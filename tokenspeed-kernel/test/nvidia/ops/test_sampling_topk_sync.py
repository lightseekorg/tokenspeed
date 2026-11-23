# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Check radix publication with a delayed CTA, including graph replay."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def topk_sync_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler is required for the native radix regression")
    source = Path(__file__).with_name("sampling_topk_sync.cu")
    kernel = Path(__file__).resolve().parents[3]
    include = kernel / "python/tokenspeed_kernel/thirdparty/cuda/csrc/fused_topk_topp"
    binary = tmp_path_factory.mktemp("topk-sync") / "test"
    compiled = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_90,code=[sm_90,compute_90]",
            "-I",
            str(include),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return binary


@pytest.mark.parametrize("distribution", ["uniform", "boundary"])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_topk_waits_for_histogram_writers(
    topk_sync_binary: Path, distribution: str, mode: str
) -> None:
    completed = subprocess.run(
        [str(topk_sync_binary), distribution, mode],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode == 77:
        pytest.skip("No CUDA device")
    assert completed.returncode == 0, completed.stdout + completed.stderr
