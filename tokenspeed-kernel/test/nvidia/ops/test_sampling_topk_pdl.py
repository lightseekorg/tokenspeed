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

"""Check the fused Top-K/Top-P pipeline behind a producer that signals early."""

import shutil
import subprocess
from pathlib import Path

import pytest

# The producer stores the probabilities this many clock cycles after it has
# allowed its dependents to launch, far longer than the whole pipeline runs.
PRODUCER_DELAY_CYCLES = 100_000_000


@pytest.fixture(scope="module")
def topk_pdl_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler is required for the native PDL regression")
    source = Path(__file__).with_name("sampling_topk_pdl.cu")
    kernel = Path(__file__).resolve().parents[3]
    include = kernel / "python/tokenspeed_kernel/thirdparty/cuda/csrc/fused_topk_topp"
    binary = tmp_path_factory.mktemp("topk-pdl") / "test"
    compiled = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
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


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_fused_pipeline_waits_for_pdl_producer(
    topk_pdl_binary: Path, mode: str
) -> None:
    completed = subprocess.run(
        [str(topk_pdl_binary), mode, str(PRODUCER_DELAY_CYCLES)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode == 77:
        pytest.skip("No CUDA device")
    assert completed.returncode == 0, completed.stdout + completed.stderr
