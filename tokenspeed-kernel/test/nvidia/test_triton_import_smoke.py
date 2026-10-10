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

import subprocess
import sys
from pathlib import Path

import pytest
from tokenspeed_kernel.platform import current_platform


@pytest.mark.skipif(not current_platform().is_hopper, reason="requires SM90")
def test_dots3_triton_imports():
    # Use real provider imports in a fresh process, without loading unrelated
    # families or launching GPU kernels. Package presence alone is insufficient.
    script = """
import importlib
import sys

sys.path.insert(0, sys.argv[1])
for family in (
    "attention.dsa", "attention.mla", "attention.prologue", "kvcache",
    "quantization", "gemm", "layernorm", "moe", "activation", "embedding",
    "transform",
):
    importlib.import_module("tokenspeed_kernel.ops." + family)

from tokenspeed_kernel.registry import KernelRegistry
registry = KernelRegistry.get()
for name in (
    "triton_dsa_decode", "triton_dsa_prefill", "triton_dsa_decode_topk_fp8",
    "triton_dsa_prefill_topk_fp8", "triton_mla_prefill",
    "triton_mla_decode_with_kvcache", "triton_mla_prologue",
    "triton_fp8_precomputed_moe_apply",
):
    assert registry.get_by_name(name) is not None, name
    assert callable(registry.get_impl(name)), name
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve().parents[2] / "python"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
