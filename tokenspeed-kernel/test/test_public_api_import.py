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

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"


def _run_import_check(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(_PYTHON_DIR)
        if not env.get("PYTHONPATH")
        else f"{_PYTHON_DIR}{os.pathsep}{env['PYTHONPATH']}"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        check=True,
    )


def test_top_level_import_does_not_load_builtin_backends_or_vendor_modules() -> None:
    _run_import_check("""
import sys
before = set(sys.modules)
import tokenspeed_kernel
from tokenspeed_kernel.registry import KernelRegistry
loaded = set(sys.modules) - before

backend_prefixes = (
    "tokenspeed_kernel.numerics.reference.",
    "tokenspeed_kernel.ops.attention.cuda",
    "tokenspeed_kernel.ops.attention.flash_attn",
    "tokenspeed_kernel.ops.attention.flashinfer",
    "tokenspeed_kernel.ops.attention.gluon",
    "tokenspeed_kernel.ops.attention.triton",
    "tokenspeed_kernel.ops.embedding.cuda",
    "tokenspeed_kernel.ops.embedding.triton",
    "tokenspeed_kernel.ops.gemm.deep_gemm",
    "tokenspeed_kernel.ops.gemm.flashinfer",
    "tokenspeed_kernel.ops.gemm.triton",
    "tokenspeed_kernel.ops.gemm.trtllm",
    "tokenspeed_kernel.ops.moe.cuda",
    "tokenspeed_kernel.ops.moe.deepep",
    "tokenspeed_kernel.ops.moe.flashinfer",
    "tokenspeed_kernel.ops.moe.gluon",
    "tokenspeed_kernel.ops.moe.triton",
    "tokenspeed_kernel.ops.moe.triton_kernels",
    "tokenspeed_kernel.ops.moe.trtllm",
    "tokenspeed_kernel.ops.quantization.flashinfer",
    "tokenspeed_kernel.ops.quantization.triton",
    "tokenspeed_kernel.ops.quantization.trtllm",
)
vendor_modules = {
    "deep_ep",
    "deep_gemm",
    "flash_attn",
    "flash_attn_interface",
    "flashinfer",
    "trtllm_kernel",
}

unexpected = [
    name
    for name in loaded
    if name.startswith(backend_prefixes)
    or name.startswith("tokenspeed_kernel.thirdparty")
    or name in vendor_modules
]
assert unexpected == [], unexpected
assert KernelRegistry.get().list_kernels() == []
assert callable(tokenspeed_kernel.mm)
""")


def test_public_op_imports_do_not_register_builtin_backends() -> None:
    _run_import_check("""
import sys
try:
    import torch  # noqa: F401
except ImportError:
    raise SystemExit(0)

before = set(sys.modules)
import tokenspeed_kernel.ops.attention
import tokenspeed_kernel.ops.embedding
import tokenspeed_kernel.ops.gemm
import tokenspeed_kernel.ops.moe
import tokenspeed_kernel.ops.quantization
from tokenspeed_kernel.registry import KernelRegistry
loaded = set(sys.modules) - before

backend_prefixes = (
    "tokenspeed_kernel.numerics.reference.",
    "tokenspeed_kernel.ops.attention.cuda",
    "tokenspeed_kernel.ops.attention.flash_attn",
    "tokenspeed_kernel.ops.attention.flashinfer",
    "tokenspeed_kernel.ops.attention.gluon",
    "tokenspeed_kernel.ops.attention.triton",
    "tokenspeed_kernel.ops.embedding.cuda",
    "tokenspeed_kernel.ops.embedding.triton",
    "tokenspeed_kernel.ops.gemm.deep_gemm",
    "tokenspeed_kernel.ops.gemm.flashinfer",
    "tokenspeed_kernel.ops.gemm.triton",
    "tokenspeed_kernel.ops.gemm.trtllm",
    "tokenspeed_kernel.ops.moe.cuda",
    "tokenspeed_kernel.ops.moe.deepep",
    "tokenspeed_kernel.ops.moe.flashinfer",
    "tokenspeed_kernel.ops.moe.gluon",
    "tokenspeed_kernel.ops.moe.triton",
    "tokenspeed_kernel.ops.moe.triton_kernels",
    "tokenspeed_kernel.ops.moe.trtllm",
    "tokenspeed_kernel.ops.quantization.flashinfer",
    "tokenspeed_kernel.ops.quantization.triton",
    "tokenspeed_kernel.ops.quantization.trtllm",
)
vendor_modules = {
    "deep_ep",
    "deep_gemm",
    "flash_attn",
    "flash_attn_interface",
    "flashinfer",
    "trtllm_kernel",
}

unexpected = [
    name
    for name in loaded
    if name.startswith(backend_prefixes)
    or name.startswith("tokenspeed_kernel.thirdparty")
    or name in vendor_modules
]
assert unexpected == [], unexpected
assert KernelRegistry.get().list_kernels() == []
""")
