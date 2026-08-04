# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""RMSNorm ops: rmsnorm_fused_parallel (fused parallel RMSNorm for two inputs)."""

from __future__ import annotations

import functools
from pathlib import Path

from tokenspeed_kernel.registry import error_fn

try:
    import torch
    import tvm_ffi
except ImportError:
    torch = None
    tvm_ffi = None


@functools.cache
def _load_rmsnorm_module():
    objs_dir = Path(__file__).parent / "objs" / "rmsnorm_fused_parallel"
    so_path = objs_dir / "rmsnorm_fused_parallel.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel rmsnorm library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def rmsnorm_fused_parallel(
    input1: torch.Tensor,
    weight1: torch.Tensor,
    output1: torch.Tensor,
    input2: torch.Tensor,
    weight2: torch.Tensor,
    output2: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    _load_rmsnorm_module().rmsnorm_fused_parallel(
        input1,
        weight1,
        output1,
        input2,
        weight2,
        output2,
        float(eps),
        bool(enable_pdl),
    )


if tvm_ffi is None:
    rmsnorm_fused_parallel = error_fn

__all__ = ["rmsnorm_fused_parallel"]
