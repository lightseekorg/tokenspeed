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

"""Marlin helper ops (GPTQ repacking).

Provides Marlin helper ops for GPTQ repacking.
"""

import functools
from pathlib import Path

import torch


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_marlin_module():
    """Load the pre-compiled marlin shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "marlin" / "marlin.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel marlin library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack GPTQ quantized weights into Marlin layout.

    Args:
        b_q_weight: int32 CUDA, shape [size_k / pack_factor, size_n]
        perm: int32 CUDA, 1D; empty (numel==0) means no act_order
        size_k: number of input features
        size_n: number of output features
        num_bits: quantization bits (4 or 8)

    Returns:
        int32 CUDA, shape [size_k / 16, size_n * 16 / pack_factor]
    """
    if num_bits not in (4, 8):
        raise ValueError("num_bits must be 4 or 8")
    pack_factor = 32 // int(num_bits)
    out = torch.empty(
        (int(size_k) // 16, int(size_n) * 16 // pack_factor),
        device=b_q_weight.device,
        dtype=torch.int32,
    )
    _load_marlin_module().gptq_marlin_repack(
        out, b_q_weight, perm, int(size_k), int(size_n), int(num_bits)
    )
    return out
