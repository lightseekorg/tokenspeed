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

"""DeepSeek V3 fused-A projection GEMM, adapted from TensorRT-LLM."""

import functools
from pathlib import Path

import torch
from tokenspeed_kernel.platform import pdl_enabled

FUSED_A_HIDDEN = 7168
FUSED_A_OUT = 2112
FUSED_A_MAX_TOKENS = 16


@functools.cache
def _load_module():
    """Load the pre-compiled dsv3_fused_a_gemm shared library via TVM FFI."""
    import tvm_ffi

    so_path = (
        Path(__file__).resolve().parent
        / "objs"
        / "dsv3_fused_a_gemm"
        / "dsv3_fused_a_gemm.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel dsv3_fused_a_gemm library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    """Fused q_a/kv_a down-projection for 1 to 16 tokens on SM90+.

    Args:
        mat_a: bf16 ``[num_tokens, 7168]`` row-major activations.
        mat_b: bf16 ``[7168, 2112]`` column-major weight, i.e. ``weight.T`` of
            a contiguous ``[2112, 7168]`` weight.
        enable_pdl: whether to launch with PDL; defaults to the platform setting.

    Returns:
        bf16 ``[num_tokens, 2112]`` output.
    """
    if mat_a.dim() != 2 or mat_b.dim() != 2:
        raise ValueError("dsv3_fused_a_gemm takes 2-D operands")
    if mat_a.dtype != torch.bfloat16 or mat_b.dtype != torch.bfloat16:
        raise ValueError("dsv3_fused_a_gemm takes bf16 operands")
    num_tokens = mat_a.shape[0]
    if not 1 <= num_tokens <= FUSED_A_MAX_TOKENS:
        raise ValueError(
            f"dsv3_fused_a_gemm takes 1 to {FUSED_A_MAX_TOKENS} tokens, got {num_tokens}"
        )
    if mat_a.shape[1] != FUSED_A_HIDDEN or tuple(mat_b.shape) != (
        FUSED_A_HIDDEN,
        FUSED_A_OUT,
    ):
        raise ValueError(
            f"dsv3_fused_a_gemm computes [M, {FUSED_A_HIDDEN}] x "
            f"[{FUSED_A_HIDDEN}, {FUSED_A_OUT}], got {tuple(mat_a.shape)} x "
            f"{tuple(mat_b.shape)}"
        )
    output = torch.empty(
        (num_tokens, FUSED_A_OUT), device=mat_a.device, dtype=torch.bfloat16
    )
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
    _load_module().dsv3_fused_a_gemm(output, mat_a, mat_b, bool(enable_pdl))
    return output
