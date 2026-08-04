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

"""Fused NoPE MLA decode-query assembly + fp8 quantization.

The absorbed decode query is ``[nope | pe]`` per head: the nope half comes
out of the absorb bmm, the pe half is a strided slice of the q projection.
Without rope in between (NoPE models), assembling the bf16 query and then
casting it to fp8 costs two elementwise launches and an extra full pass
over the query; this kernel reads both halves once and stores the fp8
query directly.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["mla_nope_query_fp8"]


@triton.jit
def _mla_nope_query_fp8_kernel(
    nope_ptr,
    pe_ptr,
    out_ptr,
    stride_nt,
    stride_nh,
    stride_pt,
    stride_ph,
    stride_ot,
    stride_oh,
    NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    n_offs = tl.arange(0, NOPE_DIM)
    nv = tl.load(nope_ptr + t * stride_nt + h * stride_nh + n_offs)
    tl.store(
        out_ptr + t * stride_ot + h * stride_oh + n_offs,
        nv.to(out_ptr.dtype.element_ty),
    )

    p_offs = tl.arange(0, PE_DIM)
    pv = tl.load(pe_ptr + t * stride_pt + h * stride_ph + p_offs)
    tl.store(
        out_ptr + t * stride_ot + h * stride_oh + NOPE_DIM + p_offs,
        pv.to(out_ptr.dtype.element_ty),
    )


def mla_nope_query_fp8(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``cat([q_nope, q_pe], -1).to(float8_e4m3fn)`` in one launch.

    Args:
        q_nope: ``[T, H, nope_dim]`` absorbed query half (any float dtype;
            unit inner stride).
        q_pe: ``[T, H, pe_dim]`` pe query half (may be a strided column
            slice; unit inner stride).
        out: optional ``[T, H, nope_dim + pe_dim]`` fp8 destination.

    Returns:
        ``[T, H, nope_dim + pe_dim]`` fp8 query.
    """
    T, H, nope_dim = q_nope.shape
    pe_dim = q_pe.shape[-1]
    assert q_pe.shape[:2] == (T, H)
    assert q_nope.stride(-1) == 1 and q_pe.stride(-1) == 1
    # Power-of-two dims keep tl.arange legal (512/64 in every MLA config).
    assert nope_dim & (nope_dim - 1) == 0 and pe_dim & (pe_dim - 1) == 0
    if out is None:
        out = torch.empty(
            T, H, nope_dim + pe_dim, dtype=torch.float8_e4m3fn, device=q_nope.device
        )
    _mla_nope_query_fp8_kernel[(T, H)](
        q_nope,
        q_pe,
        out,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        out.stride(0),
        out.stride(1),
        NOPE_DIM=nope_dim,
        PE_DIM=pe_dim,
        num_warps=4,
    )
    return out
