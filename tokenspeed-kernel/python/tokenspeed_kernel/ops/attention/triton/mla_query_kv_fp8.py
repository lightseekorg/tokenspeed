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

"""Fused NoPE MLA decode: fp8 query assembly + latent KV-cache commit.

On the NoPE + fp8-KV decode path two independent glue kernels run back to
back on the main stream each MLA layer: the query assemble+quantize
(``mla_nope_query_fp8``) and the latent KV-cache write
(``set_mla_kv_buffer``). Both are tiny (a handful of CTAs, ~1.5-2us each) and
purely launch/latency bound, so a second launch nearly doubles the cost of
the pair. This kernel runs both in ONE launch: programs ``(t, h < H)``
assemble and quantize the per-head query, programs ``(t, H)`` sanitize and
commit token ``t``'s latent row into the fp8 cache at ``loc[t]``.

Provenance: fusion validated offline with torch.compile/Inductor as a
codegen oracle over our own repro module (Inductor kept the pair as two
launches; the flat-index/select codegen it emitted is mirrored here by the
two-segment load/store). No third-party code involved.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["mla_nope_query_kv_fp8"]


@triton.jit
def _mla_nope_query_kv_fp8_kernel(
    nope_ptr,
    pe_ptr,
    latent_ptr,
    out_ptr,
    kv_ptr,
    loc_ptr,
    stride_nt,
    stride_nh,
    stride_pt,
    stride_ph,
    stride_lt,
    stride_ot,
    stride_oh,
    stride_kv,
    H: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
    SANITIZE: tl.constexpr,
    MAX_FINITE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    t = tl.program_id(0)
    h = tl.program_id(1)

    n_offs = tl.arange(0, NOPE_DIM)
    p_offs = tl.arange(0, PE_DIM)

    if h < H:
        # Query lane: [nope | pe] -> fp8, one CTA per (token, head).
        nv = tl.load(nope_ptr + t * stride_nt + h * stride_nh + n_offs)
        tl.store(
            out_ptr + t * stride_ot + h * stride_oh + n_offs,
            nv.to(out_ptr.dtype.element_ty),
        )
        pv = tl.load(pe_ptr + t * stride_pt + h * stride_ph + p_offs)
        tl.store(
            out_ptr + t * stride_ot + h * stride_oh + NOPE_DIM + p_offs,
            pv.to(out_ptr.dtype.element_ty),
        )
    else:
        # KV lane: commit token t's latent row [nope | rope] to the cache.
        loc = tl.load(loc_ptr + t).to(tl.int64)
        dst = kv_ptr + loc * stride_kv
        src = latent_ptr + t * stride_lt

        kn = tl.load(src + n_offs)
        kr = tl.load(src + NOPE_DIM + p_offs)
        if SANITIZE:
            kn = kn.to(tl.float32)
            kn = tl.where(kn != kn, 0.0, kn)
            kn = tl.where(kn == float("inf"), MAX_FINITE, kn)
            kn = tl.where(kn == -float("inf"), -MAX_FINITE, kn)
            kr = kr.to(tl.float32)
            kr = tl.where(kr != kr, 0.0, kr)
            kr = tl.where(kr == float("inf"), MAX_FINITE, kr)
            kr = tl.where(kr == -float("inf"), -MAX_FINITE, kr)
        tl.store(dst + n_offs, kn.to(kv_ptr.dtype.element_ty))
        tl.store(dst + NOPE_DIM + p_offs, kr.to(kv_ptr.dtype.element_ty))

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def mla_nope_query_kv_fp8(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    latent: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    *,
    sanitize: bool = True,
    enable_pdl: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused ``cat([q_nope, q_pe], -1).to(fp8)`` + latent KV-cache commit.

    Equivalent to ``mla_nope_query_fp8(q_nope, q_pe)`` followed by
    ``set_mla_kv_buffer(kv_buffer, loc, latent[..., :nope], latent[..., nope:])``
    but in a single launch (the pair is launch-latency bound at decode batch
    sizes).

    Args:
        q_nope: ``[T, H, nope_dim]`` absorbed query half (any float dtype;
            unit inner stride).
        q_pe: ``[T, H, pe_dim]`` pe query half (may be a strided column
            slice; unit inner stride).
        latent: ``[T, nope_dim + pe_dim]`` latent KV rows to commit
            (``[kv_a | k_pe]``; unit inner stride). A trailing singleton head
            dim (``[T, 1, D]``) is accepted.
        kv_buffer: ``[num_rows, 1, nope_dim + pe_dim]`` (or ``[num_rows,
            nope_dim + pe_dim]``) cache buffer with unit inner stride; the
            store casts to its dtype (fp8 for the K3 FlatKV pool).
        loc: ``[T]`` int32/int64 absolute row locations into ``kv_buffer``.
        sanitize: Replace NaN with 0 and clamp +-Inf to the largest value
            finite in both the source and cache dtypes (mirrors
            ``set_mla_kv_buffer_triton``'s FlatKV default). Applies to the KV
            commit only; the query lane stores raw casts like
            ``mla_nope_query_fp8``.
        enable_pdl: Launch with programmatic dependent launch and fence with
            ``gdc_wait``/``gdc_launch_dependents``.
        out: Optional ``[T, H, nope_dim + pe_dim]`` fp8 query destination.

    Returns:
        ``[T, H, nope_dim + pe_dim]`` fp8 query tensor.
    """
    T, H, nope_dim = q_nope.shape
    pe_dim = q_pe.shape[-1]
    assert q_pe.shape[:2] == (T, H)
    if latent.dim() == 3:
        assert latent.shape[1] == 1
        latent = latent.squeeze(1)
    assert latent.shape == (T, nope_dim + pe_dim)
    kv2d = kv_buffer.view(kv_buffer.shape[0], -1)
    assert kv2d.shape[-1] == nope_dim + pe_dim
    assert loc.shape == (T,) and loc.dtype in (torch.int32, torch.int64)
    assert (
        q_nope.stride(-1) == 1
        and q_pe.stride(-1) == 1
        and latent.stride(-1) == 1
        and kv2d.stride(-1) == 1
    )
    # Power-of-two dims keep tl.arange legal (512/64 in every MLA config).
    assert nope_dim & (nope_dim - 1) == 0 and pe_dim & (pe_dim - 1) == 0

    if out is None:
        out = torch.empty(
            T, H, nope_dim + pe_dim, dtype=torch.float8_e4m3fn, device=q_nope.device
        )
    # Largest value representable in both the latent source and the cache
    # dtype: bf16 sources into an fp8 cache must clamp at the fp8 bound or
    # the store would round +-Inf back to a non-finite fp8 encoding.
    float_maxes = [
        torch.finfo(t.dtype).max
        for t in (latent, kv_buffer)
        if t.dtype.is_floating_point
    ]
    max_finite = min(float_maxes) if float_maxes else float("inf")

    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _mla_nope_query_kv_fp8_kernel[(T, H + 1)](
        q_nope,
        q_pe,
        latent,
        out,
        kv2d,
        loc,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        latent.stride(0),
        out.stride(0),
        out.stride(1),
        kv2d.stride(0),
        H=H,
        NOPE_DIM=nope_dim,
        PE_DIM=pe_dim,
        SANITIZE=sanitize,
        MAX_FINITE=max_finite,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **extra_kwargs,
    )
    return out
