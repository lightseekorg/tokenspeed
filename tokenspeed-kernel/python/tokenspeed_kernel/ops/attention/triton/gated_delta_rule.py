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

"""Portable Triton gated delta net chunked prefill + decode/MTP.

The public ``gdn_chunk_prefill`` contract (initial_state/final_state/h) is
K-last ``[N, H, V, K]`` (v-major), matching the flashinfer backend's native
layout and the runtime's SSM state pool. This Triton backend's own FLA math
is natively ``[N, H, K, V]``, so this wrapper (not the ``.jit`` kernel itself)
transposes K-last -> FLA on the way in and FLA -> K-last on the way out.

``gdn_decode_step``/``gdn_decode_mtp`` (below) are the portable fallback for
flashinfer's Hopper+-only decode/MTP kernels (see
``flashinfer/gated_delta_rule.py``): same K-last state-pool contract, but the
``.jit`` kernel itself addresses the pool K-last directly (no transpose --
the state pool is large and mutated in place every decode step, so a
transpose-and-materialize wrapper like the prefill one above would be far too
expensive here).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from tokenspeed_kernel.ops.attention.gdn_utils import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
)
from tokenspeed_kernel.ops.attention.triton.linear.chunk import chunk_gated_delta_rule
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "gdn_chunk_prefill",
    name="triton_gdn_chunk_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={
        "qk_l2norm": frozenset({False, True}),
        "output_h": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
) -> GdnChunkPrefillResult:
    # initial_state arrives K-last [N, H, V, K] (the shared contract); this
    # backend's FLA math needs [N, H, K, V], so transpose in (materialized:
    # the Triton kernel below has no dynamic-stride support for h0).
    initial_state_fla = initial_state.transpose(-2, -1).contiguous()
    result = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state_fla,
        output_final_state=output_final_state or output_h,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=qk_l2norm,
        output_h=output_h,
    )
    if output_h:
        out, final_state, h = result
        final_state_klast = (
            final_state.transpose(-2, -1) if final_state is not None else None
        )
        return GdnChunkPrefillResult(
            out=out,
            final_state=final_state_klast,
            h=h.transpose(-2, -1),
            h_layout=GdnCheckpointLayout.FLA,
        )
    out, final_state = result
    final_state_klast = (
        final_state.transpose(-2, -1) if final_state is not None else None
    )
    return GdnChunkPrefillResult(out=out, final_state=final_state_klast)


# ===-----------------------------------------------------------------------===#
# GDN decode / MTP (K-last, portable fallback for flashinfer's Hopper+ kernels)
# ===-----------------------------------------------------------------------===#


@triton.jit(do_not_specialize=["T"])
def _fused_gdn_decode_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    output_state_indices,
    intermediate_states_buffer,
    per_token_output_state_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr,
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
    HAS_OUTPUT_STATE_INDICES: tl.constexpr,
    HAS_PER_TOKEN_OUTPUT_STATE_INDICES: tl.constexpr,
):
    """Dense-batch (no varlen) sigmoid-gating delta-rule decode/MTP update.

    q, k: [B, T, H, K]; v: [B, T, HV, V]; a, b: [B, T, HV]; o: [B, T, HV, V]
    (all contiguous -- enforced by the Python launcher, so strides below
    derive directly from the constexpr dims). h0_source is the K-last
    ``[pool_size, HV, V, K]`` SSM state pool (matches gdn_chunk_prefill's
    contract); h0_indices ([B]) selects each batch entry's read row, ``-1``
    skipped (state contribution treated as zero -- mirrors flashinfer's
    float32 legacy decode path; output for that batch entry is undefined).

    T=1 is a plain decode step; T>1 is an MTP verify step. Three independent,
    mutually-exclusive-by-convention state-output mechanisms:
    - HAS_OUTPUT_STATE_INDICES: after the LAST processed step, write to the
      single row ``output_state_indices[i_n]`` (``[B]``-shaped; T=1 decode's
      dual-index paging remap).
    - CACHE_INTERMEDIATE_STATES: after EVERY step, write to the batch-scoped
      ``intermediate_states_buffer[i_n, step]`` (``[B, T, HV, V, K]``,
      K-last -- matches flashinfer's MTP intermediate-state-buffer convention).
    - HAS_PER_TOKEN_OUTPUT_STATE_INDICES: after EVERY step, write directly to
      the pool row ``per_token_output_state_indices[i_n, step]`` (``[B, T]``),
      matching FlashInfer 0.6.15's ``ssm_state_indices`` contract.
    DISABLE_STATE_UPDATE additionally gates a final write-back to
    ``h0_indices[i_n]`` (the read row) when neither of the above applies.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (i_n * T) * H * K + i_h * K + o_k
    p_k = k + (i_n * T) * H * K + i_h * K + o_k
    p_v = v + (i_n * T) * HV * V + i_hv * V + o_v
    p_b = b + (i_n * T) * HV + i_hv
    p_o = o + (i_n * T * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    p_a = a + (i_n * T) * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n)
    if idx >= 0:
        # K-last [.., HV, V, K]: row v (stride K), col k (stride 1) -- the
        # opposite of this kernel's own [BK, BV] (K-row, V-col) tile layout,
        # which stays FLA-native in registers; only the pool addressing below
        # is K-last.
        p_h0 = (
            h0_source
            + idx * HV * V * K
            + i_hv * V * K
            + o_v[None, :] * K
            + o_k[:, None]
        )
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for step_idx in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale

        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if CACHE_INTERMEDIATE_STATES:
            cache_ptr = (
                intermediate_states_buffer
                + ((i_n * T + step_idx) * HV + i_hv) * V * K
                + o_v[None, :] * K
                + o_k[:, None]
            )
            tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        if HAS_PER_TOKEN_OUTPUT_STATE_INDICES:
            out_idx = tl.load(per_token_output_state_indices + i_n * T + step_idx).to(
                tl.int64
            )
            if out_idx >= 0:
                p_step_out = (
                    h0_source
                    + out_idx * HV * V * K
                    + i_hv * V * K
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(
                    p_step_out,
                    b_h.to(p_step_out.dtype.element_ty),
                    mask=mask_h,
                )

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_b += HV
        p_o += HV * V
        p_a += HV

    if HAS_OUTPUT_STATE_INDICES:
        out_idx = tl.load(output_state_indices + i_n).to(tl.int64)
        if out_idx >= 0:
            p_out = (
                h0_source
                + out_idx * HV * V * K
                + i_hv * V * K
                + o_v[None, :] * K
                + o_k[:, None]
            )
            tl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=mask_h)
    elif not DISABLE_STATE_UPDATE and not HAS_PER_TOKEN_OUTPUT_STATE_INDICES:
        if idx >= 0:
            p_out = (
                h0_source
                + idx * HV * V * K
                + i_hv * V * K
                + o_v[None, :] * K
                + o_k[:, None]
            )
            tl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=mask_h)


def _launch_fused_gdn_decode_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    output_state_indices: torch.Tensor | None,
    intermediate_states_buffer: torch.Tensor | None,
    per_token_output_state_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Shared launcher for the ``gdn_decode_step`` (T=1) / ``gdn_decode_mtp``
    (T>1) Triton fallback kernels. q/k: [B, T, H, K]; v: [B, T, HV, V]; a/b:
    [B, T, HV]; initial_state is the K-last [pool_size, HV, V, K] SSM state
    pool (the shared runtime contract -- see module docstring).
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    if scale is None:
        scale = K**-0.5

    o = q.new_empty(B, T, HV, V)

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    grid = (NK, NV, B * HV)
    _fused_gdn_decode_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state,
        h0_indices=initial_state_indices,
        output_state_indices=output_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        per_token_output_state_indices=per_token_output_state_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_OUTPUT_STATE_INDICES=output_state_indices is not None,
        HAS_PER_TOKEN_OUTPUT_STATE_INDICES=(per_token_output_state_indices is not None),
        num_warps=1,
        num_stages=3,
    )
    return o


@register_kernel(
    "attention",
    "gdn_decode_step",
    name="triton_gdn_decode_step",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Portable Triton fallback for ``gdn_decode_step`` (see
    ``flashinfer/gated_delta_rule.py`` for the shared contract). State is
    always updated: to ``output_state_indices`` when given, else back to
    ``initial_state_indices`` in place.
    """
    return _launch_fused_gdn_decode_update(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=False,
        output_state_indices=output_state_indices,
        intermediate_states_buffer=None,
        per_token_output_state_indices=None,
    )


@register_kernel(
    "attention",
    "gdn_decode_mtp",
    name="triton_gdn_decode_mtp",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    tags={"portability", "speculative-decoding"},
)
def triton_gdn_decode_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    disable_state_update: bool = True,
    use_qk_l2norm: bool = True,
    intermediate_states_buffer: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Portable Triton fallback for ``gdn_decode_mtp`` (see
    ``flashinfer/gated_delta_rule.py`` for the shared contract). Supports both
    batch-scoped intermediate-state caching and direct per-token pool scatter
    through ``output_state_indices``.
    """
    return _launch_fused_gdn_decode_update(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        output_state_indices=None,
        intermediate_states_buffer=intermediate_states_buffer,
        per_token_output_state_indices=output_state_indices,
    )
