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
from tokenspeed_kernel._triton import gl, gluon, libdevice, tl, triton
from tokenspeed_kernel.ops.attention.gdn import (
    GDN_TREE_VERIFY_CHUNKED_MAX_NODES,
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
    GdnStateCheckpoints,
)
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    CAUSAL_CONV1D_BLOCK_M,
    CausalConv1dPrefillMetadata,
    build_causal_conv1d_capacity_metadata,
    build_causal_conv1d_prefill_metadata,
    refresh_causal_conv1d_capacity_metadata,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chain_verify import (
    gdn_chain_verify_warm_batch_sizes,
    launch_gdn_chain_verify,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chunk import (
    chunk_gated_delta_rule,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chunk_delta_h import CHUNK_SIZE
from tokenspeed_kernel.ops.attention.gdn._triton.index import (
    prepare_chunk_indices,
    set_total_chunks_hint,
    set_total_chunks_hint_uniform,
)
from tokenspeed_kernel.ops.attention.gdn._triton.prefill_state_inputs import (
    prepare_prefill_state_inputs,
)
from tokenspeed_kernel.ops.attention.gdn._triton.qkv_split import (
    fused_qkv_split_gdn_prefill,
    pack_gdn_projection,
)
from tokenspeed_kernel.platform import (
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# The payload stores take ptxas to 172 registers, a CTA per SM fewer; 168 keeps three CTAs.
_TREE_VERIFY_REPLAY_MAX_REGISTERS = 168


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
        "output_h": frozenset({False, True}),
        "qk_l2norm": frozenset({False, True}),
        "state_checkpoints": frozenset({False}),
    },
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
    state_checkpoints: GdnStateCheckpoints | None = None,
) -> GdnChunkPrefillResult:
    if state_checkpoints is not None:
        raise ValueError("triton_gdn_chunk_prefill stores no state checkpoints")
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


@triton.jit
def _latest_ancestor(ancestors_row, step, valid):
    """Parent of ``step`` (scalar or vector) in a tree's ancestor-or-self bitmask row: the highest bit below its own, -1 for a root."""
    above = tl.load(ancestors_row + step, mask=valid, other=0) & (
        (tl.full([], 1, tl.int64) << step.to(tl.int64)) - 1
    )
    return 63 - libdevice.clz(above)


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
    ht_dest,
    h0_indices,
    output_state_indices,
    intermediate_states_buffer,
    per_token_output_state_indices,
    tree_ancestors,
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
    HAS_TREE: tl.constexpr,
    T_BLOCK: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Dense-batch (no varlen) sigmoid-gating delta-rule decode/MTP update.

    q, k: [B, T, H, K]; v: [B, T, HV, V]; a, b: [B, T, HV]; o: [B, T, HV, V]
    (input strides are explicit so decode can consume packed QKV views
    without intermediate copies). h0_source is the K-last
    ``[pool_size, HV, V, K]`` SSM state pool (matches gdn_chunk_prefill's
    contract); h0_indices ([B]) selects each batch entry's read row, ``-1``
    skipped (state contribution treated as zero -- mirrors flashinfer's
    float32 legacy decode path; output for that batch entry is undefined).
    ``ht_dest`` is the K-last pool both output-state index forms address; it
    may be ``h0_source`` itself.

    T=1 is a plain decode step; T>1 is an MTP verify step. Three independent,
    mutually-exclusive-by-convention state-output mechanisms:
    - HAS_OUTPUT_STATE_INDICES: after the LAST processed step, write to the
      single ``ht_dest`` row ``output_state_indices[i_n]`` (``[B]``-shaped;
      T=1 decode's dual-index paging remap).
    - CACHE_INTERMEDIATE_STATES: after EVERY step (every branch point under
      HAS_TREE), write to the batch-scoped
      ``intermediate_states_buffer[i_n, step]`` (``[B, T, HV, V, K]``,
      K-last -- matches flashinfer's MTP intermediate-state-buffer convention).
    - HAS_PER_TOKEN_OUTPUT_STATE_INDICES: after EVERY step, write directly to
      the ``ht_dest`` row ``per_token_output_state_indices[i_n, step]``
      (``[B, T]``), matching FlashInfer 0.6.15's ``ssm_state_indices`` contract.
    DISABLE_STATE_UPDATE additionally gates a final write-back to
    ``h0_indices[i_n]`` (the read row) when neither of the above applies.

    HAS_TREE (draft trees, with per-token output rows or the intermediate
    buffer): step t continues from the state after its parent, its latest
    ancestor in ``tree_ancestors[i_n, t]`` (the initial state for a root),
    instead of step t - 1, reloading it from that step's output row or
    intermediate-buffer entry when the parent is not t - 1. The intermediate buffer then receives
    only the branch points' states (steps with a child other than the next
    step), the only entries a later step reloads.
    """
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Release successor setup; its wait still guards all dependent reads.
        tl.extra.cuda.gdc_launch_dependents()
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + i_n * Q_STRIDES[0] + i_h * Q_STRIDES[2] + o_k * Q_STRIDES[3]
    p_k = k + i_n * K_STRIDES[0] + i_h * K_STRIDES[2] + o_k * K_STRIDES[3]
    p_v = v + i_n * V_STRIDES[0] + i_hv * V_STRIDES[2] + o_v * V_STRIDES[3]
    p_b = b + i_n * B_STRIDES[0] + i_hv * B_STRIDES[2]
    p_o = o + (i_n * T * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    p_a = a + i_n * A_STRIDES[0] + i_hv * A_STRIDES[2]
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices + i_n).to(tl.int64)
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

    if HAS_TREE and CACHE_INTERMEDIATE_STATES:
        nodes = tl.arange(0, T_BLOCK)
        parents = _latest_ancestor(tree_ancestors + i_n * T, nodes, nodes < T)

    # Prefetch the next step's operands to overlap the current state update.
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)
    n_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    n_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    n_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    n_b = tl.load(p_b).to(tl.float32)
    n_a = tl.load(p_a).to(tl.float32)
    for step_idx in range(0, T):
        b_q = n_q
        b_k = n_k
        b_v = n_v
        b_b = n_b
        b_a = n_a
        has_next = step_idx + 1 < T
        n_q = tl.load(p_q + Q_STRIDES[1], mask=mask_k & has_next, other=0).to(
            tl.float32
        )
        n_k = tl.load(p_k + K_STRIDES[1], mask=mask_k & has_next, other=0).to(
            tl.float32
        )
        n_v = tl.load(p_v + V_STRIDES[1], mask=mask_v & has_next, other=0).to(
            tl.float32
        )
        n_b = tl.load(p_b + B_STRIDES[1], mask=has_next, other=0).to(tl.float32)
        n_a = tl.load(p_a + A_STRIDES[1], mask=has_next, other=0).to(tl.float32)
        if HAS_TREE:
            parent = _latest_ancestor(tree_ancestors + i_n * T, step_idx, True)
            if parent != step_idx - 1:
                # The parent's row was written by this program over the same tile.
                tl.debug_barrier()
                if CACHE_INTERMEDIATE_STATES:
                    # Replay verify: node states live in the intermediate buffer only.
                    if parent < 0:
                        p_parent = (
                            h0_source
                            + idx * HV * V * K
                            + i_hv * V * K
                            + o_v[None, :] * K
                            + o_k[:, None]
                        )
                        b_h = tl.load(p_parent, mask=mask_h & (idx >= 0), other=0).to(
                            tl.float32
                        )
                    else:
                        p_parent = (
                            intermediate_states_buffer
                            + ((i_n * T + parent).to(tl.int64) * HV + i_hv) * V * K
                            + o_v[None, :] * K
                            + o_k[:, None]
                        )
                        b_h = tl.load(p_parent, mask=mask_h, other=0).to(tl.float32)
                elif parent < 0:
                    p_parent = (
                        h0_source
                        + idx * HV * V * K
                        + i_hv * V * K
                        + o_v[None, :] * K
                        + o_k[:, None]
                    )
                    b_h = tl.load(p_parent, mask=mask_h & (idx >= 0), other=0).to(
                        tl.float32
                    )
                else:
                    # Non-root parents were written to the destination pool.
                    row = tl.load(per_token_output_state_indices + i_n * T + parent).to(
                        tl.int64
                    )
                    p_parent = (
                        ht_dest
                        + row * HV * V * K
                        + i_hv * V * K
                        + o_v[None, :] * K
                        + o_k[:, None]
                    )
                    b_h = tl.load(p_parent, mask=mask_h & (row >= 0), other=0).to(
                        tl.float32
                    )
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
                + ((i_n * T + step_idx).to(tl.int64) * HV + i_hv) * V * K
                + o_v[None, :] * K
                + o_k[:, None]
            )
            if HAS_TREE:
                child = (parents == step_idx) & (nodes > step_idx + 1)
                if tl.max(child.to(tl.int32), axis=0) != 0:
                    tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)
            else:
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        if HAS_PER_TOKEN_OUTPUT_STATE_INDICES:
            out_idx = tl.load(per_token_output_state_indices + i_n * T + step_idx).to(
                tl.int64
            )
            if out_idx >= 0:
                p_step_out = (
                    ht_dest
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

        p_q += Q_STRIDES[1]
        p_k += K_STRIDES[1]
        p_v += V_STRIDES[1]
        p_b += B_STRIDES[1]
        p_o += HV * V
        p_a += A_STRIDES[1]

    if HAS_OUTPUT_STATE_INDICES:
        out_idx = tl.load(output_state_indices + i_n).to(tl.int64)
        if out_idx >= 0:
            p_out = (
                ht_dest
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
    output_state: torch.Tensor | None,
    scale: float | None,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    output_state_indices: torch.Tensor | None,
    intermediate_states_buffer: torch.Tensor | None,
    per_token_output_state_indices: torch.Tensor | None,
    tree_ancestors: torch.Tensor | None,
) -> torch.Tensor:
    """Shared launcher for the ``gdn_decode_step`` (T=1) / ``gdn_decode_mtp``
    (T>1) Triton fallback kernels. q/k: [B, T, H, K]; v: [B, T, HV, V]; a/b:
    [B, T, HV]; initial_state is the K-last [pool_size, HV, V, K] SSM state
    pool (the shared runtime contract -- see module docstring); output_state
    is the K-last pool the output-state indices address (None: initial_state).
    """
    enable_pdl = pdl_enabled()
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    if scale is None:
        scale = K**-0.5
    # The kernel addresses output rows densely (row * HV * V * K).
    if output_state is not None and not output_state.is_contiguous():
        raise ValueError(
            "Triton GDN decode requires a contiguous output_state, got strides "
            f"{output_state.stride()}"
        )

    o = q.new_empty(B, T, HV, V)

    # Serial tree verify needs narrow V tiles for parallelism; chains keep BV=32.
    BK, BV = triton.next_power_of_2(K), min(
        triton.next_power_of_2(V), 8 if tree_ancestors is not None else 32
    )
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
        ht_dest=initial_state if output_state is None else output_state,
        h0_indices=initial_state_indices,
        output_state_indices=output_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        per_token_output_state_indices=per_token_output_state_indices,
        tree_ancestors=tree_ancestors,
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
        HAS_TREE=tree_ancestors is not None,
        T_BLOCK=1 if tree_ancestors is None else triton.next_power_of_2(T),
        Q_STRIDES=q.stride(),
        K_STRIDES=k.stride(),
        V_STRIDES=v.stride(),
        A_STRIDES=a.stride(),
        B_STRIDES=b.stride(),
        num_warps=1,
        num_stages=3,
        ENABLE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
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
    traits={"strided_qkv": frozenset({False, True})},
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
        output_state=None,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=False,
        output_state_indices=output_state_indices,
        intermediate_states_buffer=None,
        per_token_output_state_indices=None,
        tree_ancestors=None,
    )


@triton.jit(do_not_specialize=["T"])
def _gdn_tree_verify_chunked_kernel(
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
    tree_ancestors,
    replay,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    T_BLOCK: tl.constexpr,
    HAS_REPLAY: tl.constexpr,
    LOG2_T_BLOCK: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """ReplaySSM draft-tree GDN verify of up to ``T_BLOCK`` nodes at once, in the chunked form.

    q, k, v, a, b, o and the K-last state pool are as in
    ``_fused_gdn_decode_update_kernel``; ``tree_ancestors`` is the ``[B, T]``
    int64 ancestor-or-self bitmask. With ``G_t`` the gate summed over node t's
    root path, ``D[t, s] = exp(G_t - G_s)`` for each ancestor s (summed over
    the path segment, not subtracted), and ``U`` the beta-scaled delta updates
    solving ``(I + beta * D_strict * k k^T) U = beta * (v - exp(G) * k h0)``,
    node t's output is ``exp(G_t) q_t h0 + sum_s D[t, s] (q_t . k_s) U_s``:
    the recurrence along that path, without its serial steps. The pool is left
    untouched. With ``HAS_REPLAY`` it also copies each node's k, v, a and b, as
    loaded, into its ``replay`` row ``[k | v | a | b]`` (the ReplaySSM payload).
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    nodes = tl.arange(0, T_BLOCK)
    valid = nodes < T
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Release successor setup; its wait still guards all dependent reads.
        tl.extra.cuda.gdc_launch_dependents()

    b_A = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    p_q = q + i_n * Q_STRIDES[0] + nodes[:, None] * Q_STRIDES[1] + i_h * Q_STRIDES[2]
    p_k = k + i_n * K_STRIDES[0] + nodes[:, None] * K_STRIDES[1] + i_h * K_STRIDES[2]
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    p_h0 = h0_source + idx * HV * V * K + i_hv * V * K + o_v[:, None] * K

    b_v = tl.load(
        v
        + i_n * V_STRIDES[0]
        + nodes[:, None] * V_STRIDES[1]
        + i_hv * V_STRIDES[2]
        + o_v[None, :] * V_STRIDES[3],
        mask=valid[:, None] & mask_v[None, :],
        other=0,
    )
    b_a = tl.load(
        a + i_n * A_STRIDES[0] + nodes * A_STRIDES[1] + i_hv * A_STRIDES[2],
        mask=valid,
        other=0,
    )
    b_b = tl.load(
        b + i_n * B_STRIDES[0] + nodes * B_STRIDES[1] + i_hv * B_STRIDES[2],
        mask=valid,
        other=0,
    )
    REPLAY_WIDTH: tl.constexpr = H * K + HV * V + 2 * HV
    p_replay = replay + (i_n * T + nodes).to(tl.int64) * REPLAY_WIDTH
    # Gates come from each value head's first V tile, k from each key head's first value head.
    writes_k = HAS_REPLAY & (i_v == 0) & (i_hv % (HV // H) == 0)
    if HAS_REPLAY:
        tl.store(
            p_replay[:, None] + H * K + i_hv * V + o_v[None, :],
            b_v,
            mask=valid[:, None] & mask_v[None, :],
        )
        tl.store(p_replay + H * K + HV * V + i_hv, b_a, mask=valid & (i_v == 0))
        tl.store(p_replay + H * K + HV * V + HV + i_hv, b_b, mask=valid & (i_v == 0))
    b_v = b_v.to(tl.float32)
    b_a = b_a.to(tl.float32)
    b_b = b_b.to(tl.float32)
    bits = tl.load(tree_ancestors + i_n * T + nodes, mask=valid, other=0)

    r_q = tl.full([T_BLOCK], 1.0, tl.float32) * scale
    r_k = tl.full([T_BLOCK], 1.0, tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        sq_q = tl.zeros([T_BLOCK], dtype=tl.float32)
        sq_k = tl.zeros([T_BLOCK], dtype=tl.float32)
        for kc in tl.static_range(0, K, BK):
            o_k = kc + tl.arange(0, BK)
            mask_qk = valid[:, None] & (o_k < K)[None, :]
            x_q = tl.load(p_q + o_k[None, :] * Q_STRIDES[3], mask=mask_qk, other=0)
            x_k = tl.load(p_k + o_k[None, :] * K_STRIDES[3], mask=mask_qk, other=0)
            sq_q += tl.sum(x_q.to(tl.float32) * x_q.to(tl.float32), 1)
            sq_k += tl.sum(x_k.to(tl.float32) * x_k.to(tl.float32), 1)
        r_q = scale / tl.sqrt(sq_q + 1e-6)
        r_k = 1.0 / tl.sqrt(sq_k + 1e-6)

    # K-chunked so the h0 tile and the normalized rows never all sit in registers.
    kk = tl.zeros([T_BLOCK, T_BLOCK], dtype=tl.float32)
    qk = tl.zeros([T_BLOCK, T_BLOCK], dtype=tl.float32)
    kh = tl.zeros([T_BLOCK, BV], dtype=tl.float32)
    qh = tl.zeros([T_BLOCK, BV], dtype=tl.float32)
    # Pipelined: a small grid is latency-bound on the cold state rows.
    for kc in tl.range(0, K, BK, num_stages=3):
        o_k = kc + tl.arange(0, BK)
        mask_qk = valid[:, None] & (o_k < K)[None, :]
        c_q = tl.load(p_q + o_k[None, :] * Q_STRIDES[3], mask=mask_qk, other=0)
        c_q = c_q.to(tl.float32) * r_q[:, None]
        c_k = tl.load(p_k + o_k[None, :] * K_STRIDES[3], mask=mask_qk, other=0)
        if HAS_REPLAY:
            tl.store(
                p_replay[:, None] + i_h * K + o_k[None, :], c_k, mask=mask_qk & writes_k
            )
        c_k = c_k.to(tl.float32) * r_k[:, None]
        # The K-last pool row loads as [BV, BK] with K contiguous.
        c_h = tl.load(
            p_h0 + o_k[None, :],
            mask=(idx >= 0) & mask_v[:, None] & (o_k < K)[None, :],
            other=0,
        ).to(tl.float32)
        kk += tl.dot(c_k, tl.trans(c_k), input_precision=DOT_PRECISION)
        qk += tl.dot(c_q, tl.trans(c_k), input_precision=DOT_PRECISION)
        kh += tl.dot(c_k, tl.trans(c_h), input_precision=DOT_PRECISION)
        qh += tl.dot(c_q, tl.trans(c_h), input_precision=DOT_PRECISION)

    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    # A finite floor keeps an infinite gate's zero decay out of 0 * inf below.
    b_g = tl.maximum(-b_A * softplus_x, -1e30)
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    anc = ((bits[:, None] >> nodes[None, :].to(tl.int64)) & 1) != 0
    path_gates = tl.where(anc, b_g[None, :], 0.0)
    gate = tl.sum(path_gates, 1)
    # Sum each segment's gates exactly: the gates on t's path off s's path.
    segment = tl.dot(
        path_gates, tl.trans(tl.where(anc, 0.0, 1.0)), input_precision="ieee"
    )
    decay = tl.where(anc, tl.exp(segment), 0.0)
    m = tl.where(
        anc & (nodes[:, None] != nodes[None, :]), b_beta[:, None] * decay * kk, 0.0
    )
    # (I + m)^-1 by doubling diagonal blocks: [[A, 0], [C, B]]^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]].
    block = nodes[:, None] ^ nodes[None, :]
    inv = (block == 0).to(tl.float32) - tl.where(block == 1, m, 0.0)
    for level in tl.static_range(1, LOG2_T_BLOCK):
        c = tl.where((block >> level) == 1, m, 0.0)
        inv -= tl.dot(
            inv,
            tl.dot(c, inv, input_precision=DOT_PRECISION),
            input_precision=DOT_PRECISION,
        )

    decay_h0 = tl.exp(gate)[:, None]
    # A vanished decay drops the h0 terms even where an extreme state overflowed them.
    kh = tl.where(decay_h0 > 0, decay_h0 * kh, 0.0)
    qh = tl.where(decay_h0 > 0, decay_h0 * qh, 0.0)
    u = tl.dot(inv, b_beta[:, None] * (b_v - kh), input_precision=DOT_PRECISION)
    b_o = qh + tl.dot(decay * qk, u, input_precision=DOT_PRECISION)
    tl.store(
        o + ((i_n * T + nodes[:, None]) * HV + i_hv) * V + o_v[None, :],
        b_o.to(o.dtype.element_ty),
        mask=valid[:, None] & mask_v[None, :],
    )


def _launch_gdn_tree_verify_chunked(
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
    tree_ancestors: torch.Tensor,
    replay_payload: torch.Tensor | None,
) -> torch.Tensor:
    enable_pdl = pdl_enabled()
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5
    if replay_payload is not None and (
        replay_payload.shape != (B * T, H * K + HV * V + 2 * HV)
        or not replay_payload.is_contiguous()
        or {replay_payload.dtype, k.dtype, v.dtype, a.dtype, b.dtype} != {q.dtype}
    ):
        raise ValueError(
            "replay_payload must be a contiguous [B * T, H * K + HV * V + 2 * HV] "
            "tensor of the dtype of q, k, v, a and b"
        )
    o = q.new_empty(B, T, HV, V)
    # Once the (request, head) programs outnumber the SMs, fewer and wider V tiles are faster.
    wide = B * HV > current_platform().sm_count
    BV = min(triton.next_power_of_2(V), 128 if wide else 64)
    # Half-width V tiles (>= 16 for tl.dot) fill the SMs at a few requests, up to two programs per SM.
    if BV >= 32 and triton.cdiv(V, BV // 2) * B * HV <= 2 * current_platform().sm_count:
        BV //= 2
    T_BLOCK = triton.next_power_of_2(GDN_TREE_VERIFY_CHUNKED_MAX_NODES)
    _gdn_tree_verify_chunked_kernel[(triton.cdiv(V, BV), B * HV)](
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
        tree_ancestors=tree_ancestors,
        replay=o if replay_payload is None else replay_payload,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=min(triton.next_power_of_2(K), 32),
        BV=BV,
        T_BLOCK=T_BLOCK,
        HAS_REPLAY=replay_payload is not None,
        LOG2_T_BLOCK=T_BLOCK.bit_length() - 1,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        # Three TF32 products per FP32 one keep FP32 accuracy on tensor cores.
        DOT_PRECISION="tf32x3" if current_platform().is_nvidia else "ieee",
        Q_STRIDES=q.stride(),
        K_STRIDES=k.stride(),
        V_STRIDES=v.stride(),
        A_STRIDES=a.stride(),
        B_STRIDES=b.stride(),
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
        **(
            {"maxnreg": _TREE_VERIFY_REPLAY_MAX_REGISTERS}
            if replay_payload is not None and current_platform().is_nvidia
            else {}
        ),
    )
    return o


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
    traits={
        "separate_output_state": frozenset({False, True}),
        "strided_qkv": frozenset({False, True}),
    },
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
    output_state: torch.Tensor | None = None,
    tree_ancestors: torch.Tensor | None,
    replay_payload: torch.Tensor | None = None,
) -> torch.Tensor:
    """Portable Triton fallback for ``gdn_decode_mtp`` (see
    ``flashinfer/gated_delta_rule.py`` for the shared contract). Supports both
    batch-scoped intermediate-state caching and direct per-token pool scatter
    through ``output_state_indices``, and draft trees: ReplaySSM trees given no
    ``intermediate_states_buffer`` in the chunked form, which also fills
    ``replay_payload``, the others step by step.
    """
    if (
        tree_ancestors is not None
        and output_state_indices is None
        and intermediate_states_buffer is None
    ):
        return _launch_gdn_tree_verify_chunked(
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
            tree_ancestors=tree_ancestors,
            replay_payload=replay_payload,
        )
    if replay_payload is not None:
        raise ValueError("only a chunked ReplaySSM tree verify fills replay_payload")
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
        output_state=output_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        disable_state_update=disable_state_update,
        output_state_indices=None,
        intermediate_states_buffer=intermediate_states_buffer,
        per_token_output_state_indices=output_state_indices,
        tree_ancestors=tree_ancestors,
    )


@register_kernel(
    "attention",
    "gdn_chain_verify",
    name="triton_gdn_chain_verify",
    solution="triton",
    # Validated (and selected) on NVIDIA only; its small 16-bit dots are untested on AMD.
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
)
def triton_gdn_chain_verify(
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
) -> torch.Tensor:
    """Read-only chain verify in the chunked form (see ``gdn_chain_verify``)."""
    return launch_gdn_chain_verify(
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
    )


# ===-----------------------------------------------------------------------===#
# ReplaySSM accepted-prefix commit
# ===-----------------------------------------------------------------------===#


@gluon.jit(do_not_specialize=["B", "T", "PAYLOAD_LAYER_STRIDE"])
def _gdn_replay_commit_kernel(
    payload,
    parameters,
    state_addresses,
    state_row_strides,
    read_indices,
    write_indices,
    accepted_length,
    B,
    T,
    PAYLOAD_LAYER_STRIDE,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    STATE_DTYPE_CODE: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    WARP_SIZE: gl.constexpr,
    THREADS_V: gl.constexpr,
    SPT_K: gl.constexpr,
    TOKENS_PER_CHUNK: gl.constexpr,
):
    """Recompute accepted GDN states with one Triton program per state tile."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Release successor setup; its wait still guards all dependent reads.
        tl.extra.cuda.gdc_launch_dependents()
    # Keep batch out of grid division so serving batch changes remain runtime.
    i_hv = (gl.program_id(0) // gl.cdiv(V, BV)) % HV
    i_v = gl.program_id(0) % gl.cdiv(V, BV)
    i_k = 0
    i_n = gl.program_id(0) // (HV * gl.cdiv(V, BV))
    i_l = gl.program_id(1)
    i_h = i_hv // (HV // H)

    # Adjacent CTAs cover contiguous V tiles; lanes own contiguous K.
    # Warps own independent V columns.
    layout: gl.constexpr = gl.BlockedLayout(
        [SPT_K, 1], [WARP_SIZE // THREADS_V, THREADS_V], [1, gl.num_warps()], [0, 1]
    )
    o_k = i_k * BK + gl.arange(0, BK, layout=gl.SliceLayout(1, layout))
    o_v = i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, layout))
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    key_width: gl.constexpr = H * K
    value_width: gl.constexpr = HV * V
    payload_width: gl.constexpr = key_width + value_width + 2 * HV
    layer_base = i_l.to(gl.int64) * PAYLOAD_LAYER_STRIDE
    request_base = layer_base + i_n.to(gl.int64) * T * payload_width

    layer_request = i_l * B + i_n
    read_idx = gl.load(read_indices + layer_request).to(gl.int64)
    steps = gl.load(accepted_length + i_n).to(gl.int32)
    steps = gl.minimum(gl.maximum(steps, 0), T)
    state_address = gl.load(state_addresses + i_l)
    if STATE_DTYPE_CODE == 0:
        state_pool = state_address.to(gl.pointer_type(gl.bfloat16))
    elif STATE_DTYPE_CODE == 1:
        state_pool = state_address.to(gl.pointer_type(gl.float16))
    else:
        state_pool = state_address.to(gl.pointer_type(gl.float32))
    state_row_stride = gl.load(state_row_strides + i_l).to(gl.int64)
    parameter_base = i_l * 2 * HV + i_hv
    raw_A = gl.load(parameters + parameter_base).to(gl.float32)
    b_dt_bias = gl.load(parameters + parameter_base + HV).to(gl.float32)
    write_idx = gl.load(write_indices + layer_request).to(gl.int64)
    # Pool row strides can break alignment; check each actual row address.
    STATE_BITS: gl.constexpr = 32 if STATE_DTYPE_CODE == 2 else 16
    PACK: gl.constexpr = 64 // STATE_BITS
    PACK_WORDS: gl.constexpr = SPT_K // PACK
    STATE_ALIGNMENT: gl.constexpr = 8 * PACK_WORDS
    if STATE_DTYPE_CODE == 2 and K % PACK == 0:
        # Each 64-bit word holds two FP32 elements.
        packed_3d: gl.constexpr = gl.BlockedLayout(
            [PACK_WORDS, PACK, 1],
            [WARP_SIZE // THREADS_V, 1, THREADS_V],
            [1, 1, gl.num_warps()],
            [1, 0, 2],
        )
        packed_layout: gl.constexpr = gl.SliceLayout(1, packed_3d)
        packed_k = i_k * (BK // PACK) + gl.arange(
            0, BK // PACK, layout=gl.SliceLayout(1, packed_layout)
        )
        packed_v = i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, packed_layout))
        pack_shift = (
            gl.arange(0, PACK, layout=gl.SliceLayout(0, gl.SliceLayout(2, packed_3d)))
            * STATE_BITS
        )
        packed_mask = (packed_k < K // PACK)[:, None] & (packed_v < V)[None, :]
    b_h = gl.zeros([BK, BV], dtype=gl.float32, layout=layout)
    if read_idx >= 0:
        base = state_pool + read_idx * state_row_stride + i_hv * V * K
        if K % PACK == 0:
            if (base.to(gl.uint64) & (STATE_ALIGNMENT - 1)) == 0:
                # Unsigned pointers isolate alignment hints from the fallback.
                if STATE_DTYPE_CODE == 2:
                    packed_ptr = base.to(gl.pointer_type(gl.uint64))
                    packed_ptr = gl.multiple_of(packed_ptr, STATE_ALIGNMENT)
                    raw = gl.load(
                        packed_ptr
                        + packed_v[None, :] * (K // PACK)
                        + packed_k[:, None],
                        mask=packed_mask,
                        other=0,
                    )
                    bits = raw[:, None, :] >> pack_shift[None, :, None]
                    values = bits.to(gl.uint32).to(gl.float32, bitcast=True)
                    b_h = gl.convert_layout(gl.reshape(values, [BK, BV]), layout)
                else:
                    aligned_ptr = base.to(gl.pointer_type(gl.uint16))
                    aligned_ptr = gl.multiple_of(aligned_ptr, STATE_ALIGNMENT)
                    raw = gl.load(
                        aligned_ptr + o_v[None, :] * K + o_k[:, None],
                        mask=mask_h,
                        other=0,
                    )
                    if STATE_DTYPE_CODE == 0:
                        b_h = raw.to(gl.bfloat16, bitcast=True).to(gl.float32)
                    else:
                        b_h = raw.to(gl.float16, bitcast=True).to(gl.float32)
            else:
                p_h0 = base + o_v[None, :] * K + o_k[:, None]
                b_h = gl.load(p_h0, mask=mask_h, other=0).to(gl.float32)
        else:
            p_h0 = base + o_v[None, :] * K + o_k[:, None]
            b_h = gl.load(p_h0, mask=mask_h, other=0).to(gl.float32)

    b_A = -gl.exp(raw_A)
    # Preserve normalization order while matching shared reads to state packing.
    key_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2 if STATE_DTYPE_CODE == 2 else 4],
        [THREADS_V, WARP_SIZE // THREADS_V],
        [gl.num_warps(), 1],
        [1, 0],
    )
    key_read_layout: gl.constexpr = gl.BlockedLayout(
        [1, SPT_K], [THREADS_V, WARP_SIZE // THREADS_V], [gl.num_warps(), 1], [1, 0]
    )
    value_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [4, WARP_SIZE // 4], [1, gl.num_warps()], [1, 0]
    )
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    shared_k = gl.allocate_shared_memory(
        gl.float32, [TOKENS_PER_CHUNK, BK], shared_layout
    )
    shared_v = gl.allocate_shared_memory(
        gl.float32, [TOKENS_PER_CHUNK, BV], shared_layout
    )
    key_nodes = gl.arange(0, TOKENS_PER_CHUNK, layout=gl.SliceLayout(1, key_layout))
    keys = gl.arange(0, BK, layout=gl.SliceLayout(0, key_layout))
    value_nodes = gl.arange(0, TOKENS_PER_CHUNK, layout=gl.SliceLayout(1, value_layout))
    values = i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, value_layout))
    for start in range(0, steps, TOKENS_PER_CHUNK):
        key_tokens = start + key_nodes
        key_rows = payload + request_base + key_tokens.to(gl.int64) * payload_width
        keys_raw = gl.load(
            key_rows[:, None] + i_h * K + keys[None, :],
            mask=(key_tokens < steps)[:, None] & (keys < K)[None, :],
            other=0,
        ).to(gl.float32)
        keys_normalized = (
            keys_raw / gl.sqrt(gl.sum(keys_raw * keys_raw, 1) + 1e-6)[:, None]
        )
        shared_k.store(keys_normalized)
        value_tokens = start + value_nodes
        value_rows = payload + request_base + value_tokens.to(gl.int64) * payload_width
        values_raw = gl.load(
            value_rows[:, None] + key_width + i_hv * V + values[None, :],
            mask=(value_tokens < steps)[:, None] & (values < V)[None, :],
            other=0,
        ).to(gl.float32)
        shared_v.store(values_raw)
        raw_a = gl.load(
            key_rows + key_width + value_width + i_hv, mask=key_tokens < steps, other=0
        ).to(gl.float32)
        raw_b = gl.load(
            key_rows + key_width + value_width + HV + i_hv,
            mask=key_tokens < steps,
            other=0,
        ).to(gl.float32)
        x = raw_a + b_dt_bias
        softplus = gl.where(x <= 20.0, gl.log(1.0 + gl.exp(x)), x)
        decay = gl.exp(b_A * softplus)
        beta = 1.0 / (1.0 + gl.exp(-raw_b))
        gate_layout: gl.constexpr = gl.BlockedLayout(
            [1], [WARP_SIZE], [gl.num_warps()], [0]
        )
        decay = gl.convert_layout(decay, gate_layout)
        beta = gl.convert_layout(beta, gate_layout)
        gl.barrier()
        for step in gl.static_range(0, TOKENS_PER_CHUNK):
            if start + step < steps:
                index = gl.full([1], step, gl.int32, layout=gate_layout)
                b_decay = gl.sum(gl.gather(decay, index, 0), 0)
                b_beta = gl.sum(gl.gather(beta, index, 0), 0)
                b_k = gl.convert_layout(
                    gl.reshape(shared_k.slice(step, 1).load(key_read_layout), [BK]),
                    gl.SliceLayout(1, layout),
                )
                b_v = gl.convert_layout(
                    gl.reshape(shared_v.slice(step, 1).load(layout), [BV]),
                    gl.SliceLayout(0, layout),
                )
                next_h = b_h * b_decay
                delta = b_v - gl.sum(next_h * b_k[:, None], 0)
                next_h += b_k[:, None] * (delta * b_beta)[None, :]
                b_h = next_h
        # Finish every consumer before overwriting the next shared chunk.
        gl.barrier()

    if write_idx >= 0:
        base = state_pool + write_idx * state_row_stride + i_hv * V * K
        if K % PACK == 0:
            if (base.to(gl.uint64) & (STATE_ALIGNMENT - 1)) == 0:
                # Unsigned pointers isolate alignment hints from the fallback.
                if STATE_DTYPE_CODE == 2:
                    packed_ptr = base.to(gl.pointer_type(gl.uint64))
                    packed_ptr = gl.multiple_of(packed_ptr, STATE_ALIGNMENT)
                    packed_values = gl.convert_layout(
                        gl.reshape(b_h, [BK // PACK, PACK, BV]), packed_3d
                    )
                    bits = packed_values.to(gl.uint32, bitcast=True).to(gl.uint64)
                    raw = gl.sum(bits << pack_shift[None, :, None], 1)
                    gl.store(
                        packed_ptr
                        + packed_v[None, :] * (K // PACK)
                        + packed_k[:, None],
                        raw,
                        mask=packed_mask,
                    )
                else:
                    aligned_ptr = base.to(gl.pointer_type(gl.uint16))
                    if STATE_DTYPE_CODE == 0:
                        bits = b_h.to(gl.bfloat16).to(gl.uint16, bitcast=True)
                    else:
                        bits = b_h.to(gl.float16).to(gl.uint16, bitcast=True)
                    aligned_ptr = gl.multiple_of(aligned_ptr, STATE_ALIGNMENT)
                    gl.store(
                        aligned_ptr + o_v[None, :] * K + o_k[:, None], bits, mask=mask_h
                    )
            else:
                p_out = base + o_v[None, :] * K + o_k[:, None]
                gl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=mask_h)
        else:
            p_out = base + o_v[None, :] * K + o_k[:, None]
            gl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=mask_h)


@register_kernel(
    "attention",
    "gdn_replay_commit",
    name="triton_gdn_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    traits={"flat_state": frozenset({True})},
)
def triton_gdn_replay_commit(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    state_dtype: torch.dtype,
) -> None:
    """Replay all GDN layers and write their final accepted states.

    This portable Triton implementation is shared by every registered GPU
    vendor.

    Args:
        payload: Packed K/V/a/b storage shaped ``[L, rows, width]``.
        parameters: Layer-major A_log/dt_bias table shaped ``[L, 2, HV]``.
        state_addresses: Recurrent-state pool base addresses shaped ``[L]``.
        state_row_strides: Pool row strides in elements, shaped ``[L]``.
        read_indices: Committed-state pages shaped ``[L, B]``.
        write_indices: Accepted-state destination pages shaped ``[L, B]``.
        accepted_length: Number of verified tokens to replay per request,
            shaped ``[B]`` and bounded by ``T``.
        draft_token_num: Verify-window length ``T``.
        num_k_heads: Key-head count ``H``.
        num_v_heads: Value/state-head count ``HV``.
        head_k_dim: Key/state dimension ``K``.
        head_v_dim: Value/state dimension ``V``.
        state_dtype: Common recurrent-state element dtype.

    Returns:
        ``None``. Every layer's accepted state is written through its pool
        address in one kernel launch.
    """
    enable_pdl = pdl_enabled()
    wide_state = (
        current_platform().is_nvidia
        and state_dtype != torch.float32
        and head_k_dim == 128
    )
    num_layers = payload.shape[0]
    batch_size = accepted_length.numel()
    state_dtype_codes = {
        torch.bfloat16: 0,
        torch.float16: 1,
        torch.float32: 2,
    }
    if state_dtype not in state_dtype_codes:
        raise ValueError(f"unsupported GDN replay state dtype: {state_dtype}")

    BK = triton.next_power_of_2(head_k_dim)
    BV = min(triton.next_power_of_2(head_v_dim), 64)
    NK, NV = triton.cdiv(head_k_dim, BK), triton.cdiv(head_v_dim, BV)
    if NK != 1:
        raise ValueError("GDN replay does not support head dimensions above one tile")
    _gdn_replay_commit_kernel[(batch_size * num_v_heads * NV, num_layers)](
        payload=payload,
        parameters=parameters,
        state_addresses=state_addresses,
        state_row_strides=state_row_strides,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        B=batch_size,
        T=draft_token_num,
        PAYLOAD_LAYER_STRIDE=payload.stride(0),
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BK=BK,
        BV=BV,
        STATE_DTYPE_CODE=state_dtype_codes[state_dtype],
        num_warps=4,
        num_stages=3,
        ENABLE_PDL=enable_pdl,
        TOKENS_PER_CHUNK=8,
        WARP_SIZE=triton.runtime.driver.active.get_current_target().warp_size,
        THREADS_V=4,
        SPT_K=(
            (4 if current_platform().is_nvidia and head_k_dim == 128 else 2)
            if state_dtype == torch.float32
            else (8 if wide_state else 4)
        ),
        **({"launch_pdl": True} if enable_pdl else {}),
        **({"maxnreg": 128, "sched4reg": True} if wide_state else {}),
    )
