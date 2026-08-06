# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""Registered MXFP4 MoE apply using MORI all-to-all EP.

Native MXFP4 path (NO dequant): weights are prepared once at load via the gfx950 Gluon
MXFP4 preprocessor; each forward MORI dispatches bf16 hidden states, the tokens (already
grouped per local expert in MORI's 3D [E,cap,H] buffer) are gathered into a contiguous
expert-sorted buffer and run through the gfx950 Gluon grouped MXFP4 SwiGLU FFN
(``_grouped_mxfp4_swiglu_ffn`` -> ``gluon_mxfp_dispatch_swiglu`` + ``gluon_mxfp_combine``)
directly on the packed MXFP4 weights, scattered back into the 3D buffer, and combined.
Selected for Kimi-K2.5-MXFP4 with --enable-expert-parallel --all2all-backend mori.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_amd:
    from tokenspeed_kernel.ops.communication.mori_ep import get_dispatcher
    from tokenspeed_kernel_amd.ops.gfx950.moe._common import (
        make_ragged_tensor_metadata,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
        _quantize_mxfp4_activation,
        gluon_mxfp_combine,
        gluon_mxfp_dispatch_swiglu,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (
        preprocess_gluon_mxfp4_gfx950_moe_weights,
    )

    def _mori_mxfp4_weight_preprocessor(plan: dict, w: torch.nn.Module) -> None:
        """Prepare MXFP4 expert weights in the gfx950 Gluon layout the grouped primitives
        below consume: K-packed uint8 storage + swizzled CDNA4 E8M0 scales, gate/up
        interleaved, Gluon-preshuffled. Sets ``w13_weight_triton_tensor`` /
        ``w2_weight_triton_tensor`` and their ``*_precision_config.b_mx_scale``."""
        return preprocess_gluon_mxfp4_gfx950_moe_weights(plan, w, preshuffle=True)

    def _swiglu_args(w: torch.nn.Module) -> tuple[float, float, float]:
        """Gate activation params for the SwiGLU reducer. Mirrors
        ``ops.moe.gluon.mxfp4._swiglu_args``: default ``(alpha=1, limit=0, beta=0)`` is an
        unclamped SiLU gate times the linear branch (what Kimi-K2.5 uses)."""
        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is None:
            return 1.0, 0.0, 0.0
        swiglu_beta = getattr(w, "swiglu_beta", None)
        return (
            1.0 if swiglu_arg.alpha is None else swiglu_arg.alpha,
            0.0 if swiglu_arg.limit is None else swiglu_arg.limit,
            0.0 if swiglu_beta is None else swiglu_beta,
        )

    def _grouped_mxfp4_swiglu_ffn(
        x_flat: torch.Tensor, meta, w: torch.nn.Module
    ) -> torch.Tensor:
        """Pure grouped MXFP4 SwiGLU FFN over expert-sorted CONTIGUOUS rows, on the gfx950
        Gluon primitives. ``x_flat`` [total, H] bf16 (rows grouped by local expert per
        ``meta``); returns [total, H] bf16. No routing / gather / scatter / gate-weighting --
        MORI owns dispatch/combine and applies the gate weights in combine.

        GEMM1 = ``gluon_mxfp_dispatch_swiglu`` (gate/up matmul + SiLU, requantized to mxfp4);
        GEMM2 = ``gluon_mxfp_combine`` with ``scatter_indx=None`` (down-proj, output left in
        the same expert-sorted ragged order as the input). This mirrors the standard dynamic
        path (_gluon_mxfp_dynamic_mxfp4_fused_moe_from_route) minus the routing indices.
        """
        alpha, limit, beta = _swiglu_args(w)
        w13 = _extract_gluon_raw_w(w.w13_weight_triton_tensor)
        s13 = _extract_gluon_raw_s(w.w13_precision_config.b_mx_scale)
        w2 = _extract_gluon_raw_w(w.w2_weight_triton_tensor)
        s2 = _extract_gluon_raw_s(w.w2_precision_config.b_mx_scale)
        w13_bias = (
            None
            if getattr(w, "_gluon_w13_bias_is_zero", False)
            else getattr(w, "w13_weight_bias", None)
        )
        w2_bias = (
            None
            if getattr(w, "_gluon_w2_bias_is_zero", False)
            else getattr(w, "w2_weight_bias", None)
        )

        gemm1_input, gemm1_scale = _quantize_mxfp4_activation(
            x_flat, gather_indx=None, ragged_metadata=meta
        )
        inter, inter_scale = gluon_mxfp_dispatch_swiglu(
            gemm1_input,
            w13,
            s13,
            x_scale=_extract_gluon_raw_s(gemm1_scale),
            x_format="e2m1",
            x_global_scale=1.0,
            bias=w13_bias,
            a_ragged_metadata=meta,
            gather_indx=None,
            out_dtype=torch.bfloat16,
            swiglu_alpha=alpha,
            swiglu_limit=limit,
            swiglu_beta=beta,
            scale_load_mode="swizzle",
            w_transpose=True,
            out_quant_scale=None,
            out_quant_format="mxfp4",
            w_preshuffle=bool(getattr(w13, "is_shuffled_for_gluon_dot", False)),
            x_scale_ragged_padded=True,
        )
        return gluon_mxfp_combine(
            inter.view(torch.uint8) if inter.dtype != torch.uint8 else inter,
            w2,
            s2,
            x_scale=_extract_gluon_raw_s(inter_scale),
            x_format="e2m1",
            x_global_scale=1.0,
            bias=w2_bias,
            a_ragged_metadata=meta,
            scatter_indx=None,
            gate_scal=None,
            n_tokens=None,
            n_expts_act=None,
            out_dtype=torch.bfloat16,
            scale_load_mode="swizzle",
            w_transpose=True,
            w_preshuffle=bool(getattr(w2, "is_shuffled_for_gluon_dot", False)),
            x_scale_ragged_padded=True,
        )

    def _grouped_mxfp4_gemm_3d(
        packed_x: torch.Tensor,
        counts: torch.Tensor,
        w: torch.nn.Module,
        n_recv_bound: int,
    ) -> None:
        """Native MXFP4 grouped SwiGLU FFN over MORI's 3D [E,cap,H] padded buffer, IN PLACE.

        Bridges MORI's padded 3D layout to the triton ragged (contiguous, expert-sorted)
        layout with NO host sync (in eager OR under HIP-graph capture), so the whole MoE forward
        captures and the eager path never drains the GPU pipeline per layer. ``n_recv_bound`` is
        a STATIC (shape-derived, not ``.item()``) upper bound on the rows this rank received this
        step; the ragged buffer is sized to ``m = min(E*cap, n_recv_bound)`` so the gather/scatter
        + GEMM stay proportional to real work rather than the full E*cap capacity (the decisive
        factor for graph-mode decode throughput -- sizing to E*cap moves ~E*cap rows and launches
        a padded-capacity grid every layer).

        Build a fixed-size [E*cap] permutation of slots (valid rows -- local < counts[e] --
        first in expert-major order, padding rows after), take the first ``m`` (>= sum(counts))
        as the ragged buffer, gather, run the mxfp4 grouped GEMM on the raw packed weights (no
        dequant) with the real per-expert counts, then scatter back into ``packed_x`` itself.
        Padding rows are written back unchanged; MORI combine reads only valid rows via
        src_info, so padding is ignored. Pure FFN; combine applies the gate weights.
        """
        E, cap, H = packed_x.shape
        dev = packed_x.device
        n_slots = E * cap
        counts_long = counts.to(torch.long)
        total = counts_long.sum()  # sum(counts), 0-dim tensor (no host sync)

        # Ragged-row count ``m`` (>= sum(counts)): the static, shape-derived ``n_recv_bound``
        # (ep_size * max-tokens-per-rank * top_k, capped at E*cap; see the caller) is a true upper
        # bound on the rows this rank can receive -- every rank dispatches at most
        # ``max_tokens_per_rank*top_k`` slots, so no rank receives more than ep_size of those.
        # Use it in BOTH capture AND eager. The obvious "tighter"
        # eager choice ``sum(counts).item()`` needs a device->host sync, and because the MoE runs
        # this per layer (x61) that sync DRAINS the GPU pipeline every layer -- on the eager
        # prefill path (batches above the captured decode sizes) that serialization, not compute,
        # was the dominant cost: multi-second worker stalls -> gateway health timeouts and mass
        # request failures at concurrency. The ragged GEMM is bounded by the device-side per-expert
        # ``counts`` (via ``meta``) regardless of ``m``, so an ``m`` above sum(counts) only widens
        # the gather buffer (cheap bandwidth), never the GEMM work; the padding rows (row >= total)
        # are written back unchanged by the device-side ``row < total`` mask in the scatter below.
        # For decode ``n_recv_bound`` is already tight (small batch); for prefill the buffer
        # genuinely fills, so it is near-exact there too.
        m = min(n_slots, n_recv_bound)
        if m == 0:
            return

        # Fixed-size [E*cap] permutation of slots -> valid rows (local < counts[e]) first in
        # expert-major order, padding rows after; take the first ``m`` as the ragged buffer.
        # argsort on a per-slot key -- no ``.item()``, no data-dependent shape -- so the op
        # stays HIP-graph capturable; the [:m] prefix is still collision-free (a permutation
        # prefix), so the scatter below never double-writes a slot.
        slot = torch.arange(n_slots, device=dev)  # index into flat [E,cap]
        e_of_slot = slot // cap
        l_of_slot = slot - e_of_slot * cap
        is_pad = (l_of_slot >= counts_long[e_of_slot]).to(torch.int64)
        gather_idx = torch.argsort(is_pad * n_slots + slot)[:m]  # [m] unique slots
        row = torch.arange(m, device=dev)

        flat = packed_x.reshape(n_slots, H)
        x_flat = flat[gather_idx]  # [m,H] valid front, pad back
        meta = make_ragged_tensor_metadata(counts.to(torch.int32), m)
        y_flat = _grouped_mxfp4_swiglu_ffn(x_flat, meta, w).to(
            packed_x.dtype
        )  # first sum(counts) rows set

        # Scatter back IN PLACE. Comparing against the 0-dim ``total`` needs no host sync;
        # gather_idx is a permutation prefix, so valid rows land in their slots collision-free
        # and padding rows (row >= total, present whenever m > sum(counts) -- the common case now
        # that ``m`` is the static bound) write their own original value back -- left exactly
        # as-is for MORI combine.
        flat[gather_idx] = torch.where((row < total)[:, None], y_flat, x_flat)

    @register_kernel(
        "moe",
        "apply",
        name="mori_ep_mxfp4_moe_apply",
        solution="mori",
        weight_preprocessor=_mori_mxfp4_weight_preprocessor,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            # Kimi-K2.5 uses an unclamped SiLU gate (_swiglu_args default alpha=1,limit=0,
            # beta=0). The dispatch_swiglu reducer does support clamped SwiGLU via swiglu_arg,
            # but keep selection scoped to "silu" (the validated config) here.
            "activation": frozenset({"silu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({1}),
            # The gfx950 Gluon path quantizes activations (and the intermediate) to MXFP4
            # dynamically (_quantize_mxfp4_activation -> e2m1), so the internal activation is
            # mxfp4 -- matches the planner's requested trait for MXFP4 EP.
            "internal_activation_dtype": frozenset({"mxfp4"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def mori_ep_mxfp4_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
        low_latency: bool | None = None,
        overlap_fn: Callable[[], None] | None = None,
    ):
        # ``low_latency`` / ``overlap_fn`` are forwarded to every all-to-all apply kernel
        # (DeepEP low-latency contract). MORI has a single dispatch/combine path, so
        # ``low_latency`` is accepted and ignored; ``overlap_fn`` (if any) is run after the
        # dispatch send, mirroring the DeepEP kernels.
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        ep_size = int(getattr(w, "ep_size", dist.get_world_size()))
        ep_rank = int(getattr(w, "ep_rank", dist.get_rank()))
        num_local = int(getattr(w, "num_local_experts"))
        # MORI dispatch capacity (tokens/rank). MORI allocates symmetric buffers
        # ~[num_local, ws*cap, H] at op-creation, so cap must upper-bound tokens/rank. Derive
        # it from the runtime's per-GPU token capacity (falling back to this step's token
        # count) so prefills up to that capacity fit -- a prefill above cap overflows the
        # buffer. ``MORI_EP_MAX_TOKENS_PER_RANK`` is an explicit override.
        import os as _os

        _cap_env = _os.environ.get("MORI_EP_MAX_TOKENS_PER_RANK")
        cap = (
            int(_cap_env)
            if _cap_env
            else max(int(max_num_tokens_per_gpu or 0), int(x.shape[0]))
        )
        dispatcher = get_dispatcher(
            rank=ep_rank,
            world_size=ep_size,
            hidden_dim=x.shape[1],
            num_local_experts=num_local,
            num_experts_per_token=int(getattr(w, "top_k")),
            max_num_inp_token_per_rank=cap,
            data_type=torch.bfloat16,
        )

        handle = dispatcher.dispatch(x, topk_weights.float(), topk_ids)
        if overlap_fn is not None:
            overlap_fn()
        packed = handle["packed_x"]  # [E_local, cap, H]
        # Static upper bound on rows THIS rank can receive after dispatch: (max tokens on any EP
        # rank this forward) * ep_size senders * top_k. It must NOT be derived from this rank's
        # own token count ``x.shape[0]``: after dispatch a rank owns experts for tokens sent by
        # every OTHER rank, so under DP attention an idle/imbalanced rank (few or zero local
        # tokens while peers are busy) still receives their tokens -- a local-only bound would
        # truncate (or, at x.shape[0]==0, skip) those rows, leaving raw dispatched input in place
        # of the FFN output and corrupting the combine result on the sender ranks. ``max_num_
        # tokens_per_gpu`` is the per-rank max the runtime reports (identical on every rank), so
        # this stays a true upper bound under eager imbalance and equals ep_size*batch under the
        # uniform decode-graph padding -- all without a host sync (keeps the GEMM capturable).
        per_rank = (
            int(max_num_tokens_per_gpu) if max_num_tokens_per_gpu else int(x.shape[0])
        )
        n_recv_bound = ep_size * per_rank * int(getattr(w, "top_k"))
        _grouped_mxfp4_gemm_3d(packed, handle["counts"], w, n_recv_bound)  # in place
        # Return the COMPLETE per-token routed result on every rank; do NOT reduce/scale here.
        return dispatcher.combine(handle)
