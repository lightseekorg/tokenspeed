# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""Registered MXFP4 MoE apply using MORI all-to-all EP.

Native MXFP4 path (NO dequant): weights are swizzled once at load via the shared
``triton_mxfp4_moe_weights`` preprocessor; each forward MORI dispatches bf16 hidden
states, the tokens (already grouped per local expert in MORI's 3D [E,cap,H] buffer)
are gathered into a contiguous expert-sorted buffer and run through the triton mxfp4
grouped-GEMM (``grouped_mxfp4_ffn``) directly on the packed MXFP4 weights, scattered
back into the 3D buffer, and combined. Selected for Kimi-K2.5-MXFP4 with
--enable-expert-parallel --all2all-backend mori.
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
    from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

    with redirect_triton_to_tokenspeed_triton():
        from triton_kernels.tensor import make_ragged_tensor_metadata

    from tokenspeed_kernel.ops.communication.mori_ep import get_dispatcher
    from tokenspeed_kernel.ops.moe.triton.mxfp4 import (
        grouped_mxfp4_ffn,
        triton_mxfp4_moe_weights,
    )

    def _grouped_mxfp4_gemm_3d(
        packed_x: torch.Tensor,
        counts: torch.Tensor,
        w: torch.nn.Module,
        n_recv_bound: int,
    ) -> None:
        """Native MXFP4 grouped SwiGLU FFN over MORI's 3D [E,cap,H] padded buffer, IN PLACE.

        Bridges MORI's padded 3D layout to the triton ragged (contiguous, expert-sorted)
        layout with no host sync under HIP-graph capture, so the whole MoE forward captures.
        ``n_recv_bound`` is a STATIC (shape-derived, not ``.item()``) upper bound on the rows
        this rank received this step, used only while capturing; the ragged buffer is sized to
        ``m = min(E*cap, received)`` so the gather/scatter + GEMM stay proportional to real work
        rather than the full E*cap capacity (the decisive factor for graph-mode decode
        throughput -- sizing to E*cap moves ~E*cap rows and launches a padded-capacity grid
        every layer).

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

        # Ragged-row count ``m`` (>= sum(counts)). Under HIP-graph capture host syncs are
        # forbidden, so use the static ``n_recv_bound`` -- a true upper bound there because the
        # decode graph pads every rank to the SAME batch (global tokens = ep_size * batch), so
        # this rank receives at most ep_size*batch*top_k rows. Outside capture (eager, incl.
        # non-uniform dp batches) a sync is fine, so use the EXACT count -- always correct and
        # tighter. Either way ``m`` bounds the gather/scatter + GEMM to real work, not E*cap.
        if torch.cuda.is_current_stream_capturing():
            m = min(n_slots, n_recv_bound)
        else:
            total_i = int(total.item())
            if total_i == 0:
                return
            m = min(n_slots, total_i)

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
        y_flat = grouped_mxfp4_ffn(x_flat, meta, w).to(
            packed_x.dtype
        )  # first sum(counts) rows set

        # Scatter back IN PLACE. Comparing against the 0-dim ``total`` needs no host sync;
        # gather_idx is a permutation prefix, so valid rows land in their slots collision-free
        # and padding rows (row >= total, only present when m > sum(counts) under capture) write
        # their own original value back -- left exactly as-is for MORI combine.
        flat[gather_idx] = torch.where((row < total)[:, None], y_flat, x_flat)

    @register_kernel(
        "moe",
        "apply",
        name="mori_ep_mxfp4_moe_apply",
        solution="mori",
        weight_preprocessor=triton_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            # silu only: grouped_mxfp4_ffn computes plain silu(gate)*up and does not apply
            # swiglu_arg/alpha/limit, so do not advertise "swiglu" (would select this kernel
            # for clamped-swiglu configs and silently compute the wrong activation).
            "activation": frozenset({"silu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input", "fp8"}),
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
        # Static upper bound on rows this rank can receive: global tokens (ep_size * this
        # step's batch, uniform across ranks under decode graph capture) times top_k. Derived
        # from x.shape (no host sync) so the GEMM stays graph-capturable while its ragged grid
        # tracks real work instead of the full E_local*cap buffer.
        n_recv_bound = ep_size * int(x.shape[0]) * int(getattr(w, "top_k"))
        _grouped_mxfp4_gemm_3d(packed, handle["counts"], w, n_recv_bound)  # in place
        # Return the COMPLETE per-token routed result on every rank; do NOT reduce/scale here.
        return dispatcher.combine(handle)
