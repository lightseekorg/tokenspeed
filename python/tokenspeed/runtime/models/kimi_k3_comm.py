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

"""Kimi-K3 communication routing and collective workspace ownership.

BT/HT finalize and normalize routed latents before a sharded up-projection.
The shared all-reduce also gathers the projection shards and adds the residual.
Small graph batches retain the multicast tail; other layouts use separate reduce.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.activation.triton import add3
from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
    MNNVLCuteDSLBTFinalizeTuning,
    MNNVLCuteDSLFinalizeAllReduceRMSNorm,
    MNNVLCuteDSLHTFinalizeAllReduceRMSNorm,
    MNNVLCuteDSLHTFinalizeTuning,
    mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported,
    mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported,
)
from tokenspeed_kernel.ops.communication.multimem import (
    multimem_available,
)
from tokenspeed_kernel.ops.moe.latent_tail import (
    KimiK3LatentTailOp,
    attn_reduce_shape_supported,
    build_attn_reduce_collective,
    latent_tail_supported,
    multicast_backend_available,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    prepare_all_reduce_buffers,
    prepare_all_reduce_fusion,
    prepare_all_reduce_lane,
)
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.layers.layernorm import RMSNorm, _get_process_group
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

_IRIS_MAX_TOKENS = 8192
_IRIS_BASELINE_PRODUCER_DIRECT_MAX_TOKENS = 48

# Widest reduce this instance is built for; it becomes the collective's max_m.
ATTN_AR_MAX_TOKENS = 8


def attn_ar_eligible(
    *, armed: bool, has_prefix: bool, num_tokens: int, fusion_max_tokens: int
) -> bool:
    """Whether the tokenspeed collective, not the vendor AR, serves this reduce.

    ``fusion_max_tokens`` is the operator's window; it goes negative to forbid a
    fused attention all-reduce outright, and this path is one.
    """
    window = min(ATTN_AR_MAX_TOKENS, fusion_max_tokens)
    return armed and has_prefix and 0 < num_tokens <= window


class K3MoETailTier(IntEnum):
    """How the K3 MoE tail combines routed/shared partials, best first.

    Declaration order *is* the priority order (mirrors the selector's
    branch order). Values never escape the process (identity comparisons
    only, no serialization), so inserting mid-list is safe — keep new
    tiers at their semantic rank rather than appending.
    """

    TAIL_FUSION = 0  # fused decode kernel (aka the multicast latent tail)
    MNNVL_HT = 1  # large M: HT finalize, all-reduce, and RMSNorm
    MNNVL_BT = 2  # medium M: BT finalize, all-reduce, and RMSNorm
    SEPARATE_REDUCE = 3  # portable: reduce each partial on its own


# Measured profit edge of the fused tail; the kernel's own capacity is larger.
TAIL_FUSION_MAX_TOKENS = 32

# Continuous deferred intervals, shared by eager and graph execution.
MNNVL_BT_MIN_TOKENS = 33
MNNVL_BT_MAX_TOKENS = 1024
MNNVL_HT_MIN_TOKENS = 1025
MNNVL_HT_MAX_TOKENS = 8192


def _deferred_tail_applicable(
    *,
    mapping,
    hidden_size: int,
    latent_size: int,
    top_k: int,
    is_blackwell: bool,
    fused_moe_ar: bool,
    has_routed_norm: bool,
    shard_up_projection: bool,
    experts_supports_deferred_finalize: bool,
) -> bool:
    """Select the automatic deferred path from static model/backend traits.

    Args:
        mapping: MoE TP/EP and attention DP/CP topology.
        hidden_size: Full shared/residual width.
        latent_size: Routed latent width before up-projection.
        top_k: Number of routed experts per token.
        is_blackwell: Whether this rank uses NVIDIA SM100/SM103.
        fused_moe_ar: Whether the execution plan supports the routed AR lane.
        has_routed_norm: Whether routed RMSNorm is present.
        shard_up_projection: Whether up-projection owns a rank-local weight shard.
        experts_supports_deferred_finalize: Actual expert-backend output trait.

    Returns:
        True for the supported K3 TP8/EP1 layout. Rank agreement, communication
        support and PDL are validated collectively before allocation; failure
        must not silently select an old in-range tail. Graph capture settings
        and experimental environment variables never select the implementation.
    """
    return (
        is_blackwell
        and mapping.moe.tp_size == 8
        and mapping.moe.ep_size == 1
        and mapping.moe.tp_ep_size == 8
        and mapping.attn.dp_size == 1
        and mapping.attn.cp_size == 1
        and hidden_size == 7168
        and latent_size == 3584
        and top_k == 16
        and fused_moe_ar
        and has_routed_norm
        and shard_up_projection
        and experts_supports_deferred_finalize
    )


def select_deferred_k3_moe_tail_tier(num_tokens: int) -> K3MoETailTier | None:
    """Return the deferred interval tier; None preserves the old small path.

    Args:
        num_tokens: Actual forward M, or graph-padded M, not request count.

    Returns:
        BT for (32,1024], HT for (1024,8192], separate above
        8192, and None for empty/small forwards. No sampled-point whitelist.
    """
    if type(num_tokens) is not int or num_tokens < 0:
        raise ValueError("token count must be a nonnegative integer")
    if num_tokens <= 32:
        return None
    if num_tokens <= 1024:
        return K3MoETailTier.MNNVL_BT
    if num_tokens <= 8192:
        return K3MoETailTier.MNNVL_HT
    return K3MoETailTier.SEPARATE_REDUCE


def select_k3_moe_tail_tier(
    *,
    num_tokens: int,
    graph_phase: bool,
    tail_fusion_max_tokens: int,
) -> K3MoETailTier:
    """Select the small graph tail or the portable separate-reduce path.

    Args:
        num_tokens: Rank-uniform token count for this forward.
        graph_phase: Whether the forward runs under the CUDA-graph phase.
        tail_fusion_max_tokens: Available small-tail capacity, already capped
            at the profitable token range; zero when unavailable.

    Returns:
        TAIL_FUSION for a supported small graph forward, otherwise
        SEPARATE_REDUCE. The deferred medium/large route is selected first
        by K3MoeTailComm.plan.
    """
    if graph_phase and 1 <= num_tokens <= tail_fusion_max_tokens:
        return K3MoETailTier.TAIL_FUSION
    return K3MoETailTier.SEPARATE_REDUCE


def prepare_k3_all_reduce_buffers(
    *,
    mapping,
    hidden_size: int,
    routed_hidden_size: int,
    max_num_tokens: int,
) -> bool:
    """Prepare the node-local AMD all-reduce buffers used by Kimi-K3."""
    if not current_platform().is_cdna4:
        return False

    max_num_tokens = min(max_num_tokens, _IRIS_MAX_TOKENS)
    if max_num_tokens <= 0:
        return False

    from tokenspeed_kernel.ops.communication.triton import (
        allreduce_residual_attnres_max_tokens,
    )

    attnres_max_rows = min(
        max_num_tokens,
        allreduce_residual_attnres_max_tokens(mapping.attn.tp_size),
    )
    groups_are_equal = mapping.attn.tp_group == mapping.moe.tp_ep_group
    # The Lamport crossover was measured with attention TP8 and MoE TP8.
    enable_lamport = (
        groups_are_equal
        and mapping.attn.tp_size == 8
        and mapping.moe.tp_size == 8
        and mapping.moe.ep_size == 1
    )
    # Keep the full producer-direct window for equal TP8 groups. Its 50K/500
    # C16 gain survives content-sensitive EAGLE3 trajectories; retain 48 tokens
    # for other mappings.
    expand_moe_window = (
        groups_are_equal and mapping.attn.tp_size == 8 and mapping.moe.tp_ep_size == 8
    )
    producer_direct_max_tokens = (
        max_num_tokens
        if expand_moe_window
        else min(max_num_tokens, _IRIS_BASELINE_PRODUCER_DIRECT_MAX_TOKENS)
    )
    prepared = False
    if mapping.attn.tp_size > 1:
        prepared = prepare_all_reduce_buffers(
            mapping.attn.tp_group,
            staged_max_numel=max_num_tokens * hidden_size,
            producer_direct_max_numel=(
                producer_direct_max_tokens * (hidden_size + routed_hidden_size)
                if groups_are_equal and mapping.moe.tp_ep_size > 1
                else 0
            ),
            attnres_max_numel=attnres_max_rows * hidden_size,
            attnres_max_rows=attnres_max_rows,
            enable_lamport=enable_lamport,
            dtype=torch.bfloat16,
            backend=None,
        )
    if mapping.moe.tp_ep_size > 1 and not groups_are_equal:
        prepared = (
            prepare_all_reduce_buffers(
                mapping.moe.tp_ep_group,
                staged_max_numel=max_num_tokens * hidden_size,
                producer_direct_max_numel=producer_direct_max_tokens
                * (hidden_size + routed_hidden_size),
                attnres_max_numel=0,
                attnres_max_rows=0,
                enable_lamport=False,
                dtype=torch.bfloat16,
                backend=None,
            )
            or prepared
        )
    return prepared


class K3AttnCommState:
    """Process-wide attention-AR fusion arming for Kimi-K3 (attn TP group).

    Construction is collective: every rank must call :meth:`get` with
    identical arguments, in lockstep, before any forward (model ``__init__``
    satisfies this). The first call runs the collective allocators exactly
    once per process — per-layer callers reuse the singleton. Splitting the
    arming per layer is what previously left several independent ways to
    strand a peer inside a rendezvous.

    Collective constructors are deliberately unguarded: a rank that fails
    mid-build has already stranded its peers, so propagating the exception
    and killing the whole job is the good outcome.
    """

    _instance: "K3AttnCommState | None" = None

    @classmethod
    def get(cls, *, mapping, hidden_size: int) -> "K3AttnCommState":
        """Return the singleton, constructing it on first use (any decoder
        layer; construction is per-process, the collective prepares inside
        run lockstep on the attention TP group).

        Args:
            mapping: Parallel mapping (attn/moe groups) — rank-uniform.
            hidden_size: Model hidden width.
        """
        if cls._instance is None:
            cls._instance = cls(mapping=mapping, hidden_size=hidden_size)
        elif cls._instance.hidden_size != hidden_size:
            # The singleton would otherwise silently hand a second model
            # (e.g. an MTP draft sharing the process with its base) arming
            # done for the wrong width.
            raise ValueError(
                "K3AttnCommState is armed once per process for "
                f"hidden_size={cls._instance.hidden_size}, but a later caller "
                f"asked for hidden_size={hidden_size}; the current "
                "implementation assumes every model in the process (base + "
                "draft) shares this width."
            )
        return cls._instance

    def __init__(self, *, mapping, hidden_size: int):
        self.mapping = mapping
        self.hidden_size = hidden_size
        hidden = hidden_size
        # --- attention AR+residual fusion arming (was per decoder layer) ---
        # Fused AR+residual for the attention reduce: a ones-weight RMSNorm
        # rides the one-shot pattern and its norm output is discarded.
        self.attn_ar_fusion_ok = dist.is_initialized() and (
            mapping.attn.tp_size > 1
            and prepare_all_reduce_lane(mapping.attn.tp_group, hidden)
            and prepare_all_reduce_fusion(
                mapping.attn.tp_group,
                hidden,
                max(int(global_server_args_dict["comm_fusion_max_num_tokens"]), 1),
            )
        )
        # Plain attribute (not a registered submodule): the model loader
        # never migrates it, so the device must be pinned explicitly here.
        # The eps only shapes the discarded ones-weight norm output.
        self.dummy_norm = RMSNorm(hidden, eps=1e-6)
        self.dummy_norm.weight.data = torch.ones(
            hidden,
            dtype=torch.bfloat16,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        self.dummy_norm.weight.requires_grad_(False)

        # A rank that skipped the build would strand its peers in the rendezvous.
        self.cute_ar = None
        if dist.is_initialized() and mapping.attn.tp_size > 1:
            group = _get_process_group(mapping.attn.tp_group)
            # Gate first: a forbidden window should not pay the rendezvous.
            local_ok = (
                attn_ar_eligible(
                    armed=True,
                    has_prefix=True,
                    num_tokens=1,
                    fusion_max_tokens=global_server_args_dict[
                        "comm_fusion_max_num_tokens"
                    ],
                )
                and self.attn_ar_fusion_ok
                and multicast_backend_available(group)
                and attn_reduce_shape_supported(
                    tp_size=mapping.attn.tp_size, hidden_size=hidden
                )
            )
            vote = torch.tensor([int(local_ok)], dtype=torch.int32, device="cuda")
            dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
            if bool(vote.item()):
                self.cute_ar = build_attn_reduce_collective(
                    group=group,
                    rank=mapping.attn.tp_rank,
                    tp_size=mapping.attn.tp_size,
                    hidden_size=hidden,
                    max_tokens=ATTN_AR_MAX_TOKENS,
                )
        attention_reduce_backend = (
            f"tokenspeed CuteDSL collective at M<={ATTN_AR_MAX_TOKENS}"
            if self.cute_ar is not None
            else "not armed; the existing backends serve every M"
        )
        logger.info(f"Kimi K3 attention reduce: {attention_reduce_backend}")


class K3MoeTailCommState:
    """Process-wide negotiated MoE-tail backends for Kimi-K3 (moe tp_ep group).

    Constructed by the first ``K3MoeTailComm`` — every rank builds MoE layers
    in the same order, so the single MIN all-reduce and the collective
    allocators below stay lockstep. Collective constructors are deliberately
    unguarded: a failed rank has already stranded its peers, so killing the
    whole job is the good outcome.
    """

    _instance: "K3MoeTailCommState | None" = None

    @classmethod
    def get(
        cls,
        *,
        mapping,
        hidden_size: int,
        latent_size: int,
        top_k: int,
        rms_eps: float,
        allow_latent_tail: bool,
        deferred_tail: bool,
    ) -> "K3MoeTailCommState":
        if cls._instance is None:
            cls._instance = cls(
                mapping=mapping,
                hidden_size=hidden_size,
                latent_size=latent_size,
                top_k=top_k,
                rms_eps=rms_eps,
                allow_latent_tail=allow_latent_tail,
                deferred_tail=deferred_tail,
            )
        else:
            inst = cls._instance
            if (
                inst.hidden_size != hidden_size
                or inst.latent_size != latent_size
                or inst.top_k != top_k
                or inst.rms_eps != float(rms_eps)
                or inst.allow_latent_tail != allow_latent_tail
                or inst.deferred_tail != deferred_tail
            ):
                # The singleton would otherwise silently hand a second model
                # (e.g. an MTP draft sharing the process with its base) a
                # negotiation done for the wrong shapes.
                raise ValueError(
                    "K3MoeTailCommState is negotiated once per process for "
                    f"hidden={inst.hidden_size} latent={inst.latent_size} "
                    f"top_k={inst.top_k} rms_eps={inst.rms_eps} "
                    f"allow_latent_tail={inst.allow_latent_tail} "
                    f"deferred_tail={inst.deferred_tail}, but a later "
                    f"caller asked for hidden={hidden_size} "
                    f"latent={latent_size} top_k={top_k} "
                    f"rms_eps={float(rms_eps)} "
                    f"allow_latent_tail={allow_latent_tail} "
                    f"deferred_tail={deferred_tail}; the current "
                    "implementation assumes every model in the process "
                    "(base + draft) shares these parameters."
                )
        return cls._instance

    def __init__(
        self,
        *,
        mapping,
        hidden_size,
        latent_size,
        top_k,
        rms_eps,
        allow_latent_tail,
        deferred_tail,
    ):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.top_k = top_k
        self.rms_eps = float(rms_eps)
        self.allow_latent_tail = allow_latent_tail
        # Derived from model/backend traits by K3MoeTailComm, never an opt-in.
        self.deferred_tail = deferred_tail
        self.mnnvl_bt_deferred = None
        self.mnnvl_ht_deferred = None
        self.latent_tail_ok = False  # per-layer ops built by K3MoeTailComm
        if not dist.is_initialized():
            return

        world = dist.get_world_size()
        hidden, latent = hidden_size, latent_size
        # Graph capture is a subset of supported M, not an allocation limit.
        bt_capacity = MNNVL_BT_MAX_TOKENS
        ht_capacity = MNNVL_HT_MAX_TOKENS

        # --- local probes (pure local; failures just vote False) ---
        # All ranks must agree on eligibility before any collective
        # allocator runs.
        deferred_comm_local = (
            self.deferred_tail
            and mapping.moe.tp_ep_size > 1
            and mapping.moe.tp_ep_size == world
            and mapping.attn.dp_size == 1
            and mapping.attn.cp_size == 1
            and multimem_available()
            # Cross-node symmetric-memory rendezvous requires fabric/IMEX.
            and (
                world <= torch.cuda.device_count()
                or fabric_allocation_supported(torch.cuda.current_device())
            )
        )
        mnnvl_bt_local = (
            self.deferred_tail
            and deferred_comm_local
            and mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported(
                group=dist.group.WORLD,
                tp_size=mapping.moe.tp_ep_size,
                hidden_size=latent,
                top_k=top_k,
                dtype=torch.bfloat16,
                candidate_min_tokens=MNNVL_BT_MIN_TOKENS,
                candidate_max_tokens=bt_capacity,
            )
        )
        mnnvl_ht_local = (
            self.deferred_tail
            and deferred_comm_local
            and mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported(
                group=dist.group.WORLD,
                tp_size=mapping.moe.tp_ep_size,
                hidden_size=latent,
                top_k=top_k,
                dtype=torch.bfloat16,
                candidate_min_tokens=MNNVL_HT_MIN_TOKENS,
                candidate_max_tokens=ht_capacity,
            )
        )
        tail_local = False
        # PDL is a requirement of the unchanged BT/HT kernels, not an opt-out.
        configuration_valid = not self.deferred_tail or not global_server_args_dict.get(
            "disable_pdl", False
        )
        if (
            allow_latent_tail
            # The fused tail requires tp_ep to span WORLD.
            and mapping.moe.tp_ep_size == world
            and mapping.attn.dp_size == 1
            and mapping.attn.cp_size == 1
        ):
            tail_local = latent_tail_supported(
                tp_size=mapping.moe.tp_ep_size,
                hidden_size=hidden,
                latent_size=latent,
                dtype=torch.bfloat16,
                group=dist.group.WORLD,
            )
        # --- the single agreement point: every rank executes unconditionally ---
        votes = torch.tensor(
            [
                int(deferred_comm_local),
                int(tail_local),
                int(mnnvl_bt_local),
                int(mnnvl_ht_local),
                int(configuration_valid),
                int(self.deferred_tail),
                -int(self.deferred_tail),
            ],
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(votes, op=dist.ReduceOp.MIN)
        values = votes.tolist()
        configuration_ok, deferred_min, deferred_negative_min = values[4:]
        if not configuration_ok or deferred_min != -deferred_negative_min:
            raise ValueError(
                "K3 tail capabilities must agree across ranks; the automatic "
                "deferred BT/HT path requires PDL"
            )
        deferred_comm_ok, tail_ok, mnnvl_bt_ok, mnnvl_ht_ok = (
            bool(v) for v in values[:4]
        )
        if self.deferred_tail and not all(
            (deferred_comm_ok, tail_ok, mnnvl_bt_ok, mnnvl_ht_ok)
        ):
            raise RuntimeError(
                "deferred K3 tail requires every TP8 capability; no fallback"
            )

        # --- collective builds, fixed order, unguarded ---
        # BT/HT own the routed buffers; shared partials use the ordinary all-reduce.
        if mnnvl_bt_ok and deferred_comm_ok:
            self.mnnvl_bt_deferred = MNNVLCuteDSLFinalizeAllReduceRMSNorm.initialize(
                group=dist.group.WORLD,
                hidden_size=latent,
                top_k=top_k,
                rms_eps=self.rms_eps,
                candidate_min_tokens=MNNVL_BT_MIN_TOKENS,
                candidate_max_tokens=bt_capacity,
                tuning_routes=(
                    MNNVLCuteDSLBTFinalizeTuning(
                        max_tokens=bt_capacity,
                        elements_per_thread=2,
                        threads=256,
                        prefetch_group=1,
                        reduction_threads=224,
                        rms_threads=448,
                        enable_pdl=True,
                    ),
                ),
            )
        if mnnvl_ht_ok and deferred_comm_ok:
            self.mnnvl_ht_deferred = MNNVLCuteDSLHTFinalizeAllReduceRMSNorm.initialize(
                group=dist.group.WORLD,
                hidden_size=latent,
                top_k=top_k,
                rms_eps=self.rms_eps,
                candidate_min_tokens=MNNVL_HT_MIN_TOKENS,
                candidate_max_tokens=ht_capacity,
                tuning_routes=(
                    MNNVLCuteDSLHTFinalizeTuning(
                        max_tokens=ht_capacity,
                        persistent_ctas=None,
                        consumer_threads=448,
                        vectors_per_thread=1,
                        stages=10,
                        reduction_warps=2,
                        reduction_cta_groups=None,
                        rms_token_groups=2,
                        rms_pipeline_stages=3,
                        rms_shard_major=False,
                        enable_pdl=True,
                    ),
                ),
            )
        self.latent_tail_ok = tail_ok
        logger.info(
            f"K3 comm negotiated: deferred_comm={deferred_comm_ok!s} "
            f"mnnvl_bt_deferred={self.mnnvl_bt_deferred is not None!s} "
            f"mnnvl_ht_deferred={self.mnnvl_ht_deferred is not None!s} "
            f"latent_tail={self.latent_tail_ok!s}",
        )


class K3AttnComm:
    """Per-decoder-layer handle over the negotiated K3 communication state.

    Model code states semantics (``attn_reduce``); backend choice,
    thresholds and workspace access all live behind this class.
    """

    def __init__(self, state: K3AttnCommState) -> None:
        self.state = state
        self.mapping = state.mapping

    # ------------------------------------------------------------------
    # Attention-side reduction, hoisted from KimiLinearDecoderLayer.
    # ------------------------------------------------------------------
    def fused_attnres_reduce_available(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        combine: tuple,
        score_weight: torch.Tensor | None,
    ) -> bool:
        """Whether the communication path can consume the AttnRes epilogue."""
        scratch, _, _, output_weight, _ = combine
        if score_weight is None or output_weight is None:
            return False
        from tokenspeed_kernel.ops.communication.triton import (
            allreduce_residual_attnres_combine_supported,
        )

        return not global_server_args_dict.get(
            "force_deterministic_rsag", False
        ) and allreduce_residual_attnres_combine_supported(
            partial,
            residual,
            score_weight,
            output_weight,
            scratch,
            rank=self.mapping.attn.tp_rank,
            group=_get_process_group(self.mapping.attn.tp_group),
            local_world_size=self.mapping.nprocs_per_node,
        )

    def attn_reduce(
        self,
        attn_partial: torch.Tensor,
        prefix_sum: torch.Tensor | None,
        combine: tuple | None = None,
        *,
        mlp_wp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All-reduce the attention partial and accumulate the residual.

        Small batches fold the residual add into the one-shot AR kernel;
        with ``combine = (scratch, res_w, rms_w, out_norm_w, eps)`` the
        mlp-side AttnRes prefix combine also rides its epilogue and the mixed
        hidden comes back as the second return (else None -- block-write
        layers, large batches and the plain-reduce fallback).

        The tokenspeed collective is the exception: it serves the narrow window
        ahead of those branches and returns None for the mixed hidden even when
        ``combine`` is set, so the caller runs the combine as its own kernel.
        Measured net faster despite the extra launch at the width that
        actually reaches it -- one token per step, where every layer but the
        block-write ones arrives with a residual. Wider steps mostly take the
        fused AttnRes graph instead, and the block-write layers that still
        arrive pass no prefix: instrumented at eight tokens on a DSpark
        deployment, this window was armed and served nothing. A layer that
        declines the fused graph for some other reason does reach it with a
        prefix, so that is a property of the configuration, not of the width.

        Like the vendor branch below it, that window does not consult
        ``force_deterministic_rsag``: the collective reduces in ascending rank
        order with an fp32 accumulator, so it is already run-to-run stable.

        ``mlp_wp`` is the calling layer's precomputed ``rms_w * res_w``
        product (per-layer state, filled in post_load_weights); the B1
        combine kernels consume it in place of the separate weights.
        """
        num_tokens = attn_partial.shape[0]
        if attn_ar_eligible(
            armed=self.state.cute_ar is not None,
            has_prefix=prefix_sum is not None,
            num_tokens=num_tokens,
            fusion_max_tokens=global_server_args_dict["comm_fusion_max_num_tokens"],
        ):
            # Any later reduce in this process overwrites it; this layer is done by then.
            residual_out, _ = self.state.cute_ar(
                attn_partial,
                prefix_sum,
                self.state.dummy_norm.weight,
                include_reduce_scatter=False,
                include_routed=True,
            )
            return residual_out, None
        if (
            prefix_sum is not None
            and self.state.attn_ar_fusion_ok
            and 0 < num_tokens
            and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
        ):
            if combine is not None:
                from tokenspeed_kernel.ops.communication.trtllm import (
                    allreduce_residual_attnres_combine,
                )

                scratch, res_w, rms_w, out_norm_w, eps = combine
                h, residual_out = allreduce_residual_attnres_combine(
                    attn_partial,
                    prefix_sum,
                    res_w,
                    rms_w,
                    out_norm_w,
                    scratch=scratch,
                    rank=self.mapping.attn.tp_rank,
                    group=_get_process_group(self.mapping.attn.tp_group),
                    eps=eps,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                )
                return residual_out, h
            _, residual_out, *_ = self.state.dummy_norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                attn_partial,
                prefix_sum,
            )
            if residual_out is not None:
                return residual_out, None
        if combine is not None and prefix_sum is not None and num_tokens > 0:
            scratch, _, _, out_norm_w, eps = combine
            if out_norm_w is not None and self.fused_attnres_reduce_available(
                attn_partial,
                prefix_sum,
                combine,
                mlp_wp,
            ):
                from tokenspeed_kernel.ops.communication.triton import (
                    allreduce_residual_attnres_combine,
                )

                group = _get_process_group(self.mapping.attn.tp_group)
                h, residual_out = allreduce_residual_attnres_combine(
                    attn_partial,
                    prefix_sum,
                    mlp_wp,
                    out_norm_w,
                    scratch,
                    rank=self.mapping.attn.tp_rank,
                    group=group,
                    local_world_size=self.mapping.nprocs_per_node,
                    eps=eps,
                )
                return residual_out, h
        reduced = all_reduce(attn_partial, self.mapping.attn.tp_group)
        return (reduced if prefix_sum is None else prefix_sum + reduced), None


@dataclass
class TailPlan:
    """Per-forward contract between the model and the MoE tail.

    Attributes:
        tier: The negotiated tail tier for this token count.
        defer_finalize: The experts kernel must run with
            ``do_finalize=False``. BT/HT own finalize; the small multicast
            tail does so when it supports deferred expert output.
        routed_in_fork: Whether the routed partial must be reduced and
            projected inside the fork (SEPARATE_REDUCE overlap).
        split_shared_rs: Start the shared ReduceScatter on the auxiliary
            stream before the routed collective is ready.
    """

    tier: "K3MoETailTier"
    defer_finalize: bool = False
    routed_in_fork: bool = False
    split_shared_rs: bool = False


def _tail_finalize_top_k(
    top_k: int,
    execution_plan,
    experts_supports_deferred_finalize: bool,
) -> int | None:
    """Deferred-finalize arming decision for the latent tail (rank-uniform).

    Both inputs are identical on every rank: ``fused_moe_ar`` comes from the
    negotiated execution plan, and ``experts_supports_deferred_finalize`` is
    the experts kernel plan's capability bit (``MoELayer.plan``), so all
    ranks arm — or don't — together. Returns ``top_k`` to request the
    deferred triple from the experts kernel, or ``None`` for the
    materialized-input tail.
    """
    if execution_plan.fused_moe_ar and experts_supports_deferred_finalize:
        return top_k
    return None


class K3MoeTailComm:
    """MoE-tail routing and execution for one KimiLinearMoE module.

    Holds the negotiated ``K3MoeTailCommState`` plus this module's own
    resources (per-module multicast mailbox, norm/up-proj weights).
    """

    def __init__(
        self,
        *,
        mapping,
        hidden_size: int,
        prefix: str,
        layer_index: int,
        model_scope: str,
        routed_hidden: int,
        top_k: int,
        routed_norm,
        up_proj,
        execution_plan,
        experts_supports_deferred_finalize: bool,
    ) -> None:
        # Derived from the projection itself (built with a shard group iff
        # _shard_k3_up_projection held), so comm and module cannot disagree.
        self._shard_up_projection = up_proj.shard_group is not None
        self._experts_supports_deferred_finalize = experts_supports_deferred_finalize
        self.state = K3MoeTailCommState.get(
            mapping=mapping,
            hidden_size=hidden_size,
            latent_size=routed_hidden,
            top_k=top_k,
            rms_eps=(routed_norm.variance_epsilon if routed_norm is not None else 1e-5),
            allow_latent_tail=(
                not execution_plan.use_native and routed_norm is not None
            ),
            deferred_tail=_deferred_tail_applicable(
                mapping=mapping,
                hidden_size=hidden_size,
                latent_size=routed_hidden,
                top_k=top_k,
                is_blackwell=current_platform().is_blackwell,
                fused_moe_ar=execution_plan.fused_moe_ar,
                has_routed_norm=routed_norm is not None,
                shard_up_projection=self._shard_up_projection,
                experts_supports_deferred_finalize=experts_supports_deferred_finalize,
            ),
        )
        self.mapping = mapping
        self.hidden_size = hidden_size
        self.routed_hidden = routed_hidden
        self.top_k = top_k
        self.routed_norm = routed_norm
        self.up_proj = up_proj
        self.execution_plan = execution_plan
        # Derived from the projection itself (built with a shard group iff
        # _shard_k3_latent_projection held), so comm and module cannot disagree.
        self._shard_up_projection = up_proj.shard_group is not None
        self.latent_tail = None
        if self.state.latent_tail_ok:
            # Deferred-finalize arming (rank-uniform). The gate is the
            # experts kernel plan's own supports_deferred_finalize bit,
            # passed in by the model (this comm layer never sees the experts
            # module itself) — NOT a use_trtllm proxy: the trtllm solution
            # spans kernels with either capability (the SiTU variants emit
            # the deferred triple, mxfp4 SwiGLU does not). The backstops stay
            # explicit: the experts layer raises on do_finalize=False without
            # the trait, and KimiK3LatentTailOp.call_deferred raises on
            # non-BF16 scales (no silent down-cast), so a mis-armed or
            # fp32-scale producer fails loudly instead of silently degrading.
            tail_finalize_top_k = _tail_finalize_top_k(
                top_k,
                execution_plan,
                experts_supports_deferred_finalize,
            )
            # Per-module mailbox. Constructor failures must propagate because
            # peers are already rendezvousing: a rank that failed mid-way has
            # stranded them, and killing the whole job is the good outcome.
            _device = torch.device("cuda", torch.cuda.current_device())
            self.latent_tail = KimiK3LatentTailOp.initialize(
                group=dist.group.WORLD,
                hidden_size=hidden_size,
                latent_size=routed_hidden,
                rms_eps=self.state.rms_eps,
                device=_device,
                layer_index=layer_index,
                model_scope=model_scope,
                # Staging may alias the workspace pool; each barrier-free
                # mailbox must remain private.
                scratch_allocator=workspace_pool(_device).allocate,
                finalize_top_k=tail_finalize_top_k,
                split_collective=current_platform().is_blackwell,
            )
            logger.info(
                "multicast latent tail engaged "
                f"({prefix!s}, deferred_finalize={tail_finalize_top_k is not None!s}, "
                f"split_shared_rs={self.latent_tail.supports_split_collective!s})",
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def plan(
        self,
        num_tokens: int,
    ) -> TailPlan:
        """Pick the tail tier and its forward-side obligations.

        Every input must be rank-uniform (token count, graph phase and the
        negotiated capabilities) so all ranks take identical branches.
        """
        if self.state.deferred_tail:
            deferred_tier = select_deferred_k3_moe_tail_tier(num_tokens)
            if deferred_tier is K3MoETailTier.SEPARATE_REDUCE:
                return TailPlan(tier=deferred_tier, routed_in_fork=True)
            if deferred_tier is not None:
                finalize = (
                    self.state.mnnvl_bt_deferred
                    if deferred_tier is K3MoETailTier.MNNVL_BT
                    else self.state.mnnvl_ht_deferred
                )
                if (
                    finalize is None
                    or not finalize.supports_num_tokens(num_tokens)
                    or not self._experts_supports_deferred_finalize
                ):
                    raise RuntimeError(f"deferred K3 tail missing M={num_tokens} route")
                return TailPlan(
                    tier=deferred_tier,
                    defer_finalize=True,
                )
        # Only small M or unsupported model/backend layouts reach compatibility dispatch.
        # Supported M33..8192 returned above; a missing deferred route raises.
        # Graph warmup, capture, and replay must select the same tier.
        tier = select_k3_moe_tail_tier(
            num_tokens=num_tokens,
            graph_phase=get_is_cuda_graph_phase(),
            tail_fusion_max_tokens=(
                min(self.latent_tail.max_num_tokens, TAIL_FUSION_MAX_TOKENS)
                if self.latent_tail is not None
                else 0
            ),
        )
        if tier is K3MoETailTier.TAIL_FUSION:
            # Full fusion: with the trtllm fused-AR plan armed and a
            # deferred-capable tail op, the multicast tail consumes the
            # experts kernel's deferred-finalize triple directly — no
            # standalone finalize kernel, no [M, latent] intermediate.
            # Otherwise the materialized-input mode remains.
            return TailPlan(
                tier=tier,
                defer_finalize=(
                    self.execution_plan.fused_moe_ar
                    and self.latent_tail is not None
                    and self.latent_tail.supports_deferred_finalize
                ),
                split_shared_rs=(
                    self.latent_tail is not None
                    and self.latent_tail.supports_split_collective
                    and num_tokens >= self.latent_tail.split_collective_min_tokens
                    and get_is_capture_mode()
                ),
            )
        return TailPlan(
            tier=tier,
            routed_in_fork=tier is K3MoETailTier.SEPARATE_REDUCE,
        )

    # ------------------------------------------------------------------
    # Fork-side helpers (called by the model inside its stream fork)
    # ------------------------------------------------------------------
    def reduce_shared(self, shared_partial: torch.Tensor) -> torch.Tensor:
        """Reduce the shared experts' TP partial on the current stream."""
        if self.mapping.moe.tp_ep_size > 1:
            return all_reduce(shared_partial, self.mapping.moe.tp_ep_group)
        return shared_partial

    def reduce_scatter_shared(self, shared_partial: torch.Tensor) -> torch.Tensor:
        """Launch the split shared ReduceScatter on the current stream."""
        if self.latent_tail is None:
            raise RuntimeError("split shared ReduceScatter requires the latent tail")
        return self.latent_tail.reduce_scatter_shared(
            shared_partial,
            self.routed_norm.weight,
        )

    def reduce_project_routed(self, routed_out: torch.Tensor) -> torch.Tensor:
        """Reduce, norm and up-project the routed partial (SEPARATE_REDUCE).

        Runs in the model's ``forward`` inside the stream fork so it overlaps
        the shared-expert branch; the tier method then receives an
        already-projected routed output, unlike the other tiers.
        """
        routed_reduced = routed_out
        if self.mapping.moe.has_tp_ep:
            routed_reduced = all_reduce(routed_reduced, self.mapping.moe.tp_ep_group)
        if self.routed_norm is not None:
            routed_reduced = self.routed_norm(routed_reduced)
        if self._shard_up_projection:
            # This block must wait for the fork before folding into the shared reduction.
            return self.up_proj.project_shard(routed_reduced)
        return self.up_proj(routed_reduced)[0]

    # ------------------------------------------------------------------
    # Tail dispatch (moved verbatim from KimiLinearMoE._moe_tail and the
    # tier methods)
    # ------------------------------------------------------------------
    def run(
        self,
        plan: TailPlan,
        routed_out,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
        prepared_shared_shard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch the selected tier over the partials.

        Raw partials everywhere except SEPARATE_REDUCE, whose routed side
        already ran inside the fork scope to overlap the shared branch, and
        deferred-finalize tiers (``plan.defer_finalize``), whose ``routed_out``
        is the experts kernel's deferred triple.
        """
        tier = plan.tier
        if tier in (
            K3MoETailTier.MNNVL_BT,
            K3MoETailTier.MNNVL_HT,
        ):
            gemm2_out, expert_weights, expanded_idx = routed_out
            finalize = (
                self.state.mnnvl_bt_deferred
                if tier is K3MoETailTier.MNNVL_BT
                else self.state.mnnvl_ht_deferred
            )
            routed_norm = finalize(
                gemm2_out, expert_weights, expanded_idx, self.routed_norm.weight
            )
            shared_partial = self._project_and_inject_local_block(
                routed_norm, shared_partial, prefix_sum, num_tokens, hidden_size
            )
            return self.reduce_shared(shared_partial).view(num_tokens, hidden_size)
        if tier is K3MoETailTier.TAIL_FUSION:
            if plan.defer_finalize:
                gemm2_out, expert_weights, expanded_idx = routed_out
                return self._tail_fusion_deferred(
                    gemm2_out,
                    expert_weights,
                    expanded_idx,
                    shared_partial,
                    prefix_sum,
                    num_tokens,
                    prepared_shared_shard,
                )
            return self._tail_fusion(
                routed_out,
                shared_partial,
                prefix_sum,
                prepared_shared_shard,
            )
        return self._tail_separate_reduce(
            routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
        )

    def _tail_fusion(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        prepared_shared_shard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.latent_tail(
            routed_out,
            shared_partial,
            self.routed_norm.weight,
            self.up_proj.weight,
            prefix=prefix_sum,
            prepared_shared_shard=prepared_shared_shard,
        )

    def _tail_fusion_deferred(
        self,
        gemm2_out: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        prepared_shared_shard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TAIL_FUSION over the deferred-finalize triple (finalize in-kernel)."""
        return self.latent_tail.call_deferred(
            gemm2_out,
            expert_weights,
            expanded_idx,
            shared_partial,
            self.routed_norm.weight,
            self.up_proj.weight,
            num_tokens=num_tokens,
            prefix=prefix_sum,
            prepared_shared_shard=prepared_shared_shard,
        )

    def _project_and_inject_local_block(
        self,
        routed_reduced: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        start, width = self.up_proj.shard_slice
        shared_partial = shared_partial.view(num_tokens, hidden_size)
        target = shared_partial[:, start : start + width]
        target += prefix_sum.view(num_tokens, hidden_size)[:, start : start + width]
        target.addmm_(routed_reduced, self.up_proj.weight.t())
        return shared_partial

    def _inject_local_block(
        self,
        routed_projected_shard: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        """Add this rank's projection block into its columns of the shared
        partial, in place, so the shared reduction also gathers the projection.

        The column blocks are disjoint across ranks, so summing the partials
        concatenates the blocks and adds ``prefix`` exactly once per column.
        Works with any sum all-reduce; the caller runs it right after this.
        """
        # One extra bf16 rounding vs the joined tiers; measured nil on GPQA, not bitwise.
        start, width = self.up_proj.shard_slice
        shared_partial = shared_partial.view(num_tokens, hidden_size)
        shared_partial[
            :, start : start + width
        ] += routed_projected_shard + prefix_sum.view(num_tokens, hidden_size).narrow(
            -1, start, width
        )
        return shared_partial

    def _tail_separate_reduce(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        if self._shard_up_projection:
            return self._tail_separate_reduce_sharded(
                routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
            )
        return self._tail_separate_reduce_replicated(
            routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
        )

    def _tail_separate_reduce_sharded(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        routed_projected_shard = routed_out
        shared_partial = self._inject_local_block(
            routed_projected_shard,
            shared_partial,
            prefix_sum,
            num_tokens,
            hidden_size,
        )
        shared_reduced = self.reduce_shared(shared_partial)
        return shared_reduced.view(num_tokens, hidden_size)

    def _tail_separate_reduce_replicated(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        routed_projected = routed_out
        shared_reduced = self.reduce_shared(shared_partial)
        # routed_scaling_factor already applied in TopK (matches reference).
        return add3(
            prefix_sum,
            routed_projected.view(num_tokens, hidden_size),
            shared_reduced.view(num_tokens, hidden_size),
        )


__all__ = [
    "K3AttnComm",
    "K3AttnCommState",
    "K3MoETailTier",
    "K3MoeTailComm",
    "K3MoeTailCommState",
    "TailPlan",
    "select_k3_moe_tail_tier",
]
