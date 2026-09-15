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

"""Kimi-K3 communication layer: capability negotiation and fused-reduction
routing for the sites where K3's AttnRes/latent-lane semantics bypass the
generic ``CommManager`` (decision D4).

Layering: this module owns *which backend runs where* (votes, workspace
lifecycle, M-window routing); the kernels themselves stay behind
``tokenspeed_kernel.ops.communication`` / ``ops.moe``. Model code
(``kimi_k3.py``) states semantics only and never names a backend.

All M thresholds for the K3 tail live here (single source of truth):

======================  =========================================
decode fused tail       ``1 <= M <= latent-tail capacity``
                        (multicast tail, tp_ep spanning WORLD) — see
                        ``select_k3_moe_tail_tier``
multimem AR window      ``MULTIMEM_AR_MIN_TOKENS..MAX`` (prefill)
attention reduce        ``1 <= M <= ATTN_AR_MAX_TOKENS`` (tokenspeed
                        CuteDSL collective, attn TP group)
fused-lane one-shot     everything else with a fused plan
======================  =========================================
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.activation.triton import add3
from tokenspeed_kernel.ops.communication import allreduce_fusion_lane
from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving_config import (
    MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
    MNNVLCuteDSLBTFinalizeTuning,
    MNNVLCuteDSLFinalizeAllReduceRMSNorm,
    MNNVLCuteDSLHTFinalizeAllReduceRMSNorm,
    MNNVLCuteDSLHTFinalizeTuning,
    mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported,
    mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported,
)
from tokenspeed_kernel.ops.communication.multimem import (
    multimem_all_reduce_staged,
    multimem_available,
    multimem_prealloc,
    multimem_stage,
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
    acquire_all_reduce_outputs,
    all_reduce,
    can_acquire_all_reduce_outputs,
    prepare_all_reduce_buffers,
    prepare_all_reduce_fusion,
    prepare_all_reduce_lane,
)
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
    get_is_prefill_graph_phase,
)
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.layers.layernorm import RMSNorm, _get_process_group
from tokenspeed.runtime.layers.moe.latent import kimi3_join_reduce_moe
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

_IRIS_MAX_TOKENS = 8192
_IRIS_PRODUCER_DIRECT_MAX_BYTES = 1024 * 1024

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
    FUSED_RS_UP_AG = 1  # prefill: HT finalize/norm + shared-RS/up-projection/AG
    MEDIUM_FUSED_RS_UP_AG = 2  # opt-in provisional BT + medium fused back half
    MNNVL_BT_DEFERRED = 3  # graph: finalize + balanced-tree AR + RMSNorm
    MNNVL_HT_DEFERRED = 4  # graph: finalize + persistent HT AR + RMSNorm
    MULTIMEM_AR = 5  # in-switch (ld_reduce) reduces, then the replicated tail
    FUSED_LANE_AR = 6  # join tier: lane one-shot / cat+one-shot / grouped NCCL
    SEPARATE_REDUCE = 7  # portable: reduce each partial on its own


# Above plain decode-graph buckets: at decode sizes the two staged reduces
# lose ~4% TPOT to the single fused-lane AR; the ld_reduce win is prefill's.
# Caveat: spec-decode buckets reach bs*q tokens and can re-enter this window
# in-graph (correct, but decode-suboptimal) — a follow-up should add an
# is_decode axis rather than gate on the graph phase, which prefill graphs
# legitimately share.
# Measured profit edge of the fused tail; the kernel's own capacity is larger.
TAIL_FUSION_MAX_TOKENS = 32

MULTIMEM_AR_MIN_TOKENS = 256
# Upper edge of the measured window; larger batches take the join's grouped path.
MULTIMEM_AR_MAX_TOKENS = 8192

# FlashInfer #4358 protocol specializations qualified on TP8 GB300.  These
# paths are CUDA-Graph-only: eager CuTe DSL dispatch adds roughly 300 us of
# host enqueue latency, while graph replay retains the kernel-side wins.
# The complete sharded-tail benchmark qualifies only the explicit production
# graph buckets below.  Every other M keeps the established tier; no result is
# extrapolated across the gaps between measured capture sizes.
MNNVL_BT_MIN_TOKENS = 256
MNNVL_BT_MAX_TOKENS = 1024
MNNVL_HT_MIN_TOKENS = 1280
MNNVL_HT_MAX_TOKENS = 8192
MNNVL_BT_QUALIFIED_TOKENS = frozenset((256, 384, 512, 768, 1024))
MNNVL_HT_QUALIFIED_TOKENS = frozenset((1280, 2048, 4096, 6144, 8192))
MNNVL_DEFERRED_ENABLE_ENV = "TOKENSPEED_K3_MNNVL_CUTEDSL"
FUSED_RS_UP_AG_ENABLE_ENV = "TOKENSPEED_K3_FUSED_RS_UP_AG"
FUSED_RS_UP_AG_TOKENS = frozenset((4096, 8192))
MNNVL_BT_ONLY_ENABLE_ENV = "TOKENSPEED_K3_MNNVL_BT_ONLY"
MEDIUM_FUSED_RS_UP_AG_ENABLE_ENV = "TOKENSPEED_K3_MEDIUM_FUSED_RS_UP_AG"
INTEGRATED_TAIL_ENABLE_ENV = "TOKENSPEED_K3_INTEGRATED_FUSED_TAIL"


def _integrated_tail_enabled() -> bool:
    """Enable both first stages and the continuous-range fused back half."""
    return os.environ.get(INTEGRATED_TAIL_ENABLE_ENV) == "1"


def select_integrated_k3_moe_tail_tier(num_tokens: int) -> K3MoETailTier | None:
    """Return the integrated interval tier; None preserves the old small path.

    Args:
        num_tokens: Actual forward M, or graph-padded M, not request count.

    Returns:
        BT/fused for (32,1024], HT/fused for (1024,8192], separate above
        8192, and None for empty/small forwards. No sampled-point whitelist.
    """
    if type(num_tokens) is not int or num_tokens < 0:
        raise ValueError("token count must be a nonnegative integer")
    if num_tokens <= 32:
        return None
    if num_tokens <= 1024:
        return K3MoETailTier.MEDIUM_FUSED_RS_UP_AG
    if num_tokens <= 8192:
        return K3MoETailTier.FUSED_RS_UP_AG
    return K3MoETailTier.SEPARATE_REDUCE


def _mnnvl_bt_only_enabled() -> bool:
    """Opt in to the provisional medium bucket policy without enabling HT."""
    return os.environ.get(MNNVL_BT_ONLY_ENABLE_ENV) == "1"


def _medium_fused_rs_up_ag_enabled() -> bool:
    """Opt in to the unqualified medium back half; BT-only is also required."""
    return os.environ.get(MEDIUM_FUSED_RS_UP_AG_ENABLE_ENV) == "1"


def _medium_flags_valid(bt_only: bool, medium_fused: bool, legacy: bool) -> bool:
    """Return whether explicit experimental flags have unambiguous scope."""
    return not (
        (medium_fused and not bt_only) or ((bt_only or medium_fused) and legacy)
    )


def _fused_rs_up_ag_enabled() -> bool:
    """Whether the experimental complete large-M tail is explicitly opted in."""

    return os.environ.get(FUSED_RS_UP_AG_ENABLE_ENV) == "1"


def _mnnvl_deferred_enabled() -> bool:
    """Whether the qualified experimental K3 CuTe DSL route is opted in."""

    return os.environ.get(MNNVL_DEFERRED_ENABLE_ENV) == "1" or _fused_rs_up_ag_enabled()


def _mnnvl_graph_max_tokens() -> int:
    """Return the resolved largest prefill graph bucket, or zero."""

    return int(global_server_args_dict.get("prefill_graph_max_tokens") or 0)


def _mnnvl_tp8_layout(mapping) -> bool:
    """Whether the MoE routes and weights use the qualified TP8/EP1 layout."""

    return mapping.moe.tp_size == 8 and mapping.moe.ep_size == 1


def select_k3_moe_tail_tier(
    *,
    num_tokens: int,
    graph_phase: bool,
    tail_fusion_max_tokens: int,
    fused_moe_ar: bool,
    multimem_ok: bool,
    is_decode: bool,
    join_moe_reduce: bool,
    mnnvl_bt_deferred_ok: bool,
    mnnvl_ht_deferred_ok: bool,
    fused_rs_up_ag_ok: bool,
    prefill_graph_phase: bool,
) -> K3MoETailTier:
    """Pick the tail tier; every input must be rank-uniform.

    Args:
        num_tokens: Tokens in this forward (identical on every rank).
        graph_phase: Whether the forward runs under the CUDA-graph phase.
        tail_fusion_max_tokens: Largest token count the fused tail is both
            able and worth running at, 0 when absent.
        fused_moe_ar: Whether the fused-AR execution plan is armed (implies a
            backend-owned lane, so TRT-LLM only).
        join_moe_reduce: Whether the routed and shared partials can be reduced
            together without a lane, via a concatenated one-shot or a grouped
            all-reduce. Portable, so this is what lets non-TRT-LLM backends
            reach the join tier.
        multimem_ok: Collectively-agreed multimem availability.
        is_decode: Whether this forward is a decode (spec-verify included);
            rank-uniform and stable between graph capture and replay.
        prefill_graph_phase: Whether this forward is being warmed or captured
            by the breakable prefill graph.  It is deliberately separate from
            ``graph_phase`` so prefill capture does not enable full-graph
            auxiliary-stream branches across an eager attention break.
        fused_rs_up_ag_ok: Rank-negotiated complete-tail capability. The
            selector additionally restricts it to the two qualified buckets.

    Returns:
        The best applicable ``K3MoETailTier``.
    """
    # Tested first, so a fused-tail capacity that ever reached into the
    # multimem window would still resolve here rather than overlap.
    if graph_phase and 1 <= num_tokens <= tail_fusion_max_tokens:
        return K3MoETailTier.TAIL_FUSION
    if (
        prefill_graph_phase
        and not is_decode
        and fused_rs_up_ag_ok
        and num_tokens in FUSED_RS_UP_AG_TOKENS
    ):
        return K3MoETailTier.FUSED_RS_UP_AG
    # The MNNVL CuTe DSL protocols consume TRT-LLM's deferred-finalize triple.
    # They are only profitable under graph replay and are initially confined
    # to prefill; speculative-decode buckets retain their separately tuned
    # tail tiers until they are measured with the same acceptance workload.
    if prefill_graph_phase and not is_decode and mnnvl_bt_deferred_ok:
        return K3MoETailTier.MNNVL_BT_DEFERRED
    if prefill_graph_phase and not is_decode and mnnvl_ht_deferred_ok:
        return K3MoETailTier.MNNVL_HT_DEFERRED
    if not fused_moe_ar:
        # No lane, but the join only needs a concatenated or grouped
        # all-reduce. Taking it halves the tail's collectives -- SEPARATE_REDUCE
        # reduces the routed and shared partials over the same group one after
        # the other -- and each collective is a rendezvous whose cost does not
        # amortize with batch, so the saving is largest at low concurrency.
        #
        # SEPARATE_REDUCE stays the fallback for layouts that cannot join,
        # which includes a sharded up projection: its tail folds the projection
        # between two sequential all-reduces instead of calling
        # kimi3_join_reduce_moe, so it would save no collective while still
        # giving up the routed_in_fork overlap.
        #
        # MULTIMEM_AR is deliberately still not reachable here. It already
        # required fused_moe_ar before this join existed, so promoting
        # lane-less backends into the multimem window would be a separate
        # behavioural change rather than part of this one.
        if join_moe_reduce:
            return K3MoETailTier.FUSED_LANE_AR
        return K3MoETailTier.SEPARATE_REDUCE
    if (
        multimem_ok
        # Decode buckets skip multimem: same bytes, but it leaves the GPU idle there.
        and not is_decode
        and MULTIMEM_AR_MIN_TOKENS <= num_tokens <= MULTIMEM_AR_MAX_TOKENS
    ):
        return K3MoETailTier.MULTIMEM_AR
    return K3MoETailTier.FUSED_LANE_AR


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
    # Iris and RCCL sum in different orders. Using Iris for K3 prefill changed
    # the greedy EAGLE3 trajectory enough to reduce acceptance and end-to-end
    # output throughput, despite making this collective faster in isolation.
    producer_direct_max_numel = min(
        max_num_tokens * (hidden_size + routed_hidden_size),
        _IRIS_PRODUCER_DIRECT_MAX_BYTES // torch.bfloat16.itemsize,
    )
    prepared = False
    if mapping.attn.tp_size > 1:
        prepared = prepare_all_reduce_buffers(
            mapping.attn.tp_group,
            staged_max_numel=max_num_tokens * hidden_size,
            producer_direct_max_numel=(
                producer_direct_max_numel
                if groups_are_equal and mapping.moe.tp_ep_size > 1
                else 0
            ),
            attnres_max_numel=attnres_max_rows * hidden_size,
            attnres_max_rows=attnres_max_rows,
            dtype=torch.bfloat16,
            backend=None,
        )
    if mapping.moe.tp_ep_size > 1 and not groups_are_equal:
        prepared = (
            prepare_all_reduce_buffers(
                mapping.moe.tp_ep_group,
                staged_max_numel=max_num_tokens * hidden_size,
                producer_direct_max_numel=producer_direct_max_numel,
                attnres_max_numel=0,
                attnres_max_rows=0,
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
        logger.info(
            "Kimi K3 attention reduce: %s",
            (
                "tokenspeed CuteDSL collective at M<=%d" % ATTN_AR_MAX_TOKENS
                if self.cute_ar is not None
                else "not armed; the existing backends serve every M"
            ),
        )


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
        allow_mnnvl_deferred: bool,
        allow_fused_rs_up_ag: bool,
        mnnvl_bt_only: bool,
        allow_medium_fused_rs_up_ag: bool,
        mnnvl_graph_max_tokens: int,
    ) -> "K3MoeTailCommState":
        if cls._instance is None:
            cls._instance = cls(
                mapping=mapping,
                hidden_size=hidden_size,
                latent_size=latent_size,
                top_k=top_k,
                rms_eps=rms_eps,
                allow_latent_tail=allow_latent_tail,
                allow_mnnvl_deferred=allow_mnnvl_deferred,
                allow_fused_rs_up_ag=allow_fused_rs_up_ag,
                mnnvl_bt_only=mnnvl_bt_only,
                allow_medium_fused_rs_up_ag=allow_medium_fused_rs_up_ag,
                mnnvl_graph_max_tokens=mnnvl_graph_max_tokens,
            )
        else:
            inst = cls._instance
            if (
                inst.hidden_size != hidden_size
                or inst.latent_size != latent_size
                or inst.top_k != top_k
                or inst.rms_eps != float(rms_eps)
                or inst.allow_latent_tail != allow_latent_tail
                or inst.allow_mnnvl_deferred != allow_mnnvl_deferred
                or inst.allow_fused_rs_up_ag != allow_fused_rs_up_ag
                or inst.mnnvl_bt_only != mnnvl_bt_only
                or inst.allow_medium_fused_rs_up_ag != allow_medium_fused_rs_up_ag
                or inst.mnnvl_graph_max_tokens != mnnvl_graph_max_tokens
            ):
                # The singleton would otherwise silently hand a second model
                # (e.g. an MTP draft sharing the process with its base) a
                # negotiation done for the wrong shapes.
                raise ValueError(
                    "K3MoeTailCommState is negotiated once per process for "
                    f"hidden={inst.hidden_size} latent={inst.latent_size} "
                    f"top_k={inst.top_k} rms_eps={inst.rms_eps} "
                    f"allow_latent_tail={inst.allow_latent_tail} "
                    f"allow_mnnvl_deferred={inst.allow_mnnvl_deferred} "
                    f"allow_fused_rs_up_ag={inst.allow_fused_rs_up_ag} "
                    f"mnnvl_bt_only={inst.mnnvl_bt_only} "
                    f"allow_medium_fused_rs_up_ag={inst.allow_medium_fused_rs_up_ag} "
                    f"mnnvl_graph_max_tokens={inst.mnnvl_graph_max_tokens}, but a later "
                    f"caller asked for hidden={hidden_size} "
                    f"latent={latent_size} top_k={top_k} "
                    f"rms_eps={float(rms_eps)} "
                    f"allow_latent_tail={allow_latent_tail} "
                    f"allow_mnnvl_deferred={allow_mnnvl_deferred} "
                    f"allow_fused_rs_up_ag={allow_fused_rs_up_ag} "
                    f"mnnvl_bt_only={mnnvl_bt_only} "
                    f"allow_medium_fused_rs_up_ag={allow_medium_fused_rs_up_ag} "
                    f"mnnvl_graph_max_tokens={mnnvl_graph_max_tokens}; the current "
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
        allow_mnnvl_deferred,
        allow_fused_rs_up_ag,
        mnnvl_bt_only,
        allow_medium_fused_rs_up_ag,
        mnnvl_graph_max_tokens,
    ):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.top_k = top_k
        self.rms_eps = float(rms_eps)
        self.allow_latent_tail = allow_latent_tail
        self.allow_mnnvl_deferred = allow_mnnvl_deferred
        self.allow_fused_rs_up_ag = allow_fused_rs_up_ag
        self.mnnvl_bt_only = mnnvl_bt_only
        self.allow_medium_fused_rs_up_ag = allow_medium_fused_rs_up_ag
        self.integrated_tail = _integrated_tail_enabled()
        bt_min_tokens = 33 if self.integrated_tail else MNNVL_BT_MIN_TOKENS
        ht_min_tokens = 1025 if self.integrated_tail else MNNVL_HT_MIN_TOKENS
        self.bt_candidate_tokens = (
            MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS
            if mnnvl_bt_only
            else MNNVL_BT_QUALIFIED_TOKENS
        )
        self.mnnvl_graph_max_tokens = mnnvl_graph_max_tokens
        self.multimem_ar_ok = False
        self.mnnvl_bt_deferred = None
        self.mnnvl_ht_deferred = None
        self.fused_rs_up_ag_finalize = None
        self.fused_rs_up_ag_workspace = None
        self.fused_rs_up_ag_capacity = 0
        self.integrated_tail_output_pool = None
        self.medium_fused_rs_up_ag_workspace = None
        self.medium_fused_rs_up_ag_capacity = 0
        self.latent_tail_ok = False  # per-layer ops built by K3MoeTailComm
        if not dist.is_initialized():
            return

        world = dist.get_world_size()
        hidden, latent = hidden_size, latent_size
        bt_capacity = min(MNNVL_BT_MAX_TOKENS, mnnvl_graph_max_tokens)
        ht_capacity = min(MNNVL_HT_MAX_TOKENS, mnnvl_graph_max_tokens)
        fused_capacity = max(
            (m for m in FUSED_RS_UP_AG_TOKENS if m <= mnnvl_graph_max_tokens),
            default=0,
        )
        if self.integrated_tail:
            fused_capacity = 8192
        medium_capacity = max(
            (m for m in self.bt_candidate_tokens if m <= mnnvl_graph_max_tokens),
            default=0,
        )

        # --- local probes (pure local; failures just vote False) ---
        # All ranks must agree on eligibility before any collective
        # allocator runs.
        multimem_local = (
            mapping.moe.tp_ep_size > 1
            and mapping.moe.tp_ep_size == world
            and mapping.attn.dp_size == 1
            and mapping.attn.cp_size == 1
            # Equal widths would alias the two per-width staging buffers.
            and latent != hidden
            and multimem_available()
            # Cross-node symmetric-memory rendezvous requires fabric/IMEX.
            and (
                world <= torch.cuda.device_count()
                or fabric_allocation_supported(torch.cuda.current_device())
            )
        )
        mnnvl_bt_local = (
            allow_mnnvl_deferred
            and bt_capacity >= bt_min_tokens
            and multimem_local
            and mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported(
                group=dist.group.WORLD,
                tp_size=mapping.moe.tp_ep_size,
                hidden_size=latent,
                top_k=top_k,
                dtype=torch.bfloat16,
                candidate_min_tokens=bt_min_tokens,
                candidate_max_tokens=bt_capacity,
            )
        )
        mnnvl_ht_local = (
            allow_mnnvl_deferred
            and not mnnvl_bt_only
            and ht_capacity >= ht_min_tokens
            and multimem_local
            and mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported(
                group=dist.group.WORLD,
                tp_size=mapping.moe.tp_ep_size,
                hidden_size=latent,
                top_k=top_k,
                dtype=torch.bfloat16,
                candidate_min_tokens=ht_min_tokens,
                candidate_max_tokens=ht_capacity,
            )
        )
        tail_local = False
        fused_local = (
            (allow_fused_rs_up_ag or self.integrated_tail)
            and mnnvl_ht_local
            and fused_capacity > 0
            and hidden == 7168
            and latent == 3584
            and top_k == 16
            and _mnnvl_tp8_layout(mapping)
        )
        medium_local = (
            allow_medium_fused_rs_up_ag
            and mnnvl_bt_only
            and mnnvl_bt_local
            and medium_capacity > 0
            and hidden == 7168
            and latent == 3584
            and top_k == 16
            and _mnnvl_tp8_layout(mapping)
        )
        flags_valid = _medium_flags_valid(
            mnnvl_bt_only, allow_medium_fused_rs_up_ag, _mnnvl_deferred_enabled()
        )
        if self.integrated_tail:
            flags_valid = (
                flags_valid
                and not any(
                    os.environ.get(flag) == "1"
                    for flag in (
                        MNNVL_BT_ONLY_ENABLE_ENV,
                        MEDIUM_FUSED_RS_UP_AG_ENABLE_ENV,
                        MNNVL_DEFERRED_ENABLE_ENV,
                        FUSED_RS_UP_AG_ENABLE_ENV,
                    )
                )
                and mnnvl_graph_max_tokens == 8192
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
                int(multimem_local),
                int(tail_local),
                int(mnnvl_bt_local),
                int(mnnvl_ht_local),
                int(fused_local),
                int(mnnvl_bt_only),
                -int(mnnvl_bt_only),
                int(allow_medium_fused_rs_up_ag),
                -int(allow_medium_fused_rs_up_ag),
                int(flags_valid),
                int(medium_local),
                int(self.integrated_tail),
                -int(self.integrated_tail),
            ],
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(votes, op=dist.ReduceOp.MIN)
        values = votes.tolist()
        (
            bt_min,
            bt_negative_min,
            medium_min,
            medium_negative_min,
            flags_ok,
            medium_ok,
        ) = values[5:11]
        if (
            not flags_ok
            or bt_min != -bt_negative_min
            or medium_min != -medium_negative_min
            or values[11] != -values[12]
        ):
            raise ValueError(
                "medium flags must agree across ranks, require BT_ONLY=1 for medium "
                "fusion, and cannot coexist with legacy MNNVL/full-fusion flags; "
                "integrated mode requires graph max 8192 and no ablation flags"
            )
        multimem_ok, tail_ok, mnnvl_bt_ok, mnnvl_ht_ok, fused_ok = (
            bool(v) for v in values[:5]
        )
        if self.integrated_tail and not all(
            (multimem_ok, tail_ok, mnnvl_bt_ok, mnnvl_ht_ok, fused_ok)
        ):
            raise RuntimeError(
                "integrated K3 tail requires every TP8 capability; no fallback"
            )

        # --- collective builds, fixed order, unguarded ---
        if multimem_ok:
            # Collective buffers must reach peak size before serving.
            self.multimem_ar_ok = multimem_prealloc(
                MULTIMEM_AR_MAX_TOKENS,
                (latent, hidden),
                dist.group.WORLD.group_name,
            )
        if mnnvl_bt_ok and self.multimem_ar_ok:
            self.mnnvl_bt_deferred = MNNVLCuteDSLFinalizeAllReduceRMSNorm.initialize(
                group=dist.group.WORLD,
                hidden_size=latent,
                top_k=top_k,
                rms_eps=self.rms_eps,
                candidate_min_tokens=bt_min_tokens,
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
        if mnnvl_ht_ok and self.multimem_ar_ok and not self.integrated_tail:
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
                        stages=7,
                        reduction_warps=2,
                        reduction_cta_groups=None,
                        rms_token_groups=2,
                        rms_pipeline_stages=3,
                        rms_shard_major=False,
                        enable_pdl=True,
                    ),
                ),
            )
        if fused_ok and self.multimem_ar_ok:
            from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import (
                SharedRsWorkspace,
            )

            self.fused_rs_up_ag_finalize = (
                MNNVLCuteDSLHTFinalizeAllReduceRMSNorm.initialize(
                    group=dist.group.WORLD,
                    hidden_size=latent,
                    top_k=top_k,
                    rms_eps=self.rms_eps,
                    candidate_min_tokens=(
                        ht_min_tokens
                        if self.integrated_tail
                        else min(FUSED_RS_UP_AG_TOKENS)
                    ),
                    candidate_max_tokens=fused_capacity,
                    tuning_routes=(
                        MNNVLCuteDSLHTFinalizeTuning(
                            max_tokens=fused_capacity,
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
            )
            self.fused_rs_up_ag_workspace = SharedRsWorkspace.allocate(
                dist.group.WORLD,
                fused_capacity,
                torch.device("cuda", torch.cuda.current_device()),
            )
            self.fused_rs_up_ag_capacity = fused_capacity
            if self.integrated_tail:
                self.mnnvl_ht_deferred = self.fused_rs_up_ag_finalize
        if medium_ok and self.multimem_ar_ok:
            from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import (
                SharedRsWorkspace,
            )

            self.medium_fused_rs_up_ag_workspace = SharedRsWorkspace.allocate(
                dist.group.WORLD,
                medium_capacity,
                torch.device("cuda", torch.cuda.current_device()),
            )
            self.medium_fused_rs_up_ag_capacity = medium_capacity
        self.latent_tail_ok = tail_ok
        logger.info(
            "K3 comm negotiated: multimem=%s mnnvl_bt_deferred=%s "
            "mnnvl_ht_deferred=%s fused_rs_up_ag_capacity=%d latent_tail=%s",
            self.multimem_ar_ok,
            self.mnnvl_bt_deferred is not None,
            self.mnnvl_ht_deferred is not None,
            self.fused_rs_up_ag_capacity,
            self.latent_tail_ok,
        )
        if mnnvl_bt_only:
            logger.warning(
                "Unqualified medium experiment: BT-only bucket policy=%s, "
                "fused capacity=%d; no HT enabled",
                sorted(self.bt_candidate_tokens),
                self.medium_fused_rs_up_ag_capacity,
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
            if out_norm_w is not None:
                from tokenspeed_kernel.ops.communication.triton import (
                    allreduce_residual_attnres_combine,
                    allreduce_residual_attnres_combine_supported,
                )

                group = _get_process_group(self.mapping.attn.tp_group)
                fused_supported = not global_server_args_dict.get(
                    "force_deterministic_rsag", False
                ) and allreduce_residual_attnres_combine_supported(
                    attn_partial,
                    prefix_sum,
                    mlp_wp,
                    out_norm_w,
                    scratch,
                    rank=self.mapping.attn.tp_rank,
                    group=group,
                    local_world_size=self.mapping.nprocs_per_node,
                )
                if fused_supported:
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
            ``do_finalize=False`` (the tail owns finalize). Set for
            TAIL_FUSION when the multicast tail was armed to inline
            finalize (trtllm fused-AR deployments).
        lane: Pre-materialized fused-lane buffer, or None; when set the
            experts kernel writes its routed partial into
            ``lane[:, :routed_hidden]`` and the shared experts into the rest.
        symm_outputs: Producer-direct (routed, shared) views of symmetric
            memory, or None. When set the producers write into these and the
            tail consumes them in place; ``lane`` is None in that case.
            The routed view is None for deferred-finalize complete fusion:
            only the shared producer writes a materialized partial.
        routed_in_fork: Whether the routed partial must be reduced and
            projected inside the fork (SEPARATE_REDUCE overlap).
        split_shared_rs: Start the shared ReduceScatter on the auxiliary
            stream before the routed collective is ready.
    """

    tier: "K3MoETailTier"
    defer_finalize: bool = False
    lane: torch.Tensor | None = None
    symm_outputs: tuple[torch.Tensor | None, torch.Tensor] | None = None
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


def _acquire_symm_join_outputs(
    *, mapping, routed_hidden: int, hidden_size: int, like, enabled: bool
):
    """Acquire the routed/shared pair from symmetric memory, or None.

    Returns two consecutive views of the Iris input buffer -- not fresh
    allocations -- so the producers write where the reduction already reads and
    the pointers stay put across graph capture.

    The pair matters, not just the memory: ``AutoBackend.all_reduce`` only
    consults ``can_reduce_outputs`` on its tuple branch, so a single
    concatenated operand can never reach the symmetric kernel however it was
    allocated. ``latent_moe_expert_shared_all_reduce`` uses this pair on AMD
    for both TP/TP and TP/EP mappings; the collective group is MoE TP x EP.

    Collective in the same sense the acquire is: every rank of the MoE TP x EP
    group reaches this with rank-uniform shapes, so none can disagree about
    whether the pair exists.
    """
    if not enabled or mapping.moe.tp_ep_size <= 1 or like is None:
        return None
    if like.ndim != 2:
        return None
    num_tokens = like.shape[0]
    if num_tokens <= 0:
        return None
    shapes = ((num_tokens, routed_hidden), (num_tokens, hidden_size))
    group = mapping.moe.tp_ep_group
    if not can_acquire_all_reduce_outputs(shapes, like, group):
        return None
    return acquire_all_reduce_outputs(shapes, like, group)


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
        mnnvl_graph_max_tokens = _mnnvl_graph_max_tokens()
        self.state = K3MoeTailCommState.get(
            mapping=mapping,
            hidden_size=hidden_size,
            latent_size=routed_hidden,
            top_k=top_k,
            rms_eps=(routed_norm.variance_epsilon if routed_norm is not None else 1e-5),
            allow_latent_tail=(
                not execution_plan.use_native and routed_norm is not None
            ),
            allow_mnnvl_deferred=(
                (
                    _mnnvl_deferred_enabled()
                    or _mnnvl_bt_only_enabled()
                    or _integrated_tail_enabled()
                )
                and execution_plan.fused_moe_ar
                and _mnnvl_tp8_layout(mapping)
                and routed_norm is not None
                and experts_supports_deferred_finalize
                and self._shard_up_projection
                and (
                    _integrated_tail_enabled()
                    or not global_server_args_dict["enforce_eager"]
                )
                and (
                    _integrated_tail_enabled()
                    or not global_server_args_dict["disable_prefill_graph"]
                )
                and not global_server_args_dict["disable_pdl"]
                and mnnvl_graph_max_tokens >= MNNVL_BT_MIN_TOKENS
            ),
            allow_fused_rs_up_ag=_fused_rs_up_ag_enabled(),
            mnnvl_bt_only=_mnnvl_bt_only_enabled(),
            allow_medium_fused_rs_up_ag=_medium_fused_rs_up_ag_enabled(),
            mnnvl_graph_max_tokens=mnnvl_graph_max_tokens,
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
        self.fused_rs_up_ag = None
        self.medium_fused_rs_up_ag = None
        if self.state.medium_fused_rs_up_ag_workspace is not None:
            from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving import (
                MediumFusedRsUpProjectionServing,
            )

            self.medium_fused_rs_up_ag = MediumFusedRsUpProjectionServing(
                self.state.medium_fused_rs_up_ag_workspace,
                self.state.medium_fused_rs_up_ag_capacity,
            )
        if self.state.fused_rs_up_ag_workspace is not None:
            from tokenspeed_kernel.ops.communication.cutedsl_fused_rs_up_projection import (
                FusedRsUpProjectionServing,
            )

            # Integrated sequential layers alternate two symmetric outputs.
            # AttnRes snapshots and EAGLE3 taps copy before either slot is reused.
            # Historical profiles retain their separate per-layer outputs.
            if self.state.integrated_tail:
                from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving import (
                    IntegratedFusedRsOutputPool,
                )

                if self.state.integrated_tail_output_pool is None:
                    self.state.integrated_tail_output_pool = (
                        IntegratedFusedRsOutputPool(
                            self.state.fused_rs_up_ag_workspace,
                            self.state.fused_rs_up_ag_capacity,
                        )
                    )
                self.fused_rs_up_ag = self.state.integrated_tail_output_pool.bind_layer(
                    layer_index
                )
            else:
                self.fused_rs_up_ag = FusedRsUpProjectionServing(
                    self.state.fused_rs_up_ag_workspace,
                    self.state.fused_rs_up_ag_capacity,
                )
            if self.state.integrated_tail:
                # Same raw storage and parity output; only first-stage and
                # fused GEMM geometry depend on M. No duplicate large buffers.
                self.medium_fused_rs_up_ag = self.fused_rs_up_ag
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
                "(%s, deferred_finalize=%s, split_shared_rs=%s)",
                prefix,
                tail_finalize_top_k is not None,
                self.latent_tail.supports_split_collective,
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def plan(
        self,
        num_tokens: int,
        hidden_states: torch.Tensor,
        *,
        is_decode: bool = False,
    ) -> TailPlan:
        """Pick the tail tier and its forward-side obligations.

        Every input must be rank-uniform (token count, graph phase and the
        negotiated capabilities) so all ranks take identical branches.
        """
        bt_workspace = getattr(self.state, "mnnvl_bt_deferred", None)
        ht_workspace = getattr(self.state, "mnnvl_ht_deferred", None)
        if getattr(self.state, "integrated_tail", False):
            integrated_tier = select_integrated_k3_moe_tail_tier(num_tokens)
            if integrated_tier is K3MoETailTier.SEPARATE_REDUCE:
                return TailPlan(tier=integrated_tier, routed_in_fork=True)
            if integrated_tier is not None:
                finalize = bt_workspace if num_tokens <= 1024 else ht_workspace
                if (
                    self.fused_rs_up_ag is None
                    or finalize is None
                    or not finalize.supports_num_tokens(num_tokens)
                    or not self._experts_supports_deferred_finalize
                ):
                    raise RuntimeError(
                        f"integrated K3 tail missing M={num_tokens} route"
                    )
                return TailPlan(
                    tier=integrated_tier,
                    defer_finalize=True,
                    symm_outputs=(None, self.fused_rs_up_ag.input_view(num_tokens)),
                )
        # Graph warmup, capture, and replay must select the same tier.
        tier = select_k3_moe_tail_tier(
            num_tokens=num_tokens,
            graph_phase=get_is_cuda_graph_phase(),
            tail_fusion_max_tokens=(
                min(self.latent_tail.max_num_tokens, TAIL_FUSION_MAX_TOKENS)
                if self.latent_tail is not None
                else 0
            ),
            fused_moe_ar=self.execution_plan.fused_moe_ar,
            join_moe_reduce=self.execution_plan.join_moe_reduce,
            multimem_ok=self.state.multimem_ar_ok,
            is_decode=is_decode,
            mnnvl_bt_deferred_ok=(
                getattr(self, "_experts_supports_deferred_finalize", False)
                and bt_workspace is not None
                and num_tokens
                in getattr(self.state, "bt_candidate_tokens", MNNVL_BT_QUALIFIED_TOKENS)
                and bt_workspace.supports_num_tokens(num_tokens)
            ),
            mnnvl_ht_deferred_ok=(
                getattr(self, "_experts_supports_deferred_finalize", False)
                and ht_workspace is not None
                and num_tokens in MNNVL_HT_QUALIFIED_TOKENS
                and ht_workspace.supports_num_tokens(num_tokens)
            ),
            fused_rs_up_ag_ok=(
                getattr(self, "fused_rs_up_ag", None) is not None
                and num_tokens <= self.state.fused_rs_up_ag_capacity
            ),
            prefill_graph_phase=get_is_prefill_graph_phase(),
        )
        # Upgrade only an already eligible BT prefill tier. Both new arms use
        # the same provisional bucket policy and exact same BT workspace.
        if (
            tier is K3MoETailTier.MNNVL_BT_DEFERRED
            and getattr(self, "medium_fused_rs_up_ag", None) is not None
        ):
            return TailPlan(
                tier=K3MoETailTier.MEDIUM_FUSED_RS_UP_AG,
                defer_finalize=True,
                symm_outputs=(None, self.medium_fused_rs_up_ag.input_view(num_tokens)),
            )
        if tier is K3MoETailTier.FUSED_RS_UP_AG:
            return TailPlan(
                tier=tier,
                defer_finalize=True,
                symm_outputs=(None, self.fused_rs_up_ag.input_view(num_tokens)),
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
        if tier in (
            K3MoETailTier.MNNVL_BT_DEFERRED,
            K3MoETailTier.MNNVL_HT_DEFERRED,
        ):
            return TailPlan(tier=tier, defer_finalize=True)
        # Shard mode splits the joined reduction, so it cannot use the packed lane.
        lane_enabled = (
            tier is K3MoETailTier.FUSED_LANE_AR and not self._shard_up_projection
        )
        # Preferred: the producers write straight into symmetric memory and the
        # join reduces them there. Falls back to the packed lane, which holds
        # ordinary memory the collective has to stage first.
        symm_outputs = _acquire_symm_join_outputs(
            mapping=self.mapping,
            routed_hidden=self.routed_hidden,
            hidden_size=self.hidden_size,
            like=hidden_states,
            enabled=lane_enabled,
        )
        lane = (
            None
            if symm_outputs is not None
            else allreduce_fusion_lane(
                hidden_states,
                self.routed_hidden + self.hidden_size,
                enabled=lane_enabled,
            )
        )
        return TailPlan(
            tier=tier,
            lane=lane,
            symm_outputs=symm_outputs,
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
        if tier is K3MoETailTier.MEDIUM_FUSED_RS_UP_AG:
            gemm2_out, expert_weights, expanded_idx = routed_out
            routed_norm = self.state.mnnvl_bt_deferred(
                gemm2_out, expert_weights, expanded_idx, self.routed_norm.weight
            )
            return self.medium_fused_rs_up_ag(
                routed_norm,
                self.up_proj.weight,
                prefix_sum.view(num_tokens, hidden_size),
                shared_partial,
            )
        if tier is K3MoETailTier.FUSED_RS_UP_AG:
            gemm2_out, expert_weights, expanded_idx = routed_out
            routed_norm = self.state.fused_rs_up_ag_finalize(
                gemm2_out, expert_weights, expanded_idx, self.routed_norm.weight
            )
            return self.fused_rs_up_ag(
                routed_norm,
                self.up_proj.weight,
                prefix_sum.view(num_tokens, hidden_size),
                shared_partial,
            )
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
        if tier in (
            K3MoETailTier.MNNVL_BT_DEFERRED,
            K3MoETailTier.MNNVL_HT_DEFERRED,
        ):
            gemm2_out, expert_weights, expanded_idx = routed_out
            workspace = (
                self.state.mnnvl_bt_deferred
                if tier is K3MoETailTier.MNNVL_BT_DEFERRED
                else self.state.mnnvl_ht_deferred
            )
            if workspace is None:
                raise RuntimeError(
                    "deferred MNNVL tier selected without an initialized workspace"
                )
            return self._tail_mnnvl_deferred_sharded(
                workspace,
                gemm2_out,
                expert_weights,
                expanded_idx,
                shared_partial,
                prefix_sum,
                num_tokens,
                hidden_size,
                use_multimem=(
                    tier is K3MoETailTier.MNNVL_HT_DEFERRED
                    or num_tokens >= MULTIMEM_AR_MIN_TOKENS
                ),
            )
        if tier is K3MoETailTier.MULTIMEM_AR:
            return self._tail_multimem_ar(
                routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
            )
        if tier is K3MoETailTier.FUSED_LANE_AR:
            return self._tail_fused_lane_ar(
                routed_out,
                shared_partial,
                prefix_sum,
                plan.lane,
                plan.symm_outputs,
                num_tokens,
                hidden_size,
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

    def _tail_mnnvl_deferred_sharded(
        self,
        workspace,
        gemm2_out: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
        *,
        use_multimem: bool,
    ) -> torch.Tensor:
        """Fuse routed finalize/AR/norm, then retain the production tail.

        The candidate is restricted to K3's sharded up projection.  Below the
        production multimem threshold its second collective remains the
        runtime-selected all-reduce backend;
        above it, shared staging is issued before the persistent candidate so
        both the baseline and candidate preserve the established copy order.
        """

        if not self._shard_up_projection:
            raise RuntimeError("deferred MNNVL tail requires a sharded up projection")
        group_name = dist.group.WORLD.group_name
        shared_stage = None
        if use_multimem:
            shared_stage = multimem_stage(
                shared_partial, group_name, MULTIMEM_AR_MAX_TOKENS
            )
            if shared_stage is None:
                raise RuntimeError(
                    "shared multimem staging failed after deferred tier selection"
                )

        routed_norm = workspace(
            gemm2_out,
            expert_weights,
            expanded_idx,
            self.routed_norm.weight,
        )
        shared_work = shared_stage if shared_stage is not None else shared_partial
        shared_work = self._project_and_inject_local_block(
            routed_norm,
            shared_work,
            prefix_sum,
            num_tokens,
            hidden_size,
        )
        if shared_stage is not None:
            shared_reduced = multimem_all_reduce_staged(shared_work, group_name)
            return shared_reduced.view(num_tokens, hidden_size).clone()
        return all_reduce(shared_work, self.mapping.moe.tp_ep_group)

    def _tail_multimem_ar(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        if self._shard_up_projection:
            return self._tail_multimem_ar_sharded(
                routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
            )
        return self._tail_multimem_ar_replicated(
            routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
        )

    def _tail_multimem_ar_sharded(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        # Eligibility guarantees tp_ep spans WORLD for these symmetric reductions.
        group_name = dist.group.WORLD.group_name
        routed_stage = multimem_stage(routed_out, group_name, MULTIMEM_AR_MAX_TOKENS)
        shared_stage = (
            multimem_stage(shared_partial, group_name, MULTIMEM_AR_MAX_TOKENS)
            if routed_stage is not None
            else None
        )
        # Eligibility guarantees both stages exist; a miss is a contract violation.
        if routed_stage is None or shared_stage is None:
            raise RuntimeError(
                "multimem staging failed after the tier was selected; the "
                "init-time capability vote and the runtime shapes disagree"
            )
        routed_reduced = multimem_all_reduce_staged(routed_stage, group_name)
        # Disjoint projection shards concatenate when injected into the shared sum.
        routed_reduced = (
            self.routed_norm(routed_reduced)
            if self.routed_norm is not None
            else routed_reduced
        )
        start, width = self.up_proj.shard_slice
        target = shared_stage[:, start : start + width]
        target += prefix_sum.view(num_tokens, hidden_size).narrow(-1, start, width)
        target.addmm_(routed_reduced, self.up_proj.weight.t())
        shared_reduced = multimem_all_reduce_staged(shared_stage, group_name)
        # Clone: the staging buffer is recycled by the next layer's stage copy.
        return shared_reduced.view(num_tokens, hidden_size).clone()

    def _tail_multimem_ar_replicated(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        # Eligibility guarantees tp_ep spans WORLD for these symmetric reductions.
        group_name = dist.group.WORLD.group_name
        routed_stage = multimem_stage(routed_out, group_name, MULTIMEM_AR_MAX_TOKENS)
        shared_stage = (
            multimem_stage(shared_partial, group_name, MULTIMEM_AR_MAX_TOKENS)
            if routed_stage is not None
            else None
        )
        # Eligibility guarantees both stages exist; a miss is a contract violation.
        if routed_stage is None or shared_stage is None:
            raise RuntimeError(
                "multimem staging failed after the tier was selected; the "
                "init-time capability vote and the runtime shapes disagree"
            )
        routed_reduced = multimem_all_reduce_staged(routed_stage, group_name)
        shared_reduced = multimem_all_reduce_staged(shared_stage, group_name)
        if self.routed_norm is not None:
            routed_reduced = self.routed_norm(routed_reduced)
        return self._projection_tail(
            routed_reduced, shared_reduced, prefix_sum, num_tokens, hidden_size
        )

    def _projection_tail(
        self,
        routed_reduced: torch.Tensor,
        shared_reduced: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        return self.up_proj.forward_add3(
            routed_reduced,
            prefix_sum,
            shared_reduced,
        ).view(num_tokens, hidden_size)

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

    def _tail_fused_lane_ar(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        lane: torch.Tensor | None,
        symm_outputs: tuple[torch.Tensor, torch.Tensor] | None,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        if self._shard_up_projection:
            return self._tail_fused_lane_ar_sharded(
                routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
            )
        return self._tail_fused_lane_ar_replicated(
            routed_out,
            shared_partial,
            prefix_sum,
            lane,
            symm_outputs,
            num_tokens,
            hidden_size,
        )

    def _tail_fused_lane_ar_sharded(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        # Sharded projection folds into the shared reduction; the packed lane is disabled here.
        routed_reduced = all_reduce(routed_out, self.mapping.moe.tp_ep_group)
        if self.routed_norm is not None:
            routed_reduced = self.routed_norm(routed_reduced)
        shared_partial = self._project_and_inject_local_block(
            routed_reduced, shared_partial, prefix_sum, num_tokens, hidden_size
        )
        return all_reduce(shared_partial, self.mapping.moe.tp_ep_group)

    def _tail_fused_lane_ar_replicated(
        self,
        routed_out: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
        lane: torch.Tensor | None,
        symm_outputs: tuple[torch.Tensor, torch.Tensor] | None,
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        routed_reduced, shared_reduced = kimi3_join_reduce_moe(
            routed_out,
            shared_partial,
            lane=lane,
            symm_outputs=symm_outputs,
            routed_hidden=self.routed_hidden,
            routed_norm=self.routed_norm,
            group=self.mapping.moe.tp_ep_group,
            enable_lane_norm=self.execution_plan.lane_latent_norm_ar,
            max_token_num=self.execution_plan.comm_fusion_max_num_tokens,
        )
        return self._projection_tail(
            routed_reduced, shared_reduced, prefix_sum, num_tokens, hidden_size
        )

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
    "MULTIMEM_AR_MAX_TOKENS",
    "MULTIMEM_AR_MIN_TOKENS",
    "TailPlan",
    "select_k3_moe_tail_tier",
]
