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
fused-lane one-shot     everything else with a fused plan
======================  =========================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.activation.triton import add3
from tokenspeed_kernel.ops.communication import allreduce_fusion_lane
from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
from tokenspeed_kernel.ops.communication.multimem import (
    multimem_all_reduce_staged,
    multimem_available,
    multimem_prealloc,
    multimem_stage,
)
from tokenspeed_kernel.ops.moe.latent_tail import (
    KimiK3LatentTailOp,
    latent_tail_supported,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    prepare_all_reduce_fusion,
    prepare_all_reduce_lane,
)
from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.layers.layernorm import RMSNorm, _get_process_group
from tokenspeed.runtime.layers.moe.latent import kimi3_join_reduce_moe
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)


class K3MoETailTier(IntEnum):
    """How the K3 MoE tail combines routed/shared partials, best first.

    Declaration order *is* the priority order (mirrors the selector's
    branch order). Values never escape the process (identity comparisons
    only, no serialization), so inserting mid-list is safe — keep new
    tiers at their semantic rank rather than appending.
    """

    TAIL_FUSION = 0  # fused decode kernel (aka the multicast latent tail)
    MULTIMEM_AR = 1  # in-switch (ld_reduce) reduces, then the replicated tail
    FUSED_LANE_AR = 2  # join tier: lane one-shot / cat+one-shot / grouped NCCL
    SEPARATE_REDUCE = 3  # portable: reduce each partial on its own


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


def select_k3_moe_tail_tier(
    *,
    num_tokens: int,
    graph_phase: bool,
    tail_fusion_max_tokens: int,
    fused_moe_ar: bool,
    multimem_ok: bool,
    is_decode: bool = False,
    join_moe_reduce: bool = False,
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

    Returns:
        The best applicable ``K3MoETailTier``.
    """
    # Tested first, so a fused-tail capacity that ever reached into the
    # multimem window would still resolve here rather than overlap.
    if graph_phase and 1 <= num_tokens <= tail_fusion_max_tokens:
        return K3MoETailTier.TAIL_FUSION
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
    ) -> "K3MoeTailCommState":
        if cls._instance is None:
            cls._instance = cls(
                mapping=mapping,
                hidden_size=hidden_size,
                latent_size=latent_size,
                top_k=top_k,
                rms_eps=rms_eps,
                allow_latent_tail=allow_latent_tail,
            )
        else:
            inst = cls._instance
            if (
                inst.hidden_size != hidden_size
                or inst.latent_size != latent_size
                or inst.top_k != top_k
                or inst.rms_eps != float(rms_eps)
                or inst.allow_latent_tail != allow_latent_tail
            ):
                # The singleton would otherwise silently hand a second model
                # (e.g. an MTP draft sharing the process with its base) a
                # negotiation done for the wrong shapes.
                raise ValueError(
                    "K3MoeTailCommState is negotiated once per process for "
                    f"hidden={inst.hidden_size} latent={inst.latent_size} "
                    f"top_k={inst.top_k} rms_eps={inst.rms_eps} "
                    f"allow_latent_tail={inst.allow_latent_tail}, but a later "
                    f"caller asked for hidden={hidden_size} "
                    f"latent={latent_size} top_k={top_k} "
                    f"rms_eps={float(rms_eps)} "
                    f"allow_latent_tail={allow_latent_tail}; the current "
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
    ):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.top_k = top_k
        self.rms_eps = float(rms_eps)
        self.allow_latent_tail = allow_latent_tail
        self.multimem_ar_ok = False
        self.latent_tail_ok = False  # per-layer ops built by K3MoeTailComm
        if not dist.is_initialized():
            return

        world = dist.get_world_size()
        hidden, latent = hidden_size, latent_size

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
        tail_local = False
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
            [int(multimem_local), int(tail_local)],
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(votes, op=dist.ReduceOp.MIN)
        multimem_ok, tail_ok = (bool(v) for v in votes.tolist())

        # --- collective builds, fixed order, unguarded ---
        if multimem_ok:
            # Collective buffers must reach peak size before serving.
            self.multimem_ar_ok = multimem_prealloc(
                MULTIMEM_AR_MAX_TOKENS,
                (latent, hidden),
                dist.group.WORLD.group_name,
            )
        self.latent_tail_ok = tail_ok
        logger.info(
            "K3 comm negotiated: multimem=%s latent_tail=%s",
            self.multimem_ar_ok,
            self.latent_tail_ok,
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
    # Attention-side reduction (moved verbatim from
    # KimiLinearDecoderLayer._reduce_attn_accumulate; behavior unchanged).
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

        ``mlp_wp`` is the calling layer's precomputed ``rms_w * res_w``
        product (per-layer state, filled in post_load_weights); the B1
        combine kernels consume it in place of the separate weights.
        """
        num_tokens = attn_partial.shape[0]
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
        routed_in_fork: Whether the routed partial must be reduced and
            projected inside the fork (SEPARATE_REDUCE overlap).
        split_shared_rs: Start the shared ReduceScatter on the auxiliary
            stream before the routed collective is ready.
    """

    tier: "K3MoETailTier"
    defer_finalize: bool = False
    lane: torch.Tensor | None = None
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
        self.state = K3MoeTailCommState.get(
            mapping=mapping,
            hidden_size=hidden_size,
            latent_size=routed_hidden,
            top_k=top_k,
            rms_eps=(routed_norm.variance_epsilon if routed_norm is not None else 1e-6),
            allow_latent_tail=(
                not execution_plan.use_native and routed_norm is not None
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
        # Shard mode splits the joined reduction, so it cannot use the packed lane.
        lane = allreduce_fusion_lane(
            hidden_states,
            self.routed_hidden + self.hidden_size,
            enabled=(
                tier is K3MoETailTier.FUSED_LANE_AR and not self._shard_up_projection
            ),
        )
        return TailPlan(
            tier=tier,
            lane=lane,
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
        deferred-finalize TAIL_FUSION (``plan.defer_finalize``), whose
        ``routed_out`` is the experts kernel's deferred-finalize triple.
        """
        tier = plan.tier
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
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        if self._shard_up_projection:
            return self._tail_fused_lane_ar_sharded(
                routed_out, shared_partial, prefix_sum, num_tokens, hidden_size
            )
        return self._tail_fused_lane_ar_replicated(
            routed_out, shared_partial, prefix_sum, lane, num_tokens, hidden_size
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
        num_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor:
        routed_reduced, shared_reduced = kimi3_join_reduce_moe(
            routed_out,
            shared_partial,
            lane=lane,
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
