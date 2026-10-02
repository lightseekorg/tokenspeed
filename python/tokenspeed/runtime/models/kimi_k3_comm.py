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

The routed all-reduce and RMSNorm run before joining the shared-expert stream.
The up-projection and shared all-reduce assemble the output after the join.
AMD and attention-DP execution are owned by the model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ClassVar

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication import (
    AllReduceFusionPattern,
    AllReduceFusionWorkspace,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)
from tokenspeed_kernel.ops.communication.multimem import (
    multimem_all_reduce_staged,
    multimem_prealloc,
    multimem_stage,
)
from tokenspeed_kernel.ops.moe.latent_tail import (
    K3_SHARED_RS_MAX_TOKENS,
    KimiK3LatentTailOp,
    attn_reduce_shape_supported,
    build_attn_reduce_collective,
    multicast_backend_available,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_ops import (
    acquire_all_reduce_outputs,
    all_reduce,
    can_acquire_all_reduce_outputs,
    prepare_all_reduce_fusion,
    prepare_all_reduce_lane,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm, _get_process_group
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)

_MULTIMEM_AR_MIN_TOKENS = 1024

_IRIS_MAX_TOKENS = 8192
_IRIS_ATTN_PRODUCER_DIRECT_MIN_TOKENS = 16
_IRIS_ATTN_SHARDED_PREFIX_MIN_TOKENS = 56

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


def _unified_tail_applicable(
    *,
    mapping,
    hidden_size: int,
    latent_size: int,
    top_k: int,
    is_blackwell: bool,
    has_routed_norm: bool,
) -> bool:
    return (
        is_blackwell
        and mapping.moe.tp_ep_size in (4, 8, 16)
        and mapping.attn.tp_size == mapping.moe.tp_ep_size
        and hidden_size == 7168
        and latent_size == 3584
        and top_k == 16
        and has_routed_norm
    )


class K3AttnComm:
    """Attention all-reduce, residual fusion, and shared workspaces for Kimi-K3.

    Each decoder layer owns an instance. The first instance initializes the
    class-level workspaces, which subsequent instances reuse. Layer-specific
    weights are supplied to each reduction call.
    """

    _prepared_hidden_size: ClassVar[int | None] = None
    attn_ar_fusion_ok: ClassVar[bool] = False
    dummy_norm: ClassVar[RMSNorm | None] = None
    cute_ar: ClassVar[Callable | None] = None

    def __init__(self, *, mapping, hidden_size: int) -> None:
        self.mapping = mapping
        self.hidden_size = hidden_size
        if self._prepared_hidden_size is not None:
            if hidden_size != self._prepared_hidden_size:
                raise ValueError(
                    "K3 attention workspace hidden size differs from its initial configuration"
                )
            return
        hidden = hidden_size
        # Fused AR+residual for the attention reduce: a ones-weight RMSNorm
        # rides the one-shot pattern and its norm output is discarded.
        K3AttnComm.attn_ar_fusion_ok = dist.is_initialized() and (
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
        K3AttnComm.dummy_norm = RMSNorm(hidden, eps=1e-6)
        self.dummy_norm.weight.data = torch.ones(
            hidden,
            dtype=torch.bfloat16,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        self.dummy_norm.weight.requires_grad_(False)

        if (
            self.attn_ar_fusion_ok
            and global_server_args_dict["comm_fusion_max_num_tokens"] > 0
        ):
            group = _get_process_group(mapping.attn.tp_group)
            if multicast_backend_available(group) and attn_reduce_shape_supported(
                tp_size=mapping.attn.tp_size, hidden_size=hidden
            ):
                K3AttnComm.cute_ar = build_attn_reduce_collective(
                    group=group,
                    rank=mapping.attn.tp_rank,
                    tp_size=mapping.attn.tp_size,
                    hidden_size=hidden,
                    max_tokens=ATTN_AR_MAX_TOKENS,
                )
        K3AttnComm._prepared_hidden_size = hidden_size
        attention_reduce_backend = (
            f"tokenspeed CuteDSL collective at M<={ATTN_AR_MAX_TOKENS}"
            if self.cute_ar is not None
            else "not armed; the existing backends serve every M"
        )
        logger.info(f"Kimi K3 attention reduce: {attention_reduce_backend}")

    def acquire_projection_output(
        self,
        like: torch.Tensor,
        projection,
    ) -> torch.Tensor | None:
        """Return prepared storage for an eligible attention producer, or None."""
        from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod

        if (
            not current_platform().is_cdna4
            or like.ndim != 2
            or not _IRIS_ATTN_PRODUCER_DIRECT_MIN_TOKENS
            <= like.shape[0]
            <= _IRIS_MAX_TOKENS
            or like.dtype != torch.bfloat16
            or self.mapping.attn.tp_size != 8
            or self.mapping.moe.tp_size != 8
            or self.mapping.moe.ep_size != 1
            or self.mapping.attn.tp_group != self.mapping.moe.tp_ep_group
            or self.mapping.pp_size != 1
            or type(projection.quant_method) is not UnquantizedLinearMethod
            or projection.weight.dtype != torch.bfloat16
            or projection.weight.shape[0] != 7168
            or projection.bias is not None
            or projection.reduce_results
            or not projection.input_is_parallel
        ):
            return None
        shapes = ((like.shape[0], 7168),)
        group = self.mapping.attn.tp_group
        if not can_acquire_all_reduce_outputs(
            shapes, like, group, backend=None, op=dist.ReduceOp.SUM
        ):
            return None
        return acquire_all_reduce_outputs(
            shapes, like, group, backend=None, op=dist.ReduceOp.SUM
        )[0]

    def reduce_for_attnres(
        self,
        partial: torch.Tensor,
        prefix: torch.Tensor | None,
        *,
        producer_direct: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the residual and optional delta consumed by the AttnRes mixer."""
        if producer_direct:
            # Prepared Iris inputs reduce into owned storage, which remains
            # valid as a residual after the next producer reuses its input.
            reduced = all_reduce((partial,), self.mapping.attn.tp_group)[0]
        else:
            reduced = all_reduce(partial, self.mapping.attn.tp_group)
        return (reduced, None) if prefix is None else (prefix, reduced)

    def mix_for_moe(
        self,
        partial: torch.Tensor,
        prefix: torch.Tensor | None,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        *,
        eps: float,
        out_norm_weight: torch.Tensor,
        out_norm_eps: float,
        num_valid_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Mix a prepared projection, retaining the MoE's local residual rows."""
        if (
            not current_platform().is_cdna4
            or partial.ndim != 2
            or not _IRIS_ATTN_SHARDED_PREFIX_MIN_TOKENS
            <= partial.shape[0]
            <= _IRIS_MAX_TOKENS
            or partial.shape[0] % 8 != 0
        ):
            return None
        from tokenspeed_kernel.ops.communication.iris import (
            iris_attention_mix,
        )

        return iris_attention_mix(
            partial,
            prefix,
            block_residual,
            res_weight,
            rms_weight,
            eps=eps,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            num_valid_blocks=num_valid_blocks,
            group=_get_process_group(self.mapping.attn.tp_group),
        )

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
        producer_direct: bool,
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
            armed=self.cute_ar is not None,
            has_prefix=prefix_sum is not None,
            num_tokens=num_tokens,
            fusion_max_tokens=global_server_args_dict["comm_fusion_max_num_tokens"],
        ):
            # Any later reduce in this process overwrites it; this layer is done by then.
            residual_out, _ = self.cute_ar(
                attn_partial,
                prefix_sum,
                self.dummy_norm.weight,
                include_reduce_scatter=False,
                include_routed=True,
            )
            return residual_out, None
        if (
            prefix_sum is not None
            and self.attn_ar_fusion_ok
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
            _, residual_out, *_ = self.dummy_norm.forward_with_allreduce_fusion(
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
        residual, delta = self.reduce_for_attnres(
            attn_partial, prefix_sum, producer_direct=producer_direct
        )
        return (residual if delta is None else residual + delta), None


class K3MoeTailComm:
    """Combine routed and shared expert outputs over 4, 8, or 16 MoE ranks.

    Stage 1 -- routed_ar_fusion:
        Finalize deferred routed output, all-reduce, and apply RMSNorm.
        Runs on the routed branch before the stream join.

    Stage 2 -- shared_rs + up_proj_ag, for 1..32 tokens:
        Reduce-scatter the shared output in the shared-expert branch. After
        the join, up-project the routed latent, add the reduced shared shard,
        and all-gather the output.

    Stage 2 -- up_proj_inject_ar, for 33 or more tokens:
        After the join, accumulate this rank's up-projection block into the
        full shared-expert partial, then all-reduce the combined output.

    Both stage-2 patterns add the attention residual exactly once. Token count
    selects the pattern; the model controls stream scheduling independently.
    Workspaces are prepared once and reused by sequential layers, while each
    layer supplies its own normalization and projection weights.
    """

    _routed_workspace: ClassVar[AllReduceFusionWorkspace | None] = None
    _latent_tail: ClassVar[KimiK3LatentTailOp | None] = None
    _stage2_capacity: ClassVar[int | None] = None
    _multimem_group_name: ClassVar[str | None] = None
    _workspace_config: ClassVar[tuple[tuple[int, ...], int, int, float] | None] = None

    def __init__(
        self,
        *,
        mapping,
        hidden_size: int,
        routed_hidden: int,
        top_k: int,
        routed_norm,
        up_proj,
        experts_supports_deferred_finalize: bool,
    ) -> None:
        self.mapping = mapping
        self.hidden_size = hidden_size
        self.routed_hidden = routed_hidden
        self.top_k = top_k
        self.routed_norm = routed_norm
        self.up_proj = up_proj
        self.use_allreduce_fusion = _unified_tail_applicable(
            mapping=mapping,
            hidden_size=hidden_size,
            latent_size=routed_hidden,
            top_k=top_k,
            is_blackwell=current_platform().is_blackwell,
            has_routed_norm=routed_norm is not None,
        )
        self.defer_finalize = (
            self.use_allreduce_fusion and experts_supports_deferred_finalize
        )
        self.rms_eps = (
            float(routed_norm.variance_epsilon) if routed_norm is not None else 1e-5
        )
        if self.use_allreduce_fusion:
            configuration = (
                tuple(mapping.moe.tp_ep_group),
                routed_hidden,
                top_k,
                self.rms_eps,
            )
            if K3MoeTailComm._workspace_config is None:
                K3MoeTailComm._workspace_config = configuration
            elif K3MoeTailComm._workspace_config != configuration:
                raise ValueError(
                    "K3 routed workspace geometry differs from its initial configuration"
                )

    def prepare(self, max_num_tokens: int) -> bool:
        """Prepare both tail stages once, before KV sizing and graph capture."""
        if self._stage2_capacity is not None:
            if max_num_tokens > self._stage2_capacity:
                raise RuntimeError("Cannot grow a prepared K3 MoE workspace")
            return True
        group = _get_process_group(self.mapping.moe.tp_ep_group)
        if self.use_allreduce_fusion:
            K3MoeTailComm._routed_workspace = create_allreduce_fusion_workspace(
                group=group,
                hidden_size=self.routed_hidden,
                top_k=self.top_k,
                max_num_tokens=max_num_tokens,
                rms_eps=self.rms_eps,
            )
        K3MoeTailComm._latent_tail = KimiK3LatentTailOp(
            group=group, hidden_size=self.hidden_size, latent_size=self.routed_hidden
        )
        if max_num_tokens > _MULTIMEM_AR_MIN_TOKENS:
            if not multimem_prealloc(
                max_num_tokens, (self.hidden_size,), group.group_name
            ):
                raise RuntimeError("K3 MoE tail requires Multimem all-reduce")
            K3MoeTailComm._multimem_group_name = group.group_name
        K3MoeTailComm._stage2_capacity = max_num_tokens
        logger.info(f"K3 MoE tail workspaces prepared through M={max_num_tokens}")
        return True

    def shared_rs(self, shared_partial: torch.Tensor) -> torch.Tensor:
        """Reduce-scatter the shared-expert partial in the shared branch.

        Args:
            shared_partial: Rank-local BF16 [M,H], with M in [1,32].

        Returns:
            Padded [64,H/TP] reduced shard, with the first M rows live.
        """
        tail = self._latent_tail
        if tail is None:
            raise RuntimeError("K3 shared RS must be prepared before forward")
        return tail.reduce_scatter_shared(shared_partial)

    def routed_ar_fusion(
        self,
        routed_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor:
        """Finalize, all-reduce and normalize routed output before the stream join.

        ``routed_out`` is deferred expert rows with weights and indices when
        ``defer_finalize`` is true, otherwise a finalized rank-local tensor.
        Returns replicated [M, latent] output, normalized when configured.
        """
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError("K3 tail expects a positive token count")
        if self.use_allreduce_fusion:
            workspace = self._routed_workspace
            if workspace is None:
                raise RuntimeError(
                    "K3 routed workspace must be prepared before forward"
                )
            if not workspace.supports_num_tokens(num_tokens):
                raise ValueError("K3 token count exceeds the prepared routed workspace")
            if self.defer_finalize:
                routed_input, weights, indices = routed_out
                pattern = AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
            else:
                routed_input, weights, indices = routed_out, None, None
                pattern = AllReduceFusionPattern.ALLREDUCE_RMSNORM
            return allreduce_fusion(
                routed_input,
                workspace,
                pattern=pattern,
                rms_gamma=self.routed_norm.weight,
                num_tokens=num_tokens,
                expert_weights=weights,
                expanded_idx_to_permuted_idx=indices,
            )
        if self.mapping.moe.has_tp_ep:
            routed_out = all_reduce(routed_out, self.mapping.moe.tp_ep_group)
        if self.routed_norm is not None:
            routed_out = self.routed_norm(routed_out)
        return routed_out

    def up_proj_ag(
        self,
        routed_latent: torch.Tensor,
        shared_shard: torch.Tensor,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Up-project, add the reduced shared shard, and gather after the join.

        Args:
            routed_latent: Replicated normalized BF16 [M,L], with M in [1,32].
            shared_shard: Padded rank-local output returned by shared_rs.
            prefix_sum: Replicated BF16 [M,H] attention residual.

        Returns:
            BF16 [M,H] combined output on every rank. Projection uses SIMT
            for M1..5 and TensorCore for M6..32.
        """
        capacity = self._stage2_capacity
        if capacity is None:
            raise RuntimeError("K3 MoE tail must be prepared before forward")
        if routed_latent.shape[0] > capacity:
            raise ValueError("K3 token count exceeds the prepared MoE workspace")
        return self._latent_tail.project_and_gather(
            routed_latent, self.up_proj.weight, shared_shard, prefix_sum
        )

    def up_proj_inject_ar(
        self,
        routed_latent: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Inject the projection block and residual, then all-reduce after the join.

        Args:
            routed_latent: Replicated normalized BF16 [M,L].
            shared_partial: Unreduced rank-local BF16 [M,H] shared-expert output.
            prefix_sum: Replicated BF16 [M,H] attention residual.

        Returns:
            BF16 [M,H] combined output on every rank. M33..1024 uses ordinary
            AR; M>1024 uses Multimem AR.
        """
        num_tokens = routed_latent.shape[0]
        capacity = self._stage2_capacity
        if capacity is None:
            raise RuntimeError("K3 MoE tail must be prepared before forward")
        if num_tokens > capacity:
            raise ValueError("K3 token count exceeds the prepared MoE workspace")
        shared_partial = shared_partial.view(num_tokens, self.hidden_size)
        if num_tokens > _MULTIMEM_AR_MIN_TOKENS:
            shared_partial = multimem_stage(
                shared_partial, self._multimem_group_name, capacity
            )
            if shared_partial is None:
                raise RuntimeError("K3 Multimem staging was not prepared")
        start, width = self.up_proj.shard_slice
        target = shared_partial[:, start : start + width]
        target += prefix_sum.view(num_tokens, self.hidden_size)[
            :, start : start + width
        ]
        target.addmm_(routed_latent, self.up_proj.weight.t())
        if num_tokens > _MULTIMEM_AR_MIN_TOKENS:
            # The next layer reuses the symmetric staging buffer.
            return multimem_all_reduce_staged(
                shared_partial, self._multimem_group_name
            ).clone()
        return all_reduce(shared_partial, self.mapping.moe.tp_ep_group)


__all__ = [
    "K3_SHARED_RS_MAX_TOKENS",
    "K3AttnComm",
    "K3MoeTailComm",
]
