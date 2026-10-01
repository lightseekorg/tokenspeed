"""Public communication-kernel interfaces."""

from __future__ import annotations

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.allreduce_fusion import (
    AllReduceFusionPattern,
    AllReduceFusionWorkspace,
    allreduce_fusion,
    allreduce_fusion_supported,
    create_allreduce_fusion_workspace,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allgather_dual_rmsnorm as _allgather_dual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_lane_latent_norm as _allreduce_lane_latent_norm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_residual_rmsnorm as _allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    reducescatter_residual_rmsnorm as _reducescatter_residual_rmsnorm,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled

_ALLREDUCE_FUSION_LANE: torch.Tensor | None = None


def allreduce_fusion_lane(
    like: torch.Tensor,
    width: int,
    *,
    enabled: bool = True,
) -> torch.Tensor | None:
    """Return a persistent one-row lane when fused all-reduce can use it.

    Args:
        like: Tensor providing the row count, dtype, and device.
        width: Width of the fused reduction lane.
        enabled: Whether the caller prepared fused all-reduce support.

    Returns:
        A zero-initialized ``[1, width]`` lane, or ``None`` when this invocation
        should use the ordinary reduction path.
    """

    if not enabled or like.ndim != 2 or like.shape[0] != 1:
        return None
    global _ALLREDUCE_FUSION_LANE
    lane = _ALLREDUCE_FUSION_LANE
    if (
        lane is None
        or lane.dtype != like.dtype
        or lane.device != like.device
        or lane.shape != (1, width)
    ):
        lane = torch.zeros(1, width, dtype=like.dtype, device=like.device)
        _ALLREDUCE_FUSION_LANE = lane
    return lane


def allreduce_lane_latent_norm_supported(
    lane: torch.Tensor,
    *,
    enabled: bool = True,
) -> bool:
    """Return whether this invocation can use the fused lane-norm epilogue."""

    return enabled and lane.ndim == 2 and lane.shape[0] == 1


def prepare_allreduce_fusion(
    *,
    rank: int,
    group: dist.ProcessGroup,
    max_token_num: int,
    hidden_dim: int,
    use_fp32_lamport: bool = False,
) -> bool:
    """Prepare the selected fused-all-reduce implementation for graph capture."""

    if not current_platform().is_nvidia:
        return False
    from tokenspeed_kernel.ops.communication.trtllm import (
        ensure_workspace_initialized,
    )

    return bool(
        ensure_workspace_initialized(
            rank=rank,
            group=group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=use_fp32_lamport,
        )
    )


def allreduce_lane_latent_norm(
    lane: torch.Tensor,
    gamma: torch.Tensor,
    latent_width: int,
    *,
    rank: int,
    group: dist.ProcessGroup,
    eps: float,
    max_token_num: int,
    trigger_completion_at_end: bool = False,
) -> torch.Tensor:
    """Reduce a routed/shared lane and normalize its routed prefix."""

    return _allreduce_lane_latent_norm(
        lane,
        gamma,
        latent_width,
        rank=rank,
        group=group,
        eps=eps,
        max_token_num=max_token_num,
        launch_with_pdl=pdl_enabled(),
        trigger_completion_at_end=trigger_completion_at_end,
    )


def allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: bool | None = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    block_quant_fp8: bool = False,
    residual_reduce_scattered: bool = False,
    has_partial_norm_out: bool = False,
    max_sm_to_use: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused all-reduce, residual addition, and RMS normalization."""

    return _allreduce_residual_rmsnorm(
        input_tensor=input_tensor,
        residual=residual,
        weight=weight,
        rank=rank,
        group=group,
        eps=eps,
        max_token_num=max_token_num,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        block_quant_fp8=block_quant_fp8,
        residual_reduce_scattered=residual_reduce_scattered,
        has_partial_norm_out=has_partial_norm_out,
        max_sm_to_use=max_sm_to_use,
        launch_with_pdl=pdl_enabled(),
    )


def reducescatter_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: bool | None = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    block_quant_fp8: bool = False,
    add_in: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run fused reduce-scatter, residual addition, and RMS normalization."""

    return _reducescatter_residual_rmsnorm(
        input_tensor=input_tensor,
        residual=residual,
        weight=weight,
        rank=rank,
        group=group,
        eps=eps,
        max_token_num=max_token_num,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        block_quant_fp8=block_quant_fp8,
        add_in=add_in,
        launch_with_pdl=pdl_enabled(),
    )


def allgather_dual_rmsnorm(
    qkv: torch.Tensor,
    total_num_tokens: int,
    weight_q_a: torch.nn.Parameter,
    weight_kv_a: torch.nn.Parameter,
    rank: int,
    group: dist.ProcessGroup,
    eps_q: float,
    eps_kv: float,
    max_token_num: int,
    block_quant_fp8: bool = False,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run fused all-gather with dual RMS normalization."""

    return _allgather_dual_rmsnorm(
        qkv=qkv,
        total_num_tokens=total_num_tokens,
        weight_q_a=weight_q_a,
        weight_kv_a=weight_kv_a,
        rank=rank,
        group=group,
        eps_q=eps_q,
        eps_kv=eps_kv,
        max_token_num=max_token_num,
        block_quant_fp8=block_quant_fp8,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        launch_with_pdl=pdl_enabled(),
    )


def attention_reduce_mix(
    partial: torch.Tensor,
    residual: torch.Tensor | None,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_norm_eps: float,
    num_valid_blocks: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Reduce attention by row, mix its history, and gather normalized activations.

    Args:
        partial: Prepared projection output, [tokens, hidden].
        residual: Replicated residual, or None to start a new residual block.
        block_residual: Replicated history, [blocks, tokens, hidden].
        res_weight: Attention-residual scorer weight, [hidden].
        rms_weight: Score RMSNorm weight, [hidden].
        eps: Score RMSNorm epsilon.
        out_norm_weight: Output RMSNorm weight, [hidden].
        out_norm_eps: Output RMSNorm epsilon.
        num_valid_blocks: Number of leading history snapshots to mix.
        group: Group owning the prepared projection storage.

    Returns:
        Owned local residual rows and a borrowed replicated activation, or None
        before launch when unsupported. Consume the activation on the calling
        stream before the next attention/MoE fusion reuses its prepared storage.
        All ranks must supply the same shapes and collective order.
    """
    if not current_platform().is_cdna4:
        return None
    from tokenspeed_kernel.ops.communication.iris import iris_attn_mix

    return iris_attn_mix(
        partial,
        residual,
        block_residual,
        res_weight,
        rms_weight,
        eps=eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=out_norm_eps,
        num_valid_blocks=num_valid_blocks,
        group=group,
    )


def moe_reduce_project(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    prefix: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    prefix_is_sharded: bool,
    norm_weight: torch.Tensor | None,
    eps: float | None,
    group: dist.ProcessGroup,
) -> torch.Tensor | None:
    """Reduce MoE partials by row, project routed rows, add the prefix, and gather.

    Args:
        routed_partial: Prepared routed output, [tokens, latent].
        shared_partial: Prepared shared output, [tokens, hidden].
        prefix: Replicated residual or this rank's consecutive residual rows.
        projection_weight: Replicated projection, [hidden, latent].
        prefix_is_sharded: Whether prefix contains only this rank's rows.
        norm_weight: Routed RMSNorm weight, [latent], or None.
        eps: RMSNorm epsilon, or None when norm_weight is None.
        group: Group owning both prepared producer outputs.

    Returns:
        A borrowed replicated result, or None before launch when unsupported.
        Calls sharing prepared storage must run in order on one stream. Consume
        or clone the result before the next attention/MoE fusion overwrites it.
        All ranks must supply the same shapes and collective order.
    """
    if not current_platform().is_cdna4:
        return None
    from tokenspeed_kernel.ops.communication.iris import iris_kimi3_moe_tail

    return iris_kimi3_moe_tail(
        routed_partial,
        shared_partial,
        prefix,
        projection_weight,
        prefix_is_sharded=prefix_is_sharded,
        norm_weight=norm_weight,
        eps=eps,
        group=group,
    )


__all__ = [
    "AllReduceFusionPattern",
    "AllReduceFusionWorkspace",
    "allreduce_fusion",
    "allreduce_fusion_supported",
    "create_allreduce_fusion_workspace",
    "allgather_dual_rmsnorm",
    "allreduce_fusion_lane",
    "allreduce_lane_latent_norm",
    "allreduce_lane_latent_norm_supported",
    "allreduce_residual_rmsnorm",
    "attention_reduce_mix",
    "moe_reduce_project",
    "prepare_allreduce_fusion",
    "reducescatter_residual_rmsnorm",
]
