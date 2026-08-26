"""Public communication-kernel interfaces."""

from __future__ import annotations

import torch
import torch.distributed as dist
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


__all__ = [
    "allgather_dual_rmsnorm",
    "allreduce_fusion_lane",
    "allreduce_lane_latent_norm",
    "allreduce_lane_latent_norm_supported",
    "allreduce_residual_rmsnorm",
    "prepare_allreduce_fusion",
    "reducescatter_residual_rmsnorm",
]
