"""Public communication-kernel interfaces."""

from __future__ import annotations

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.trtllm import (
    allgather_dual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_lane_latent_norm as _allreduce_lane_latent_norm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_residual_rmsnorm,
    reducescatter_residual_rmsnorm,
)
from tokenspeed_kernel.platform import current_platform

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
    launch_with_pdl: bool = False,
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
        launch_with_pdl=launch_with_pdl,
        trigger_completion_at_end=trigger_completion_at_end,
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
