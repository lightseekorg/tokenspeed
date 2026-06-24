"""Communication fusion-compatible operators for Ascend NPU."""

from __future__ import annotations

import torch
import torch.distributed as dist

from tokenspeed_kernel_ascend.ops.layernorm.rmsnorm import rmsnorm

__all__ = [
    "allgather_dual_rmsnorm",
    "allreduce_residual_rmsnorm",
    "reducescatter_residual_rmsnorm",
]


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
    launch_with_pdl: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
    """All-reduce followed by residual add and RMSNorm on Ascend."""
    _ = (
        rank,
        max_token_num,
        use_oneshot,
        trigger_completion_at_end,
        fp32_acc,
        max_sm_to_use,
        launch_with_pdl,
    )

    if residual is None:
        raise ValueError("Ascend allreduce_residual_rmsnorm requires residual")

    if block_quant_fp8 or residual_reduce_scattered or has_partial_norm_out:
        raise NotImplementedError(
            "Ascend allreduce_residual_rmsnorm does not support "
            "block_quant_fp8, residual_reduce_scattered, or has_partial_norm_out"
        )

    reduced = input_tensor
    if group.size() > 1:
        reduced = input_tensor.clone()
        dist.all_reduce(reduced, group=group)

    norm_out, residual_out = rmsnorm(
        reduced,
        weight,
        eps,
        residual=residual,
    )
    return norm_out, residual_out, None, None


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
    launch_with_pdl: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Reduce-scatter followed by residual add and RMSNorm on Ascend."""
    _ = (
        rank,
        max_token_num,
        use_oneshot,
        trigger_completion_at_end,
        fp32_acc,
        launch_with_pdl,
    )

    if residual is None:
        raise ValueError("Ascend reducescatter_residual_rmsnorm requires residual")
    if block_quant_fp8:
        raise NotImplementedError(
            "Ascend reducescatter_residual_rmsnorm does not support block_quant_fp8"
        )

    world_size = group.size()
    if world_size == 1:
        reduced = input_tensor
    else:
        if input_tensor.shape[0] % world_size != 0:
            raise ValueError(
                "Ascend reducescatter_residual_rmsnorm requires input tokens "
                f"({input_tensor.shape[0]}) to be divisible by world_size "
                f"({world_size})"
            )
        reduced = torch.empty(
            (input_tensor.shape[0] // world_size, *input_tensor.shape[1:]),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        dist.reduce_scatter_tensor(reduced, input_tensor.contiguous(), group=group)

    if add_in is not None:
        reduced = reduced + add_in

    norm_out, residual_out = rmsnorm(reduced, weight, eps, residual=residual)
    return norm_out, residual_out, None


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
    launch_with_pdl: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """All-gather followed by RMSNorm on q and kv slices on Ascend."""
    _ = (
        rank,
        max_token_num,
        trigger_completion_at_end,
        fp32_acc,
        launch_with_pdl,
    )

    if block_quant_fp8:
        raise NotImplementedError(
            "Ascend allgather_dual_rmsnorm does not support block_quant_fp8"
        )

    world_size = group.size()
    if qkv.shape[0] * world_size != total_num_tokens:
        raise ValueError(
            "Ascend allgather_dual_rmsnorm requires "
            f"qkv.shape[0] * world_size ({qkv.shape[0]} * {world_size}) "
            f"to equal total_num_tokens ({total_num_tokens})"
        )

    allgather_out = torch.empty(
        (total_num_tokens, *qkv.shape[1:]),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    if world_size == 1:
        allgather_out.copy_(qkv)
    else:
        dist.all_gather_into_tensor(allgather_out, qkv.contiguous(), group=group)

    q_lora_rank = weight_q_a.shape[0]
    kv_lora_rank = weight_kv_a.shape[0]
    q = allgather_out[..., :q_lora_rank]
    k_nope = allgather_out[..., q_lora_rank : q_lora_rank + kv_lora_rank]

    q_contiguous = torch.empty_like(q)
    rmsnorm(q.contiguous(), weight_q_a, eps_q, out=q_contiguous)
    rmsnorm(k_nope.contiguous(), weight_kv_a, eps_kv, out=k_nope)
    return allgather_out, q_contiguous, k_nope, None
