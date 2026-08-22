"""CUDA MoE kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cuda.routing import (
    hash_softplus_sqrt_topk_flash,
    softplus_sqrt_topk_flash,
)

try:
    from tokenspeed_kernel.thirdparty.cuda.moe import moe_finalize_fuse_shared
except ImportError:
    moe_finalize_fuse_shared = error_fn


@register_kernel(
    "moe",
    "deepseek_v4_select_experts",
    name="cuda_deepseek_v4_select_experts",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=frozenset(
        format_signature(router_logits=dense_tensor_format(dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ),
    traits={
        "experts": frozenset({256, 384}),
        "top_k": frozenset({6}),
        "renormalize": frozenset({True}),
        "routing_kind": frozenset({"bias", "hash"}),
    },
    priority=Priority.SPECIALIZED,
    tags={"nvidia", "routing", "latency"},
)
def cuda_deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused CUDA sqrt-softplus expert selector."""
    logits_f32 = router_logits.float().contiguous()
    topk_weights = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=torch.int32,
        device=router_logits.device,
    )
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        hash_softplus_sqrt_topk_flash(
            logits_f32,
            input_ids.reshape(-1).to(device=router_logits.device).contiguous(),
            hash_indices_table.to(
                device=router_logits.device,
                dtype=torch.int32,
            ).contiguous(),
            topk_ids,
            topk_weights,
            1.0,
            renormalize,
        )
    elif correction_bias is not None:
        softplus_sqrt_topk_flash(
            logits_f32,
            correction_bias.to(
                device=router_logits.device,
                dtype=torch.float32,
            ).contiguous(),
            topk_ids,
            topk_weights,
            1.0,
            renormalize,
        )
    else:
        raise ValueError("fused DeepSeek V4 selection requires bias or hash routing")
    scores = (
        torch.sqrt(F.softplus(router_logits.float())) if need_scores else router_logits
    )
    return topk_weights, topk_ids, scores


__all__ = ["cuda_deepseek_v4_select_experts", "moe_finalize_fuse_shared"]
