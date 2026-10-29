"""CUDA MoE kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.cuda.kimi3_router import sigmoid_bias_topk
from tokenspeed_kernel.thirdparty.cuda.moe import moe_finalize_fuse_shared
from tokenspeed_kernel.thirdparty.cuda.routing import (
    hash_softplus_sqrt_topk_flash,
    routing_available,
    softplus_sqrt_topk_flash,
)


def cuda_sqrt_softplus_topk(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    need_scores: bool,
    routed_scaling_factor: float,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused CUDA sqrt-softplus expert selector."""
    logits_f32 = router_logits.float().contiguous()
    topk_weights = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=weights_dtype,
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
            routed_scaling_factor,
            renormalize,
            enable_pdl=pdl_enabled(),
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
            routed_scaling_factor,
            renormalize,
            enable_pdl=pdl_enabled(),
        )
    else:
        raise ValueError("fused DeepSeek V4 selection requires bias or hash routing")
    scores = (
        torch.sqrt(F.softplus(router_logits.float())) if need_scores else router_logits
    )
    return topk_weights, topk_ids, scores


if routing_available():
    cuda_sqrt_softplus_topk = register_kernel(
        "moe",
        "topk",
        name="cuda_sqrt_softplus_topk",
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
            "score_function": frozenset({"sqrt_softplus"}),
        },
        priority=Priority.SPECIALIZED,
    )(cuda_sqrt_softplus_topk)


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="cuda_kimi3_sigmoid_bias_topk",
    solution="cuda",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {format_signature(router_logits=dense_tensor_format(torch.float32))}
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "tokens_min": frozenset({2560}),
        "experts": frozenset({896}),
        "topk": frozenset({16}),
        "bias_dtype": frozenset({torch.float32}),
        "inputs_contiguous": frozenset({True}),
        "normalize_topk_weights": frozenset({True}),
        "routed_scaling_factor": frozenset({1.0}),
        "weights_dtype": frozenset({torch.float32}),
        "ids_dtype": frozenset({torch.int32}),
        "supports_output_buffers": frozenset({True}),
    },
)
def cuda_kimi3_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 normalized top-16 weights and INT32 IDs; see sigmoid_bias_topk."""
    if (
        topk != 16
        or routed_scaling_factor != 1.0
        or not normalize_topk_weights
        or (out[0].dtype if out is not None else weights_dtype) != torch.float32
        or (out is not None and out[1].dtype != torch.int32)
    ):
        raise ValueError(
            "Kimi-K3 CUDA routing requires normalized top-16 weights "
            "with scale 1, FP32 weights, and INT32 ids"
        )
    return sigmoid_bias_topk(router_logits, correction_bias, out=out)


__all__ = [
    "cuda_sqrt_softplus_topk",
    "cuda_kimi3_sigmoid_bias_topk",
    "moe_finalize_fuse_shared",
]
