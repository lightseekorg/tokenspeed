# Copyright (c) 2026 LightSeek Foundation

"""Registration shims for AMD Gluon sigmoid top-k routing."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_gluon,
        invoke_sigmoid_bias_topk_route_prefill_gluon,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_prefill_gluon as invoke_sigmoid_bias_topk_route_prefill_gfx1250,
    )

    @register_kernel(
        "moe",
        "sigmoid_bias_topk",
        name="gluon_sigmoid_bias_topk_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "router_logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
        tags={"prefill", "routing"},
    )
    def gluon_sigmoid_bias_topk_gfx950(
        *,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        routed_scaling_factor: float,
        normalize_topk_weights: bool,
        weights_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decode_supported = router_logits.shape[0] * topk <= 128
        route = (
            invoke_sigmoid_bias_topk_route_gluon
            if decode_supported
            and (
                router_logits.shape[0] == 1
                or router_logits.dtype != torch.float32
                or topk > 16
            )
            else invoke_sigmoid_bias_topk_route_prefill_gluon
        )
        topk_ids, topk_weights = route(
            router_logits,
            correction_bias,
            topk,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
        )
        # The gluon route has no output-dtype argument to store through.
        return topk_weights.to(weights_dtype), topk_ids

    @register_kernel(
        "moe",
        "sigmoid_bias_topk",
        name="gluon_sigmoid_bias_topk_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures("router_logits", "dense", {torch.float32}),
        priority=Priority.SPECIALIZED,
        tags={"prefill", "routing", "gfx1250"},
        traits={
            "tokens": range(2, 1 << 31),
            "experts": frozenset({896}),
            "topk": frozenset({16}),
        },
    )
    def gluon_sigmoid_bias_topk_gfx1250(
        *,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        routed_scaling_factor: float,
        normalize_topk_weights: bool,
        weights_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_ids, topk_weights = invoke_sigmoid_bias_topk_route_prefill_gfx1250(
            router_logits,
            correction_bias,
            topk,
            routed_scaling_factor=routed_scaling_factor,
            normalize_topk_weights=normalize_topk_weights,
        )
        return topk_weights.to(weights_dtype), topk_ids

    __all__ = [
        "gluon_sigmoid_bias_topk_gfx1250",
        "gluon_sigmoid_bias_topk_gfx950",
    ]
else:
    __all__ = []
