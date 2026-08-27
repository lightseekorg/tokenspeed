# Copyright (c) 2026 LightSeek Foundation

"""Registration shim for gfx950 Gluon sigmoid top-k routing."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_gluon,
        invoke_sigmoid_bias_topk_route_prefill_gluon,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc
    invoke_sigmoid_bias_topk_route_gluon = None
    invoke_sigmoid_bias_topk_route_prefill_gluon = None
else:
    _IMPORT_ERROR = None


if invoke_sigmoid_bias_topk_route_prefill_gluon is not None:

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

else:

    def gluon_sigmoid_bias_topk_gfx950(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise ImportError(
            "gluon_sigmoid_bias_topk_gfx950 requires tokenspeed-kernel-amd"
        ) from _IMPORT_ERROR


__all__ = ["gluon_sigmoid_bias_topk_gfx950"]
