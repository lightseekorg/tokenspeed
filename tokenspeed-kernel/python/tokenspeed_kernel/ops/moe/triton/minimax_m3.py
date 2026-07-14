# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Registered Triton routing kernel for MiniMax-M3."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.triton.minimax_m3 import (
    minimax_m3_topk as _triton_minimax_m3_topk,
)

_ROUTING_SIGNATURE = format_signature(
    hidden_states=dense_tensor_format(torch.bfloat16),
    gating_output=dense_tensor_format(torch.float32),
    correction_bias=dense_tensor_format(torch.float32),
)


@register_kernel(
    "moe",
    "minimax_m3_topk",
    name="triton_minimax_m3_topk",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=frozenset({_ROUTING_SIGNATURE}),
    traits={
        "topk": frozenset({4}),
        "renormalize": frozenset({True}),
    },
    priority=Priority.PERFORMANT,
    tags={"determinism", "fail_closed"},
)
def triton_minimax_m3_topk(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    return _triton_minimax_m3_topk(*args, **kwargs)


__all__ = ["triton_minimax_m3_topk"]
