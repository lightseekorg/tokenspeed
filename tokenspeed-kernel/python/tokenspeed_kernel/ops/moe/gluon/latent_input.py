# Copyright (c) 2026 LightSeek Foundation

"""Gluon registrations for latent-MoE input projections."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.moe.latent_input import packed_projection_weight_view
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_decode import (
        gluon_latent_input_decode_gfx950 as _decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_small_batch import (
        gluon_latent_input_small_batch_gfx950 as _small_batch_impl,
    )

    _SIGNATURES = frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                router_weight=dense_tensor_format(torch.bfloat16),
                routed_weight=dense_tensor_format(torch.bfloat16),
                shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
            )
        }
    )
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_decode_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "hidden_size": frozenset({7168}),
            "num_experts": frozenset({896}),
            "latent_size": frozenset({3584}),
            "shared_size": frozenset({768}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_latent_input_decode_gfx950(**kwargs):
        return _decode_impl(
            kwargs["hidden_states"],
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "moe",
        "latent_input",
        name="gluon_latent_input_small_batch_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        # Below the single-token specialist and above the portable Triton
        # kernel. Past 128 tokens this schedule loses to the portable kernel.
        priority=Priority.SPECIALIZED - 1,
        traits={
            "weights_packed": frozenset({True}),
            "hidden_size_multiple_64": frozenset({True}),
            "inputs_contiguous": frozenset({True}),
            "tokens": frozenset(range(1, 129)),
        },
    )
    def gluon_latent_input_small_batch_gfx950(**kwargs):
        weights = (
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
        )
        return _small_batch_impl(
            kwargs["hidden_states"],
            *weights,
            packed_projection_weight_view(*weights),
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )
