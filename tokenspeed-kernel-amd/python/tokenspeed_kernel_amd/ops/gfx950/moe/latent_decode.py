# Copyright (c) 2026 LightSeek Foundation

"""Registration for gfx950 joint latent/shared MoE decode."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.latent_shared_decode import (
        gluon_latent_expert_shared_decode_gfx950 as _joint_decode_impl,
    )

    @register_kernel(
        "moe",
        "latent_expert_shared",
        name="gluon_latent_expert_shared_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                w13_weight=dense_tensor_format(torch.uint8),
                w13_scale=dense_tensor_format(torch.uint8),
                w2_weight=dense_tensor_format(torch.uint8),
                w2_scale=dense_tensor_format(torch.uint8),
                topk_weights=dense_tensor_format(topk_weight_dtype),
                topk_ids=dense_tensor_format(torch.int32),
                shared_input=dense_tensor_format(torch.bfloat16),
                shared_weight=dense_tensor_format(torch.bfloat16),
                routed_out=dense_tensor_format(torch.bfloat16),
                shared_out=dense_tensor_format(torch.bfloat16),
            )
            for topk_weight_dtype in (torch.bfloat16, torch.float32)
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "latent_size": frozenset({3584}),
            "topk": frozenset({16}),
            "num_local_experts": frozenset({112}),
            "intermediate_size": frozenset({3072}),
            "shared_size": frozenset({768}),
            "output_size": frozenset({7168}),
            "linear_weights": frozenset({True}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_latent_expert_shared_gfx950(**kwargs):
        return _joint_decode_impl(
            kwargs["hidden_states"],
            kwargs["w13_weight"],
            kwargs["w13_scale"],
            kwargs["w2_weight"],
            kwargs["w2_scale"],
            kwargs["topk_weights"],
            kwargs["topk_ids"],
            situ_beta=kwargs["activation_clamp"],
            situ_linear_beta=kwargs["linear_clamp"],
            expert_start=kwargs["expert_start"],
            w13_interleaved=kwargs["w13_interleaved"],
            shared_input=kwargs["shared_input"],
            shared_weight=kwargs["shared_weight"],
            routed_out=kwargs["routed_out"],
            shared_out=kwargs["shared_out"],
        )
