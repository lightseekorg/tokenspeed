# Copyright (c) 2026 LightSeek Foundation

"""Registration shim for the AMD gfx950 Gluon AttnRes kernel."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

gluon_attn_res_fwd_gfx950 = error_fn

if current_platform().is_cdna4:
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
        attn_res_rmsnorm_gfx950 as _attn_res_rmsnorm_impl,
    )

    @register_kernel(
        "attn_res",
        "fwd",
        name="gluon_attn_res_fwd_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("layer_residual", "block_residual"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "fused_output_norm": frozenset({True}),
            "large_prefill": frozenset({True}),
        },
        tags={"prefill", "fusion"},
    )
    def gluon_attn_res_fwd_gfx950(
        *,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        eps: float,
        out_norm_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt block-major runtime storage to the Gluon token-major kernel."""
        if out_norm_weight is None:
            raise ValueError("Gluon AttnRes forward requires an output RMSNorm")
        return _attn_res_rmsnorm_impl(
            layer_residual=layer_residual,
            block_residual=block_residual.transpose(0, 1),
            res_weight=res_weight,
            score_rms_weight=rms_weight,
            score_eps=eps,
            output_rms_weight=out_norm_weight,
            output_eps=eps,
            num_valid_blocks=block_residual.shape[0],
        )


__all__ = ["gluon_attn_res_fwd_gfx950"]
