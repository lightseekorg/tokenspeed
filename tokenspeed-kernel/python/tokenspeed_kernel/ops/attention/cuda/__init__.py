from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()

if platform.is_nvidia and platform.is_hopper_plus:
    from tokenspeed_kernel.thirdparty.cuda.merge_state import merge_state

    @register_kernel(
        "attention",
        "attn_merge_state",
        name="cuda_attn_merge_state",
        solution="cuda",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("out_a", "out_b"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED + 2,
        traits={},
        tags={"throughput"},
    )
    def cuda_attn_merge_state(
        out_a: torch.Tensor,
        lse_a: torch.Tensor,
        out_b: torch.Tensor,
        lse_b: torch.Tensor,
        lse_scale_log2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return merge_state(
            out_a,
            lse_a,
            out_b,
            lse_b,
            lse_scale_log2=lse_scale_log2,
        )
