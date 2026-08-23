"""CUDA GEMM kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

try:
    from tokenspeed_kernel.thirdparty.cuda.dsv3_gemm import dsv3_router_gemm
except ImportError:
    dsv3_router_gemm = error_fn


@register_kernel(
    "gemm",
    "dsv4_linear_fp32",
    name="cuda_dsv3_dsv4_linear_fp32",
    solution="cuda",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 9),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        format_signature(
            hidden_states=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(weight_dtype),
        )
        for weight_dtype in (torch.bfloat16, torch.float32)
    ),
    traits={
        "hidden_rank": frozenset({2}),
        "weight_rank": frozenset({2}),
        "has_tokens": frozenset({True}),
        "k_match": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
    tags={"nvidia", "latency"},
)
def cuda_dsv3_dsv4_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run the DSV3 router GEMM specialization."""
    return dsv3_router_gemm(
        hidden_states,
        weight,
        out_dtype=torch.float32,
        enable_pdl=enable_pdl,
    )


__all__ = ["cuda_dsv3_dsv4_linear_fp32", "dsv3_router_gemm"]
