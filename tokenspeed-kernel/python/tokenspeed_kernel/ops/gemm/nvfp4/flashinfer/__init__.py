"""FlashInfer NVFP4 GEMM implementation."""

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format

_NVFP4_SCALE_DTYPES = frozenset({torch.float32, torch.uint8, torch.float8_e4m3fn})
_NVFP4_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=a_scale_dtype,
                granularity="block",
                block_shape=(16,),
            ),
        ),
        b=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=b_scale_dtype,
                granularity="block",
                block_shape=(16,),
            ),
        ),
    )
    for storage_dtype in {torch.uint8, torch.float4_e2m1fn_x2}
    for a_scale_dtype in _NVFP4_SCALE_DTYPES
    for b_scale_dtype in _NVFP4_SCALE_DTYPES
)

platform = current_platform()
# ---- FlashInfer FP4 -----------------------------------------------------

mm_fp4 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    from flashinfer import mm_fp4

if mm_fp4 is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_nvfp4",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_NVFP4_FORMAT_SIGNATURES,
        traits={},
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_mm_nvfp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        enable_pdl: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # backend="cutlass" (not "auto") to skip flashinfer's cuDNN-graph plan compile.
        output = mm_fp4(
            A,
            B,
            A_scales,
            B_scales,
            alpha,
            out_dtype,
            backend="cutlass",
            enable_pdl=enable_pdl,
        )
        if out is not None:
            out.copy_(output)
            return out
        return output
