"""FlashInfer FP16 and BF16 GEMM implementations."""

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

tinygemm_bf16 = error_fn

if current_platform().is_hopper_plus:
    from flashinfer.gemm import tinygemm_bf16

__all__ = ["tinygemm_bf16"]
