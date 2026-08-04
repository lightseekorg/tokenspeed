"""CUDA FP16 and BF16 GEMM kernels."""

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

dsv3_router_gemm = error_fn
lm_head_gemm = error_fn
should_use_fused = error_fn

platform = current_platform()
if platform.is_nvidia:
    from .lm_head import lm_head_gemm, should_use_fused

if platform.is_hopper_plus:
    from .dsv3_router import dsv3_router_gemm

__all__ = ["dsv3_router_gemm", "lm_head_gemm", "should_use_fused"]
