"""CUDA FP16 and BF16 GEMM kernels."""

from tokenspeed_kernel.registry import error_fn

try:
    from .dsv3_router import dsv3_router_gemm
except ImportError:
    dsv3_router_gemm = error_fn

from .lm_head import lm_head_gemm, should_use_fused

__all__ = ["dsv3_router_gemm", "lm_head_gemm", "should_use_fused"]
