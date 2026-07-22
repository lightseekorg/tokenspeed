"""CUDA GEMM kernels."""

from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_kernel.thirdparty.cuda.dsv3_gemm import dsv3_router_gemm
except ImportError:
    dsv3_router_gemm = error_fn

__all__ = ["dsv3_router_gemm"]
