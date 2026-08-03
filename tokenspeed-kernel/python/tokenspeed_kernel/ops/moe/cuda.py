"""CUDA MoE kernels."""

from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_kernel.thirdparty.cuda.moe import moe_finalize_fuse_shared
except ImportError:
    moe_finalize_fuse_shared = error_fn

__all__ = ["moe_finalize_fuse_shared"]
