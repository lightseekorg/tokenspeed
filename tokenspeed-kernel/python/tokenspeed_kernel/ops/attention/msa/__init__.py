"""MiniMax sparse attention implementations."""

import tokenspeed_kernel.ops.attention.msa.triton  # noqa: E402,F401
from tokenspeed_kernel.platform import current_platform

__all__ = []

if current_platform().is_nvidia and current_platform().is_blackwell:
    try:
        from tokenspeed_kernel.ops.attention.msa.cuda.runtime import *  # noqa: F403
        from tokenspeed_kernel.ops.attention.msa.cuda.runtime import __all__
        from tokenspeed_kernel.ops.attention.msa.cute_dsl.attention import *  # noqa: F403
    except ImportError:
        pass
