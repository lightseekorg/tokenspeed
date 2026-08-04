"""MiniMax sparse attention implementations."""

import tokenspeed_kernel.ops.attention.msa.triton  # noqa: E402,F401
from tokenspeed_kernel.platform import current_platform

__all__ = []

platform = current_platform()
if platform.is_nvidia and platform.is_blackwell:
    import tokenspeed_kernel.ops.attention.msa.cuda.runtime  # noqa: F401
    import tokenspeed_kernel.ops.attention.msa.cute_dsl.attention  # noqa: F401
