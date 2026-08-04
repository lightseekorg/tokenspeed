"""Platform-selected direct FlashMLA APIs."""

from __future__ import annotations

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

flash_mla_with_kvcache = error_fn
flash_mla_sparse_fwd = error_fn
get_mla_metadata = error_fn

if current_platform().is_nvidia and current_platform().is_hopper_plus:
    try:
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
    except ImportError:
        pass

__all__ = [
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "get_mla_metadata",
]
