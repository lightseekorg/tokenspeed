# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Optional FlashMLA dependency, provided by tokenspeed-flashmla."""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=1)
def flash_mla_api() -> ModuleType:
    """Return the optional FlashMLA API, or raise an actionable import error.

    The import is lazy so unrelated kernel families require no FlashMLA.
    Only Python API references are cached; attention schedules are not.
    """
    try:
        import flash_mla.flash_mla_interface as api
    except (ImportError, OSError) as exc:
        raise ImportError(
            "FlashMLA requires the optional tokenspeed-flashmla package "
            "including its native CUDA extension."
        ) from exc
    return api


def is_flash_mla_v41_available() -> bool:
    """Return whether the optional V4.1 Python API and extension import."""
    try:
        flash_mla_api()
        # This API was introduced with the V4.1 packed-cache formats. Older
        # FlashMLA releases can still serve other attention implementations.
        from flash_mla import fused_norm_rope_attn_rope_cast  # noqa: F401
    except (ImportError, OSError):
        return False
    return True
