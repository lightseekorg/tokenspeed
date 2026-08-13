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

import os
import sys
from functools import lru_cache

from tokenspeed_kernel.platform import current_platform

# Diagnostic bisect: comma-separated substrings; a pdl_enabled() call whose
# CALLER's filename contains any of them reports False while everyone else
# keeps PDL. Lets an accuracy A/B disable one component's PDL at a time
# without touching every call site. Inert (and overhead-free) when unset.
_PDL_OFF_MODULES = [
    m for m in os.environ.get("TOKENSPEED_PDL_OFF_MODULES", "").split(",") if m
]


@lru_cache(maxsize=1)
def _pdl_platform_enabled() -> bool:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    if global_server_args_dict.get("disable_pdl", False):
        return False
    try:
        return current_platform().is_hopper_plus
    except Exception:
        return False


_CENSUS_PATH = os.environ.get("TOKENSPEED_PDL_CALLER_CENSUS")
_census_seen: set = set()


def pdl_enabled() -> bool:
    """Return whether Programmatic Dependent Launch is enabled."""
    if _PDL_OFF_MODULES or _CENSUS_PATH:
        caller = sys._getframe(1).f_code.co_filename
        if _CENSUS_PATH and caller not in _census_seen:
            # Census mode: record every distinct caller file once. The
            # caller-keyword gating is only sound if this set contains no
            # wrappers (e.g. torch custom-op trampolines) standing in for
            # the real consumers.
            _census_seen.add(caller)
            with open(_CENSUS_PATH, "a") as f:
                f.write(caller + "\n")
        if _PDL_OFF_MODULES and any(m in caller for m in _PDL_OFF_MODULES):
            return False
    return _pdl_platform_enabled()


# Callers reset this after server-arg changes; keep the lru interface.
pdl_enabled.cache_clear = _pdl_platform_enabled.cache_clear  # type: ignore[attr-defined]
