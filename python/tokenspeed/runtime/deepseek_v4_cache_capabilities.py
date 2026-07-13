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

"""Fail-closed capability checks for DeepSeek V4 cache management.

This module deliberately has no runtime, torch, or scheduler imports.  The
primary check runs before distributed initialization; the pool check protects
alternate PD entry points before they enumerate device pointers.
"""

from __future__ import annotations

from typing import Any

V4_ACTIVE_PD_UNSUPPORTED = (
    "DeepSeek V4 KV cache does not support active PD transfer; "
    "group-aware PD transfer is required."
)
V4_FLAT_HOST_TIER_UNSUPPORTED = (
    "DeepSeek V4 flat KV cache is device-side L1 only; host/L2 KV cache, "
    "kvstore, and CPU cache copy are unsupported."
)
PD_POOL_UNSUPPORTED = (
    "KV cache pool does not support PD transfer; group-aware PD transfer "
    "is required."
)

_ACTIVE_KV_PD_MODES = frozenset(("prefill", "decode"))


def _v4_roles(*, target_is_v4: bool, draft_is_v4: bool) -> str:
    roles: list[str] = []
    if target_is_v4:
        roles.append("target")
    if draft_is_v4:
        roles.append("draft")
    return ",".join(roles)


def validate_deepseek_v4_cache_capabilities(
    *,
    target_is_v4: bool,
    draft_is_v4: bool,
    disaggregation_mode: str,
    flat_kvcache_ext: bool,
    enable_kvstore: bool,
) -> None:
    """Reject unsupported V4 cache combinations before runtime resources exist."""

    roles = _v4_roles(target_is_v4=target_is_v4, draft_is_v4=draft_is_v4)
    if not roles:
        return

    if disaggregation_mode in _ACTIVE_KV_PD_MODES:
        raise RuntimeError(f"{V4_ACTIVE_PD_UNSUPPORTED} model_role={roles}")

    if flat_kvcache_ext and enable_kvstore:
        raise RuntimeError(f"{V4_FLAT_HOST_TIER_UNSUPPORTED} model_role={roles}")


def validate_pd_transfer_pools(target_pool: Any, draft_pool: Any | None) -> None:
    """Validate every pool before the caller reads any device buffer pointer."""

    for role, pool in (("target", target_pool), ("draft", draft_pool)):
        if pool is not None and not getattr(pool, "supports_pd_transfer", True):
            raise RuntimeError(f"{PD_POOL_UNSUPPORTED} pool_role={role}")
