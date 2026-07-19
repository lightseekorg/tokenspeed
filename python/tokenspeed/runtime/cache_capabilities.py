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

"""Import-light cache capability validation shared by runtime entry points."""

from __future__ import annotations

from typing import Any

_ACTIVE_KV_PD_MODES = frozenset(("prefill", "decode"))


def _validate_pools(
    target_pool: Any,
    draft_pool: Any | None,
    *,
    require_pd_transfer: bool,
    require_host_tier: bool,
) -> None:
    for role, pool in (("target", target_pool), ("draft", draft_pool)):
        if pool is None:
            continue
        if require_pd_transfer and not getattr(pool, "supports_pd_transfer", True):
            raise RuntimeError(
                "KV cache pool does not support active PD transfer; "
                f"pool_role={role}"
            )
        if require_host_tier and not getattr(
            pool, "supports_hierarchical_kv_cache", True
        ):
            raise RuntimeError(
                "KV cache pool does not support the host/L2 cache tier; "
                f"pool_role={role}"
            )


def validate_pd_transfer_pools(target_pool: Any, draft_pool: Any | None) -> None:
    """Validate every pool before a PD caller reads device buffer pointers."""
    _validate_pools(
        target_pool,
        draft_pool,
        require_pd_transfer=True,
        require_host_tier=False,
    )


def validate_cache_runtime_capabilities(
    *,
    target_pool: Any,
    draft_pool: Any | None,
    disaggregation_mode: str,
    enable_kvstore: bool,
) -> None:
    """Fail before runtime wiring when a cache lacks a requested feature."""
    _validate_pools(
        target_pool,
        draft_pool,
        require_pd_transfer=disaggregation_mode in _ACTIVE_KV_PD_MODES,
        require_host_tier=enable_kvstore,
    )


__all__ = [
    "validate_cache_runtime_capabilities",
    "validate_pd_transfer_pools",
]
