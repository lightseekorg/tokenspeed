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

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)


def cache_debug_enabled() -> bool:
    """Whether expensive, GPU-synchronizing cache validation is enabled."""
    return os.environ.get("TOKENSPEED_CACHE_DEBUG") == "1"


def require_positive_int(name: str, value: object) -> int:
    """Validate that ``value`` is a positive, non-boolean integer.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Returns:
        ``value`` unchanged, typed as ``int``.

    Raises:
        ValueError: If ``value`` is a bool, not an int, or not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class CacheRuntimeContract:
    prefix_granularity: int
    num_lcm_blocks: int
    token_capacity: int
    group_specs: tuple[CacheGroupSpec, ...]
    # Both projected from the memory plan, which owns physical geometry: how
    # many CacheBlocks each group has, and how many share one LCM parent.
    group_page_counts: Mapping[str, int]
    group_packing: Mapping[str, int]

    def __post_init__(self) -> None:
        prefix_granularity = require_positive_int(
            "prefix_granularity", self.prefix_granularity
        )
        num_lcm_blocks = require_positive_int("num_lcm_blocks", self.num_lcm_blocks)
        token_capacity = require_positive_int("token_capacity", self.token_capacity)
        if not isinstance(self.group_specs, tuple) or not self.group_specs:
            raise ValueError("group_specs must be a non-empty tuple")
        if any(not isinstance(spec, CacheGroupSpec) for spec in self.group_specs):
            raise ValueError("group_specs must contain CacheGroupSpec values")
        group_ids = tuple(spec.group_id for spec in self.group_specs)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("group_specs contain duplicate group IDs")
        counts = dict(self.group_page_counts)
        actual_group_ids = set(counts)
        expected_group_ids = set(group_ids)
        if actual_group_ids != expected_group_ids:
            raise ValueError(
                "group_page_counts keys must match group_specs: "
                f"missing={sorted(expected_group_ids - actual_group_ids)} "
                f"extra={sorted(actual_group_ids - expected_group_ids)}"
            )
        counts = {
            group_id: require_positive_int(
                f"group page count for {group_id!r}", counts[group_id]
            )
            for group_id in group_ids
        }
        packing = dict(self.group_packing)
        if set(packing) != expected_group_ids:
            raise ValueError(
                "group_packing keys must match group_specs: "
                f"missing={sorted(expected_group_ids - set(packing))} "
                f"extra={sorted(set(packing) - expected_group_ids)}"
            )
        packing = {
            group_id: require_positive_int(
                f"cache_blocks_per_lcm_block for {group_id!r}", packing[group_id]
            )
            for group_id in group_ids
        }
        # Both sides come from the plan, so this checks the plan's own
        # arithmetic: every group's blocks are its packing per parent times
        # the parent count, plus the reserved null block.
        expected_counts = {
            group_id: num_lcm_blocks * packing[group_id] + 1 for group_id in group_ids
        }
        if counts != expected_counts:
            raise ValueError(
                "group page counts must equal num_lcm_blocks * "
                "cache_blocks_per_lcm_block + 1: "
                f"expected={expected_counts}, got={counts}"
            )
        max_child_pages = max(counts.values()) - 1
        if token_capacity > max_child_pages * prefix_granularity:
            raise ValueError(
                "token_capacity exceeds the largest group's child-page capacity"
            )
        object.__setattr__(self, "group_page_counts", MappingProxyType(counts))
        object.__setattr__(self, "group_packing", MappingProxyType(packing))
