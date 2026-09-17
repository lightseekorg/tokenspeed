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

"""The recipes' side of the scheduler contract.

Two things cross here. A :class:`CacheGroupSpec` becomes the
``CacheGroupConfig`` the C++ scheduler reads -- the one conversion, whether
the pool is being sized or has been built. And the scheduler's
``CapacityModel`` answers how many pages the configured concurrency needs,
so a recipe sizes its pool with the same per-request working-set model the
Scheduler later bounds requests with. Neither side restates the other's
formula.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tokenspeed_scheduler import (
    CacheGroupConfig,
    CacheGroupFamily,
    CacheRetention,
    CacheTransferPolicy,
    CapacityModel,
    SchedulerConfig,
)

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

# Spec vocabulary -> scheduler enum.
_RETENTION_MAP = {
    "full_history": CacheRetention.FullHistory,
    "sliding_window": CacheRetention.SlidingWindow,
}
_FAMILY_MAP = {
    "history": CacheGroupFamily.History,
    "state": CacheGroupFamily.State,
}
_TRANSFER_POLICY_MAP = {
    "full_suffix": CacheTransferPolicy.FullSuffix,
    "latest_snapshot": CacheTransferPolicy.LatestSnapshot,
}
_ROLE_MAP = {
    "prefill": SchedulerConfig.Role.P,
    "decode": SchedulerConfig.Role.D,
}


def scheduler_role(disaggregation_mode: str) -> SchedulerConfig.Role:
    """The scheduler role a server disaggregation mode selects.

    ``prefill`` and ``decode`` are the two halves of the cache-transfer PD
    protocol; every other mode runs the fused scheduler.
    """
    return _ROLE_MAP.get(disaggregation_mode, SchedulerConfig.Role.Fused)


def cache_group_config(
    spec: CacheGroupSpec,
    *,
    total_pages: int,
    cache_blocks_per_lcm_block: int,
) -> CacheGroupConfig:
    """One spec as the scheduler reads it.

    Args:
        spec: The recipe's declaration.
        total_pages: Virtual block count including the null block, as the
            runtime contract publishes it -- or 0 while the pool is still
            being sized; ``CapacityModel`` never reads it.
        cache_blocks_per_lcm_block: Virtual packing (physical packing times
            ``spec.shard_count``).

    Raises:
        ValueError: A retention, family or transfer policy the scheduler has
            no enum for.
    """
    retention = _RETENTION_MAP.get(spec.retention)
    if retention is None:
        raise ValueError(
            f"cache_group_config: unsupported retention {spec.retention!r} "
            f"for group {spec.group_id!r}"
        )
    family = _FAMILY_MAP.get(spec.family)
    if family is None:
        raise ValueError(
            f"cache_group_config: unsupported family {spec.family!r} "
            f"for group {spec.group_id!r}"
        )
    # The declaration shape (row geometry or state checkpoint) stops here:
    # the scheduler only learns how many tokens one block-table slot spans.
    kwargs = dict(
        group_id=spec.group_id,
        block_granularity=int(spec.block_granularity),
        total_pages=int(total_pages),
        retention=retention,
        family=family,
        cache_blocks_per_lcm_block=int(cache_blocks_per_lcm_block),
        shard_count=spec.shard_count,
    )
    if spec.transfer_policy is not None:
        mapped_policy = _TRANSFER_POLICY_MAP.get(spec.transfer_policy)
        if mapped_policy is None:
            raise ValueError(
                "cache_group_config: unsupported transfer policy "
                f"{spec.transfer_policy!r} for group {spec.group_id!r}"
            )
        kwargs["transfer_policy"] = mapped_policy
    if spec.retention == "sliding_window":
        if spec.sliding_window_tokens is None or spec.sliding_window_tokens <= 0:
            raise ValueError(
                f"cache_group_config: sliding group {spec.group_id!r} needs a "
                "positive sliding_window_tokens"
            )
        kwargs["sliding_window_tokens"] = int(spec.sliding_window_tokens)
    # Always stated, False included: a group silently left cached when its
    # recipe declared it replayable would change what the prefix hit means.
    kwargs["replayable"] = bool(spec.replayable)
    return CacheGroupConfig(**kwargs)


@dataclass(frozen=True)
class SchedulerLimits:
    """The scheduler-side inputs a recipe sizes its pool against.

    Read once from the server and attention configs so per-group demand and
    the capacity search cannot size against different numbers, then handed
    to the ``CapacityModel`` field for field.
    """

    role: SchedulerConfig.Role
    max_live_requests: int
    max_scheduled_tokens: int
    max_context_len: int
    decode_input_tokens: int
    overlap_schedule_depth: int
    disable_prefix_cache: bool


def capacity_model(
    specs: Sequence[CacheGroupSpec],
    *,
    prefix_granularity: int,
    virtual_packing: Mapping[str, int],
    limits: SchedulerLimits,
) -> CapacityModel:
    """The scheduler's capacity model over these groups, before any pool exists.

    Group results index like ``specs``. The model validates the sizing
    inputs it reads and raises ``ValueError`` for the same violations the
    Scheduler would later reject.

    Args:
        specs: The recipe's declarations, in declaration order.
        prefix_granularity: The scheduler prefix domain the layout was packed
            for.
        virtual_packing: Virtual children per LCM block by group id.
        limits: The concurrency and reserve widths to size for.
    """
    config = SchedulerConfig()
    config.role = limits.role
    config.prefix_granularity = prefix_granularity
    config.max_scheduled_tokens = limits.max_scheduled_tokens
    config.max_batch_size = limits.max_live_requests
    config.decode_input_tokens = limits.decode_input_tokens
    config.overlap_schedule_depth = limits.overlap_schedule_depth
    config.disable_prefix_cache = limits.disable_prefix_cache
    config.cache_groups = [
        cache_group_config(
            spec,
            total_pages=0,
            cache_blocks_per_lcm_block=virtual_packing[spec.group_id],
        )
        for spec in specs
    ]
    return CapacityModel(config)


__all__ = [
    "SchedulerLimits",
    "cache_group_config",
    "capacity_model",
    "scheduler_role",
]
