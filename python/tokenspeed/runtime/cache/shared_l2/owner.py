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

"""The cache index of the pool's owner: the scheduler's ``SharedL2Pool``.

The owner is one process serving the pool's cache index to every client; it
holds no bytes. ``python -m tokenspeed.runtime.cache.shared_l2 owner`` starts
it, and its service runs in the engine's host extension (``shared_l2/owner.rs``
of ``ts-control``, which lists the requests it answers), over the pool that
``make_shared_l2_pool`` builds here: once, when every expected provider has
joined and the first client reports its cache geometry.

A client's geometry is its scheduler's view of the cache: ``prefix_granularity``,
``lcm_block_bytes`` and ``groups``, which holds, per cache group in the scheduler's
order, ``group_id``, ``attn_kind``, ``block_granularity``,
``cache_blocks_per_lcm_block``, ``host_group_packing``, ``shard_count``,
``sliding_window_tokens``, ``sink_tokens``, ``replayable`` and
``checkpoint_granularity``.
"""

from __future__ import annotations

from typing import Any


def make_shared_l2_pool(
    geometry: dict[str, Any],
    num_lcm_blocks: int,
    *,
    headroom_lcm_blocks: int,
    checkpoint_retention_requests: int,
    checkpoint_retention_leaf_distance: int,
):
    """The pool's cache index, laid out as a client's Host tier is.

    Args:
        geometry: The first client's cache geometry.
        num_lcm_blocks: LCM blocks of every provider together.
        headroom_lcm_blocks: Empty LCM blocks a reservation evicts back up to.
        checkpoint_retention_requests: The scheduler's setting of that name,
            counted over the lookups of every client, hit or miss; 0 ranks
            every entry by its own use.
        checkpoint_retention_leaf_distance: The scheduler's setting of that
            name.
    """
    import tokenspeed_engine.scheduler as ts

    groups = []
    for group in geometry["groups"]:
        config = ts.CacheGroupConfig()
        config.group_id = str(group["group_id"])
        config.attn_kind = ts.AttnKind[group["attn_kind"]]
        config.block_granularity = int(group["block_granularity"])
        config.cache_blocks_per_lcm_block = int(group["cache_blocks_per_lcm_block"])
        config.shard_count = int(group["shard_count"])
        config.sliding_window_tokens = group["sliding_window_tokens"]
        config.sink_tokens = int(group["sink_tokens"])
        config.replayable = bool(group["replayable"])
        config.checkpoint_granularity = int(group["checkpoint_granularity"])
        groups.append(config)
    scheduler_config = ts.SchedulerConfig()
    scheduler_config.prefix_granularity = int(geometry["prefix_granularity"])
    scheduler_config.cache_groups = groups
    scheduler_config.host_group_packing = [
        int(group["host_group_packing"]) for group in geometry["groups"]
    ]
    scheduler_config.checkpoint_retention_requests = checkpoint_retention_requests
    scheduler_config.checkpoint_retention_leaf_distance = (
        checkpoint_retention_leaf_distance
    )
    return ts.SharedL2Pool(scheduler_config, num_lcm_blocks, headroom_lcm_blocks)
