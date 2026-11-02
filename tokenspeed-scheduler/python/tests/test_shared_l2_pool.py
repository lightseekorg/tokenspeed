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

"""Two clients sharing one L2 pool through the Python binding."""

from __future__ import annotations

import tokenspeed_scheduler as ts


def config():
    full = ts.CacheGroupConfig()
    full.group_id, full.block_granularity, full.total_pages = "full", 4, 16
    full.cache_blocks_per_lcm_block, full.attn_kind = 2, ts.AttnKind.Full
    state = ts.CacheGroupConfig()
    state.group_id, state.block_granularity, state.total_pages = "state", 0, 8
    state.cache_blocks_per_lcm_block, state.attn_kind = 1, ts.AttnKind.State
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 4
    cfg.cache_groups = [full, state]
    return cfg


def test_a_client_reads_the_pages_another_client_wrote():
    pool = ts.SharedL2Pool(config(), 8, 0)
    keys = [(0, "p1", 0), (0, "p2", 0), (1, "p2", 0)]
    ticket, pages = pool.reserve(1, keys)
    assert (
        ticket and len(set(zip((k[0] for k in keys), pages))) == 3 and min(pages) >= 0
    )
    pool.commit(1, ticket, [(list(range(8)), ["p1", "p2"], None)])

    ticket, num_tokens, kv_tokens, tail, slots = pool.probe(
        2, ["p1", "p2", "p3"], b"", 12
    )
    assert (num_tokens, tail) == (8, None)
    assert sorted(slot[:3] for slot in slots) == sorted(keys)
    assert {(slot[0], slot[3]) for slot in slots} == set(
        zip((k[0] for k in keys), pages)
    )
    assert pool.stats()["pinned_blocks"] == 3
    pool.release(2, ticket)
    assert pool.stats()["pinned_blocks"] == 0
    assert pool.reserve(2, keys) == (0, [-1, -1, -1])
