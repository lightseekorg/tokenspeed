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

"""Clients sharing one L2 pool through the Python binding, as the pool's
owner service drives it: keys, tails and tokens are plain values it relays."""

from __future__ import annotations

import array
import hashlib
import struct

import pytest
import tokenspeed_scheduler as ts

GRAIN = 4


def config():
    full = ts.CacheGroupConfig()
    full.group_id, full.block_granularity, full.total_pages = "full", GRAIN, 16
    full.cache_blocks_per_lcm_block, full.attn_kind = 2, ts.AttnKind.Full
    state = ts.CacheGroupConfig()
    state.group_id, state.block_granularity, state.total_pages = "state", 0, 8
    state.cache_blocks_per_lcm_block, state.attn_kind = 1, ts.AttnKind.State
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = GRAIN
    cfg.cache_groups = [full, state]
    cfg.host_group_packing = [2, 1]
    cfg.checkpoint_retention_requests = 64
    return cfg


def tokens_bytes(tokens):
    return array.array("i", tokens).tobytes()


def key(group, content_hash):
    return (0, group, content_hash, 0)


def page_hash(tokens, prior):
    """The scheduler's chain hash of one page (or tail) after `prior`."""
    prior_bytes = bytes.fromhex(prior)
    return hashlib.sha256(
        struct.pack("<I", len(prior_bytes))
        + prior_bytes
        + struct.pack(f"<I{len(tokens)}i", len(tokens), *tokens)
    ).hexdigest()


class Prompt:
    """A prompt hashed as a rank's scheduler hashes it: its full pages, and
    its partial page as a tail."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.hashes = []
        pages = len(self.tokens) // GRAIN
        for page in range(pages):
            prior = self.hashes[-1] if self.hashes else ""
            self.hashes.append(
                page_hash(self.tokens[page * GRAIN : (page + 1) * GRAIN], prior)
            )
        self.tail = None
        if len(self.tokens) % GRAIN:
            prior = self.hashes[-1] if self.hashes else ""
            self.tail = (
                len(self.tokens) % GRAIN,
                page_hash(self.tokens[pages * GRAIN :], prior),
            )

    @property
    def end(self):
        return self.tail[1] if self.tail else self.hashes[-1]

    def keys(self):
        """Full pages, the tail's block, then the State checkpoint."""
        full = [key(0, content_hash) for content_hash in self.hashes]
        if self.tail:
            full.append(key(0, self.tail[1]))
        return full + [key(1, self.end)]

    def publication(self, publisher):
        return (tokens_bytes(self.tokens), self.hashes, self.tail, publisher)

    def probe(self, pool, client):
        return pool.probe(
            client, self.hashes, tokens_bytes(self.tokens), len(self.tokens)
        )


def write(pool, client, prompt):
    keys = prompt.keys()
    ticket, pages = pool.reserve(client, keys, [0] * len(keys))
    assert ticket and min(pages) >= 1
    assert pool.commit(client, ticket, [prompt.publication(1)])
    return dict(zip(keys, pages))


def test_a_client_reads_the_pages_another_client_wrote():
    pool = ts.SharedL2Pool(config(), 8, 0)
    prompt = Prompt(range(10))
    pages = write(pool, 1, prompt)
    # A group's pages are numbered on their own.
    assert len({(k[1], page) for k, page in pages.items()}) == 4

    ticket, num_tokens, kv_tokens, tail, slots = Prompt(range(14)).probe(pool, 2)
    assert (num_tokens, kv_tokens, tail) == (10, 10, prompt.tail)
    assert {k: page for k, _, page in slots} == pages
    assert all(bucket == 0 for _, bucket, _ in slots)
    assert pool.stats()["pinned_blocks"] == 4
    assert pool.stats()["lookups"] == 1
    pool.release(2, ticket)
    assert pool.stats()["pinned_blocks"] == 0
    keys = prompt.keys()
    assert pool.reserve(2, keys, [0] * len(keys)) == (0, [-1] * len(keys))
    assert pool.stats()["checkpoints"] == 1


def test_keys_and_tails_survive_a_wire_that_turns_tuples_into_lists():
    pool = ts.SharedL2Pool(config(), 8, 0)
    prompt = Prompt(range(10))
    keys = [list(k) for k in prompt.keys()]
    ticket, pages = pool.reserve(1, keys, [0] * len(keys))
    tokens, hashes, tail, publisher = prompt.publication(5)
    assert pool.commit(1, ticket, [[tokens, list(hashes), list(tail), publisher]])
    pool.touch(keys)
    ticket, num_tokens, _, tail, slots = prompt.probe(pool, 2)
    assert num_tokens == 10 and tail == prompt.tail
    assert sorted(k for k, _, _ in slots) == sorted(prompt.keys())
    assert sorted(page for _, _, page in slots) == sorted(pages)


def test_misuse_is_refused():
    cfg = config()
    cfg.checkpoint_retention_requests = None
    with pytest.raises(ValueError, match="checkpoint_retention_requests"):
        ts.SharedL2Pool(cfg, 8, 0)
    cfg = config()
    cfg.host_group_packing = []
    with pytest.raises(ValueError, match="Host packing"):
        ts.SharedL2Pool(cfg, 8, 0)

    pool = ts.SharedL2Pool(config(), 8, 0)
    prompt = Prompt(range(8))
    with pytest.raises(ValueError, match="int32 bytes"):
        pool.probe(1, prompt.hashes, b"abc", 8)
    assert pool.stats()["lookups"] == 0
