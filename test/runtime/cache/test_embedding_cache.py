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

"""Unit tests for the EPD vision-embedding cache.

These exercise pure Python logic (no GPU, no model, no torch import): the
bytes-bounded LRU and the two-tier (L1 VRAM + L2 host DRAM) cache that let
duplicate images skip the vision tower.
"""

from __future__ import annotations

from tokenspeed.runtime.cache.embedding_cache import (
    EmbeddingCache,
    TieredEmbeddingCache,
)


# --------------------------------------------------------------------------- #
# EmbeddingCache
# --------------------------------------------------------------------------- #
def test_cache_hit_miss_and_counters():
    c = EmbeddingCache(capacity_bytes=1000)
    assert c.get("a") is None
    assert c.misses == 1
    c.put("a", "embA", nbytes=100)
    assert c.get("a") == "embA"
    assert c.hits == 1
    assert len(c) == 1
    assert c.current_bytes == 100


def test_cache_lru_eviction_by_bytes():
    c = EmbeddingCache(capacity_bytes=250)
    c.put("a", "A", 100)
    c.put("b", "B", 100)
    # Touch "a" so "b" becomes the LRU victim.
    assert c.get("a") == "A"
    c.put("c", "C", 100)  # 300 > 250 -> evict LRU ("b")
    assert "a" in c
    assert "c" in c
    assert "b" not in c
    assert c.current_bytes == 200


def test_cache_skips_item_larger_than_capacity():
    c = EmbeddingCache(capacity_bytes=50)
    c.put("a", "A", 10)
    c.put("big", "B", 100)  # too big to ever retain; must not evict "a"
    assert "big" not in c
    assert "a" in c
    assert c.current_bytes == 10


# --------------------------------------------------------------------------- #
# TieredEmbeddingCache (L1 VRAM + L2 host DRAM)
#
# Inject identity / tagging copy functions so the tiering logic is exercised
# without a GPU (the production defaults call .cpu()/.to(device) on real
# tensors). Sizes are tiny so eviction is easy to drive.
# --------------------------------------------------------------------------- #
def _tiered(l1, l2, *, to_host=lambda v: v, to_device=lambda v: v):
    return TieredEmbeddingCache(l1, l2, to_host=to_host, to_device=to_device)


def test_tiered_demote_on_l1_eviction():
    # L1 holds one 100B entry; the second put evicts the first down to L2.
    tc = _tiered(150, 1000)
    tc.put("a", "A", 100)
    tc.put("b", "B", 100)  # 200 > 150 -> "a" demoted to L2
    assert "a" not in tc.l1 and "a" in tc.l2  # exclusive
    assert "b" in tc.l1 and "b" not in tc.l2
    assert tc.demotions == 1
    # Aggregate views span both tiers.
    assert "a" in tc and "b" in tc  # TieredEmbeddingCache.__contains__
    assert len(tc) == 2  # __len__ = len(l1) + len(l2)


def test_tiered_promote_on_l2_hit_is_exclusive():
    tc = _tiered(150, 1000)
    tc.put("a", "A", 100)
    tc.put("b", "B", 100)  # "a" -> L2
    # Hit on "a": promoted back to L1, removed from L2; "b" (now LRU) demotes.
    assert tc.get("a") == "A"
    assert (tc.l1_hits, tc.l2_hits, tc.misses) == (0, 1, 0)
    assert tc.promotions == 1
    assert "a" in tc.l1 and "a" not in tc.l2
    assert "b" in tc.l2 and "b" not in tc.l1
    assert tc.demotions == 2  # "a" out, then "b" out
    # Aggregate `hits` includes L2 hits (here l1_hits=0, l2_hits=1).
    assert tc.hits == 1


def test_tiered_l2_disabled_behaves_single_tier():
    # L2 capacity 0: demote is a no-op, nothing is ever promoted.
    tc = _tiered(150, 0)
    tc.put("a", "A", 100)
    tc.put("b", "B", 100)  # "a" evicted; L2 can't hold it -> truly dropped
    assert "a" not in tc.l1 and "a" not in tc.l2
    assert tc.demotions == 0
    assert tc.get("a") is None
    assert tc.misses == 1
    assert "b" in tc.l1
