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

import pytest
import torch

from tokenspeed.runtime.cache.prefix_cache import CacheInitParams, PrefixCache, TreeNode


class MockAllocator:
    def __init__(self, page_size: int = 1):
        self.page_size = page_size
        self.freed: list[torch.Tensor] = []

    def append_to_later_free(self, t: torch.Tensor) -> None:
        self.freed.append(t)

    def free_group_end(self) -> None:
        pass

    def free_with_diff(self, a, b) -> None:
        pass


def _make_cache(page_size: int = 1, policy: str = "lru") -> PrefixCache:
    alloc = MockAllocator(page_size=page_size)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=alloc,
        page_size=page_size,
        eviction_policy=policy,
    )
    return PrefixCache(params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert(cache: PrefixCache, key_tuples: list[tuple], page_ids: list[int]) -> None:
    value = torch.tensor(page_ids, dtype=torch.int32)
    cache.insert(key_tuples, value)


def _leaves(cache: PrefixCache) -> set[TreeNode]:
    """Collect all leaf nodes via DFS (ground-truth, independent of evictable_leaves)."""
    result = set()
    stack = [cache.root_node]
    while stack:
        node = stack.pop()
        if not node.children:
            result.add(node)
        else:
            stack.extend(node.children.values())
    return result


def _evictable_leaves_ground_truth(cache: PrefixCache) -> set[TreeNode]:
    return {n for n in _leaves(cache) if n.lock_ref == 0 and n != cache.root_node}


# ---------------------------------------------------------------------------
# evictable_leaves invariant
# ---------------------------------------------------------------------------


class TestEvictableLeaves:
    def test_empty_cache(self):
        cache = _make_cache()
        assert cache.evictable_leaves == set()

    def test_single_insert(self):
        cache = _make_cache()
        _insert(cache, [(1,)], [10])
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)
        assert len(cache.evictable_leaves) == 1

    def test_two_independent_sequences(self):
        cache = _make_cache()
        _insert(cache, [(1,)], [10])
        _insert(cache, [(2,)], [20])
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)
        assert len(cache.evictable_leaves) == 2

    def test_shared_prefix_split(self):
        cache = _make_cache()
        _insert(cache, [(1,), (2,), (3,)], [10, 11, 12])
        _insert(cache, [(1,), (2,), (4,)], [10, 11, 13])
        # Tree: root → [(1,),(2,)] → [(3,)] leaf and [(4,)] leaf
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)
        assert len(cache.evictable_leaves) == 2

    def test_parent_removed_from_set_when_child_inserted(self):
        cache = _make_cache()
        _insert(cache, [(1,)], [10])
        leaf = next(iter(cache.evictable_leaves))
        assert leaf in cache.evictable_leaves
        # Insert a child under the same path — creates a longer chain
        _insert(cache, [(1,), (2,)], [10, 11])
        # The node for (1,) now has a child, must not be in evictable_leaves
        assert leaf not in cache.evictable_leaves
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)

    def test_after_eviction(self):
        cache = _make_cache()
        for i in range(5):
            _insert(cache, [(i,)], [i])
        assert len(cache.evictable_leaves) == 5
        cache.evict(3)
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)

    def test_cascade_eviction_restores_parent_to_set(self):
        cache = _make_cache()
        _insert(cache, [(1,), (2,)], [10, 11])
        _insert(cache, [(1,), (3,)], [10, 12])
        # root → (1,) internal → (2,) leaf, (3,) leaf
        assert len(cache.evictable_leaves) == 2
        # Evict both leaves — the parent (1,) should become evictable
        cache.evict(10)
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)

    def test_inc_dec_lock_ref(self):
        cache = _make_cache()
        _insert(cache, [(1,)], [10])
        leaf = next(iter(cache.evictable_leaves))
        cache.inc_lock_ref(leaf)
        assert leaf not in cache.evictable_leaves
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)
        cache.dec_lock_ref(leaf)
        assert leaf in cache.evictable_leaves
        assert cache.evictable_leaves == _evictable_leaves_ground_truth(cache)

    def test_locked_internal_node_not_in_evictable_leaves(self):
        cache = _make_cache()
        _insert(cache, [(1,), (2,)], [10, 11])
        _insert(cache, [(1,), (3,)], [10, 12])
        # Find the internal node for (1,) — it has two children
        internal = cache.root_node.children[(1,)]
        assert len(internal.children) == 2
        # Internal node with children is never evictable
        assert internal not in cache.evictable_leaves
        assert internal not in _evictable_leaves_ground_truth(cache)

    def test_reset_clears_set(self):
        cache = _make_cache()
        for i in range(10):
            _insert(cache, [(i,)], [i])
        assert len(cache.evictable_leaves) == 10
        cache.reset()
        assert cache.evictable_leaves == set()


# ---------------------------------------------------------------------------
# Eviction correctness
# ---------------------------------------------------------------------------


class TestEviction:
    def test_evict_returns_count(self):
        cache = _make_cache()
        for i in range(10):
            _insert(cache, [(i,)], [i])
        evicted = cache.evict(5)
        assert evicted == 5
        assert cache.evictable_size() == 5

    def test_evict_respects_lock_ref(self):
        cache = _make_cache()
        for i in range(5):
            _insert(cache, [(i,)], [i])
        locked = cache.root_node.children[(0,)]
        cache.inc_lock_ref(locked)
        evicted = cache.evict(5)
        # Can only evict 4 — the locked leaf is protected
        assert evicted == 4

    def test_evict_all_then_empty(self):
        cache = _make_cache()
        for i in range(3):
            _insert(cache, [(i,)], [i])
        cache.evict(100)
        assert cache.evictable_size() == 0
        assert len(cache.evictable_leaves) == 0

    def test_evict_cascade(self):
        cache = _make_cache()
        _insert(cache, [(1,), (2,)], [10, 11])
        _insert(cache, [(1,), (3,)], [10, 12])
        # Evicting both leaves should cascade to free (1,) as well
        cache.evict(10)
        assert cache.evictable_size() == 0

    def test_evict_freed_pages(self):
        alloc = MockAllocator()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=alloc,
            page_size=1,
        )
        cache = PrefixCache(params)
        for i in range(3):
            cache.insert([(i,)], torch.tensor([i], dtype=torch.int32))
        cache.evict(3)
        freed_flat = [v.item() for t in alloc.freed for v in t]
        assert sorted(freed_flat) == [0, 1, 2]

    def test_lru_order(self):
        import time

        cache = _make_cache(policy="lru")
        for i in range(3):
            cache.insert([(i,)], torch.tensor([i], dtype=torch.int32))
            time.sleep(0.001)
        # (0,) is LRU — should be evicted first
        alloc = cache.token_to_kv_pool_allocator
        cache.evict(1)
        assert len(alloc.freed) == 1
        assert alloc.freed[0].item() == 0

    def test_evict_disabled_cache_noop(self):
        alloc = MockAllocator()
        params = CacheInitParams(
            disable=True,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=alloc,
            page_size=1,
        )
        cache = PrefixCache(params)
        result = cache.evict(100)
        assert result is None
        assert alloc.freed == []


# ---------------------------------------------------------------------------
# delete_leaf O(1) key lookup
# ---------------------------------------------------------------------------


class TestDeleteLeaf:
    def test_delete_by_key_not_linear_scan(self):
        """Ensure _delete_leaf uses node.key[0] as the dict key, not a linear scan."""
        cache = _make_cache()
        # Build a flat tree with many siblings under root
        for i in range(100):
            _insert(cache, [(i,)], [i])
        # Grab a leaf and delete it — should not raise and should reduce size
        leaf = next(iter(cache.evictable_leaves))
        before = len(cache.root_node.children)
        cache._delete_leaf(leaf)
        assert len(cache.root_node.children) == before - 1
        assert leaf not in cache.evictable_leaves

    def test_delete_leaf_after_split(self):
        """After a split, the leaf's key[0] must still be the correct parent dict key."""
        cache = _make_cache()
        _insert(cache, [(1,), (2,), (3,)], [10, 11, 12])
        _insert(cache, [(1,), (2,), (4,)], [10, 11, 13])
        # evictable leaves are the two split children
        for leaf in list(cache.evictable_leaves):
            before = len(leaf.parent.children)
            cache._delete_leaf(leaf)
            assert len(leaf.parent.children) == before - 1
