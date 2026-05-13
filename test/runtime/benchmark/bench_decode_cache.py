"""
Benchmark the cache operations that run on the hot decode path.

During decode, for each completed page tokenspeed calls:
  insert(paged_key, page_ids)
  match_prefix(paged_key)             # to confirm hit
  dec_lock_ref(old_last_node)
  inc_lock_ref(new_last_node)

We simulate a steady-state server: a large cache of pre-existing sequences
(simulating the populated prefix tree after many requests) plus N concurrent
"requests" each decoding D pages one at a time.

Run with:
  python/.venv/bin/python test/runtime/benchmark/bench_decode_cache.py
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import torch

from tokenspeed.runtime.cache.prefix_cache import CacheInitParams, PrefixCache, TreeNode

# ---------------------------------------------------------------------------
# Minimal mock allocator — matches what real decode needs
# ---------------------------------------------------------------------------


class MockAllocator:
    def __init__(self, page_size: int = 16):
        self.page_size = page_size
        self._next_page = 0
        # Lazily allocated on first access; avoids 128MB upfront per benchmark run.
        self._req_to_page: torch.Tensor | None = None

    @property
    def req_to_page(self) -> torch.Tensor:
        if self._req_to_page is None:
            self._req_to_page = torch.zeros(512, 512, dtype=torch.int32)
        return self._req_to_page

    @property
    def req_to_page_cpu(self) -> torch.Tensor:
        return self.req_to_page

    def alloc_page(self) -> int:
        p = self._next_page
        self._next_page += 1
        return p

    # Called by insert/match_prefix indirectly
    def free_with_diff(self, new_ids, old_ids):
        diff = new_ids != old_ids
        return diff

    def append_to_later_free(self, t):
        pass

    def free_group_end(self):
        pass


def _make_cache(page_size: int = 16) -> tuple[PrefixCache, MockAllocator]:
    alloc = MockAllocator(page_size=page_size)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=alloc,
        page_size=page_size,
    )
    return PrefixCache(params), alloc


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _populate_cache(
    cache: PrefixCache, alloc: MockAllocator, n_background: int, prefix_pages: int = 0
) -> None:
    """Fill cache with n_background independent sequences (background traffic)."""
    page_size = cache.page_size
    for s in range(n_background):
        if prefix_pages > 0:
            key = [tuple([0] * page_size)] * prefix_pages + [
                tuple([s + 1] + [0] * (page_size - 1))
            ]
        else:
            key = [tuple([s + 1] + [0] * (page_size - 1))]
        n_pages = len(key)
        value = torch.tensor(
            [alloc.alloc_page() for _ in range(n_pages)], dtype=torch.int32
        )
        cache.insert(key, value)


# ---------------------------------------------------------------------------
# Decode simulation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimReq:
    req_id: int
    prefix_key: list  # shared prefix already in cache
    decode_pages: int  # how many pages this req will generate
    current_pages: int = 0
    last_node: Optional[TreeNode] = None


def _simulate_decode_step(
    cache: PrefixCache, alloc: MockAllocator, req: SimReq
) -> bool:
    """
    Simulate one page completion for req.
    Returns True if the request is done.
    """
    page_size = cache.page_size
    req.current_pages += 1

    # Build key: shared prefix + unique decode pages so far
    decode_part = [
        tuple([req.req_id * 10000 + p + 1] + [0] * (page_size - 1))
        for p in range(req.current_pages)
    ]
    full_key = req.prefix_key + decode_part

    value = torch.tensor(
        [alloc.alloc_page() for _ in range(len(full_key))], dtype=torch.int32
    )

    # Hot path: insert → match → dec_lock_ref → inc_lock_ref
    cache.insert(full_key, value)
    result = cache.match_prefix(full_key)
    new_last_node = result.last_device_node

    cache.dec_lock_ref(req.last_node)
    cache.inc_lock_ref(new_last_node)
    req.last_node = new_last_node

    return req.current_pages >= req.decode_pages


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def bench(
    label: str,
    n_background: int,
    n_concurrent: int,
    decode_pages: int,
    prefix_pages: int = 0,
    page_size: int = 16,
    repeats: int = 5,
) -> dict:
    """
    Measure total time for n_concurrent requests each decoding decode_pages pages,
    against a background cache of n_background entries.
    """
    times = []
    for _ in range(repeats):
        cache, alloc = _make_cache(page_size)
        _populate_cache(cache, alloc, n_background, prefix_pages)

        # Build shared prefix key (same for all requests)
        if prefix_pages > 0:
            prefix_key = [tuple([0] * page_size)] * prefix_pages
        else:
            prefix_key = []

        requests = [
            SimReq(req_id=i, prefix_key=prefix_key, decode_pages=decode_pages)
            for i in range(n_concurrent)
        ]

        # Lock each request's last_node at their prefix (simulating prefill done)
        for req in requests:
            if prefix_key:
                result = cache.match_prefix(prefix_key)
                req.last_node = result.last_device_node
                cache.inc_lock_ref(req.last_node)

        total_steps = n_concurrent * decode_pages
        step = 0
        active = list(requests)

        t0 = time.perf_counter()
        while active:
            next_active = []
            for req in active:
                done = _simulate_decode_step(cache, alloc, req)
                step += 1
                if not done:
                    next_active.append(req)
            active = next_active
        elapsed = time.perf_counter() - t0

        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    total_steps = n_concurrent * decode_pages
    return {
        "label": label,
        "n_background": n_background,
        "n_concurrent": n_concurrent,
        "decode_pages": decode_pages,
        "prefix_pages": prefix_pages,
        "total_decode_steps": total_steps,
        "median_ms": round(median * 1e3, 3),
        "us_per_step": round(median * 1e6 / total_steps, 2),
    }


if __name__ == "__main__":
    import json

    scenarios = [
        # (label,                   n_background, n_concurrent, decode_pages, prefix_pages)
        ("small_cache_short_seq", 1_000, 64, 32, 0),
        ("small_cache_long_seq", 1_000, 64, 128, 0),
        ("large_cache_short_seq", 20_000, 64, 32, 0),
        ("large_cache_long_seq", 20_000, 64, 128, 0),
        ("large_cache_shared_pfx", 20_000, 64, 32, 100),
        ("xlarge_cache_short_seq", 50_000, 64, 32, 0),
    ]

    for args in scenarios:
        r = bench(*args)
        print(json.dumps(r))
