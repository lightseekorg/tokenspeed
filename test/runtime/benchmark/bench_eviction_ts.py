"""
Benchmark KV cache eviction speed for tokenspeed PrefixCache.

Measures how long evict() takes across different tree sizes and shapes.
Output is JSON lines so bench_eviction.sh can parse and compare.
"""

import json
import time

import torch

from tokenspeed.runtime.cache.prefix_cache import CacheInitParams, PrefixCache

# ---------------------------------------------------------------------------
# Minimal mock allocator – no actual GPU memory needed
# ---------------------------------------------------------------------------


class MockAllocator:
    def __init__(self, page_size: int = 16):
        self.page_size = page_size
        self._pending: list[torch.Tensor] = []
        self.free_slots: list[int] = list(range(1, 10_000_001))

    def append_to_later_free(self, page_ids: torch.Tensor) -> None:
        self._pending.append(page_ids)

    def free_group_end(self) -> None:
        self._pending.clear()

    def free_with_diff(self, a, b) -> None:
        pass

    def free_req_cache(self, *args, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Tree-building helpers
# ---------------------------------------------------------------------------


def _make_cache(page_size: int = 16) -> PrefixCache:
    allocator = MockAllocator(page_size=page_size)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
    )
    return PrefixCache(params)


def _fill_flat(cache: PrefixCache, n: int, pages_per_seq: int = 1) -> None:
    """Insert n unique sequences, each pages_per_seq pages, no prefix sharing."""
    page_size = cache.page_size
    page_id = 0
    for s in range(n):
        key = [tuple([s] + [0] * (page_size - 1))]  # unique first page
        for p in range(1, pages_per_seq):
            key.append(tuple([0] * page_size))
        value = torch.arange(page_id, page_id + pages_per_seq, dtype=torch.int32)
        page_id += pages_per_seq
        cache.insert(key, value)


def _fill_shared_prefix(cache: PrefixCache, n: int, prefix_pages: int = 100) -> None:
    """n leaves, all sharing a common prefix of prefix_pages pages."""
    page_size = cache.page_size
    shared_prefix = [tuple([0] * page_size) for _ in range(prefix_pages)]
    page_id = 0
    for s in range(n):
        key = shared_prefix + [tuple([s + 1] + [0] * (page_size - 1))]
        value = torch.arange(page_id, page_id + len(key), dtype=torch.int32)
        page_id += len(key)
        cache.insert(key, value)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _bench(
    label: str,
    n: int,
    pages_per_seq: int,
    fill_fn,
    evict_fraction: float = 0.5,
    repeats: int = 7,
) -> dict:
    cache = _make_cache()
    fill_fn(cache, n, pages_per_seq)

    evict_pages = int(cache.evictable_size() * evict_fraction)
    if evict_pages == 0:
        evict_pages = 1

    times = []
    for _ in range(repeats):
        # Refill after each eviction so the tree size stays consistent
        t0 = time.perf_counter()
        cache.evict(evict_pages)
        times.append(time.perf_counter() - t0)
        # Refill
        fill_fn(cache, n, pages_per_seq)

    times.sort()
    median = times[len(times) // 2]
    return {
        "system": "tokenspeed",
        "label": label,
        "n_seq": n,
        "pages_per_seq": pages_per_seq,
        "evict_pages": evict_pages,
        "median_ms": round(median * 1e3, 4),
        "min_ms": round(times[0] * 1e3, 4),
        "max_ms": round(times[-1] * 1e3, 4),
    }


def _bench_insert(
    label: str, n: int, pages_per_seq: int, fill_fn, repeats: int = 7
) -> dict:
    """Measure amortized insert cost per sequence (N inserts into fresh cache)."""
    times = []
    for _ in range(repeats):
        cache = _make_cache()
        t0 = time.perf_counter()
        fill_fn(cache, n, pages_per_seq)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    return {
        "system": "tokenspeed",
        "label": label,
        "n_seq": n,
        "pages_per_seq": pages_per_seq,
        "op": "insert",
        "median_ms": round(median * 1e3, 4),
        "min_ms": round(times[0] * 1e3, 4),
        "us_per_insert": round(median * 1e6 / n, 3),
    }


if __name__ == "__main__":
    results = []

    # Eviction benchmarks: 50% and 5% eviction fractions
    for n in [1_000, 5_000, 20_000, 50_000]:
        for frac, frac_label in [(0.5, "evict50pct"), (0.05, "evict5pct")]:
            results.append(
                _bench(
                    label=f"flat_1page_n{n}_{frac_label}",
                    n=n,
                    pages_per_seq=1,
                    fill_fn=_fill_flat,
                    evict_fraction=frac,
                )
            )
            results.append(
                _bench(
                    label=f"shared100_1page_n{n}_{frac_label}",
                    n=n,
                    pages_per_seq=100,
                    fill_fn=_fill_shared_prefix,
                    evict_fraction=frac,
                )
            )

    # Insert overhead (decode preparation cost)
    for n in [1_000, 5_000, 20_000]:
        results.append(
            _bench_insert(
                label=f"insert_flat_1page_n{n}",
                n=n,
                pages_per_seq=1,
                fill_fn=_fill_flat,
            )
        )

    for r in results:
        print(json.dumps(r))
