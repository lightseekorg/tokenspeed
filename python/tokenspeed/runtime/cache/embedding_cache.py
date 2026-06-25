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

"""Vision-embedding cache for the EPD encode stage.

The encode server runs the vision tower only; a cache lets duplicate images
(same ``MultimodalDataItem.hash``) reuse a previously computed embedding instead
of re-running the tower. This is the encode-stage analog of the LM KV/prefix
caches that already live in this subsystem (see :mod:`prefix_cache`), so it
lives here rather than beside the transport hooks in
``runtime/disaggregation/embedding``.

Two interchangeable implementations share a ``get`` / ``put`` surface:
:class:`EmbeddingCache` (single-tier, bytes-bounded LRU) and
:class:`TieredEmbeddingCache` (L1 GPU VRAM + L2 host DRAM). Both are
framework-agnostic and unit-testable without a GPU -- the caller supplies byte
sizes and the device<->host copies are injectable.
"""

import collections
from typing import Callable, Hashable, Optional, Tuple


class EmbeddingCache:
    """Bytes-bounded LRU cache of vision-tower outputs, keyed by content hash.

    Caches vision-tower outputs so identical images (same
    ``MultimodalDataItem.hash``) reuse a previously computed embedding instead
    of re-running the tower. Values are opaque (typically a ``torch.Tensor``);
    the caller supplies the byte size so this stays framework-agnostic and
    testable without a GPU.

    ``on_evict`` is an optional ``(key, value, nbytes)`` callback fired for each
    entry dropped by *capacity overflow* (not by an update to an existing key,
    which is an in-place replace, and not by an explicit :meth:`pop`). It exists
    so a second tier can capture LRU victims and demote them (see
    :class:`TieredEmbeddingCache`); when unset the cache behaves exactly as
    before.
    """

    def __init__(
        self,
        capacity_bytes: int,
        on_evict: Optional[Callable[[Hashable, object, int], None]] = None,
    ):
        if capacity_bytes < 0:
            raise ValueError(f"capacity_bytes must be >= 0, got {capacity_bytes}")
        self.capacity_bytes = capacity_bytes
        self._on_evict = on_evict
        # key -> (value, nbytes); ordered by recency (oldest first).
        self._store: "collections.OrderedDict[Hashable, Tuple[object, int]]" = (
            collections.OrderedDict()
        )
        self._cur_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Optional[object]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key: Hashable, value: object, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError(f"nbytes must be >= 0, got {nbytes}")
        # An item larger than the whole cache can never be retained; skip it
        # rather than evicting everything for a value we must immediately drop.
        if nbytes > self.capacity_bytes:
            return
        existing = self._store.pop(key, None)
        if existing is not None:
            self._cur_bytes -= existing[1]
        self._store[key] = (value, nbytes)
        self._cur_bytes += nbytes
        self._evict()

    def pop(self, key: Hashable) -> Optional[Tuple[object, int]]:
        """Remove ``key`` and return its ``(value, nbytes)``, or ``None`` if
        absent. A structural removal: it neither counts as a hit/miss nor fires
        ``on_evict`` (the caller is taking ownership of the value, e.g. to
        promote it to another tier)."""
        existing = self._store.pop(key, None)
        if existing is None:
            return None
        self._cur_bytes -= existing[1]
        return existing

    def _evict(self) -> None:
        while self._cur_bytes > self.capacity_bytes and self._store:
            key, (value, nbytes) = self._store.popitem(last=False)
            self._cur_bytes -= nbytes
            if self._on_evict is not None:
                self._on_evict(key, value, nbytes)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    @property
    def current_bytes(self) -> int:
        return self._cur_bytes


def _embedding_to_host(value: object) -> object:
    """Default L1->L2 demote: copy a vision embedding from GPU to host memory.

    A cache value is a ``(main, deepstack)`` tuple (deepstack half may be
    ``None`` for non-deepstack models) or, for legacy/test-seeded entries, a bare
    tensor. Pageable host memory (plain ``.cpu()``) keeps the first cut simple; a
    pinned, RDMA-direct L2 is a possible follow-up.
    """
    if isinstance(value, tuple):
        return tuple(None if t is None else t.cpu() for t in value)
    return value.cpu()


def _embedding_to_device(value: object, device) -> object:
    """Default L2->L1 promote: copy a host-resident embedding back to ``device``
    so the executor can stage it into the GPU send ring."""
    if isinstance(value, tuple):
        return tuple(None if t is None else t.to(device) for t in value)
    return value.to(device)


class TieredEmbeddingCache:
    """Two-tier vision-embedding cache: L1 in GPU VRAM, L2 in host DRAM.

    Exposes the same ``get`` / ``put`` surface as :class:`EmbeddingCache`, so the
    encode worker uses either interchangeably. The lone VRAM tier (4 GiB by
    default) holds only ~150 Kimi-K2.5 images; once a worker's working set
    exceeds that, an LRU victim would otherwise be re-encoded by the tower on its
    next hit. L2 catches those victims in far cheaper host DRAM: a hit there
    skips the (much more expensive) ViT and only pays a host->device copy.

    Tiers are kept *exclusive* -- a key lives in exactly one. An L1 eviction
    demotes the victim to L2 (device->host copy); an L2 hit promotes the entry
    back to L1 (host->device copy) and removes it from L2; ``put`` always lands
    in L1 and drops any stale L2 duplicate. There is no distributed L3 tier:
    image-hash routing pins each image to one worker, so a per-worker local
    cache already captures the reuse and no cross-instance store is needed.

    The device<->host copies are injectable (``to_host`` / ``to_device``) so the
    tiering logic is unit-testable without a GPU; the defaults copy real tensors
    to/from ``device``.
    """

    def __init__(
        self,
        l1_capacity_bytes: int,
        l2_capacity_bytes: int,
        *,
        device: object = None,
        to_host: Optional[Callable[[object], object]] = None,
        to_device: Optional[Callable[[object], object]] = None,
    ):
        # L1 demotes its evictions into L2 via the on_evict hook; L2 is the
        # bottom tier (no hook), so its evictions are true drops -- no recursion.
        self.l1 = EmbeddingCache(l1_capacity_bytes, on_evict=self._demote)
        self.l2 = EmbeddingCache(l2_capacity_bytes)
        self._device = device
        self._to_host = to_host or _embedding_to_host
        self._to_device = to_device or (lambda v: _embedding_to_device(v, self._device))
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.promotions = 0
        self.demotions = 0

    def get(self, key: Hashable) -> Optional[object]:
        value = self.l1.get(key)
        if value is not None:
            self.l1_hits += 1
            return value
        promoted = self.l2.pop(key)
        if promoted is None:
            self.misses += 1
            return None
        host_value, nbytes = promoted
        device_value = self._to_device(host_value)
        self.l2_hits += 1
        self.promotions += 1
        # Re-home as MRU in L1. May evict colder L1 entries, which demote to L2;
        # the just-promoted key is MRU so it is never the victim.
        self.l1.put(key, device_value, nbytes)
        return device_value

    def put(self, key: Hashable, value: object, nbytes: int) -> None:
        # L1 is the write tier. Drop any stale L2 copy first so the tiers stay
        # exclusive (e.g. a key demoted earlier and now re-encoded on a miss).
        self.l2.pop(key)
        self.l1.put(key, value, nbytes)

    def _demote(self, key: Hashable, value: object, nbytes: int) -> None:
        """L1 eviction hook: stash the victim in host DRAM instead of dropping
        it. host nbytes == device nbytes (same dtype/numel). A victim larger than
        all of L2 is silently dropped by ``L2.put`` (same as having no L2)."""
        self.l2.put(key, self._to_host(value), nbytes)
        if key in self.l2:
            self.demotions += 1

    @property
    def hits(self) -> int:
        return self.l1_hits + self.l2_hits

    def __contains__(self, key: Hashable) -> bool:
        return key in self.l1 or key in self.l2

    def __len__(self) -> int:
        return len(self.l1) + len(self.l2)
