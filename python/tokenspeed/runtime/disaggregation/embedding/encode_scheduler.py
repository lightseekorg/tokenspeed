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

"""Encode-stage batching for EPD disaggregation.

The encode server runs the vision tower only and is orchestrated here in
Python rather than through the C++ KV scheduler: an encode request carries no
KV cache, runs no LM forward, and has no chunked-prefill / retract semantics,
so the paged-cache machinery does not apply. What it does need is a way to
batch pending vision items into a single ViT forward under a token budget --
that is :class:`EncodeScheduler` below.

The companion duplicate-image cache lives in
:mod:`tokenspeed.runtime.cache.embedding_cache` (it is a cache, so it sits with
the other caches). The two are intentionally decoupled: the encode loop checks
the cache on arrival (resolving hits immediately) and only feeds *misses* to
the scheduler.
"""

import dataclasses
from typing import Dict, List, Tuple


@dataclasses.dataclass(frozen=True)
class PendingEncodeItem:
    """One vision item awaiting encode, identified within its request.

    ``cost`` is the item's vision-token (patch) count, used as the batching
    budget unit.
    """

    request_id: str
    item_index: int
    cost: int

    @property
    def key(self) -> Tuple[str, int]:
        return (self.request_id, self.item_index)


class EncodeScheduler:
    """Deterministic patch-budget batcher for the encode (vision-tower) stage.

    Collects pending vision items (cache misses) and packs them into batches
    bounded by a per-batch token budget and a max item count. Ordering is
    deterministic across tensor-parallel ranks -- items are sorted by
    ``(request_id, item_index)`` -- so every rank forms an identical ViT batch.
    This matters because the vision tower can be tensor-parallel; non-identical
    batches across ranks would deadlock the NCCL collectives inside it. (It is
    the same reason the C++ LM scheduler tie-breaks its candidates by request
    id.)

    Greedy packing: items are admitted in order until the next one would exceed
    ``max_tokens_per_batch`` or ``max_items_per_batch``. A single item whose
    cost alone exceeds the token budget is still returned alone, so it always
    makes progress.
    """

    def __init__(self, max_tokens_per_batch: int, max_items_per_batch: int):
        if max_tokens_per_batch <= 0:
            raise ValueError(
                f"max_tokens_per_batch must be > 0, got {max_tokens_per_batch}"
            )
        if max_items_per_batch <= 0:
            raise ValueError(
                f"max_items_per_batch must be > 0, got {max_items_per_batch}"
            )
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_items_per_batch = max_items_per_batch
        self._pending: Dict[Tuple[str, int], PendingEncodeItem] = {}

    def add(self, item: PendingEncodeItem) -> None:
        # Idempotent on (request_id, item_index): a re-added item overwrites.
        self._pending[item.key] = item

    def pending_size(self) -> int:
        return len(self._pending)

    def _ordered_pending(self) -> List[PendingEncodeItem]:
        # Sort by (request_id, item_index) for cross-rank determinism.
        return [self._pending[k] for k in sorted(self._pending.keys())]

    def next_batch(self) -> List[PendingEncodeItem]:
        """Pop and return the next deterministic batch of items to encode.

        Empty when nothing is pending. Removes the returned items from the
        pending set.
        """
        batch: List[PendingEncodeItem] = []
        used = 0
        for it in self._ordered_pending():
            if batch and (
                used + it.cost > self.max_tokens_per_batch
                or len(batch) >= self.max_items_per_batch
            ):
                break
            batch.append(it)
            used += it.cost
            del self._pending[it.key]
        return batch
