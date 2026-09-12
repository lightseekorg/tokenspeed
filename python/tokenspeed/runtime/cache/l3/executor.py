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

"""Copy packed Host CacheBlocks to/from the L3 store under flat KV."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from tokenspeed.runtime.cache.l3.backend import KvStoreStorage, storage_object_key

logger = logging.getLogger(__name__)

StoragePage = tuple[
    int, int, str, int
]  # group_index, host_block_id, content_hash, page_offset


class L3HostStore:
    """Zero-copy adapter from compact Host pages to a ``KvStoreStorage``."""

    def __init__(
        self,
        backend: KvStoreStorage,
        host_storage: Any,
        *,
        key_prefix: str,
        rank: int,
        cp_rank: int,
    ):
        self.backend = backend
        self.host_storage = host_storage
        self._base_key_prefix = key_prefix
        self.rank = int(rank)
        self.cp_rank = int(cp_rank)

    @property
    def key_prefix(self) -> str:
        return self._base_key_prefix

    def set_key_prefix(self, key_prefix: str) -> None:
        """Publish and restore under a new namespace without deleting it.

        Live weight updates change ``weight_version`` after Device/Host
        have been flushed and the GPU load succeeds. Flush /
        ``rotate_namespace`` still deletes the *current* prefix first so
        stale KV is gone before this switches the writers.
        """

        if not key_prefix:
            raise ValueError("key_prefix must be non-empty")
        self._base_key_prefix = key_prefix

    def rotate_namespace(self) -> bool:
        """Delete objects under this stable namespace.

        The prefix does not change, so a restarted rank in *this* job still
        probes the same keys and observes the deletion through
        ``batch_exists``. TP/CP/PP/DP MIN-reduce clearability and the
        delete result inside one TokenSpeed process group. Independent
        deployments that share a tenant are not in those groups: a
        fleet-wide wipe is an operator flush of every instance. A peer
        PUT after this delete is a later ``batch_exists`` hit, not a
        lease; vanished-L3 prefetch recovers a miss.

        Returns True when the store reports the prefix is gone. A False
        must not be followed by Device/Host ``ClearCache``.
        """

        try:
            return bool(self.backend.remove_by_prefix(f"{self.key_prefix}_"))
        except Exception:
            logger.exception("L3 namespace delete failed")
            return False

    def object_key(self, content_hash: str, group_id: int, page_offset: int) -> str:
        return storage_object_key(
            content_hash,
            group_id,
            page_offset,
            prefix=self.key_prefix,
            rank=self.rank,
            cp_rank=self.cp_rank,
        )

    def exists(self, pages: Sequence[StoragePage]) -> list[bool]:
        keys = [
            self.object_key(content_hash, group_id, page_offset)
            for group_id, _host_block, content_hash, page_offset in pages
        ]
        return self.backend.batch_exists(keys)

    def present_keys(
        self,
        group_ids: Sequence[int],
        content_hashes: Sequence[str],
        page_offsets: Sequence[int],
        exists: Sequence[bool] | None,
    ) -> tuple[list[int], list[str], list[int]]:
        """Return the subset of keys that exist in the store.

        ``exists`` is the replica-converged mask from ``batch_exists`` (or
        ``None`` to probe this rank's backend locally). Callers must pass
        it explicitly so a forgotten mask cannot silently diverge across
        cache-owning ranks.
        """

        if not (len(group_ids) == len(content_hashes) == len(page_offsets)):
            raise ValueError("ragged L3 key lists")
        if exists is None:
            pages = [
                (int(group_id), 0, content_hash, int(page_offset))
                for group_id, content_hash, page_offset in zip(
                    group_ids, content_hashes, page_offsets
                )
            ]
            exists = self.exists(pages)
        if len(exists) != len(group_ids):
            raise ValueError("exists mask length must match keys")
        hit_groups: list[int] = []
        hit_hashes: list[str] = []
        hit_offsets: list[int] = []
        for group_id, content_hash, page_offset, present in zip(
            group_ids, content_hashes, page_offsets, exists
        ):
            if not present:
                continue
            hit_groups.append(int(group_id))
            hit_hashes.append(content_hash)
            hit_offsets.append(int(page_offset))
        return hit_groups, hit_hashes, hit_offsets

    def _ranges(
        self, pages: Sequence[StoragePage]
    ) -> tuple[list[str], list[int], list[int]]:
        keys = []
        offsets = []
        sizes = []
        for group_id, host_block_id, content_hash, page_offset in pages:
            offset, size = self.host_storage.host_block_range(
                int(group_id), int(host_block_id)
            )
            keys.append(self.object_key(content_hash, int(group_id), int(page_offset)))
            offsets.append(offset)
            sizes.append(size)
        return keys, offsets, sizes

    def backup(self, pages: Sequence[StoragePage]) -> list[bool]:
        if not pages:
            return []
        keys, offsets, sizes = self._ranges(pages)
        results = self.backend.batch_put_from(
            keys, self.host_storage.host_buffer, offsets, sizes
        )
        logger.info(
            "[L3] backup pages=%d ok=%d", len(pages), sum(1 for ok in results if ok)
        )
        return results

    def prefetch(self, pages: Sequence[StoragePage]) -> list[bool]:
        if not pages:
            return []
        keys, offsets, sizes = self._ranges(pages)
        results = self.backend.batch_get_into(
            keys, self.host_storage.host_buffer, offsets, sizes
        )
        logger.info(
            "[L3] prefetch pages=%d ok=%d", len(pages), sum(1 for ok in results if ok)
        )
        return results

    def close(self) -> None:
        self.backend.close()
