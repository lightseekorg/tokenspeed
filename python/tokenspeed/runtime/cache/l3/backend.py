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

"""Storage-backend interface for Host CacheBlocks under flat KV."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol


def storage_object_key(
    content_hash: str,
    group_id: int,
    page_offset: int,
    *,
    prefix: str = "",
    rank: int = 0,
) -> str:
    """Return the L3 object key for one packed Host CacheBlock.

    TokenSpeed's Host pool is one compact byte buffer (flat KV). One Mooncake
    object stores the packed bytes of a single CacheBlock, keyed by the
    scheduler content hash plus the group/offset/rank that uniquely identify
    the shard. Rank isolation matches SGLang/vLLM: each TP rank owns a
    different KV slice.
    """

    if not content_hash:
        raise ValueError("content_hash must be non-empty")
    tagged = f"{prefix}_{content_hash}" if prefix else content_hash
    return f"{tagged}|g{int(group_id)}|o{int(page_offset)}|r{int(rank)}"


def cache_layout_signature(layout: Any, *, cache_dtype: str) -> str:
    """Return a stable fingerprint of the bytes stored in one L3 page."""

    groups = []
    for group in layout.groups:
        fields = [
            {
                "id": field.field_id,
                "buffer": int(field.device_buffer_index),
                "zero": int(field.device_block_zero_offset_bytes),
                "stride": int(field.block_stride_bytes),
                "payload": int(field.payload_bytes),
            }
            for field in group.fields
        ]
        groups.append(
            {
                "id": group.group_id,
                "blocks_per_lcm": int(group.cache_blocks_per_lcm_block),
                "fields": fields,
            }
        )
    payload = json.dumps(
        {"cache_dtype": str(cache_dtype), "groups": groups},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def storage_key_prefix(
    model_name: str,
    *,
    revision: str,
    weight_version: str,
    cache_signature: str,
    pipeline_rank: int,
    draft_model: str,
    draft_revision: str,
    draft_weight_version: str,
) -> str:
    """Return a collision-resistant namespace for compatible L3 objects.

    Every component is required so a new caller cannot omit the checkpoint
    identity, cache layout, pipeline stage, or draft pool and silently
    collide with an incompatible deployment. Empty strings are valid and
    mean "unset" (no Hugging Face revision, no speculative draft).
    """

    payload = json.dumps(
        {
            "model": str(model_name),
            "revision": str(revision),
            "weight_version": str(weight_version),
            "cache_signature": str(cache_signature),
            "pipeline_rank": int(pipeline_rank),
            "draft_model": str(draft_model),
            "draft_revision": str(draft_revision),
            "draft_weight_version": str(draft_weight_version),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "tsl3v1-" + hashlib.sha256(payload.encode()).hexdigest()


def host_buffer_ptr(host_buffer: Any) -> int:
    data_ptr = getattr(host_buffer, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    raise TypeError(f"host buffer {type(host_buffer)!r} has no data_ptr()")


def copy_host_bytes(host_buffer: Any, offset: int, size: int) -> bytes:
    view = host_buffer[offset : offset + size]
    tobytes = getattr(view, "tobytes", None)
    if callable(tobytes):
        return bytes(tobytes())
    numpy = getattr(view, "numpy", None)
    if callable(numpy):
        return bytes(numpy())
    return bytes(view)


def write_host_bytes(host_buffer: Any, offset: int, payload: bytes) -> None:
    size = len(payload)
    dest = host_buffer[offset : offset + size]
    copy_ = getattr(dest, "copy_", None)
    if callable(copy_):
        import torch

        copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        return
    host_buffer[offset : offset + size] = payload


class KvStoreStorage(Protocol):
    """Byte store for packed Host CacheBlocks.

    Implementations must be safe to call from the runtime thread that owns
    the Host buffer. ``batch_get_into`` / ``batch_put_from`` operate on
    offsets into that registered buffer (SGLang HiCacheStorage v1).
    """

    def batch_exists(self, keys: Sequence[str]) -> list[bool]:
        """Return per-key existence, aligned with ``keys``."""

    def batch_get_into(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        """Read objects into Host buffer slices. True means the copy succeeded."""

    def batch_put_from(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        """Write Host buffer slices into the store. True means the put succeeded."""

    def remove_by_prefix(self, prefix: str) -> None:
        """Remove every object whose key starts with ``prefix``."""

    def close(self) -> None:
        """Release backend resources. Idempotent."""


class MemoryKvStore:
    """In-process dict store used by tests and as a Mooncake-free reference."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def batch_exists(self, keys: Sequence[str]) -> list[bool]:
        return [key in self._objects for key in keys]

    def batch_get_into(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 get")
        results = []
        for key, offset, size in zip(keys, offsets, sizes):
            payload = self._objects.get(key)
            if payload is None or len(payload) != size:
                results.append(False)
                continue
            write_host_bytes(host_buffer, int(offset), payload)
            results.append(True)
        return results

    def batch_put_from(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 put")
        results = []
        for key, offset, size in zip(keys, offsets, sizes):
            if key in self._objects:
                results.append(True)
                continue
            self._objects[key] = copy_host_bytes(host_buffer, int(offset), int(size))
            results.append(True)
        return results

    def remove_by_prefix(self, prefix: str) -> None:
        self._objects = {
            key: payload
            for key, payload in self._objects.items()
            if not key.startswith(prefix)
        }

    def close(self) -> None:
        self._objects.clear()
