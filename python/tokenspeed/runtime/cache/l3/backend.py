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


def storage_key_prefix(model_name: str | None) -> str:
    """Sanitize a model name into an L3 object-key prefix (SGLang-style)."""

    if not model_name:
        return ""
    return str(model_name).replace("/", "_")


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

    def close(self) -> None:
        self._objects.clear()
