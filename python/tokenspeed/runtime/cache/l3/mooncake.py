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

"""Mooncake Store backend for packed Host CacheBlocks.

Mirrors SGLang HiCache ``MooncakeStore`` / vLLM ``MooncakeStoreConnector``:
``MooncakeDistributedStore.setup`` then zero-copy ``batch_put_from`` /
``batch_get_into`` against the registered Host L2 buffer.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.cache.l3.backend import host_buffer_ptr
from tokenspeed.runtime.cache.l3.io_types import StorageIOResult

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024
_DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024
_DEFAULT_TENANT_ID = "default"


def _require_result_len(results: Sequence[Any], expected: int, *, op: str) -> None:
    """Reject a truncated Mooncake reply before zip can drop unmatched keys."""

    if len(results) != expected:
        raise ValueError(
            f"Mooncake {op} returned {len(results)} results for {expected} keys"
        )


def _parse_global_segment_size(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("global_segment_size must be an int or size string")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("gb"):
            number = text[:-2].strip()
            if not number:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(number) * 1024 * 1024 * 1024
        if text.endswith("mb"):
            number = text[:-2].strip()
            if not number:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'mb'"
                )
            return int(number) * 1024 * 1024
        return int(text)
    return int(value)


@dataclass(frozen=True, slots=True)
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    client_server_address: str
    # Required so a missed argument cannot silently join the shared default tenant.
    tenant_id: str

    @staticmethod
    def from_mapping(extra_config: dict[str, Any] | None) -> MooncakeStoreConfig:
        extra_config = extra_config or {}
        master = extra_config.get("master_server_address") or os.environ.get(
            "MOONCAKE_MASTER", ""
        )
        client = extra_config.get("client_server_address") or os.environ.get(
            "MOONCAKE_CLIENT", ""
        )
        if not master:
            master = client
        if not master:
            raise ValueError(
                "Mooncake Store requires master_server_address or client_server_address "
                "in --kvstore-storage-backend-extra-config, or MOONCAKE_MASTER / "
                "MOONCAKE_CLIENT in the environment"
            )
        local_hostname = (
            extra_config.get("local_hostname")
            or os.environ.get("MOONCAKE_LOCAL_HOSTNAME")
            or os.environ.get("LOCAL_HOSTNAME")
            or "localhost"
        )
        return MooncakeStoreConfig(
            local_hostname=str(local_hostname),
            metadata_server=str(
                extra_config.get(
                    "metadata_server",
                    os.environ.get("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
                )
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size",
                    os.environ.get(
                        "MOONCAKE_GLOBAL_SEGMENT_SIZE", _DEFAULT_GLOBAL_SEGMENT_SIZE
                    ),
                )
            ),
            protocol=str(
                extra_config.get("protocol", os.environ.get("MOONCAKE_PROTOCOL", "tcp"))
            ),
            device_name=str(
                extra_config.get("device_name", os.environ.get("MOONCAKE_DEVICE", ""))
            ),
            master_server_address=str(master),
            client_server_address=str(client),
            tenant_id=str(extra_config.get("tenant_id") or _DEFAULT_TENANT_ID),
        )


class MooncakeKvStore:
    """Zero-copy Mooncake Store adapter for one compact Host buffer.

    ``extra_config`` is required. Pass ``None`` to fall back to
    ``MOONCAKE_MASTER`` / ``MOONCAKE_CLIENT`` and the other env defaults.
    """

    def __init__(
        self,
        extra_config: dict[str, Any] | None,
        *,
        host_buffer: Any,
        tp_size: int,
        pp_size: int,
    ):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to use --kvstore-storage-backend mooncake."
            ) from exc

        # An exceptional SDK return does not prove that RDMA stopped using Host.
        self.failures_retire = False
        self.config = MooncakeStoreConfig.from_mapping(extra_config)
        self.store = MooncakeDistributedStore()
        tp_size = max(int(tp_size), 1)
        pp_size = max(int(pp_size), 1)
        per_rank_segment = self.config.global_segment_size // (tp_size * pp_size)
        # Zero explicitly selects a client backed by independently mounted
        # store segments, so inference restarts need not remove cached objects.
        if self.config.global_segment_size < 0 or (
            self.config.global_segment_size > 0 and per_rank_segment == 0
        ):
            raise ValueError(
                "Mooncake global_segment_size "
                f"{self.config.global_segment_size} is too small to split "
                f"across tp_size={tp_size} pp_size={pp_size}"
            )
        setup_kwargs: dict[str, Any] = {}
        if self.config.tenant_id != _DEFAULT_TENANT_ID:
            setup_kwargs["tenant_id"] = self.config.tenant_id
        try:
            ret = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                per_rank_segment,
                _DEFAULT_LOCAL_BUFFER_SIZE,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
                **setup_kwargs,
            )
        except TypeError as exc:
            if setup_kwargs:
                raise RuntimeError(
                    "Installed Mooncake does not support tenant_id; refusing to "
                    f"connect tenant {self.config.tenant_id!r} as the default tenant"
                ) from exc
            ret = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                per_rank_segment,
                _DEFAULT_LOCAL_BUFFER_SIZE,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
        if ret:
            raise RuntimeError(f"Failed to setup Mooncake store, error code: {ret}")
        self._register_host_buffer(host_buffer)
        logger.info(
            f"Mooncake Store L3 ready (protocol={self.config.protocol} "
            f"master={self.config.master_server_address} segment={per_rank_segment})"
        )

    def _register_host_buffer(self, host_buffer: Any) -> None:
        ptr = host_buffer_ptr(host_buffer)
        numel = int(host_buffer.numel())
        itemsize = (
            int(host_buffer.element_size())
            if hasattr(host_buffer, "element_size")
            else 1
        )
        size = numel * itemsize
        if size <= 0:
            return
        ret = self.store.register_buffer(ptr, size)
        if ret != 0:
            raise RuntimeError(
                f"Failed to register Host L2 buffer with Mooncake Store, error code: {ret}"
            )

    def batch_exists(self, keys: Sequence[str]) -> list[bool]:
        if not keys:
            return []
        results = self.store.batch_is_exist(list(keys))
        _require_result_len(results, len(keys), op="batch_is_exist")
        if any(int(flag) not in (0, 1) for flag in results):
            raise RuntimeError(f"Mooncake existence query failed: {results}")
        return [int(flag) == 1 for flag in results]

    def batch_get_into(self, keys, host_buffer, offsets, sizes) -> list[bool]:
        return list(
            self.batch_get_into_result(keys, host_buffer, offsets, sizes).success
        )

    def batch_get_into_result(
        self, keys, host_buffer, offsets, sizes
    ) -> StorageIOResult:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 get")
        if not keys:
            return StorageIOResult((), True, ())
        base = host_buffer_ptr(host_buffer)
        requested = [int(size) for size in sizes]
        results = self.store.batch_get_into(
            list(keys), [base + int(x) for x in offsets], requested
        )
        _require_result_len(results, len(keys), op="batch_get_into")
        return StorageIOResult(
            tuple(int(result) == size for result, size in zip(results, requested)),
            # OBJECT_NOT_FOUND is a metadata-query miss before native transfer
            # submission. Other negative errors may leave writes in flight.
            # A nonnegative short read is terminal but invalid.
            all(int(result) >= 0 or int(result) == -704 for result in results),
            tuple(
                int(result) == -704 or (int(result) >= 0 and int(result) != size)
                for result, size in zip(results, requested)
            ),
        )

    def batch_put_from(self, keys, host_buffer, offsets, sizes) -> list[bool]:
        return list(
            self.batch_put_from_result(keys, host_buffer, offsets, sizes).success
        )

    def batch_put_from_result(
        self, keys, host_buffer, offsets, sizes
    ) -> StorageIOResult:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 put")
        if not keys:
            return StorageIOResult((), True, ())
        # Metadata failures precede any native access to the source buffer.
        try:
            exists = self.batch_exists(keys)
        except Exception:
            logger.exception("Mooncake existence query before PUT failed")
            return StorageIOResult((False,) * len(keys), True, (False,) * len(keys))
        indices = [i for i, present in enumerate(exists) if not present]
        flags = list(exists)
        if not indices:
            return StorageIOResult(tuple(flags), True, (False,) * len(keys))
        base = host_buffer_ptr(host_buffer)
        results = self.store.batch_put_from(
            [keys[i] for i in indices],
            [base + int(offsets[i]) for i in indices],
            [int(sizes[i]) for i in indices],
        )
        if len(results) != len(indices):
            logger.error(
                f"Mooncake PUT returned {len(results)} results for {len(indices)} keys"
            )
            return StorageIOResult((False,) * len(keys), False, (False,) * len(keys))
        # Mooncake's public ErrorCode::OBJECT_ALREADY_EXISTS (-705) is a
        # create-only rejection before transfer. NO_AVAILABLE_HANDLE (-200)
        # is the master's allocation rejection, likewise before any source
        # transfer; capacity pressure retires as a per-page failure the
        # scheduler re-offers. Other errors do not prove retirement, even if
        # another writer's object now exists.
        for index, result in zip(indices, results):
            flags[index] = int(result) in (0, -705)
        return StorageIOResult(
            tuple(flags),
            all(int(result) in (0, -705, -200) for result in results),
            (False,) * len(keys),
        )

    def remove_by_prefix(self, prefix: str) -> bool:
        remover = getattr(self.store, "remove_by_regex", None)
        if not callable(remover):
            logger.error(
                "Installed Mooncake does not support remove_by_regex; refusing "
                "to report a successful L3 cache clear"
            )
            return False
        try:
            ret = remover(f"^{re.escape(prefix)}.*", True)
        except Exception:
            logger.exception(f"Mooncake remove_by_regex raised while clearing {prefix}")
            return False
        # Mooncake returns the number of removed objects; negative values are
        # error codes.
        if ret < 0:
            logger.error(f"Failed to clear Mooncake L3 namespace, error code: {ret}")
            return False
        return True

    def close(self) -> None:
        store = self.store
        self.store = None
        closer = getattr(store, "close", None)
        if callable(closer):
            closer()
