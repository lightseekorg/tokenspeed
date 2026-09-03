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

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from tokenspeed.runtime.engine.scheduler_utils import (
    block_tables_from_forward_op,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
    require_positive_int,
)


@dataclass(frozen=True, init=False)
class CacheBatchMetadata:
    """Factory-only views tied to one scheduler forward operation.

    Attributes:
        group_ids: Cache group IDs in runtime-contract order.
    """

    group_ids: tuple[str, ...]
    _group_tables: Mapping[str, torch.Tensor] = field(repr=False, compare=False)
    _forward_op: Any = field(repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("CacheBatchMetadata is factory-only; use from_forward_op()")

    @classmethod
    def from_forward_op(
        cls,
        forward_op: Any,
        *,
        device: torch.device | str,
        contract: CacheRuntimeContract,
        num_requests: int,
    ) -> CacheBatchMetadata:
        """Validate CPU exports, pack once, and retain operation provenance."""
        if forward_op is None:
            raise ValueError("forward_op must not be None")
        require_positive_int("num_reqs", num_requests)
        group_ids = tuple(spec.group_id for spec in contract.group_specs)
        max_page_ids = {
            group_id: require_positive_int(
                f"max page ID for {group_id!r}",
                contract.group_page_counts[group_id] - 1,
            )
            for group_id in group_ids
        }
        if (
            not group_ids
            or any(
                not isinstance(group_id, str) or not group_id for group_id in group_ids
            )
            or len(group_ids) != len(set(group_ids))
        ):
            raise ValueError(
                "runtime contract must provide ordered nonempty unique group IDs"
            )
        tables = block_tables_from_forward_op(
            forward_op,
            device,
            num_reqs=num_requests,
            expected_group_ids=group_ids,
            max_page_ids=max_page_ids,
        )
        return cls._from_validated_tables(
            group_ids=group_ids,
            group_tables=tables,
            num_requests=num_requests,
            forward_op=forward_op,
        )

    @classmethod
    def _from_validated_tables(
        cls,
        *,
        group_ids: tuple[str, ...],
        group_tables: Mapping[str, torch.Tensor],
        num_requests: int,
        forward_op: Any,
    ) -> CacheBatchMetadata:
        if tuple(group_tables) != group_ids:
            raise ValueError(
                "cache group table mapping must exactly match contract order"
            )
        table_device: torch.device | None = None
        ordered = dict(group_tables)
        for group_id, table in ordered.items():
            if not isinstance(table, torch.Tensor):
                raise ValueError(f"cache group {group_id!r} must be a tensor")
            if table.dtype != torch.int32:
                raise ValueError(f"cache group {group_id!r} must use int32")
            if table.ndim != 2 or table.shape[0] != num_requests:
                raise ValueError(f"cache group {group_id!r} has invalid shape")
            if table.shape[1] == 0:
                raise ValueError(f"cache group {group_id!r} has zero width")
            if table.device.type not in ("cpu", "cuda", "npu"):
                raise ValueError(
                    f"cache group {group_id!r} must be on CPU, CUDA, or NPU"
                )
            if table_device is None:
                table_device = table.device
            elif table.device != table_device:
                raise ValueError("cache group tables must use one CPU/CUDA/NPU device")
        nonempty = [table for table in ordered.values() if table.numel()]
        pointers = {table.untyped_storage().data_ptr() for table in nonempty}
        if len(pointers) != 1:
            raise ValueError("cache group tables must share packed storage")

        metadata = object.__new__(cls)
        object.__setattr__(metadata, "group_ids", group_ids)
        object.__setattr__(metadata, "_group_tables", MappingProxyType(ordered))
        # A strong reference makes Python/nanobind object identity safe against
        # id reuse until all metadata views become unreachable.
        object.__setattr__(metadata, "_forward_op", forward_op)
        return metadata

    def _validate_active_forward_op(self, active_forward_op: Any) -> None:
        if active_forward_op is not self._forward_op:
            raise RuntimeError(
                "stale cache metadata does not match the active forward operation"
            )

    def tables(self, *, active_forward_op: Any) -> Mapping[str, torch.Tensor]:
        """Return all immutable table views after freshness validation."""
        self._validate_active_forward_op(active_forward_op)
        return self._group_tables
