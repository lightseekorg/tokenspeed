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
from tokenspeed.runtime.layers.attention.page_table import expand_page_table


@dataclass(frozen=True, init=False)
class CacheBatchMetadata:
    """Factory-only views tied to one scheduler forward operation.

    Attributes:
        group_ids: Cache group IDs in runtime-contract order.
        num_requests: Number of request rows in each group table.
        max_page_ids: Inclusive maximum page ID accepted for each group.
        block_granularity: Full-history table grain in tokens (equals the
            contract prefix_granularity by the 1:1 convention).
        full_attention_group_id: The unique ``family="history"`` +
            ``retention="full_history"`` group ID, or ``None`` when the
            contract does not contain exactly one such group.
    """

    group_ids: tuple[str, ...]
    _group_tables: Mapping[str, torch.Tensor] = field(repr=False, compare=False)
    num_requests: int
    max_page_ids: Mapping[str, int]
    # Grain of the scheduler's full-history table (equals prefix_granularity
    # by the 1:1 prefix-page <-> CacheBlock convention).
    block_granularity: int
    full_attention_group_id: str | None
    _forward_op: Any = field(repr=False, compare=False)
    # Kernel-page expansions memoized per (group_id, kernel_page_size, max_pages);
    # metadata lives exactly one forward operation, so entries never go stale.
    _kernel_tables: dict = field(repr=False, compare=False)

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
        block_granularity = require_positive_int(
            "contract prefix_granularity", contract.prefix_granularity
        )
        full_attention_ids = tuple(
            spec.group_id
            for spec in contract.group_specs
            if spec.family == "history" and spec.retention == "full_history"
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
            max_page_ids=max_page_ids,
            block_granularity=block_granularity,
            full_attention_group_id=(
                full_attention_ids[0] if len(full_attention_ids) == 1 else None
            ),
            forward_op=forward_op,
        )

    @classmethod
    def _from_validated_tables(
        cls,
        *,
        group_ids: tuple[str, ...],
        group_tables: Mapping[str, torch.Tensor],
        num_requests: int,
        max_page_ids: Mapping[str, int],
        block_granularity: int,
        full_attention_group_id: str | None,
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
            if table.device.type not in ("cpu", "cuda"):
                raise ValueError(f"cache group {group_id!r} must be on CPU or CUDA")
            if table_device is None:
                table_device = table.device
            elif table.device != table_device:
                raise ValueError("cache group tables must use one CPU/CUDA device")
        nonempty = [table for table in ordered.values() if table.numel()]
        pointers = {table.untyped_storage().data_ptr() for table in nonempty}
        if len(pointers) != 1:
            raise ValueError("cache group tables must share packed storage")

        metadata = object.__new__(cls)
        object.__setattr__(metadata, "group_ids", group_ids)
        object.__setattr__(metadata, "_group_tables", MappingProxyType(ordered))
        object.__setattr__(metadata, "num_requests", num_requests)
        object.__setattr__(
            metadata, "max_page_ids", MappingProxyType(dict(max_page_ids))
        )
        object.__setattr__(metadata, "block_granularity", block_granularity)
        object.__setattr__(metadata, "full_attention_group_id", full_attention_group_id)
        # A strong reference makes Python/nanobind object identity safe against
        # id reuse until all metadata views become unreachable.
        object.__setattr__(metadata, "_forward_op", forward_op)
        object.__setattr__(metadata, "_kernel_tables", {})
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

    def require_table(
        self,
        group_id: str,
        *,
        active_forward_op: Any,
    ) -> torch.Tensor:
        """Return one required table after freshness validation."""
        self._validate_active_forward_op(active_forward_op)
        try:
            return self._group_tables[group_id]
        except KeyError:
            raise KeyError(f"missing cache group {group_id!r}") from None

    def require_full_attention_table(self, *, active_forward_op: Any) -> torch.Tensor:
        """Return the unique full-history history-group table.

        Args:
            active_forward_op: The scheduler forward operation this batch is
                executing; must be the operation the metadata was built from.

        Returns:
            The ``[num_requests, max_pages]`` int32 page table of the single
            ``family="history"``, ``retention="full_history"`` group.

        Raises:
            RuntimeError: If the metadata is stale, or the contract does not
                contain exactly one full-attention history group.
        """
        self._validate_active_forward_op(active_forward_op)
        if self.full_attention_group_id is None:
            raise RuntimeError(
                "runtime contract does not define exactly one full-history "
                "history group; the MLA cache path requires it"
            )
        return self.require_table(
            self.full_attention_group_id,
            active_forward_op=active_forward_op,
        )

    def kernel_table(
        self,
        group_id: str | None = None,
        *,
        kernel_page_size: int,
        max_pages: int | None = None,
        active_forward_op: Any,
    ) -> torch.Tensor:
        """Return a group's table expanded into a backend's kernel pages.

        Scheduler tables address prefix pages of ``block_granularity`` tokens; a
        backend's kernel walks pages of ``kernel_page_size`` tokens. This expands
        page ids once per (group, kernel_page_size, max_pages) and memoizes the
        result on the metadata, so every consumer of the same geometry within
        one forward (eager init, graph replay, chunked prefill) shares one
        expansion. The token-location invariant holds by construction:
        ``expanded[i, t // p] * p + t % p == table[i, t // P] * P + t % P``
        for every token position ``t``, so write-location math on the
        expanded table with ``kernel_page_size`` is exact.

        Args:
            group_id: Cache group to expand; ``None`` selects the unique
                full-attention history group.
            kernel_page_size: The consuming kernel's page size in tokens. Must
                divide ``block_granularity``.
            max_pages: Width of the returned table in kernel pages. ``None``
                keeps the full expanded width. The returned tensor is padded
                with the null page 0 past the source width.
            active_forward_op: Freshness token; must be the forward operation
                the metadata was built from.

        Returns:
            ``[num_requests, max_pages]`` int32 tensor of kernel page ids.
            -1 holes clamp into the null page's kernel range (page 0 spans
            kernel pages ``0..ratio-1``, all inside the physical null page).

        Raises:
            RuntimeError: If the metadata is stale or ``block_granularity`` is not a
                positive multiple of ``kernel_page_size``.
        """
        if group_id is None:
            table = self.require_full_attention_table(
                active_forward_op=active_forward_op
            )
            group_id = self.full_attention_group_id
        else:
            table = self.require_table(group_id, active_forward_op=active_forward_op)
        if self.block_granularity % kernel_page_size:
            raise RuntimeError(
                f"block granularity {self.block_granularity} is not a positive "
                f"multiple of the kernel page size {kernel_page_size}"
            )
        key = (group_id, int(kernel_page_size), max_pages)
        cached = self._kernel_tables.get(key)
        if cached is not None:
            return cached
        if table.stride(0) != table.shape[1] and table.shape[0] > 1:
            # Chunked-prefill kernels derive the row stride from shape[1].
            table = table.contiguous()
        if self.block_granularity == kernel_page_size and (
            max_pages is None or max_pages <= table.shape[1]
        ):
            expanded = table if max_pages is None else table[:, :max_pages]
        else:
            expanded = expand_page_table(
                table,
                block_granularity=self.block_granularity,
                kernel_page_size=kernel_page_size,
                max_kernel_pages=max_pages,
            )
        self._kernel_tables[key] = expanded
        return expanded

    def validate_live_pages(
        self, seq_lens: torch.Tensor, *, active_forward_op: Any
    ) -> None:
        """Debug check (GPU sync): no -1 hole or null page inside a live range.

        Validates the LOGICAL full-attention table once per forward; kernel
        expansions derived from a valid logical table are valid by
        construction, so backends need no per-expansion re-check.
        """
        table = self.require_full_attention_table(active_forward_op=active_forward_op)
        if table.numel() == 0 or seq_lens.numel() == 0:
            return
        batch_size = min(int(seq_lens.shape[0]), int(table.shape[0]))
        live_pages = (
            (seq_lens[:batch_size].to(torch.int64) + self.block_granularity - 1)
            // self.block_granularity
        ).clamp_max_(table.shape[1])
        columns = torch.arange(table.shape[1], device=table.device)
        live_entries = table[:batch_size][
            columns.unsqueeze(0) < live_pages.unsqueeze(1)
        ]
        if live_entries.numel() and not bool((live_entries > 0).all().item()):
            raise RuntimeError(
                "full-attention table contains -1 or the null page 0 "
                "inside a live range"
            )
