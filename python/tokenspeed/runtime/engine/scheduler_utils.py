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

"""Helper functions for constructing scheduler specs and events."""

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tokenspeed_scheduler import (
    Cache,
    ExecutionEvent,
    FlatBlockPoolConfig,
    ForwardEvent,
    PagedCacheGroupConfig,
    PagedCacheGroupFamily,
    PagedCachePrefixRole,
    PagedCacheRetention,
    PagedCacheTableLayout,
    PrefixCacheAdjunctSpec,
    RequestSpec,
    SchedulerConfig,
)

from tokenspeed.runtime.configs.paged_cache_spec import require_flat_table_cols

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.flat_memory_plan import V4FlatMemoryPlan
    from tokenspeed.runtime.execution.types import FlatKVCompletion

_CACHE_EVENT_TYPES = {
    "WriteBackDoneEvent": Cache.WriteBackDoneEvent,
    "PrefetchDoneEvent": Cache.PrefetchDoneEvent,
}
# Emitted only by the flat host tier (FlatMemoryExecutor); the radix executors
# never produce it, so radix behavior is unchanged. hasattr-guarded: the flat
# tier requires a flat-built (post-C3) ext anyway, and an older radix ext must
# keep importing this module.
if hasattr(Cache, "LoadBackDoneEvent"):
    _CACHE_EVENT_TYPES["LoadBackDoneEvent"] = Cache.LoadBackDoneEvent
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

# Pool-spec string -> scheduler enum (pool_to_paged_cache_groups).
_RETENTION_MAP = {
    "full_history": PagedCacheRetention.FullHistory,
    "sliding_window": PagedCacheRetention.SlidingWindow,
}
_FAMILY_MAP = {
    "history": PagedCacheGroupFamily.History,
    "state": PagedCacheGroupFamily.State,
}
_PREFIX_ROLE_MAP = {
    "history_anchor": PagedCachePrefixRole.HistoryAnchor,
    "continuation_state": PagedCachePrefixRole.ContinuationState,
    "none": PagedCachePrefixRole.None_,
}
_TABLE_LAYOUT_MAP = {
    "absolute": PagedCacheTableLayout.Absolute,
    "bounded_window": PagedCacheTableLayout.BoundedWindow,
}


def resolve_scheduler_block_size(page_size: int, paged_cache_groups) -> int:
    """Scheduler block_size = hash-grain BASE: gcd of group block sizes, not the KV page geometry."""
    base = page_size
    for group in paged_cache_groups or ():
        gb = int(getattr(group, "block_size", 0) or 0) or page_size
        base = math.gcd(base, gb)
    return base


def make_spec(rid: str, tokens: list[int]) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = rid
    spec.tokens = tokens
    return spec


def make_config(
    num_device_pages: int,
    max_scheduled_tokens: int,
    max_batch_size: int,
    page_size: int,
    num_host_pages: int,
    disable_l2_cache: bool,
    enable_l3_storage: bool,
    prefetch_threshold: int,
    role: str,
    enable_kv_cache_events: bool = False,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    disable_prefix_cache: bool = False,
    enable_mamba: bool = False,
    mamba_cache_chunk_size: int = 64,
    mamba_pool_total_chunks: int = 0,
    enable_mamba_l2: bool = False,
    mamba_l2_host_slots: int = 0,
    paged_cache_groups: Sequence["PagedCacheGroupConfig"] | None = None,
    flat_block_pools: Sequence["FlatBlockPoolConfig"] | None = None,
    enable_mixed_prefill_decode: bool = False,
    enable_structured_flat_kv_completion: bool = False,
    prefix_cache_adjunct: "PrefixCacheAdjunctSpec | None" = None,
) -> SchedulerConfig:
    if flat_block_pools and num_device_pages != 0:
        raise ValueError(
            "explicit flat block pools require num_device_pages=0; "
            "per-pool total_blocks is the only device page authority"
        )
    cfg = SchedulerConfig()
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = max_batch_size
    cfg.block_size = resolve_scheduler_block_size(page_size, paged_cache_groups)

    cfg.num_host_pages = num_host_pages
    cfg.enable_l3_storage = enable_l3_storage
    cfg.prefetch_threshold = prefetch_threshold
    cfg.enable_kv_cache_events = enable_kv_cache_events

    if role == "prefill":
        cfg.role = SchedulerConfig.Role.P
    elif role == "decode":
        cfg.role = SchedulerConfig.Role.D
    else:
        cfg.role = SchedulerConfig.Role.Fused
    cfg.decode_input_tokens = decode_input_tokens
    cfg.overlap_schedule_depth = overlap_schedule_depth
    cfg.disable_prefix_cache = disable_prefix_cache
    cfg.disable_l2_cache = disable_l2_cache

    cfg.enable_mamba = enable_mamba
    cfg.mamba_cache_chunk_size = mamba_cache_chunk_size
    cfg.mamba_pool_total_chunks = mamba_pool_total_chunks
    cfg.enable_mamba_l2 = enable_mamba_l2
    cfg.mamba_l2_host_slots = mamba_l2_host_slots
    cfg.enable_mixed_prefill_decode = enable_mixed_prefill_decode
    cfg.enable_structured_flat_kv_completion = enable_structured_flat_kv_completion
    if paged_cache_groups:
        cfg.paged_cache_groups = list(paged_cache_groups)
    if flat_block_pools:
        cfg.flat_block_pools = list(flat_block_pools)
    # Opt-in; unset means paged-cache groups are transport-only.
    if prefix_cache_adjunct is not None:
        cfg.prefix_cache_adjunct = prefix_cache_adjunct
    return cfg


def pool_to_paged_cache_groups(pool: Any) -> list:
    """Convert the pool's scheduler-authoritative group union to configs."""
    specs = getattr(pool, "scheduler_group_specs", pool.paged_cache_group_specs)
    if not specs:
        return []
    counts = getattr(
        pool,
        "scheduler_group_page_counts",
        pool.paged_cache_group_page_counts,
    )
    out = []
    for spec in specs:
        retention = _RETENTION_MAP.get(spec.retention)
        if retention is None:
            raise ValueError(
                f"pool_to_paged_cache_groups: unsupported retention "
                f"{spec.retention!r} for group {spec.group_id!r}"
            )
        family = _FAMILY_MAP.get(spec.family)
        if family is None:
            raise ValueError(
                f"pool_to_paged_cache_groups: unsupported family "
                f"{spec.family!r} for group {spec.group_id!r}"
            )
        prefix_role = _PREFIX_ROLE_MAP.get(spec.prefix_role)
        if prefix_role is None:
            raise ValueError(
                "pool_to_paged_cache_groups: unsupported prefix_role "
                f"{spec.prefix_role!r} for group {spec.group_id!r}"
            )
        table_layout = _TABLE_LAYOUT_MAP.get(spec.table_layout)
        if table_layout is None:
            raise ValueError(
                "pool_to_paged_cache_groups: unsupported table_layout "
                f"{spec.table_layout!r} for group {spec.group_id!r}"
            )
        kwargs = dict(
            group_id=spec.group_id,
            rows_per_page=int(spec.rows_per_page),
            entry_stride_tokens=int(spec.entry_stride_tokens),
            total_pages=int(counts[spec.group_id]),
            retention=retention,
            family=family,
            block_size=int(spec.block_size_tokens),
            pool_id=str(spec.pool_id),
            prefix_role=prefix_role,
            table_layout=table_layout,
            required_producer_domain_mask=int(spec.required_producer_domain_mask),
            owner_mask=int(spec.owner_mask),
        )
        if spec.retention == "sliding_window":
            kwargs["sliding_window_tokens"] = int(spec.sliding_window_tokens)
        cfg = PagedCacheGroupConfig(**kwargs)
        # Ctor default 0 = global base; a spec block_size sets the per-group granularity.
        if getattr(spec, "block_size", None):
            cfg.block_size = int(spec.block_size)
        out.append(cfg)
    return out


def pool_to_flat_block_pools(pool: Any) -> list:
    """Convert a V4 arena plan's canonical physical pools for the scheduler.

    Legacy/radix pools publish no ``flat_memory_plan`` and therefore preserve
    the existing empty-list fallback to the scheduler's single default pool.
    """
    plan = getattr(pool, "flat_memory_plan", None)
    if plan is None:
        return []
    out = []
    for pool_plan in plan.pools:
        config = FlatBlockPoolConfig()
        config.pool_id = str(pool_plan.pool_id)
        config.total_blocks = int(pool_plan.total_blocks)
        config.bytes_per_block = int(pool_plan.bytes_per_block)
        out.append(config)
    return out


def pool_to_prefix_cache_adjunct_spec(
    required_group_ids: Sequence[str],
) -> "PrefixCacheAdjunctSpec":
    """Build a PrefixCacheAdjunctSpec from required group ids."""
    if not required_group_ids:
        raise ValueError(
            "pool_to_prefix_cache_adjunct_spec: required_group_ids must be non-empty"
        )
    spec = PrefixCacheAdjunctSpec()
    spec.required_groups = [str(gid) for gid in required_group_ids]
    return spec


def should_use_overlap_schedule(
    *,
    disable_overlap_schedule: bool,
    disaggregation_mode: str,
) -> bool:
    """Return whether the runtime can use the overlapped scheduler loop."""

    if disable_overlap_schedule:
        return False
    if disaggregation_mode in ("prefill", "encode"):
        # prefill drain + KV send run only on the non-overlap loop; encode has no LM loop.
        return False
    return True


def _bind_flat_kv_completion(
    request_id: str, completion: "FlatKVCompletion"
) -> "ForwardEvent.FlatKVCompletion":
    """Copy a validated runtime POD into explicit nanobind value objects."""
    if completion.request_id != request_id:
        raise ValueError(
            "flat KV completion request_id differs from ExtendResult: "
            f"{completion.request_id!r} != {request_id!r}"
        )

    bound_groups = []
    for group in completion.groups:
        bound_group = ForwardEvent.FlatKVGroupCompletion()
        bound_group.group_id = group.group_id
        bound_group.completed_domain_mask = group.completed_domain_mask
        bound_group.domain_valid_ends = list(group.domain_valid_ends)
        bound_groups.append(bound_group)

    bound = ForwardEvent.FlatKVCompletion()
    bound.request_id = completion.request_id
    bound.table_generation = completion.table_generation
    bound.dispatch_seq = completion.dispatch_seq
    bound.accepted_raw_end = completion.accepted_raw_end
    bound.protected_raw_end = completion.protected_raw_end
    bound.groups = bound_groups
    return bound


def make_extend_result_event(
    request_id: str,
    tokens: Sequence[int] = (),
    *,
    flat_kv_completion: "FlatKVCompletion | None" = None,
) -> "ForwardEvent.ExtendResult":
    bound_completion = (
        _bind_flat_kv_completion(request_id, flat_kv_completion)
        if flat_kv_completion is not None
        else None
    )
    fe = ForwardEvent.ExtendResult()
    fe.request_id = request_id
    fe.tokens = list(tokens)
    if bound_completion is not None:
        fe.flat_kv_completion = bound_completion
    return fe


def make_finish_event(request_id: str) -> "ForwardEvent.Finish":
    fe = ForwardEvent.Finish()
    fe.request_id = request_id
    return fe


def make_abort_event(request_id: str) -> "ForwardEvent.Abort":
    """Finish without caching: AbortEvent skips the radix-tree insert and
    never enters Draining, so no host-KV writeback (target or draft) is
    issued. Used for numerically-corrupted requests whose KV must not be
    reused.
    """
    fe = ForwardEvent.Abort()
    fe.request_id = request_id
    return fe


def make_update_reserve_tokens_event(request_id: str, new_reserve_num_tokens: int):
    fe = ForwardEvent.UpdateReserveNumTokens()
    fe.request_id = request_id
    fe.reserve_num_tokens_in_next_schedule_event = new_reserve_num_tokens
    return fe


def advance_forward(scheduler, forward_events: list) -> None:
    ec = ExecutionEvent()
    for fe in forward_events:
        ec.add_event(fe)
    scheduler.advance(ec)


def cache_event_to_payload(event) -> dict:
    kind = type(event).__name__
    if kind not in _CACHE_EVENT_TYPES:
        raise ValueError(f"Unsupported cache event type: {kind}")
    return {
        "kind": kind,
        "op_id": int(event.op_id),
        "success": bool(event.success),
        "request_id": getattr(event, "request_id", ""),
    }


def cache_event_from_payload(payload: dict):
    kind = payload["kind"]
    if kind not in _CACHE_EVENT_TYPES:
        raise ValueError(f"Unsupported cache event type: {kind}")
    event = _CACHE_EVENT_TYPES[kind]()
    event.op_id = int(payload["op_id"])
    event.success = bool(payload["success"])
    request_id = payload.get("request_id", "")
    if request_id:
        event.request_id = request_id
    return event


def cache_event_key(payload: dict) -> tuple[str, int]:
    return payload["kind"], int(payload["op_id"])


def pop_common_cache_event_payloads(
    pending_payloads_by_rank: Sequence[Sequence[dict]],
) -> list[dict]:
    if not pending_payloads_by_rank:
        return []

    rank_maps = []
    common_keys = None
    for payloads in pending_payloads_by_rank:
        rank_map = {cache_event_key(payload): payload for payload in payloads}
        rank_maps.append(rank_map)
        rank_keys = set(rank_map)
        common_keys = rank_keys if common_keys is None else common_keys & rank_keys
        if not common_keys:
            return []

    ready_payloads = []
    for key in sorted(common_keys, key=lambda item: (item[1], item[0])):
        payload = dict(rank_maps[0][key])
        payload["success"] = all(rank_map[key]["success"] for rank_map in rank_maps)
        ready_payloads.append(payload)
    return ready_payloads


def cache_sync_debug_enabled() -> bool:
    value = os.getenv("TS_DEBUG_CACHE_SYNC", "")
    return value.strip().lower() in _TRUTHY_ENV_VALUES


@dataclass(slots=True)
class _FlatBlockTableStagingSlot:
    host_tables: dict[str, torch.Tensor]
    host_bases: dict[str, torch.Tensor]
    device_tables: dict[str, torch.Tensor]
    device_bases: dict[str, torch.Tensor]


class FlatBlockTableStagingBuffers:
    """Persistent table/base H2D buffers for one flat memory plan.

    The scheduler binding writes each rectangular C++ export directly into a
    planned pinned host slot.  A matching persistent device slot receives the
    active rectangle with a non-blocking copy.  Slots rotate at the plan's
    declared forward depth, so Python neither flattens nested rows nor allocates
    pinned/device table tensors in the per-forward hot path.
    """

    def __init__(
        self, plan: "V4FlatMemoryPlan", *, device: "torch.device | str"
    ) -> None:
        self._device = torch.device(device) if isinstance(device, str) else device
        self._depth = self._positive_plan_int(plan, "forward_buffer_depth")
        self._max_rows = self._positive_plan_int(plan, "max_scheduled_batch_rows")
        group_plans = tuple(getattr(plan, "group_table_plans", ()))
        if not group_plans:
            raise ValueError("flat table staging requires planned cache groups")

        self._max_cols_by_group: dict[str, int] = {}
        for group_plan in group_plans:
            group_id = str(group_plan.group_id)
            if not group_id:
                raise ValueError("flat table staging group_id must be non-empty")
            if group_id in self._max_cols_by_group:
                raise ValueError(f"flat table staging has duplicate group {group_id!r}")
            max_cols = self._positive_plan_int(group_plan, "max_export_cols")
            self._max_cols_by_group[group_id] = max_cols
        self._group_ids = tuple(sorted(self._max_cols_by_group))

        pool_capacity_by_id: dict[str, int] = {}
        for pool in tuple(getattr(plan, "pools", ())):
            pool_id = str(pool.pool_id)
            if not pool_id or pool_id in pool_capacity_by_id:
                raise ValueError(
                    f"flat table staging has an invalid/duplicate pool {pool_id!r}"
                )
            pool_capacity_by_id[pool_id] = self._positive_plan_int(pool, "total_blocks")
        self._page_id_upper_bound_by_group: dict[str, int] = {}
        for spec in tuple(getattr(plan, "scheduler_group_specs", ())):
            group_id = str(spec.group_id)
            pool_id = str(spec.pool_id)
            if group_id in self._page_id_upper_bound_by_group:
                raise ValueError(
                    f"flat table staging has duplicate scheduler group {group_id!r}"
                )
            if pool_id not in pool_capacity_by_id:
                raise ValueError(
                    f"flat table staging group {group_id!r} refers to unknown "
                    f"pool {pool_id!r}"
                )
            self._page_id_upper_bound_by_group[group_id] = pool_capacity_by_id[pool_id]
        if set(self._page_id_upper_bound_by_group) != set(self._group_ids):
            raise ValueError(
                "flat table staging group plans and scheduler groups differ: "
                f"tables={sorted(self._group_ids)}, "
                f"scheduler={sorted(self._page_id_upper_bound_by_group)}"
            )

        planned_bytes = (
            4
            * self._depth
            * self._max_rows
            * sum(self._max_cols_by_group[group_id] + 1 for group_id in self._group_ids)
        )
        for field in ("forward_input_bytes", "cpu_forward_staging_bytes"):
            actual = getattr(plan, field, None)
            if isinstance(actual, bool) or not isinstance(actual, int):
                raise ValueError(f"flat memory plan {field} must be an integer")
            if actual != planned_bytes:
                raise RuntimeError(
                    f"flat table staging bytes disagree with plan {field}: "
                    f"shape_bytes={planned_bytes}, plan={actual}"
                )

        pin_memory = self._device.type == "cuda"
        self._slots: list[_FlatBlockTableStagingSlot] = []
        for _ in range(self._depth):
            host_tables = {
                group_id: torch.full(
                    (self._max_rows * self._max_cols_by_group[group_id],),
                    -1,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for group_id in self._group_ids
            }
            host_bases = {
                group_id: torch.zeros(
                    (self._max_rows,),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                for group_id in self._group_ids
            }
            device_tables = {
                group_id: torch.empty(
                    (self._max_rows * self._max_cols_by_group[group_id],),
                    dtype=torch.int32,
                    device=self._device,
                )
                for group_id in self._group_ids
            }
            device_bases = {
                group_id: torch.empty(
                    (self._max_rows,),
                    dtype=torch.int32,
                    device=self._device,
                )
                for group_id in self._group_ids
            }
            if pin_memory and not all(
                tensor.is_pinned()
                for tensor in (*host_tables.values(), *host_bases.values())
            ):
                raise RuntimeError(
                    "flat CUDA table staging must use pinned host tensors"
                )
            self._slots.append(
                _FlatBlockTableStagingSlot(
                    host_tables=host_tables,
                    host_bases=host_bases,
                    device_tables=device_tables,
                    device_bases=device_bases,
                )
            )

        actual_host_bytes = sum(
            self._tensor_nbytes(tensor)
            for slot in self._slots
            for tensor in (*slot.host_tables.values(), *slot.host_bases.values())
        )
        actual_device_bytes = sum(
            self._tensor_nbytes(tensor)
            for slot in self._slots
            for tensor in (*slot.device_tables.values(), *slot.device_bases.values())
        )
        if actual_host_bytes != planned_bytes or actual_device_bytes != planned_bytes:
            raise RuntimeError(
                "flat table staging allocation bytes disagree with memory plan: "
                f"host={actual_host_bytes}, device={actual_device_bytes}, "
                f"expected_each={planned_bytes}"
            )

        self._next_slot = 0
        self._current_slot: int | None = None
        self._current_rows = 0
        self._current_cols_by_group: dict[str, int] = {}

    @staticmethod
    def _positive_plan_int(owner: Any, field: str) -> int:
        value = getattr(owner, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"flat memory plan {field} must be a positive integer")
        return value

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def validate_forward_op_schema(self, forward_op: Any, *, num_reqs: int) -> None:
        """Validate the cheap group header without materializing table rows."""
        if num_reqs == 0:
            return
        group_ids_fn = getattr(forward_op, "flat_block_table_group_ids", None)
        if not callable(group_ids_fn):
            raise RuntimeError(
                "planned flat table staging requires a scheduler binding with "
                "flat_block_table_group_ids()"
            )
        actual_list = [str(group_id) for group_id in group_ids_fn()]
        if len(actual_list) != len(set(actual_list)):
            raise RuntimeError("flat forward export contains duplicate group ids")
        actual = set(actual_list)
        expected = set(self._group_ids)
        if actual != expected:
            raise RuntimeError(
                "flat forward export groups disagree with the memory plan: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

    def stage(
        self, forward_op: Any, *, num_reqs: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Copy one forward's atomic table/base union into the next ring slot."""
        if isinstance(num_reqs, bool) or not isinstance(num_reqs, int) or num_reqs < 0:
            raise ValueError("flat table staging num_reqs must be an integer >= 0")
        if num_reqs > self._max_rows:
            raise ValueError(
                "flat table staging row capacity exceeded: "
                f"rows={num_reqs}, capacity={self._max_rows}"
            )
        self.validate_forward_op_schema(forward_op, num_reqs=num_reqs)

        slot_index = self._next_slot
        slot = self._slots[slot_index]
        cols_by_group: dict[str, int] = {}
        if num_reqs == 0:
            # CUDA graph idle replay needs a real page-0 column.  The padded
            # rows are filled in-place by pad_current_for_graph().
            cols_by_group = {group_id: 1 for group_id in self._group_ids}
        else:
            copy_to = getattr(forward_op, "copy_flat_block_table_to", None)
            if not callable(copy_to):
                raise RuntimeError(
                    "planned flat table staging requires a scheduler binding "
                    "with copy_flat_block_table_to()"
                )
            for group_id in self._group_ids:
                copied = copy_to(
                    group_id,
                    slot.host_tables[group_id],
                    slot.host_bases[group_id],
                    self._page_id_upper_bound_by_group[group_id],
                )
                if not isinstance(copied, Sequence) or len(copied) != 2:
                    raise RuntimeError(
                        f"flat staging copy for {group_id!r} returned an invalid header"
                    )
                rows, cols = copied
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (rows, cols)
                ):
                    raise RuntimeError(
                        f"flat staging copy for {group_id!r} returned non-integer metadata"
                    )
                if rows != num_reqs:
                    raise RuntimeError(
                        f"flat staging copy for {group_id!r} returned {rows} "
                        f"rows, expected {num_reqs}"
                    )
                capacity = self._max_cols_by_group[group_id]
                if cols < 0 or cols > capacity:
                    raise RuntimeError(
                        f"flat staging copy for {group_id!r} returned {cols} "
                        f"columns outside [0, {capacity}]"
                    )
                cols_by_group[group_id] = cols

                active_values = num_reqs * cols
                slot.device_tables[group_id][:active_values].copy_(
                    slot.host_tables[group_id][:active_values],
                    non_blocking=self._device.type == "cuda",
                )
                slot.device_bases[group_id][:num_reqs].copy_(
                    slot.host_bases[group_id][:num_reqs],
                    non_blocking=self._device.type == "cuda",
                )

        self._current_slot = slot_index
        self._current_rows = num_reqs
        self._current_cols_by_group = cols_by_group
        self._next_slot = (slot_index + 1) % self._depth
        return self._current_device_views()

    def stage_idle(
        self, *, padded_rows: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Prepare a graph-idle page-0/base-0 view without allocating tensors.

        An idle DP rank has no ``FlatForwardOperation`` and therefore cannot use
        :meth:`stage`. Rotate the same lifetime ring as active forwards, then
        expose one persistent page-0 column per group. This is valid both before
        the first active forward and while an earlier slot remains in flight.
        """

        if (
            isinstance(padded_rows, bool)
            or not isinstance(padded_rows, int)
            or padded_rows <= 0
        ):
            raise ValueError("flat table idle padded_rows must be a positive integer")
        if padded_rows > self._max_rows:
            raise ValueError(
                "flat table idle row capacity exceeded: "
                f"rows={padded_rows}, capacity={self._max_rows}"
            )

        slot_index = self._next_slot
        slot = self._slots[slot_index]
        for group_id in self._group_ids:
            # One column is sufficient for an idle dummy row. Page/base zero is
            # the canonical non-owning mapping in every local pool.
            slot.device_tables[group_id][:padded_rows].zero_()
            slot.device_bases[group_id][:padded_rows].zero_()

        self._current_slot = slot_index
        self._current_rows = 0
        self._current_cols_by_group = {group_id: 1 for group_id in self._group_ids}
        self._next_slot = (slot_index + 1) % self._depth
        return (
            {
                group_id: slot.device_tables[group_id][:padded_rows].view(
                    padded_rows, 1
                )
                for group_id in self._group_ids
            },
            {
                group_id: slot.device_bases[group_id][:padded_rows]
                for group_id in self._group_ids
            },
        )

    def _current_device_views(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self._current_slot is None:
            raise RuntimeError("flat table staging has no current forward slot")
        slot = self._slots[self._current_slot]
        tables = {
            group_id: slot.device_tables[group_id][
                : self._current_rows * self._current_cols_by_group[group_id]
            ].view(self._current_rows, self._current_cols_by_group[group_id])
            for group_id in self._group_ids
        }
        bases = {
            group_id: slot.device_bases[group_id][: self._current_rows]
            for group_id in self._group_ids
        }
        return tables, bases

    def pad_current_for_graph(
        self,
        flat_block_tables: Mapping[str, torch.Tensor],
        flat_block_table_base_offsets: Mapping[str, torch.Tensor],
        *,
        actual_rows: int,
        padded_rows: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return padded views backed by the current persistent device slot."""
        if self._current_slot is None:
            raise RuntimeError("flat table graph padding has no staged forward")
        if actual_rows != self._current_rows:
            raise RuntimeError(
                "flat table graph padding rows disagree with staged forward: "
                f"actual={actual_rows}, staged={self._current_rows}"
            )
        if padded_rows < actual_rows or padded_rows > self._max_rows:
            raise RuntimeError(
                "flat table graph padded rows exceed the planned staging shape: "
                f"actual={actual_rows}, padded={padded_rows}, "
                f"capacity={self._max_rows}"
            )
        if set(flat_block_tables) != set(self._group_ids) or set(
            flat_block_table_base_offsets
        ) != set(self._group_ids):
            raise RuntimeError(
                "flat table graph padding did not receive the staged group union"
            )

        slot = self._slots[self._current_slot]
        for group_id in self._group_ids:
            cols = self._current_cols_by_group[group_id]
            table = flat_block_tables[group_id]
            bases = flat_block_table_base_offsets[group_id]
            expected_table_shape = (actual_rows, cols)
            if tuple(table.shape) != expected_table_shape or tuple(bases.shape) != (
                actual_rows,
            ):
                raise RuntimeError(
                    f"flat graph input shape for {group_id!r} is not the "
                    f"current staging view: table={tuple(table.shape)}, "
                    f"base={tuple(bases.shape)}, expected={expected_table_shape}"
                )
            if table.data_ptr() != slot.device_tables[group_id].data_ptr() or (
                bases.data_ptr() != slot.device_bases[group_id].data_ptr()
            ):
                raise RuntimeError(
                    f"flat graph input for {group_id!r} is not backed by the "
                    "current persistent staging slot"
                )
            if padded_rows > actual_rows:
                # Page 0/base 0 is the canonical dummy row for graph padding.
                slot.device_tables[group_id][
                    actual_rows * cols : padded_rows * cols
                ].zero_()
                slot.device_bases[group_id][actual_rows:padded_rows].zero_()

        return (
            {
                group_id: slot.device_tables[group_id][
                    : padded_rows * self._current_cols_by_group[group_id]
                ].view(padded_rows, self._current_cols_by_group[group_id])
                for group_id in self._group_ids
            },
            {
                group_id: slot.device_bases[group_id][:padded_rows]
                for group_id in self._group_ids
            },
        )


def _block_tables_from_forward_op(
    forward_op: Any,
    *,
    attr: str,
    device: "torch.device | str",
    num_reqs: int | None,
    max_cols_by_group: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    raw_tables = getattr(forward_op, attr, None)
    if raw_tables is None:
        if max_cols_by_group is not None and (num_reqs or 0) > 0:
            raise ValueError(f"{attr} is missing for a planned flat cache batch")
        return {}
    device = torch.device(device) if isinstance(device, str) else device
    items = (
        list(raw_tables.items())
        if isinstance(raw_tables, Mapping)
        else list(raw_tables)
    )
    if max_cols_by_group is not None and (num_reqs or 0) > 0:
        actual_groups = {str(key) for key, _ in items}
        planned_groups = {str(key) for key in max_cols_by_group}
        if actual_groups != planned_groups:
            raise ValueError(
                f"{attr} groups do not match the memory plan: "
                f"missing={sorted(planned_groups - actual_groups)}, "
                f"extra={sorted(actual_groups - planned_groups)}"
            )
    # One packed pinned H2D for all groups; reuse is safe — every step ends in a commit sync.
    flat_values: list[int] = []
    spans: list[tuple[str, int, int, int]] = []  # key, offset, rows, cols
    out: dict[str, torch.Tensor] = {}
    for key_obj, table in items:
        key = str(key_obj)
        rows = list(table)
        if num_reqs is not None and len(rows) != num_reqs:
            # No exemption for empty row lists: a silently dropped group
            # would hand the flat CUDA-graph replay a per-group hole.
            raise ValueError(
                f"{attr}[{key}] has {len(rows)} rows but forward op reported "
                f"num_reqs={num_reqs}"
            )
        if not rows:
            # Idle/empty op: callers treat the resulting {} as "no tables".
            continue
        max_pages = max((len(row) for row in rows), default=0)
        if max_cols_by_group is not None:
            require_flat_table_cols(
                group_id=key,
                purpose="forward export",
                actual_cols=int(max_cols_by_group[key]),
                required_cols=max_pages,
            )
        if max_pages == 0:
            out[key] = torch.empty((len(rows), 0), dtype=torch.int32, device=device)
            continue
        spans.append((key, len(flat_values), len(rows), max_pages))
        for row in rows:
            row_values = list(row)
            flat_values.extend(row_values)
            # Holes stay 0, ragged tails pad -1 (never read past cache_seqlens).
            flat_values.extend([-1] * (max_pages - len(row_values)))
    if not spans:
        return out
    total = len(flat_values)
    # Fresh (never persistent) pinned staging per step: reuse races with
    # overlap scheduling; fresh allocations are event-fenced.
    staged = torch.tensor(
        flat_values, dtype=torch.int32, pin_memory=device.type == "cuda"
    )
    dev_buf = _device_staging(attr, total, device)
    dev_buf[:total].copy_(staged, non_blocking=True)
    for key, off, rows_n, cols in spans:
        out[key] = dev_buf[off : off + rows_n * cols].view(rows_n, cols)
    return out


# Persistent device staging per forward-op attr; grows to high-water, stream-ordered.
_DEVICE_STAGING: dict[str, "torch.Tensor"] = {}


def _device_staging(key: str, numel: int, device) -> "torch.Tensor":
    buf = _DEVICE_STAGING.get(key)
    if buf is None or buf.numel() < numel or buf.device != device:
        buf = torch.zeros(max(numel, 4096), dtype=torch.int32, device=device)
        _DEVICE_STAGING[key] = buf
    return buf


def paged_cache_block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
) -> dict[str, torch.Tensor]:
    return _block_tables_from_forward_op(
        forward_op,
        attr="paged_cache_block_tables",
        device=device,
        num_reqs=num_reqs,
    )


def flat_block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
    max_cols_by_group: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Bridge flat per-group live tables to packed GPU int32 tensors.

    Null hole 0 and ragged-row padding -1 are preserved. Pair the result with
    :func:`flat_block_table_base_offsets_from_forward_op`; a row column is
    relative to that request/group's logical base.

    All groups stage into ONE pinned buffer and ride ONE H2D copy; the
    returned per-group views share a single storage, which is the
    precondition of the backends' one-launch packed replay fill
    (``_flat_try_packed_unpack``). Per-group uploads would fail its
    same-storage check and fall back to per-group copy/fill chains
    (~40 tiny transfers per decode step).
    """
    arrays = getattr(forward_op, "flat_block_tables_arrays", None)
    if not callable(arrays):
        legacy_tables = getattr(forward_op, "flat_block_tables", None)
        if legacy_tables is None:
            # Radix builds / idle ops carry no flat tables at all.
            if max_cols_by_group is not None and (num_reqs or 0) > 0:
                raise ValueError(
                    "flat_block_tables_arrays is missing for a planned flat "
                    "cache batch"
                )
            return {}
        # Preserve plan validation as a fail-closed diagnostic even when an old
        # extension only exposes the removed nested-list ABI. A valid payload
        # still fails below and requires rebuilding the scheduler extension.
        legacy_items = (
            list(legacy_tables.items())
            if isinstance(legacy_tables, Mapping)
            else list(legacy_tables)
        )
        if max_cols_by_group is not None and (num_reqs or 0) > 0:
            actual_groups = {str(key) for key, _ in legacy_items}
            planned_groups = {str(key) for key in max_cols_by_group}
            if actual_groups != planned_groups:
                raise ValueError(
                    "flat_block_tables groups do not match the memory plan: "
                    f"missing={sorted(planned_groups - actual_groups)}, "
                    f"extra={sorted(actual_groups - planned_groups)}"
                )
            for key_obj, table in legacy_items:
                key = str(key_obj)
                rows = list(table)
                required_cols = max((len(row) for row in rows), default=0)
                require_flat_table_cols(
                    group_id=key,
                    purpose="forward export",
                    actual_cols=int(max_cols_by_group[key]),
                    required_cols=required_cols,
                )
        raise RuntimeError(
            "flat scheduler ext does not expose flat_block_tables_arrays; "
            "rebuild tokenspeed-scheduler (the per-element nested-list export "
            "path was removed)."
        )
    device = torch.device(device) if isinstance(device, str) else device
    out: dict[str, torch.Tensor] = {}
    packable: list[tuple[str, Any, int]] = []
    total = 0
    array_items = list(arrays().items())
    if max_cols_by_group is not None and (num_reqs or 0) > 0:
        actual_groups = {str(key) for key, _ in array_items}
        planned_groups = {str(key) for key in max_cols_by_group}
        if actual_groups != planned_groups:
            raise ValueError(
                "flat_block_tables_arrays groups do not match the memory plan: "
                f"missing={sorted(planned_groups - actual_groups)}, "
                f"extra={sorted(actual_groups - planned_groups)}"
            )
    for key_obj, arr in array_items:
        key = str(key_obj)
        if num_reqs is not None and arr.shape[0] != num_reqs:
            raise ValueError(
                f"flat_block_tables_arrays[{key}] has {arr.shape[0]} rows "
                f"but forward op reported num_reqs={num_reqs}"
            )
        if arr.shape[0] == 0:
            continue
        if max_cols_by_group is not None:
            require_flat_table_cols(
                group_id=key,
                purpose="forward export",
                actual_cols=int(max_cols_by_group[key]),
                required_cols=int(arr.shape[1]),
            )
        if arr.shape[1] == 0:
            # Kept out of the pack: a zero-width table must stay loud in the
            # replay fill's cols >= 1 assert, not be silently tail-padded.
            out[key] = torch.empty((arr.shape[0], 0), dtype=torch.int32, device=device)
            continue
        packable.append((key, arr, total))
        total += arr.shape[0] * arr.shape[1]
    if not packable:
        return out
    # Fresh pinned stage per step (event-fenced; reuse races overlap).
    # arr is a read-only zero-copy view over the C++ buffer; np.copyto
    # reads it into our own writable pinned tensor (never writes back).
    staged = torch.empty(total, dtype=torch.int32, pin_memory=device.type == "cuda")
    staged_np = staged.numpy()
    for key, arr, offset in packable:
        np.copyto(staged_np[offset : offset + arr.size].reshape(arr.shape), arr)
    packed = staged.to(device, non_blocking=True)
    for key, arr, offset in packable:
        out[key] = packed[offset : offset + arr.size].view(arr.shape[0], arr.shape[1])
    return out


def _block_table_base_offsets_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    attr: str,
    num_reqs: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    raw = getattr(forward_op, attr, None)
    if raw is None:
        return {}, {}
    device = torch.device(device) if isinstance(device, str) else device
    items = list(raw.items()) if isinstance(raw, Mapping) else list(raw)
    out: dict[str, torch.Tensor] = {}
    max_per_group: dict[str, int] = {}
    for key_obj, offsets in items:
        key = str(key_obj)
        rows = list(offsets)
        if num_reqs is not None and len(rows) != num_reqs:
            raise ValueError(
                f"{attr}[{key}] has {len(rows)} "
                f"rows but forward op reported num_reqs={num_reqs}"
            )
        if not rows:
            max_per_group[key] = 0
            continue
        if any(int(offset) < 0 for offset in rows):
            raise ValueError(f"{attr}[{key}] contains a negative logical base")
        max_per_group[key] = int(max(rows))
        cpu = torch.tensor(rows, dtype=torch.int32, device="cpu")
        if device.type == "cuda":
            out[key] = cpu.pin_memory().to(device, non_blocking=True)
        else:
            out[key] = cpu.to(device)
    return out, max_per_group


def paged_cache_block_table_base_offsets_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Convert radix compact-table bases; missing full-history keys mean zero."""
    return _block_table_base_offsets_from_forward_op(
        forward_op,
        attr="paged_cache_block_table_base_offsets",
        device=device,
        num_reqs=num_reqs,
    )


def flat_block_table_base_offsets_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Convert flat bases; every delivered flat group must have a key."""
    return _block_table_base_offsets_from_forward_op(
        forward_op,
        attr="flat_block_table_base_offsets",
        device=device,
        num_reqs=num_reqs,
    )
