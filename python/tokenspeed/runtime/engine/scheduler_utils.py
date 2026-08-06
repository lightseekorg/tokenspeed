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
from typing import Any

import numpy as np
import torch
from tokenspeed_scheduler import (
    Cache,
    ExecutionEvent,
    ForwardEvent,
    PagedCacheGroupConfig,
    PagedCacheGroupFamily,
    PagedCacheRetention,
    PagedCacheTransferPolicy,
    RequestSpec,
    SchedulerConfig,
)

from tokenspeed.runtime.configs.cache_runtime import require_positive_int

_CACHE_EVENT_TYPES = {
    "WriteBackDoneEvent": Cache.WriteBackDoneEvent,
}
# Emitted only by the host tier. Keep the lookup guarded so an older extension
# still imports this module and fails later with a targeted compatibility error.
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
_TRANSFER_POLICY_MAP = {
    "full_suffix": PagedCacheTransferPolicy.FullSuffix,
    "latest_snapshot": PagedCacheTransferPolicy.LatestSnapshot,
}


@dataclass(frozen=True)
class SchedulerCacheGeometry:
    page_size: int
    num_device_pages: int
    num_usable_pages: int
    token_capacity: int


def scheduler_cache_geometry_from_pool(
    pool: Any,
    *,
    fallback_token_capacity: int,
    fallback_page_size: int,
) -> SchedulerCacheGeometry:
    contract = getattr(pool, "runtime_contract", None)
    num_lcm_blocks = getattr(pool, "num_lcm_blocks", None)
    if num_lcm_blocks is not None:
        num_lcm_blocks = require_positive_int("num_lcm_blocks", num_lcm_blocks)
        if contract is not None and contract.num_lcm_blocks != num_lcm_blocks:
            raise ValueError("pool and runtime contract disagree on num_lcm_blocks")
        return SchedulerCacheGeometry(
            page_size=require_positive_int(
                "contract.block_size" if contract is not None else "fallback_page_size",
                contract.block_size if contract is not None else fallback_page_size,
            ),
            # Parent 0 is reserved as the null LCM block.
            num_device_pages=num_lcm_blocks + 1,
            num_usable_pages=num_lcm_blocks,
            token_capacity=require_positive_int(
                (
                    "contract.token_capacity"
                    if contract is not None
                    else "fallback_token_capacity"
                ),
                (
                    contract.token_capacity
                    if contract is not None
                    else fallback_token_capacity
                ),
            ),
        )
    if contract is not None:
        num_lcm_blocks = require_positive_int(
            "contract.num_lcm_blocks", contract.num_lcm_blocks
        )
        return SchedulerCacheGeometry(
            page_size=contract.block_size,
            num_device_pages=num_lcm_blocks + 1,
            num_usable_pages=num_lcm_blocks,
            token_capacity=contract.token_capacity,
        )
    if fallback_page_size <= 0 or fallback_token_capacity <= 0:
        raise ValueError("fallback scheduler cache geometry must be positive")
    if fallback_token_capacity % fallback_page_size:
        raise ValueError(
            "fallback token capacity must be divisible by fallback page size"
        )
    pages = fallback_token_capacity // fallback_page_size
    return SchedulerCacheGeometry(
        page_size=fallback_page_size,
        # Ordinary pools allocate page 0 separately from their profiled,
        # usable token capacity, just like LCM pools reserve parent 0.
        num_device_pages=pages + 1,
        num_usable_pages=pages,
        token_capacity=fallback_token_capacity,
    )


def aligned_max_scheduled_tokens(
    max_scheduled_tokens: int,
    paged_cache_groups,
) -> int:
    """Floor ``max_scheduled_tokens`` to the state-snapshot grain, if any.

    Recurrent-state groups (family=State, retention=FullHistory — the C++
    ``final_state_manager`` criterion) register their state snapshot only when
    a prefill chunk ends exactly on a CacheBlock boundary
    (``RegistersAlignedFinalPageOnly``); interior boundaries never received a
    state write. A chunk size that is not a multiple of every such group's
    CacheBlock token span therefore never registers a state block. Since the
    admission probe takes the minimum hit across groups, prefix-cache reuse
    silently degrades to zero for the whole model.

    Args:
        max_scheduled_tokens: Requested per-step token budget
            (``--chunked-prefill-size``).
        paged_cache_groups: Scheduler ``PagedCacheGroupConfig`` sequence, or
            None/empty when the model declares no paged cache groups.
    Returns:
        ``max_scheduled_tokens`` floored to the LCM of the state groups'
        CacheBlock token spans. Returned unchanged when no such group exists
        or the value is already aligned.

    Raises:
        ValueError: If the configured budget is smaller than one state CacheBlock.
            Raising is safer than increasing a limit that may already have
            sized executor buffers.
    """
    require_positive_int("max_scheduled_tokens", max_scheduled_tokens)
    grain = 1
    for group in paged_cache_groups or ():
        if group.family != PagedCacheGroupFamily.State:
            continue
        if group.retention == PagedCacheRetention.SlidingWindow:
            continue
        grain = math.lcm(
            grain,
            int(group.rows_per_page) * int(group.entry_stride_tokens),
        )
    if grain == 1:
        return max_scheduled_tokens
    if max_scheduled_tokens < grain:
        raise ValueError(
            "chunked_prefill_size must be at least one recurrent-state CacheBlock: "
            f"got {max_scheduled_tokens}, minimum {grain}"
        )
    return max_scheduled_tokens - max_scheduled_tokens % grain


def make_spec(rid: str, tokens: list[int], max_new_tokens: int = 0) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = rid
    spec.tokens = tokens
    spec.max_new_tokens = max_new_tokens
    return spec


def make_config(
    num_device_pages: int,
    max_scheduled_tokens: int,
    max_batch_size: int,
    page_size: int,
    num_host_pages: int,
    disable_l2_cache: bool,
    enable_l3_storage: bool,
    role: str,
    enable_kv_cache_events: bool = False,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    disable_prefix_cache: bool = False,
    paged_cache_groups: Sequence["PagedCacheGroupConfig"] | None = None,
    enable_mixed_prefill_decode: bool = False,
) -> SchedulerConfig:
    cfg = SchedulerConfig()
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = max_batch_size
    cfg.block_size = page_size

    cfg.num_host_pages = num_host_pages
    cfg.enable_l3_storage = enable_l3_storage
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

    cfg.enable_mixed_prefill_decode = enable_mixed_prefill_decode
    if paged_cache_groups:
        cfg.paged_cache_groups = list(paged_cache_groups)
    return cfg


def pool_to_paged_cache_groups(pool: Any) -> list:
    """Convert authoritative contract specs, or legacy pool properties."""
    contract = getattr(pool, "runtime_contract", None)
    if contract is not None:
        specs = contract.group_specs
        counts = contract.group_page_counts
    else:
        specs = pool.paged_cache_group_specs
        counts = pool.paged_cache_group_page_counts
    if not specs:
        return []
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
        kwargs = dict(
            group_id=spec.group_id,
            rows_per_page=int(spec.rows_per_page),
            entry_stride_tokens=int(spec.entry_stride_tokens),
            total_pages=int(counts[spec.group_id]),
            retention=retention,
            family=family,
            cache_blocks_per_lcm_block=int(
                getattr(spec, "cache_blocks_per_lcm_block", 1)
            ),
        )
        transfer_policy = getattr(spec, "transfer_policy", None)
        if transfer_policy is not None:
            mapped_policy = _TRANSFER_POLICY_MAP.get(transfer_policy)
            if mapped_policy is None:
                raise ValueError(
                    "pool_to_paged_cache_groups: unsupported transfer policy "
                    f"{transfer_policy!r} for group {spec.group_id!r}"
                )
            kwargs["transfer_policy"] = mapped_policy
        if spec.retention == "sliding_window":
            kwargs["sliding_window_tokens"] = int(spec.sliding_window_tokens)
        out.append(PagedCacheGroupConfig(**kwargs))
    return out


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


def make_extend_result_event(
    request_id: str, tokens: Sequence[int] = ()
) -> "ForwardEvent.ExtendResult":
    fe = ForwardEvent.ExtendResult()
    fe.request_id = request_id
    fe.tokens = list(tokens)
    return fe


def make_finish_event(request_id: str) -> "ForwardEvent.Finish":
    fe = ForwardEvent.Finish()
    fe.request_id = request_id
    return fe


def make_abort_event(request_id: str) -> "ForwardEvent.Abort":
    """Finish without caching: AbortEvent skips the single-table-tree insert and
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
    }


def cache_event_from_payload(payload: dict):
    kind = payload["kind"]
    if kind not in _CACHE_EVENT_TYPES:
        raise ValueError(f"Unsupported cache event type: {kind}")
    event = _CACHE_EVENT_TYPES[kind]()
    event.op_id = int(payload["op_id"])
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
        ready_payloads.append(dict(rank_maps[0][key]))
    return ready_payloads


def cache_sync_debug_enabled() -> bool:
    value = os.getenv("TS_DEBUG_CACHE_SYNC", "")
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
    expected_group_ids: tuple[str, ...] | None = None,
    max_page_id: int | None = None,
    max_page_ids: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Bridge the per-group block tables to GPU int32 tensors: absolute
    page indices, null hole = 0 preserved, ragged-row padding -1. No
    base-offset companion -- the cache path never compacts.

    All groups stage into ONE pinned buffer and ride ONE H2D copy; the
    returned per-group views share a single storage, which is the
    precondition of the backends' one-launch packed replay fill
    (``_try_packed_group_unpack``). Per-group uploads would fail its
    same-storage check and fall back to per-group copy/fill chains
    (~40 tiny transfers per decode step).

    Args:
        forward_op: Scheduler forward operation exporting CPU NumPy tables.
        device: Destination device for the packed tensor.
        num_reqs: Optional expected row count for every group.
        expected_group_ids: Optional contract order and exact key set.
        max_page_id: Optional common inclusive upper bound for page IDs.
        max_page_ids: Optional per-group inclusive upper bounds.

    Returns:
        Per-group tensor views in ``expected_group_ids`` order when supplied,
        otherwise preserving producer order.

    Raises:
        ValueError: If strict contract validation fails before device transfer.
    """
    if max_page_id is not None and max_page_ids is not None:
        raise ValueError("pass max_page_id or max_page_ids, not both")
    strict_validation = (
        expected_group_ids is not None
        or max_page_id is not None
        or max_page_ids is not None
    )
    if strict_validation and num_reqs is not None:
        require_positive_int("num_reqs", num_reqs)
    if expected_group_ids is not None:
        seen_group_ids: set[str] = set()
        duplicate_group_ids: set[str] = set()
        for group_id in expected_group_ids:
            if group_id in seen_group_ids:
                duplicate_group_ids.add(group_id)
            seen_group_ids.add(group_id)
        if duplicate_group_ids:
            raise ValueError(
                f"expected_group_ids contains duplicates: {sorted(duplicate_group_ids)}"
            )
    array_items = [
        (str(key), arr) for key, arr in forward_op.block_tables_arrays().items()
    ]
    if strict_validation:
        normalized_group_ids: set[str] = set()
        collided_ids: set[str] = set()
        for group_id, _ in array_items:
            if group_id in normalized_group_ids:
                collided_ids.add(group_id)
            normalized_group_ids.add(group_id)
        if collided_ids:
            raise ValueError(
                "cache group keys collide after string normalization: "
                f"{sorted(collided_ids)}"
            )
    arrays_by_id = dict(array_items)
    if expected_group_ids is not None:
        actual = set(arrays_by_id)
        expected = set(expected_group_ids)
        if actual != expected:
            raise ValueError(
                f"cache group keys disagree: missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        ordered_items = [
            (group_id, arrays_by_id[group_id]) for group_id in expected_group_ids
        ]
    else:
        ordered_items = array_items
    if strict_validation:
        for group_id, arr in ordered_items:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"cache group {group_id!r} must be a NumPy array")
            if arr.dtype != np.int32:
                raise ValueError(f"cache group {group_id!r} must use int32")
            if arr.ndim != 2:
                raise ValueError(f"cache group {group_id!r} has invalid shape")
            if arr.shape[0] == 0:
                raise ValueError(f"cache group {group_id!r} has zero rows")
            # rows-vs-num_reqs is checked once in the packing loop below.
            if arr.shape[1] == 0:
                raise ValueError(f"cache group {group_id!r} has zero width")
            group_max_page_id = (
                max_page_ids.get(group_id) if max_page_ids is not None else max_page_id
            )
            if max_page_ids is not None and group_max_page_id is None:
                raise ValueError(f"max_page_ids is missing cache group {group_id!r}")
            if group_max_page_id is not None:
                invalid = (arr < -1) | (arr > group_max_page_id)
                if bool(invalid.any()):
                    raise ValueError(
                        f"cache group {group_id!r} contains a page ID outside "
                        f"-1..{group_max_page_id}"
                    )
    device = torch.device(device) if isinstance(device, str) else device
    out: dict[str, torch.Tensor] = {}
    packable: list[tuple[str, Any, int]] = []
    total = 0
    for key, arr in ordered_items:
        if num_reqs is not None and arr.shape[0] != num_reqs:
            raise ValueError(
                f"block_tables_arrays[{key}] has {arr.shape[0]} rows "
                f"but forward op reported num_reqs={num_reqs}"
            )
        if arr.shape[0] == 0:
            continue
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
