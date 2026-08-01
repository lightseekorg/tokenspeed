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
    PrefixCacheAdjunctSpec,
    RequestSpec,
    SchedulerConfig,
)

from tokenspeed.runtime.configs.flat_cache_runtime import require_positive_int

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
        num_device_pages=pages,
        num_usable_pages=pages,
        token_capacity=fallback_token_capacity,
    )


def resolve_scheduler_block_size(page_size: int, paged_cache_groups) -> int:
    """Scheduler block_size = hash-grain BASE: gcd of group block sizes, not the KV page geometry."""
    base = page_size
    for group in paged_cache_groups or ():
        gb = int(getattr(group, "block_size", 0) or 0) or page_size
        base = math.gcd(base, gb)
    return base


def aligned_max_scheduled_tokens(
    max_scheduled_tokens: int,
    paged_cache_groups,
    page_size: int,
) -> int:
    """Floor ``max_scheduled_tokens`` to the state-snapshot grain, if any.

    Recurrent-state groups (family=State, retention=FullHistory — the C++
    ``final_state_manager`` criterion) register their state snapshot only when
    a prefill chunk ends exactly on a page boundary
    (``RegistersAlignedFinalPageOnly``); interior boundaries never received a
    state write. A chunk size that is not a multiple of every such group's
    page size therefore never registers a state page, and since the admission
    probe takes the minimum hit across groups, prefix-cache reuse silently
    degrades to zero for the whole model.

    Args:
        max_scheduled_tokens: Requested per-step token budget
            (``--chunked-prefill-size``).
        paged_cache_groups: Scheduler ``PagedCacheGroupConfig`` sequence, or
            None/empty when the model declares no paged cache groups.
        page_size: Global page size in tokens; the fallback grain for groups
            whose ``block_size`` is 0 (= unset, global base).

    Returns:
        ``max_scheduled_tokens`` floored to the LCM of the state groups' page
        sizes, but never below one page (a smaller chunk could not register a
        snapshot at all). Returned unchanged when no such group exists or the
        value is already aligned.
    """
    require_positive_int("max_scheduled_tokens", max_scheduled_tokens)
    require_positive_int("page_size", page_size)
    grain = 1
    for group in paged_cache_groups or ():
        if group.family != PagedCacheGroupFamily.State:
            continue
        if group.retention == PagedCacheRetention.SlidingWindow:
            continue
        group_block = int(getattr(group, "block_size", 0) or 0) or page_size
        grain = math.lcm(grain, group_block)
    if grain == 1:
        return max_scheduled_tokens
    return max(max_scheduled_tokens - max_scheduled_tokens % grain, grain)


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
    paged_cache_host_group_pages: Mapping[str, int] | None = None,
    enable_mixed_prefill_decode: bool = False,
    prefix_cache_adjunct: "PrefixCacheAdjunctSpec | None" = None,
) -> SchedulerConfig:
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
    if paged_cache_groups:
        cfg.paged_cache_groups = list(paged_cache_groups)
    if paged_cache_host_group_pages:
        cfg.paged_cache_host_group_pages = {
            str(group_id): int(page_count)
            for group_id, page_count in paged_cache_host_group_pages.items()
        }
    # Opt-in; unset means paged-cache groups are transport-only.
    if prefix_cache_adjunct is not None:
        cfg.prefix_cache_adjunct = prefix_cache_adjunct
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
        cfg = PagedCacheGroupConfig(**kwargs)
        # Ctor default 0 = global base; a spec block_size sets the per-group granularity.
        if getattr(spec, "block_size", None):
            cfg.block_size = int(spec.block_size)
        out.append(cfg)
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


def _block_tables_from_forward_op(
    forward_op: Any,
    *,
    attr: str,
    device: "torch.device | str",
    num_reqs: int | None,
) -> dict[str, torch.Tensor]:
    raw_tables = getattr(forward_op, attr, None)
    if raw_tables is None:
        return {}
    device = torch.device(device) if isinstance(device, str) else device
    items = (
        list(raw_tables.items())
        if isinstance(raw_tables, Mapping)
        else list(raw_tables)
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
    expected_group_ids: tuple[str, ...] | None = None,
    max_page_id: int | None = None,
    max_page_ids: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Bridge the flat per-group block tables to GPU int32 tensors: absolute
    page indices, null hole = 0 preserved, ragged-row padding -1. No
    base-offset companion -- the flat path never compacts.

    All groups stage into ONE pinned buffer and ride ONE H2D copy; the
    returned per-group views share a single storage, which is the
    precondition of the backends' one-launch packed replay fill
    (``_flat_try_packed_unpack``). Per-group uploads would fail its
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
    arrays = getattr(forward_op, "flat_block_tables_arrays", None)
    if not callable(arrays):
        if getattr(forward_op, "flat_block_tables", None) is None:
            if expected_group_ids is not None:
                raise ValueError(
                    "flat group keys disagree: "
                    f"missing={sorted(expected_group_ids)} extra=[]"
                )
            # Callers without a runtime contract preserve the no-table path
            # used by idle and non-flat forward operations.
            return {}
        raise RuntimeError(
            "flat scheduler ext does not expose flat_block_tables_arrays; "
            "rebuild tokenspeed-scheduler (the per-element nested-list export "
            "path was removed)."
        )
    array_items = [(str(key), arr) for key, arr in arrays().items()]
    if strict_validation:
        normalized_group_ids: set[str] = set()
        collided_ids: set[str] = set()
        for group_id, _ in array_items:
            if group_id in normalized_group_ids:
                collided_ids.add(group_id)
            normalized_group_ids.add(group_id)
        if collided_ids:
            raise ValueError(
                "flat group keys collide after string normalization: "
                f"{sorted(collided_ids)}"
            )
    arrays_by_id = dict(array_items)
    if expected_group_ids is not None:
        actual = set(arrays_by_id)
        expected = set(expected_group_ids)
        if actual != expected:
            raise ValueError(
                f"flat group keys disagree: missing={sorted(expected - actual)} "
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
                raise ValueError(f"flat group {group_id!r} must be a NumPy array")
            if arr.dtype != np.int32:
                raise ValueError(f"flat group {group_id!r} must use int32")
            if arr.ndim != 2:
                raise ValueError(f"flat group {group_id!r} has invalid shape")
            if arr.shape[0] == 0:
                raise ValueError(f"flat group {group_id!r} has zero rows")
            # rows-vs-num_reqs is checked once in the packing loop below.
            if arr.shape[1] == 0:
                raise ValueError(f"flat group {group_id!r} has zero width")
            group_max_page_id = (
                max_page_ids.get(group_id) if max_page_ids is not None else max_page_id
            )
            if max_page_ids is not None and group_max_page_id is None:
                raise ValueError(f"max_page_ids is missing flat group {group_id!r}")
            if group_max_page_id is not None:
                invalid = (arr < -1) | (arr > group_max_page_id)
                if bool(invalid.any()):
                    raise ValueError(
                        f"flat group {group_id!r} contains a page ID outside "
                        f"-1..{group_max_page_id}"
                    )
    device = torch.device(device) if isinstance(device, str) else device
    out: dict[str, torch.Tensor] = {}
    packable: list[tuple[str, Any, int]] = []
    total = 0
    for key, arr in ordered_items:
        if num_reqs is not None and arr.shape[0] != num_reqs:
            raise ValueError(
                f"flat_block_tables_arrays[{key}] has {arr.shape[0]} rows "
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


def paged_cache_block_table_base_offsets_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Convert forward op compact-table base offsets to int32 tensors.

    Returns (gpu_offsets_per_group, cpu_max_per_group). The CPU max is captured
    before H2D so callers can size graph-replay buffers without a GPU max + D2H
    sync. Empty rows yield max=0; missing keys are absent from the max dict.
    """
    raw = getattr(forward_op, "paged_cache_block_table_base_offsets", None)
    if raw is None:
        return {}, {}
    device = torch.device(device) if isinstance(device, str) else device
    items = list(raw.items()) if isinstance(raw, Mapping) else list(raw)
    out: dict[str, torch.Tensor] = {}
    max_per_group: dict[str, int] = {}
    for key_obj, offsets in items:
        key = str(key_obj)
        rows = list(offsets)
        if num_reqs is not None and rows and len(rows) != num_reqs:
            raise ValueError(
                f"paged_cache_block_table_base_offsets[{key}] has {len(rows)} "
                f"rows but forward op reported num_reqs={num_reqs}"
            )
        if not rows:
            max_per_group[key] = 0
            continue
        max_per_group[key] = int(max(rows))
        cpu = torch.tensor(rows, dtype=torch.int32, device="cpu")
        if device.type == "cuda":
            out[key] = cpu.pin_memory().to(device, non_blocking=True)
        else:
            out[key] = cpu.to(device)
    return out, max_per_group
