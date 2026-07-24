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
    PrefixCacheAdjunctSpec,
    RequestSpec,
    SchedulerConfig,
)

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
    """Convert a KV pool's paged_cache_group_specs to scheduler configs."""
    specs = pool.paged_cache_group_specs
    if not specs:
        return []
    counts = pool.paged_cache_group_page_counts
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
        )
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
    """
    arrays = getattr(forward_op, "flat_block_tables_arrays", None)
    if not callable(arrays):
        if getattr(forward_op, "flat_block_tables", None) is None:
            # Radix builds / idle ops carry no flat tables at all.
            return {}
        raise RuntimeError(
            "flat scheduler ext does not expose flat_block_tables_arrays; "
            "rebuild tokenspeed-scheduler (the per-element nested-list export "
            "path was removed)."
        )
    device = torch.device(device) if isinstance(device, str) else device
    out: dict[str, torch.Tensor] = {}
    packable: list[tuple[str, Any, int]] = []
    total = 0
    for key_obj, arr in arrays().items():
        key = str(key_obj)
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
