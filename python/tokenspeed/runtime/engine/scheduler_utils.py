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

import os
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from tokenspeed_scheduler import (
    Cache,
    ExecutionEvent,
    ForwardEvent,
    PagedCacheGroupConfig,
    PagedCacheRetention,
    RequestSpec,
    SchedulerConfig,
)

_CACHE_EVENT_TYPES = {
    "WriteBackDoneEvent": Cache.WriteBackDoneEvent,
    "PrefetchDoneEvent": Cache.PrefetchDoneEvent,
}
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


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
    decode_input_tokens: int = 1,
    disable_prefix_cache: bool = False,
    enable_mamba: bool = False,
    mamba_cache_chunk_size: int = 64,
    mamba_pool_total_chunks: int = 0,
    paged_cache_groups: Sequence["PagedCacheGroupConfig"] | None = None,
    enable_mixed_prefill_decode: bool | None = None,
) -> SchedulerConfig:
    cfg = SchedulerConfig()
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = max_batch_size
    cfg.page_size = page_size

    cfg.num_host_pages = num_host_pages
    cfg.enable_l3_storage = enable_l3_storage
    cfg.prefetch_threshold = prefetch_threshold

    if role == "prefill":
        cfg.role = SchedulerConfig.Role.P
    elif role == "decode":
        cfg.role = SchedulerConfig.Role.D
    else:
        cfg.role = SchedulerConfig.Role.Fused
    cfg.num_device_pages = num_device_pages
    cfg.decode_input_tokens = decode_input_tokens
    cfg.disable_prefix_cache = disable_prefix_cache
    cfg.disable_l2_cache = disable_l2_cache

    cfg.enable_mamba = enable_mamba
    cfg.mamba_cache_chunk_size = mamba_cache_chunk_size
    cfg.mamba_pool_total_chunks = mamba_pool_total_chunks
    if enable_mixed_prefill_decode is None:
        enable_mixed_prefill_decode = (
            os.getenv("TOKENSPEED_ENABLE_MIXED_PREFILL_DECODE", "").strip().lower()
            in _TRUTHY_ENV_VALUES
        )
    cfg.enable_mixed_prefill_decode = bool(enable_mixed_prefill_decode)
    if paged_cache_groups:
        cfg.paged_cache_groups = list(paged_cache_groups)
    return cfg


def pool_to_paged_cache_groups(pool: Any) -> list:
    """Convert a KV pool's paged_cache_group_specs to scheduler configs."""
    specs = getattr(pool, "paged_cache_group_specs", ())
    if not specs:
        return []
    counts = pool.paged_cache_group_page_counts
    out = []
    for spec in specs:
        if spec.retention == "full_history":
            retention = PagedCacheRetention.FullHistory
        elif spec.retention == "sliding_window":
            retention = PagedCacheRetention.SlidingWindow
        else:
            raise ValueError(
                f"pool_to_paged_cache_groups: unsupported retention "
                f"{spec.retention!r} for group {spec.group_id!r}"
            )
        kwargs = dict(
            group_id=spec.group_id,
            rows_per_page=int(spec.rows_per_page),
            entry_stride_tokens=int(spec.entry_stride_tokens),
            total_pages=int(counts[spec.group_id]),
            retention=retention,
        )
        if spec.retention == "sliding_window":
            kwargs["sliding_window_tokens"] = int(spec.sliding_window_tokens)
        out.append(PagedCacheGroupConfig(**kwargs))
    return out


def make_extend_result_event(request_id: str, tokens: list[int] = ()) -> None:
    fe = ForwardEvent.ExtendResult()
    fe.request_id = request_id
    fe.tokens = list(tokens)
    return fe


def make_finish_event(request_id: str) -> None:
    fe = ForwardEvent.Finish()
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
    out: dict[str, torch.Tensor] = {}
    for key_obj, table in items:
        key = str(key_obj)
        rows = list(table)
        if num_reqs is not None and rows and len(rows) != num_reqs:
            raise ValueError(
                f"{attr}[{key}] has {len(rows)} rows but forward op reported "
                f"num_reqs={num_reqs}"
            )
        if not rows:
            continue
        max_pages = max((len(row) for row in rows), default=0)
        if max_pages == 0:
            out[key] = torch.empty((len(rows), 0), dtype=torch.int32, device=device)
            continue
        flat = torch.full(
            (len(rows), max_pages),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        for row_idx, row in enumerate(rows):
            row_len = len(row)
            if row_len:
                flat[row_idx, :row_len] = torch.as_tensor(list(row), dtype=torch.int32)
        out[key] = flat.to(device, non_blocking=True)
    return out


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


def paged_cache_block_table_base_offsets_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Convert forward op compact-table base offsets to int32 tensors."""
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
