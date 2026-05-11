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

from tokenspeed_scheduler import (
    Cache,
    ExecutionEvent,
    ForwardEvent,
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
    enable_kv_cache_events: bool = False,
    decode_input_tokens: int = 1,
    num_mamba_slots: int = 0,
) -> SchedulerConfig:
    cfg = SchedulerConfig()
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = max_batch_size
    cfg.page_size = page_size

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
    cfg.num_device_pages = num_device_pages
    cfg.decode_input_tokens = decode_input_tokens
    cfg.disable_l2_cache = disable_l2_cache
    cfg.num_mamba_slots = num_mamba_slots
    return cfg


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
