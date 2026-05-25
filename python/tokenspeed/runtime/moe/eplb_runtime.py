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

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import torch

from tokenspeed.runtime.moe import eplb_logger
from tokenspeed.runtime.moe.eplb_host_cache import HostWeightCache, _iter_moe_layers
from tokenspeed.runtime.moe.eplb_layer_state import SingleLayerEplbState
from tokenspeed.runtime.moe.eplb_planner import (
    EplbPlanResult,
)
from tokenspeed.runtime.moe.eplb_planner import _balancedness as _logical_balancedness
from tokenspeed.runtime.moe.eplb_planner import (
    run_planner_with_broadcast,
)
from tokenspeed.runtime.moe.eplb_relocator import WeightRelocator
from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

_GLOBAL_EPLB_RUNTIME: "EplbRuntime | None" = None


def set_global_eplb_runtime(runtime: "EplbRuntime | None") -> None:
    global _GLOBAL_EPLB_RUNTIME
    _GLOBAL_EPLB_RUNTIME = runtime


def get_global_eplb_runtime() -> "EplbRuntime | None":
    return _GLOBAL_EPLB_RUNTIME


def get_dispatch_info(layer_idx: int):
    runtime = get_global_eplb_runtime()
    if runtime is None:
        return None
    return runtime.get_dispatch_info(layer_idx)


_EVENT_FIELD_ALIASES = {
    "PlanDone": {"layer_ids": "layers_changed", "balancedness": "balancedness_pred"},
}


def _make_event(name: str, **kwargs):
    try:
        from tokenspeed_scheduler import EPLB

        cls = getattr(EPLB, name)
        event = cls()
        binding_event = True
    except Exception:
        event = type(name, (), {})()
        binding_event = False

    aliases = _EVENT_FIELD_ALIASES.get(name, {})
    for key, value in kwargs.items():
        target = aliases.get(key, key)
        try:
            setattr(event, target, value)
        except AttributeError:
            if not binding_event:
                setattr(event, key, value)
    return event


def _type_name(obj) -> str:
    return type(obj).__name__


def _getattr(obj, name: str, default=None):
    return getattr(obj, name, default)


class EplbRuntime:
    def __init__(
        self,
        *,
        server_args,
        model_config,
        initial_metadata,
        recorder,
        rank: int,
        ep_size: int,
        event_sink: Callable[[object], None] | None = None,
        eplb_pg=None,
        eplb_control_pg=None,
    ):
        self.server_args = server_args
        self.model_config = model_config
        self.metadata = initial_metadata
        self.recorder = recorder
        self.rank = int(rank)
        self.ep_size = int(ep_size)
        self.eplb_pg = eplb_pg
        self.eplb_control_pg = eplb_control_pg
        self.layer_states: dict[int, SingleLayerEplbState] = {}
        self._event_sink = event_sink
        self._event_queue: deque[object] = deque()
        self._stats_handles: dict[int, torch.Tensor] = {}
        self._plan_handles: dict[int, object] = {}
        self._planner_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="eplb"
        )
        self._planner_futures: dict[int, Future] = {}
        self._planner_contexts: dict[int, tuple[int, torch.Tensor]] = {}
        self._relocate_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="eplb_relocate"
        )
        self._relocate_futures: dict[int, tuple[Future, int, list[int]]] = {}
        self._relocate_contexts: dict[int, dict[str, object]] = {}
        self._metric_history = {size: deque(maxlen=size) for size in (10, 100, 1000)}
        self._host_cache: HostWeightCache | None = None
        self._relocator: WeightRelocator | None = None
        persist_dir = getattr(server_args, "eplb_persist_dir", None)
        self._persist_dir = Path(persist_dir) if persist_dir else None
        eplb_logger.set_rank(rank)

    def bind_to_model(self, model, *, build_host_cache: bool = True) -> None:
        include_drafter = bool(getattr(self.server_args, "eplb_include_drafter", False))
        for layer in _iter_moe_layers(model):
            if (not include_drafter) and str(getattr(layer, "prefix", "")).startswith(
                "drafter"
            ):
                continue
            layer_idx = int(layer.layer_index)
            device = next(layer.parameters()).device
            dispatch_info = None
            if hasattr(self.metadata, "dispatch_info_for_layer"):
                dispatch_info = self.metadata.dispatch_info_for_layer(layer_idx)
            self.layer_states[layer_idx] = SingleLayerEplbState(
                layer_index=layer_idx,
                num_local=int(layer.num_local_experts),
                num_experts=int(layer.num_experts),
                device=device,
                dispatch_info=dispatch_info,
            )
        if build_host_cache:
            share_host_cache = (
                self.ep_size > 1
                and getattr(self.server_args, "eplb_relocate_strategy", "host_first")
                == "host_first"
            )
            self.bind_host_cache(
                HostWeightCache.from_model(
                    model,
                    self.server_args,
                    shared_pg=self.eplb_control_pg,
                    rank=self.rank,
                    ep_size=self.ep_size,
                    share_across_ep=share_host_cache,
                )
            )
        set_global_eplb_runtime(self)

    def bind_host_cache(self, host_cache: HostWeightCache) -> None:
        self._host_cache = host_cache
        self._relocator = WeightRelocator(
            host_cache,
            self.layer_states,
            rank=self.rank,
            ep_size=self.ep_size,
            strategy=getattr(self.server_args, "eplb_relocate_strategy", "host_first"),
            eplb_pg=self.eplb_pg,
        )

    def _persist(self, kind: str, payload: dict) -> None:
        if self._persist_dir is None:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        path = self._persist_dir / f"{time.time_ns()}_{kind}.pt"
        torch.save(payload, path)

    def get_dispatch_info(self, layer_idx: int):
        state = self.layer_states.get(int(layer_idx))
        if state is not None and state.dispatch_info is not None:
            return state.dispatch_info
        if hasattr(self.metadata, "dispatch_info_for_layer"):
            return self.metadata.dispatch_info_for_layer(int(layer_idx))
        return None

    def handle(self, op) -> None:
        name = _type_name(op)
        if "CollectStats" in name:
            self._on_collect(op)
        elif "Plan" in name:
            self._on_plan(op)
        elif "Relocate" in name:
            self._on_relocate(op)
        elif "Swap" in name:
            self._on_swap(op)
        else:
            raise NotImplementedError(f"Unknown EPLB operation {name}")

    def drain_events(self) -> list[object]:
        self._drain_planner_futures()
        self._drain_relocate_futures()
        events = list(self._event_queue)
        self._event_queue.clear()
        return events

    def _emit(self, event) -> None:
        if self._event_sink is not None:
            self._event_sink(event)
        else:
            self._event_queue.append(event)

    def _log_collect_metrics(
        self, op_id: int, logical_count: torch.Tensor, total_count: int
    ) -> None:
        if self.rank != 0:
            return
        if not getattr(self.server_args, "enable_expert_distribution_metrics", False):
            return
        balancedness = _logical_balancedness(logical_count)
        for history in self._metric_history.values():
            history.append(balancedness)
        history_msg = "".join(
            f"last_{size}_average_balancedness={sum(history) / len(history):.03f} "
            for size, history in self._metric_history.items()
            if history
        )
        eplb_logger.debug(
            "[EPLB Balancedness] collect_op_id=%s global_balancedness=%.03f %sglobal_logical_count_sum=%s",
            op_id,
            balancedness,
            history_msg,
            total_count,
        )

    def _on_collect(self, op) -> None:
        op_id = int(_getattr(op, "op_id", 0))
        logical_count = self.recorder.snapshot_logical_count(reset=True)
        if logical_count is None or logical_count.numel() == 0:
            self._emit(_make_event("StatsEmpty", op_id=op_id))
            return
        logical_count = logical_count.detach().cpu()
        total_count = int(logical_count.sum().item())
        if total_count == 0:
            self._emit(_make_event("StatsEmpty", op_id=op_id))
            return
        self._stats_handles[op_id] = logical_count
        self._log_collect_metrics(op_id, logical_count, total_count)
        self._emit(
            _make_event(
                "StatsCollected",
                op_id=op_id,
                stats_handle=op_id,
                total_count=total_count,
            )
        )

    def _on_plan(self, op) -> None:
        op_id = int(_getattr(op, "op_id", 0))
        stats_handle = int(_getattr(op, "stats_handle", op_id))
        logical_count = self._stats_handles.pop(stats_handle, None)
        if logical_count is None:
            self._emit(
                _make_event(
                    "PlanFailed",
                    op_id=op_id,
                    reason=f"missing stats handle {stats_handle}",
                )
            )
            return
        self._planner_contexts[op_id] = (stats_handle, logical_count)
        self._planner_futures[op_id] = self._planner_pool.submit(
            self._run_plan, op_id, stats_handle, logical_count
        )

    def _run_plan(self, op_id: int, stats_handle: int, logical_count: torch.Tensor):
        try:
            result = run_planner_with_broadcast(
                logical_count,
                rank=self.rank,
                current_metadata=self.metadata,
                server_args=self.server_args,
                model_config=self.model_config,
                eplb_pg=self.eplb_pg,
                eplb_control_pg=self.eplb_control_pg,
            )
            return result
        except Exception as exc:
            return _make_event(
                "PlanFailed",
                op_id=op_id,
                reason=str(exc),
            )

    def _finish_plan_result(
        self,
        op_id: int,
        stats_handle: int,
        logical_count: torch.Tensor,
        result: EplbPlanResult,
    ):
        if self.rank == 0 and getattr(
            self.server_args, "enable_expert_distribution_metrics", False
        ):
            eplb_logger.debug(
                "[EPLB Plan] plan_op_id=%s stats_handle=%s changed_layers=%s stats_balancedness=%.03f",
                op_id,
                stats_handle,
                result.changed_layers,
                result.balancedness,
            )
        if not result.changed_layers:
            return _make_event(
                "PlanIdentical",
                op_id=op_id,
                balancedness=result.balancedness,
            )
        new_metadata = ExpertLocationMetadata.init_by_mapping(
            self.server_args,
            self.model_config,
            result.physical_to_logical_map_cpu,
        )
        self._plan_handles[op_id] = new_metadata
        self._persist(
            "plan",
            {
                "op_id": op_id,
                "stats_handle": stats_handle,
                "layers_changed": result.changed_layers,
                "logical_count": logical_count.detach().cpu(),
                "physical_to_logical_map": result.physical_to_logical_map_cpu.clone(),
                "balancedness": result.balancedness,
            },
        )
        return _make_event(
            "PlanDone",
            op_id=op_id,
            plan_handle=op_id,
            layers_changed=result.changed_layers,
            balancedness_before=result.balancedness,
            balancedness_pred=result.balancedness,
        )

    def _run_relocate(
        self, relocator, old_metadata, new_metadata, layer_ids: list[int]
    ):
        ready_layers = list(relocator.submit(old_metadata, new_metadata, layer_ids))
        for layer_id in ready_layers or layer_ids:
            state = self.layer_states.get(int(layer_id))
            if state is not None:
                state.relocate_done_event.synchronize()
        return ready_layers

    def _drain_relocate_futures(self) -> None:
        for op_id, item in list(self._relocate_futures.items()):
            future, plan_handle, layer_ids = item
            if not future.done():
                continue
            self._relocate_futures.pop(op_id)
            context = self._relocate_contexts.pop(op_id, {})
            started_at = context.get("started_at")
            wall_ms = None
            if isinstance(started_at, float):
                wall_ms = (time.perf_counter() - started_at) * 1000.0
            try:
                ready_layers = list(future.result())
                event_layers = ready_layers or layer_ids
                eplb_logger.debug(
                    "[EPLB Relocate] phase=done op_id=%s plan_handle=%s layers=%s "
                    "ready_layers=%s wall_ms=%.3f transfer_entries=%s local_dst_entries=%s "
                    "local_src_entries=%s expected_h2d_bytes=%s",
                    op_id,
                    plan_handle,
                    layer_ids,
                    event_layers,
                    wall_ms if wall_ms is not None else -1.0,
                    context.get("transfer_entries", "unknown"),
                    context.get("local_dst_entries", "unknown"),
                    context.get("local_src_entries", "unknown"),
                    context.get("expected_h2d_bytes", "unknown"),
                )
                self._emit(
                    _make_event(
                        "RelocateDone",
                        op_id=op_id,
                        plan_handle=plan_handle,
                        layer_ids=event_layers,
                    )
                )
            except Exception as exc:
                eplb_logger.error(
                    "[EPLB Relocate] phase=fail op_id=%s plan_handle=%s layers=%s "
                    "wall_ms=%.3f reason=%s",
                    op_id,
                    plan_handle,
                    layer_ids,
                    wall_ms if wall_ms is not None else -1.0,
                    str(exc),
                )
                self._emit(
                    _make_event(
                        "RelocateFailed",
                        op_id=op_id,
                        layer_id=layer_ids[0] if layer_ids else -1,
                        reason=str(exc),
                    )
                )

    def _drain_planner_futures(self) -> None:
        for op_id, future in list(self._planner_futures.items()):
            if not future.done():
                continue
            self._planner_futures.pop(op_id)
            stats_handle, logical_count = self._planner_contexts.pop(
                op_id, (op_id, torch.empty(0))
            )
            result = future.result()
            if isinstance(result, EplbPlanResult):
                try:
                    result = self._finish_plan_result(
                        op_id, stats_handle, logical_count, result
                    )
                except Exception as exc:
                    result = _make_event(
                        "PlanFailed",
                        op_id=op_id,
                        reason=str(exc),
                    )
            self._emit(result)

    def _on_relocate(self, op) -> None:
        op_id = int(_getattr(op, "op_id", 0))
        plan_handle = int(_getattr(op, "plan_handle", op_id))
        layer_ids = [int(x) for x in list(_getattr(op, "layer_ids", []))]
        new_metadata = self._plan_handles.get(plan_handle)
        if new_metadata is None:
            self._emit(
                _make_event(
                    "RelocateFailed",
                    op_id=op_id,
                    layer_id=layer_ids[0] if layer_ids else -1,
                    reason=f"missing plan handle {plan_handle}",
                )
            )
            return
        if self._relocator is None:
            self._emit(
                _make_event(
                    "RelocateFailed",
                    op_id=op_id,
                    layer_id=layer_ids[0] if layer_ids else -1,
                    reason="relocator is not bound to a model",
                )
            )
            return
        try:
            transfer_stats: dict[str, object] = {
                "transfer_entries": "unknown",
                "local_dst_entries": "unknown",
                "local_src_entries": "unknown",
                "expected_h2d_bytes": "unknown",
            }
            estimator = getattr(self._relocator, "estimate_transfer_stats", None)
            if estimator is not None:
                try:
                    transfer_stats = estimator(self.metadata, new_metadata, layer_ids)
                except Exception as exc:
                    eplb_logger.warning(
                        "[EPLB Relocate] phase=estimate_fail op_id=%s "
                        "plan_handle=%s layers=%s reason=%s",
                        op_id,
                        plan_handle,
                        layer_ids,
                        str(exc),
                    )
            eplb_logger.debug(
                "[EPLB Relocate] phase=start op_id=%s plan_handle=%s layers=%s "
                "strategy=%s transfer_entries=%s local_dst_entries=%s "
                "local_src_entries=%s expected_h2d_bytes=%s",
                op_id,
                plan_handle,
                layer_ids,
                getattr(self.server_args, "eplb_relocate_strategy", "host_first"),
                transfer_stats["transfer_entries"],
                transfer_stats["local_dst_entries"],
                transfer_stats["local_src_entries"],
                transfer_stats["expected_h2d_bytes"],
            )
            future = self._relocate_pool.submit(
                self._run_relocate,
                self._relocator,
                self.metadata,
                new_metadata,
                layer_ids,
            )
            self._relocate_futures[op_id] = (future, plan_handle, layer_ids)
            self._relocate_contexts[op_id] = {
                "started_at": time.perf_counter(),
                **transfer_stats,
            }
            self._drain_relocate_futures()
        except Exception as exc:
            eplb_logger.error(
                "[EPLB Relocate] phase=fail op_id=%s plan_handle=%s layers=%s reason=%s",
                op_id,
                plan_handle,
                layer_ids,
                str(exc),
            )
            self._emit(
                _make_event(
                    "RelocateFailed",
                    op_id=op_id,
                    layer_id=layer_ids[0] if layer_ids else -1,
                    reason=str(exc),
                )
            )

    def _on_swap(self, op) -> None:
        op_id = int(_getattr(op, "op_id", 0))
        plan_handle = int(_getattr(op, "plan_handle", op_id))
        layer_ids = [int(x) for x in list(_getattr(op, "layer_ids", []))]
        start = time.perf_counter()
        eplb_logger.debug(
            "[EPLB Swap] phase=start op_id=%s plan_handle=%s layers=%s",
            op_id,
            plan_handle,
            layer_ids,
        )
        try:
            new_metadata = self._plan_handles.get(plan_handle)
            if new_metadata is None:
                raise RuntimeError(f"missing EPLB plan handle {plan_handle}")
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("EPLB swap cannot run during CUDA graph capture")
            for layer_id in layer_ids:
                state = self.layer_states.get(int(layer_id))
                if state is not None:
                    state.relocate_done_event.synchronize()
            self.metadata.update(new_metadata, layer_ids)
            for layer_id in layer_ids:
                state = self.layer_states.get(int(layer_id))
                if state is not None and hasattr(
                    self.metadata, "dispatch_info_for_layer"
                ):
                    state.dispatch_info = self.metadata.dispatch_info_for_layer(
                        int(layer_id)
                    )
            self.recorder.rebind(self.metadata)
            blocked_us = int((time.perf_counter() - start) * 1_000_000)
            self._persist(
                "swap",
                {
                    "op_id": op_id,
                    "plan_handle": plan_handle,
                    "layer_ids": layer_ids,
                    "blocked_us": blocked_us,
                    "physical_to_logical_map": (
                        self.metadata.physical_to_logical_map_cpu.clone()
                        if hasattr(self.metadata, "physical_to_logical_map_cpu")
                        else None
                    ),
                },
            )
            eplb_logger.debug(
                "[EPLB Swap] phase=done op_id=%s plan_handle=%s "
                "layers=%s blocked_us=%s",
                op_id,
                plan_handle,
                layer_ids,
                blocked_us,
            )
            self._emit(
                _make_event(
                    "SwapDone",
                    op_id=op_id,
                    layer_ids=layer_ids,
                    blocked_us=blocked_us,
                )
            )
        except Exception as exc:
            blocked_us = int((time.perf_counter() - start) * 1_000_000)
            eplb_logger.error(
                "[EPLB Swap] phase=fail op_id=%s plan_handle=%s "
                "layers=%s blocked_us=%s reason=%s",
                op_id,
                plan_handle,
                layer_ids,
                blocked_us,
                str(exc),
            )
            raise
