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

"""CUDA graph scope identity and bounded per-replay integer metadata collection.

GPU work is owned by the execution stream. Reading pinned values requires the
existing result-copy event to have been synchronized by the result owner.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import queue
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    ProfileSession,
    loaded_library_manifest,
)
from tokenspeed_kernel.profile_utilization import HardwareProfile

if TYPE_CHECKING:
    from tokenspeed_kernel.profile_nvtx import NvtxTimeline


class ProfileWindow:
    """A finite serving window armed by the existing profiling request."""

    def __init__(self, *, start_step: int, num_steps: int):
        self.start_step, self.num_steps = start_step, num_steps
        self.armed = False
        self.observed = 0
        self.selected = 0
        self._started = False

    def arm(self):
        if self._started:
            raise ValueError(
                "A utilization run has one window; choose a new run/output to profile again"
            )
        self._started, self.armed = True, True

    def select(self) -> bool:
        if not self.armed:
            return False
        self.observed += 1
        if self.start_step <= self.observed < self.start_step + self.num_steps:
            self.selected += 1
            return True
        return False

    def stop(self):
        self.armed = False


@dataclass(frozen=True)
class MetadataLease:
    slot: int
    generation: int
    step_id: str


@dataclass
class _MetadataSlot:
    generation: int = 0
    state: str = "free"
    step_id: str | None = None
    event: Any = None


class MetadataSlotPool:
    """Host-only generation ownership; a queued slot is never overwritten."""

    def __init__(self, capacity: int, *, identity_capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Metadata slot capacity must be positive")
        if type(identity_capacity) is not int or identity_capacity < 1:
            raise ValueError("Metadata identity capacity must be positive")
        self.identity_capacity = identity_capacity
        self._slots = [_MetadataSlot() for _ in range(capacity)]
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def reserve(self, step_id: str) -> tuple[MetadataLease | None, str | None]:
        with self._lock:
            if step_id in self._seen:
                return None, "duplicate_step_identity"
            if len(self._seen) >= self.identity_capacity:
                return None, "step_identity_capacity"
            self._seen.add(step_id)
            for index, slot in enumerate(self._slots):
                if slot.state == "free":
                    slot.generation += 1
                    slot.state, slot.step_id, slot.event = "reserved", step_id, None
                    return MetadataLease(index, slot.generation, step_id), None
            return None, "ring_full"

    def _owned(self, lease: MetadataLease) -> _MetadataSlot:
        slot = self._slots[lease.slot]
        if (slot.generation, slot.step_id) != (lease.generation, lease.step_id):
            raise ValueError("Stale or mismatched metadata generation")
        return slot

    def queue(self, lease: MetadataLease, event: Any) -> None:
        with self._lock:
            slot = self._owned(lease)
            if slot.state != "reserved" or event is None:
                raise ValueError("Queue metadata exactly once with its result event")
            slot.state, slot.event = "queued", event

    def ready_after_sync(self, lease: MetadataLease, event: Any) -> None:
        with self._lock:
            slot = self._owned(lease)
            if slot.state != "queued" or slot.event is not event:
                raise ValueError("Metadata needs its exact synchronized result event")
            slot.state = "ready"

    def release(self, lease: MetadataLease) -> None:
        with self._lock:
            slot = self._owned(lease)
            if slot.state != "ready":
                raise ValueError("Persist ready metadata before releasing its slot")
            slot.state, slot.step_id, slot.event = "free", None, None

    def status(self) -> dict[str, int]:
        with self._lock:
            return {
                state: sum(slot.state == state for slot in self._slots)
                for state in ("free", "reserved", "queued", "ready")
            }


class GraphNodes:
    """Read capture node IDs from the exact CUPTI instance used by Nsight."""

    def __init__(self, *, cuda_runtime_library: str, cupti_library: str):
        runtime, cupti = (
            Path(path).resolve(strict=True)
            for path in (cuda_runtime_library, cupti_library)
        )
        loaded = {
            str(Path(line.split()[-1]).resolve())
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split()[-1].startswith("/")
        }
        if str(cupti) not in loaded:
            raise ValueError(
                "Configured CUPTI must already be loaded by the external profiler"
            )
        self.runtime, self.cupti = C.CDLL(str(runtime)), C.CDLL(str(cupti))
        self.capture = self.runtime.cudaStreamGetCaptureInfo_v2
        self.capture.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_int),
            C.POINTER(C.c_ulonglong),
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_void_p,
        ]
        self.nodes = self.runtime.cudaGraphGetNodes
        self.nodes.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t)]
        self.node_id = self.cupti.cuptiGetGraphNodeId
        self.node_id.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
        self.graph_id = self.cupti.cuptiGetGraphId
        self.graph_id.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
        self.node_type = self.runtime.cudaGraphNodeGetType
        self.node_type.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        self._cached: dict[tuple[int, int], int] = {}
        self.node_types: dict[int, dict] = {}

    @staticmethod
    def _check(code, name):
        if code:
            raise RuntimeError(f"{name} returned {code}")

    def snapshot(self) -> tuple[int, set[int]]:
        status, sequence, graph = C.c_int(), C.c_ulonglong(), C.c_void_p()
        self._check(
            self.capture(
                torch.cuda.current_stream().cuda_stream,
                C.byref(status),
                C.byref(sequence),
                C.byref(graph),
                None,
                None,
            ),
            "cudaStreamGetCaptureInfo_v2",
        )
        if status.value != 1 or not graph.value:
            raise RuntimeError("Graph scope observation requires active capture")
        identity = C.c_uint32()
        self._check(self.graph_id(graph, C.byref(identity)), "cuptiGetGraphId")
        count = C.c_size_t()
        self._check(self.nodes(graph, None, C.byref(count)), "cudaGraphGetNodes(count)")
        if not count.value:
            return identity.value, set()
        handles = (C.c_void_p * count.value)()
        self._check(self.nodes(graph, handles, C.byref(count)), "cudaGraphGetNodes")
        result = set()
        for handle in handles[: count.value]:
            key = identity.value, handle
            if key not in self._cached:
                node = C.c_uint64()
                self._check(self.node_id(handle, C.byref(node)), "cuptiGetGraphNodeId")
                self._cached[key] = node.value
                kind = C.c_int()
                code = self.node_type(handle, C.byref(kind))
                self.node_types[node.value] = dict(
                    graph_node_id=node.value,
                    graph_id=identity.value,
                    cuda_graph_node_type=kind.value if code == 0 else None,
                    unavailable_reason=(
                        None if code == 0 else f"cudaGraphNodeGetType returned {code}"
                    ),
                )
            result.add(self._cached[key])
        return identity.value, result


@dataclass(frozen=True)
class IntegerSnapshot:
    trace_label: str
    name: str
    offset: int
    count: int
    dtype: str


class GraphCapture:
    """Capture once; metadata copies have distinct observer node identities."""

    def __init__(
        self,
        *,
        session: ProfileSession,
        costs: CostRegistry,
        hardware: HardwareProfile,
        capture_id: str,
        graph_key: tuple,
        nodes: GraphNodes,
        max_metadata_bytes: int,
        max_scopes: int,
        metadata: dict,
        output: Path,
        modules: list,
    ):
        self.session, self.costs, self.hardware = session, costs, hardware
        self.capture_id, self.graph_key, self.nodes = capture_id, graph_key, nodes
        self.max_metadata_bytes, self.max_scopes = max_metadata_bytes, max_scopes
        self.metadata, self.output, self.modules = metadata, output, modules
        self.storage = torch.zeros(
            max_metadata_bytes, dtype=torch.uint8, device=torch.cuda.current_device()
        )
        self.used = 0
        self.snapshots: list[IntegerSnapshot] = []
        self.scopes: list[dict] = []
        self.observer_nodes: set[int] = set()
        self.unavailable: list[dict] = []
        self._stack: list[int] = []
        self._thread = threading.get_native_id()
        self._contexts = ExitStack()

    def __enter__(self):
        for module, layer_id, name in self.modules:
            self._contexts.enter_context(
                self.session.watch_module(module, layer_id=layer_id, name=name)
            )
        self._contexts.enter_context(self.session.observe(self))
        self._contexts.enter_context(
            self.session.iteration(iter_id=0, metadata=self.metadata)
        )
        return self

    def __exit__(self, kind, error, traceback):
        self._contexts.__exit__(kind, error, traceback)
        if error is None:
            targets = {
                str(id(op.tensor_values)): op.trace_label for op in self.session.ops
            }
            self.snapshots = [
                replace(item, trace_label=targets[item.trace_label])
                for item in self.snapshots
            ]
            for item in self.unavailable:
                item["trace_label"] = targets[item["trace_label"]]
            for scope in self.scopes:
                scope["graph_node_ids"] = sorted(
                    set(scope["graph_node_ids"]) - self.observer_nodes
                )
            report = self.session.report(costs=self.costs)
            report.update(
                execution_mode="cuda_graph_capture",
                capture_id=self.capture_id,
                graph_key=self.graph_key,
                session_id=self.session.run_id,
                run_id=self.metadata["run_id"],
                process_id=os.getpid(),
                thread_id=self._thread,
                hardware=asdict(self.hardware),
                graph_scopes=self.scopes,
                graph_node_types=[
                    self.nodes.node_types[node]
                    for node in sorted(
                        self.observer_nodes.union(
                            *(set(scope["graph_node_ids"]) for scope in self.scopes)
                        )
                    )
                ],
                graph_node_type_api="cudaGraphNodeGetType",
                observer_graph_node_ids=sorted(self.observer_nodes),
                integer_snapshots=[asdict(item) for item in self.snapshots],
                integer_metadata_bytes=self.used,
                unavailable_integer_metadata=self.unavailable,
                static_metadata=self.metadata,
            )
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("x") as stream:
                json.dump(report, stream, allow_nan=False)
                stream.write("\n")

    def snapshot_integer_selection(self, producer, target, inputs, name):
        from tokenspeed_kernel.op_profile import _describe

        before = self.nodes.snapshot()
        source = producer()
        if (
            source.dtype not in (torch.int32, torch.int64)
            or not source.is_contiguous()
            or not 0 < source.numel() <= 4096
        ):
            raise ValueError("Integer selection must be bounded and contiguous")
        _describe(source, name, inputs)
        self.snapshot_integer_values(source.view(-1), target, name)
        graph, after = self.nodes.snapshot()
        if graph != before[0] or not before[1] <= after:
            raise RuntimeError("Metadata selection changed capture graph identity")
        self.observer_nodes.update(after - before[1])

    def snapshot_integer_values(
        self, source: torch.Tensor, target: dict, name: str
    ) -> None:
        target[name] = None
        if source.device.type != "cuda":
            self.unavailable.append(
                dict(
                    trace_label=str(id(target)),
                    name=name,
                    reason="cpu_integer_argument_not_dynamic_capture_bound",
                )
            )
            return
        if source.device.index != torch.cuda.current_device():
            self.unavailable.append(
                dict(
                    trace_label=str(id(target)),
                    name=name,
                    reason="integer_argument_on_other_device",
                )
            )
            return
        start = (self.used + 7) // 8 * 8
        end = start + source.numel() * source.element_size()
        if end > self.max_metadata_bytes:
            self.unavailable.append(
                dict(
                    trace_label=str(id(target)),
                    name=name,
                    reason="capture_metadata_byte_capacity",
                )
            )
            return
        before = self.nodes.snapshot()
        self.storage[start:end].view(source.dtype).copy_(source, non_blocking=True)
        graph, after = self.nodes.snapshot()
        if graph != before[0] or not before[1] <= after:
            raise RuntimeError("Metadata copy changed capture graph identity")
        self.observer_nodes.update(after - before[1])
        # target identity is resolved to its real OpRecord after collection.
        self.snapshots.append(
            IntegerSnapshot(
                str(id(target)), name, start, source.numel(), str(source.dtype)
            )
        )
        self.used = end

    @contextmanager
    def record_func(self, label: str, *, display_name: str):
        if threading.get_native_id() != self._thread:
            raise RuntimeError("Graph capture must remain on its execution thread")
        if len(self.scopes) >= self.max_scopes:
            raise RuntimeError("Graph profile max_scopes exhausted")
        parent = self._stack[-1] if self._stack else None
        scope = dict(
            scope_id=len(self.scopes),
            parent_scope_id=parent,
            trace_label=label,
            display_name=display_name,
        )
        self.scopes.append(scope)
        self._stack.append(scope["scope_id"])
        before = self.nodes.snapshot()
        try:
            yield
        finally:
            self._stack.pop()
            graph, after = self.nodes.snapshot()
            if graph != before[0] or not before[1] <= after:
                raise RuntimeError(
                    "Capture graph identity changed or nodes were removed"
                )
            scope.update(graph_id=graph, graph_node_ids=sorted(after - before[1]))


class ProfileStep:
    """Exact optional result identity; device readiness does not prove commit."""

    def __init__(
        self,
        *,
        owner: GraphReplayCollector,
        run_id: str,
        rank: int,
        serving_step: int,
        request_ids: tuple[str, ...],
        pool_indices: tuple[int, ...],
    ):
        self.owner, self.run_id, self.rank = owner, run_id, rank
        self.serving_step = serving_step
        self.step_id = f"{run_id}:r{rank}:s{serving_step}"
        self.request_ids, self.pool_indices = request_ids, pool_indices
        self.forward_id: int | None = None
        self.capture: GraphCapture | None = None
        self.lease: MetadataLease | None = None
        self.eager_timeline: NvtxTimeline | None = None
        self.unavailable_reason: str | None = None
        self._event: Any = None
        self._ready = False
        self.device_ready_sequence: int | None = None
        self.replay_thread_id: int | None = None
        self.replay_stream: int | None = None
        self.geometry: dict = {}
        self.owner.selected_steps += 1

    @property
    def nvtx_label(self) -> str:
        return f"tokenspeed.profile.replay | {self.step_id}"

    def bind_forward_id(self, forward_id: int) -> None:
        if type(forward_id) is not int or forward_id < 0:
            raise ValueError("ForwardId must be a nonnegative integer")
        if self.forward_id is not None and self.forward_id != forward_id:
            raise ValueError("One profile result cannot bind two ForwardIds")
        if self.forward_id == forward_id:
            return
        self.forward_id = forward_id
        if self._ready:
            try:
                self.owner.emit({**self.identity(), "event": "forward_identity_bound"})
            except Exception as error:
                self.owner._writer_error = (
                    f"Forward identity index unavailable: {error}"
                )

    def identity(self) -> dict:
        return dict(
            run_id=self.run_id,
            rank=self.rank,
            step_id=self.step_id,
            serving_step=self.serving_step,
            forward_id=self.forward_id,
            request_ids=self.request_ids,
            request_pool_indices=self.pool_indices,
        )

    def unavailable(self, reason: str) -> None:
        self.unavailable_reason = reason

    def bind_eager_timeline(self, timeline: NvtxTimeline) -> None:
        if self.eager_timeline is not None or self.replay_thread_id is not None:
            raise ValueError("A serving step must have one native execution path")
        self.eager_timeline = timeline

    def bind_result_event(self, event: Any) -> None:
        if self._event is not None:
            raise ValueError("Bind the result-copy event exactly once")
        self._event = event
        if self.eager_timeline is not None:
            self.eager_timeline.verify_result_stream()
        elif self.capture is None and self.unavailable_reason is None:
            self.unavailable("no_graph_metadata_for_step")
        if self.lease is not None:
            self.owner.slots.queue(self.lease, event)

    def device_results_ready(self, event: Any) -> None:
        """Called only after the result owner's existing exact-once sync gate."""
        if self._ready or self._event is not event or event is None:
            raise ValueError(
                "Profile metadata requires its exact result synchronization"
            )
        self._ready = True
        if self.lease is not None:
            self.owner.slots.ready_after_sync(self.lease, event)
        self.owner.persist_ready(self)

    def ready_record(self) -> dict:
        return {
            **self.identity(),
            "event": "device_results_ready",
            "device_ready_sequence": self.device_ready_sequence,
            "process_id": os.getpid(),
            "replay_thread_id": self.replay_thread_id,
            "replay_stream": self.replay_stream,
            "replay_nvtx_label": self.nvtx_label,
            "capture_id": self.capture.capture_id if self.capture is not None else None,
            "capture_file": (
                str(self.capture.output) if self.capture is not None else None
            ),
            "metadata_lease": asdict(self.lease) if self.lease is not None else None,
            "unavailable_reason": self.unavailable_reason,
            "geometry": self.geometry,
            "completion_identity_state": "completion_identity_unavailable",
        }


class GraphReplayCollector:
    """One rank's bounded pinned slots, paired to exact async results.

    Graph rows are flushed before reusing their slots. Eager sidecars retain
    independently bounded reservations until the profiling window stops.
    """

    def __init__(
        self,
        *,
        output: Path,
        metadata_slots: int,
        max_metadata_bytes: int,
        max_steps: int,
        source_context: dict | None = None,
    ):
        self.output = output
        self._source_context = json.loads(json.dumps(source_context))
        self._source_baseline = loaded_library_manifest(
            source_context=self._source_context, collection_phase="collector_start"
        ).get("semantic_provenance")
        self.slots = MetadataSlotPool(metadata_slots, identity_capacity=max_steps)
        self.host_buffers = [
            torch.empty(max_metadata_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(metadata_slots)
        ]
        self._write_lock = threading.Lock()
        self.selected_steps = 0
        self.device_ready_steps = 0
        self.persisted_payloads = 0
        self.persisted_eager = 0
        self.dropped_metadata = 0
        if output.exists():
            raise ValueError("Profile replay output already exists; choose a new run")
        output.parent.mkdir(parents=True, exist_ok=True)
        self._ready_queue = queue.Queue(maxsize=metadata_slots)
        self._eager_capacity = metadata_slots
        self._eager_lock = threading.Lock()
        self._eager_pending: dict[str, tuple[ProfileStep, dict] | None] = {}
        self._writer_error: str | None = None
        self._closed = False
        self._index_stream = output.open("x", encoding="utf-8")
        self._writer = threading.Thread(
            target=self._drain, name="profile-metadata-writer", daemon=True
        )
        self._writer.start()

    @contextmanager
    def reserve_eager_metadata(self, step: ProfileStep):
        """Reserve a bounded serving eager capture before allocating its buffer."""
        with self._eager_lock:
            if self._closed or step.step_id in self._eager_pending:
                raise ValueError("Eager metadata reservation is closed or duplicated")
            reserved = len(self._eager_pending) < self._eager_capacity
            if reserved:
                self._eager_pending[step.step_id] = None
        if not reserved:
            step.unavailable("metadata_dropped:eager_postwindow_capacity")
        try:
            yield reserved
        finally:
            if reserved and step.eager_timeline is None:
                with self._eager_lock:
                    self._eager_pending.pop(step.step_id, None)

    def persist_ready(self, step: ProfileStep) -> None:
        self.device_ready_steps += 1
        step.device_ready_sequence = self.device_ready_steps
        if step.eager_timeline is not None:
            with self._eager_lock:
                if (
                    step.step_id not in self._eager_pending
                    or self._eager_pending[step.step_id] is not None
                ):
                    raise ValueError(
                        "Ready eager metadata requires its one reservation"
                    )
                self._eager_pending[step.step_id] = (step, step.ready_record())
            return
        if step.lease is None:
            try:
                self.emit(step.ready_record())
            except Exception as error:
                self._writer_error = f"Ready identity index unavailable: {error}"
            return
        # Exactly one queue entry per owned slot; no CUDA calls or value reads.
        try:
            self._ready_queue.put_nowait(step)
        except queue.Full:
            step.unavailable("metadata_dropped:persistence_queue_full")
            try:
                self.emit(step.ready_record())
            except Exception as error:
                self._writer_error = f"Dropped identity index unavailable: {error}"
            if step.lease is not None:
                self.slots.release(step.lease)

    def _drain(self) -> None:
        while True:
            step = self._ready_queue.get()
            try:
                if step is None:
                    return
                row = step.ready_record()
                if step.unavailable_reason is None and step.capture is not None:
                    host = self.host_buffers[step.lease.slot]
                    if host.device.type != "cpu":
                        raise RuntimeError(
                            "Metadata writer accepts ready host memory only"
                        )
                    payload = C.string_at(host.data_ptr(), step.capture.used)
                    path = self.output.with_name(
                        f"rank-{step.rank}.step-{step.serving_step}.metadata.bin"
                    )
                    with path.open("xb") as stream:
                        stream.write(payload)
                    row["binary_payload"] = dict(
                        path=str(path),
                        bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                    )
                self.emit(row)
                if "binary_payload" in row:
                    self.persisted_payloads += 1
                if step.lease is not None:
                    self.slots.release(step.lease)
            except BaseException as error:
                # Keep its slot owned: failed persistence cannot expose old bytes
                # to a new generation. Future captures explicitly drop on capacity.
                self._writer_error = f"{type(error).__name__}: {error}"
                if step is not None:
                    step.unavailable(
                        f"metadata_persistence_failed:{self._writer_error}"
                    )
                    try:
                        self.emit(step.ready_record())
                    except BaseException as emit_error:
                        # Failure reporting may fail for the same filesystem.
                        # Retain the owned slot/error, but always acknowledge
                        # this CPU queue item so stop/flush cannot deadlock.
                        self._writer_error += f"; failure index unavailable: {type(emit_error).__name__}: {emit_error}"
            finally:
                self._ready_queue.task_done()

    def _flush_eager_ready(self) -> None:
        with self._eager_lock:
            pending = [
                record for record in self._eager_pending.values() if record is not None
            ]
            for step, row in sorted(
                pending, key=lambda record: record[1]["device_ready_sequence"]
            ):
                timeline = step.eager_timeline
                try:
                    timeline.finish_after_results()
                    with timeline.output.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                        size = os.fstat(stream.fileno()).st_size
                    self.emit(
                        {
                            **row,
                            "execution_mode": "eager",
                            "eager_sidecar": dict(
                                path=str(timeline.output.resolve()),
                                bytes=size,
                                sha256=digest,
                            ),
                        }
                    )
                    self.persisted_eager += 1
                except Exception as error:
                    self._writer_error = f"{type(error).__name__}: {error}"
                    step.unavailable(
                        f"metadata_persistence_failed:{self._writer_error}"
                    )
                    try:
                        self.emit(
                            {**row, "unavailable_reason": step.unavailable_reason}
                        )
                    except Exception as emit_error:
                        self._writer_error += f"; failure index unavailable: {type(emit_error).__name__}: {emit_error}"
                finally:
                    step.eager_timeline = None
                    del self._eager_pending[step.step_id]

    def flush_ready(self) -> dict:
        """Drain only CPU-ready work; never query or wait for a CUDA event."""
        self._ready_queue.join()
        self._flush_eager_ready()
        library_manifest = loaded_library_manifest(source_context=self._source_context)
        if "semantic_provenance" in library_manifest:
            library_manifest["semantic_provenance"]["baseline"] = self._source_baseline
        return dict(
            selected_steps=self.selected_steps,
            device_results_ready=self.device_ready_steps,
            persisted_payloads=self.persisted_payloads,
            persisted_eager=self.persisted_eager,
            profile_config_sha256=(self._source_context or {}).get(
                "profile_config_sha256"
            ),
            dropped_metadata=self.dropped_metadata,
            unresolved_result_metadata=self.selected_steps - self.device_ready_steps,
            writer_flushed=True,
            loaded_library_manifest=library_manifest,
            writer_error=self._writer_error,
            slot_states=self.slots.status(),
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._ready_queue.join()
            self._ready_queue.put(None)
            self._writer.join()
            self._flush_eager_ready()
            with self._write_lock:
                self._index_stream.close()

    def emit(self, row: dict) -> None:
        with self._write_lock:
            self._index_stream.write(json.dumps(row, allow_nan=False) + "\n")
            if (
                row.get("event") == "device_results_ready"
                and row.get("unavailable_reason") is not None
            ):
                self.dropped_metadata += 1
            self._index_stream.flush()

    @contextmanager
    def replay(
        self, step: ProfileStep, capture: GraphCapture | None, *, geometry: dict
    ):
        if step.eager_timeline is not None:
            raise ValueError("A serving step must have one native execution path")
        if step.capture is not None or step.lease is not None:
            step.unavailable("multiple_graph_launches_for_step")
        else:
            step.capture, step.geometry = capture, geometry
            if capture is None:
                step.unavailable("same_run_graph_capture_unavailable")
            elif (
                geometry.get("batch_size") != 1
                or geometry.get("padded_batch_size") != 1
            ):
                step.unavailable("padding_or_batch_geometry_unsupported")
            elif geometry.get("query_shard"):
                step.unavailable("query_shard_unsupported")
            else:
                step.lease, reason = self.slots.reserve(step.step_id)
                if reason:
                    step.unavailable(f"metadata_dropped:{reason}")
        step.replay_thread_id = threading.get_native_id()
        step.replay_stream = torch.cuda.current_stream().cuda_stream
        torch.cuda.nvtx.range_push(step.nvtx_label)
        try:
            yield
            if step.lease is not None and step.unavailable_reason is None:
                host = self.host_buffers[step.lease.slot]
                host[: capture.used].copy_(
                    capture.storage[: capture.used], non_blocking=True
                )
                capture.storage.record_stream(torch.cuda.current_stream())
        except BaseException:
            step.unavailable("replay_failed")
            raise
        finally:
            torch.cuda.nvtx.range_pop()
