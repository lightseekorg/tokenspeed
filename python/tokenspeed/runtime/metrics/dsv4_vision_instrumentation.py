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

"""Default-off observability for DeepSeek V4 vision integration tests.

The recorder deliberately owns no tensors and never makes a scheduling or
model-execution decision.  Producers hand it scalar metadata only.  Rank zero
exports the state snapshot and the optional JSONL stream; other ranks retain
the same hook shape without opening a file or exposing an IPC payload.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TextIO

DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION = 3
DSV4_VISION_DISPATCH_LOG_ENV = "TOKENSPEED_DSV4_VISION_DISPATCH_LOG"
_DEFAULT_MAX_KEYED_ENTRIES = 4096


@dataclasses.dataclass(frozen=True)
class MediaBindingRecord:
    """Scalar proof that one request bound one encoded media item in order."""

    request_id: str
    item_index: int
    modality: str
    content_hash_u64: int | None
    placeholder_token_id: int | None
    pad_value: int | None
    offsets: tuple[tuple[int, int], ...]
    compress_pad: int | None
    encoded_rows: int

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["offsets"] = [list(offset) for offset in self.offsets]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MediaBindingRecord:
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(value) != expected:
            raise ValueError(
                "DeepSeek V4 vision media-binding fields do not match schema"
            )
        return cls(
            request_id=str(value["request_id"]),
            item_index=int(value["item_index"]),
            modality=str(value["modality"]),
            content_hash_u64=(
                None
                if value["content_hash_u64"] is None
                else int(value["content_hash_u64"])
            ),
            placeholder_token_id=(
                None
                if value["placeholder_token_id"] is None
                else int(value["placeholder_token_id"])
            ),
            pad_value=None if value["pad_value"] is None else int(value["pad_value"]),
            offsets=tuple(
                (int(offset[0]), int(offset[1])) for offset in value["offsets"]
            ),
            compress_pad=(
                None if value["compress_pad"] is None else int(value["compress_pad"])
            ),
            encoded_rows=int(value["encoded_rows"]),
        )


@dataclasses.dataclass(frozen=True)
class DispatchRecord:
    schema_version: int
    forward_index: int
    path: str
    num_tokens: int
    batch_size: int
    request_ids: tuple[str, ...]
    num_extends: int
    extend_prefix_lens: tuple[int, ...]
    intersects_span: bool
    media_bindings: tuple[MediaBindingRecord, ...]
    encoder_wall_ns: int
    wall_ns: int

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["request_ids"] = list(self.request_ids)
        value["extend_prefix_lens"] = list(self.extend_prefix_lens)
        value["media_bindings"] = [item.to_dict() for item in self.media_bindings]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DispatchRecord:
        expected = {field.name for field in dataclasses.fields(cls)}
        actual = set(value)
        if actual != expected:
            raise ValueError(
                "DeepSeek V4 vision dispatch record fields do not match schema: "
                f"missing={sorted(expected - actual)} "
                f"unknown={sorted(actual - expected)}"
            )
        version = int(value["schema_version"])
        if version != DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported DeepSeek V4 vision instrumentation schema version "
                f"{version}; expected {DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION}"
            )
        path = str(value["path"])
        if path not in ("eager", "replay"):
            raise ValueError(f"Invalid DeepSeek V4 vision dispatch path: {path!r}")
        return cls(
            schema_version=version,
            forward_index=int(value["forward_index"]),
            path=path,
            num_tokens=int(value["num_tokens"]),
            batch_size=int(value["batch_size"]),
            request_ids=tuple(str(item) for item in value["request_ids"]),
            num_extends=int(value["num_extends"]),
            extend_prefix_lens=tuple(int(item) for item in value["extend_prefix_lens"]),
            intersects_span=bool(value["intersects_span"]),
            media_bindings=tuple(
                MediaBindingRecord.from_dict(item) for item in value["media_bindings"]
            ),
            encoder_wall_ns=int(value["encoder_wall_ns"]),
            wall_ns=int(value["wall_ns"]),
        )


@dataclasses.dataclass(frozen=True)
class CurrentStreamTiming:
    """Start/end events bound to one execution stream."""

    start_event: Any
    end_event: Any
    stream: Any


class DSV4VisionInstrumentationRecorder:
    """Process-local scalar recorder with bounded keyed retention."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        global_rank: int = 0,
        dispatch_log_path: str | None = None,
        max_keyed_entries: int = _DEFAULT_MAX_KEYED_ENTRIES,
        opener=open,
    ) -> None:
        if max_keyed_entries <= 0:
            raise ValueError("max_keyed_entries must be positive")
        self.enabled = bool(enabled)
        self.global_rank = int(global_rank)
        self._max_keyed_entries = int(max_keyed_entries)
        self._lock = threading.Lock()
        self._snapshot_id = 0
        self._dispatch_forward_index = 0
        self._totals = {
            "mm_encoder_calls": 0,
            "mm_encoded_items": 0,
            "mm_encoded_rows": 0,
        }
        self._by_item: OrderedDict[str, dict[str, int]] = OrderedDict()
        self._by_request: OrderedDict[str, dict[str, int]] = OrderedDict()
        self._by_item_evicted = 0
        self._by_request_evicted = 0
        self._scheduler_config: dict[str, int | bool] = {}
        self._model: dict[str, Any] = {}
        self._pending_encoder_wall_ns = 0
        self._stream: TextIO | None = None

        # Flag-off and non-rank-zero construction must not even consult the
        # sink environment, much less open a file.
        if self.enabled and self.global_rank == 0:
            path = dispatch_log_path
            if path is None:
                path = os.environ.get(DSV4_VISION_DISPATCH_LOG_ENV)
            if path:
                # Evidence destinations are preflighted as fresh.  Exclusive
                # creation prevents accidental overwrite if that contract is
                # violated.
                self._stream = opener(path, "x", encoding="utf-8", buffering=1)

    @property
    def has_dispatch_stream(self) -> bool:
        return self._stream is not None

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()

    def _mutated(self) -> None:
        self._snapshot_id += 1

    def _evict_items_if_needed(self) -> None:
        while len(self._by_item) > self._max_keyed_entries:
            self._by_item.popitem(last=False)
            self._by_item_evicted += 1

    def _evict_requests_if_needed(self) -> None:
        while len(self._by_request) > self._max_keyed_entries:
            self._by_request.popitem(last=False)
            self._by_request_evicted += 1

    def record_encoder_call(
        self,
        modality: str,
        items: Sequence[tuple[str, int]],
        *,
        wall_ns: int = 0,
    ) -> None:
        """Record one successful modality encoder call.

        ``items`` contains the canonical item hash and encoded row count for
        each item in that call. Deduplicated aliases never reach this hook.
        """
        if not self.enabled:
            return
        normalized = [
            (f"{modality}:{item_hash}", int(rows)) for item_hash, rows in items
        ]
        wall_ns = int(wall_ns)
        if wall_ns < 0:
            raise ValueError("encoder wall time must be non-negative")
        with self._lock:
            self._totals["mm_encoder_calls"] += 1
            self._totals["mm_encoded_items"] += len(normalized)
            self._totals["mm_encoded_rows"] += sum(rows for _, rows in normalized)
            self._pending_encoder_wall_ns += wall_ns
            for key, rows in normalized:
                entry = self._by_item.get(key)
                if entry is None:
                    entry = {"encoder_calls": 0, "encoded_rows": 0}
                    self._by_item[key] = entry
                entry["encoder_calls"] += 1
                entry["encoded_rows"] += rows
            self._evict_items_if_needed()
            self._mutated()

    def begin_dispatch(self) -> None:
        """Start one instrumented forward's encoder-time accumulation."""
        if not self.enabled:
            return
        with self._lock:
            self._pending_encoder_wall_ns = 0

    def record_dispatch(
        self,
        *,
        path: str,
        num_tokens: int,
        batch_size: int,
        request_ids: Sequence[str],
        num_extends: int,
        extend_prefix_lens: Sequence[int],
        intersects_span: bool,
        wall_ns: int,
        media_bindings: Sequence[MediaBindingRecord] = (),
    ) -> DispatchRecord | None:
        if not self.enabled:
            return None
        if path not in ("eager", "replay"):
            raise ValueError(f"Invalid DeepSeek V4 vision dispatch path: {path!r}")
        request_ids = tuple(str(item) for item in request_ids)
        extend_prefix_lens = tuple(int(item) for item in extend_prefix_lens)
        num_extends = int(num_extends)
        if num_extends < 0 or num_extends > len(request_ids):
            raise ValueError("num_extends must index a prefix of request_ids")
        if len(extend_prefix_lens) != num_extends:
            raise ValueError(
                "extend_prefix_lens must contain one value per extend request"
            )

        with self._lock:
            self._dispatch_forward_index += 1
            record = DispatchRecord(
                schema_version=DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION,
                forward_index=self._dispatch_forward_index,
                path=path,
                num_tokens=int(num_tokens),
                batch_size=int(batch_size),
                request_ids=request_ids,
                num_extends=num_extends,
                extend_prefix_lens=extend_prefix_lens,
                intersects_span=bool(intersects_span),
                media_bindings=tuple(media_bindings),
                encoder_wall_ns=self._pending_encoder_wall_ns,
                wall_ns=int(wall_ns),
            )
            self._pending_encoder_wall_ns = 0
            for request_id, accepted_prefix in zip(
                request_ids[:num_extends], extend_prefix_lens
            ):
                entry = self._by_request.get(request_id)
                if entry is None:
                    entry = {
                        "accepted_prefix_tokens": accepted_prefix,
                        "first_extend_forward_index": record.forward_index,
                        "extend_forwards": 0,
                    }
                    self._by_request[request_id] = entry
                entry["extend_forwards"] += 1
            self._evict_requests_if_needed()
            self._mutated()
            if self._stream is not None:
                self._stream.write(
                    json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                self._stream.flush()
            return record

    def record_scheduler_config(
        self,
        *,
        disable_prefix_cache: bool,
        max_scheduled_tokens: int,
        prefix_granularity: int,
    ) -> None:
        if not self.enabled:
            return
        value = {
            "disable_prefix_cache": bool(disable_prefix_cache),
            "max_scheduled_tokens": int(max_scheduled_tokens),
            "prefix_granularity": int(prefix_granularity),
        }
        with self._lock:
            if self._scheduler_config != value:
                self._scheduler_config = value
                self._mutated()

    def record_model_parameters(self, parameter_names: Iterable[str]) -> None:
        if not self.enabled:
            return
        names = sorted(str(name) for name in parameter_names)
        digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        value = {"param_count": len(names), "param_names_sha256": digest}
        with self._lock:
            changed = any(self._model.get(key) != item for key, item in value.items())
            if changed:
                self._model.update(value)
                self._mutated()

    def record_logits_dtype(self, dtype: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if "logits_dtype" not in self._model:
                self._model["logits_dtype"] = str(dtype)
                self._mutated()

    def record_prefill_index_buffer(
        self, shape: Sequence[int], allocation_bytes: int
    ) -> None:
        if not self.enabled:
            return
        shape = [int(dim) for dim in shape]
        allocation_bytes = int(allocation_bytes)
        with self._lock:
            # Capture the first configured allocation once. A running maximum
            # makes this baseline depend on unrelated later request traffic.
            if "prefill_index_buffer_bytes" not in self._model:
                self._model["prefill_index_buffer_shape"] = shape
                self._model["prefill_index_buffer_bytes"] = allocation_bytes
                self._mutated()

    def reset(self) -> None:
        """Clear bounded observations while preserving monotone watermarks."""
        if not self.enabled:
            return
        with self._lock:
            self._totals = {
                "mm_encoder_calls": 0,
                "mm_encoded_items": 0,
                "mm_encoded_rows": 0,
            }
            self._by_item.clear()
            self._by_request.clear()
            self._by_item_evicted = 0
            self._by_request_evicted = 0
            self._pending_encoder_wall_ns = 0
            self._mutated()

    def internal_state(self) -> dict[str, Any]:
        if not self.enabled or self.global_rank != 0:
            return {}
        with self._lock:
            return {
                "dsv4_instrumentation": {
                    "schema_version": DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION,
                    "snapshot_id": self._snapshot_id,
                    "dispatch_forward_index": self._dispatch_forward_index,
                    "totals": dict(self._totals),
                    "by_item": {
                        key: dict(value) for key, value in self._by_item.items()
                    },
                    "by_request": {
                        key: dict(value) for key, value in self._by_request.items()
                    },
                    "by_item_evicted": self._by_item_evicted,
                    "by_request_evicted": self._by_request_evicted,
                    "scheduler_config": dict(self._scheduler_config),
                    "model": dict(self._model),
                }
            }


_recorder = DSV4VisionInstrumentationRecorder()


def configure_dsv4_vision_instrumentation(
    *,
    enabled: bool,
    global_rank: int,
    dispatch_log_path: str | None = None,
) -> DSV4VisionInstrumentationRecorder:
    global _recorder
    replacement = DSV4VisionInstrumentationRecorder(
        enabled=enabled,
        global_rank=global_rank,
        dispatch_log_path=dispatch_log_path,
    )
    previous = _recorder
    _recorder = replacement
    previous.close()
    return replacement


def get_dsv4_vision_instrumentation() -> DSV4VisionInstrumentationRecorder:
    return _recorder


def begin_current_stream_timing(
    enabled: bool, device_module: Any
) -> CurrentStreamTiming | None:
    """Record a start event without synchronizing the accelerator device."""
    if not enabled:
        return None
    is_capturing = getattr(device_module, "is_current_stream_capturing", None)
    if callable(is_capturing) and is_capturing():
        return None
    if not hasattr(device_module, "current_stream") or not hasattr(
        device_module, "Event"
    ):
        return None
    stream = device_module.current_stream()
    timing = CurrentStreamTiming(
        start_event=device_module.Event(enable_timing=True),
        end_event=device_module.Event(enable_timing=True),
        stream=stream,
    )
    timing.start_event.record(stream)
    return timing


def finish_current_stream_timing(
    timing: CurrentStreamTiming | None,
) -> int | None:
    """Wait only for the end event and return elapsed accelerator time in ns."""
    if timing is None:
        return None
    timing.end_event.record(timing.stream)
    timing.end_event.synchronize()
    elapsed_ms = timing.start_event.elapsed_time(timing.end_event)
    return max(0, int(round(elapsed_ms * 1_000_000)))


def record_encoder_call(
    modality: Any,
    items: Sequence[Any],
    outputs: Sequence[Any],
    *,
    wall_ns: int = 0,
) -> None:
    recorder = _recorder
    if not recorder.enabled:
        return
    modality_name = getattr(modality, "name", str(modality)).lower()
    facts = []
    for item, output in zip(items, outputs):
        item_hash = getattr(item, "hash", None)
        rows = int(output.shape[0])
        facts.append(("none" if item_hash is None else str(item_hash), rows))
    recorder.record_encoder_call(modality_name, facts, wall_ns=wall_ns)


def _optional_scalar(value: Any) -> int | None:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        if int(numel()) != 1:
            raise ValueError(
                "DeepSeek V4 instrumentation expected one scalar metadata value"
            )
        value = value.reshape(-1)[0].item()
    return int(value)


def build_request_media_bindings(context: Any) -> tuple[MediaBindingRecord, ...]:
    """Read scalar request/media binding evidence after an instrumented forward."""
    if context is None:
        return ()
    request_ids = list(getattr(context, "request_ids", ()))
    records = []
    for request_index, mm_inputs in enumerate(getattr(context, "mm_inputs", ())):
        if mm_inputs is None or request_index >= len(request_ids):
            continue
        request_id = str(request_ids[request_index])
        placeholder_token_id = _optional_scalar(getattr(mm_inputs, "im_token_id", None))
        for item_index, item in enumerate(mm_inputs.mm_items):
            model_specific = getattr(item, "model_specific_data", None) or {}
            if "dsv4_compress_pad" not in model_specific:
                continue
            encoded = getattr(item, "encoded", None)
            encoded_rows = int(encoded.shape[0]) if encoded is not None else 0
            records.append(
                MediaBindingRecord(
                    request_id=request_id,
                    item_index=item_index,
                    modality=getattr(
                        getattr(item, "modality", None), "name", "unknown"
                    ).lower(),
                    content_hash_u64=_optional_scalar(getattr(item, "hash", None)),
                    placeholder_token_id=placeholder_token_id,
                    pad_value=_optional_scalar(getattr(item, "pad_value", None)),
                    offsets=tuple(
                        (int(start), int(end))
                        for start, end in (getattr(item, "offsets", None) or ())
                    ),
                    compress_pad=_optional_scalar(model_specific["dsv4_compress_pad"]),
                    encoded_rows=encoded_rows,
                )
            )
    return tuple(records)


def record_logits_dtype(dtype: Any) -> None:
    recorder = _recorder
    if recorder.enabled:
        recorder.record_logits_dtype(str(dtype))


def record_prefill_index_buffer(buffer: Any) -> None:
    recorder = _recorder
    if recorder.enabled:
        recorder.record_prefill_index_buffer(
            tuple(int(dim) for dim in buffer.shape),
            int(buffer.numel()) * int(buffer.element_size()),
        )


def read_dispatch_log(
    path: str | os.PathLike[str],
    *,
    after_forward_index: int = 0,
    through_forward_index: int | None = None,
) -> list[DispatchRecord]:
    records = []
    previous_index = 0
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid DeepSeek V4 vision dispatch JSON on line {line_number}"
                ) from exc
            record = DispatchRecord.from_dict(value)
            if record.forward_index <= previous_index:
                raise ValueError(
                    "DeepSeek V4 vision dispatch forward_index is not strictly "
                    "increasing"
                )
            previous_index = record.forward_index
            if record.forward_index <= after_forward_index:
                continue
            if (
                through_forward_index is not None
                and record.forward_index > through_forward_index
            ):
                continue
            records.append(record)
    return records


def _validate_snapshot(value: Mapping[str, Any]) -> Mapping[str, Any]:
    version = int(value.get("schema_version", -1))
    if version != DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported DeepSeek V4 vision instrumentation snapshot schema "
            f"version {version}; expected {DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION}"
        )
    required = {
        "schema_version",
        "snapshot_id",
        "dispatch_forward_index",
        "totals",
        "by_item",
        "by_request",
        "by_item_evicted",
        "by_request_evicted",
        "scheduler_config",
        "model",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"DeepSeek V4 vision instrumentation snapshot is missing {sorted(missing)}"
        )
    return value


def read_internal_states(
    internal_states: Sequence[Mapping[str, Any]],
    *,
    item_keys: Sequence[str] = (),
    request_ids: Sequence[str] = (),
) -> Mapping[str, Any]:
    snapshots = [
        state["dsv4_instrumentation"]
        for state in internal_states
        if "dsv4_instrumentation" in state
    ]
    if len(snapshots) != 1:
        raise ValueError(
            "Expected exactly one rank-zero DeepSeek V4 vision instrumentation "
            f"snapshot, found {len(snapshots)}"
        )
    snapshot = _validate_snapshot(snapshots[0])
    by_item = snapshot["by_item"]
    by_request = snapshot["by_request"]
    missing_items = [key for key in item_keys if key not in by_item]
    missing_requests = [key for key in request_ids if key not in by_request]
    if missing_items or missing_requests:
        raise KeyError(
            "DeepSeek V4 vision instrumentation correlation key is absent: "
            f"items={missing_items} requests={missing_requests}"
        )
    return snapshot


def read_counters(
    engine: Any,
    *,
    item_keys: Sequence[str] = (),
    request_ids: Sequence[str] = (),
) -> Mapping[str, Any]:
    server_info = engine.get_server_info()
    return read_internal_states(
        server_info.get("internal_states", ()),
        item_keys=item_keys,
        request_ids=request_ids,
    )


__all__ = [
    "DSV4_VISION_DISPATCH_LOG_ENV",
    "DSV4_VISION_INSTRUMENTATION_SCHEMA_VERSION",
    "CurrentStreamTiming",
    "DSV4VisionInstrumentationRecorder",
    "DispatchRecord",
    "MediaBindingRecord",
    "begin_current_stream_timing",
    "build_request_media_bindings",
    "configure_dsv4_vision_instrumentation",
    "finish_current_stream_timing",
    "get_dsv4_vision_instrumentation",
    "read_counters",
    "read_dispatch_log",
    "read_internal_states",
    "record_encoder_call",
    "record_logits_dtype",
    "record_prefill_index_buffer",
]
