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
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping

import torch

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_DEFAULT_LOGGER = logging.getLogger(__name__)
_STATS: dict[str, tuple[int, float, float]] = {}
_ENABLED = os.getenv("TOKENSPEED_GLM5_PERF_DEBUG", "").lower() in _TRUE_VALUES
_LOG_EVERY = 32
try:
    _LOG_EVERY = max(1, int(os.getenv("TOKENSPEED_GLM5_PERF_DEBUG_EVERY", "32")))
except ValueError:
    pass
_PARITY_ENABLED = os.getenv("TOKENSPEED_GLM5_PARITY_DEBUG", "").lower() in _TRUE_VALUES
_PARITY_PREVIEW = 8
try:
    _PARITY_PREVIEW = max(
        1, int(os.getenv("TOKENSPEED_GLM5_PARITY_DEBUG_PREVIEW", "8"))
    )
except ValueError:
    pass


def _optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        _DEFAULT_LOGGER.warning("Ignoring invalid integer env %s=%r", name, value)
        return None


_PARITY_SEQ_START = _optional_int_env("TOKENSPEED_GLM5_PARITY_DEBUG_SEQ_START")
_PARITY_SEQ_END = _optional_int_env("TOKENSPEED_GLM5_PARITY_DEBUG_SEQ_END")
_PARITY_TOPK_LAYERS = os.getenv(
    "TOKENSPEED_GLM5_PARITY_DEBUG_TOPK_LAYERS",
    "0,-1",
)
_ALWAYS_LOG_PREFIXES = (
    "glm5:deepep_adapter_replay",
    "glm5:execute_forward_op",
    "glm5:forward_step",
    "glm5:graph_replay",
    "glm5:graph_replay_metadata",
    "glm5:output_d2h",
    "glm5:sampling",
    "glm5:sampling_prep",
    "glm5:target_forward",
)


def _enabled() -> bool:
    return _ENABLED


def _log_every() -> int:
    return _LOG_EVERY


def parity_debug_enabled() -> bool:
    return _PARITY_ENABLED


def parity_debug_preview() -> int:
    return _PARITY_PREVIEW


def _is_stream_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _tensor_preview(tensor: torch.Tensor, *, limit: int | None = None) -> dict:
    limit = _PARITY_PREVIEW if limit is None else max(1, int(limit))
    detached = tensor.detach()
    flat = detached.reshape(-1)
    preview = flat[:limit].to("cpu").tolist()
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "device": str(detached.device),
        "preview": preview,
    }


def _summarize_value(value):
    if isinstance(value, torch.Tensor):
        return _tensor_preview(value)
    if isinstance(value, Mapping):
        return {str(key): _summarize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item) for item in value[:_PARITY_PREVIEW]]
    return value


def _seq_lens_to_list(seq_lens: torch.Tensor | None) -> list[int]:
    if seq_lens is None:
        return []
    return [int(value) for value in seq_lens.detach().reshape(-1).to("cpu").tolist()]


def parity_debug_seq_matches(seq_lens: torch.Tensor | None) -> bool:
    if not parity_debug_enabled():
        return False
    if _is_stream_capturing():
        return False
    if _PARITY_SEQ_START is None and _PARITY_SEQ_END is None:
        return True
    seq_values = _seq_lens_to_list(seq_lens)
    if not seq_values:
        return False
    start = _PARITY_SEQ_START if _PARITY_SEQ_START is not None else min(seq_values)
    end = _PARITY_SEQ_END if _PARITY_SEQ_END is not None else max(seq_values)
    return any(start <= value <= end for value in seq_values)


def parity_debug_selected_layers(layer_ids) -> list[int]:
    """Select layer ids for verbose GLM5 top-k replay logging.

    The env value accepts comma/space-separated ids, negative indexes into the
    sorted layer-id list, or `all`/`*`. Defaults to the first and last layer to
    keep replay logs small unless the caller opts into more detail.
    """

    if not parity_debug_enabled():
        return []
    sorted_ids = sorted({int(layer_id) for layer_id in layer_ids})
    if not sorted_ids:
        return []

    raw_value = (_PARITY_TOPK_LAYERS or "").strip().lower()
    if raw_value in {"all", "*"}:
        return sorted_ids
    if not raw_value:
        raw_value = "0,-1"

    selected = []
    seen = set()
    for item in raw_value.replace(",", " ").split():
        try:
            layer_or_index = int(item)
        except ValueError:
            _DEFAULT_LOGGER.warning(
                "Ignoring invalid GLM5 parity top-k layer selector %r", item
            )
            continue

        if layer_or_index in sorted_ids:
            layer_id = layer_or_index
        elif -len(sorted_ids) <= layer_or_index < 0:
            layer_id = sorted_ids[layer_or_index]
        elif 0 <= layer_or_index < len(sorted_ids):
            layer_id = sorted_ids[layer_or_index]
        else:
            continue

        if layer_id not in seen:
            selected.append(layer_id)
            seen.add(layer_id)
    return selected


def parity_debug_log(
    name: str,
    *,
    logger: logging.Logger | None = None,
    seq_lens: torch.Tensor | None = None,
    **fields,
) -> None:
    if not parity_debug_seq_matches(seq_lens):
        return

    active_logger = logger or _DEFAULT_LOGGER
    payload = {key: _summarize_value(value) for key, value in fields.items()}
    if seq_lens is not None:
        payload["seq_lens"] = _seq_lens_to_list(seq_lens)
    active_logger.info("GLM5 parity %s %s", name, payload)


def _record(name: str, elapsed_ms: float, logger: logging.Logger) -> None:
    count, total_ms, max_ms = _STATS.get(name, (0, 0.0, 0.0))
    count += 1
    total_ms += elapsed_ms
    max_ms = max(max_ms, elapsed_ms)
    _STATS[name] = (count, total_ms, max_ms)

    should_log = (
        count <= 3
        or count % _log_every() == 0
        or any(name.startswith(prefix) for prefix in _ALWAYS_LOG_PREFIXES)
    )
    if should_log:
        logger.info(
            "GLM5 perf %s count=%d last=%.3fms avg=%.3fms max=%.3fms",
            name,
            count,
            elapsed_ms,
            total_ms / count,
            max_ms,
        )


class _CudaPerfRange:
    """Measure a CUDA range when GLM5 perf debug logging is enabled.

    This intentionally synchronizes the end event, so it must stay behind the
    environment flag. Keep the disabled path lightweight because the helper is
    used in decode hot paths.
    """

    __slots__ = (
        "enabled",
        "end_event",
        "logger",
        "name",
        "start_event",
        "start_time",
        "use_cuda_events",
    )

    def __init__(self, name: str, logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger or _DEFAULT_LOGGER
        self.enabled = _enabled()
        self.use_cuda_events = False
        self.start_time = 0.0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if not self.enabled:
            return self

        self.use_cuda_events = (
            torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing()
        )
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False

        if self.use_cuda_events:
            assert self.start_event is not None
            assert self.end_event is not None
            self.end_event.record()
            self.end_event.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        _record(self.name, elapsed_ms, self.logger)
        return False


def cuda_perf_range(name: str, *, logger: logging.Logger | None = None):
    return _CudaPerfRange(name, logger)
