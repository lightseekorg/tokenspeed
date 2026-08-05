"""Opt-in tensor recording for DSpark engine-parity investigations.

Set ``TOKENSPEED_DSPARK_PARITY_DIR`` to enable. The recorder is intentionally
eager-only: copying tensors to CPU during graph capture would either be captured
or perturb the graph. Production behavior is unchanged when the variable is
unset.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

_SAFE_STAGE = re.compile(r"[^a-zA-Z0-9_.-]+")
_RECORDER = None
_RECORDER_ENV: str | None = None
_RECORDER_LOCK = threading.Lock()


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


class DSparkParityRecorder:
    def __init__(
        self,
        root: str | Path,
        *,
        max_records_per_stage: int = 4,
        ranks: set[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_records_per_stage = max_records_per_stage
        self.ranks = {0} if ranks is None else set(ranks)
        self._counts: dict[tuple[int, str], int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(
        self,
        stage: str,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        rank = _rank()
        if rank not in self.ranks:
            return None
        stage = _SAFE_STAGE.sub("_", stage).strip("._") or "unnamed"
        key = (rank, stage)
        with self._lock:
            index = self._counts[key]
            if index >= self.max_records_per_stage:
                return None
            self._counts[key] += 1

        value = tensor.detach().contiguous().cpu()
        path = self.root / f"{stage}-{index:04d}-rank{rank}.pt"
        torch.save({"tensor": value, "metadata": metadata or {}}, path)

        numeric = value.float()
        summary = {
            "stage": stage,
            "index": index,
            "rank": rank,
            "path": str(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "min": float(numeric.min()) if value.numel() else None,
            "max": float(numeric.max()) if value.numel() else None,
            "mean": float(numeric.mean()) if value.numel() else None,
            "l2": float(torch.linalg.vector_norm(numeric)) if value.numel() else 0.0,
            "metadata": _json_value(metadata or {}),
        }
        manifest = self.root / f"manifest-rank{rank}.jsonl"
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
        return path


def _get_recorder() -> DSparkParityRecorder | None:
    global _RECORDER, _RECORDER_ENV
    root = os.environ.get("TOKENSPEED_DSPARK_PARITY_DIR", "").strip()
    if not root:
        return None
    with _RECORDER_LOCK:
        if _RECORDER is not None and _RECORDER_ENV == root:
            return _RECORDER
        max_records = int(os.environ.get("TOKENSPEED_DSPARK_PARITY_MAX_RECORDS", "4"))
        rank_text = os.environ.get("TOKENSPEED_DSPARK_PARITY_RANKS", "0")
        ranks = {int(item.strip()) for item in rank_text.split(",") if item.strip()}
        _RECORDER = DSparkParityRecorder(
            root, max_records_per_stage=max_records, ranks=ranks
        )
        _RECORDER_ENV = root
        return _RECORDER


def record_dspark_parity_tensor(
    stage: str,
    tensor: torch.Tensor | None,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    recorder = _get_recorder()
    if recorder is None or tensor is None:
        return None
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    return recorder.record(stage, tensor, metadata)


def reset_dspark_parity_recorder_for_tests() -> None:
    global _RECORDER, _RECORDER_ENV
    with _RECORDER_LOCK:
        _RECORDER = None
        _RECORDER_ENV = None
