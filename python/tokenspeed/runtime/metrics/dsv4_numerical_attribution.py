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

"""Default-off full-logit and semantic-boundary recorder for DeepSeek V4."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

_PROFILES = {"tokenspeed-full", "tokenspeed-incremental"}
_MODES = {"series", "checkpoint"}
METADATA_KEY = "dsv4_numerical_attribution"
_POPULATIONS = {"text-control", "image-conditioned"}
_REQUIRED_METADATA_KEYS = {
    "profile",
    "case_id",
    "population",
    "prompt_length",
    "anchor",
    "anchor_sha256",
    "observations",
    "requested_step",
    "checkpoint_step",
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _attribution_metadata(sampling_params: Any) -> dict[str, Any] | None:
    custom = getattr(sampling_params, "custom_params", None)
    if not isinstance(custom, dict):
        return None
    value = custom.get(METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{METADATA_KEY} must be an object")
    return value


def build_sampling_directive(
    sampling_params: Any,
    *,
    enabled: bool,
    batch_size: int,
    enforce_eager: bool,
    speculative_algorithm: str | None,
    sampling_backend: str,
    forward_mode: str,
    start_position: int,
    call_input_length: int,
    accepted_prefix_tokens: int,
    request_id: str,
    trajectory_step: int,
) -> dict[str, Any] | None:
    """Validate the narrow attribution request and derive its forced token."""

    metadata = _attribution_metadata(sampling_params)
    if not enabled:
        if metadata is not None:
            raise RuntimeError(
                "numerical-attribution request metadata requires the server flag"
            )
        return None
    if metadata is None:
        raise RuntimeError(
            "an attribution-enabled server accepts only attributed requests"
        )
    if set(metadata) != _REQUIRED_METADATA_KEYS:
        raise RuntimeError("numerical-attribution metadata has missing or extra fields")
    if batch_size != 1 or not enforce_eager or speculative_algorithm is not None:
        raise RuntimeError(
            "numerical attribution requires eager single-request non-speculative execution"
        )
    if sampling_backend != "greedy":
        raise RuntimeError("numerical attribution requires the greedy sampling backend")
    if not request_id:
        raise RuntimeError("numerical-attribution request id is empty")
    profile = str(metadata["profile"])
    population = str(metadata["population"])
    if profile not in _PROFILES or population not in _POPULATIONS:
        raise RuntimeError("invalid numerical-attribution profile or population")
    prompt_length = int(metadata["prompt_length"])
    anchor = metadata["anchor"]
    observations = metadata["observations"]
    if (
        prompt_length <= 0
        or not isinstance(anchor, list)
        or len(anchor) != 32
        or any(not isinstance(token, int) or token < 0 for token in anchor)
        or _canonical_sha256(anchor) != metadata["anchor_sha256"]
    ):
        raise RuntimeError("numerical-attribution anchor or hash is invalid")
    if (
        not isinstance(observations, list)
        or len(observations) != 32
        or [row.get("step") for row in observations] != list(range(32))
        or len({row.get("comparison_key") for row in observations}) != 32
    ):
        raise RuntimeError("numerical-attribution observations are not 32 ordered keys")
    if profile == "tokenspeed-incremental":
        if metadata["requested_step"] is not None:
            raise RuntimeError("incremental attribution derives its step at runtime")
        step = int(trajectory_step)
        expected_mode = "prefill" if step == 0 else "decode"
        expected_start = 0 if step == 0 else prompt_length + step - 1
        expected_call_length = prompt_length if step == 0 else 1
        forced_token_id: int | None = int(anchor[step]) if step < len(anchor) else None
    else:
        step = int(metadata["requested_step"])
        if int(trajectory_step) != step:
            raise RuntimeError("full-reprefill attribution step differs")
        expected_mode = "prefill"
        expected_start = 0
        expected_call_length = prompt_length + step
        forced_token_id = None
    if not 0 <= step < 32:
        raise RuntimeError("numerical-attribution step is outside 0..31")
    if (
        forward_mode != expected_mode
        or start_position != expected_start
        or call_input_length != expected_call_length
        or accepted_prefix_tokens != 0
    ):
        raise RuntimeError(
            "numerical-attribution scheduling differs from the frozen profile"
        )
    observation = observations[step]
    if (
        observation.get("case_id") != metadata["case_id"]
        or observation.get("population") != population
        or int(observation.get("scored_anchor_id", -1)) != int(anchor[step])
    ):
        raise RuntimeError("numerical-attribution observation identity differs")
    checkpoint_step = metadata["checkpoint_step"]
    if checkpoint_step is not None and not 0 <= int(checkpoint_step) < 32:
        raise RuntimeError("checkpoint step is outside 0..31")
    return {
        **observation,
        "profile": profile,
        "method": ("full-reprefill" if profile == "tokenspeed-full" else "incremental"),
        "request_id": request_id,
        "forced_id": forced_token_id,
        "schedule_mode": forward_mode,
        "start_position": start_position,
        "call_input_length": call_input_length,
        "accepted_prefix_tokens": accepted_prefix_tokens,
        "capture_checkpoint": (
            checkpoint_step is not None and int(checkpoint_step) == step
        ),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def checkpoint_tensor_order(num_layers: int = 43) -> tuple[str, ...]:
    names = ["embedding_pre_hc"]
    for layer in range(num_layers):
        names.extend(
            (
                f"layer.{layer}.attn_pre_norm",
                f"layer.{layer}.attn_output",
                f"layer.{layer}.attn_post_hc",
                f"layer.{layer}.ffn_pre_norm",
                f"layer.{layer}.router_topk_ids",
                f"layer.{layer}.router_weights",
                f"layer.{layer}.ffn_output",
                f"layer.{layer}.block_post_hc",
            )
        )
    names.extend(("head_input", "final_norm", "logits"))
    return tuple(names)


class DSV4NumericalAttributionRecorder:
    """Rank-local create-exclusive recorder used only by the guarded harness."""

    def __init__(
        self,
        *,
        enabled: bool,
        global_rank: int,
        mode: str = "series",
        profile: str = "tokenspeed-full",
        spool_dir: str | Path | None = None,
        output_root: str | Path | None = None,
        expected_vocab_size: int = 129280,
        expected_observations: int = 64,
        num_layers: int = 43,
    ) -> None:
        self.enabled = bool(enabled)
        self.global_rank = int(global_rank)
        self.mode = mode
        self.profile = profile
        self.spool_dir = None if spool_dir is None else Path(spool_dir)
        self.output_root = None if output_root is None else Path(output_root)
        self.expected_vocab_size = int(expected_vocab_size)
        self.expected_observations = int(expected_observations)
        self.num_layers = int(num_layers)
        self._rows: list[dict[str, Any]] = []
        self._logits: dict[str, torch.Tensor] = {}
        self._active_row: dict[str, Any] | None = None
        self._checkpoint: dict[str, torch.Tensor] = {}
        self._finalized = False
        if not self.enabled:
            return
        if mode not in _MODES or profile not in _PROFILES:
            raise RuntimeError("invalid numerical-attribution recorder profile")
        if not 0 <= self.global_rank < 4:
            raise RuntimeError("numerical attribution requires global rank 0..3")
        if mode == "series" and self.spool_dir is None:
            raise RuntimeError("series attribution requires a spool directory")
        if mode == "checkpoint" and self.output_root is None:
            raise RuntimeError("checkpoint attribution requires an output root")

    def begin_forward(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            raise RuntimeError("disabled numerical-attribution recorder was invoked")
        if row.get("profile") != self.profile:
            raise RuntimeError("request attribution profile differs from recorder")
        self._active_row = dict(row)
        self._checkpoint = {}

    def record_boundary(
        self, name: str, tensor: torch.Tensor, *, preserve_all_rows: bool = False
    ) -> None:
        row = self._active_row
        if not self.enabled or row is None or not row.get("capture_checkpoint"):
            return
        value = tensor.detach()
        if preserve_all_rows:
            if value.ndim < 2:
                raise RuntimeError(f"{name} must contain token and hidden dimensions")
            value = value.reshape(-1, value.shape[-1])
        else:
            if value.ndim == 0:
                raise RuntimeError(f"{name} checkpoint tensor cannot be scalar")
            if name.endswith(("attn_post_hc", "block_post_hc")):
                value = value.reshape(-1, *value.shape[-2:])[-1]
            elif value.ndim >= 2:
                value = value.reshape(-1, value.shape[-1])[-1]
        if name in self._checkpoint:
            raise RuntimeError(f"duplicate numerical-attribution boundary: {name}")
        self._checkpoint[name] = value.to("cpu").contiguous()

    def _validate_checkpoint(self, call_input_length: int) -> None:
        expected_names = checkpoint_tensor_order(self.num_layers)
        if tuple(self._checkpoint) != expected_names:
            missing = [name for name in expected_names if name not in self._checkpoint]
            extra = [name for name in self._checkpoint if name not in expected_names]
            raise RuntimeError(
                f"checkpoint inventory differs: missing={missing[:5]} extra={extra[:5]}"
            )
        hidden = 4096
        for name, tensor in self._checkpoint.items():
            if name == "embedding_pre_hc":
                expected_shape = (call_input_length, hidden)
                expected_dtype = torch.bfloat16
            elif name.endswith(("attn_post_hc", "block_post_hc")):
                expected_shape = (4, hidden)
                expected_dtype = torch.bfloat16
            elif name.endswith("router_topk_ids"):
                expected_shape = (6,)
                expected_dtype = torch.int64
            elif name.endswith("router_weights"):
                expected_shape = (6,)
                expected_dtype = torch.float32
            elif name == "logits":
                expected_shape = (self.expected_vocab_size,)
                expected_dtype = torch.bfloat16
            else:
                expected_shape = (hidden,)
                expected_dtype = torch.bfloat16
            if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
                raise RuntimeError(
                    f"checkpoint {name} has {tuple(tensor.shape)}/{tensor.dtype}, "
                    f"expected {expected_shape}/{expected_dtype}"
                )

    def _finish_checkpoint(self, row: dict[str, Any], logits: torch.Tensor) -> None:
        if "logits" not in self._checkpoint:
            self.record_boundary("logits", logits)
        self._validate_checkpoint(int(row["call_input_length"]))
        destination = (
            self.output_root
            / "checkpoints"
            / self.profile
            / str(row["population"])
            / f"{row['case_id']}-step-{int(row['step']):02d}-rank-{self.global_rank}.safetensors"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"checkpoint destination exists: {destination}")
        save_file(self._checkpoint, str(destination))

    def record_sampling(
        self,
        logits: torch.Tensor,
        raw_argmax: torch.Tensor,
        rows: list[dict[str, Any]],
    ) -> None:
        if not self.enabled:
            raise RuntimeError("disabled numerical-attribution recorder was invoked")
        if logits.ndim != 2 or logits.shape[0] != len(rows):
            raise RuntimeError("attribution logits and request rows are not aligned")
        if logits.shape[1] != self.expected_vocab_size:
            raise RuntimeError("attribution logits do not contain the full vocabulary")
        if logits.dtype != torch.bfloat16:
            raise RuntimeError("TokenSpeed attribution logits must retain BF16")
        argmax_values = raw_argmax.detach().to("cpu").tolist()
        for index, row in enumerate(rows):
            vector = logits[index].detach().to("cpu").contiguous()
            if self.mode == "checkpoint":
                if row.get("capture_checkpoint"):
                    self._finish_checkpoint(row, vector)
                continue
            key = str(row["comparison_key"])
            if key in self._logits:
                raise RuntimeError(f"duplicate attribution comparison key: {key}")
            self._logits[key] = vector
            self._rows.append(
                {
                    **row,
                    "raw_argmax": int(argmax_values[index]),
                    "logit_tensor_key": key,
                    "logit_sha256": _tensor_sha256(vector),
                    "logit_dtype": str(vector.dtype),
                    "logit_shape": list(vector.shape),
                }
            )
        if self.mode == "series" and len(self._rows) == self.expected_observations:
            self.finalize_series()
        self._active_row = None
        self._checkpoint = {}

    def finalize_series(self) -> None:
        if self._finalized:
            raise RuntimeError("numerical-attribution series was finalized twice")
        if len(self._rows) != self.expected_observations:
            raise RuntimeError(
                f"attribution series has {len(self._rows)} observations, "
                f"expected {self.expected_observations}"
            )
        assert self.spool_dir is not None
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = self.spool_dir / f"rank-{self.global_rank}.safetensors"
        metadata_path = self.spool_dir / f"rank-{self.global_rank}.json"
        if tensor_path.exists():
            raise RuntimeError(f"attribution spool exists: {tensor_path}")
        save_file(self._logits, str(tensor_path))
        _write_json_exclusive(
            metadata_path,
            {
                "schema": "dsv4-numerical-attribution-rank-series/v1",
                "profile": self.profile,
                "rank": self.global_rank,
                "observations": self._rows,
            },
        )
        self._finalized = True


_lock = threading.Lock()
_recorder = DSV4NumericalAttributionRecorder(enabled=False, global_rank=0)


def configure_dsv4_numerical_attribution(
    *, enabled: bool, global_rank: int, expected_vocab_size: int
) -> DSV4NumericalAttributionRecorder:
    global _recorder
    with _lock:
        if enabled:
            _recorder = DSV4NumericalAttributionRecorder(
                enabled=True,
                global_rank=global_rank,
                mode=os.environ.get("DSV4_NUMERICAL_ATTRIBUTION_MODE", ""),
                profile=os.environ.get("DSV4_NUMERICAL_ATTRIBUTION_PROFILE", ""),
                spool_dir=os.environ.get("DSV4_NUMERICAL_ATTRIBUTION_SPOOL"),
                output_root=os.environ.get("DSV4_NUMERICAL_ATTRIBUTION_OUTPUT_ROOT"),
                expected_vocab_size=expected_vocab_size,
            )
        else:
            _recorder = DSV4NumericalAttributionRecorder(
                enabled=False, global_rank=global_rank
            )
        return _recorder


def numerical_attribution_enabled() -> bool:
    return _recorder.enabled


def numerical_attribution_checkpoint_active() -> bool:
    """Return whether the active attributed forward needs boundary tensors."""

    row = _recorder._active_row
    return bool(
        _recorder.enabled
        and _recorder.mode == "checkpoint"
        and row is not None
        and row.get("capture_checkpoint")
    )


def begin_numerical_attribution_forward(row: dict[str, Any]) -> None:
    _recorder.begin_forward(row)


def record_numerical_attribution_boundary(
    name: str, tensor: torch.Tensor, *, preserve_all_rows: bool = False
) -> None:
    _recorder.record_boundary(name, tensor, preserve_all_rows=preserve_all_rows)


def record_numerical_attribution_sampling(
    logits: torch.Tensor,
    raw_argmax: torch.Tensor,
    rows: list[dict[str, Any]],
) -> None:
    _recorder.record_sampling(logits, raw_argmax, rows)


__all__ = [
    "METADATA_KEY",
    "DSV4NumericalAttributionRecorder",
    "begin_numerical_attribution_forward",
    "build_sampling_directive",
    "checkpoint_tensor_order",
    "configure_dsv4_numerical_attribution",
    "numerical_attribution_checkpoint_active",
    "numerical_attribution_enabled",
    "record_numerical_attribution_boundary",
    "record_numerical_attribution_sampling",
]
