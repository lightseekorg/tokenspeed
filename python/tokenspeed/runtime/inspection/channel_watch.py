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

"""Opt-in, sparse channel observation during eager dense Qwen3 inference.

A workload profile supplies the channels to sample and their contribution
limits. Forward hooks gather those channels from the MLP activation output.
After the forward, callers can export the rows and channels that exceeded a
limit to locate differences from the profiling workload.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.mlp import GatedMLP, GatedMLPFusion
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True)
class RowIdentity:
    """Caller-supplied identity of one physical row in the recorded forward."""

    request_id: str
    token_position: int


@dataclass(frozen=True)
class LayerChannelSample:
    """Sparse tensors from one MLP, with shape [selected rows, channels]."""

    layer: int
    channels: tuple[int, ...]
    activations: torch.Tensor
    contribution_estimates: torch.Tensor
    contribution_limits: torch.Tensor
    exceeds_limit: torch.Tensor


@dataclass(frozen=True)
class ChannelWatchSnapshot:
    """Independent sparse samples from one completed forward.

    ``contribution_estimates`` are abs(activation) times the L2 norm of the
    corresponding down-projection column. This estimates the magnitude of a
    channel's contribution at the MLP output. ``rows`` names all physical rows.
    ``selected_rows`` maps each sample row back to its original physical row.
    Tensors remain on the model device until the caller exports events.
    """

    model_identity: str
    rows: tuple[RowIdentity, ...]
    selected_rows: tuple[int, ...]
    samples: tuple[LayerChannelSample, ...]
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ChannelWatchEvent:
    """One watched channel above its frozen contribution threshold."""

    model_identity: str
    request_id: str
    token_position: int
    physical_row: int
    layer: int
    channel: int
    activation: float
    contribution_estimate: float
    contribution_limit: float


@dataclass(frozen=True)
class _LayerWatchState:
    """Device-resident watch settings for one MLP layer."""

    channels: tuple[int, ...]
    width: int
    dtype: torch.dtype
    indices: torch.Tensor
    limits: torch.Tensor
    down_norms: torch.Tensor


def _validate_execution_model(
    model: Qwen3ForCausalLM,
    *,
    execution: Literal["eager"],
    speculative: bool,
    monokernel: bool,
) -> None:
    if execution != "eager" or speculative or monokernel:
        raise ValueError(
            "Channel observation requires eager, non-speculative, "
            "non-monokernel execution"
        )
    if type(model) is not Qwen3ForCausalLM:
        raise ValueError("Channel observation supports ordinary dense Qwen3 only")
    if model.mapping.world_size != 1 or model.mapping.pp_size != 1:
        raise ValueError("Channel observation requires TP=PP=DP=1")
    if model.quant_config is not None or model._weight_update_active:
        raise ValueError("Channel observation requires loaded, unquantized weights")
    if model.training:
        raise ValueError("Channel observation requires evaluation mode")


def _watchlist_entries(
    watchlist: Mapping[str, object], model_identity: str
) -> list[object]:
    if type(model_identity) is not str or not model_identity.strip():
        raise ValueError("model_identity must be a nonempty caller-supplied string")
    if not isinstance(watchlist, Mapping) or set(watchlist) != {
        "schema",
        "model_identity",
        "entries",
    }:
        raise ValueError("Expected a channel-watchlist-v1 artifact")
    if watchlist["schema"] != "channel-watchlist-v1":
        raise ValueError("Unsupported channel watchlist schema")
    if watchlist["model_identity"] != model_identity:
        raise ValueError(
            "Watchlist model identity does not match the loaded model identity"
        )
    entries = watchlist["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Watchlist must contain at least one channel entry")
    return entries


def _supported_layers_and_device(
    model: Qwen3ForCausalLM,
) -> tuple[Sequence[nn.Module], torch.device]:
    layers = model.model.layers
    if not layers or len(layers) != model.config.num_hidden_layers:
        raise ValueError("Channel observation requires all local Qwen3 layers")
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise ValueError("Channel observation requires all model weights on one device")
    device = next(iter(devices))
    for layer in layers:
        mlp = layer.mlp
        if type(mlp) is not GatedMLP:
            raise ValueError("Channel observation requires ordinary GatedMLP layers")
        if (
            mlp.fusion is not GatedMLPFusion.NONE
            or type(mlp.act_fn) is not SiluAndMul
            or mlp.act_fn.swiglu_limit is not None
            or type(mlp.gate_up_proj.quant_method) is not UnquantizedLinearMethod
            or type(mlp.down_proj.quant_method) is not UnquantizedLinearMethod
        ):
            raise ValueError(
                "Channel observation requires an unfused, unquantized SwiGLU MLP"
            )
        down = mlp.down_proj.weight
        gate_up = mlp.gate_up_proj.weight
        if (
            down.dtype not in (torch.bfloat16, torch.float32)
            or gate_up.dtype != down.dtype
            or down.device != device
            or gate_up.device != device
        ):
            raise ValueError(
                "Channel observation requires unfused BF16/FP32 SwiGLU MLPs"
            )
        if (
            down.ndim != 2
            or gate_up.ndim != 2
            or gate_up.shape[0] != 2 * down.shape[1]
            or gate_up.shape[1] != down.shape[0]
        ):
            raise ValueError("Unsupported MLP projection shapes")
    return layers, device


def _group_watchlist_entries(
    entries: Sequence[object], layers: Sequence[nn.Module]
) -> dict[int, list[tuple[int, float]]]:
    by_layer: dict[int, list[tuple[int, float]]] = {}
    previous = (-1, -1)
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "layer",
            "channel",
            "contribution_limit",
        }:
            raise ValueError("Malformed channel watchlist entry")
        layer_id = entry["layer"]
        channel = entry["channel"]
        limit = entry["contribution_limit"]
        if type(layer_id) is not int or not 0 <= layer_id < len(layers):
            raise ValueError("Watchlist layer is outside the model")
        width = layers[layer_id].mlp.down_proj.weight.shape[1]
        if type(channel) is not int or not 0 <= channel < width:
            raise ValueError("Watchlist channel is outside its MLP")
        if isinstance(limit, bool) or not isinstance(limit, (float, int)):
            raise ValueError("contribution_limit must be positive and finite")
        try:
            limit_value = float(limit)
        except (OverflowError, ValueError) as exc:
            raise ValueError("contribution_limit must be positive and finite") from exc
        if (
            not math.isfinite(limit_value)
            or limit_value < math.ldexp(1.0, -149)
            or limit_value > torch.finfo(torch.float32).max
        ):
            raise ValueError(
                "contribution_limit must be positive and finite in float32"
            )
        if (layer_id, channel) <= previous:
            raise ValueError("Watchlist entries must be unique and sorted")
        previous = (layer_id, channel)
        by_layer.setdefault(layer_id, []).append((channel, limit_value))
    return by_layer


def _build_layer_watch_state(
    mlp: GatedMLP,
    selected: Sequence[tuple[int, float]],
    device: torch.device,
) -> _LayerWatchState:
    channels = tuple(channel for channel, _ in selected)
    indices = torch.tensor(channels, dtype=torch.int64, device=device)
    limits = torch.tensor(
        [limit for _, limit in selected], dtype=torch.float32, device=device
    )
    down = mlp.down_proj.weight
    with torch.no_grad():
        down_norms = torch.linalg.vector_norm(
            down.index_select(1, indices).float(), dim=0
        )
        if not bool(torch.isfinite(down_norms).all()):
            raise ValueError("Watched down-projection norms must be finite")
    return _LayerWatchState(
        channels=channels,
        width=down.shape[1],
        dtype=down.dtype,
        indices=indices,
        limits=limits,
        down_norms=down_norms,
    )


def _sample_layer_output(
    output: torch.Tensor,
    state: _LayerWatchState,
    row_indices: torch.Tensor | None,
    layer_id: int,
) -> LayerChannelSample:
    if row_indices is None:
        activation = output.detach().index_select(1, state.indices)
    else:
        # Gather only watched rows and channels from the full MLP output.
        activation = output.detach()[row_indices[:, None], state.indices[None, :]]
    contribution = activation.float().abs() * state.down_norms
    return LayerChannelSample(
        layer=layer_id,
        channels=state.channels,
        activations=activation,
        contribution_estimates=contribution,
        contribution_limits=state.limits,
        exceeds_limit=contribution > state.limits,
    )


def _copy_layer_sample(sample: LayerChannelSample) -> LayerChannelSample:
    return LayerChannelSample(
        layer=sample.layer,
        channels=sample.channels,
        activations=sample.activations.clone(),
        contribution_estimates=sample.contribution_estimates.clone(),
        contribution_limits=sample.contribution_limits.clone(),
        exceeds_limit=sample.exceeds_limit.clone(),
    )


class ChannelWatch:
    """Record watched channels in one eager Qwen3 forward at a time.

    Attach with a context manager, wrap each forward in ``record(rows)``, then
    call ``snapshot()``. Pass ``selected_rows`` to sample only specified
    physical token rows. ``None`` samples them all. After the forward,
    ``events(snapshot)`` exports over-limit samples with the request IDs and
    token positions supplied in ``rows``. The caller's ``model_identity``
    must match the label in the watchlist.

    Use the model's forward thread and stream with fixed weights. The watcher
    checks parameter identity and version. Raw storage writes via ``.data``
    bypass those checks. Hooks require eager execution with unfused,
    unquantized MLPs on one device.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        watchlist: Mapping[str, object],
        model_identity: str,
        execution: Literal["eager"],
        speculative: bool,
        monokernel: bool,
    ) -> None:
        _validate_execution_model(
            model,
            execution=execution,
            speculative=speculative,
            monokernel=monokernel,
        )
        entries = _watchlist_entries(watchlist, model_identity)
        layers, self._device = _supported_layers_and_device(model)
        by_layer = _group_watchlist_entries(entries, layers)

        self._model = model
        self._model_identity = model_identity
        self._layer_ids = tuple(by_layer)
        self._layer_states: dict[int, _LayerWatchState] = {}
        for layer_id, selected in by_layer.items():
            mlp = layers[layer_id].mlp
            self._layer_states[layer_id] = _build_layer_watch_state(
                mlp, selected, self._device
            )

        self._parameters = tuple(
            (name, parameter, parameter._version)
            for name, parameter in model.named_parameters()
        )

        self._owner = object()
        self._handles: list[RemovableHandle] = []
        self._armed = False
        self._complete = False
        self._rows: tuple[RowIdentity, ...] = ()
        self._selected_rows: tuple[int, ...] = ()
        self._row_indices: torch.Tensor | None = None
        self._samples: dict[int, LayerChannelSample] = {}

    def _require_eager(self) -> None:
        if self._device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ChannelWatch does not support graph capture or replay")

    def _require_unchanged_parameters(self) -> None:
        current = tuple(self._model.named_parameters())
        if len(current) != len(self._parameters):
            raise RuntimeError("Model parameters changed since watchlist construction")
        for (expected_name, expected, version), (name, parameter) in zip(
            self._parameters, current
        ):
            if (
                name != expected_name
                or parameter is not expected
                or parameter._version != version
            ):
                raise RuntimeError(
                    "Model parameters changed since watchlist construction"
                )

    def __enter__(self) -> ChannelWatch:
        self._require_eager()
        self._require_unchanged_parameters()
        if self._handles:
            raise RuntimeError("ChannelWatch is already attached")
        try:
            for layer_id in self._layer_ids:
                activation = self._model.model.layers[layer_id].mlp.act_fn
                self._handles.append(
                    activation.register_forward_hook(self._make_hook(layer_id))
                )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Detach hooks. An idle, closed watcher may be attached again."""
        if self._armed:
            raise RuntimeError("Cannot close ChannelWatch during record()")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def record(
        self,
        rows: Sequence[RowIdentity],
        selected_rows: Sequence[int] | None = None,
    ) -> Iterator[None]:
        """Arm one forward, optionally sampling a subset of physical rows."""
        self._require_eager()
        if not self._handles or self._armed:
            raise RuntimeError("record() requires an attached, idle ChannelWatch")
        self._rows = ()
        self._selected_rows = ()
        self._row_indices = None
        self._samples.clear()
        self._complete = False
        row_tuple = tuple(rows)
        if not row_tuple or any(
            type(row) is not RowIdentity
            or not isinstance(row.request_id, str)
            or not row.request_id
            or type(row.token_position) is not int
            or row.token_position < 0
            for row in row_tuple
        ):
            raise ValueError(
                "rows must contain valid caller-supplied RowIdentity values"
            )
        if selected_rows is None:
            selected = tuple(range(len(row_tuple)))
            row_indices = None
        else:
            selected = tuple(selected_rows)
            if (
                not selected
                or any(
                    type(row) is not int or not 0 <= row < len(row_tuple)
                    for row in selected
                )
                or len(set(selected)) != len(selected)
            ):
                raise ValueError(
                    "selected_rows must contain unique physical row indices"
                )
            row_indices = torch.tensor(selected, dtype=torch.int64, device=self._device)
        if self._model._weight_update_active or self._model.training:
            raise RuntimeError("Cannot observe during a weight update or training")
        self._require_unchanged_parameters()
        self._rows = row_tuple
        self._selected_rows = selected
        self._row_indices = row_indices
        self._armed = True
        try:
            with torch.no_grad():
                yield
            if set(self._samples) != set(self._layer_ids):
                missing = sorted(set(self._layer_ids) - set(self._samples))
                raise RuntimeError(
                    f"Forward bypassed watched MLP hooks: {missing}; "
                    "graph replay is unsupported"
                )
            self._complete = True
        finally:
            self._armed = False

    def _make_hook(self, layer_id: int):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            if not self._armed:
                return
            self._require_eager()
            if layer_id in self._samples:
                raise RuntimeError(
                    f"Expected one forward per record(), repeated layer {layer_id}"
                )
            state = self._layer_states[layer_id]
            if (
                not isinstance(output, torch.Tensor)
                or output.ndim != 2
                or output.shape[0] != len(self._rows)
                or output.shape[1] != state.width
                or output.dtype != state.dtype
                or output.device != self._device
            ):
                raise ValueError(
                    "Watched MLP output does not match physical row mapping"
                )
            self._samples[layer_id] = _sample_layer_output(
                output, state, self._row_indices, layer_id
            )

        return hook

    def snapshot(self) -> ChannelWatchSnapshot:
        """Return independent sparse tensors from the last completed forward."""
        self._require_eager()
        if self._armed or not self._complete:
            raise RuntimeError("No completed channel record is available")
        samples = tuple(
            _copy_layer_sample(self._samples[layer_id]) for layer_id in self._layer_ids
        )
        return ChannelWatchSnapshot(
            model_identity=self._model_identity,
            rows=self._rows,
            selected_rows=self._selected_rows,
            samples=samples,
            _owner=self._owner,
        )

    def events(self, snapshot: ChannelWatchSnapshot) -> tuple[ChannelWatchEvent, ...]:
        """Export only over-limit records after the forward; host access may sync."""
        if (
            type(snapshot) is not ChannelWatchSnapshot
            or snapshot._owner is not self._owner
        ):
            raise ValueError("Snapshot does not belong to this ChannelWatch")
        events: list[ChannelWatchEvent] = []
        for sample in snapshot.samples:
            # nonzero may synchronize to learn its output shape, but this is
            # explicitly after the forward. It yields row-major coordinates,
            # preserving the prior event order without copying quiet values.
            coordinates = torch.nonzero(sample.exceeds_limit, as_tuple=False)
            if coordinates.numel() == 0:
                continue
            sample_rows = coordinates[:, 0]
            column_ids = coordinates[:, 1]
            activations = sample.activations[sample_rows, column_ids].float().cpu()
            contributions = sample.contribution_estimates[sample_rows, column_ids].cpu()
            limits = sample.contribution_limits.index_select(0, column_ids).cpu()
            host_coordinates = coordinates.cpu().tolist()
            for event_id, (sample_row, column_id) in enumerate(host_coordinates):
                physical_row = snapshot.selected_rows[sample_row]
                row = snapshot.rows[physical_row]
                events.append(
                    ChannelWatchEvent(
                        model_identity=snapshot.model_identity,
                        request_id=row.request_id,
                        token_position=row.token_position,
                        physical_row=physical_row,
                        layer=sample.layer,
                        channel=sample.channels[column_id],
                        activation=float(activations[event_id]),
                        contribution_estimate=float(contributions[event_id]),
                        contribution_limit=float(limits[event_id]),
                    )
                )
        return tuple(events)
