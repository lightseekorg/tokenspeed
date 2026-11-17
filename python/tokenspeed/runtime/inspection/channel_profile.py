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

"""Opt-in, all-layer MLP activity profiling for an eager dense Qwen3 model.

Each record covers one complete forward and counts its physical token rows.
The resulting profile describes only the caller's sampled workload; a channel
that is quiet here is not necessarily quiet for other inputs or checkpoints.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.mlp import GatedMLP, GatedMLPFusion
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM


@dataclass
class _ChannelSums:
    rows: int
    sum_abs: torch.Tensor
    sum_square: torch.Tensor
    max_abs: torch.Tensor
    active_count: torch.Tensor
    sum_contribution: torch.Tensor
    max_contribution: torch.Tensor
    histogram: torch.Tensor


def _nonnegative_finite(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


class ChannelActivityProfiler:
    """Collect every MLP channel's activity on explicitly recorded forwards.

    Args:
        model: Unquantized dense Qwen3 on one device, held unchanged during
            profiling. It must be in evaluation mode.
        model_identity: Nonempty checkpoint identifier supplied by the caller.
            It is metadata, not a verified signature or safety assertion.
        execution: Only eager execution is supported; graph replay bypasses
            Python forward hooks.
        speculative: Must be False because draft/verify rows need separate
            attribution.
        monokernel: Must be False because fused layers bypass these hooks.
        activation_threshold: Absolute activation above this value counts as
            active. The threshold is fixed for all channels and records.
        histogram_edges: Positive, ascending absolute-activation boundaries.
            Counts use bins (-inf,e0], (e0,e1], ..., (eN,+inf).

    Enter the profiler to install hooks, then wrap each full model forward in
    ``record()``. Omit ``row_indices`` to profile all physical rows, or pass
    explicit zero-based physical row indices to profile a position-specific
    workload. ``profile()`` and ``freeze_watchlist()`` work after one or more
    complete records, including after detaching the profiler. Inference outputs
    are never replaced. Call on one forward thread and stream only.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        model_identity: str,
        execution: Literal["eager"],
        speculative: bool,
        monokernel: bool,
        activation_threshold: float,
        histogram_edges: Sequence[float],
    ) -> None:
        if execution != "eager" or speculative or monokernel:
            raise ValueError(
                "Profiling requires eager, non-speculative, non-monokernel execution"
            )
        if type(model) is not Qwen3ForCausalLM:
            raise ValueError("Profiling supports the ordinary dense Qwen3 model only")
        if model.mapping.world_size != 1 or model.mapping.pp_size != 1:
            raise ValueError("Profiling requires one device (TP=PP=DP=1)")
        if model.quant_config is not None or model._weight_update_active:
            raise ValueError("Profiling requires unquantized, unchanged weights")
        if model.training:
            raise ValueError("Profiling requires evaluation mode")
        if type(model_identity) is not str or not model_identity.strip():
            raise ValueError("model_identity must be a nonempty string")
        activation_threshold = _nonnegative_finite(
            activation_threshold, "activation_threshold"
        )
        edges = tuple(
            _nonnegative_finite(value, "histogram edge") for value in histogram_edges
        )
        if (
            not edges
            or edges[0] <= 0
            or any(right <= left for left, right in zip(edges, edges[1:]))
        ):
            raise ValueError("histogram_edges must be positive and strictly ascending")
        try:
            edge_tensor = torch.tensor(edges, dtype=torch.float32)
            threshold_tensor = torch.tensor(activation_threshold, dtype=torch.float32)
        except (OverflowError, RuntimeError) as error:
            raise ValueError("Profile thresholds must fit in float32") from error
        if (
            not bool(torch.isfinite(edge_tensor).all())
            or edge_tensor[0] <= 0
            or not bool((edge_tensor[1:] > edge_tensor[:-1]).all())
        ):
            raise ValueError(
                "histogram_edges must stay finite, positive and distinct in float32"
            )
        if not bool(torch.isfinite(threshold_tensor)) or (
            activation_threshold > 0 and threshold_tensor <= 0
        ):
            raise ValueError(
                "activation_threshold cannot overflow or underflow float32"
            )

        layers = tuple(model.model.layers)
        if not layers or len(layers) != model.config.num_hidden_layers:
            raise ValueError("Profiling requires all local Qwen3 layers")
        mlps = tuple(layer.mlp for layer in layers)
        first_weight = mlps[0].down_proj.weight
        if first_weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Profiling supports BF16 or FP32 tensors only")
        for layer, mlp in zip(layers, mlps):
            if (
                type(mlp) is not GatedMLP
                or mlp.fusion is not GatedMLPFusion.NONE
                or type(mlp.act_fn) is not SiluAndMul
                or mlp.act_fn.swiglu_limit is not None
                or type(mlp.gate_up_proj.quant_method) is not UnquantizedLinearMethod
                or type(mlp.down_proj.quant_method) is not UnquantizedLinearMethod
            ):
                raise ValueError("Profiling requires unfused, unquantized SwiGLU MLPs")
            down = mlp.down_proj.weight
            gate_up = mlp.gate_up_proj.weight
            if (
                down.ndim != 2
                or down.shape[1] <= 0
                or gate_up.ndim != 2
                or gate_up.shape[0] != 2 * down.shape[1]
                or gate_up.shape[1] != down.shape[0]
            ):
                raise ValueError("Unsupported MLP projection shapes")
            if any(
                weight.device != first_weight.device
                or weight.dtype != first_weight.dtype
                for weight in (down, gate_up)
            ):
                raise ValueError("MLP projections must share a dtype and device")

        self._model = model
        self._model_identity = model_identity
        self._layers = layers
        self._mlps = mlps
        self._device = first_weight.device
        self._activation_threshold = float(threshold_tensor.item())
        self._histogram_edges = tuple(float(value) for value in edge_tensor.tolist())
        self._edge_tensor = edge_tensor.to(self._device)
        self._down_norms = tuple(
            torch.linalg.vector_norm(mlp.down_proj.weight.detach().float(), dim=0)
            for mlp in mlps
        )
        if any(not bool(torch.isfinite(norm).all()) for norm in self._down_norms):
            raise ValueError("MLP down projection has nonfinite column norms")
        self._parameters = tuple(model.named_parameters())
        self._parameter_versions = tuple(
            parameter._version for _, parameter in self._parameters
        )
        self._channel_indices = tuple(
            torch.arange(norm.numel(), dtype=torch.int64, device=self._device)
            for norm in self._down_norms
        )
        self._totals = [self._empty_sums(norm.numel()) for norm in self._down_norms]
        self._handles: list[RemovableHandle] = []
        self._pending: dict[int, _ChannelSums] = {}
        self._armed = False
        self._record_count = 0
        self._physical_rows_seen = 0
        self._selected_row_records = 0
        self._row_indices: tuple[int, ...] | None = None
        self._row_selector: torch.Tensor | None = None
        self._physical_rows_in_record: int | None = None

    def _empty_sums(self, width: int) -> _ChannelSums:
        zeros = torch.zeros(width, dtype=torch.float64, device=self._device)
        return _ChannelSums(
            rows=0,
            sum_abs=zeros.clone(),
            sum_square=zeros.clone(),
            max_abs=zeros.clone(),
            active_count=torch.zeros(width, dtype=torch.int64, device=self._device),
            sum_contribution=zeros.clone(),
            max_contribution=zeros.clone(),
            histogram=torch.zeros(
                (width, len(self._histogram_edges) + 1),
                dtype=torch.int64,
                device=self._device,
            ),
        )

    def _require_eager(self) -> None:
        if self._device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Profiling does not support graph capture or replay")
        if self._model._weight_update_active or self._model.training:
            raise RuntimeError("Model state changed during profiling")

    def _require_unchanged_parameters(self) -> None:
        current = tuple(self._model.named_parameters())
        if len(current) != len(self._parameters) or any(
            current_name != original_name
            or current_parameter is not original_parameter
            or current_parameter._version != original_version
            for (current_name, current_parameter), (
                original_name,
                original_parameter,
            ), original_version in zip(
                current, self._parameters, self._parameter_versions
            )
        ):
            raise RuntimeError("Model parameters changed during profiling")

    def __enter__(self) -> ChannelActivityProfiler:
        if self._handles:
            raise RuntimeError("Profiler is already attached")
        self._require_eager()
        try:
            for layer_index, mlp in enumerate(self._mlps):
                self._handles.append(
                    mlp.act_fn.register_forward_hook(
                        partial(self._on_activation, layer_index)
                    )
                )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Remove all hooks; accumulated statistics remain available."""
        if self._armed:
            raise RuntimeError("Cannot detach during a record")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def _on_activation(
        self, layer_index: int, _module: nn.Module, _inputs: tuple, output: torch.Tensor
    ) -> None:
        if not self._armed:
            return
        self._require_eager()
        if layer_index in self._pending:
            raise RuntimeError(f"Layer {layer_index} was repeated in one record")
        width = self._down_norms[layer_index].numel()
        if (
            type(output) is not torch.Tensor
            or output.ndim != 2
            or output.shape[0] <= 0
            or output.shape[1] != width
            or output.device != self._device
        ):
            raise ValueError(f"Layer {layer_index} has an unsupported activation shape")
        if self._physical_rows_in_record is None:
            self._physical_rows_in_record = output.shape[0]
        elif output.shape[0] != self._physical_rows_in_record:
            raise ValueError("Profiled MLP layers have different physical row counts")
        if self._row_indices is not None:
            if self._row_indices[-1] >= output.shape[0]:
                raise ValueError("Selected row index is outside the physical forward")
            selector = self._row_selector
            if selector is None:
                raise RuntimeError("Selected row indices have no device selector")
            selected = output.detach().index_select(0, selector)
        else:
            selected = output.detach()
        activation = selected.float().abs()
        if not bool(torch.isfinite(activation).all()):
            raise ValueError(f"Layer {layer_index} has nonfinite activations")
        contribution = activation * self._down_norms[layer_index]
        bins = torch.bucketize(activation, self._edge_tensor)
        bin_count = len(self._histogram_edges) + 1
        flat_bins = self._channel_indices[layer_index].unsqueeze(0) * bin_count + bins
        histogram = torch.bincount(
            flat_bins.reshape(-1), minlength=width * bin_count
        ).reshape(width, bin_count)
        self._pending[layer_index] = _ChannelSums(
            rows=selected.shape[0],
            sum_abs=activation.sum(dim=0, dtype=torch.float64),
            sum_square=activation.square().sum(dim=0, dtype=torch.float64),
            max_abs=activation.max(dim=0).values.to(torch.float64),
            active_count=(activation > self._activation_threshold).sum(dim=0),
            sum_contribution=contribution.sum(dim=0, dtype=torch.float64),
            max_contribution=contribution.max(dim=0).values.to(torch.float64),
            histogram=histogram,
        )

    @contextmanager
    def record(self, row_indices: Sequence[int] | None = None) -> Iterator[None]:
        """Commit one full eager forward, profiling selected physical rows.

        ``row_indices=None`` includes every physical row. Otherwise supply
        nonempty, unique, ascending zero-based indices. The full forward still
        runs, and every layer must execute exactly once. Row positions refer
        to this forward's physical tensor, not a request or global token ID.
        """
        if not self._handles:
            raise RuntimeError("Profiler must be attached before recording")
        if self._armed:
            raise RuntimeError("Profiler already has an active record")
        self._require_eager()
        self._require_unchanged_parameters()
        if row_indices is not None:
            indices = tuple(row_indices)
            if (
                not indices
                or any(type(index) is not int or index < 0 for index in indices)
                or any(right <= left for left, right in zip(indices, indices[1:]))
            ):
                raise ValueError(
                    "row_indices must be unique ascending nonnegative integers"
                )
            self._row_indices = indices
            self._row_selector = torch.tensor(
                indices, dtype=torch.int64, device=self._device
            )
        else:
            self._row_indices = None
            self._row_selector = None
        self._physical_rows_in_record = None
        self._pending = {}
        self._armed = True
        try:
            with torch.no_grad():
                yield
            if len(self._pending) != len(self._layers):
                missing = sorted(set(range(len(self._layers))) - set(self._pending))
                raise RuntimeError(f"Model forward bypassed profiled layers: {missing}")
            for layer_index, part in self._pending.items():
                total = self._totals[layer_index]
                total.rows += part.rows
                total.sum_abs.add_(part.sum_abs)
                total.sum_square.add_(part.sum_square)
                torch.maximum(total.max_abs, part.max_abs, out=total.max_abs)
                total.active_count.add_(part.active_count)
                total.sum_contribution.add_(part.sum_contribution)
                torch.maximum(
                    total.max_contribution,
                    part.max_contribution,
                    out=total.max_contribution,
                )
                total.histogram.add_(part.histogram)
            self._record_count += 1
            physical_rows = self._physical_rows_in_record
            if physical_rows is None:
                raise RuntimeError("No physical rows were observed")
            self._physical_rows_seen += physical_rows
            self._selected_row_records += int(self._row_indices is not None)
        finally:
            self._armed = False
            self._pending = {}
            self._row_indices = None
            self._row_selector = None
            self._physical_rows_in_record = None

    def profile(self) -> dict:
        """Return JSON-ready statistics for every channel in every layer."""
        if self._armed or self._record_count == 0:
            raise RuntimeError("Profile requires at least one complete idle record")
        self._require_unchanged_parameters()
        layers = []
        for layer_index, total in enumerate(self._totals):
            rows = total.rows
            # Copy each aggregate once. Per-channel .item() would force one
            # device synchronization for every field of every channel.
            sum_abs = total.sum_abs.cpu().tolist()
            sum_square = total.sum_square.cpu().tolist()
            max_abs = total.max_abs.cpu().tolist()
            active_count = total.active_count.cpu().tolist()
            sum_contribution = total.sum_contribution.cpu().tolist()
            max_contribution = total.max_contribution.cpu().tolist()
            histograms = total.histogram.cpu().tolist()
            down_column_norms = self._down_norms[layer_index].cpu().tolist()
            down_norm_median = statistics.median(down_column_norms)
            channels = []
            for channel in range(len(sum_abs)):
                channels.append(
                    {
                        "layer": layer_index,
                        "channel": channel,
                        "sample_count": rows,
                        "mean_abs": sum_abs[channel] / rows,
                        "rms": math.sqrt(sum_square[channel] / rows),
                        "max_abs": max_abs[channel],
                        "active_count": active_count[channel],
                        "active_fraction": active_count[channel] / rows,
                        "down_column_norm": down_column_norms[channel],
                        "down_norm_ratio_to_layer_median": (
                            down_column_norms[channel] / down_norm_median
                            if down_norm_median > 0
                            else None
                        ),
                        "mean_contribution": sum_contribution[channel] / rows,
                        "max_contribution": max_contribution[channel],
                        "histogram_counts": histograms[channel],
                    }
                )
            layers.append(
                {
                    "layer": layer_index,
                    "down_norm_median": down_norm_median,
                    "channels": channels,
                }
            )
        return {
            "schema": "channel-activity-profile-v1",
            "model_identity": self._model_identity,
            "record_count": self._record_count,
            "physical_rows_seen": self._physical_rows_seen,
            "selected_row_records": self._selected_row_records,
            "activation_threshold": self._activation_threshold,
            "histogram_edges": list(self._histogram_edges),
            "layers": layers,
        }

    def rank_candidates(self, *, limit: int | None) -> list[dict]:
        """Rank measured channels by active fraction, then mean contribution.

        This ordering helps review a supplied workload's quiet channels; it
        does not classify any channel as inherently safe, unsafe or dormant.
        ``limit=None`` returns all channels. Ties use layer and channel index.
        """
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None")
        rows = [
            channel
            for layer in self.profile()["layers"]
            for channel in layer["channels"]
        ]
        rows.sort(
            key=lambda channel: (
                channel["active_fraction"],
                channel["mean_contribution"],
                channel["layer"],
                channel["channel"],
            )
        )
        return rows if limit is None else rows[:limit]

    def freeze_watchlist(
        self,
        *,
        max_active_fraction: float,
        max_mean_contribution: float,
        max_observed_contribution: float,
        min_down_norm_ratio: float,
        max_outlier_active_fraction: float,
        min_observed_rows: int,
        runtime_multiplier: float,
        contribution_floor: float,
    ) -> dict:
        """Select low-activity channels using explicit workload thresholds.

        A channel qualifies through either the low-activity/contribution gate
        or a large down-column norm ratio with limited observed activity.
        Ratios are undefined when a layer's median norm is zero, and such
        layers cannot qualify through the norm-ratio gate. The runtime limit
        is the greater of the observed maximum contribution times
        ``runtime_multiplier`` and ``contribution_floor``. It flags a later
        change relative to these observations; it does not certify benign or
        malicious behavior.
        """
        max_active_fraction = _nonnegative_finite(
            max_active_fraction, "max_active_fraction"
        )
        max_mean_contribution = _nonnegative_finite(
            max_mean_contribution, "max_mean_contribution"
        )
        max_observed_contribution = _nonnegative_finite(
            max_observed_contribution, "max_observed_contribution"
        )
        min_down_norm_ratio = _nonnegative_finite(
            min_down_norm_ratio, "min_down_norm_ratio"
        )
        max_outlier_active_fraction = _nonnegative_finite(
            max_outlier_active_fraction, "max_outlier_active_fraction"
        )
        runtime_multiplier = _nonnegative_finite(
            runtime_multiplier, "runtime_multiplier"
        )
        contribution_floor = _nonnegative_finite(
            contribution_floor, "contribution_floor"
        )
        if (
            max_active_fraction > 1
            or min_down_norm_ratio <= 0
            or max_outlier_active_fraction > 1
            or runtime_multiplier < 1
            or contribution_floor <= 0
        ):
            raise ValueError(
                "Require fractions <= 1, norm ratio > 0, multiplier >= 1, "
                "and floor > 0"
            )
        if type(min_observed_rows) is not int or min_observed_rows <= 0:
            raise ValueError("min_observed_rows must be a positive integer")
        profile = self.profile()
        if any(
            layer["channels"][0]["sample_count"] < min_observed_rows
            for layer in profile["layers"]
        ):
            raise ValueError("Profile has fewer rows than min_observed_rows")
        entries = []
        for layer in profile["layers"]:
            for channel in layer["channels"]:
                quiet = (
                    channel["active_fraction"] <= max_active_fraction
                    and channel["mean_contribution"] <= max_mean_contribution
                    and channel["max_contribution"] <= max_observed_contribution
                )
                ratio = channel["down_norm_ratio_to_layer_median"]
                norm_outlier = (
                    ratio is not None
                    and ratio >= min_down_norm_ratio
                    and channel["active_fraction"] <= max_outlier_active_fraction
                )
                if quiet or norm_outlier:
                    contribution_limit = max(
                        channel["max_contribution"] * runtime_multiplier,
                        contribution_floor,
                    )
                    if not math.isfinite(contribution_limit):
                        raise ValueError("Calculated contribution limit is nonfinite")
                    entries.append(
                        {
                            "layer": channel["layer"],
                            "channel": channel["channel"],
                            "contribution_limit": contribution_limit,
                        }
                    )
        entries.sort(key=lambda entry: (entry["layer"], entry["channel"]))
        return {
            "schema": "channel-watchlist-v1",
            "model_identity": self._model_identity,
            "entries": entries,
        }
