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

"""Eager, single-device Qwen3 final-MLP observation and local suffix replay.

This is an opt-in evidence tool, not a detector or a serving callback API.
Native hooks copy the actual intermediates; inference outputs are not replaced.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
from tokenspeed.runtime.layers.mlp import GatedMLP, GatedMLPFusion
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True)
class FinalMLPObservation:
    """Independent tensors from one completed record, on the model's device.

    The first axis retains the whole physical forward shape, except for
    ``selected_logits`` and ``logit_rows``. ``gate_up`` is [gate | up].
    ``logit_rows`` names the effective hidden rows used by the processor, not
    request IDs or absolute token positions. Tensors are caller-owned copies.
    Replay is bound to the observer and unchanged model that produced them.
    """

    mlp_input: torch.Tensor
    residual_before_mlp: torch.Tensor
    gate_up: torch.Tensor
    activation: torch.Tensor
    mlp_output: torch.Tensor
    normalized_hidden: torch.Tensor
    selected_logits: torch.Tensor
    logit_rows: torch.Tensor
    layer_id: int
    layer_boundary_norm: str
    selects_rows: bool
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class FinalMLPReplay:
    """Three suffix computations, with the original physical GEMM shape.

    ``suffix_logits`` reuses the recorded down output. ``recomputed_logits``
    recomputes down from the recorded activation. ``intervened_logits`` first
    zeros the requested activation channels. Compare both baselines with the
    observation's logits before interpreting an intervention. No live tensors,
    model weights, or KV state are changed.
    """

    suffix_logits: torch.Tensor
    recomputed_logits: torch.Tensor
    intervened_logits: torch.Tensor
    recomputed_mlp_output: torch.Tensor
    intervened_mlp_output: torch.Tensor


class FinalMLPObserver:
    """Observe one eager Qwen3 forward at a time using private fixed buffers.

    Args:
        model: An unquantized dense Qwen3 on one device, with unchanged weights
            and kernel/numerics configuration throughout this object's life.
        num_rows: Exact physical token rows in every recorded forward.
        num_logits_rows: Exact selected rows returned by the logits processor.
        execution: Only ``"eager"`` is supported. Graph capture and replay are
            refused; Python hooks do not execute on an existing graph replay.
        speculative: Must be False; draft/verify attribution is unsupported.
        monokernel: Must be False; fused layers may bypass the observed modules.

    Enter the observer to install hooks, and wrap each forward in ``record()``.
    ``snapshot()`` owns copies that survive later records. Use on the model's
    forward thread and stream, without concurrent forwards, reloads or updates.
    When no observer is attached, the model has no additional hooks or work.
    """

    _REQUIRED = frozenset(("post_attention", "activation", "mlp", "norm", "logits"))

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        num_rows: int,
        num_logits_rows: int,
        execution: Literal["eager"],
        speculative: bool,
        monokernel: bool,
    ) -> None:
        if execution != "eager" or speculative or monokernel:
            raise ValueError(
                "Observation requires eager, non-speculative, non-monokernel execution"
            )
        if type(model) is not Qwen3ForCausalLM:
            raise ValueError(
                "Observation supports the ordinary dense Qwen3ForCausalLM only"
            )
        if model.mapping.world_size != 1 or model.mapping.pp_size != 1:
            raise ValueError("Observation requires one device (TP=PP=DP=1)")
        if model.quant_config is not None or model._weight_update_active:
            raise ValueError(
                "Observation requires unquantized weights outside a weight update"
            )
        if (
            type(num_rows) is not int
            or type(num_logits_rows) is not int
            or num_rows <= 0
            or not 0 < num_logits_rows <= num_rows
        ):
            raise ValueError("Require 0 < num_logits_rows <= num_rows")
        layer = model.model.layers[-1]
        mlp = layer.mlp
        if (
            type(mlp) is not GatedMLP
            or mlp.fusion is not GatedMLPFusion.NONE
            or type(mlp.act_fn) is not SiluAndMul
            or mlp.act_fn.swiglu_limit is not None
            or type(mlp.gate_up_proj.quant_method) is not UnquantizedLinearMethod
            or type(mlp.down_proj.quant_method) is not UnquantizedLinearMethod
        ):
            raise ValueError("Observation requires an unfused, unquantized SwiGLU MLP")
        if (
            type(model.model.norm) is not RMSNorm
            or type(layer.post_attention_layernorm) is not RMSNorm
            or type(model.logits_processor) is not LogitsProcessor
            or type(model.lm_head.quant_method)
            not in (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)
        ):
            raise ValueError(
                "Observation requires the ordinary unquantized RMSNorm/head suffix"
            )
        processor = model.logits_processor
        if (
            processor.dp_sampling_enabled
            or processor.do_argmax
            or processor._sampled_shard_start is not None
        ):
            raise ValueError("Observation requires ordinary full-vocabulary logits")
        mode = layer.comm_manager.layer_boundary_norm
        if mode not in ("fused", "unfused"):
            raise ValueError(f"Unsupported residual-add/norm mode: {mode}")
        weight = mlp.down_proj.weight
        if weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Observation supports BF16 or FP32 tensors only")
        weights = (
            mlp.gate_up_proj.weight,
            weight,
            layer.post_attention_layernorm.weight,
            model.model.norm.weight,
            model.lm_head.weight,
        )
        if any(w.device != weight.device or w.dtype != weight.dtype for w in weights):
            raise ValueError(
                "Observed projections and norms must share a dtype and device"
            )
        self._model = model
        self._layer = layer
        self._mlp = mlp
        self._device = weight.device
        self._dtype = weight.dtype
        self._num_rows = num_rows
        self._num_logits_rows = num_logits_rows
        self._norm_mode = mode
        self._owner = object()
        self._handles: list[RemovableHandle] = []
        self._armed = False
        self._complete = False
        self._seen: set[str] = set()
        self._selects_rows = False
        hidden, intermediate = weight.shape
        shapes = {
            "mlp_input": (num_rows, hidden),
            "residual_before_mlp": (num_rows, hidden),
            "gate_up": (num_rows, 2 * intermediate),
            "activation": (num_rows, intermediate),
            "mlp_output": (num_rows, hidden),
            "normalized_hidden": (num_rows, hidden),
            "selected_logits": (num_logits_rows, model.config.vocab_size),
        }
        self._buffers = {
            name: torch.empty(shape, dtype=self._dtype, device=self._device)
            for name, shape in shapes.items()
        }
        self._logit_rows = torch.empty(
            num_logits_rows, dtype=torch.int64, device=self._device
        )

    def _require_eager(self) -> None:
        if self._device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FinalMLPObserver does not support graph capture or replay"
            )

    def __enter__(self) -> FinalMLPObserver:
        self._require_eager()
        if self._handles:
            raise RuntimeError("Observer is already attached")
        try:
            self._handles.append(
                self._layer.post_attention_layernorm.register_forward_hook(
                    self._post_attention
                )
            )
            self._handles.append(
                self._mlp.act_fn.register_forward_hook(self._activation)
            )
            self._handles.append(self._mlp.register_forward_hook(self._mlp_output))
            self._handles.append(
                self._model.model.norm.register_forward_hook(self._norm)
            )
            self._handles.append(
                self._model.logits_processor.register_forward_hook(
                    self._logits, with_kwargs=True
                )
            )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Remove this observer's hooks; repeated closes are harmless."""
        if self._armed:
            raise RuntimeError("Cannot close an observer during record()")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def record(self) -> Iterator[None]:
        """Record exactly one forward; incomplete or bypassed hooks fail closed.

        A failed forward invalidates the previous workspace. Snapshot copies
        already returned remain valid. This context disables autograd.
        """
        self._require_eager()
        if not self._handles or self._armed:
            raise RuntimeError("record() requires an attached, idle observer")
        if (
            self._model._weight_update_active
            or self._layer.comm_manager.layer_boundary_norm != self._norm_mode
        ):
            raise RuntimeError(
                "Model configuration changed while observation was attached"
            )
        self._complete = False
        self._seen.clear()
        self._armed = True
        try:
            with torch.no_grad():
                yield
            if self._seen != self._REQUIRED:
                missing = sorted(self._REQUIRED - self._seen)
                raise RuntimeError(
                    f"Forward bypassed required observation hooks: {missing}; graph replay is unsupported"
                )
            self._complete = True
        finally:
            self._armed = False

    def _mark(self, name: str) -> bool:
        if not self._armed:
            return False
        self._require_eager()
        if name in self._seen:
            raise RuntimeError(f"Expected one forward per record(), repeated {name}")
        return True

    def _copy(self, name: str, value: torch.Tensor) -> None:
        destination = self._buffers[name]
        if (
            value.shape != destination.shape
            or value.dtype != destination.dtype
            or value.device != destination.device
        ):
            raise ValueError(
                f"{name} does not match the prepared shape, dtype and device"
            )
        destination.copy_(value.detach())

    def _post_attention(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        if self._mark("post_attention"):
            self._copy("mlp_input", output[0])
            # The final fused norm may overwrite this residual storage.
            self._copy("residual_before_mlp", output[1])
            self._seen.add("post_attention")

    def _activation(
        self, module: nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        if self._mark("activation"):
            self._copy("gate_up", inputs[0])
            self._copy("activation", output)
            self._seen.add("activation")

    def _mlp_output(
        self, module: nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        if self._mark("mlp"):
            self._copy("mlp_output", output)
            self._seen.add("mlp")

    def _norm(
        self, module: nn.Module, inputs: tuple, output: torch.Tensor | tuple
    ) -> None:
        if self._mark("norm"):
            self._copy(
                "normalized_hidden", output[0] if isinstance(output, tuple) else output
            )
            self._seen.add("norm")

    def _logits(
        self,
        module: nn.Module,
        inputs: tuple,
        kwargs: dict,
        output: LogitsProcessorOutput,
    ) -> None:
        if not self._mark("logits"):
            return
        metadata = inputs[3] if len(inputs) > 3 else kwargs["logits_metadata"]
        if (
            metadata.query_shard is not None
            or metadata.logits_rows_selected
            or metadata.capture_hidden_mode != CaptureHiddenMode.NULL
        ):
            raise ValueError(
                "Observation does not support narrowed, sharded or speculative hidden rows"
            )
        if (
            output.next_token_logits is None
            or output.logits_layout_plan is not None
            or output.vocab_shard is not None
        ):
            raise ValueError("Observation requires full selected logits")
        gather = metadata.gather_ids
        self._selects_rows = gather is not None and gather.shape[0] != self._num_rows
        if self._selects_rows:
            if gather.shape != self._logit_rows.shape:
                raise ValueError("Logits row selection does not match num_logits_rows")
            self._logit_rows.copy_(gather)
        else:
            if self._num_logits_rows != self._num_rows:
                raise ValueError(
                    "Unselected logits require num_logits_rows == num_rows"
                )
            torch.arange(self._num_rows, device=self._device, out=self._logit_rows)
        self._copy("selected_logits", output.next_token_logits)
        self._seen.add("logits")

    @torch.no_grad()
    def snapshot(self) -> FinalMLPObservation:
        """Copy the completed record; callers own all returned tensor storage."""
        self._require_eager()
        if self._armed or not self._complete:
            raise RuntimeError("No completed record is available")
        return FinalMLPObservation(
            **{name: value.clone() for name, value in self._buffers.items()},
            logit_rows=self._logit_rows.clone(),
            layer_id=self._layer.layer_id,
            layer_boundary_norm=self._norm_mode,
            selects_rows=self._selects_rows,
            _owner=self._owner,
        )

    def _suffix(
        self, down: torch.Tensor, observation: FinalMLPObservation
    ) -> torch.Tensor:
        norm = self._model.model.norm
        residual = observation.residual_before_mlp.clone()
        # Direct forward calls intentionally bypass the recording hooks. The
        # norm and head kernel implementations remain the model's originals.
        if self._norm_mode == "fused":
            normalized, _ = norm.forward(down.clone(), residual)
        else:
            normalized = norm.forward(down + residual)
        if observation.selects_rows:
            normalized = normalized[observation.logit_rows]
        metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
        return self._model.logits_processor._get_logits(
            normalized,
            self._model.lm_head,
            metadata,
            require_full_vocab=False,
        )

    @torch.no_grad()
    def replay(
        self, observation: FinalMLPObservation, *, ablate_channels: Sequence[int]
    ) -> FinalMLPReplay:
        """Replay the suffix and zero selected final-MLP activation channels.

        Args:
            observation: A snapshot from this observer and unchanged model.
            ablate_channels: Unique local channel indices; empty means no
                intervention. Full physical rows are replayed, preserving GEMM
                shape. This is not subtraction of a reconstructed rank-one
                write, which has different finite-precision semantics.
        Returns:
            Separate recorded-output, recomputed-output, and intervention
            logits, plus both recomputed down outputs. Compare baseline errors
            before treating a difference as an intervention effect.
        """
        self._require_eager()
        if self._armed or observation._owner is not self._owner:
            raise ValueError("Replay requires an idle observer and its own snapshot")
        if (
            self._model._weight_update_active
            or self._layer.comm_manager.layer_boundary_norm != self._norm_mode
        ):
            raise RuntimeError(
                "Model configuration changed while observation was attached"
            )
        channels = tuple(ablate_channels)
        width = observation.activation.shape[1]
        if any(type(i) is not int or i < 0 or i >= width for i in channels) or len(
            set(channels)
        ) != len(channels):
            raise ValueError("ablate_channels must contain unique in-range integers")
        activation = observation.activation.clone()
        recomputed, _ = self._mlp.down_proj.forward(activation)
        intervention = activation.clone()
        intervention[:, list(channels)] = 0
        changed, _ = self._mlp.down_proj.forward(intervention)
        return FinalMLPReplay(
            suffix_logits=self._suffix(observation.mlp_output, observation),
            recomputed_logits=self._suffix(recomputed, observation),
            intervened_logits=self._suffix(changed, observation),
            recomputed_mlp_output=recomputed,
            intervened_mlp_output=changed,
        )
