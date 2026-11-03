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

"""How a model runs its decoder layers in one forward.

A model's forward hands its layer loop to a layer stack, chosen at
construction through the model's ``layer_stack_cls``. The stack owns which
implementation runs each layer for a step; the model owns everything around
the loop (embedding, final norm, logits). The default stack runs every layer
in order, as the model was written. A model with its own forward defines the
stack its forward calls, with the boundary values its layers pass.

``prepare`` runs once at startup, after the weights are loaded and before
the KV cache is sized or a graph is captured. A stack that runs some layers
through other kernels (a fused decoder layer, for one) prepares them there
and decides per step, from the step's shape alone, which layers they run;
every other step runs the default loop. See
``docs/design/fused-decoder-layers.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.context import ForwardContext, TargetCaptureSink

if TYPE_CHECKING:
    from tokenspeed.runtime.models.base.transformer_model import BaseTransformerModel


def notify_target_capture(
    sink: TargetCaptureSink,
    capture_idx_map: dict[int, int],
    layer_idx: int,
    aux_hidden_states: list[torch.Tensor],
) -> None:
    """Hand the tap capture layer ``layer_idx`` just appended to the
    forward's sink.

    Skipped when the capture layer appended no tap: an idle forward skips
    BaseDecoderLayer's attention block, so nothing was appended. Layers that
    append an empty tap on an idle forward still hand over the zero-row tensor.

    Args:
        sink: The forward's ``ctx.target_capture_sink``.
        capture_idx_map: Capture layer index to the draft's positional tap
            index; layers absent from it are not handed over.
        layer_idx: The capture layer that just ran.
        aux_hidden_states: The taps collected so far in this forward.
    """
    capture_idx = capture_idx_map.get(layer_idx)
    if capture_idx is None or len(aux_hidden_states) != capture_idx + 1:
        return
    sink.on_target_capture(capture_idx, aux_hidden_states[capture_idx])


class DecoderLayerStack:
    """Run every decoder layer in order: ``BaseTransformerModel``'s loop."""

    def heap_bytes(self, model: BaseTransformerModel, world_size: int) -> int:
        """Communication-heap bytes ``prepare`` will allocate, reserved when
        the heap is created; none here."""
        return 0

    def prepare(self, model: BaseTransformerModel) -> None:
        """Prepare other implementations of the model's layers, once at
        startup before the KV cache is sized or a graph is captured; none
        here."""

    def run(
        self,
        model: BaseTransformerModel,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model's layers on one step. A model with its own forward
        overrides ``run`` with the boundary values its layers pass.

        Args:
            model: The model whose layers run.
            positions: The step's positions.
            hidden_states: The first layer's input.
            ctx: The step's forward context.
            residual: The first layer's residual, or None.
            aux_hidden_states: Taps for a draft; capture layers append theirs.

        Returns:
            ``(hidden_states, residual)`` after the last layer.
        """
        for i, layer in enumerate(model.layers):
            capture = i in model.layers_to_capture
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states=aux_hidden_states if capture else None,
            )
            if capture and ctx.target_capture_sink is not None:
                notify_target_capture(
                    ctx.target_capture_sink,
                    model._dflash_capture_idx_map,
                    i,
                    aux_hidden_states,
                )
        return hidden_states, residual
