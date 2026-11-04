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

"""Muse Spark's decoder-layer stack: every layer regular, or the MXFP4-expert
layers as monokernel launches on the decode shapes the group serves.

``MuseSparkModel.layer_stack_cls``. The model's forward embeds the tokens,
picks the row layout and runs the final norm; the stack runs the layers in
between (``models/base/layer_stack.py`` describes the hook). Monokernel
layers are another implementation of the same layers, chosen per step: the
regular layers before and after them run as written, and every step the stack
does not serve runs the regular loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.monokernel import MonokernelLayer, PlanKey

from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.monokernel import (
    MAX_WAIT_DEADLINE_SECONDS,
    MIN_WAIT_DEADLINE_SECONDS,
    agree_served_shapes,
    current_deployment,
    deployment_decline_reasons,
    wait_deadline_seconds,
)
from tokenspeed.runtime.models.base.layer_stack import DecoderLayerStack
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.failure_records import rank_record_path

if TYPE_CHECKING:
    from tokenspeed.runtime.models.muse_spark import MuseSparkModel

logger = logging.getLogger(__name__)

# Requests per decode step that the monokernel layers are prepared for: most
# speculative decode steps carry one or two requests.
FUSED_REQUESTS = (1, 2)
RECIPE = "muse_spark"


class MuseSparkLayerStack(DecoderLayerStack):
    """Muse Spark's decoder layers for one forward.

    Monokernels cover the contiguous layers with MXFP4 experts (the fused
    range); the layers before and after them stay regular, and the hand-offs
    at both ends use the regular layers' own boundary: the first monokernel
    layer's query comes from its attention's regular prologue, and the last
    launch ends with the next layer's input norm.
    """

    def __init__(self) -> None:
        # The prepared monokernel layers and the layer indices they run, set by
        # ``prepare`` when the group serves any shape; empty otherwise.
        self.fused_layers: list[MonokernelLayer] = []
        self.fused_indices = range(0)
        # (requests, tokens per request) decode shapes every rank serves.
        self.served_shapes: frozenset[tuple[int, int]] = frozenset()
        # The launch state every monokernel layer of the group shares.
        self.state: object | None = None

    @staticmethod
    def _candidate_indices(model: MuseSparkModel) -> range:
        """The contiguous MXFP4-expert layers, with a regular layer before
        and after them, or an empty range."""
        indices = [
            i for i, layer in enumerate(model.layers) if layer.mlp.quantized_experts
        ]
        if (
            not indices
            or indices[0] == 0
            or indices[-1] + 1 >= len(model.layers)
            or indices != list(range(indices[0], indices[-1] + 1))
        ):
            return range(0)
        return range(indices[0], indices[-1] + 1)

    def heap_bytes(self, model: MuseSparkModel, world_size: int) -> int:
        """Symmetric heap bytes the monokernel state takes (0 when no layer
        can run as a monokernel); the Iris heap is created with them."""
        from tokenspeed_kernel.ops.monokernel import (
            monokernel_state_heap_bytes,
        )

        if not self._candidate_indices(model):
            return 0
        return monokernel_state_heap_bytes(
            recipe=RECIPE, world_size=world_size, max_requests=max(FUSED_REQUESTS)
        )

    @staticmethod
    def selection_traits(server_args: dict, mapping) -> dict:
        """The shape traits a fused layer leaf is selected by: the decode
        rows per request (the verify width, 1 without speculation), the
        attention TP and MoE EP sizes, and the KV cache dtype."""
        return {
            "verify_width": (
                int(server_args["speculative_num_draft_tokens"])
                if server_args["speculative_algorithm"]
                else 1
            ),
            "tp_size": mapping.attn.tp_size,
            "ep_size": mapping.moe.ep_size,
            "kv_cache_dtype": server_args["kv_cache_dtype"],
        }

    def prepare_layers(
        self,
        model: MuseSparkModel,
        state: object,
        traits: dict,
        deadline_seconds: float,
    ) -> list[MonokernelLayer | None]:
        """One prepared fused layer per candidate layer (None where no leaf
        serves ``traits``), each for every request count in
        ``FUSED_REQUESTS``, reading the layer's weights and the next layer's
        input norm and query projection in place; its waits for other GPUs
        give up after ``deadline_seconds``."""
        from tokenspeed_kernel.ops.monokernel import prepare_monokernel_layer

        candidates = self._candidate_indices(model)
        prepared = []
        for index in candidates:
            layer, next_layer = model.layers[index], model.layers[index + 1]
            next_kind = next_layer.layer_kind if index + 1 in candidates else "regular"
            prepared.append(
                prepare_monokernel_layer(
                    recipe=RECIPE,
                    plan_keys=[
                        PlanKey(requests, layer.layer_kind, next_kind)
                        for requests in FUSED_REQUESTS
                    ],
                    traits=traits,
                    weights=layer.fused_weights(next_layer),
                    state=state,
                    deadline_seconds=deadline_seconds,
                )
            )
        return prepared

    def prepare(self, model: MuseSparkModel) -> None:
        """Prepare the monokernel layers and agree, across the attention
        group, on the decode shapes they serve; log the startup report. Runs
        once, after the Iris heap exists and before the KV cache is sized or a
        graph is captured."""
        from tokenspeed_kernel.ops.monokernel import prepare_monokernel_state

        candidates = self._candidate_indices(model)
        if self.state is not None or not candidates:
            return
        args = global_server_args_dict
        mapping = model.mapping
        group = mapping.attn.tp_group
        rank_in_group = group.index(dist.get_rank())
        state = prepare_monokernel_state(
            recipe=RECIPE,
            group=process_group_manager.get_device_process_group(group),
            rank_in_group=rank_in_group,
            max_requests=max(FUSED_REQUESTS),
            failure_record_path=rank_record_path("monokernel", dist.get_rank()),
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        if state is None:
            # No leaf serves this platform: nothing was allocated.
            return
        self.state = state
        traits = self.selection_traits(args, mapping)
        shapes = [(requests, traits["verify_width"]) for requests in FUSED_REQUESTS]
        deployment = deployment_decline_reasons(
            current_deployment(args, forward_side_stream=model.alt_stream is not None)
        )
        reasons = {shape: dict.fromkeys(deployment) for shape in shapes}
        deadline = wait_deadline_seconds(float(args["watchdog_timeout"]))
        prepared = self.prepare_layers(model, state, traits, deadline)
        for index, fused in zip(candidates, prepared):
            for shape in shapes:
                if fused is None:
                    reasons[shape]["no monokernel"] = None
                    continue
                for reason in fused.decline_reasons(shape[0]):
                    reasons[shape][f"layer {index}: {reason}"] = None
        process_group_manager.init_process_group(group, backend="gloo")
        served = agree_served_shapes(
            shapes,
            {shape: list(lines) for shape, lines in reasons.items()},
            process_group_manager.get_process_group("gloo", group),
        )
        if rank_in_group == 0:
            logger.info(served.report("monokernel layers"))
            logger.info(
                f"monokernel layers: a wait for another GPU gives up after "
                f"{deadline:g} s (half of --watchdog-timeout "
                f"{args['watchdog_timeout']:g} s, kept within "
                f"{MIN_WAIT_DEADLINE_SECONDS:g}-{MAX_WAIT_DEADLINE_SECONDS:g} s)"
            )
        if served.shapes:
            self.fused_layers = prepared
            self.fused_indices = candidates
            self.served_shapes = served.shapes

    def fused_range(
        self,
        model: MuseSparkModel,
        rows: int,
        ctx: ForwardContext,
        owned_global_rows: int | None,
    ) -> range:
        """The layers this step runs as monokernels, or an empty range: the
        regular loop runs every layer.

        A step runs monokernels only as a decode forward (it carries decode
        metadata: an extend of the same shape does not) on replicated rows
        whose padded (requests, tokens per request) shape every rank agreed
        to serve. Nothing rank-local enters, so a captured graph records the
        same choice on every rank.
        """
        if (
            not self.served_shapes
            or owned_global_rows is not None
            or ctx.bs <= 0
            or rows % ctx.bs
            or (ctx.bs, rows // ctx.bs) not in self.served_shapes
        ):
            return range(0)
        first = model.layers[self.fused_indices.start]
        if first.self_attn.attn.decode_kv_view(ctx) is None:
            return range(0)
        return self.fused_indices

    def run(  # type: ignore[override]
        self,
        model: MuseSparkModel,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        attention_input: torch.Tensor | None,
        owned_global_rows: int | None,
        aux_hidden_states: list[torch.Tensor],
        taps_live: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model's layers on one step, with the boundary values Muse
        Spark's layers pass (its forward overrides the default arguments).

        Args:
            model: The model whose layers run.
            positions: The step's positions.
            hidden_states: The FP32 residual rows (owned rows under
                ``owned_global_rows``).
            ctx: The step's forward context.
            attention_input: The first layer's normed input, or None for it
                to normalize its own.
            owned_global_rows: The step's global rows when each rank holds
                only its owned rows, else None.
            aux_hidden_states: Taps for a draft; capture layers append theirs.
            taps_live: Whether this step captures taps.

        Returns:
            ``(hidden_states, attention_input)`` after the last layer.
        """
        fused = self.fused_range(model, hidden_states.shape[0], ctx, owned_global_rows)
        prepared_query = None
        for index, layer in enumerate(model.layers):
            if taps_live and index in model.layers_to_capture:
                model._capture(
                    index, hidden_states, owned_global_rows, aux_hidden_states, ctx
                )
            if index in fused:
                if index == fused.start:
                    prepared_query = layer.prepare_fused_query(
                        positions, attention_input, ctx
                    )
                hidden_states, prepared_query, attention_input = layer.forward_fused(
                    self.fused_layers[index - fused.start],
                    positions,
                    prepared_query,
                    hidden_states,
                    ctx,
                    model.layers[index + 1],
                    index + 1 in fused,
                )
                continue
            # Each FFN boundary emits the next layer's attention input rows.
            next_input_layernorm = (
                model.layers[index + 1].input_layernorm
                if index + 1 < len(model.layers)
                else None
            )
            # Preserve the owned path's direct call without Module hooks.
            layer_forward = layer.forward if owned_global_rows is not None else layer
            hidden_states, attention_input = layer_forward(
                positions,
                hidden_states,
                ctx,
                attention_input,
                next_input_layernorm,
                owned_global_rows,
            )
        return hidden_states, attention_input
