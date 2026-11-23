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

"""FlashInfer MoEEpLayer adapter for SM120 W4A8 routed experts."""

from __future__ import annotations

import torch
from torch import nn


def configure_sm120_cake(plan: dict, group, capacity: int) -> None:
    if plan["apply_kernel_name"] != "flashinfer_sm12x_mxfp4_moe_apply":
        raise ValueError("SM120 Cake requires the SM120 FlashInfer MXFP4 expert plan")
    if plan.get("a2a_backend") != "none":
        raise ValueError("SM120 Cake supplies its own expert transport")

    plan.update(
        a2a_backend="flashinfer_moe_ep",
        process_group=group,
        persistent_max_num_tokens_per_gpu=capacity,
        weight_preprocessor=prepare_sm120_cake_weights,
        warmup=warmup_sm120_cake,
    )


def prepare_sm120_cake_weights(plan, w) -> None:
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        preprocess_mega_weights,
    )

    limit = None if w.swiglu_arg is None else w.swiglu_arg.limit
    if limit is not None and limit != int(limit):
        raise ValueError("SM120 Cake requires an integer SwiGLU clamp")
    if w.w13_input_layout != "concatenated" or w._swiglu_form() != "standard":
        raise ValueError("SM120 Cake requires canonical weights and standard SwiGLU")

    group = plan["process_group"]
    capacity = plan["persistent_max_num_tokens_per_gpu"]
    weights = MoEWeightPack(
        w.w13_weight,
        w.w2_weight,
        w.w13_weight_scale.view(torch.float8_e8m0fnu),
        w.w2_weight_scale.view(torch.float8_e8m0fnu),
    )
    packed = preprocess_mega_weights(
        weights,
        hidden_size=w.hidden_size,
        intermediate_size=w.intermediate_size,
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=group.size(),
            rank=group.rank(),
            process_group=group,
            device=torch.cuda.current_device(),
        ),
        fleet_params=FleetParams(
            num_experts=w.num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=w.hidden_size,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                intermediate_size=w.intermediate_size,
                top_k=w.top_k,
                swiglu_limit=0 if limit is None else int(limit),
                transport_table="all_gin",
            ),
            quantize_input=True,
            preprocess_weights=False,
            transformed_weights=packed,
        ),
    )
    # Keep the module parameters and backend aliased to one expert weight copy.
    w.w13_weight = nn.Parameter(packed.w1_fp4, requires_grad=False)
    w.w13_weight_scale = nn.Parameter(packed.w1_sfb, requires_grad=False)
    w.w2_weight = nn.Parameter(packed.w2_fp4, requires_grad=False)
    w.w2_weight_scale = nn.Parameter(packed.w2_sfb, requires_grad=False)
    w._moe_backend_state = Sm120CakeRoutedExperts(
        layer, capacity, w.hidden_size, w.top_k, w.num_experts, group.rank()
    )


def warmup_sm120_cake(plan, w) -> None:
    w._moe_backend_state.warmup()


class Sm120CakeRoutedExperts:
    """Stage a rank-uniform token bucket without host synchronization."""

    def __init__(self, layer, capacity, hidden, top_k, experts, rank):
        self.layer = layer
        self.capacity = capacity
        self.hidden = hidden
        self.top_k = top_k
        self.experts = experts
        self.rank = rank
        self._protocol_error = None

    def warmup(self) -> None:
        if self._protocol_error is not None:
            return
        rows = min(16, self.capacity)
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.zeros((rows, self.hidden), device=device, dtype=torch.bfloat16)
        ids = (
            (torch.arange(rows * self.top_k, device=device) + self.rank * rows)
            % self.experts
        ).view(rows, self.top_k)
        weights = torch.full(
            (rows, self.top_k), 1.0 / self.top_k, device=device, dtype=torch.float32
        )
        self._stage(x, ids, weights, rows)
        # FlashInfer 0.7.5 exposes only a synchronizing status read at the
        # backend boundary. Retain the device flag for a stream-ordered check.
        self._protocol_error = self.layer._workspace.storage["protocol_error"].view(
            torch.int32
        )
        self._compute()
        torch.cuda.current_stream().synchronize()

    def run(self, x, ids, weights, maximum: int) -> torch.Tensor:
        if self._protocol_error is None:
            raise RuntimeError("SM120 Cake workspace must be warmed up before use")
        if not 0 <= x.shape[0] <= maximum <= self.capacity:
            raise ValueError("SM120 Cake requires a rank-uniform token bound")
        self._stage(x, ids, weights, maximum)
        return self._compute()

    def _stage(self, x, ids, weights, maximum: int) -> None:
        from flashinfer.moe_ep import MoEEpTensors

        inputs = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
        self.layer.stage_inputs(inputs, compile_tokens_per_rank=maximum)

    def _compute(self) -> torch.Tensor:
        output = self.layer.compute_staged(output=None)
        # The next forward can clear the flag in the shared workspace.
        torch._assert_async(self._protocol_error == 0, "SM120 Cake protocol error")
        return output
