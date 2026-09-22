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

from __future__ import annotations

import torch
from tokenspeed_kernel import moe_topk, pack_topk_router_logits

from tokenspeed.runtime.layers.moe.topk import (
    BypassedTopKOutput,
    ExpertLocationDispatchInfo,
    StandardTopKOutput,
    TopK,
    TopKOutput,
    TopKOutputFormat,
)


class DeepseekV4TopK(TopK):
    def __init__(
        self,
        top_k: int,
        renormalize: bool,
        correction_bias: torch.Tensor | None,
        routed_scaling_factor: float,
        output_format: TopKOutputFormat,
        hash_routing: bool,
    ) -> None:
        super().__init__(
            top_k=top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            output_format=output_format,
        )
        self.hash_routing = hash_routing

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        output_format: TopKOutputFormat | None = None,
        num_token_non_padded: torch.Tensor | None = None,
        expert_location_dispatch_info: ExpertLocationDispatchInfo | None = None,
        routing_correction_bias: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> TopKOutput:
        output_format = (
            output_format or self.topk_config.output_format or TopKOutputFormat.STANDARD
        )
        correction_bias = (
            self.topk_config.correction_bias
            if routing_correction_bias is None
            else routing_correction_bias
        )
        topk_weights, topk_ids = moe_topk(
            router_logits,
            self.topk_config.top_k,
            "sqrt_softplus",
            "hash" if self.hash_routing else "topk",
            self.topk_config.renormalize,
            self.topk_config.routed_scaling_factor,
            correction_bias=correction_bias,
            hash_indices_table=hash_indices_table,
            input_ids=input_ids,
        )
        if output_format == TopKOutputFormat.BYPASSED:
            output_scale = topk_weights.sum(dim=-1, keepdim=True)
            packed_logits = pack_topk_router_logits(
                topk_weights,
                topk_ids,
                router_logits.shape[1],
            )
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=packed_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
                output_scale=output_scale,
            )
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)


__all__ = ["DeepseekV4TopK"]
