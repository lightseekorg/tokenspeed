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

"""Inference-only Qwen3 MoE model compatible with HuggingFace weights."""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    MoECheckpointLoader,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from tokenspeed.runtime.models.qwen3_5_moe import (
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
    _is_moe_layer,
)
from tokenspeed.runtime.utils import add_prefix


class Qwen3MoeDecoderLayer(Qwen3DecoderLayer):
    @property
    def is_moe_layer(self) -> bool:
        return _is_moe_layer(self.layer_id, self.config)

    @property
    def is_prev_moe_layer(self) -> bool:
        return _is_moe_layer(self.layer_id - 1, self.config)

    def resolve_mlp(self, prefix: str) -> nn.Module:
        if self.is_moe_layer:
            return Qwen3_5MoeSparseMoeBlock(
                config=self.config,
                mapping=self.mapping,
                quant_config=self.quant_config,
                layer_index=self.layer_id,
                prefix=add_prefix("mlp", prefix),
            )
        return Qwen3_5MoeMLP(
            parallelism="dense",
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            mapping=self.mapping,
            quant_config=self.quant_config,
            reduce_results=False,
            prefix=add_prefix("mlp", prefix),
        )

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        if not self.is_moe_layer:
            return super().forward_mlp(
                hidden_states, residual, ctx, num_global_tokens, max_num_tokens_per_gpu
            )
        # The MoE block runs its own communication.
        return self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx)


class Qwen3MoeModel(Qwen3Model):
    layer_cls = Qwen3MoeDecoderLayer


class Qwen3MoeForCausalLM(Qwen3ForCausalLM):
    model_cls = Qwen3MoeModel

    def get_expert_checkpoint_loader(self) -> MoECheckpointLoader:
        return build_moe_checkpoint_loader(
            params_dict=dict(self.named_parameters(remove_duplicate=False)),
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            fused_schema=ExpertCheckpointSchema(
                gate_up_fused_name="gate_up_proj",
                down_proj_name="down_proj",
            ),
            num_experts=self.config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )


EntryClass = Qwen3MoeForCausalLM
