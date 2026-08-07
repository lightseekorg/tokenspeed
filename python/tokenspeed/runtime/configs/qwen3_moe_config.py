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

"""Qwen3 MoE model configuration definitions."""

from dataclasses import field

from tokenspeed.runtime.configs.qwen3_config import Qwen3Config


class Qwen3MoeConfig(Qwen3Config):
    """Configuration for Qwen3 MoE causal LMs such as Qwen3-30B-A3B."""

    model_type = "qwen3_moe"

    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    shared_expert_intermediate_size: int = 0
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list = field(default_factory=list)

    def __post_init__(self, **kwargs):
        self.mlp_only_layers = (
            [] if self.mlp_only_layers is None else self.mlp_only_layers
        )

        super().__post_init__(**kwargs)


__all__ = ["Qwen3MoeConfig"]
