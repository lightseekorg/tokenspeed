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

"""MiniMax-M2 model configuration definitions."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from tokenspeed.runtime.configs.utils import rope_config_validation

logger = logging.get_logger(__name__)


class MiniMaxM2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniMaxM2Model`].
    It is used to instantiate MiniMax-M2 family models according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    """

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.block_sparse_moe.gate": "colwise_rep",
        "layers.*.block_sparse_moe.experts.*.w1": "colwise",
        "layers.*.block_sparse_moe.experts.*.w2": "rowwise",
        "layers.*.block_sparse_moe.experts.*.w3": "colwise",
    }

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=1536,
        num_hidden_layers=62,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=196608,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5_000_000,
        rope_scaling=None,
        rotary_dim=64,
        attention_bias=False,
        attention_dropout=0.0,
        # MoE
        num_local_experts=256,
        num_experts_per_tok=8,
        scoring_func="sigmoid",
        use_routing_bias=True,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        # QK-Norm
        use_qk_norm=True,
        qk_norm_type="per_layer",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rotary_dim = rotary_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate rope
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        # MoE
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        # QK-Norm
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        # Preserve extra public checkpoint metadata through PretrainedConfig
        # without making it part of the MiniMax-M2 serving runtime.
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MiniMaxM2Config"]
