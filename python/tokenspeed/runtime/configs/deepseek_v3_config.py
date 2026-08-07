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

from typing import Any

from tokenspeed.runtime.configs.base_config import TextConfigBase


class DeepseekV3Config(TextConfigBase):
    model_type = "deepseek_v3"

    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "num_mtp_layers": "num_nextn_predict_layers",
    }

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int | None = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int | None = 128
    qk_nope_head_dim: int = 128
    n_group: int | None = 8
    topk_group: int | None = 4
    num_experts_per_tok: int | None = 8
    first_k_dense_replace: int | None = 3
    norm_topk_prob: bool | None = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    rope_interleave: bool | None = True
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    num_nextn_predict_layers: int = 1

    def __post_init__(self, **kwargs: Any) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim
        super().__post_init__(**kwargs)


__all__ = ["DeepseekV3Config"]
