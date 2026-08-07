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


class LongcatFlashConfig(TextConfigBase):
    model_type = "longcat_flash"

    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 10000000.0
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "num_experts_per_tok": "moe_topk",
        "intermediate_size": "ffn_hidden_size",
    }
    base_model_tp_plan = {
        "layers.*.self_attn.*.q_b_proj": "colwise",
        "layers.*.self_attn.*.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.*.kv_b_proj": "colwise",
        "layers.*.self_attn.*.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts.identity_expert": "moe_identity_expert",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlps.*.gate_proj": "colwise",
        "layers.*.mlps.*.up_proj": "colwise",
        "layers.*.mlps.*.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 131072
    hidden_size: int = 6144
    num_hidden_layers: int = 56
    num_layers: int = 28
    num_attention_heads: int = 64
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    ffn_hidden_size: int = 12288
    q_lora_rank: int | None = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    head_dim: int = 64
    v_head_dim: int = 128
    qk_head_dim: int | None = None
    moe_topk: int = 12
    n_routed_experts: int = 512
    zero_expert_num: int = 256
    expert_ffn_hidden_size: int = 2048
    routed_scaling_factor: float = 6.0
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs: Any) -> None:
        if self.qk_head_dim is None:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        super().__post_init__(**kwargs)


__all__ = ["LongcatFlashConfig"]
