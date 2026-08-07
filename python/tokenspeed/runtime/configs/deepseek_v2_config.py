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

"""Local DeepSeek-V2 checkpoint configuration."""

from typing import Any

from tokenspeed.runtime.configs.base_config import TextConfigBase


class DeepseekV2Config(TextConfigBase):
    """Configuration fields consumed by the DeepSeek-V2-compatible loaders."""

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None
    first_k_dense_replace: int = 0
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    n_group: int | None = None
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    routed_scaling_factor: float = 1.0
    topk_group: int | None = None
    topk_method: str | None = "greedy"
    norm_topk_prob: bool | None = False
    v_head_dim: int = 128
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int = 1407

    def __post_init__(self, **kwargs: Any) -> None:
        # DeepSeek MLA stores only the rotary portion in the generic head_dim
        # field; qk_nope_head_dim is accounted for separately by the runtime.
        self.head_dim = self.qk_rope_head_dim
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)
        self.validate_architecture()

    def validate_architecture(self) -> None:
        """Validate the attention geometry used by the local MLA loader."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of "
                f"the number of attention heads ({self.num_attention_heads})."
            )


__all__ = ["DeepseekV2Config"]
