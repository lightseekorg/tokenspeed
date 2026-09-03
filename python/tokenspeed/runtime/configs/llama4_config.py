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

"""Local Llama 4 text checkpoint configuration."""

from typing import Any

from tokenspeed.runtime.configs.base_config import TextConfigBase


class Llama4TextConfig(TextConfigBase):
    """Llama 4 text configuration used by compatible Eagle3 checkpoints."""

    model_type = "llama4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.shared_expert.gate_proj": "colwise",
        "layers.*.feed_forward.shared_expert.up_proj": "colwise",
        "layers.*.feed_forward.shared_expert.down_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "packed_rowwise",
        "layers.*.feed_forward.experts.down_proj": "colwise",
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "grouped_gemm",
        "layers.*.feed_forward.experts.down_proj": "grouped_gemm",
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
        "layers.*.feed_forward.router": "ep_router",
    }

    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    moe_layers: list[int] | None = None
    interleave_moe_layer_step: int = 1
    use_qk_norm: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    rope_parameters: dict | None = None
    no_rope_layers: list[int] | None = None
    no_rope_layer_interval: int = 4
    attention_chunk_size: int | None = 8192
    layer_types: list[str] | None = None
    attn_temperature_tuning: bool = True
    floor_scale: int = 8192
    attn_scale: float = 0.1
    attention_bias: bool = False

    def __post_init__(self, **kwargs: Any) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        default_no_rope_layers = [
            int((layer_idx + 1) % self.no_rope_layer_interval != 0)
            for layer_idx in range(self.num_hidden_layers)
        ]
        self.no_rope_layers = self.no_rope_layers or default_no_rope_layers
        self.head_dim = (
            self.head_dim
            if self.head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )
        self.moe_layers = (
            self.moe_layers
            if self.moe_layers is not None
            else list(
                range(
                    self.interleave_moe_layer_step - 1,
                    self.num_hidden_layers,
                    self.interleave_moe_layer_step,
                )
            )
        )
        if self.layer_types is None:
            self.layer_types = [
                "chunked_attention" if no_rope else "full_attention"
                for no_rope in self.no_rope_layers
            ]

        super().__post_init__(**kwargs)


__all__ = ["Llama4TextConfig"]
