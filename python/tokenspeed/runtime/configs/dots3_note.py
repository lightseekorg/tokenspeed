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

"""Text-backbone configuration for ``Dot3NoteForCausalLM``.

Checkpoint fields selecting model behavior are required, rather than borrowed
from a different MLA model. The checkpoint omits the MoE grouping fields: this
architecture has one group, so both are explicitly supplied to the shared MoE.
Cache groups and the generation EOS policy are owned by runtime registration.
"""

from transformers.configuration_utils import PretrainedConfig


class Dots3NoteConfig(PretrainedConfig):
    model_type = "dots3_note"
    keys_to_ignore_at_inference = ["past_key_values"]
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_hidden_layers: int,
        layer_types: list[str],
        num_attention_heads: int,
        num_key_value_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_theta: float,
        swa_num_attention_heads: int,
        swa_num_key_value_heads: int,
        swa_q_lora_rank: int,
        swa_kv_lora_rank: int,
        swa_qk_nope_head_dim: int,
        swa_qk_rope_head_dim: int,
        swa_v_head_dim: int,
        swa_rope_theta: float,
        sliding_window_size: int,
        attention_gate_type: str,
        swa_attention_gate_type: str,
        apply_mla_qkv_lora_rescale: bool,
        attention_bias: bool,
        attention_dropout: float,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        n_routed_experts: int,
        n_shared_experts: int,
        num_experts_per_tok: int,
        first_k_dense_replace: int,
        moe_layer_freq: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool,
        scoring_func: str,
        topk_method: str,
        hidden_act: str,
        rms_norm_eps: float,
        max_position_embeddings: int,
        tie_word_embeddings: bool,
        rope_scaling: dict | None = None,
        pad_token_id: int | None = None,
        vision_config: dict | None = None,
        audio_config: dict | None = None,
        **kwargs,
    ) -> None:
        if len(layer_types) != num_hidden_layers or set(layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("layer_types must classify every backbone layer")
        if (
            attention_gate_type != "headwise"
            or swa_attention_gate_type != "headwise"
            or not apply_mla_qkv_lora_rescale
            or attention_bias
            or attention_dropout != 0
            or rope_scaling is not None
            or tie_word_embeddings
        ):
            raise ValueError(
                "dots3_note requires headwise gates, LoRA rescaling, untied "
                "embeddings, and no attention bias/dropout or RoPE scaling"
            )
        if (
            num_attention_heads != num_key_value_heads
            or swa_num_attention_heads != swa_num_key_value_heads
            or qk_rope_head_dim != 64
            or swa_qk_rope_head_dim != 64
            or (index_n_heads, index_head_dim, index_topk) != (64, 128, 2048)
            or sliding_window_size != 513
        ):
            raise ValueError("Unsupported dots3_note attention geometry")
        if (
            scoring_func != "sigmoid"
            or topk_method != "noaux_tc"
            or not norm_topk_prob
            or routed_scaling_factor != 1.0
            or hidden_act != "silu"
            or kwargs.pop("n_group", 1) != 1
            or kwargs.pop("topk_group", 1) != 1
        ):
            raise ValueError(
                "dots3_note requires normalized sigmoid MoE with one group"
            )
        kwargs.setdefault("architectures", ["Dot3NoteForCausalLM"])
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.pad_token_id = pad_token_id
        self.vision_config = vision_config
        self.audio_config = audio_config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = list(layer_types)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_q_lora_rank = swa_q_lora_rank
        self.swa_kv_lora_rank = swa_kv_lora_rank
        self.swa_qk_nope_head_dim = swa_qk_nope_head_dim
        self.swa_qk_rope_head_dim = swa_qk_rope_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.swa_rope_theta = swa_rope_theta
        self.sliding_window_size = sliding_window_size
        self.attention_gate_type = attention_gate_type
        self.swa_attention_gate_type = swa_attention_gate_type
        self.apply_mla_qkv_lora_rescale = apply_mla_qkv_lora_rescale
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = 1
        self.topk_group = 1
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings

    @property
    def paged_cache_layer_types(self) -> list[str]:
        return self.layer_types
