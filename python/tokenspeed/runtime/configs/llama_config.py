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


class LlamaConfig(TextConfigBase):
    model_type = "llama"

    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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
    attention_dropout: int | float | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)

        self.validate_architecture()

        if not 0.0 <= self.initializer_range <= 1.0:
            raise ValueError(
                "initializer_range must be between 0.0 and 1.0"
                f" (got {self.initializer_range})."
            )

    def validate_architecture(self):
        """Validate the attention dimensions."""

        if self.num_attention_heads <= 0:
            raise ValueError(
                "num_attention_heads must be greater than 0"
                f" (got {self.num_attention_heads})."
            )

        if self.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be greater than 0 (got {self.hidden_size})."
            )

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple "
                f"of the number of attention heads "
                f"({self.num_attention_heads})."
            )


__all__ = ["LlamaConfig"]
