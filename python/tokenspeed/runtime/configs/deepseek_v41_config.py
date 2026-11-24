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

"""Checkpoint configuration for DeepSeek V4.1's text and vision backbones."""

from transformers import PretrainedConfig


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"
    base_config_key = "text_config"

    def __init__(self, **kwargs):
        # Transformers standardizes RoPE before assigning additional kwargs.
        for name in (
            "max_position_embeddings",
            "rope_theta",
            "head_dim",
            "hidden_size",
            "num_attention_heads",
            "rope_scaling",
        ):
            if name in kwargs:
                setattr(self, name, kwargs[name])
        self.rope_parameters = kwargs.get("rope_scaling", {})
        super().__init__(**kwargs)
        self.expert_dtype: str | None = kwargs.get("expert_dtype")
        self.moe_quant_algo: str | None = kwargs.get("moe_quant_algo")
        self.engram_embed_quant: str = kwargs.get("engram_embed_quant", "mxfp8")
        # These are architectural facts, not V4 defaults: V4.1 has learned,
        # ungrouped routing in every layer and no hash-routed MoE layers.
        self.num_hash_layers = 0
        self.n_group = 1
        self.topk_group = 1

    @property
    def kv_source_layers(self):
        return self.kv_source_layer_ids

    @property
    def index_source_layers(self):
        return self.index_source_layer_ids

    @property
    def candidate_source_layer(self):
        return self.candidate_source_layer_id

    @property
    def ngram_context_len(self):
        return 3 if getattr(self, "engram_layer_ids", ()) else 0


class DeepseekV41VisionConfig(PretrainedConfig):
    """ViT and aligner dimensions with Flash checkpoint defaults."""

    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        *,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        downsample_ratio: int = 3,
        rope_theta: float = 10000.0,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.rope_theta = rope_theta
        super().__init__(**kwargs)


class DeepseekV41Config(PretrainedConfig):
    """DeepSeek V4.1 text and vision configuration."""

    model_type = "deepseek_v41"

    def __init__(
        self,
        *,
        text_config: dict | DeepseekV41TextConfig | None = None,
        vision_config: dict | DeepseekV41VisionConfig | None = None,
        **kwargs,
    ) -> None:
        if text_config is None:
            text_config = DeepseekV41TextConfig()
        elif isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        if vision_config is None:
            vision_config = DeepseekV41VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = DeepseekV41VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)
        quantization = kwargs.get("quantization_config", {})
        expert_dtype = quantization.get("expert_dtype")
        if expert_dtype is not None:
            self.text_config.expert_dtype = expert_dtype
        self.text_config.moe_quant_algo = quantization.get("moe_quant_algo")
        if self.text_config.moe_quant_algo is not None:
            if (
                self.text_config.moe_quant_algo != "NVFP4"
                or quantization.get("group_size") != 16
                or quantization.get("weight_block_size") != [32, 32]
                or quantization.get("scale_fmt") != "ue8m0"
            ):
                raise ValueError(
                    "V4.1 NVFP4 experts require group16 and original FP8 dense weights"
                )
            expected = {
                f"layers.{i}.ffn.experts"
                for i in range(self.text_config.num_hidden_layers)
            }
            entries = quantization.get("quantized_layers", {})
            if set(entries) != expected or any(
                entry != {"group_size": 16, "quant_algo": "NVFP4"}
                for entry in entries.values()
            ):
                raise ValueError(
                    "V4.1 NVFP4 must cover exactly the main routed experts"
                )
        self.text_config.engram_embed_quant = quantization.get(
            "engram_embed_quant", "mxfp8"
        )
        if self.text_config.engram_embed_quant == "mxfp4":
            if (
                quantization.get("engram_embed_block_size") != 32
                or quantization.get("engram_embed_scale_dtype") != "ue8m0"
                or quantization.get("engram_embed_layers")
                != self.text_config.engram_layer_ids
            ):
                raise ValueError(
                    "V4.1 MXFP4 Engram requires group32 E8M0 on all Engram layers"
                )
        elif self.text_config.engram_embed_quant != "mxfp8":
            raise ValueError("Unsupported V4.1 Engram checkpoint format")
        for name in ("dtype", "bos_token_id", "eos_token_id", "pad_token_id"):
            if name in kwargs:
                setattr(self.text_config, name, getattr(self, name))

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
