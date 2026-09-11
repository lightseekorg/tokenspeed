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

"""Checkpoint configuration for DeepSeek V4.1's text backbone."""

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


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    def __init__(self, **kwargs):
        text = kwargs.pop("text_config", {})
        self.text_config = text
        super().__init__(**kwargs)
        for name in ("dtype", "bos_token_id", "eos_token_id", "pad_token_id"):
            if name in kwargs:
                setattr(self.text_config, name, getattr(self, name))

    def __setattr__(self, name, value):
        if name == "text_config" and isinstance(value, dict):
            value = DeepseekV41TextConfig(**value)
        super().__setattr__(name, value)

    def __getattr__(self, name):
        text = self.__dict__.get("text_config")
        if name.startswith("_") or name == "text_config" or text is None:
            raise AttributeError(name)
        return getattr(text, name)
