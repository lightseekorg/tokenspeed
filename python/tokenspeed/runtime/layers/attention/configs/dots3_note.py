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

"""Attention geometry for the alternating DSA and SWA layers."""

from dataclasses import dataclass

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    model_wide_kwargs,
    resolve_dtype,
)
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(kw_only=True)
class Dots3NoteAttentionConfig(DSAConfig):
    swa: MLAConfig

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> AttnConfig:
        if is_draft:
            if server_args.drafter_attention_backend not in (None, "mla"):
                raise ValueError("dots3-note MTP requires the mla backend")
            return AttnConfig(
                components=(
                    MLAConfig(
                        **MLAConfig._spec_kwargs(server_args, model_config, is_draft),
                        sliding_window_tokens=model_config.hf_text_config.sliding_window_size,
                    ),
                ),
                kernel_page_size=32,
                **model_wide_kwargs(
                    server_args,
                    model_config,
                    is_draft,
                    kv_cache_dtype=resolve_dtype(server_args.kv_cache_dtype),
                ),
            )
        if server_args.attention_backend not in (None, "dsa"):
            raise ValueError("dots3-note requires the dsa/mla cache-group router")
        hf = model_config.hf_text_config
        dsa_kwargs = DSAConfig._spec_kwargs(server_args, model_config, is_draft)
        swa = MLAConfig(
            backend_name="mla",
            num_attention_heads=hf.swa_num_attention_heads,
            num_kv_heads=hf.swa_num_key_value_heads,
            head_dim=hf.swa_qk_nope_head_dim + hf.swa_qk_rope_head_dim,
            attn_tp_size=dsa_kwargs["attn_tp_size"],
            kv_lora_rank=hf.swa_kv_lora_rank,
            qk_nope_head_dim=hf.swa_qk_nope_head_dim,
            qk_rope_head_dim=hf.swa_qk_rope_head_dim,
            v_head_dim=hf.swa_v_head_dim,
            scaling=(hf.swa_qk_nope_head_dim + hf.swa_qk_rope_head_dim) ** -0.5,
            kv_cache_dim=hf.swa_kv_lora_rank + hf.swa_qk_rope_head_dim,
            sliding_window_tokens=hf.sliding_window_size,
        )
        # Hopper uses the tested portable FP8 path, without GLM's platform gate.
        return AttnConfig(
            components=(cls(**dsa_kwargs, swa=swa),),
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=resolve_dtype(server_args.kv_cache_dtype),
            ),
        )
