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

"""Attention geometry for Hy4-flash hybrid compressed-DSA and SWA-MLA layers."""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed.runtime.configs.hy4_flash_config import HYV4FlashTextConfig
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
class Hy4FlashAttentionConfig(DSAConfig):
    """Target attention geometry pairing 32-head DSA layers with 64-head SWA-MLA."""

    swa: MLAConfig
    dsa_layer_ids: tuple[int, ...]
    swa_layer_ids: tuple[int, ...]

    @classmethod
    def generate(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
        is_draft: bool = False,
    ) -> AttnConfig:
        if is_draft:
            raise ValueError(
                "Hy4-flash DFly2 shares the target KV cache and does not allocate "
                "a standalone draft attention config"
            )
        if server_args.attention_backend not in (None, "dsa"):
            raise ValueError(
                "Hy4-flash requires the hybrid DSA/SWA cache-group router "
                f"(attention_backend=None or 'dsa'), got {server_args.attention_backend!r}"
            )
        hf = model_config.hf_text_config
        if not isinstance(hf, HYV4FlashTextConfig):
            raise TypeError(
                "Hy4FlashAttentionConfig requires HYV4FlashTextConfig, "
                f"got {type(hf).__name__}"
            )
        dsa_kwargs = DSAConfig._spec_kwargs(server_args, model_config, is_draft)
        dsa_kwargs["index_k_format"] = "fp8_scaled"
        dsa_kwargs["index_kpool"] = hf.index_kpool
        dsa_kwargs["cache_layer_types"] = tuple(hf.paged_cache_layer_types)

        swa_qk_head_dim = hf.swa_qk_nope_head_dim + hf.swa_qk_rope_head_dim
        swa = MLAConfig(
            backend_name="mla",
            num_attention_heads=hf.swa_num_attention_heads,
            num_kv_heads=hf.swa_num_attention_heads,
            head_dim=swa_qk_head_dim,
            attn_tp_size=dsa_kwargs["attn_tp_size"],
            kv_lora_rank=hf.kv_lora_rank,
            qk_nope_head_dim=hf.swa_qk_nope_head_dim,
            qk_rope_head_dim=hf.swa_qk_rope_head_dim,
            v_head_dim=hf.swa_v_head_dim,
            scaling=swa_qk_head_dim**-0.5,
            kv_cache_dim=hf.kv_lora_rank + hf.swa_qk_rope_head_dim,
            sliding_window_tokens=hf.sliding_window,
            cache_layer_types=tuple(hf.paged_cache_layer_types),
        )
        return AttnConfig(
            components=(
                cls(
                    **dsa_kwargs,
                    swa=swa,
                    dsa_layer_ids=hf.dsa_layer_ids,
                    swa_layer_ids=hf.swa_layer_ids,
                ),
            ),
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=resolve_dtype(server_args.kv_cache_dtype),
            ),
        )


__all__ = ["Hy4FlashAttentionConfig"]
