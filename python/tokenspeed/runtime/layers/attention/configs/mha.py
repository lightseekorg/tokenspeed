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

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
    is_block_drafter,
    model_wide_kwargs,
    resolve_cache_layer_types,
    resolve_dtype,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(kw_only=True)
class MHAConfig(SoftmaxAttnConfig):
    # Mixed full+sliding models are ONE component carrying the inherited
    # per-layer cache_layer_types/window vectors; the per-layer kv-head counts
    # live on the cache-pool spec (``CachePoolSpec.layer_kv_head_counts``).

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> AttnConfig:
        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = is_block_drafter(
            server_args.speculative_algorithm, is_draft
        )
        if draft_block_decode and server_args.drafter_attention_backend != "trtllm":
            kv_cache_dtype = "bfloat16"

        hf_config = model_config.hf_config
        cache_layer_types = resolve_cache_layer_types(
            hf_config,
            num_layers=model_config.num_attention_layers,
            is_draft=is_draft,
            draft_block_decode=draft_block_decode,
        )
        # The retention window belongs to the storage labels: a block drafter
        # has none of its own (its window is a compute mask on its layers).
        sliding_window_tokens = (
            None if draft_block_decode else getattr(hf_config, "sliding_window", None)
        )
        spec = cls(
            backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            cache_layer_types=cache_layer_types,
            sliding_window_tokens=sliding_window_tokens,
        )
        return AttnConfig(
            components=(spec,),
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=resolve_dtype(kv_cache_dtype),
                kv_cache_mxfp8=kv_cache_dtype == "mxfp8",
                draft_block_decode=draft_block_decode,
            ),
        )

    def cache_cell_size(self, config: AttnConfig) -> int:
        cell = (
            max(self.num_kv_heads // self.attn_tp_size, 1)
            * self.head_dim
            * 2
            * torch._utils._element_size(config.kv_cache_dtype)
        )
        if config.kv_cache_mxfp8:
            # One UE8M0 byte per 32 fp8 data bytes.
            cell += cell // 32
        return cell
