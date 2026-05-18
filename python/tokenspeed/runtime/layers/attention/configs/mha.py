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
    BaseAttnConfig,
    resolve_dtype,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass
class MHAConfig(BaseAttnConfig):
    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ):
        return cls(
            device=server_args.device,
            context_len=model_config.context_len,
            backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            dtype=model_config.dtype,
            kv_cache_dtype=resolve_dtype(server_args.kv_cache_dtype),
            page_size=server_args.block_size,
            max_bs=server_args.max_num_seqs
            // (server_args.data_parallel_size or server_args.mapping.attn.dp_size),
            max_graph_bs=server_args.max_cudagraph_capture_size,
            kv_cache_quant_method=server_args.kv_cache_quant_method,
            speculative_num_steps=server_args.speculative_num_steps,
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            is_draft=is_draft,
        )

    def cache_cell_size(self) -> int:
        return (
            max(self.num_kv_heads // self.attn_tp_size, 1)
            * self.head_dim
            * 2
            * torch._utils._element_size(self.kv_cache_dtype)
        )

    def create_pool(
        self,
        num_layers: int,
        max_total_num_tokens: int,
        rank: int,
        enable_memory_saver: bool,
    ) -> BaseTokenToKVPool:
        from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool

        return MHATokenToKVPool(
            size=max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            head_num=max(self.num_kv_heads // self.attn_tp_size, 1),
            head_dim=self.head_dim,
            layer_num=num_layers,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=self.max_bs,
            max_context_len=self.context_len,
            page_size=self.page_size,
            rank=rank,
        )
