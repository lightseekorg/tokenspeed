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
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.utils.server_args import ServerArgs


def resolve_dtype(kv_cache_dtype_str: str) -> torch.dtype:
    if kv_cache_dtype_str == "auto":
        return torch.bfloat16
    elif kv_cache_dtype_str == "bfloat16":
        return torch.bfloat16
    elif kv_cache_dtype_str in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype_str!r}")


@dataclass(kw_only=True)
class BaseAttnConfig:
    device: str
    backend_name: str
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    attn_tp_size: int
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    page_size: int
    context_len: int
    max_bs: int
    max_graph_bs: int
    kv_cache_quant_method: str
    speculative_num_steps: int = 0
    speculative_num_draft_tokens: int = 0
    is_draft: bool = False

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ):
        raise NotImplementedError("Not Implemented!")

    def cache_cell_size(self) -> int:
        raise NotImplementedError("Not Implemented!")

    def create_pool(
        self,
        num_layers: int,
        max_total_num_tokens: int,
        rank: int,
        enable_memory_saver: bool,
    ) -> BaseTokenToKVPool:
        raise NotImplementedError("Not Implemented!")
