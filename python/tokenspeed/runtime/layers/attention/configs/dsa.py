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
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.utils.server_args import ServerArgs

_INDEX_K_FP8_GROUP_SIZE = 128
_INDEX_K_SCALE_BYTES = torch._utils._element_size(torch.float32)


def _is_blackwell_device(device: str) -> bool:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError, ValueError):
        major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def dsa_index_k_row_bytes(index_head_dim: int) -> int:
    if index_head_dim % _INDEX_K_FP8_GROUP_SIZE != 0:
        raise ValueError(
            "DSA index_head_dim must be divisible by "
            f"{_INDEX_K_FP8_GROUP_SIZE}, got {index_head_dim}"
        )
    return (
        index_head_dim
        + index_head_dim // _INDEX_K_FP8_GROUP_SIZE * _INDEX_K_SCALE_BYTES
    )


@dataclass
class DSAConfig(MLAConfig):
    index_topk: int
    index_head_dim: int
    index_n_heads: int

    @classmethod
    def generate(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
        is_draft: bool = False,
    ):
        base = MLAConfig.generate(server_args, model_config, is_draft)
        if base.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            if not _is_blackwell_device(server_args.device):
                raise ValueError(
                    "GLM DSA FP8 KV cache currently requires the Blackwell TRTLLM "
                    "sparse attention path; use --kv-cache-dtype auto or bfloat16 "
                    f"on this platform, got {server_args.kv_cache_dtype}."
                )
        return cls(
            **base.__dict__,
            index_topk=model_config.index_topk,
            index_head_dim=model_config.index_head_dim,
            index_n_heads=model_config.index_n_heads,
        )

    def cache_cell_size(self) -> int:
        index_k_cell_size = dsa_index_k_row_bytes(
            self.index_head_dim,
        )
        return super().cache_cell_size() + index_k_cell_size

    def create_pool(
        self,
        num_layers: int,
        max_total_num_tokens: int,
        rank: int,
        enable_memory_saver: bool,
    ) -> BaseTokenToKVPool:
        from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool

        return DSATokenToKVPool(
            size=max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            model_dtype=self.dtype,
            quant_method=self.kv_cache_quant_method,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            layer_num=num_layers,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=self.max_bs,
            max_context_len=self.context_len,
            page_size=self.page_size,
            rank=rank,
            index_head_dim=self.index_head_dim,
        )
