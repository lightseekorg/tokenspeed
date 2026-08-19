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
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.utils.server_args import ServerArgs

_INDEX_K_FP8_GROUP_SIZE = 128
_INDEX_K_SCALE_BYTES = torch._utils._element_size(torch.float32)


def dsa_index_k_row_bytes(index_head_dim: int) -> int:
    if index_head_dim <= 0 or index_head_dim % _INDEX_K_FP8_GROUP_SIZE != 0:
        raise ValueError(
            f"DSA index_head_dim must be a positive multiple of {_INDEX_K_FP8_GROUP_SIZE}, got {index_head_dim}"
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
        if base.dcp_size > 1:
            raise ValueError(
                "decode context parallelism currently supports dense MLA, not DSA"
            )
        if base.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            platform = current_platform()
            if not (platform.is_blackwell_plus or platform.is_cdna4_plus):
                raise ValueError(
                    "GLM DSA FP8 KV cache currently requires NVIDIA Blackwell "
                    "or AMD CDNA4 sparse attention support; use --kv-cache-dtype "
                    "auto or bfloat16 on this platform, got "
                    f"{server_args.kv_cache_dtype}."
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
