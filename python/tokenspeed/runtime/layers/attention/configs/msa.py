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

"""MiniMax sparse-attention cache and index configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
    model_wide_kwargs,
    resolve_dtype,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(kw_only=True)
class MSAConfig(SoftmaxAttnConfig):
    """Runtime and cache contract for MiniMax sparse attention.

    One component: the sparse label selects dense-vs-sparse COMPUTE per layer,
    while every layer retains full history in the one shared cache — so the
    dense sub-backend is this component's business (``full_attn_backend_name``),
    not a second component.
    """

    full_attn_backend_name: str | None = None

    # Compute-layer labels select dense versus sparse execution. Cache-layer
    # labels (inherited layer_types) remain empty because every MiniMax layer
    # retains full history.
    compute_layer_types: tuple[str, ...] = ()
    sparse_layer_ids: frozenset[int] = frozenset()
    sliding_window_tokens: None = None

    index_head_dim: int = 0
    index_n_heads: int = 0
    index_topk_blocks: int = 0
    index_init_blocks: int = 0
    index_local_blocks: int = 0

    @classmethod
    def generate(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
        is_draft: bool = False,
    ) -> AttnConfig:
        text_config = model_config.hf_text_config
        sparse_layer_type = model_config.hf_config.runtime_attention_layer_type

        compute_layer_types = tuple(text_config.layer_types)
        sparse_layer_ids = frozenset(
            layer_id
            for layer_id, layer_type in enumerate(compute_layer_types)
            if layer_type == sparse_layer_type
        )

        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = bool(
            is_draft and server_args.speculative_algorithm == "DFLASH"
        )
        if draft_block_decode:
            kv_cache_dtype = "bfloat16"
        if kv_cache_dtype == "mxfp8":
            raise ValueError(
                "MiniMax sparse attention does not support kv_cache_dtype='mxfp8'."
            )
        if server_args.kv_cache_quant_method != "none":
            raise ValueError(
                f"MiniMax sparse attention does not support kv_cache_quant_method={server_args.kv_cache_quant_method!r}."
            )

        spec = cls(
            backend_name="msa",
            full_attn_backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            compute_layer_types=compute_layer_types,
            sparse_layer_ids=sparse_layer_ids,
            layer_types=(),
            sliding_window_tokens=None,
            index_head_dim=int(text_config.index_head_dim),
            index_n_heads=int(text_config.index_n_heads),
            index_topk_blocks=int(text_config.index_topk_blocks),
            index_init_blocks=int(getattr(text_config, "index_init_blocks", 0)),
            index_local_blocks=int(text_config.index_local_blocks),
        )
        # MSA verify uses the raw configured width (no DSpark width convention).
        return AttnConfig(
            components=(spec,),
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=resolve_dtype(kv_cache_dtype),
                draft_block_decode=draft_block_decode,
                speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            ),
        )

    def cache_cell_size(self, config: AttnConfig) -> int:
        kv_cache_bytes = (
            max(self.num_kv_heads // self.attn_tp_size, 1)
            * self.head_dim
            * 2
            * torch._utils._element_size(config.kv_cache_dtype)
        )
        index_cache_bytes = self.index_head_dim * torch._utils._element_size(
            config.dtype
        )
        num_layers = len(self.compute_layer_types)
        if num_layers:
            # The common profiler multiplies this per-layer value by num_layers.
            index_cache_bytes = (
                index_cache_bytes * len(self.sparse_layer_ids) + num_layers - 1
            ) // num_layers
        return kv_cache_bytes + index_cache_bytes
