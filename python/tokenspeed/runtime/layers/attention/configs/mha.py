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
from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec
from tokenspeed.runtime.layers.attention.configs.base import (
    BaseAttnConfig,
    resolve_dtype,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass
class MHAConfig(BaseAttnConfig):
    # Per-layer attention-type labels + window, forwarded to the KV pool for
    # paged_cache_group_specs publication (empty -> single full-history group).
    layer_types: tuple[str, ...] = ()
    sliding_window_tokens: int | tuple[int | None, ...] | None = None
    max_scheduled_tokens: int = 0
    # True iff server_args.disaggregation_mode != "null"; the pool's slab
    # guards consume it.
    pd_disaggregation_enabled: bool = False
    # Extra model-declared paged-cache groups (e.g. Inkling paged sconv); forwarded to publication
    extra_paged_groups: tuple[PagedCacheGroupSpec, ...] = ()
    # Slot span in tokens (largest group's block)
    slot_tokens: int | None = None
    # Per-group page sizes (hetero zero-padding slots)
    group_page_sizes: dict[str, int] | None = None
    layer_kv_head_counts: tuple[int, ...] | None = None

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ):
        kwargs = {}
        if server_args.speculative_algorithm is not None:
            kwargs.update(
                speculative_num_steps=server_args.speculative_num_steps,
                speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            )
        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = bool(
            is_draft and server_args.speculative_algorithm in ("DFLASH", "DSPARK")
        )
        if draft_block_decode and server_args.drafter_attention_backend != "trtllm":
            kv_cache_dtype = "bfloat16"

        hf_config = getattr(model_config, "hf_config", None)
        # paged_cache_layer_types wins: it can carry labels outside transformers' ALLOWED_LAYER_TYPES
        layer_types = tuple(
            getattr(hf_config, "paged_cache_layer_types", None)
            or getattr(hf_config, "layer_types", None)
            or ()
        )
        if (
            is_draft
            and layer_types
            and len(layer_types) != model_config.num_attention_layers
        ):
            # Target-stack labels don't fit the draft depth; drop so the pool falls back to full attn
            layer_types = ()
        sliding_window_tokens = getattr(hf_config, "sliding_window", None)
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
            kv_cache_dtype=resolve_dtype(kv_cache_dtype),
            kv_cache_mxfp8=kv_cache_dtype == "mxfp8",
            page_size=server_args.block_size,
            max_bs=server_args.max_num_seqs
            // (server_args.data_parallel_size or server_args.mapping.attn.dp_size),
            max_graph_bs=server_args.max_cudagraph_capture_size,
            kv_cache_quant_method=server_args.kv_cache_quant_method,
            is_draft=is_draft,
            draft_block_decode=draft_block_decode,
            layer_types=layer_types,
            sliding_window_tokens=sliding_window_tokens,
            max_scheduled_tokens=getattr(server_args, "chunked_prefill_size", 8192),
            pd_disaggregation_enabled=getattr(
                server_args, "disaggregation_mode", "null"
            )
            != "null",
            **kwargs,
        )

    def cache_cell_size(self) -> int:
        cell = (
            max(self.num_kv_heads // self.attn_tp_size, 1)
            * self.head_dim
            * 2
            * torch._utils._element_size(self.kv_cache_dtype)
        )
        if self.kv_cache_mxfp8:
            # One UE8M0 byte per 32 fp8 data bytes.
            cell += cell // 32
        return cell

    def create_pool(
        self,
        num_layers: int,
        max_total_num_tokens: int,
        rank: int,
        enable_memory_saver: bool,
    ) -> BaseTokenToKVPool:
        raise RuntimeError(
            "Every KV pool now runs on the shared LCM arena with a "
            "PagedCacheRuntimeContract; the classic MHA pool path was removed. "
            "This model family has no LCM recipe yet: add one in "
            "lcm_setup.prepare_lcm_setup (see the 'plain_mha' and 'msa' "
            "recipes for the pattern) and gate it in registry.lcm_family."
        )
