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
    resolve_speculative_num_tokens,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass
class MHAConfig(BaseAttnConfig):
    # Resolved by the Qwen GDN cache recipe after checking the engine option,
    # verify width, device, and registered kernel support.
    replay_ssm: bool = False
    # Per-layer attention-type labels + window, forwarded to the KV pool for
    # cache_group_specs publication (empty -> single full-history group).
    layer_types: tuple[str, ...] = ()
    sliding_window_tokens: int | tuple[int | None, ...] | None = None
    max_scheduled_tokens: int = 0
    # True iff server_args.disaggregation_mode != "null"; cache recipes use
    # it to stamp transfer policies onto the cache group specs.
    pd_disaggregation_enabled: bool = False
    layer_kv_head_counts: tuple[int, ...] | None = None

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ):
        kwargs = {}
        if server_args.speculative_algorithm is not None:
            kwargs.update(
                speculative_num_steps=server_args.speculative_num_steps,
                speculative_num_draft_tokens=resolve_speculative_num_tokens(
                    server_args, is_draft
                ),
            )
        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = bool(
            is_draft and server_args.speculative_algorithm in ("DFLASH", "DSPARK")
        )
        if draft_block_decode and server_args.drafter_attention_backend != "trtllm":
            kv_cache_dtype = "bfloat16"

        hf_config = getattr(model_config, "hf_config", None)
        # cache_layer_types wins: it can carry labels outside transformers' ALLOWED_LAYER_TYPES
        layer_types = tuple(
            getattr(hf_config, "cache_layer_types", None)
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
            context_len=model_config.context_len + server_args.spec_context_pad,
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
            prefix_granularity=server_args.prefix_granularity,
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
