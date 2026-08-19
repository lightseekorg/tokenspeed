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

from dataclasses import dataclass, field

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import (
    BaseAttnConfig,
    resolve_dtype,
    resolve_speculative_num_tokens,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def resolve_mla_kv_cache_dtype(
    server_args: ServerArgs, model_config: ModelConfig, is_draft: bool
) -> torch.dtype:
    """Resolve MLA cache precision without quantizing K3 DSpark context blindly.

    The public K3 DSpark checkpoint has no FP8 KV scales and its reference vLLM
    launch uses the default BF16 cache. TokenSpeed's K3 target currently requires
    FP8 LCM storage, so the unified arena exposes a separate BF16 compute view
    for the draft continuation fields. Other MLA drafts continue to honor the
    global cache setting.
    """
    hf_config = getattr(model_config, "hf_config", None)
    if (
        is_draft
        and server_args.speculative_algorithm == "DSPARK"
        and getattr(hf_config, "model_type", None) == "k3_dspark"
    ):
        return torch.bfloat16
    return resolve_dtype(server_args.kv_cache_dtype)


@dataclass
class MLAConfig(BaseAttnConfig):
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    scaling: float
    kv_cache_dim: int
    layer_types: tuple[str, ...] = field(default=(), kw_only=True)
    max_scheduled_tokens: int = field(default=0, kw_only=True)
    pd_disaggregation_enabled: bool = field(default=False, kw_only=True)

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
        hf_config = getattr(model_config, "hf_config", None)
        layer_types = tuple(
            getattr(hf_config, "cache_layer_types", None)
            or getattr(hf_config, "layer_types", None)
            or ()
        )
        draft_block_decode = bool(
            is_draft and server_args.speculative_algorithm in ("DFLASH", "DSPARK")
        )
        mapping = server_args.mapping.attn
        dcp_size = 1 if is_draft else mapping.dcp_size
        # MLA persists one compressed latent KV stream per request. Generic HF
        # configs often omit num_key_value_heads, which ModelConfig then
        # defaults to the query-head count; that fallback is not MLA geometry.
        effective_kv_streams = 1
        if dcp_size > 1:
            backend_name = server_args.attention_backend
            if backend_name not in ("tokenspeed_mla", "hybrid_linear_attn"):
                raise ValueError(
                    "decode context parallelism currently requires the "
                    "tokenspeed_mla full-attention backend, got "
                    f"{backend_name!r}"
                )
            if server_args.kv_cache_dtype != "fp8_e4m3":
                raise ValueError(
                    "decode context parallelism currently requires "
                    "--kv-cache-dtype fp8_e4m3"
                )
            if server_args.kv_cache_quant_method != "none":
                raise ValueError(
                    "decode context parallelism does not support "
                    f"--kv-cache-quant-method {server_args.kv_cache_quant_method!r}"
                )
            if server_args.disaggregation_mode != "null":
                raise ValueError(
                    "decode context parallelism does not yet support PD "
                    "disaggregation; use --disaggregation-mode null"
                )
            if server_args.kvstore_storage_backend is not None:
                raise ValueError(
                    "decode context parallelism does not yet namespace shared "
                    "KVStore storage by DCP rank; disable the storage backend"
                )
            local_heads = model_config.num_attention_heads // mapping.tp_size
            if local_heads * dcp_size > 128:
                raise ValueError(
                    "tokenspeed_mla DCP supports at most 128 gathered query "
                    f"heads, got {local_heads * dcp_size}"
                )
            max_dcp_size = mapping.tp_size // effective_kv_streams
            if dcp_size > max_dcp_size:
                raise ValueError(
                    "decode context parallel subgroups must stay within one "
                    "KV stream: require DCP size <= attention TP size / "
                    f"num KV streams ({mapping.tp_size} / {effective_kv_streams} = "
                    f"{max_dcp_size}), got {dcp_size}"
                )
        return cls(
            device=server_args.device,
            context_len=model_config.context_len + server_args.spec_context_pad,
            backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=effective_kv_streams,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            dtype=model_config.dtype,
            kv_cache_dtype=resolve_mla_kv_cache_dtype(
                server_args, model_config, is_draft
            ),
            prefix_granularity=server_args.prefix_granularity,
            max_graph_bs=server_args.max_cudagraph_capture_size,
            max_bs=server_args.max_num_seqs
            // (server_args.data_parallel_size or server_args.mapping.attn.dp_size),
            kv_cache_quant_method=server_args.kv_cache_quant_method,
            is_draft=is_draft,
            draft_block_decode=draft_block_decode,
            kv_lora_rank=model_config.kv_lora_rank,
            qk_nope_head_dim=model_config.qk_nope_head_dim,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            v_head_dim=model_config.v_head_dim,
            scaling=model_config.scaling,
            kv_cache_dim=model_config.kv_lora_rank + model_config.qk_rope_head_dim,
            layer_types=layer_types,
            max_scheduled_tokens=getattr(server_args, "chunked_prefill_size", 8192),
            pd_disaggregation_enabled=getattr(
                server_args, "disaggregation_mode", "null"
            )
            != "null",
            dcp_size=dcp_size,
            dcp_rank=mapping.dcp_rank if dcp_size > 1 else 0,
            dcp_group=mapping.dcp_group if dcp_size > 1 else (0,),
            **kwargs,
        )

    def cache_cell_size(self) -> int:
        if self.kv_cache_quant_method == "per_token_head":
            cell_size = (
                self.kv_lora_rank * torch._utils._element_size(self.kv_cache_dtype)
                + self.qk_rope_head_dim * torch._utils._element_size(self.dtype)
                + 1 * torch._utils._element_size(torch.float32)
            )
        else:
            cell_size = (
                self.kv_lora_rank + self.qk_rope_head_dim
            ) * torch._utils._element_size(self.kv_cache_dtype)
        return cell_size
