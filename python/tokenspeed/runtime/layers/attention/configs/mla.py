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
    model_wide_kwargs,
    resolve_dtype,
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
    if (
        is_draft
        and server_args.speculative_algorithm == "DSPARK"
        and getattr(model_config.hf_config, "model_type", None) == "k3_dspark"
    ):
        return torch.bfloat16
    return resolve_dtype(server_args.kv_cache_dtype)


@dataclass(kw_only=True)
class MLAConfig(SoftmaxAttnConfig):
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    scaling: float
    kv_cache_dim: int
    # DeepSeek V4 stamps its window here post-construction (declared so the
    # write is a real field, not a __dict__ stowaway).
    sliding_window_tokens: int | None = None

    @classmethod
    def _spec_kwargs(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool
    ) -> dict:
        """MLA component fields, shared with the DSA subclass."""
        hf_config = model_config.hf_config
        layer_types = tuple(
            getattr(hf_config, "cache_layer_types", None)
            or getattr(hf_config, "layer_types", None)
            or ()
        )
        return dict(
            backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            kv_lora_rank=model_config.kv_lora_rank,
            qk_nope_head_dim=model_config.qk_nope_head_dim,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            v_head_dim=model_config.v_head_dim,
            scaling=model_config.scaling,
            kv_cache_dim=model_config.kv_lora_rank + model_config.qk_rope_head_dim,
            layer_types=layer_types,
        )

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> AttnConfig:
        draft_block_decode = bool(
            is_draft and server_args.speculative_algorithm in ("DFLASH", "DSPARK")
        )
        spec = cls(**cls._spec_kwargs(server_args, model_config, is_draft))
        return AttnConfig(
            components=(spec,),
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=resolve_mla_kv_cache_dtype(
                    server_args, model_config, is_draft
                ),
                draft_block_decode=draft_block_decode,
            ),
        )

    def cache_cell_size(self, config: AttnConfig) -> int:
        if config.kv_cache_quant_method == "per_token_head":
            cell_size = (
                self.kv_lora_rank * torch._utils._element_size(config.kv_cache_dtype)
                + self.qk_rope_head_dim * torch._utils._element_size(config.dtype)
                + 1 * torch._utils._element_size(torch.float32)
            )
        else:
            cell_size = (
                self.kv_lora_rank + self.qk_rope_head_dim
            ) * torch._utils._element_size(config.kv_cache_dtype)
        return cell_size
