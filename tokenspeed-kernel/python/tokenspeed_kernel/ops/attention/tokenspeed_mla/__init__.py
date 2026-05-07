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

"""TokenSpeed MLA kernels exposed through tokenspeed-kernel."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel

try:
    from tokenspeed_mla import (
        get_num_sm,
        mla_kv_pack_quantize_fp8,
        tokenspeed_mla_decode,
        tokenspeed_mla_prefill,
        warmup_compile_prefill,
    )
except ImportError:
    get_num_sm = error_fn
    mla_kv_pack_quantize_fp8 = error_fn
    tokenspeed_mla_decode = error_fn
    tokenspeed_mla_prefill = error_fn
    warmup_compile_prefill = error_fn


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="tokenspeed_mla_decode_with_kvcache",
    features={"mla", "paged"},
    solution="tokenspeed_mla",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    dtypes={torch.float16, torch.bfloat16, torch.float8_e4m3fn},
    priority=Priority.SPECIALIZED + 3,
    tags={"latency"},
)
def tokenspeed_mla_decode_with_kvcache(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    workspace_buffer: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    output_scale: float = 1.0,
    out: torch.Tensor | None = None,
    is_var_seq: bool = True,
    causal_mask: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    if workspace_buffer is None:
        raise ValueError("workspace_buffer is required for TokenSpeed MLA decode")
    return tokenspeed_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        softmax_scale=(
            softmax_scale
            if softmax_scale is not None
            else 1.0 / math.sqrt(query.shape[-1])
        ),
        output_scale=output_scale,
        out=out,
        is_var_seq=is_var_seq,
        causal_mask=causal_mask,
        enable_pdl=enable_pdl,
    )

__all__ = [
    "get_num_sm",
    "mla_kv_pack_quantize_fp8",
    "tokenspeed_mla_decode",
    "tokenspeed_mla_decode_with_kvcache",
    "tokenspeed_mla_prefill",
    "warmup_compile_prefill",
]
