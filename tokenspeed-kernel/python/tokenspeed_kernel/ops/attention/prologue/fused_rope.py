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

"""Prologues whose RoPE launch also stores the key and value rows."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.prologue import (
    _BOOLS,
    _CUDA_ROPE_HEAD_DIMS,
    GQAPrologueOutput,
    HeadKVCache,
    RopeStyle,
    Rotary,
)
from tokenspeed_kernel.ops.embedding import FusedSetKVBufferArg, apply_rope
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "gqa_prologue",
    name="fused_rope_gqa_attention_prologue",
    solution="fused_rope",
    # Only the CUDA embedding.rope kernel stores K/V in its launch.
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(("q",), "dense", {torch.float16, torch.bfloat16}),
    priority=Priority.PERFORMANT + 1,
    traits={
        "head_dim": _CUDA_ROPE_HEAD_DIMS,
        # Up to 512 token-heads the one-launch Triton kernel is faster.
        "token_heads_min": frozenset({513}),
        "full_write": frozenset({True}),
        "has_norm": frozenset({False}),
        "kv_format": frozenset({"native", "fp8"}),
        "kv_convert": frozenset({False}),
        "mrope": frozenset({False}),
        "partial_rotary": frozenset({False}),
        "return_kv": _BOOLS,
        "rope_style": frozenset({"neox", "gptj"}),
    },
)
def fused_rope_gqa_attention_prologue(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    norm: None,
    rotary: Rotary,
    cache: HeadKVCache,
    return_kv: bool,
    enable_pdl: bool,
) -> GQAPrologueOutput:
    # embedding.rope reads the same platform PDL setting.
    del norm, enable_pdl
    num_tokens = q.shape[0]
    q = q.flatten(1)
    q_rope = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_rope = torch.empty(k.shape, dtype=k.dtype, device=k.device) if return_kv else None
    apply_rope(
        rotary.positions,
        q,
        k,
        cache.k_cache.shape[-1],
        rotary.cos_sin_cache,
        is_neox=rotary.style is RopeStyle.NEOX,
        fused_set_kv_buffer_arg=FusedSetKVBufferArg(
            value=v.reshape(num_tokens, *cache.v_cache.shape[1:]),
            k_buffer=cache.k_cache.view(cache.k_cache.shape[0], -1),
            v_buffer=cache.v_cache.view(cache.v_cache.shape[0], -1),
            cache_loc=cache.slots,
        ),
        q_rope_out=q_rope,
        k_rope_out=k_rope,
    )
    return GQAPrologueOutput(q=q_rope, k=k_rope, v=v if return_kv else None)
