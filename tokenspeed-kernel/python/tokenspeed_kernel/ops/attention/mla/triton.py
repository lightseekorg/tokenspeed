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


import torch
from tokenspeed_kernel.ops.attention.mla._triton.decode import (
    _triton_mla_decode_with_kvcache_impl,
)
from tokenspeed_kernel.ops.attention.mla._triton.page_table import *  # noqa: F403
from tokenspeed_kernel.ops.attention.mla._triton.prefill import (
    _triton_mla_prefill_impl,
)
from tokenspeed_kernel.ops.attention.mla._triton.write_locations import *  # noqa: F403
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})
_PORTABLE_DTYPES = frozenset({torch.float16, torch.bfloat16}) | _FP8_DTYPES
_PORTABLE_CAPABILITY = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))


@register_kernel(
    "attention",
    "mla_prefill",
    name="triton_mla_prefill",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures(("q", "k", "v"), "dense", _PORTABLE_DTYPES),
    priority=Priority.PORTABLE,
    traits={
        "is_causal": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    seq_lens_kv: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _triton_mla_prefill_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
        seq_lens_kv=seq_lens_kv,
    )


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="triton_mla_decode_with_kvcache",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures(("q", "kv_cache"), "dense", _PORTABLE_DTYPES),
    priority=Priority.PORTABLE,
    traits={
        "q_len": frozenset({1}),
        "sliding_window": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mla_decode_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    window_left: int = -1,
    noncausal_block_size: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _triton_mla_decode_with_kvcache_impl(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        logit_cap=logit_cap,
        return_lse=return_lse,
        out=out,
        window_left=window_left,
        noncausal_block_size=noncausal_block_size,
    )
