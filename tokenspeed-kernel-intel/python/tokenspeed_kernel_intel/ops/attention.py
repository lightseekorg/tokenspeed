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

"""Intel XPU attention kernels (vllm-xpu-kernels backend).

Registers XPU-specialized MHA kernels into the shared TokenSpeed registry.
Compute is delegated to ``vllm_xpu_kernels.flash_attn_interface`` (SYCL/DPC++).

vllm-xpu-kernels v0.1.7 exposes a single entry point,
``flash_attn_varlen_func``, which serves BOTH:

* non-paged variable-length prefill  -> pass ``cu_seqlens_k``
* paged decode                       -> pass ``block_table`` + ``seqused_k``

Its signature (v0.1.7)::

    flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q, cu_seqlens_q, max_seqlen_k,
        cu_seqlens_k=None,        # non-paged prefill
        seqused_k=None,           # paged: tokens used per seq
        softmax_scale=None,
        causal=False,
        window_size=None,         # [left, right]; None -> (-1, -1)
        softcap=0.0,              # == logit cap
        block_table=None,         # paged KV cache
        return_softmax_lse=False,
        s_aux=None,               # attention sinks
        ...
    ) -> out | (out, softmax_lse)
"""

from __future__ import annotations

import math

import torch

from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from vllm_xpu_kernels import flash_attn_interface as _xpu_fa

    _XPU_FA_AVAILABLE = getattr(_xpu_fa, "FA2_AVAILABLE", True)
except Exception:  # pragma: no cover - environment dependent
    _xpu_fa = None
    _XPU_FA_AVAILABLE = False


# Priority: above the portable Triton band (Priority.PORTABLE) so the selector
# prefers the XPU-specialized kernel whenever its traits match.
_XPU_ATTENTION_PRIORITY = Priority.PERFORMANT


def _window(window_left: int) -> list[int] | None:
    # TokenSpeed encodes "no sliding window" as window_left < 0. vllm expects a
    # [left, right] pair (or None). Causal attention uses right = 0.
    return [window_left, 0] if window_left >= 0 else None


if _XPU_FA_AVAILABLE:

    @register_kernel(
        "attention",
        "mha_prefill",
        name="xpu_mha_prefill",
        solution="xpu",
        capability=CapabilityRequirement(vendors=frozenset({"intel"})),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=_XPU_ATTENTION_PRIORITY,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
        tags={"xpu"},
    )
    def xpu_mha_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: list[int],
        max_seqlen: int,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Variable-length causal MHA prefill on Intel XPU.

        Non-paged self-attention: q/k/v are packed
        ``[total_tokens, num_heads, head_dim]`` sharing ``cu_seqlens`` for both
        Q and K.
        """
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        out = _xpu_fa.flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=max_seqlen,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=max_seqlen,
            cu_seqlens_k=cu_seqlens,  # non-paged prefill
            softmax_scale=softmax_scale,
            causal=True,
            window_size=_window(window_left),
            softcap=logit_cap,
            s_aux=sinks,
            return_softmax_lse=return_lse,
        )
        if return_lse:
            attn_out, lse = out
            return attn_out, lse
        return out


# ---------------------------------------------------------------------------
# Paged decode (mha_decode_with_kvcache)
# ---------------------------------------------------------------------------
# NOT registered yet: paged decode requires mapping TokenSpeed's KV-cache /
# page-table layout onto vllm's ``block_table`` + ``seqused_k`` convention, and
# constructing ``cu_seqlens_q`` for the decode queries. These layout details
# (block size, [num_blocks, block, heads, dim] ordering, per-request q lengths)
# MUST be verified on hardware against the numerics reference before enabling.
# Until then, decode falls back to the portable Triton kernel automatically
# (no XPU decode kernel is registered, so the selector skips this vendor for the
# mha_decode_with_kvcache mode).
#
# Template to finish, then wrap with @register_kernel(
#     "attention", "mha_decode_with_kvcache", name="xpu_mha_decode_with_kvcache",
#     solution="xpu", capability=CapabilityRequirement(vendors={"intel"}),
#     signatures=format_signatures(("q","k_cache","v_cache"), "dense",
#                                  {torch.float16, torch.bfloat16}),
#     priority=_XPU_ATTENTION_PRIORITY, traits={...}, tags={"xpu"}):
#
# def xpu_mha_decode_with_kvcache(q, k_cache, v_cache, page_table, cache_seqlens,
#                                 max_seqlen_k, max_seqlen_q=1, window_left=-1,
#                                 logit_cap=0.0, sinks=None, return_lse=False,
#                                 softmax_scale=None, q_scale=None, k_scale=None,
#                                 v_scale=None):
#     if softmax_scale is None:
#         softmax_scale = 1.0 / math.sqrt(q.shape[-1])
#     # Build cu_seqlens_q for the decode queries (e.g. arange(0, B+1, int32)
#     # when max_seqlen_q == 1). Verify q packing first.
#     out = _xpu_fa.flash_attn_varlen_func(
#         q, k_cache, v_cache,
#         max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
#         max_seqlen_k=max_seqlen_k,
#         seqused_k=cache_seqlens,     # paged: per-seq KV length
#         block_table=page_table,      # paged KV cache
#         softmax_scale=softmax_scale, causal=True,
#         window_size=_window(window_left), softcap=logit_cap, s_aux=sinks,
#     )
#     return out
