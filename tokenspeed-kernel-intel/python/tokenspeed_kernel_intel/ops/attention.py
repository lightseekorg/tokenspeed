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

vllm-xpu-kernels exposes a single entry point, ``flash_attn_varlen_func``,
which serves BOTH:

* non-paged variable-length prefill  -> pass ``cu_seqlens_k``
* paged decode                       -> pass ``block_table`` + ``seqused_k``

Its signature (verified against v0.1.10)::

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
        host_kv_lens=None,        # v0.1.10: host per-seq KV len -> seqused_k
        num_splits_kv=None,       # v0.1.10: split-KV plan for paged decode
        ...
    ) -> out | (out, softmax_lse)

The prefill path below uses only the kwargs shared across versions, so it is
unchanged by the v0.1.7 -> v0.1.10 upgrade.
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
# vllm-xpu-kernels >= 0.1.10 exposes a mature paged decode path through the same
# ``flash_attn_varlen_func`` entry point: pass ``block_table`` + ``seqused_k``
# for the paged KV cache and a ``cu_seqlens_q`` describing the (usually length-1)
# decode queries. The KV cache is laid out as
# ``[num_blocks, block_size, num_kv_heads, head_dim]``, which matches the layout
# TokenSpeed hands to the decode kernel (the Triton path views it identically),
# so no relayout is required.


if _XPU_FA_AVAILABLE:

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="xpu_mha_decode_with_kvcache",
        solution="xpu",
        capability=CapabilityRequirement(vendors=frozenset({"intel"})),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=_XPU_ATTENTION_PRIORITY,
        traits={
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False, True}),
            # LSE return not wired up for decode yet; those paths fall back to
            # the portable Triton kernel.
            "return_lse": frozenset({False}),
        },
        tags={"xpu"},
    )
    def xpu_mha_decode_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_k: int,
        max_seqlen_q: int = 1,
        window_left: int = -1,
        logit_cap: float = 0.0,
        sinks: torch.Tensor | None = None,
        return_lse: bool = False,
        softmax_scale: float | None = None,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Paged-KV decode MHA on Intel XPU.

        ``q`` is packed ``[batch * max_seqlen_q, num_heads, head_dim]``.
        ``k_cache`` / ``v_cache`` are paged
        ``[num_blocks, block_size, num_kv_heads, head_dim]`` and indexed through
        ``page_table`` (``block_table``) with per-request KV length
        ``cache_seqlens`` (``seqused_k``).
        """
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch = cache_seqlens.shape[0]
        # One contiguous run of ``max_seqlen_q`` query tokens per request:
        # cu_seqlens_q = [0, m, 2m, ..., batch*m], int32 on device.
        cu_seqlens_q = torch.arange(
            0,
            (batch + 1) * max_seqlen_q,
            max_seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )
        seqused_k = cache_seqlens.to(torch.int32)
        block_table = page_table.to(torch.int32)

        out = _xpu_fa.flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=seqused_k,  # paged: per-seq KV length
            block_table=block_table,  # paged KV cache
            softmax_scale=softmax_scale,
            # Pure decode (single query token) is non-causal; speculative /
            # multi-token decode (max_seqlen_q > 1) needs the causal mask.
            causal=max_seqlen_q > 1,
            window_size=_window(window_left),
            softcap=logit_cap,
            s_aux=sinks,
        )
        return out.view_as(q)
