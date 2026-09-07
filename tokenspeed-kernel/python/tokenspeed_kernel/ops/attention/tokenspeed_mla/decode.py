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

"""The CuteDSL MLA decode under the ``mla_decode_with_kvcache`` dispatcher.

Registered for a block drafter's proposal and nothing else:
``noncausal_block_size`` exceeds one only when a DFLASH or DSPARK draft
proposes a whole non-causal block (``configs/mla.py`` sets
``draft_block_decode`` from exactly that), so ordinary decode and target
verify cannot reach this registration however their other traits line up. A
target already reaches this same kernel through its own backend, which calls
``tokenspeed_mla_decode`` directly.

Within that scope both masks are served. The windowed one is what the
portable Triton kernel was the only implementation of, and is why a draft
whose layers declare ``sliding_attention`` was paying an order of magnitude
for its attention; the full-attention layers of the same draft were paying it
for no reason at all.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode

__all__ = ["tokenspeed_mla_decode_with_kvcache"]

_MLA_DECODE_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float8_e4m3fn})
# The kernel classes hard-code the MLA latent and RoPE widths, and the paged
# reader is built for these page spans.
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_PAGE_SIZES = frozenset({32, 64})
# Query rows per request in one launch, which for a block drafter is its whole
# proposal block. One would be ordinary decode, which this registration does
# not serve.
_Q_LENS = frozenset(range(2, 9))
_BLOCK_SIZES = _Q_LENS
_NUM_Q_HEADS = frozenset(range(1, 129))

_workspaces: dict[torch.device, torch.Tensor] = {}


def _workspace(device: torch.device, num_q_heads: int, q_len: int) -> torch.Tensor:
    """Split-KV accumulator scratch, one growing block per device.

    Sized by the kernel's own closed-form bound, which does not depend on the
    batch, so steady-state decode never reallocates and a captured graph
    records a block that outlives it.
    """
    required = get_num_sm(device) * num_q_heads * q_len * (_KV_LORA_RANK + 1) * 4
    workspace = _workspaces.get(device)
    if workspace is None or workspace.numel() < required:
        workspace = torch.empty(required, dtype=torch.int8, device=device)
        _workspaces[device] = workspace
    return workspace


@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="tokenspeed_mla_decode_with_kvcache",
    solution="tokenspeed_mla",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(("q", "kv_cache"), "dense", _MLA_DECODE_DTYPES),
    priority=Priority.SPECIALIZED,
    traits={
        "sliding_window": frozenset({False, True}),
        # A block drafter's proposal, never ordinary decode or target verify.
        "noncausal_block_size": _BLOCK_SIZES,
        # The block rides the query axis here. The flattened form, one row per
        # block position on the batch axis, stays with the portable kernel.
        "block_on_query_axis": frozenset({True}),
        "page_size": _PAGE_SIZES,
        "q_len": _Q_LENS,
        "num_q_heads": _NUM_Q_HEADS,
        "kv_lora_rank": frozenset({_KV_LORA_RANK}),
        "qk_rope_head_dim": frozenset({_QK_ROPE_HEAD_DIM}),
        "support_logit_cap": frozenset({False}),
        # The kernel reports log-sum-exp in log2 units, which is not this
        # dispatcher's contract; a caller that wants LSE keeps the Triton path.
        "return_lse": frozenset({False}),
    },
    tags={"latency"},
)
def tokenspeed_mla_decode_with_kvcache(
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
) -> torch.Tensor:
    """MLA decode with a drafter's proposal block on the query axis.

    ``q`` carries the whole block as ``[batch, block, heads, dim]`` with one
    page table row and one cache length per request, rather than the flattened
    one-row-per-block-position form the Triton kernel reads. Both spell the
    same mask: every row of a block sees the whole block, plus either
    ``window_left`` tokens of history or all of it.

    Raises:
        ValueError: The block is not the query axis, which means the caller is
            on the flattened contract and should have selected the portable
            kernel.
    """
    if noncausal_block_size != q.shape[1]:
        raise ValueError(
            f"the proposal block must be the query axis: q_len={q.shape[1]}, "
            f"noncausal_block_size={noncausal_block_size}"
        )
    if kv_cache.ndim == 4:
        kv_cache = kv_cache.squeeze(2)

    return tokenspeed_mla_decode(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=_workspace(q.device, q.shape[2], q.shape[1]),
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=page_table,
        seq_lens=cache_seqlens,
        max_seq_len=max_seqlen_k,
        softmax_scale=softmax_scale,
        out=out,
        # The block is non-causal by construction: every row sees the whole
        # block, so the window -- when there is one -- is the only bound.
        causal_mask=False,
        window_left=window_left,
        enable_pdl=pdl_enabled(),
    )
