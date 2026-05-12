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

import importlib
import math

import pytest
import tokenspeed_kernel.ops.attention.flash_attn as flash_attn_module
import torch
from tokenspeed_kernel.ops.attention.flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel

platform = current_platform()
torch.manual_seed(42)

pytestmark = pytest.mark.skipif(
    not (platform.is_blackwell and platform.arch_version == ArchVersion(10, 0)),
    reason="FA4 smoke tests require SM 10.0 Blackwell GPU (B100/B200/GB200).",
)


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2), (torch.bfloat16, 256, 8, 2)],
)
def test_mha(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 2
    seqlen = 128

    q = torch.randn(
        batch_size, seqlen, num_q_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype
    )

    out, lse = flash_attn_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape
    assert lse is None


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2), (torch.bfloat16, 256, 8, 2)],
)
def test_mha_ragged(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    seqlens = torch.tensor([17, 9, 12], device=device, dtype=torch.int32)
    cu_seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0))
    total_tokens = int(seqlens.sum().item())
    max_seqlen = int(seqlens.max().item())

    q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    out, lse = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape
    assert lse is None


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_ragged_with_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    decode_tokens = 1
    max_cache_seqlen = 256
    cache_seqlens = torch.tensor([63, 48, 17, 80], device=device, dtype=torch.int32)

    q = torch.randn(
        batch_size, decode_tokens, num_q_heads, head_dim, device=device, dtype=dtype
    )

    k_cache = torch.zeros(
        batch_size,
        max_cache_seqlen,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        batch_size,
        max_cache_seqlen,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        k_cache[batch_idx, :total_kv_len] = torch.randn(
            total_kv_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v_cache[batch_idx, :total_kv_len] = torch.randn(
            total_kv_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )

    out, lse = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        seqused_k=cache_seqlens,
        max_seqlen_q=decode_tokens,
        max_seqlen_k=int(cache_seqlens.max().item()),
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape
    assert lse is None


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_ragged_with_paged_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    batch_size = 4
    decode_tokens = 1
    page_size = 128
    max_cache_seqlen = 256
    cache_seqlens = torch.tensor([63, 129, 17, 191], device=device, dtype=torch.int32)
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    q = torch.randn(
        batch_size, decode_tokens, num_q_heads, head_dim, device=device, dtype=dtype
    )

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            k_cache[physical_block, :tokens_in_block] = torch.randn(
                tokens_in_block,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            v_cache[physical_block, :tokens_in_block] = torch.randn(
                tokens_in_block,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )

    out, lse = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        seqused_k=cache_seqlens,
        page_table=page_table,
        max_seqlen_q=decode_tokens,
        max_seqlen_k=int(cache_seqlens.max().item()),
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape
    assert lse is None


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 128, 8, 2)],
)
def test_mha_ragged_extend_with_paged_kvcache(
    device: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    page_size = 128
    max_cache_seqlen = 256
    query_seqlens = torch.tensor([3, 1, 2, 4], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([66, 130, 19, 195], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.cumsum(query_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0))
    total_q = int(query_seqlens.sum().item())
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=dtype)

    page_table = torch.zeros(
        cache_seqlens.shape[0],
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            k_cache[physical_block, :tokens_in_block] = torch.randn(
                tokens_in_block,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            v_cache[physical_block, :tokens_in_block] = torch.randn(
                tokens_in_block,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )

    out, lse = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        page_table=page_table,
        max_seqlen_q=int(query_seqlens.max().item()),
        max_seqlen_k=int(cache_seqlens.max().item()),
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )

    assert out.shape == q.shape
    assert lse is None


def _reload_fa4_registry_entries() -> None:
    KernelRegistry.reset()
    importlib.reload(flash_attn_module)


def test_fa4_prefill_selection_accepts_head_dim_256() -> None:
    _reload_fa4_registry_entries()
    registry = KernelRegistry.get()
    spec = registry.get_by_name("fa4_mha_prefill")
    assert spec is not None
    assert spec.solution == "fa4"
    assert 256 in spec.traits["head_dim"]

    selected = select_kernel(
        "attention",
        "mha_prefill",
        torch.bfloat16,
        traits={
            "head_dim": 256,
            "sliding_window": False,
            "support_sinks": False,
            "return_lse": False,
            "support_logit_cap": False,
        },
    )
    assert selected.name == "fa4_mha_prefill"


def test_fa4_decode_selection_keeps_head_dim_256_disabled() -> None:
    _reload_fa4_registry_entries()
    spec = KernelRegistry.get().get_by_name("fa4_mha_decode_with_kvcache")
    assert spec is not None
    assert 256 not in spec.traits["head_dim"]


def test_fa4_prefill_with_kvcache_selection_requires_prewritten_kv() -> None:
    _reload_fa4_registry_entries()
    spec = KernelRegistry.get().get_by_name("fa4_mha_prefill_with_kvcache_cached")
    assert spec is not None
    assert spec.solution == "fa4"
    assert spec.traits["prewritten_kv"] == frozenset({True})

    traits = {
        "head_dim": 128,
        "page_size": 128,
        "prewritten_kv": True,
        "sliding_window": False,
        "support_sinks": False,
        "return_lse": False,
        "support_logit_cap": False,
    }
    selected = select_kernel(
        "attention",
        "mha_prefill_with_kvcache",
        torch.bfloat16,
        traits=traits,
    )
    assert selected.name == "fa4_mha_prefill_with_kvcache_cached"

    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "mha_prefill_with_kvcache",
            torch.bfloat16,
            traits={**traits, "prewritten_kv": False},
        )
