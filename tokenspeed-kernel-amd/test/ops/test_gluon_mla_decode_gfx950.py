from __future__ import annotations

import math

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


_IS_GFX950 = _is_gfx950()
if not _IS_GFX950:
    pytest.skip(
        "Gluon MLA decode kernel is gfx950 (CDNA4) only",
        allow_module_level=True,
    )

from tokenspeed_kernel import mla_decode_with_kvcache

_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_QK_NOPE_HEAD_DIM = 128
_QK_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM


def _torch_ref(q, kv_cache, page_table, cache_seqlens, page_size, scale):
    refs, ref_lses = [], []
    for b in range(q.shape[0]):
        rows = [
            kv_cache[page_table[b, pos // page_size], pos % page_size, 0]
            for pos in range(int(cache_seqlens[b].item()))
        ]
        kv = torch.stack(rows).float()
        scores = torch.einsum("hd,kd->hk", q[b, 0].float(), kv) * scale
        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.matmul(probs, kv[:, :_KV_LORA_RANK]).unsqueeze(0))
        ref_lses.append(torch.logsumexp(scores, dim=-1).unsqueeze(0))
    return torch.stack(refs, dim=0), torch.stack(ref_lses, dim=0)


@pytest.mark.parametrize(
    "num_heads,cache_seqlens",
    [
        (16, [200, 64, 129, 1]),
        (16, [65, 128, 256, 63]),
        (8, [130, 64]),
        (1, [77]),
    ],
)
def test_gluon_mla_decode_parity(num_heads: int, cache_seqlens: list[int]) -> None:
    device = "cuda"
    page_size = 64
    torch.manual_seed(0)
    seqlens = torch.tensor(cache_seqlens, device=device, dtype=torch.int32)
    batch_size = len(cache_seqlens)
    max_seqlen_k = int(seqlens.max().item())
    max_pages = (max_seqlen_k + page_size - 1) // page_size
    num_pages = batch_size * max_pages + 4

    q = torch.randn(
        batch_size, 1, num_heads, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_pages, page_size, 1, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    page_table = (
        torch.randperm(num_pages, device=device)[: batch_size * max_pages]
        .reshape(batch_size, max_pages)
        .to(torch.int32)
        .contiguous()
    )
    scale = 1.0 / math.sqrt(_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)

    kwargs = dict(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        softmax_scale=scale,
        return_lse=True,
    )
    out_g, lse_g = mla_decode_with_kvcache(solution="gluon", **kwargs)
    out_ref, lse_ref = _torch_ref(q, kv_cache, page_table, seqlens, page_size, scale)

    assert out_g.shape == (batch_size, 1, num_heads, _KV_LORA_RANK)
    assert lse_g.shape == (batch_size, 1, num_heads)
    torch.testing.assert_close(out_g.float(), out_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(lse_g.float(), lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "num_heads,cache_seqlens,max_seqlen_k",
    [
        # Capture-time worst-case >> runtime seqlens: NUM_KV_SPLITS is sized for
        # max_seqlen_k, so short requests leave most trailing splits empty and
        # the reduce must mask them (the cudagraph-safe compromise).
        (16, [130, 64, 200, 1], 8192),  # batch 4 -> min(64, 128) = 64 splits
        (1, [65], 8192),  # batch 1 -> min(256, 128) = 128 splits, one short seq
        (8, [300, 5], 4096),  # batch 2 -> min(128, 64) = 64 splits
    ],
)
def test_gluon_mla_decode_overprovisioned_splits(
    num_heads: int, cache_seqlens: list[int], max_seqlen_k: int
) -> None:
    """Passing a large (capture-time) max_seqlen_k must still match the torch
    reference for short runtime sequences -- empty trailing splits are masked."""
    device = "cuda"
    page_size = 64
    torch.manual_seed(0)
    seqlens = torch.tensor(cache_seqlens, device=device, dtype=torch.int32)
    batch_size = len(cache_seqlens)
    # Cache / page table are sized to the *actual* sequences; only the split
    # count is driven by the (larger) max_seqlen_k.
    actual_max = int(seqlens.max().item())
    max_pages = (actual_max + page_size - 1) // page_size
    num_pages = batch_size * max_pages + 4

    q = torch.randn(
        batch_size, 1, num_heads, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_pages, page_size, 1, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    page_table = (
        torch.randperm(num_pages, device=device)[: batch_size * max_pages]
        .reshape(batch_size, max_pages)
        .to(torch.int32)
        .contiguous()
    )
    scale = 1.0 / math.sqrt(_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)

    out_g, lse_g = mla_decode_with_kvcache(
        solution="gluon",
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        softmax_scale=scale,
        return_lse=True,
    )
    out_ref, lse_ref = _torch_ref(q, kv_cache, page_table, seqlens, page_size, scale)

    torch.testing.assert_close(out_g.float(), out_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(lse_g.float(), lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "num_heads,batch_size",
    [
        (64, 64),  # base_grid 64  -> NUM_KV_SPLITS 4 (split-K + reduce)
        (128, 64),  # base_grid 128 -> NUM_KV_SPLITS 2 (split-K + reduce)
        (64, 128),  # base_grid 128 -> NUM_KV_SPLITS 2 (split-K + reduce)
        (128, 128),  # base_grid 256 -> NUM_KV_SPLITS 1 (single-split fast path)
    ],
)
def test_gluon_mla_decode_bh64_parity(num_heads: int, batch_size: int) -> None:
    """bh64 regime (num_q_heads in {64, 128}, batch divisible by 64): the 3-D
    XCD-aware grid must match the torch reference across the split-K and the
    single-split fast paths, with per-batch sequence lengths that leave some
    trailing splits empty (exercising the stage-2 mask)."""
    device = "cuda"
    page_size = 64
    torch.manual_seed(0)
    seqlens = torch.randint(1, 400, (batch_size,), device=device, dtype=torch.int32)
    max_seqlen_k = int(seqlens.max().item())
    max_pages = (max_seqlen_k + page_size - 1) // page_size
    num_pages = batch_size * max_pages + 4

    q = torch.randn(
        batch_size, 1, num_heads, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_pages, page_size, 1, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    page_table = (
        torch.randperm(num_pages, device=device)[: batch_size * max_pages]
        .reshape(batch_size, max_pages)
        .to(torch.int32)
        .contiguous()
    )
    scale = 1.0 / math.sqrt(_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)

    out_g, lse_g = mla_decode_with_kvcache(
        solution="gluon",
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        softmax_scale=scale,
        return_lse=True,
    )
    out_ref, lse_ref = _torch_ref(q, kv_cache, page_table, seqlens, page_size, scale)

    assert out_g.shape == (batch_size, 1, num_heads, _KV_LORA_RANK)
    assert lse_g.shape == (batch_size, 1, num_heads)
    torch.testing.assert_close(out_g.float(), out_ref, rtol=8e-2, atol=8e-2)
    torch.testing.assert_close(lse_g.float(), lse_ref, rtol=8e-2, atol=8e-2)


def test_gluon_mla_decode_bh64_requires_batch_multiple_of_64() -> None:
    """bh64 is large-batch only: a {64, 128}-head decode whose batch is not a
    multiple of 64 must raise (mirroring the upstream assert), not silently drop
    the uncovered batch rows."""
    device = "cuda"
    page_size = 64
    batch_size = 32  # divisible by NUM_XCDS(8) but not by 64
    seqlens = torch.full((batch_size,), 128, device=device, dtype=torch.int32)
    max_pages = (128 + page_size - 1) // page_size
    num_pages = batch_size * max_pages + 4

    q = torch.randn(
        batch_size, 1, 64, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_pages, page_size, 1, _QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    page_table = (
        torch.randperm(num_pages, device=device)[: batch_size * max_pages]
        .reshape(batch_size, max_pages)
        .to(torch.int32)
        .contiguous()
    )
    scale = 1.0 / math.sqrt(_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)

    with pytest.raises(NotImplementedError, match="divisible by 64"):
        mla_decode_with_kvcache(
            solution="gluon",
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=seqlens,
            max_seqlen_k=128,
            qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
            kv_lora_rank=_KV_LORA_RANK,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            softmax_scale=scale,
        )


def test_gluon_mla_decode_is_selected() -> None:
    """The gluon kernel wins dispatch for the Kimi-TP4 decode shape and falls
    back to triton outside its supported regime."""
    from tokenspeed_kernel.ops.attention import _attention_format_signature
    from tokenspeed_kernel.selection import select_kernel

    def pick(num_heads: int, page_size: int, logit_cap: float = 0.0) -> str:
        q = torch.empty(
            2, 1, num_heads, _QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        kv = torch.empty(
            8, page_size, 1, _QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        sig = _attention_format_signature(q=q, kv_cache=kv)
        traits = {
            "page_size": page_size,
            "q_len": 1,
            "num_q_heads": num_heads,
            "qk_nope_head_dim": _QK_NOPE_HEAD_DIM,
            "kv_lora_rank": _KV_LORA_RANK,
            "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
            "support_logit_cap": logit_cap != 0.0,
            "return_lse": False,
        }
        return select_kernel(
            "attention", "mla_decode_with_kvcache", sig, traits=traits
        ).name

    assert pick(16, 64) == "gluon_mla_decode_bf16_gfx950"  # bh16bn64
    assert pick(8, 64) == "gluon_mla_decode_bf16_gfx950"  # bh16bn64
    assert pick(64, 64) == "gluon_mla_decode_bf16_gfx950"  # bh64
    assert pick(128, 64) == "gluon_mla_decode_bf16_gfx950"  # bh64
    # Head counts between the two regimes (17-63, >128) fall through to triton.
    assert pick(32, 64) == "triton_mla_decode_with_kvcache"
    assert pick(16, 1) == "triton_mla_decode_with_kvcache"
    assert pick(16, 64, logit_cap=1.0) == "triton_mla_decode_with_kvcache"
