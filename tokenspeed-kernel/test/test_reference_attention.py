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


"""The MHA ground-truth kernels: registration band and agreement with SDPA."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.reference.attention import (
    torch_mha_decode_with_kvcache,
    torch_mha_extend_with_kvcache,
    torch_mha_prefill,
)
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.ops.attention.mla import (
    mla_decode_with_kvcache,
    mla_extend_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.registry import KernelRegistry, Priority, load_builtin_kernels
from tokenspeed_kernel.selection import is_ground_truth

_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 2
_HEAD_DIM = 64
_PAGE_SIZE = 16


@pytest.mark.parametrize(
    "name",
    [
        "torch_mha_prefill",
        "torch_mha_extend_with_kvcache",
        "torch_mha_decode_with_kvcache",
        "torch_mla_prefill",
        "torch_mla_extend_with_kvcache",
        "torch_mla_decode_with_kvcache",
        "torch_gdn_chunk_prefill",
    ],
)
def test_attention_references_are_ground_truth_only(name: str) -> None:
    load_builtin_kernels()
    spec = KernelRegistry.get().get_by_name(name)
    assert spec is not None
    assert spec.solution == "torch"
    assert spec.priority == Priority.REFERENCE
    assert is_ground_truth(spec)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """FP32 SDPA over ``[L, H, D]`` tensors with a ``[Lq, Lk]`` boolean mask."""
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(
        q.float().transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=mask,
    )
    return out.transpose(0, 1)


def _causal_window_mask(
    q_pos: torch.Tensor, kv_len: int, window_left: int
) -> torch.Tensor:
    k_pos = torch.arange(kv_len, device=q_pos.device)
    mask = q_pos[:, None] >= k_pos[None, :]
    if window_left >= 0:
        mask &= q_pos[:, None] - k_pos[None, :] <= window_left
    return mask


@pytest.mark.parametrize("window_left", [-1, 5], ids=["full", "sliding"])
def test_prefill_matches_sdpa_per_sequence(device: str, window_left: int) -> None:
    torch.manual_seed(0)
    seqlens = [7, 19, 1]
    cu_seqlens_cpu = [0]
    for seqlen in seqlens:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    total = cu_seqlens_cpu[-1]
    q = torch.randn(total, _NUM_Q_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(
        total, _NUM_KV_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)

    out, lse = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32),
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max(seqlens),
        window_left=window_left,
        return_lse=True,
        solution="reference",
    )
    assert out.dtype == q.dtype
    assert lse.shape == (total, _NUM_Q_HEADS)

    for start, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
        q_pos = torch.arange(end - start, device=device)
        mask = _causal_window_mask(q_pos, end - start, window_left)
        expected = _sdpa(q[start:end], k[start:end], v[start:end], mask)
        torch.testing.assert_close(
            out[start:end].float(), expected, rtol=2e-2, atol=2e-2
        )
        scores = torch.einsum(
            "qhd,khd->qhk",
            q[start:end].float(),
            k[start:end].float().repeat_interleave(_NUM_Q_HEADS // _NUM_KV_HEADS, 1),
        ) / math.sqrt(_HEAD_DIM)
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        torch.testing.assert_close(
            lse[start:end], torch.logsumexp(scores, dim=-1), rtol=1e-4, atol=1e-4
        )


def test_prefill_sinks_and_logit_cap(device: str) -> None:
    torch.manual_seed(1)
    seqlen = 13
    q = torch.randn(seqlen, _NUM_Q_HEADS, _HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(
        seqlen, _NUM_KV_HEADS, _HEAD_DIM, device=device, dtype=torch.float32
    )
    v = torch.randn_like(k)
    sinks = torch.randn(_NUM_Q_HEADS, device=device)
    logit_cap = 3.0
    out, lse = torch_mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor([0, seqlen], device=device, dtype=torch.int32),
        cu_seqlens_cpu=[0, seqlen],
        max_seqlen=seqlen,
        window_left=-1,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=True,
        softmax_scale=None,
        skip_softmax_threshold=0.0,
    )

    group = _NUM_Q_HEADS // _NUM_KV_HEADS
    scores = torch.einsum(
        "qhd,khd->qhk", q, k.repeat_interleave(group, dim=1)
    ) / math.sqrt(_HEAD_DIM)
    scores = logit_cap * torch.tanh(scores / logit_cap)
    mask = _causal_window_mask(torch.arange(seqlen, device=device), seqlen, -1)
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    # The sink joins the softmax denominator as one more logit with no value.
    logits = torch.cat((scores, sinks[None, :, None].expand(seqlen, -1, 1)), dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :-1]
    expected = torch.einsum("qhk,khd->qhd", probs, v.repeat_interleave(group, dim=1))
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        lse, torch.logsumexp(logits, dim=-1), rtol=1e-4, atol=1e-4
    )


def _paged_cache(
    device: str, cache_lens: list[int]
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]
]:
    """Return a shuffled paged cache plus each request's flat K/V."""
    pages_per_seq = [(n + _PAGE_SIZE - 1) // _PAGE_SIZE for n in cache_lens]
    total_pages = sum(pages_per_seq)
    k_cache = torch.randn(
        total_pages, _PAGE_SIZE, _NUM_KV_HEADS, _HEAD_DIM, device=device
    ).to(torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    order = torch.randperm(total_pages, device=device).to(torch.int32)
    page_table = torch.zeros(
        len(cache_lens), max(pages_per_seq), device=device, dtype=torch.int32
    )
    flat_k, flat_v = [], []
    next_page = 0
    for batch_idx, (num_pages, cache_len) in enumerate(zip(pages_per_seq, cache_lens)):
        pages = order[next_page : next_page + num_pages]
        page_table[batch_idx, :num_pages] = pages
        next_page += num_pages
        flat_k.append(
            k_cache[pages.long()].reshape(-1, _NUM_KV_HEADS, _HEAD_DIM)[:cache_len]
        )
        flat_v.append(
            v_cache[pages.long()].reshape(-1, _NUM_KV_HEADS, _HEAD_DIM)[:cache_len]
        )
    return k_cache, v_cache, page_table, flat_k, flat_v


@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("window_left", [-1, 9], ids=["full", "sliding"])
def test_extend_matches_sdpa_on_flat_cache(
    device: str, is_causal: bool, window_left: int
) -> None:
    torch.manual_seed(2)
    query_lens = [3, 1, 5]
    cache_lens = [40, 17, 5]
    k_cache, v_cache, page_table, flat_k, flat_v = _paged_cache(device, cache_lens)
    cu_seqlens_q = [0]
    for query_len in query_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + query_len)
    q = torch.randn(
        cu_seqlens_q[-1], _NUM_Q_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )

    out = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(
            [0, *torch.tensor(cache_lens).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        ),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor(cache_lens, device=device, dtype=torch.int32),
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(cache_lens),
        is_causal=is_causal,
        window_left=window_left,
        solution="reference",
    )

    for batch_idx, (start, end) in enumerate(zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])):
        cache_len = cache_lens[batch_idx]
        q_pos = cache_len - (end - start) + torch.arange(end - start, device=device)
        k_pos = torch.arange(cache_len, device=device)
        mask = torch.ones((end - start, cache_len), dtype=torch.bool, device=device)
        if is_causal:
            mask &= q_pos[:, None] >= k_pos[None, :]
        if window_left >= 0:
            mask &= q_pos[:, None] - k_pos[None, :] <= window_left
        expected = _sdpa(q[start:end], flat_k[batch_idx], flat_v[batch_idx], mask)
        torch.testing.assert_close(
            out[start:end].float(), expected, rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("q_len", [1, 4], ids=["q1", "q4"])
def test_decode_is_the_causal_tail_of_extend(device: str, q_len: int) -> None:
    torch.manual_seed(3)
    cache_lens = [33, 4, 20]
    k_cache, v_cache, page_table, _, _ = _paged_cache(device, cache_lens)
    q = torch.randn(
        len(cache_lens) * q_len,
        _NUM_Q_HEADS,
        _HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_seqlens = torch.tensor(cache_lens, device=device, dtype=torch.int32)

    decode = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max(cache_lens),
        max_seqlen_q=q_len,
        solution="reference",
    )
    cu_seqlens_q = torch.arange(
        0, (len(cache_lens) + 1) * q_len, q_len, device=device, dtype=torch.int32
    )
    options = dict(
        window_left=-1,
        logit_cap=0.0,
        sinks=None,
        return_lse=False,
        softmax_scale=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        enable_pdl=False,
    )
    extend = torch_mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=q_len,
        max_seqlen_k=max(cache_lens),
        is_causal=True,
        **options,
    )
    torch.testing.assert_close(decode, extend, rtol=0, atol=0)
    assert torch.equal(
        decode,
        torch_mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max(cache_lens),
            max_seqlen_q=q_len,
            **options,
        ),
    )


_KV_LORA_RANK = 32
_ROPE_DIM = 8
_MLA_SCALE = 1.0 / math.sqrt(_KV_LORA_RANK + _ROPE_DIM)


@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_mla_prefill_matches_sdpa_per_sequence(device: str, is_causal: bool) -> None:
    torch.manual_seed(4)
    q_lens, kv_lens = [3, 6, 1], [9, 6, 4]
    cu_q = [0, *torch.tensor(q_lens).cumsum(0).tolist()]
    cu_kv = [0, *torch.tensor(kv_lens).cumsum(0).tolist()]
    q = torch.randn(
        cu_q[-1], _NUM_Q_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        cu_kv[-1], _NUM_Q_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    v = torch.randn(
        cu_kv[-1], _NUM_Q_HEADS, _HEAD_DIM // 2, device=device, dtype=torch.bfloat16
    )

    out = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor(cu_q, device=device, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(cu_kv, device=device, dtype=torch.int32),
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        softmax_scale=_MLA_SCALE,
        is_causal=is_causal,
        solution="reference",
    )
    assert out.shape == (cu_q[-1], _NUM_Q_HEADS, _HEAD_DIM // 2)

    for (q_start, q_end), (kv_start, kv_end) in zip(
        zip(cu_q[:-1], cu_q[1:]), zip(cu_kv[:-1], cu_kv[1:])
    ):
        q_len, kv_len = q_end - q_start, kv_end - kv_start
        mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=device)
        if is_causal:
            q_pos = kv_len - q_len + torch.arange(q_len, device=device)
            mask = q_pos[:, None] >= torch.arange(kv_len, device=device)[None, :]
        expected = F.scaled_dot_product_attention(
            q[q_start:q_end].float().transpose(0, 1),
            k[kv_start:kv_end].float().transpose(0, 1),
            v[kv_start:kv_end].float().transpose(0, 1),
            attn_mask=mask,
            scale=_MLA_SCALE,
        ).transpose(0, 1)
        torch.testing.assert_close(
            out[q_start:q_end].float(), expected, rtol=2e-2, atol=2e-2
        )


def _mla_cache(
    device: str, cache_lens: list[int]
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    pages_per_seq = [(n + _PAGE_SIZE - 1) // _PAGE_SIZE for n in cache_lens]
    total_pages = sum(pages_per_seq)
    kv_cache = torch.randn(
        total_pages, _PAGE_SIZE, 1, _KV_LORA_RANK + _ROPE_DIM, device=device
    ).to(torch.bfloat16)
    order = torch.randperm(total_pages, device=device).to(torch.int32)
    page_table = torch.zeros(
        len(cache_lens), max(pages_per_seq), device=device, dtype=torch.int32
    )
    flat = []
    next_page = 0
    for batch_idx, (num_pages, cache_len) in enumerate(zip(pages_per_seq, cache_lens)):
        pages = order[next_page : next_page + num_pages]
        page_table[batch_idx, :num_pages] = pages
        next_page += num_pages
        flat.append(
            kv_cache[pages.long()].reshape(-1, _KV_LORA_RANK + _ROPE_DIM)[:cache_len]
        )
    return kv_cache, page_table, flat


def _mla_expected(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """``[H, D]`` query against flat ``[n, D]`` latent+rope rows."""
    scores = torch.einsum("hd,kd->hk", q.float(), kv.float()) * _MLA_SCALE
    return torch.softmax(scores, dim=-1) @ kv.float()[:, :_KV_LORA_RANK]


@pytest.mark.parametrize("q_len", [1, 3], ids=["q1", "q3"])
def test_mla_decode_is_the_causal_tail(device: str, q_len: int) -> None:
    torch.manual_seed(5)
    cache_lens = [37, 5, 20]
    kv_cache, page_table, flat = _mla_cache(device, cache_lens)
    q = torch.randn(
        len(cache_lens),
        q_len,
        _NUM_Q_HEADS,
        _KV_LORA_RANK + _ROPE_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    out, lse = mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor(cache_lens, device=device, dtype=torch.int32),
        max_seqlen_k=max(cache_lens),
        qk_nope_head_dim=_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=_MLA_SCALE,
        return_lse=True,
        solution="reference",
    )
    assert out.shape == (len(cache_lens), q_len, _NUM_Q_HEADS, _KV_LORA_RANK)
    assert lse.shape == (len(cache_lens), q_len, _NUM_Q_HEADS)
    for batch_idx, cache_len in enumerate(cache_lens):
        for query_idx in range(q_len):
            visible = flat[batch_idx][: cache_len - q_len + query_idx + 1]
            expected = _mla_expected(q[batch_idx, query_idx], visible)
            torch.testing.assert_close(
                out[batch_idx, query_idx].float(), expected, rtol=2e-2, atol=2e-2
            )


def test_mla_decode_proposal_block_window(device: str) -> None:
    """Both block layouts see the whole block plus ``window_left`` history."""
    torch.manual_seed(6)
    block, context_len, window_left = 4, 23, 9
    cache_len = context_len + block
    kv_cache, page_table, flat = _mla_cache(device, [cache_len])
    q = torch.randn(
        block,
        _NUM_Q_HEADS,
        _KV_LORA_RANK + _ROPE_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    common = dict(
        kv_cache=kv_cache,
        max_seqlen_k=cache_len,
        qk_nope_head_dim=_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=_MLA_SCALE,
        window_left=window_left,
        noncausal_block_size=block,
        solution="reference",
    )
    lens = torch.full((1,), cache_len, device=device, dtype=torch.int32)
    flattened = mla_decode_with_kvcache(
        q=q.unsqueeze(1),
        page_table=page_table.repeat(block, 1),
        cache_seqlens=lens.repeat(block),
        **common,
    )
    on_query_axis = mla_decode_with_kvcache(
        q=q.unsqueeze(0), page_table=page_table, cache_seqlens=lens, **common
    )
    torch.testing.assert_close(on_query_axis.reshape_as(flattened), flattened)
    for position in range(block):
        visible = flat[0][max(0, context_len - window_left + position) :]
        expected = _mla_expected(q[position], visible)
        torch.testing.assert_close(
            flattened[position, 0].float(), expected, rtol=2e-2, atol=2e-2
        )


def test_mla_extend_is_the_causal_suffix(device: str) -> None:
    torch.manual_seed(7)
    query_lens, cache_lens = [3, 2, 1], [3, 21, 40]
    kv_cache, page_table, flat = _mla_cache(device, cache_lens)
    cu_q = [0, *torch.tensor(query_lens).cumsum(0).tolist()]
    q = torch.randn(
        cu_q[-1],
        _NUM_Q_HEADS,
        _KV_LORA_RANK + _ROPE_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    out = mla_extend_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor(cache_lens, device=device, dtype=torch.int32),
        cu_seqlens_q=torch.tensor(cu_q, device=device, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(
            [0, *torch.tensor(cache_lens).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        ),
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(cache_lens),
        qk_nope_head_dim=_HEAD_DIM,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_ROPE_DIM,
        softmax_scale=_MLA_SCALE,
        is_causal=True,
        solution="reference",
    )
    assert out.shape == (cu_q[-1], _NUM_Q_HEADS, _KV_LORA_RANK)
    for batch_idx, (start, end) in enumerate(zip(cu_q[:-1], cu_q[1:])):
        cache_len, q_len = cache_lens[batch_idx], end - start
        for query_idx in range(q_len):
            visible = flat[batch_idx][: cache_len - q_len + query_idx + 1]
            expected = _mla_expected(q[start + query_idx], visible)
            torch.testing.assert_close(
                out[start + query_idx].float(), expected, rtol=2e-2, atol=2e-2
            )


def test_gdn_chunk_prefill_reference_is_the_token_recurrence(device: str) -> None:
    torch.manual_seed(8)
    seq_lens, num_q_heads, num_v_heads, head_dim = [5, 3], 2, 4, 16
    total = sum(seq_lens)
    q = torch.randn(1, total, num_q_heads, head_dim, device=device)
    k = torch.randn(1, total, num_q_heads, head_dim, device=device)
    v = torch.randn(1, total, num_v_heads, head_dim, device=device)
    g = F.logsigmoid(torch.randn(1, total, num_v_heads, device=device))
    beta = torch.rand(1, total, num_v_heads, device=device)
    initial_state = torch.randn(
        len(seq_lens), num_v_heads, head_dim, head_dim, device=device
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=0.5,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution="reference",
    )

    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    group = num_v_heads // num_q_heads
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        # Scalar recurrence per value head on the [K, V] state.
        state = initial_state[seq_idx].transpose(-2, -1).clone()
        for token in range(int(start), int(end)):
            for head in range(num_v_heads):
                q_t = l2norm(q[0, token, head // group])
                k_t = l2norm(k[0, token, head // group])
                s = torch.exp(g[0, token, head]) * state[head]
                s = s + torch.outer(
                    k_t, beta[0, token, head] * (v[0, token, head] - k_t @ s)
                )
                state[head] = s
                torch.testing.assert_close(
                    result.out[0, token, head], 0.5 * (q_t @ s), rtol=1e-4, atol=1e-4
                )
        torch.testing.assert_close(
            result.final_state[seq_idx], state.transpose(-2, -1), rtol=1e-4, atol=1e-4
        )
