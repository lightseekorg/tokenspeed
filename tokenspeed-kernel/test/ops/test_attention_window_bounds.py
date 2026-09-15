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

"""Sliding-window paged attention compared with a logical-KV Torch reference.

Cover cached suffixes, page and tile boundaries, ragged query lengths, attention
sinks, and disabled windows while checking that inputs stay unchanged.
"""

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache


@pytest.mark.parametrize(
    "query_lengths,prefix_lengths,window_left,is_causal,has_sink,return_lse",
    [
        pytest.param((127,), (961,), 511, True, False, True, id="swa511-q127"),
        pytest.param((128,), (65,), 511, True, True, True, id="swa511-q128-sink"),
        pytest.param(
            (129, 1, 63),
            (513, 127, 65),
            511,
            True,
            False,
            False,
            id="swa511-ragged-q129",
        ),
        pytest.param((65, 129), (31, 191), 63, True, True, True, id="swa63-ragged"),
        pytest.param((129,), (63,), 1, True, False, True, id="swa1-q129"),
        pytest.param((129,), (193,), 511, False, False, True, id="swa511-noncausal"),
        pytest.param((127,), (577,), 63, False, True, False, id="swa63-noncausal"),
        pytest.param((129, 63), (65, 191), -1, True, True, True, id="full-causal"),
        pytest.param((128,), (63,), -1, False, False, True, id="full-noncausal"),
        pytest.param((1024,), (0,), 511, True, False, True, id="swa511-fresh-1k"),
    ],
)
@torch.inference_mode()
def test_mha_extend_window_bounds_public(
    device,
    require,
    monkeypatch,
    query_lengths,
    prefix_lengths,
    window_left,
    is_causal,
    has_sink,
    return_lse,
):
    require("attention", "mha_extend_with_kvcache", "triton", torch.bfloat16, "q")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    generator = torch.Generator(device=device).manual_seed(1149)
    page_size, q_heads, kv_heads, head_dim = 64, 6, 1, 128
    lengths = [prefix + query for prefix, query in zip(prefix_lengths, query_lengths)]
    pages = [(length + page_size - 1) // page_size for length in lengths]
    cu_q, cu_kv = [0], [0]
    for query, length in zip(query_lengths, lengths):
        cu_q.append(cu_q[-1] + query)
        cu_kv.append(cu_kv[-1] + length)
    q = torch.randn(
        (cu_q[-1], q_heads, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.full(
        (sum(pages) + 1, page_size, kv_heads, head_dim),
        17.0,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.full_like(k_cache, -19.0)
    table = torch.zeros((len(lengths), max(pages)), device=device, dtype=torch.int32)
    permutation = torch.arange(sum(pages), 0, -1, device=device, dtype=torch.int32)
    logical_kv, page_offset = [], 0
    for index, (length, page_count) in enumerate(zip(lengths, pages)):
        physical = permutation[page_offset : page_offset + page_count]
        table[index, :page_count] = physical
        positions = torch.arange(length, device=device, dtype=torch.int64)
        slots = (
            physical[positions // page_size].long() * page_size + positions % page_size
        )
        k = torch.randn(
            (length, kv_heads, head_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            (length, kv_heads, head_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        k_cache.view(-1, kv_heads, head_dim)[slots] = k
        v_cache.view(-1, kv_heads, head_dim)[slots] = v
        logical_kv.append((k, v))  # Reference never gathers through the page table.
        page_offset += page_count
    cu_q_tensor = torch.tensor(cu_q, device=device, dtype=torch.int32)
    cu_kv_tensor = torch.tensor(cu_kv, device=device, dtype=torch.int32)
    seq = torch.tensor(lengths, device=device, dtype=torch.int32)
    sinks = (
        torch.randn(
            (q_heads,), generator=generator, device=device, dtype=torch.bfloat16
        )
        if has_sink
        else None
    )
    inputs = [q, k_cache, v_cache, table, cu_q_tensor, cu_kv_tensor, seq]
    if sinks is not None:
        inputs.append(sinks)
    before = [tensor.clone() for tensor in inputs]
    scale = 1.0 / math.sqrt(head_dim)
    result = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_q_tensor,
        cu_seqlens_kv=cu_kv_tensor,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=table,
        cache_seqlens=seq,
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(lengths),
        is_causal=is_causal,
        window_left=window_left,
        logit_cap=0.0,
        sinks=sinks,
        return_lse=return_lse,
        softmax_scale=scale,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution="triton",
    )
    out, lse = result if return_lse else (result, None)
    ref_outs, ref_lses = [], []
    for index, (k, v) in enumerate(logical_kv):
        start, end = cu_q[index : index + 2]
        q_i = q[start:end].float()
        k_exp = k.float().repeat_interleave(q_heads // kv_heads, dim=1)
        v_exp = v.float().repeat_interleave(q_heads // kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_i, k_exp) * scale
        q_pos = prefix_lengths[index] + torch.arange(end - start, device=device)
        k_pos = torch.arange(lengths[index], device=device)
        mask = torch.ones(
            (end - start, lengths[index]), device=device, dtype=torch.bool
        )
        if is_causal:
            mask &= q_pos[:, None] >= k_pos[None, :]
        if window_left > 0:
            mask &= q_pos[:, None] <= k_pos[None, :] + window_left
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        if sinks is not None:
            sink_scores = sinks.float()[:, None, None].expand(-1, end - start, 1)
            scores = torch.cat((scores, sink_scores), dim=-1)
        probs = torch.softmax(scores, dim=-1)[..., : lengths[index]]
        ref_outs.append(torch.einsum("hqk,khd->qhd", probs, v_exp))
        ref_lses.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(), torch.cat(ref_outs, dim=0), rtol=8e-2, atol=8e-2
    )
    if return_lse:
        assert lse.shape == (cu_q[-1], q_heads) and lse.dtype == torch.float32
        assert torch.isfinite(lse).all()
        torch.testing.assert_close(
            lse, torch.cat(ref_lses, dim=0), rtol=8e-2, atol=8e-2
        )
    for original, unchanged in zip(inputs, before):
        assert torch.equal(original, unchanged)
