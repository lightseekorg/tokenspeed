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

"""Tests for fused sampled DSpark Markov kernels."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling import (
    dspark_markov_sample_workspace,
    dspark_markov_softmax,
    dspark_probs_scatter_and_sample,
)
from tokenspeed_kernel.ops.sampling.packed_verify import (
    DRAFT_POS_HASH_MUL,
    RNG_TAG_DRAFT,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _philox4x32_10_uniform(seed: int, position: int, tag: int) -> float:
    mask32 = 0xFFFFFFFF
    bits = seed & 0xFFFFFFFFFFFFFFFF
    pos = position & 0xFFFFFFFFFFFFFFFF
    k0, k1 = bits & mask32, (bits >> 32) & mask32
    x0, x1, x2, x3 = pos & mask32, (pos >> 32) & mask32, tag & mask32, 0
    for _ in range(10):
        p0 = x0 * 0xD2511F53
        p1 = x2 * 0xCD9E8D57
        x0 = ((p1 >> 32) ^ x1 ^ k0) & mask32
        x1 = p1 & mask32
        x2 = ((p0 >> 32) ^ x3 ^ k1) & mask32
        x3 = p0 & mask32
        k0 = (k0 + 0x9E3779B9) & mask32
        k1 = (k1 + 0xBB67AE85) & mask32
    return (x0 >> 8) / float(1 << 24)


def _reference_draw(row_probs: torch.Tensor, coin: float) -> int:
    values = row_probs.float()
    mass = float(values.sum().item())
    if mass > 0.0:
        threshold = coin * mass
        cdf = values.cumsum(0)
        hits = torch.nonzero((values > 0.0) & (cdf > threshold), as_tuple=False)
        if hits.numel() > 0:
            return int(hits[0].item())
    return int(torch.argmax(values).item())


def _setup(
    rows: int,
    block: int,
    vocab: int,
    local_vocab: int,
    rank: int,
    pool_rows: int = 16,
    *,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 17,
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base_logits = torch.randn((rows, block, local_vocab), device="cuda", generator=gen)
    # Power-of-two fractions keep every BF16->FP32 dot product summation-order exact.
    embedding = (
        torch.randint(-3, 4, (vocab, rank), device="cuda", generator=gen).float() / 8.0
    ).to(dtype)
    projection = (
        torch.randint(-3, 4, (local_vocab, rank), device="cuda", generator=gen).float()
        / 8.0
    ).to(dtype)
    temperature_pool = torch.ones((pool_rows,), device="cuda", dtype=torch.float32)
    top_k_pool = torch.full((pool_rows,), 1 << 30, device="cuda", dtype=torch.int32)
    top_p_pool = torch.ones((pool_rows,), device="cuda", dtype=torch.float32)
    seed_pool = torch.arange(pool_rows, device="cuda", dtype=torch.int64) * 97 - 11
    ws = dspark_markov_sample_workspace(
        max_rows=max(rows, 1),
        vocab_size=vocab,
        rank=rank,
        embed_dtype=dtype,
        device="cuda",
    )
    return (
        base_logits,
        embedding,
        projection,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        ws,
    )


@pytest.mark.parametrize(
    ("rows", "block", "vocab", "local_vocab", "rank"),
    [
        (1, 2, 32, 48, 24),
        (5, 3, 1025, 1088, 32),
        (19, 4, 4100, 4128, 512),
    ],
)
def test_dspark_markov_softmax_and_scatter_match_reference(
    rows: int, block: int, vocab: int, local_vocab: int, rank: int
):
    pool_rows = 24
    (
        base_logits,
        embedding,
        projection,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        ws,
    ) = _setup(rows, block, vocab, local_vocab, rank, pool_rows=pool_rows)
    slots = torch.tensor([(i * 3 + 1) % pool_rows for i in range(rows)], device="cuda")
    write_slots = torch.tensor(
        [(pool_rows - 1 - i) % pool_rows for i in range(rows)], device="cuda"
    )
    for idx, slot in enumerate(slots.tolist()):
        if idx % 3 == 0:
            # Canonical lowest-index greedy tie-breaking at top_k == 1.
            top_k_pool[slot] = 1
            base_logits[idx, 0, :vocab] = 0.0
            if rows == 1:
                embedding.zero_()
        else:
            temperature_pool[slot] = 0.65 + 0.4 * (idx % 3)

    start_pos = torch.arange(rows, device="cuda", dtype=torch.int64) * 19 + 41
    # Guarded token storage (extra rows & anchor col 0) and out-of-range anchor.
    tokens = torch.full((rows + 2, block + 1), -99, device="cuda", dtype=torch.int32)
    tokens[1 : 1 + rows, 0] = torch.tensor(
        [
            -5 if i == 0 else (vocab + 7 if i == rows - 1 else (i * 11) % vocab)
            for i in range(rows)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    anchor_ids = tokens[1 : 1 + rows, 0]
    tokens_before = tokens.clone()
    draft_probs = torch.full(
        (pool_rows, block, vocab), -3.0, device="cuda", dtype=torch.float32
    )
    draft_before = draft_probs.clone()

    prev = anchor_ids.clone()
    expected_tokens = []
    expected_probs = []
    for step in range(block):
        clamped = prev.clamp(0, vocab - 1).long()
        emb = embedding[clamped]
        bias_fp32 = emb.float() @ projection[:vocab].float().T
        logits = base_logits[:, step, :vocab].float() + bias_fp32
        row_probs = []
        row_toks = []
        for r in range(rows):
            slot = int(slots[r].item())
            pos = int(start_pos[r].item()) + 1 + (step + 1) * DRAFT_POS_HASH_MUL
            if top_k_pool[slot].item() == 1:
                best = int(torch.argmax(logits[r]).item())
                p = torch.zeros((vocab,), device="cuda", dtype=torch.float32)
                p[best] = 1.0
                tok = best
            else:
                temp = float(temperature_pool[slot].item())
                p = torch.softmax(logits[r] / temp, dim=-1, dtype=torch.float32)
                coin = _philox4x32_10_uniform(
                    int(seed_pool[slot].item()), pos, RNG_TAG_DRAFT
                )
                tok = _reference_draw(p, coin)
            row_probs.append(p)
            row_toks.append(tok)
        expected_probs.append(torch.stack(row_probs, dim=0))
        prev = torch.tensor(row_toks, device="cuda", dtype=torch.int32)
        expected_tokens.append(prev)

        actual_p = dspark_markov_softmax(
            base_logits,
            step,
            anchor_ids,
            embedding,
            projection,
            slots,
            temperature_pool,
            top_k_pool,
            top_p_pool,
            seed_pool,
            ws,
            vocab,
            prepare_step_inputs=(step == 0),
        )
        torch.testing.assert_close(actual_p, expected_probs[-1], atol=2e-6, rtol=2e-6)
        dspark_probs_scatter_and_sample(
            actual_p,
            write_slots,
            draft_probs,
            step,
            ws.seeds[:rows],
            start_pos,
            1 + (step + 1) * DRAFT_POS_HASH_MUL,
            RNG_TAG_DRAFT,
            embedding,
            tokens[1 : 1 + rows, 1:],
            ws,
            vocab,
            gather_next_embed=(step + 1 < block),
        )

    assert torch.equal(tokens[:, 0], tokens_before[:, 0])
    assert torch.equal(tokens[0], tokens_before[0])
    assert torch.equal(tokens[-1], tokens_before[-1])
    expected_block = torch.stack(expected_tokens, dim=1)
    assert torch.equal(tokens[1 : 1 + rows, 1:], expected_block)
    touched = set(write_slots.tolist())
    for slot in range(pool_rows):
        if slot not in touched:
            assert torch.equal(draft_probs[slot], draft_before[slot])
    for step in range(block):
        torch.testing.assert_close(
            draft_probs[write_slots, step],
            expected_probs[step],
            atol=2e-6,
            rtol=2e-6,
        )


def test_dspark_markov_no_triton_recompilation_and_empty_rows():
    base_logits, emb, proj, t_pool, k_pool, p_pool, s_pool, ws = _setup(
        18, 5, 2048, 2112, 64, pool_rows=16
    )
    slots = torch.arange(18, device="cuda") % 16
    write_slots = (slots + 3) % 16
    start_pos = torch.arange(18, device="cuda", dtype=torch.int64) * 5 + 10
    draft_probs = torch.zeros((16, 5, 2048), device="cuda", dtype=torch.float32)
    out = torch.empty((18, 5), device="cuda", dtype=torch.int32)
    anchors = torch.arange(18, device="cuda", dtype=torch.int32) % 2048

    empty_out = dspark_markov_softmax(
        base_logits[:0],
        0,
        anchors[:0],
        emb,
        proj,
        slots[:0],
        t_pool,
        k_pool,
        p_pool,
        s_pool,
        ws,
        2048,
        prepare_step_inputs=True,
    )
    assert empty_out.shape == (0, 2048)

    # Warm up both gather_next_embed={True, False}.
    for step in range(5):
        probs = dspark_markov_softmax(
            base_logits,
            step,
            anchors,
            emb,
            proj,
            slots,
            t_pool,
            k_pool,
            p_pool,
            s_pool,
            ws,
            2048,
            prepare_step_inputs=(step == 0),
        )
        dspark_probs_scatter_and_sample(
            probs,
            write_slots,
            draft_probs,
            step,
            ws.seeds[:18],
            start_pos,
            step + 1,
            RNG_TAG_DRAFT,
            emb,
            out,
            ws,
            2048,
            gather_next_embed=(step + 1 < 5),
        )

    with assert_no_triton_compile():
        for rows in (1, 3, 7, 16, 17):
            strided_base = torch.cat(
                [base_logits[:rows], torch.zeros_like(base_logits[:rows])], dim=-1
            )[:, :, :2112]
            strided_out = torch.empty((rows, 10), device="cuda", dtype=torch.int32)[
                :, :5
            ]
            for step in range(5):
                probs = dspark_markov_softmax(
                    strided_base,
                    step,
                    anchors[:rows],
                    emb,
                    proj,
                    slots[:rows],
                    t_pool,
                    k_pool,
                    p_pool,
                    s_pool,
                    ws,
                    2048,
                    prepare_step_inputs=(step == 0),
                )
                dspark_probs_scatter_and_sample(
                    probs,
                    write_slots[:rows],
                    draft_probs,
                    step,
                    ws.seeds[:rows],
                    start_pos[:rows],
                    step + 1,
                    RNG_TAG_DRAFT,
                    emb,
                    strided_out,
                    ws,
                    2048,
                    gather_next_embed=(step + 1 < 5),
                )


def test_dspark_markov_scatter_beyond_2gb_and_matches_sample_probs():
    cute_verify = pytest.importorskip("tokenspeed_kernel.ops.sampling._cute_dsl")
    del cute_verify
    from tokenspeed_kernel.ops.sampling.packed_verify import sample_probs

    vocab = 129280
    # 5000 * 1 * 129280 fp32 = ~2.585 GB (> 2^31 bytes).
    pool_rows = 5000
    rows, block, rank = 2, 2, 64
    base_logits, emb, proj, t_pool, k_pool, p_pool, s_pool, ws = _setup(
        rows, block, vocab, vocab, rank, pool_rows=16, seed=71
    )
    slots = torch.tensor([2, 9], device="cuda", dtype=torch.int64)
    write_slots = torch.tensor([1, pool_rows - 1], device="cuda", dtype=torch.int64)
    t_pool[slots] = torch.tensor([0.75, 1.2], device="cuda")
    start_pos = torch.tensor([101, 505], device="cuda", dtype=torch.int64)
    anchors = torch.tensor([13, 99], device="cuda", dtype=torch.int32)
    out = torch.empty((rows, block), device="cuda", dtype=torch.int32)

    try:
        draft_probs = torch.empty(
            (pool_rows, 1, vocab), device="cuda", dtype=torch.float32
        )
        assert draft_probs.numel() * draft_probs.element_size() > (1 << 31)
        probs = dspark_markov_softmax(
            base_logits,
            0,
            anchors,
            emb,
            proj,
            slots,
            t_pool,
            k_pool,
            p_pool,
            s_pool,
            ws,
            vocab,
            prepare_step_inputs=True,
        )
        dspark_probs_scatter_and_sample(
            probs,
            write_slots,
            draft_probs,
            0,
            ws.seeds[:rows],
            start_pos,
            DRAFT_POS_HASH_MUL + 1,
            RNG_TAG_DRAFT,
            emb,
            out,
            ws,
            vocab,
            gather_next_embed=True,
        )
        torch.testing.assert_close(draft_probs[1, 0], probs[0], atol=0, rtol=0)
        torch.testing.assert_close(
            draft_probs[pool_rows - 1, 0], probs[1], atol=0, rtol=0
        )

        # Sample tokens must match CuTe DSL packed_verify.sample_probs bit for bit.
        cute_out = torch.empty((rows,), device="cuda", dtype=torch.int32)
        sample_probs(
            probs=probs,
            seeds=s_pool.index_select(0, slots),
            positions=start_pos + DRAFT_POS_HASH_MUL + 1,
            output=cute_out,
            rng_tag=RNG_TAG_DRAFT,
        )
        assert torch.equal(out[:, 0], cute_out)
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
