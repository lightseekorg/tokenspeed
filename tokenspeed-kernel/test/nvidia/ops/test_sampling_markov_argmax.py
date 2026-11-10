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

"""The fused greedy DSpark Markov step on one rank.

``markov_argmax_step`` folds one proposal position of the DSpark walk -- the
Markov bias GEMM, the add to the base logits and the argmax -- into one
launch. The reference is the unfused walk: ``bias = embedding[clamp(prev)] @
projection.T`` rounded to the logits dtype, ``logits = base + bias`` rounded
again, ``torch.argmax`` over the vocabulary. Integer-valued Markov weights
make every bias exact in FP32 whatever the summation order, so the kernel
must match that reference bit for bit, ties included; random weights may only
differ where the bias's FP32 summation order flips a rounding, i.e. at
near-ties. The cross-rank exchange is covered by
``test_sampling_markov_argmax_distributed.py``.
"""

from __future__ import annotations

import pytest
import torch

cute_dsl = pytest.importorskip("tokenspeed_kernel.ops.sampling.cute_dsl")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and cute_dsl.is_available()),
    reason="the fused Markov step is a CuTe DSL kernel for NVIDIA sm_90+",
)

# Kimi-K3's head shard at TP8 and its Markov rank.
VOCAB = 20480
RANK = 512
MAX_ROWS = 130
TABLE_WIDTH = 8


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _workspace(vocab: int, rank_dim: int, dtype: torch.dtype, max_rows: int):
    return cute_dsl.create_markov_argmax_workspace(
        None,
        device=_device(),
        max_rows=max_rows,
        vocab_per_rank=vocab,
        rank_dim=rank_dim,
        dtype=dtype,
    )


def _integer_weights(rows: int, rank_dim: int, dtype, gen) -> torch.Tensor:
    """Draw multiples of 1/8 up to 3/8 in magnitude.

    Every product is a multiple of 1/64 and every bias an exact FP32 sum in
    any order, of the base logits' magnitude.
    """
    values = torch.randint(-3, 4, (rows, rank_dim), generator=gen, device="cuda")
    return (values / 8).to(dtype)


def _reference(base, prev, embedding, projection, *, exact: bool) -> torch.Tensor:
    """Logits of the unfused walk for ``prev``."""
    latent = embedding[prev.long().clamp(0, embedding.shape[0] - 1)]
    if exact:
        # The GEMM's FP32 dot, then its epilogue's rounding to the dtype.
        bias = (latent.float() @ projection.float().T).to(base.dtype)
    else:
        bias = torch.matmul(latent, projection.T)
    return base + bias


def _step(workspace, base, table, prev_col, out_col, embedding, projection):
    cute_dsl.markov_argmax_step(
        workspace,
        base,
        table[:, prev_col],
        embedding,
        projection,
        0,
        table[:, out_col],
    )


def _inputs(rows: int, seed: int, dtype=torch.bfloat16, vocab=VOCAB, rank=RANK):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = (torch.randn(rows, vocab, generator=gen, device="cuda") * 4).to(dtype)
    embedding = _integer_weights(vocab, rank, dtype, gen)
    projection = _integer_weights(vocab, rank, dtype, gen)
    table = torch.randint(
        0, vocab, (rows, TABLE_WIDTH), generator=gen, device="cuda", dtype=torch.int32
    )
    return base, embedding, projection, table


@pytest.fixture(scope="module")
def workspace():
    return _workspace(VOCAB, RANK, torch.bfloat16, MAX_ROWS)


@pytest.mark.parametrize("rows", [1, 3, 8, 9, 16, 17, 40, 64, 65, 130])
def test_matches_the_unfused_walk_bit_for_bit(workspace, rows):
    """Row counts cover every row-tile kernel and several CTA rows."""
    base, embedding, projection, table = _inputs(rows, seed=rows)
    before = table.clone()
    _step(workspace, base, table, 2, 3, embedding, projection)

    logits = _reference(base, before[:, 2], embedding, projection, exact=True)
    expected = torch.argmax(logits.float(), dim=-1).to(torch.int32)
    torch.testing.assert_close(table[:, 3], expected, rtol=0, atol=0)
    # Only the output column is written.
    keep = [c for c in range(TABLE_WIDTH) if c != 3]
    torch.testing.assert_close(table[:, keep], before[:, keep], rtol=0, atol=0)


def test_random_weights_differ_only_at_near_ties(workspace):
    """Random weights disagree with the unfused path only at near-ties.

    Against the unfused path's own BF16 GEMM only the bias's FP32 summation
    order differs.
    """
    rows, agree, total = 64, 0, 0
    for seed in range(8):
        gen = torch.Generator(device="cuda").manual_seed(100 + seed)
        base = (torch.randn(rows, VOCAB, generator=gen, device="cuda") * 3).to(
            torch.bfloat16
        )
        embedding = (torch.randn(VOCAB, RANK, generator=gen, device="cuda") * 0.05).to(
            torch.bfloat16
        )
        projection = (torch.randn(VOCAB, RANK, generator=gen, device="cuda") * 0.05).to(
            torch.bfloat16
        )
        table = torch.randint(
            0,
            VOCAB,
            (rows, TABLE_WIDTH),
            generator=gen,
            device="cuda",
            dtype=torch.int32,
        )
        prev = table[:, 0].clone()
        _step(workspace, base, table, 0, 1, embedding, projection)

        logits = _reference(base, prev, embedding, projection, exact=False).float()
        expected = torch.argmax(logits, dim=-1)
        got = table[:, 1].long()
        best = logits.max(dim=-1).values
        chosen = logits.gather(1, got[:, None]).squeeze(1)
        slack = best.abs().clamp(min=1.0) * 2 * torch.finfo(torch.bfloat16).eps
        assert bool((chosen >= best - slack).all()), "a non-tie disagreed"
        agree += int((got == expected).sum())
        total += rows
    print(f"fused vs unfused agreement: {agree}/{total}")
    assert agree >= 0.99 * total


def test_out_of_range_previous_tokens_clamp_to_the_table_edges(workspace):
    base, embedding, projection, table = _inputs(6, seed=7)
    prev = torch.tensor(
        [-7, -1, VOCAB, VOCAB + 1000, 0, VOCAB - 1], dtype=torch.int32, device="cuda"
    )
    table[:, 0] = prev
    _step(workspace, base, table, 0, 1, embedding, projection)
    logits = _reference(base, prev, embedding, projection, exact=True)
    expected = torch.argmax(logits.float(), dim=-1).to(torch.int32)
    torch.testing.assert_close(table[:, 1], expected, rtol=0, atol=0)


def test_ties_go_to_the_lowest_token(workspace):
    """Equal maxima resolve to the lowest column, like torch.argmax.

    The maxima sit in different CTAs, warps and lanes; -0.0 ties with +0.0.
    """
    rows = 4
    embedding = torch.zeros(VOCAB, RANK, dtype=torch.bfloat16, device="cuda")
    projection = torch.ones(VOCAB, RANK, dtype=torch.bfloat16, device="cuda")
    base = torch.full((rows, VOCAB), -5.0, dtype=torch.bfloat16, device="cuda")
    base[0] = 1.0  # every column ties
    base[1, [20000, 9000, 5, 77]] = 2.0
    base[2, [VOCAB - 1, VOCAB - 9]] = 2.0
    base[3] = float("-inf")
    base[3, 4000] = -0.0
    base[3, 12000] = 0.0
    table = torch.zeros(rows, TABLE_WIDTH, dtype=torch.int32, device="cuda")
    _step(workspace, base, table, 0, 1, embedding, projection)
    assert table[:, 1].tolist() == [0, 5, VOCAB - 9, 4000]


def test_rows_without_a_finite_winner(workspace):
    """-inf and NaN never win; a row with nothing above -inf yields 0."""
    rows = 4
    embedding = torch.zeros(VOCAB, RANK, dtype=torch.bfloat16, device="cuda")
    projection = torch.zeros(VOCAB, RANK, dtype=torch.bfloat16, device="cuda")
    base = torch.full((rows, VOCAB), float("-inf"), dtype=torch.bfloat16, device="cuda")
    base[1] = float("nan")
    base[2] = float("nan")
    base[2, 1234] = -100.0
    base[3, 17000] = -3.0
    table = torch.full((rows, TABLE_WIDTH), 99, dtype=torch.int32, device="cuda")
    _step(workspace, base, table, 0, 1, embedding, projection)
    assert table[:, 1].tolist() == [0, 0, 1234, 17000]


def test_partial_last_cta_and_small_rank():
    """A partial last CTA still lets its one real row group compete.

    A shard of 257 row groups leaves three idle warp pairs in the last CTA;
    they must not compete.
    """
    vocab, rank = 4112, 128
    ws = _workspace(vocab, rank, torch.bfloat16, 16)
    base, embedding, projection, table = _inputs(16, seed=3, vocab=vocab, rank=rank)
    base[0] = -100.0
    base[0, vocab - 3] = 100.0
    before = table.clone()
    _step(ws, base, table, 4, 5, embedding, projection)
    logits = _reference(base, before[:, 4], embedding, projection, exact=True)
    expected = torch.argmax(logits.float(), dim=-1).to(torch.int32)
    torch.testing.assert_close(table[:, 5], expected, rtol=0, atol=0)
    assert int(table[0, 5]) == vocab - 3


def test_fp16_matches_the_unfused_walk():
    ws = _workspace(VOCAB, RANK, torch.float16, 32)
    base, embedding, projection, table = _inputs(32, seed=11, dtype=torch.float16)
    before = table.clone()
    _step(ws, base, table, 1, 6, embedding, projection)
    logits = _reference(base, before[:, 1], embedding, projection, exact=True)
    expected = torch.argmax(logits.float(), dim=-1).to(torch.int32)
    torch.testing.assert_close(table[:, 6], expected, rtol=0, atol=0)


def _walk(workspace, block, table, embedding, projection):
    """The drafter's walk: position k reads the token written for k - 1."""
    for k in range(1, block.shape[0] + 1):
        _step(workspace, block[k - 1], table, k - 1, k, embedding, projection)


@pytest.mark.parametrize("rows", [1, 24, 64])
def test_chained_walk_replays_under_cuda_graph(workspace, rows):
    """Seven chained steps captured once replay correctly on fresh inputs.

    The scratch each launch leaves behind is what the next one expects.
    """
    steps = 7
    gen = torch.Generator(device="cuda").manual_seed(rows)
    embedding = _integer_weights(VOCAB, RANK, torch.bfloat16, gen)
    projection = _integer_weights(VOCAB, RANK, torch.bfloat16, gen)
    block = torch.empty(steps, rows, VOCAB, dtype=torch.bfloat16, device="cuda")
    table = torch.zeros(rows, TABLE_WIDTH, dtype=torch.int32, device="cuda")

    def refill(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        block.copy_(torch.randn(block.shape, generator=g, device="cuda") * 4)
        table.zero_()
        table[:, 0] = torch.randint(0, VOCAB, (rows,), generator=g, device="cuda")

    refill(0)
    _walk(workspace, block, table, embedding, projection)  # warm the launch path
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _walk(workspace, block, table, embedding, projection)

    for seed in (1, 2, 3):
        refill(seed)
        graph.replay()
        torch.cuda.synchronize()
        tokens = table[:, 0].clone()
        for k in range(1, steps + 1):
            logits = _reference(block[k - 1], tokens, embedding, projection, exact=True)
            tokens = torch.argmax(logits.float(), dim=-1).to(torch.int32)
            torch.testing.assert_close(table[:, k], tokens, rtol=0, atol=0)


def test_rejects_inputs_the_workspace_does_not_cover(workspace):
    base, embedding, projection, table = _inputs(4, seed=5)
    with pytest.raises(ValueError, match="base_logits"):
        _step(workspace, base.float(), table, 0, 1, embedding, projection)
    with pytest.raises(ValueError, match="projection"):
        _step(workspace, base, table, 0, 1, embedding, projection[:-16])
    with pytest.raises(ValueError, match="prev_tokens"):
        cute_dsl.markov_argmax_step(
            workspace,
            base,
            table[:, 0].long(),
            embedding,
            projection,
            0,
            table[:, 1],
        )
    rows = MAX_ROWS + 1
    big = torch.zeros(rows, VOCAB, dtype=torch.bfloat16, device="cuda")
    big_table = torch.zeros(rows, TABLE_WIDTH, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="max_rows"):
        _step(workspace, big, big_table, 0, 1, embedding, projection)


def test_supported_geometry():
    assert cute_dsl.supports_markov_argmax_shape(VOCAB, RANK, torch.bfloat16)
    assert cute_dsl.supports_markov_argmax_shape(VOCAB, 128, torch.float16)
    assert not cute_dsl.supports_markov_argmax_shape(VOCAB, RANK, torch.float32)
    assert not cute_dsl.supports_markov_argmax_shape(VOCAB + 8, RANK, torch.bfloat16)
    assert not cute_dsl.supports_markov_argmax_shape(VOCAB, 1024, torch.bfloat16)
    assert not cute_dsl.supports_markov_argmax_shape(VOCAB, 192, torch.bfloat16)
