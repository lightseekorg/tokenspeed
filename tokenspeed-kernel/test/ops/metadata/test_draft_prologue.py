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

"""eagle_chain_prologue against the chain drafter's tensor operations."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.metadata import eagle_chain_prologue, eagle_chain_step
from tokenspeed_kernel.ops.metadata.draft_prologue import (
    _eagle_chain_prologue_kernel,
    _eagle_chain_step_kernel,
)
from utils import assert_no_triton_compile

if not torch.cuda.is_available():
    pytest.skip("a GPU is required", allow_module_level=True)

WINDOW, POOL = 4, 256


def _inputs(bs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return dict(
        tokens=torch.randint(0, 1000, (bs * WINDOW,), generator=g).int().cuda(),
        accept_lengths=torch.randint(1, WINDOW + 1, (bs,), generator=g).int().cuda(),
        gather_offsets=(torch.arange(bs) * WINDOW - 1).cuda(),
        valid_cache_lengths=torch.randint(0, 5000, (POOL,), generator=g).int().cuda(),
        req_pool_indices=torch.randperm(POOL, generator=g)[:bs].cuda(),
    )


def _run(ops, bs):
    out = dict(
        next_tokens=torch.full((bs, WINDOW), -7, dtype=torch.int32, device="cuda"),
        gather_ids=torch.zeros(bs, dtype=torch.int64, device="cuda"),
        frontier=torch.zeros(bs, dtype=torch.int32, device="cuda"),
        seq_lens=torch.zeros(bs, dtype=torch.int32, device="cuda"),
        positions=torch.zeros(bs, dtype=torch.int64, device="cuda"),
    )
    eagle_chain_prologue(**ops, **out)
    return out


@pytest.mark.parametrize("bs", [1, 3, 4, 130])
def test_matches_the_drafter_ops(bs):
    ops = _inputs(bs, bs)
    got = _run(ops, bs)
    # Eagle.draft, _accepted_frontier and _run_multi_step_decode.
    gather = ops["gather_offsets"] + ops["accept_lengths"]
    token = ops["tokens"][gather].int()
    frontier = (
        ops["valid_cache_lengths"].index_select(0, ops["req_pool_indices"])
        + ops["accept_lengths"]
    )
    assert torch.equal(got["gather_ids"], gather)
    assert torch.equal(got["next_tokens"], token[:, None].expand(-1, WINDOW))
    assert torch.equal(got["frontier"], frontier)
    assert torch.equal(got["seq_lens"], frontier + 1)
    assert torch.equal(got["positions"], frontier.long())


def test_batch_size_does_not_recompile():
    _run(_inputs(1, 0), 1)
    with assert_no_triton_compile(_eagle_chain_prologue_kernel):
        for bs in (2, 3, 4, 16, 17):
            _run(_inputs(bs, bs), bs)


@pytest.mark.parametrize("advance", [True, False])
@pytest.mark.parametrize("bs", [1, 3, 200])
def test_chain_step_writes_its_column_and_moves_the_window(bs, advance):
    """A step's tokens land in their window column, and an advancing step
    moves the positions and lengths on one token, as the separate tensor
    operations do."""
    gen = torch.Generator(device="cuda").manual_seed(bs)
    ids = torch.randint(0, 1000, (bs,), device="cuda", generator=gen)
    window = torch.randint(0, 1000, (bs, 4), dtype=torch.int32, device="cuda")
    positions = torch.randint(0, 9000, (bs,), device="cuda", generator=gen)
    lengths = (positions + 1).int()
    want = (window.clone(), positions.clone(), lengths.clone())
    want[0][:, 2] = ids.int()
    if advance:
        want[1].add_(1)
        want[2].add_(1)

    eagle_chain_step(
        ids,
        next_tokens=window,
        column=2,
        positions=positions,
        seq_lens=lengths,
        advance=advance,
    )
    assert torch.equal(window, want[0])
    assert torch.equal(positions, want[1])
    assert torch.equal(lengths, want[2])


def test_chain_step_compiles_once_across_batches_and_columns():
    window = torch.zeros(300, 4, dtype=torch.int32, device="cuda")
    positions = torch.zeros(300, dtype=torch.int64, device="cuda")
    lengths = torch.zeros(300, dtype=torch.int32, device="cuda")

    def run(bs, column):
        ids = torch.zeros(bs, dtype=torch.int64, device="cuda")
        eagle_chain_step(
            ids,
            next_tokens=window[:bs],
            column=column,
            positions=positions[:bs],
            seq_lens=lengths[:bs],
            advance=True,
        )

    run(2, 1)
    with assert_no_triton_compile(_eagle_chain_step_kernel):
        for bs, column in ((1, 2), (16, 3), (7, 1), (300, 2)):
            run(bs, column)
