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

"""The fused greedy DSpark Markov step across a vocabulary-sharded group.

Each rank holds its shard of the base logits and of the Markov projection and
the whole replicated embedding; ``markov_argmax_step`` must return, on every
rank, the argmax over the concatenated vocabulary of ``round(base +
round(bias))``, exchanging candidates through the distributed argmax's
slots. Integer-valued weights make the reference exact (see the one-GPU
test); random weights are compared with the unfused path the drafter runs
otherwise -- the BF16 bias GEMM, the BF16 add, and ``distributed_argmax`` --
and may only differ at near-ties.

Normal one-GPU pytest runs skip this file. Exercise it with:
``torchrun --standalone --nproc-per-node=8 -m pytest -q -s <this file>``
(any group size from 2 to 8).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

cute_dsl = pytest.importorskip("tokenspeed_kernel.ops.sampling.cute_dsl")

# Kimi-K3's vocabulary and Markov rank.
VOCAB = 163840
RANK = 512
MAX_ROWS = 100
TABLE_WIDTH = 8


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


pytestmark = pytest.mark.skipif(
    not 2 <= _world_size() <= 8 or VOCAB % _world_size() != 0,
    reason="launch with torchrun, 2 to 8 ranks",
)


def _setup():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), torch.device("cuda", rank)


_STATE: dict[str, object] = {}


def _state():
    """One distributed-argmax state and step workspace, agreed by all ranks."""
    rank, device = _setup()
    if "state" not in _STATE:
        ok = cute_dsl.is_available()
        vote = torch.tensor([int(ok)], dtype=torch.int32, device=device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        state = None
        if vote.item():
            state = cute_dsl.try_create_dist_argmax_state(
                group=dist.group.WORLD,
                rank_in_group=rank,
                max_M=MAX_ROWS,
                dtype=torch.bfloat16,
                device=device,
                skip_ping_pong=False,
            )
            vote.fill_(int(state is not None))
            dist.all_reduce(vote, op=dist.ReduceOp.MIN)
            if not vote.item():
                state = None
        workspace = None
        if state is not None:
            workspace = cute_dsl.create_markov_argmax_workspace(
                state,
                device=device,
                max_rows=MAX_ROWS,
                vocab_per_rank=VOCAB // _world_size(),
                rank_dim=RANK,
                dtype=torch.bfloat16,
            )
        _STATE["state"], _STATE["workspace"] = state, workspace
    if _STATE["state"] is None:
        pytest.skip("no CuTe DSL kernel or rank-agreed multicast (NVLS) support")
    return rank, device, _STATE["state"], _STATE["workspace"]


def _shard(rank: int) -> slice:
    width = VOCAB // _world_size()
    return slice(rank * width, (rank + 1) * width)


def _full_inputs(rows: int, seed: int, *, exact: bool):
    """The same full tensors on every rank (same seed)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = (torch.randn(rows, VOCAB, generator=gen, device="cuda") * 3).to(
        torch.bfloat16
    )

    def weights():
        if exact:  # multiples of 1/8: exact FP32 biases in any order
            values = torch.randint(
                -3, 4, (VOCAB, RANK), generator=gen, device="cuda", dtype=torch.int8
            )
            return values.bfloat16() / 8
        values = torch.randn(
            VOCAB, RANK, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        return values * 0.05

    embedding, projection = weights(), weights()
    table = torch.randint(
        0, VOCAB, (rows, TABLE_WIDTH), generator=gen, device="cuda", dtype=torch.int32
    )
    return base, embedding, projection, table


def _step(workspace, rank, base, table, prev_col, out_col, embedding, projection):
    shard = _shard(rank)
    cute_dsl.markov_argmax_step(
        workspace,
        base[:, shard].contiguous(),
        table[:, prev_col],
        embedding,
        projection[shard],
        shard.start,
        table[:, out_col],
    )


def _exact_logits(base, prev, embedding, projection) -> torch.Tensor:
    latent = embedding[prev.long().clamp(0, VOCAB - 1)].float()
    return base + (latent @ projection.float().T).to(base.dtype)


@pytest.mark.parametrize("rows", [1, 7, 16, 33, 64, 100])
def test_matches_the_full_vocabulary_argmax(rows):
    rank, _, _, workspace = _state()
    base, embedding, projection, table = _full_inputs(rows, seed=rows, exact=True)
    prev = table[:, 0].clone()
    _step(workspace, rank, base, table, 0, 1, embedding, projection)
    expected = torch.argmax(
        _exact_logits(base, prev, embedding, projection).float(), dim=-1
    )
    torch.testing.assert_close(table[:, 1], expected.to(torch.int32), rtol=0, atol=0)


def test_random_weights_against_the_unfused_path():
    """Random weights disagree with the drafter's other path only at near-ties.

    That path is the BF16 bias GEMM per shard, the BF16 add, then
    ``distributed_argmax``.
    """
    rank, _, state, workspace = _state()
    rows, agree, total = 64, 0, 0
    shard = _shard(rank)
    for seed in range(4):
        base, embedding, projection, table = _full_inputs(rows, 50 + seed, exact=False)
        prev = table[:, 0].clone()
        _step(workspace, rank, base, table, 0, 1, embedding, projection)
        latent = embedding[prev.long()]
        logits = base[:, shard] + torch.matmul(latent, projection[shard].T)
        _, unfused = cute_dsl.distributed_argmax(state, logits.contiguous())

        full = base + torch.matmul(latent, projection.T)
        full = full.float()
        best = full.max(dim=-1).values
        chosen = full.gather(1, table[:, 1].long()[:, None]).squeeze(1)
        slack = best.abs().clamp(min=1.0) * 2 * torch.finfo(torch.bfloat16).eps
        assert bool((chosen >= best - slack).all()), "a non-tie disagreed"
        agree += int((table[:, 1].long() == unfused).sum())
        total += rows
    if rank == 0:
        print(f"fused vs unfused (distributed_argmax) agreement: {agree}/{total}")
    assert agree >= 0.98 * total


def test_ties_across_ranks_go_to_the_lowest_token():
    rank, _, _, workspace = _state()
    world, width = _world_size(), VOCAB // _world_size()
    rows = 3
    embedding = torch.zeros(VOCAB, RANK, dtype=torch.bfloat16, device="cuda")
    projection = torch.zeros(VOCAB, RANK, dtype=torch.bfloat16, device="cuda")
    base = torch.full((rows, VOCAB), -4.0, dtype=torch.bfloat16, device="cuda")
    base[0, [(world - 1) * width + 30, width + 9, width + 2000]] = 7.0
    base[1] = float("-inf")
    base[2, [VOCAB - 1, (world - 1) * width]] = 7.0
    table = torch.zeros(rows, TABLE_WIDTH, dtype=torch.int32, device="cuda")
    _step(workspace, rank, base, table, 0, 1, embedding, projection)
    assert table[:, 1].tolist() == [width + 9, 0, (world - 1) * width]


def test_interleaves_with_distributed_argmax_on_one_state():
    """Both kernels follow the same band order on the shared state."""
    rank, _, state, workspace = _state()
    rows = 16
    shard = _shard(rank)
    for seed in range(3):
        base, embedding, projection, table = _full_inputs(rows, 70 + seed, exact=True)
        prev = table[:, 0].clone()
        _step(workspace, rank, base, table, 0, 1, embedding, projection)
        logits = _exact_logits(base, prev, embedding, projection)
        _, plain = cute_dsl.distributed_argmax(state, logits[:, shard].contiguous())
        _step(workspace, rank, base, table, 0, 2, embedding, projection)
        expected = torch.argmax(logits.float(), dim=-1).to(torch.int32)
        torch.testing.assert_close(table[:, 1], expected, rtol=0, atol=0)
        torch.testing.assert_close(table[:, 2], expected, rtol=0, atol=0)
        # distributed_argmax breaks exact in-shard ties by its scan order.
        top2 = logits.float().topk(2, dim=-1).values
        unique = top2[:, 0] > top2[:, 1]
        torch.testing.assert_close(
            plain.to(torch.int32)[unique], expected[unique], rtol=0, atol=0
        )


@pytest.mark.parametrize("rows", [1, 64])
def test_chained_walk_replays_under_cuda_graph(rows):
    rank, _, _, workspace = _state()
    steps = 7
    shard = _shard(rank)
    _, embedding, projection, _ = _full_inputs(1, seed=900, exact=True)
    block = torch.empty(steps, rows, VOCAB, dtype=torch.bfloat16, device="cuda")
    local = torch.empty(
        steps, rows, shard.stop - shard.start, dtype=torch.bfloat16, device="cuda"
    )
    table = torch.zeros(rows, TABLE_WIDTH, dtype=torch.int32, device="cuda")

    def refill(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        block.copy_(torch.randn(block.shape, generator=g, device="cuda") * 3)
        local.copy_(block[:, :, shard])
        table.zero_()
        table[:, 0] = torch.randint(0, VOCAB, (rows,), generator=g, device="cuda")

    def walk():
        for k in range(1, steps + 1):
            cute_dsl.markov_argmax_step(
                workspace,
                local[k - 1],
                table[:, k - 1],
                embedding,
                projection[shard],
                shard.start,
                table[:, k],
            )

    refill(0)
    walk()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        walk()
    for seed in (1, 2, 3):
        refill(seed)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        tokens = table[:, 0].clone()
        for k in range(1, steps + 1):
            logits = _exact_logits(block[k - 1], tokens, embedding, projection)
            tokens = torch.argmax(logits.float(), dim=-1).to(torch.int32)
            torch.testing.assert_close(table[:, k], tokens, rtol=0, atol=0)
