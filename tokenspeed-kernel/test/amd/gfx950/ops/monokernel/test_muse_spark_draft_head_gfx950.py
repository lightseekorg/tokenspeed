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

"""The DFlash2 draft head (Muse Spark) in one launch across the 8 ranks: the
vocabulary-parallel top 16 of every draft step and the selector's scores."""

from __future__ import annotations

import contextlib
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.benchmark import muse_spark_monokernel as inputs
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_head,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN = inputs.WORLD, inputs.HIDDEN
SHARD, VOCAB, RANK_DIM, TOPK = (
    inputs.HEAD_SHARD,
    inputs.HEAD_VOCAB,
    inputs.HEAD_RANK_DIM,
    inputs.HEAD_TOPK,
)
STEPS = draft_head.STEPS
SCALE, SOFTCAP = inputs.HEAD_SCALE, inputs.HEAD_SOFTCAP


def _reference_candidates(x, shard, rank):
    """The regular path (``VocabParallelTopK``): this rank's BF16 logits
    times the scale, its top 16, the 8 ranks' 16s gathered, the best 16
    sorted, softcapped."""
    logits = torch.matmul(x, shard.T)
    logits.mul_(SCALE)
    values, ids = torch.topk(logits, TOPK, dim=-1, sorted=False)
    staged = torch.cat([values.float(), (ids + rank * SHARD).float()], dim=-1).cpu()
    gathered = [torch.empty_like(staged) for _ in range(WORLD)]
    dist.all_gather(gathered, staged)
    planes = torch.stack(gathered).view(WORLD, STEPS, 2, TOPK)
    planes = planes.permute(1, 2, 0, 3).reshape(STEPS, 2, WORLD * TOPK)
    values, lanes = torch.topk(planes[:, 0], TOPK, dim=-1, sorted=True)
    ids = torch.gather(planes[:, 1], 1, lanes).long()
    return ids, values


def _softcap(v: torch.Tensor) -> torch.Tensor:
    return SOFTCAP * (2 * torch.sigmoid(2 * v / SOFTCAP) - 1)


def _reference_scores(x, hidden_proj, pred, succ, anchor, ids, unary):
    """``CandidateSelector`` on the given candidates (models/dflash2.py)."""
    hidden = (x.float() @ hidden_proj.float().T).to(torch.bfloat16)
    prev = torch.cat([anchor.view(1, 1).expand(1, TOPK), ids[:-1]])
    q = pred[prev] * hidden[:, None, :]  # BF16 [steps, p, r]
    edge = torch.einsum("lpr,lcr->lpc", q, succ[ids])
    return unary[:, None, :] + edge.float()


def _worker(rank: int, port: int, seed: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        # Two requests' plan is the larger: one scratch serves both.
        plan = draft_head.head_plan(2 * STEPS)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        x, hidden_proj, pred, succ, anchor = inputs.draft_head_shared(
            seed, STEPS, device
        )
        shard = inputs.draft_head_shard(seed, rank, device)
        weights = draft_head.pack_draft_head(
            {
                "lm_head": shard,
                "hidden_proj": hidden_proj,
                "pred_codebook": pred,
                "succ_codebook": succ,
            },
            rank,
            SCALE,
            SOFTCAP,
        )
        outs = dict(
            anchor=torch.empty(1, dtype=torch.int64, device=device),
            cand_ids=torch.empty(STEPS, TOPK, dtype=torch.int64, device=device),
            unary=torch.empty(STEPS, TOPK, dtype=torch.float32, device=device),
            scores=torch.empty(STEPS, TOPK, TOPK, dtype=torch.float32, device=device),
        )
        for step in (5, 6):
            # The second launch, at another step, runs the first one's binary.
            with (
                assert_no_triton_compile(draft_head.gluon_muse_spark_draft_head_gfx950)
                if step == 6
                else contextlib.nullcontext()
            ):
                draft_head.launch_gluon_muse_spark_draft_head_gfx950(
                    weights,
                    x=x,
                    first_token=anchor.to(torch.int32),
                    **outs,
                    scratch=scratch,
                    step=torch.full((1,), step, dtype=torch.int32, device=device),
                    peer_bases=group.bases,
                    trace_buf=None,
                    compile_only=False,
                )
            torch.cuda.synchronize()
            _check(x, shard, rank, hidden_proj, pred, succ, anchor, outs)
            dist.barrier()
        # Two requests' steps in one launch: each request's outputs are the
        # bits of its own launch (the rows are independent, the anchor and
        # step 0's predecessors per request).
        xb, _, _, _, anchor_b = inputs.draft_head_shared(seed + 1, STEPS, device)
        one = {}
        for name, xs, first, step in (("a", x, anchor, 7), ("b", xb, anchor_b, 8)):
            one[name] = {k: torch.empty_like(v) for k, v in outs.items()}
            draft_head.launch_gluon_muse_spark_draft_head_gfx950(
                weights,
                x=xs,
                first_token=first.to(torch.int32),
                **one[name],
                scratch=scratch,
                step=torch.full((1,), step, dtype=torch.int32, device=device),
                peer_bases=group.bases,
                trace_buf=None,
                compile_only=False,
            )
        both = dict(
            anchor=torch.empty(2, dtype=torch.int64, device=device),
            cand_ids=torch.empty(2 * STEPS, TOPK, dtype=torch.int64, device=device),
            unary=torch.empty(2 * STEPS, TOPK, dtype=torch.float32, device=device),
            scores=torch.empty(
                2 * STEPS, TOPK, TOPK, dtype=torch.float32, device=device
            ),
        )
        draft_head.launch_gluon_muse_spark_draft_head_gfx950(
            weights,
            x=torch.cat([x, xb]),
            first_token=torch.cat([anchor, anchor_b]).to(torch.int32),
            **both,
            scratch=scratch,
            step=torch.full((1,), 9, dtype=torch.int32, device=device),
            peer_bases=group.bases,
            trace_buf=None,
            compile_only=False,
        )
        torch.cuda.synchronize()
        for q, name in enumerate(("a", "b")):
            rows = slice(q * STEPS, (q + 1) * STEPS)
            assert torch.equal(both["anchor"][q : q + 1], one[name]["anchor"])
            for k in ("cand_ids", "unary", "scores"):
                assert torch.equal(both[k][rows], one[name][k]), (name, k)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _exact(x, shard, rank, ids):
    """Each candidate's scaled logit in FP64 (the ranks' shards summed)."""
    local = ids.to(shard.device) - rank * SHARD
    mine = (local >= 0) & (local < SHARD)
    rows = shard[local.clamp(0, SHARD - 1)].double()  # [steps, 16, hidden]
    dots = torch.einsum("lh,lch->lc", x.double(), rows) * SCALE
    dots = torch.where(mine, dots, 0.0).cpu()
    dist.all_reduce(dots)
    return dots


def _check(x, shard, rank, hidden_proj, pred, succ, anchor, outs) -> None:
    assert torch.equal(outs["anchor"].cpu(), anchor.cpu())
    want_ids, want_values = _reference_candidates(x, shard, rank)
    got_ids, got_unary = outs["cand_ids"].cpu(), outs["unary"].cpu()
    # Every rank holds the same candidates, 16 distinct tokens a step.
    gathered = [torch.empty_like(got_ids) for _ in range(WORLD)]
    dist.all_gather(gathered, got_ids)
    assert all(torch.equal(g, got_ids) for g in gathered)
    assert all(len(set(r.tolist())) == TOPK for r in got_ids)
    # The values in order: the reference's, or one BF16 step apart where the
    # kernel's FP32 sum rounds a logit the other way.
    torch.testing.assert_close(got_unary, _softcap(want_values), rtol=2**-7, atol=0)
    # A valid top 16: each candidate's score is its own logit's, and the
    # chosen logits match the reference's chosen ones (ties in BF16 may pick
    # other tokens of the same value).
    got_exact, want_exact = _exact(x, shard, rank, got_ids), _exact(
        x, shard, rank, want_ids
    )
    torch.testing.assert_close(
        got_unary, _softcap(got_exact.float()), rtol=2**-7, atol=0
    )
    torch.testing.assert_close(
        got_exact.sort(dim=-1, descending=True).values,
        want_exact.sort(dim=-1, descending=True).values,
        rtol=2**-7,
        atol=0,
    )
    # Sorted by value, the lower token id first on a tie.
    assert (got_unary[:, :-1] >= got_unary[:, 1:]).all()
    tie = got_unary[:, :-1] == got_unary[:, 1:]
    assert (got_ids[:, :-1][tie] < got_ids[:, 1:][tie]).all()
    # The scores of the kernel's own candidates.
    want = _reference_scores(
        x.cpu(),
        hidden_proj.cpu(),
        pred.cpu(),
        succ.cpu(),
        anchor.cpu(),
        got_ids,
        got_unary,
    )
    got = outs["scores"].cpu()
    torch.testing.assert_close(
        got, want, rtol=2**-7, atol=want.abs().max().item() * 2**-9
    )


@pytest.mark.parametrize("seed", [0])
def test_draft_head_in_one_launch_matches_the_vocab_parallel_top16_and_selector(
    seed: int,
):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, seed), nprocs=WORLD, join=True)


def test_one_binary_serves_every_step_and_block():
    """Each launch changes the step, the block's rows and its first token,
    all device tensors: compiling the launches for one GPU's rank finds the
    first launch's binary. Another request count is another plan, which the
    guard must see compile."""
    dev = torch.device("cuda:0")
    x, hidden_proj, pred, succ, anchor = inputs.draft_head_shared(0, STEPS, dev)
    weights = draft_head.pack_draft_head(
        {
            "lm_head": inputs.draft_head_shard(0, 0, dev),
            "hidden_proj": hidden_proj,
            "pred_codebook": pred,
            "succ_codebook": succ,
        },
        0,
        SCALE,
        SOFTCAP,
    )
    widest = draft_head.head_plan(draft_head.MAX_REQUESTS * STEPS)
    # Compiling alone reads none of these: plain device memory stands in for
    # the symmetric heap and the peers.
    scratch = torch.zeros(widest.scratch_pairs, dtype=torch.int64, device=dev)
    peer_bases = torch.zeros(WORLD, dtype=torch.int64, device=dev)

    def compile_launch(step: int, requests: int, seed: int) -> None:
        rows = requests * STEPS
        g = torch.Generator(device=dev).manual_seed(seed)
        draft_head.launch_gluon_muse_spark_draft_head_gfx950(
            weights,
            x=torch.randn(rows, HIDDEN, generator=g, device=dev).to(torch.bfloat16),
            first_token=torch.randint(
                0, VOCAB, (requests,), generator=g, device=dev, dtype=torch.int32
            ),
            anchor=torch.empty(requests, dtype=torch.int64, device=dev),
            cand_ids=torch.empty(rows, TOPK, dtype=torch.int64, device=dev),
            unary=torch.empty(rows, TOPK, dtype=torch.float32, device=dev),
            scores=torch.empty(rows, TOPK, TOPK, dtype=torch.float32, device=dev),
            scratch=scratch,
            step=torch.full((1,), step, dtype=torch.int32, device=dev),
            peer_bases=peer_bases,
            trace_buf=None,
            compile_only=True,
        )

    compile_launch(5, 1, 1)
    with assert_no_triton_compile(draft_head.gluon_muse_spark_draft_head_gfx950):
        for step in (6, 7, 1 << 20):
            compile_launch(step, 1, step)
    # The guard is live: two requests' steps are a new plan and compile.
    with pytest.raises(AssertionError, match="compiled 1 new"):
        with assert_no_triton_compile(draft_head.gluon_muse_spark_draft_head_gfx950):
            compile_launch(8, 2, 8)


def test_the_lm_head_tiles_are_built_without_a_padded_copy():
    """The packer tiles the rank's LM-head shard a chunk of 16-row blocks at
    a time, zero-padding the last block, so startup never holds a padded
    copy of the shard next to the tiles: the tiles equal those of the padded
    shard, byte for byte."""
    shard, hidden = int(draft_head.SHARD), int(draft_head.HIDDEN)
    gen = torch.Generator(device="cuda").manual_seed(3)
    head = torch.randn(shard, hidden, device="cuda", generator=gen).to(torch.bfloat16)
    padded = head.new_zeros(int(draft_head.BLOCKS) * 16, hidden)
    padded[:shard] = head
    expected = bf16_tiles(padded)
    got = draft_head._padded_tiles(head)
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
