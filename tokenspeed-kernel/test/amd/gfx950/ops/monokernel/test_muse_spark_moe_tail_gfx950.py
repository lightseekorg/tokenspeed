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

"""Muse Spark combine, post-expert projection and boundary across 8 ranks."""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import fill, payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_tail,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, POOLED, TOP_K, OWNED, OWNED_POOL = 8, 8192, 4096, 8, 1024, 512


@gluon.jit
def moe_tail_kernel(
    scratch,
    step,
    bases,
    pen_gain,
    w_post,
    post_gain,
    alpha,
    beta,
    in_gain,
    h_next,
    a_out,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "comb1")
    if active:
        moe_tail.comb_stages(mb, task, pen_gain, RANK)
    task, active = schedule.stage_task(PLAN, "unpool")
    if active:
        moe_tail.unpool_stages(mb, task, w_post, post_gain, alpha, beta, RANK)
    task, active = schedule.stage_task(PLAN, "out")
    if active:
        # No draft tap (TAP false): the tap arguments are unused. The owned
        # residual columns (WHOLE_ROWS false); the layer test covers whole rows.
        # 16 out tasks a row and no anx publish: a launch without qnext (the
        # layer test covers qnext's 8 a row).
        moe_tail.out_stage(
            mb,
            task,
            post_gain,
            beta,
            in_gain,
            a_out,
            h_next,
            h_next,
            0,
            0,
            RANK,
            False,
            False,
            16,
            False,
            "out",
        )


def _inputs(launch: int, rows: int):
    """The route partials, route weights and ``h'`` of each row (leading
    dimension ``rows``), then the layer's weights."""
    g = torch.Generator().manual_seed(300 + launch)
    parts = (torch.randn(rows, WORLD, TOP_K, POOLED, generator=g) * 4).to(
        torch.bfloat16
    )
    w = torch.rand(rows, TOP_K, generator=g) + 0.1
    w = w / w.sum(-1, keepdim=True)
    hp = torch.randn(rows, HIDDEN, generator=g) * 2
    pen = (1.0 + 0.1 * torch.randn(POOLED, generator=g)).to(torch.bfloat16).float()
    w_post = (torch.randn(HIDDEN, POOLED, generator=g) / POOLED**0.5).to(torch.bfloat16)
    post = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16).float()
    alpha = torch.rand(HIDDEN, generator=g) + 0.5
    beta = torch.rand(HIDDEN, generator=g) + 0.5
    ing = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16).float()
    return parts, w, hp, pen, w_post, post, alpha, beta, ing


def _reference(parts, w, hp, pen, w_post, post, alpha, beta, ing):
    r = torch.zeros(TOP_K, POOLED)
    for s in range(WORLD):
        r = r + parts[s].float()
    p = r.to(torch.bfloat16).float() * pen
    inv = torch.rsqrt(p.pow(2).mean(1, keepdim=True) + 1e-8)
    combined = ((p * inv) * w[:, None]).sum(0).to(torch.bfloat16)
    y = (combined.float() @ w_post.float().T).to(torch.bfloat16).float() * post
    y = y * torch.rsqrt(y.pow(2).mean() + 1e-8)
    h2 = beta * y + alpha * hp
    scale = torch.rsqrt(h2.pow(2).mean() + 1e-5)
    a = (h2 * scale * ing).to(torch.bfloat16)
    return combined, h2, a, (beta * y).abs(), scale


def _worker(rank: int, port: int, rows: int, source_rows: int, out: str) -> None:
    """Run two launches of ``rows`` rows, row ``r`` taking the inputs of row
    ``r % source_rows``, check them against the reference, and save each
    launch's ``a_out`` and ``h_next`` to ``out``."""
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
        plan = Plan(
            grid=256,
            stages=moe_tail.stages(
                {"comb": 0, "unpool": 64, "out": 128}, rows, 16, False
            ),
            regions=(
                ("route", 2 * TOP_K),
                ("hp", OWNED),
                ("rr", WORLD * TOP_K * OWNED_POOL // 2),
            )
            + moe_tail.regions(rows),
            rows=rows,
            chain=rows,
            external=("route", "hp", "rr"),
            world=WORLD,
            peers=("rr",) + moe_tail.peers(rows),
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        own_pool = slice(rank * OWNED_POOL, (rank + 1) * OWNED_POOL)
        for launch in range(2):
            step = 20 + launch
            tag = launch_tag(step, 0)
            parts, w, hp, pen, w_post, post, alpha, beta, ing = _inputs(
                launch, source_rows
            )
            src = torch.arange(rows) % source_rows
            parts, w, hp = parts[src], w[src], hp[src]
            for row in range(rows):
                route = torch.cat(
                    [torch.zeros(TOP_K, dtype=torch.int32), w[row].view(torch.int32)]
                )
                fill(scratch, plan, "route", route.to(device), tag, row=row)
                hp_own = hp[row, own].contiguous().view(torch.int32)
                fill(scratch, plan, "hp", hp_own.to(device), tag, row=row)
                rr = parts[row, :, :, own_pool].contiguous().view(torch.int32)
                fill(
                    scratch,
                    plan,
                    "rr",
                    rr.reshape(-1).to(device),
                    tag,
                    copy=step & 1,
                    row=row,
                )
            h_next = torch.empty(rows, OWNED, device=device)
            a_out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
            moe_tail_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
                pen[own_pool].to(device),
                bf16_tiles(w_post[own].contiguous()).to(device),
                post.to(device),
                alpha[own].to(device),
                beta.to(device),
                ing.to(device),
                h_next,
                a_out,
                plan.key,
                rank,
                num_warps=8,
                **extern_compile_options(),
            )
            torch.cuda.synchronize()
            dist.barrier()
            for row in range(rows):
                combined, h2, a, gated, scale = _reference(
                    parts[row], w[row], hp[row], pen, w_post, post, alpha, beta, ing
                )
                cmb = payloads(
                    scratch, plan, "cmb", POOLED // 2, tag, row=row, copy=step & 1
                )
                torch.testing.assert_close(
                    cmb.view(torch.bfloat16).float().cpu(),
                    combined.float(),
                    rtol=2**-7,
                    atol=1e-3,
                )
                # The projection output is BF16: a sum next to a rounding
                # boundary can round to the neighbour, one BF16 step of beta y
                # (up to 2^-7 of it, at the bottom of a binade).
                err = (h_next[row].cpu() - h2[own]).abs()
                bound = 2**-7 * h2[own].abs() + 2**-7 * gated[own] + 1e-4
                assert torch.all(err <= bound), row
                # Four BF16 roundings lie between the route partials and this
                # output: one step in an early one can make two here. A few
                # elements may also carry the projection's one-step flip
                # (bounded as for h_next, through the norm) plus one of a's.
                err = (a_out[row].float().cpu() - a.float()).abs()
                tight = 2e-3 + 2**-6 * a.float().abs()
                flip = 2**-7 * (scale * ing * gated + a.float().abs())
                assert torch.all(err <= tight + flip), row
                assert int((err > tight).sum()) <= 4, row
            torch.save(
                (a_out.cpu(), h_next.cpu()), f"{out}/rank{rank}-launch{launch}.pt"
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(rows: int, source_rows: int, out) -> None:
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows, source_rows, str(out)), nprocs=WORLD, join=True)


@pytest.mark.parametrize("rows", [1, 8, 16])
def test_moe_tail_combines_projects_and_closes_the_layer(rows: int, tmp_path):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    _spawn(rows, rows, tmp_path)


def test_ffn_words_past_8_rows_match_the_tagged_pairs(tmp_path):
    """16 rows hand ffn to out as words and flags, 8 rows as tagged pairs:
    two requests of the same 8 rows give each row the 8-row launch's bits."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    assert moe_tail.ffn_words(16) and not moe_tail.ffn_words(8)
    (tmp_path / "pairs").mkdir()
    (tmp_path / "words").mkdir()
    _spawn(8, 8, tmp_path / "pairs")
    _spawn(16, 8, tmp_path / "words")
    for rank in range(WORLD):
        for launch in range(2):
            name = f"rank{rank}-launch{launch}.pt"
            pairs = torch.load(tmp_path / "pairs" / name)
            words = torch.load(tmp_path / "words" / name)
            for got, want in zip(words, pairs):
                assert torch.equal(got, torch.cat([want, want])), (rank, launch)
