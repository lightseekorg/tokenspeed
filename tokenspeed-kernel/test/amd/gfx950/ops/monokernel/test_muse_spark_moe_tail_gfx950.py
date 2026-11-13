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

import math
import socket
from datetime import timedelta

import muse_spark_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import muse_spark_standins as standins  # noqa: E402
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
ALL = tuple(range(WORLD))


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


def _inputs(launch: int, rows: int, present):
    """The route partials, route weights and ``h'`` of each row (leading
    dimension ``rows``), then the layer's weights. Outside ``present`` the
    model is zero-shard: those ranks' route partials, the pooled and hidden
    columns they own (partials, h', post-projection rows) are zero, so they
    put exactly zero."""
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
    absent_rank = ~torch.isin(torch.arange(WORLD), torch.tensor(present))
    parts[:, absent_rank] = 0
    parts[..., standins.absent_columns(POOLED, OWNED_POOL, present)] = 0
    hp[:, standins.absent_columns(HIDDEN, OWNED, present)] = 0
    w_post[standins.absent_columns(HIDDEN, OWNED, present)] = 0
    return parts, w, hp, pen, w_post, post, alpha, beta, ing


def _reference(parts, w, pen):
    """The combined row of one row's route partials in FP64, and a bound on
    the kernel's BF16 row's difference from it."""
    r = torch.zeros(TOP_K, POOLED)
    for s in range(WORLD):
        r = r + parts[s].float()
    # The combined row in FP64 from the same BF16 route sums (the kernel sums
    # the ranks in the same order), and a bound on the kernel's FP32 steps:
    # p's product, the norm's 4096-term sum of squares (6 standard deviations
    # of its roundings: ~2^-18.4 relative), rsqrt and the 8-term weighted sum
    # stay under 2^-17 of the terms' size; twice that, then one BF16 rounding.
    p64 = r.to(torch.bfloat16).double() * pen.double()
    terms = p64 * torch.rsqrt(p64.pow(2).mean(1, keepdim=True) + 1e-8)
    terms = terms * w[:, None].double()
    exact = terms.sum(0)
    sums = 2**-16 * terms.abs().sum(0)
    return exact, sums + 2**-8 * (exact.abs() + sums)


# Independent per-element errors move a norm's mean square by a sum of
# independent terms, each at most its bound (a standard deviation of at most
# bound / sqrt(3)): allow 6 standard deviations.
_SIGMAS = 6 / math.sqrt(3)


def _rms_error(x, dx):
    """The relative error of ``rms(x)`` from independent element errors of at
    most ``dx``, plus 2^-16 for the FP32 sum of squares (8192 terms: ~2^-17.6
    at 6 standard deviations) and rsqrt."""
    ms = x.pow(2).mean()
    return _SIGMAS * (x * dx).pow(2).sum().sqrt() / (x.numel() * ms) + 2**-16


def _downstream(cmb, hp, w_post, post, alpha, beta, ing):
    """From the kernel's BF16 combined row ``cmb``: ``h''`` (FP32 in the
    kernel) and ``a`` (BF16), each in FP64 with a bound on the kernel's
    difference.

    The post-expert projection is a BF16 GEMM (``ref.bf16_gemm``), whose
    errors are independent per element. Its norm's scale error is shared by
    every element of ``beta * y``, so it is carried as one signed term through
    the next norm. ``h''``'s two terms add a few FP32 roundings (2^-22 of
    their size); ``a`` rounds once to BF16.
    """
    y, dy = ref.bf16_gemm(cmb[None], w_post)
    x, dx = y[0] * post.double(), dy[0] * post.double().abs()
    inv = torch.rsqrt(x.pow(2).mean() + 1e-8)
    shared = _rms_error(x, dx)  # y's error: dx * inv, and +-shared * y
    y = x * inv
    h2 = beta.double() * y + alpha.double() * hp.double()
    v = beta.double() * y  # the shared error's direction in h''
    own = beta.double() * dx * inv + 2**-22 * (
        v.abs() + (alpha.double() * hp.double()).abs()
    )
    dh2 = own + shared * v.abs()
    # a = h'' / rms(h'') * ing: the independent part, then the shared one
    # (``v - a' * sum(h'' v) / (n rms)``, exact for a signed scale error).
    rms = torch.sqrt(h2.pow(2).mean() + 1e-5)
    a = h2 / rms
    common = (v - a * (h2 * v).sum() / (h2.numel() * rms)).abs() / rms
    da = own / rms + a.abs() * _rms_error(h2, own) + shared * common
    a, da = a * ing.double(), da * ing.double().abs()
    return h2, dh2, a, da + 2**-8 * (a.abs() + da)


def _worker(me: int, port: int, rows: int, source_rows: int, out: str, present) -> None:
    """Run two launches of ``rows`` rows, row ``r`` taking the inputs of row
    ``r % source_rows``, check them against the reference, and save each
    launch's ``a_out`` and ``h_next`` to ``out``. ``present``: None for the
    whole TP8 group; or the two rank programs that run (process ``me`` runs
    ``present[me]``) in the zero-shard model of :func:`_inputs`, whose other
    ranks' puts (exactly zero) the host writes before each launch."""
    torch.cuda.set_device(me)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=me,
        world_size=WORLD if present is None else len(present),
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{me}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, me, device)
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
            flags=moe_tail.flags(rows),
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        if present is None:
            rank, model, bases = me, ALL, group.bases
        else:
            rank, model = present[me], present
            ghost = torch.zeros_like(scratch)
            bases = standins.peer_bases(group.bases, present, me, scratch, ghost)
        group.finish_prepare()
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        own_pool = slice(rank * OWNED_POOL, (rank + 1) * OWNED_POOL)
        for launch in range(2):
            step = 20 + launch
            tag = launch_tag(step, 0)
            parts, w, hp, pen, w_post, post, alpha, beta, ing = _inputs(
                launch, source_rows, model
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
            if present is not None:
                standins.standin_fill(
                    scratch,
                    plan,
                    present,
                    [(0, t % source_rows) for t in range(rows)],
                    step,
                    0,
                    step & 1,
                    True,
                )
            h_next = torch.empty(rows, OWNED, device=device)
            a_out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
            moe_tail_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                bases,
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
                cmb_exact, cmb_bound = _reference(parts[row], w[row], pen)
                cmb = payloads(
                    scratch, plan, "cmb", POOLED // 2, tag, row=row, copy=step & 1
                )
                cmb = cmb.view(torch.bfloat16).cpu()
                err = (cmb.double() - cmb_exact).abs()
                assert torch.all(err <= cmb_bound), float((err / cmb_bound).max())
                # h'' and a follow from the kernel's combined row, so that a
                # BF16 step there is not carried through the projection.
                h2, dh2, a, da = _downstream(
                    cmb, hp[row], w_post, post, alpha, beta, ing
                )
                err = (h_next[row].cpu().double() - h2[own]).abs()
                assert torch.all(err <= dh2[own]), float((err / dh2[own]).max())
                err = (a_out[row].cpu().double() - a).abs()
                assert torch.all(err <= da), float((err / da).max())
                if present is not None:
                    # The absent ranks' columns stay zero.
                    absent = standins.absent_columns(HIDDEN, OWNED, present)
                    assert not a_out[row].cpu()[absent].float().any()
                    absent = standins.absent_columns(POOLED, OWNED_POOL, present)
                    assert not cmb[absent].float().any()
            torch.save(
                (a_out.cpu(), h_next.cpu()), f"{out}/rank{rank}-launch{launch}.pt"
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(rows: int, source_rows: int, out, present) -> None:
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _worker,
        args=(port, rows, source_rows, str(out), present),
        nprocs=WORLD if present is None else len(present),
        join=True,
    )


@pytest.mark.parametrize("rows", [1, 8, 16])
def test_moe_tail_combines_projects_and_closes_the_layer(rows: int, tmp_path):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    _spawn(rows, rows, tmp_path, None)


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
@pytest.mark.parametrize("rows", [1, 8, 16])
def test_moe_tail_combines_projects_and_closes_the_layer_on_two_gpus(
    rows: int, present, tmp_path
):
    """As :func:`test_moe_tail_combines_projects_and_closes_the_layer` with
    the production rank programs ``present`` on two GPUs, in a zero-shard
    model whose six other ranks put exactly zero (the host writes it): the
    combined row, h'' and the next input against the reference, and the
    absent ranks' columns stay zero. Two non-zero terms per sum cannot check
    the rank order of an eight-rank sum (see muse_spark_standins)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    _spawn(rows, rows, tmp_path, present)


def test_ffn_words_past_8_rows_match_the_tagged_pairs(tmp_path):
    """16 rows hand ffn to out as words and flags, 8 rows as tagged pairs:
    two requests of the same 8 rows give each row the 8-row launch's bits."""
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    assert moe_tail.ffn_words(16) and not moe_tail.ffn_words(8)
    (tmp_path / "pairs").mkdir()
    (tmp_path / "words").mkdir()
    _spawn(8, 8, tmp_path / "pairs", None)
    _spawn(16, 8, tmp_path / "words", None)
    for rank in range(WORLD):
        for launch in range(2):
            name = f"rank{rank}-launch{launch}.pt"
            pairs = torch.load(tmp_path / "pairs" / name)
            words = torch.load(tmp_path / "words" / name)
            for got, want in zip(words, pairs):
                assert torch.equal(got, torch.cat([want, want])), (rank, launch)


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
def test_ffn_words_past_8_rows_match_the_tagged_pairs_on_two_gpus(present, tmp_path):
    """As :func:`test_ffn_words_past_8_rows_match_the_tagged_pairs` with the
    production rank programs ``present`` on two GPUs (zero-shard model):
    two requests of the same 8 rows give each row the 8-row launch's bits."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    (tmp_path / "pairs").mkdir()
    (tmp_path / "words").mkdir()
    _spawn(8, 8, tmp_path / "pairs", present)
    _spawn(16, 8, tmp_path / "words", present)
    for rank in present:
        for launch in range(2):
            name = f"rank{rank}-launch{launch}.pt"
            pairs = torch.load(tmp_path / "pairs" / name)
            words = torch.load(tmp_path / "words" / name)
            for got, want in zip(words, pairs):
                assert torch.equal(got, torch.cat([want, want])), (rank, launch)
