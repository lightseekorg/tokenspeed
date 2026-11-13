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

"""Muse Spark output projection and attention tail across the 8 ranks."""

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

import muse_spark_standins as standins  # noqa: E402
from monokernel_testing import fill, payloads, words  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    attn_tail,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, Q, OWNED = 8, 8192, 1024, 1024
LAUNCHES = 2
ALL = tuple(range(WORLD))


@gluon.jit
def tail_kernel(
    scratch,
    step,
    bases,
    w,
    h,
    alpha,
    beta,
    pre_gain,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "oproj")
    if active:
        attn_tail.oproj_stage(mb, task, w, RANK, "oproj")
    task, active = schedule.stage_task(PLAN, "tail1")
    if active:
        attn_tail.tail_stages(mb, task, h, alpha, beta, pre_gain, RANK)


def _plan(rows: int) -> Plan:
    return Plan(
        grid=256,
        stages=attn_tail.stages({"oproj": 0, "tail": 128}, rows),
        regions=(("z", Q // 2),) + attn_tail.regions(),
        rows=rows,
        chain=rows,
        external=("z",),
        world=WORLD,
        peers=attn_tail.PEERS,
        flags=attn_tail.FLAGS,
    )


def _rank_inputs(rank: int, launch: int, rows: int, present):
    """``rank``'s attention output ``z`` and o_proj shard ``w``. Ranks outside
    ``present`` have a zero shard, and every shard's rows into the columns
    they own are zero (a zero-shard model: those ranks put exactly zero)."""
    g = torch.Generator().manual_seed(100 * launch + rank)
    z = torch.randn(rows, Q, generator=g).to(torch.bfloat16)
    w = (torch.randn(HIDDEN, Q, generator=g) / Q**0.5).to(torch.bfloat16)
    if rank not in present:
        w = torch.zeros_like(w)
    w[standins.absent_columns(HIDDEN, OWNED, present)] = 0
    return z, w


def _shared_inputs(launch: int, rows: int, present):
    """``h`` (zero in the columns of ranks outside ``present``), ``alpha``,
    ``beta`` and the pre-expert gain."""
    g = torch.Generator().manual_seed(9000 + launch)
    h = torch.randn(rows, HIDDEN, generator=g) * 3
    h[:, standins.absent_columns(HIDDEN, OWNED, present)] = 0
    alpha = torch.rand(HIDDEN, generator=g) + 0.5
    beta = torch.rand(HIDDEN, generator=g) + 0.5
    pre_gain = (torch.randn(HIDDEN, generator=g) * 0.1 + 1.0).to(torch.bfloat16).float()
    return h, alpha, beta, pre_gain


def _reference_b(launch: int, rows: int, present):
    """The reduced, BF16-rounded ``o_proj`` output ``b``, and the sum of the
    ranks' partials' magnitudes, both ``[rows, 8192]``."""
    b = torch.zeros(rows, HIDDEN)
    spread = torch.zeros(rows, HIDDEN)
    for r in range(WORLD):
        z, w = _rank_inputs(r, launch, rows, present)
        part = (z.float() @ w.float().T).to(torch.bfloat16).float()
        b = b + part
        spread = spread + part.abs()
    return b.to(torch.bfloat16).float(), spread


def _rank_order_sum(red: torch.Tensor) -> torch.Tensor:
    """FP32 of the BF16-rounded sum, in rank order, of the 8 ranks' BF16
    partials of the owned columns (``red``'s int32 pairs, source-rank major)."""
    parts = red.view(torch.bfloat16).float().view(WORLD, OWNED)
    total = torch.zeros(OWNED)
    for p in range(WORLD):
        total = total + parts[p]
    return total.to(torch.bfloat16).float()


def _reference_tail(launch: int, b: torch.Tensor, present):
    """``(h', u)`` of ``b [rows, 8192]`` in FP64, and the size of each one's
    terms: ``|beta * b'| + |alpha * h|`` for h' (where the two cancel, FP32
    rounding is an absolute error at that size), carried through the norm
    for u."""
    h, alpha, beta, pre_gain = (
        x.double() for x in _shared_inputs(launch, b.shape[0], present)
    )
    b = b.double()
    inv = torch.rsqrt(b.pow(2).mean(-1, keepdim=True) + 1e-8)
    hp = beta * b * inv + alpha * h
    hp_size = (beta * b * inv).abs() + (alpha * h).abs()
    inv_hp = torch.rsqrt(hp.pow(2).mean(-1, keepdim=True) + 1e-5)
    return hp, hp * inv_hp * pre_gain, hp_size, hp_size * inv_hp * pre_gain.abs()


def _worker(me: int, port: int, rows: int, present) -> None:
    """``present``: None for the whole TP8 group, one process a rank; or the
    two rank programs that run (process ``me`` runs ``present[me]``) in a
    zero-shard model whose other ranks' puts (exactly zero) the host
    writes before each launch."""
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
        plan = _plan(rows)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        if present is None:
            rank, model, bases = me, ALL, group.bases
        else:
            rank, model = present[me], present
            ghost = torch.zeros_like(scratch)
            bases = standins.peer_bases(group.bases, present, me, scratch, ghost)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        # Each launch has new inputs and a new step (the peer regions'
        # copies alternate). z is this launch's input, filled by the host.
        for k in range(LAUNCHES):
            step = 7 + k
            z, w = _rank_inputs(rank, k, rows, model)
            h, alpha, beta, pre_gain = _shared_inputs(k, rows, model)
            if present is not None:
                standins.standin_fill(
                    scratch,
                    plan,
                    present,
                    [(0, t) for t in range(rows)],
                    step,
                    0,
                    step & 1,
                    True,
                )
            for row in range(rows):
                fill(
                    scratch,
                    plan,
                    "z",
                    z[row].view(torch.int32).to(device),
                    launch_tag(step, 0),
                    row=row,
                )
            tail_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                bases,
                bf16_tiles(w).to(device),
                h[:, own].contiguous().to(device),
                alpha[own].to(device),
                beta[own].to(device),
                pre_gain[own].to(device),
                plan.key,
                rank,
                num_warps=8,
                **extern_compile_options(),
            )
            torch.cuda.synchronize()
            tag = launch_tag(step, 0)
            # b, the reduced, BF16-rounded o_proj columns: the 8 ranks'
            # BF16 partials this rank received, summed in rank order as tail1
            # (and the router) sum them. A rank's BF16 partial can be one step
            # (2^-7 of its size) off the reference's, as its FP32 sums run in
            # another order; the sum's own rounding adds one more step.
            b = torch.stack(
                [
                    _rank_order_sum(
                        payloads(
                            scratch,
                            plan,
                            "red",
                            WORLD * OWNED // 2,
                            tag,
                            row=row,
                            copy=step & 1,
                        ).cpu()
                    )
                    for row in range(rows)
                ]
            )
            want_b, spread = _reference_b(k, rows, model)
            err = (b - want_b[:, own]).abs()
            bound = 2**-7 * (spread[:, own] + want_b[:, own].abs())
            assert torch.all(err <= bound), float((err / bound).max())
            # h' and u follow from the kernel's b: the residual gate can
            # cancel, and would amplify b's rounding steps. h' is FP32: a few
            # FP32 roundings and the norm's reordered sum, well within 2^-16
            # of its terms' size.
            parts = [torch.empty_like(b) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, b)
            whole_b = torch.zeros(rows, HIDDEN)  # the absent ranks' b is zero
            for r, part in zip(ALL if present is None else present, parts):
                whole_b[:, r * OWNED : (r + 1) * OWNED] = part
            want_hp, want_u, hp_size, u_size = _reference_tail(k, whole_b, model)
            for row in range(rows):
                hp = payloads(scratch, plan, "hp", OWNED, tag, row=row)
                err = (hp.view(torch.float32).cpu().double() - want_hp[row, own]).abs()
                assert torch.all(err <= 2**-16 * hp_size[row, own]), float(err.max())
                # Every rank holds the whole BF16 MoE input, rank-major, as
                # words, and every tail task's flag.
                payloads(
                    scratch,
                    plan,
                    "xflag",
                    WORLD * attn_tail.TAIL_TASKS,
                    tag,
                    row=row,
                    copy=step & 1,
                )
                xin = words(scratch, plan, "xin", HIDDEN // 2, row=row, copy=step & 1)
                # u rounds once to BF16 from h''s FP32 error: one step.
                err = (xin.view(torch.bfloat16).cpu().double() - want_u[row]).abs()
                bound = 2**-7 * want_u[row].abs() + 2**-16 * u_size[row]
                assert torch.all(err <= bound), float((err / bound).max())
                if present is not None:
                    # The absent ranks' columns of the MoE input stay zero.
                    absent = standins.absent_columns(HIDDEN, OWNED, present)
                    assert not xin.view(torch.bfloat16).cpu()[absent].float().any()
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rows", [1, 8])
def test_attention_tail_reduces_scatters_and_normalizes_across_ranks(rows: int):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows, None), nprocs=WORLD, join=True)


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
@pytest.mark.parametrize("rows", [1, 8])
def test_attention_tail_reduces_scatters_and_normalizes_on_two_gpus(rows: int, present):
    """As :func:`test_attention_tail_reduces_scatters_and_normalizes_across_ranks`
    with the production rank programs ``present`` on two GPUs, in a
    zero-shard model whose six other ranks put exactly zero (the host
    writes it): b, h' and the MoE input against the reference, and the
    absent ranks' columns stay zero. Two non-zero terms per sum cannot
    check the rank order of an eight-rank sum (see muse_spark_standins)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows, present), nprocs=2, join=True)
