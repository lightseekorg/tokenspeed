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

"""The Inkling2 projection stages on slices of the local-layer plan, against
torch and bit for bit against the regular path's kernels, and the attention
output exchange across four ranks."""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monokernel_testing import fill, fill_words, payloads, words
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel.ops.communication.triton import (  # noqa: E402
    rank_ordered_fold,
)
from tokenspeed_kernel.ops.conv import inkling_ring_sconv  # noqa: E402
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.fp32_output import (  # noqa: E402
    gluon_mm_bf16_fp32_num_counters_gfx950,
    launch_gluon_mm_bf16_fp32_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    conv,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    plan as ip,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    projection,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

NUM_WARPS = 8
STEP = 3
ROWS = [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)]
QKVR_SPLITS = gl.constexpr(ip.QKVR_SPLITS)
ROUTER_SPLITS = gl.constexpr(ip.ROUTER_SPLITS)
WO_SPLITS = gl.constexpr(ip.WO_SPLITS)
WO_SLOTS = gl.constexpr(ip.WO_SLOTS)


@gluon.jit
def _qkvr_kernel(scratch, step, w, PLAN: gl.constexpr, K: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    projection.chained_stages(
        PLAN, mb, w, "x_attn", "qkvr", "qkvr", K, QKVR_SPLITS, False
    )


@gluon.jit
def _router_kernel(scratch, step, w, PLAN: gl.constexpr, K: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    projection.chained_stages(
        PLAN, mb, w, "x_mlp", "router", "logits", K, ROUTER_SPLITS, True
    )


def _projection_plan(rows, chain, name):
    """The plan slice of projection ``name``: its chained K split groups at
    ``rows`` rows (``plan.projection_shape``)."""
    plan = ip.local_layer_plan(rows, chain, "nvfp4", False, False)
    names = [s.name for s in plan.stages if s.name.split("_k")[0] == name]
    return plan.only(tuple(names))


def _qkvr_tiles(w):
    """The packed qkvr weight: ``QKVR_PADDED`` rows, the last zero."""
    return bf16_tiles(torch.nn.functional.pad(w, (0, 0, 0, ip.QKVR_PADDED - ip.QKVR)))


def _bf16_words(x: torch.Tensor) -> torch.Tensor:
    """BF16 ``[..., 2n]`` as int32 pair words ``[..., n]``, element 0 low."""
    return x.contiguous().view(torch.int32)


def _publish_rows(scratch, plan, region, x, tag):
    for r in range(x.shape[0]):
        fill(scratch, plan, region, _bf16_words(x[r]), tag, row=r)


def _rows_of(scratch, plan, region, n, tag, rows, dtype=torch.bfloat16):
    count = n // 2 if dtype == torch.bfloat16 else n
    got = [payloads(scratch, plan, region, count, tag, row=r) for r in range(rows)]
    return torch.stack(got).view(dtype)


def _reference(x, w):
    """FP32 GEMV of BF16 operands (exact products, FP64 sums)."""
    return (x.double() @ w.double().T).float()


def _regular_projection(x, w):
    """The regular path's FP32 projection of BF16 ``x`` by ``w`` (its
    ``gluon_mm_bf16_fp32_gfx950`` launch, split-K as served)."""
    m, n, k = x.shape[0], w.shape[0], w.shape[1]
    counters = torch.zeros(
        gluon_mm_bf16_fp32_num_counters_gfx950(m, n, k),
        dtype=torch.int32,
        device=x.device,
    )
    return launch_gluon_mm_bf16_fp32_gfx950(x, w, None, counters=counters)


def _assert_bf16_close(got, want):
    # The FP32 sums run in another order than the reference's: allow one BF16
    # rounding step of the FP32 result.
    torch.testing.assert_close(got.float(), want, rtol=2**-7, atol=1e-6)


@pytest.mark.parametrize("rows,chain", ROWS)
def test_qkvr_projection_matches_torch(rows, chain):
    g = torch.Generator().manual_seed(rows)
    plan = _projection_plan(rows, chain, "qkvr")
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    x = torch.randn(rows, ip.HIDDEN, generator=g).bfloat16().cuda()
    w = (torch.randn(ip.QKVR, ip.HIDDEN, generator=g) / 64).bfloat16().cuda()
    _publish_rows(scratch, plan, "x_attn", x, tag)
    _qkvr_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        _qkvr_tiles(w),
        plan.key,
        ip.HIDDEN,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    got = _rows_of(scratch, plan, "qkvr", ip.QKVR, tag, rows)
    _assert_bf16_close(got, _reference(x, w))
    # The regular path rounds its FP32 GEMM to BF16.
    assert torch.equal(got, _regular_projection(x, w).bfloat16())


@pytest.mark.parametrize("rows,chain", ROWS)
def test_router_logits_match_torch(rows, chain):
    g = torch.Generator().manual_seed(10 + rows)
    plan = _projection_plan(rows, chain, "router")
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    x = torch.randn(rows, ip.HIDDEN, generator=g).bfloat16().cuda()
    w = (torch.randn(ip.ROUTED, ip.HIDDEN, generator=g) / 64).bfloat16().cuda()
    _publish_rows(scratch, plan, "x_mlp", x, tag)
    _router_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        bf16_tiles(w),
        plan.key,
        ip.HIDDEN,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    got = _rows_of(scratch, plan, "logits", ip.ROUTED, tag, rows, torch.float32)
    torch.testing.assert_close(got, _reference(x, w), rtol=1e-5, atol=1e-5)
    # The regular path's logits stay FP32 into the top-k.
    assert torch.equal(got, _regular_projection(x, w))


# The attention output exchange on four ranks: ``wo``, then ``red_attn``
# (the reduce with the attention output conv).
WORLD = ip.WORLD
O_DIM = ip.Q_HEADS * ip.HEAD_DIM  # this rank's gated attention output
LAUNCHES = 3
SLOTS = 12  # conv ring slots: past the requests of the largest launch
ATTN_CHANNEL0 = 512  # the ATTN stream's channels in the conv ring
FIRST_POSITION = 1  # of the first launch's first row: older taps start at 0


@gluon.jit
def _exchange_kernel(
    scratch,
    step,
    bases,
    w,
    term,
    taps,
    ring,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    K: gl.constexpr,
    N: gl.constexpr,
):
    W: gl.constexpr = schedule.world(PLAN)
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, W)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "wo")
    if active:
        projection.partial_stage(
            mb,
            task,
            w,
            "o",
            "o_flag",
            2,
            None,
            "wo_part",
            "wo_flag",
            K,
            N,
            2,
            WO_SPLITS,
            0,
            WO_SPLITS,
            WO_SLOTS,
            RANK,
            projection.wo_scatter(PLAN),
            "wo",
        )
    if schedule.has_stage(PLAN, "fold_attn"):
        task, active = schedule.stage_task(PLAN, "fold_attn")
        if active:
            projection.scatter_fold_stage(
                mb,
                task,
                "wo_part",
                "wo_flag",
                "wo_gather",
                "wo_gather_flag",
                N,
                32,
                RANK,
                "fold_attn",
            )
    task, active = schedule.stage_task(PLAN, "red_attn")
    if active:
        conv.reduce_conv_stage(
            mb,
            task,
            "wo_gather" if projection.wo_scatter(PLAN) else "wo_part",
            "wo_gather_flag" if projection.wo_scatter(PLAN) else "wo_flag",
            projection.wo_scatter(PLAN),
            "attn_sc",
            term,
            taps,
            conv.Ring(ring[0], ring[1], ring[2], schedule.chain(PLAN)),
            512,
            N,
            32,
            "red_attn",
        )


def _exchange_inputs(rank, launch, rows):
    g = torch.Generator().manual_seed(100 * launch + rank)
    o = torch.randn(rows, O_DIM, generator=g).bfloat16()
    w = (torch.randn(ip.HIDDEN, O_DIM, generator=g) / 32).bfloat16()
    return o, w


def _conv_reference(a, taps, ring, slots, positions, chain):
    """TokenSpeed's ring sconv of ``a [rows, H]`` (FP32 of BF16) on the ATTN
    stream: the FMA chain in FP64, rounded to FP32 at each step."""
    rows = a.shape[0]
    channels = slice(ATTN_CHANNEL0, ATTN_CHANNEL0 + ip.HIDDEN)
    w = taps.float()
    out = torch.empty_like(a)
    for r in range(rows):
        slot, p, tok = slots[r // chain], positions[r], r % chain
        acc = torch.zeros(a.shape[1])
        for i in range(4):
            back = 3 - i
            if back <= tok:
                x = a[r - back]
            elif slot >= 0 and p - back >= 0:
                x = ring[slot, (p - back) % 11, channels].float()
            else:
                x = torch.zeros(a.shape[1])
            acc = (x.double() * w[:, i].double() + acc.double()).float()
        out[r] = ((acc + a[r]) * 0.4472135954999579).bfloat16().float()
    return out


def _regular_reduce_conv(parts, taps, ring, slots, positions, chain):
    """The regular path's attention TP all-reduce (the all-gathered BF16
    partials ``parts [world, rows, H]`` through ``rank_ordered_fold``) and
    the ATTN stream's decode sconv (``inkling_ring_sconv``), which updates
    ``ring`` in place. Returns ``(folded, conv)``, BF16 ``[rows, H]``."""
    rows = parts.shape[1]
    device = parts.device
    folded = torch.empty(rows, ip.HIDDEN, dtype=torch.bfloat16, device=device)
    rank_ordered_fold(parts, folded)
    requests = rows // chain
    starts = [positions[q * chain] for q in range(requests)]
    conv = inkling_ring_sconv(
        folded,
        taps,
        ring[:, :, ATTN_CHANNEL0 : ATTN_CHANNEL0 + ip.HIDDEN],
        torch.arange(0, rows + 1, chain, dtype=torch.int32, device=device),
        torch.arange(rows, dtype=torch.int32, device=device) // chain,
        torch.tensor(slots, dtype=torch.int32, device=device),
        torch.ones(requests, dtype=torch.bool, device=device),
        torch.tensor([p + chain for p in starts], dtype=torch.int32, device=device),
        torch.zeros(requests, 2, dtype=torch.int32, device=device),
        torch.zeros(1, 3, ip.HIDDEN, dtype=torch.bfloat16, device=device),
        None,
        num_extends=0,
        publish_live=False,
        dynamic_weights=None,
        output_scale=5**-0.5,
        token_range=None,
    )
    return folded, conv


def _exchange_worker(rank, port, rows, chain):
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
        stages = (
            ("wo", "fold_attn", "red_attn")
            if ip.WO_SCATTER[rows]
            else ("wo", "red_attn")
        )
        plan = ip.local_layer_plan(rows, chain, "nvfp4", False, False).only(stages)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        g = torch.Generator().manual_seed(7)  # the same conv state on every rank
        taps = (torch.randn(ip.HIDDEN, 4, generator=g) / 2).bfloat16()
        ring = torch.randn(SLOTS, 11, 9728, generator=g).bfloat16()
        requests = rows // chain
        # Each request its own ring slot; the last of several is a padded
        # request (slot -1: zero taps, no ring writes).
        slots = [2 + q for q in range(requests)]
        if requests > 1:
            slots[-1] = -1
        ring_gpu = ring.to(device)
        regular_ring = ring.to(device)
        for k in range(LAUNCHES):
            step = STEP + k
            tag = launch_tag(step, 0)
            o, w = _exchange_inputs(rank, k, rows)
            positions = [
                FIRST_POSITION + 3 * (r // chain) + k * chain + r % chain
                for r in range(rows)
            ]
            # The merge's plain words and every row's two KV-head flags.
            for r in range(rows):
                fill_words(scratch, plan, "o", _bf16_words(o[r].to(device)), row=r)
                flags = torch.ones(2, dtype=torch.int32, device=device)
                fill(scratch, plan, "o_flag", flags, tag, row=r)
            term = torch.zeros(rows, ip.HIDDEN, dtype=torch.bfloat16, device=device)
            ring_before = ring_gpu.cpu()
            # Every rank's inputs are in place before any rank's launch.
            dist.barrier()
            _exchange_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
                bf16_tiles(w.to(device)),
                term,
                taps.to(device),
                (
                    ring_gpu,
                    torch.tensor(slots, dtype=torch.int32, device=device),
                    torch.tensor(positions, dtype=torch.int64, device=device),
                ),
                plan.key,
                rank,
                O_DIM,
                ip.HIDDEN,
                num_warps=NUM_WARPS,
                **extern_compile_options(),
            )
            torch.cuda.synchronize()
            got = _rows_of(scratch, plan, "attn_sc", ip.HIDDEN, tag, rows).cpu()
            copy = step & 1
            # Scattered, this rank holds every rank's partial of the 32-column
            # blocks it owns (block % WORLD == rank), and the owners' folded
            # rows; else every rank's whole partial.
            owned = torch.ones(ip.HIDDEN, dtype=torch.bool)
            if ip.WO_SCATTER[rows]:
                owned[:] = False
                for b in range(rank, ip.HIDDEN // 32, WORLD):
                    owned[32 * b : 32 * (b + 1)] = True
            parts = []
            for source in range(WORLD):
                part = (
                    torch.stack(
                        [
                            words(
                                scratch,
                                plan,
                                "wo_part",
                                ip.HIDDEN // 2 * (source + 1),
                                row=r,
                                copy=copy,
                            )[source * ip.HIDDEN // 2 :]
                            for r in range(rows)
                        ]
                    )
                    .view(torch.bfloat16)
                    .cpu()
                )
                so, sw = _exchange_inputs(source, k, rows)
                regular = _regular_projection(so.to(device), sw.to(device))
                regular = regular.bfloat16().cpu()
                # wo's partials are bit for bit each rank's regular projection.
                assert torch.equal(part[:, owned], regular[:, owned]), source
                _assert_bf16_close(regular, _reference(so, sw))
                parts.append(regular.float())
            # The fold is the rank-ordered FP32 sum of the partials, bit for bit.
            total = parts[0]
            for part in parts[1:]:
                total = total + part
            folded = total.bfloat16().float()
            want = _conv_reference(folded, taps, ring_before, slots, positions, chain)
            torch.testing.assert_close(got.float(), want, rtol=2**-7, atol=1e-6)
            # red_attn is bit for bit the regular path's all-reduce (the
            # ordered fold of the all-gathered partials) and decode sconv.
            regular_folded, regular_conv = _regular_reduce_conv(
                torch.stack([part.bfloat16() for part in parts]).to(device),
                taps.to(device),
                regular_ring,
                slots,
                positions,
                chain,
            )
            assert torch.equal(folded.bfloat16(), regular_folded.cpu())
            assert torch.equal(got, regular_conv.cpu()), f"launch {k}"
            assert torch.equal(ring_gpu, regular_ring), f"launch {k}"
            if ip.WO_SCATTER[rows]:
                # Every rank gathered the owners' folded rows.
                gathered = torch.stack(
                    [
                        words(
                            scratch, plan, "wo_gather", ip.HIDDEN // 2, row=r, copy=copy
                        )
                        for r in range(rows)
                    ]
                )
                assert torch.equal(
                    gathered.view(torch.bfloat16).cpu(), regular_folded.cpu()
                ), f"launch {k}"
            assert torch.equal(term.cpu(), got), f"launch {k}"
            # Each row wrote its raw reduced input to its ring row.
            for r in range(rows):
                if slots[r // chain] < 0:
                    continue
                ring_row = ring_gpu[slots[r // chain], positions[r] % 11].cpu()
                channels = slice(ATTN_CHANNEL0, ATTN_CHANNEL0 + ip.HIDDEN)
                assert torch.equal(ring_row[channels], folded[r].bfloat16())
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rows,chain", ROWS)
def test_attention_output_exchange_folds_ranks_in_order(rows, chain):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_exchange_worker, args=(port, rows, chain), nprocs=WORLD, join=True)
