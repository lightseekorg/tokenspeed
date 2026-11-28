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

"""The Inkling2 NVFP4 expert stages (w13, capped SwiGLU with the route weight
and its quantization, w2) of one rank against a torch reference of
TokenSpeed's block-scaled MoE routes, at each geometry."""

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

from monokernel_testing import fill, fill_words, words  # noqa: E402
from test_inkling2_attention_gfx950 import _nvfp4_reference  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel.ops.communication.triton import (  # noqa: E402
    rank_ordered_fold,
)
from tokenspeed_kernel.ops.conv import inkling_ring_sconv  # noqa: E402
from tokenspeed_kernel.ops.moe._triton.block_scaled import (  # noqa: E402
    _fold_routes,
    _quantized_buffers,
    _swiglu_quantize_routes,
)
from tokenspeed_kernel.ops.moe._triton.common import _routing  # noqa: E402
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.block_scaled import (  # noqa: E402
    gluon_block_scaled_moe_routes_mm_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    conv,
    experts,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    plan as ip,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag  # noqa: E402

NUM_WARPS = 8
STEP = 11
# (geometry, rows, chain): Inkling 2 Small's verify blocks of 8 tokens, the
# half-width geometry's of 4. The routes below (_route_ids) give the NVFP4
# expert stages at Small (36 tiles an item, 256 programs) only narrow tiles
# at 1 row, wide tiles then narrow ones at 16 rows (19 items) and only wide
# tiles at 8, 32 and 64 rows; at the half width (18 tiles an item, w2's K
# 1152 ending in a partial ring block) only narrow tiles up to 32 rows and
# only wide tiles at 64.
SMALL, HALF = ip.GEOMETRIES
SMALL_ROWS = [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)]
HALF_ROWS = [(1, 1), (4, 4), (8, 4), (16, 4), (32, 4), (64, 4)]
CASES = [(SMALL, r, c) for r, c in SMALL_ROWS] + [(HALF, r, c) for r, c in HALF_ROWS]
IDS = [f"h{dims.hidden}-r{rows}-c{chain}" for dims, rows, chain in CASES]
RANK = 1


@gluon.jit
def _experts_kernel(scratch, step, up, down, PLAN: gl.constexpr, RANK: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    routes = experts.local_routes(mb, "route", RANK)
    task, active = schedule.stage_task(PLAN, "ex1")
    if active:
        experts.up_stage(
            mb,
            task,
            experts.Bank(*up),
            routes,
            "xq",
            "xa",
            "xq_flag",
            "gate_up",
            "ex1",
            True,
        )
    G: gl.constexpr = schedule.grid(PLAN)
    task, rounds = schedule.stage_rounds(PLAN, "swiglu")
    for i in range(rounds):
        experts.swiglu_stage(
            mb,
            task + i * G,
            "route",
            "gate_up",
            "hq",
            "ha",
            "hq_flag",
            RANK,
            "swiglu",
            True,
        )
    task, active = schedule.stage_task(PLAN, "ex2")
    if active:
        experts.down_stage(
            mb,
            task,
            experts.Bank(*down),
            routes,
            "hq",
            "ha",
            "hq_flag",
            "ex2",
            "ex2",
            True,
        )


_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _bank(local, n, k, g):
    """A random NVFP4 bank of ``local`` experts ``[n, k]`` on the GPU and a
    function giving one expert's BF16 values."""
    codes = torch.randint(0, 256, (local, n, k // 2), dtype=torch.uint8, device="cuda")
    scales = (torch.rand(local, n, k // 16, device="cuda") * 1.5 + 0.25).to(
        torch.float8_e4m3fn
    )
    wa = (torch.rand(local, generator=g) + 0.5).cuda()
    wd = torch.tensor([2.0], device="cuda")
    table = _E2M1.cuda()

    def values(e):
        c = codes[e].int()
        v = torch.stack([table[c & 15], table[c >> 4]], -1).view(n, k)
        s = scales[e].float().repeat_interleave(16, dim=1)
        return (v * s).bfloat16().float()

    return (codes, scales, wa, wd), values


def _route_ids(dims, rank, rows, g):
    """Every row's 8 distinct experts, as the top-k selects them: 4 of
    ``rank``'s (3 that every row shares, one that varies), 4 remote."""
    local = dims.local_experts
    base = rank * local
    remote = [e for e in range(dims.routed) if not base <= e < base + local]
    n = len(remote)
    ids = torch.zeros(rows, 8, dtype=torch.int32)
    for r in range(rows):
        pool = [
            base + 3,
            base + local // 2 + 2,
            base + local - 1,
            base + 10 + r % (local // 2 - 8),
            remote[7],
            remote[n // 2 + 3],
            remote[20 + r % (n // 2 - 20)],
            remote[n - 4],
        ]
        ids[r] = torch.tensor(pool)[torch.randperm(8, generator=g)]
    return ids


def _route_pairs(scratch, plan, region, row, k, n, tag):
    """FP32 of the ``2 n`` BF16 values in pairs ``k n ..`` of a row's region,
    after checking their tags."""
    start = plan.offset(region) + row * plan.row_pairs + k * n
    pairs = scratch[start : start + n]
    assert torch.all((pairs >> 32) == tag), (region, row, k)
    return (pairs & 0xFFFFFFFF).to(torch.int32).view(torch.bfloat16).float().cpu()


def _regular_expert_mm(local, x, xa, bank, ids, per_token):
    """The regular decode expert GEMM (``(16, 64, 256)`` tiles, BF16
    activations; its masked loop where K is not a whole number of 256-wide
    blocks) on this rank's ``local`` experts' routes: BF16 ``[rows * 8, N]``
    by route slot. ``x``: per-token rows (``per_token``) or one row a route
    slot."""
    codes, scales, wa, wd = bank
    routes, counts = _routing(ids.cuda(), local, RANK * local)
    n, k = codes.shape[1], codes.shape[2] * 2
    y = torch.zeros((ids.numel(), n), device="cuda", dtype=torch.bfloat16)
    programs = torch.cuda.get_device_properties(0).multi_processor_count * 2
    gluon_block_scaled_moe_routes_mm_gfx950[(programs,)](
        x.bfloat16().cuda(),
        xa.cuda(),
        xa.cuda(),
        codes,
        scales,
        wa,
        wd,
        y,
        routes,
        counts,
        ids.numel(),
        programs,
        n,
        k,
        local,
        8,
        per_token,
        True,
        16,
        64,
        256,
        True,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return y.float().cpu()


def _regular_swiglu(dims, gate_up, weights, ids):
    """The regular decode SwiGLU and NVFP4 quantization of this rank's routes
    (``_swiglu_quantize_routes`` as the gfx950 expert routes launch it, with
    the BF16 copy of the quantized values): BF16 ``hq [rows * 8, inter]``
    and FP32 ``ha [rows * 8]`` by route slot, local routes only."""
    routes = ids.numel()
    inter, local = dims.inter, dims.local_experts
    second, scale, amax = _quantized_buffers(routes, inter, True, "cuda")
    dequant = torch.empty((routes, inter), device="cuda", dtype=torch.bfloat16)
    _swiglu_quantize_routes(
        gate_up.bfloat16().cuda(),
        weights.cuda(),
        ids.cuda(),
        torch.empty((routes, dims.hidden), device="cuda", dtype=torch.bfloat16),
        second,
        scale,
        amax,
        dequant,
        second_format="nvfp4",
        expert_start=RANK * local,
        num_experts=local,
        gate_cap=4.0,
        up_cap=25.0,
        zero_remote_routes=False,
        route_rows=None,
        w_amax=None,
        w_denominator=None,
        layout=0,
    )
    return dequant.cpu(), amax.cpu()


def _swiglu(gu, gamma):
    g, u = gu[0::2], gu[1::2]
    sg = torch.sigmoid(g)
    return (
        (((4 * torch.tanh(g / 4) * sg) * (25 * torch.tanh(u / 25))) * gamma)
        .bfloat16()
        .float()
    )


@pytest.mark.parametrize("dims,rows,chain", CASES, ids=IDS)
def test_expert_stages_match_tokenspeed_routes(dims, rows, chain):
    g = torch.Generator().manual_seed(rows)
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(
        ("ex1", "swiglu", "ex2")
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    H, INTER, LOCAL = dims.hidden, dims.inter, dims.local_experts
    up, up_values = _bank(LOCAL, 2 * INTER, H, g)
    down, down_values = _bank(LOCAL, H, INTER, g)
    # Routes: rows share some local experts; some routes are remote.
    base = RANK * LOCAL
    ids = _route_ids(dims, RANK, rows, g)
    weights = torch.rand(rows, 8, generator=g) + 0.2
    x = torch.randn(rows, H, generator=g)
    xq, xa = _nvfp4_reference(x)
    for r in range(rows):
        route = torch.cat([ids[r], weights[r].view(torch.int32)])
        fill(scratch, plan, "route", route.cuda(), tag, row=r)
        fill_words(
            scratch, plan, "xq", xq[r].bfloat16().view(torch.int32).cuda(), row=r
        )
        fill(scratch, plan, "xa", xa[r : r + 1].view(torch.int32).cuda(), tag, row=r)
        fill(
            scratch,
            plan,
            "xq_flag",
            torch.ones(1, dtype=torch.int32).cuda(),
            tag,
            row=r,
        )
    _experts_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        up,
        down,
        plan.key,
        RANK,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    up_wa, up_wd = up[2].cpu(), up[3].cpu()
    down_wa, down_wd = down[2].cpu(), down[3].cpu()
    # ex1 and ex2 run the regular expert GEMM's tiles: the same bits.
    regular_up = _regular_expert_mm(LOCAL, xq, xa.float(), up, ids, True)
    hq_routes = torch.cat(
        [
            words(scratch, plan, "hq", 8 * INTER // 2, row=r)
            .view(torch.bfloat16)
            .float()
            .cpu()
            .view(8, INTER)
            for r in range(rows)
        ]
    )
    ha_routes = torch.stack(
        [
            (scratch[plan.offset("ha") + r * plan.row_pairs + k] & 0xFFFFFFFF)
            .to(torch.int32)
            .view(torch.float32)
            .cpu()
            for r in range(rows)
            for k in range(8)
        ]
    )
    regular_down = _regular_expert_mm(LOCAL, hq_routes, ha_routes, down, ids, False)
    # swiglu runs the regular SwiGLU and quantizer's math: the same bits.
    regular_hq, regular_ha = _regular_swiglu(dims, regular_up, weights, ids)
    local = ((ids >= base) & (ids < base + LOCAL)).view(-1)
    got_hq = hq_routes.bfloat16()[local].view(torch.int16)
    assert torch.equal(got_hq, regular_hq[local].view(torch.int16))
    got_ha = ha_routes[local].view(torch.int32)
    assert torch.equal(got_ha, regular_ha[local].view(torch.int32))
    # Every route has the regular GEMMs' bits; the GEMMs' torch math (an
    # expert's dequantized weights a route) is checked on the first and last
    # rows' routes, which cover the shared and the varying experts.
    math_rows = (0, rows - 1)
    for r in range(rows):
        for k in range(8):
            e = int(ids[r, k]) - base
            if not 0 <= e < LOCAL:
                continue
            gate_up = _route_pairs(scratch, plan, "gate_up", r, k, INTER, tag)
            if r in math_rows:
                w13 = up_values(e).cpu()
                want_gu = (xq[r].double() @ w13.double().T).float()
                want_gu = (
                    (want_gu * ((xa[r] * up_wa[e]) * (1 / (1536.0 * up_wd[0]))))
                    .bfloat16()
                    .float()
                )
                torch.testing.assert_close(gate_up, want_gu, rtol=2**-7, atol=1e-5)
            assert torch.equal(gate_up, regular_up[r * 8 + k]), (r, k)
            # From the kernel's gate_up on: h, its quantization and w2.
            h = _swiglu(gate_up, weights[r, k])
            hq, ha = _nvfp4_reference(h.view(1, -1))
            got_hq = words(scratch, plan, "hq", 8 * INTER // 2, row=r).view(
                torch.bfloat16
            )
            got_hq = got_hq.float().cpu().view(8, INTER)[k]
            close = (got_hq == hq[0]).float().mean().item()
            assert close > 0.995, f"row {r} route {k}: {close:.4f} of hq agree"
            at = plan.offset("ha") + r * plan.row_pairs + k
            assert (scratch[at] >> 32).item() == tag
            got_ha = (scratch[at : at + 1] & 0xFFFFFFFF).to(torch.int32)
            got_ha = got_ha.view(torch.float32).cpu()[0]
            torch.testing.assert_close(got_ha, ha[0], rtol=2**-7, atol=0)
            got = _route_pairs(scratch, plan, "ex2", r, k, H // 2, tag)
            if r in math_rows:
                w2 = down_values(e).cpu()
                want = (got_hq.double() @ w2.double().T).float()
                want = (
                    (want * ((got_ha * down_wa[e]) * (1 / (1536.0 * down_wd[0]))))
                    .bfloat16()
                    .float()
                )
                torch.testing.assert_close(got, want, rtol=2**-7, atol=1e-5)
            assert torch.equal(got, regular_down[r * 8 + k]), (r, k)


@gluon.jit
def _swiglu_kernel(scratch, step, PLAN: gl.constexpr, RANK: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    G: gl.constexpr = schedule.grid(PLAN)
    task, rounds = schedule.stage_rounds(PLAN, "swiglu")
    for i in range(rounds):
        experts.swiglu_stage(
            mb,
            task + i * G,
            "route",
            "gate_up",
            "hq",
            "ha",
            "hq_flag",
            RANK,
            "swiglu",
            True,
        )


@pytest.mark.parametrize("dims,rows,chain", CASES, ids=IDS)
def test_swiglu_equals_regular_swiglu(dims, rows, chain):
    """``swiglu`` on gate / up values of a trained layer's range (both caps'
    polynomial and exponential tanh branches, saturated sigmoids, zeros of
    either sign) has the bits of the regular SwiGLU and NVFP4 quantizer."""
    g = torch.Generator().manual_seed(40 + rows)
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(
        ("swiglu",)
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    INTER, LOCAL = dims.inter, dims.local_experts
    base = RANK * LOCAL
    ids = torch.stack(
        [torch.randperm(dims.routed, generator=g)[:8] for _ in range(rows)]
    ).int()
    # At least two local routes a row.
    ids[:, 2] = base + 5
    ids[:, 6] = base + LOCAL - 6
    weights = torch.rand(rows, 8, generator=g) + 0.5
    gate = torch.randn(rows * 8, INTER, generator=g) * 3
    up = torch.randn(rows * 8, INTER, generator=g) * 12
    gate[:, :8] = torch.tensor([0.0, -0.0, -90.0, 90.0, 2.5, -2.5, 1e-30, -1e-30])
    up[:, 8:12] = torch.tensor([0.0, -0.0, 400.0, -15.625])
    gate_up = torch.stack([gate, up], -1).view(rows * 8, 2 * INTER).bfloat16()
    for r in range(rows):
        route = torch.cat([ids[r], weights[r].view(torch.int32)])
        fill(scratch, plan, "route", route.cuda(), tag, row=r)
        pairs = gate_up[r * 8 : (r + 1) * 8].view(torch.int32)
        fill(scratch, plan, "gate_up", pairs.cuda(), tag, row=r)
    _swiglu_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        plan.key,
        RANK,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    want_hq, want_ha = _regular_swiglu(dims, gate_up, weights, ids)
    for r in range(rows):
        got_hq = words(scratch, plan, "hq", 8 * INTER // 2, row=r).cpu()
        got_hq = got_hq.view(torch.int16).view(8, INTER)
        for k in range(8):
            if not 0 <= int(ids[r, k]) - base < LOCAL:
                continue
            route = r * 8 + k
            assert torch.equal(got_hq[k], want_hq[route].view(torch.int16)), (r, k)
            ha_pair = scratch[plan.offset("ha") + r * plan.row_pairs + k].item()
            assert ha_pair >> 32 == tag
            got_ha = torch.tensor([ha_pair & 0xFFFFFFFF]).to(torch.int32)
            assert torch.equal(got_ha, want_ha[route : route + 1].view(torch.int32))
            # And the torch reference of the math.
            h = _swiglu(gate_up[route].float(), weights[r, k])
            hq, ha = _nvfp4_reference(h.view(1, -1))
            close = (got_hq[k].view(torch.bfloat16).float() == hq[0]).float().mean()
            assert close > 0.995, f"row {r} route {k}: {close:.4f} of hq agree"
            torch.testing.assert_close(want_ha[route], ha[0], rtol=2**-7, atol=0)


# The expert output exchange on four ranks: ``fold`` (each rank's slot-order
# sum of its local routes), then ``red_moe`` (the rank fold with the MLP
# output conv).
WORLD = SMALL.world  # both geometries: attention TP4, experts EP4
FOLD_COLUMNS = gl.constexpr(ip.FOLD_COLUMNS)


def _ring_rows(chain):
    """Conv ring rows a slot: the server's ``3 + K`` for verify blocks of
    ``K`` tokens (11 at 8, 7 at 4; decode without verify takes 8's)."""
    return 3 + (chain if chain > 1 else 8)


@gluon.jit
def _fold_kernel(
    scratch,
    step,
    bases,
    term,
    taps,
    ring,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    RING_ROWS: gl.constexpr,
):
    D: gl.constexpr = schedule.dims(PLAN)
    W: gl.constexpr = schedule.world(PLAN)
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, W)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "fold")
    if active:
        experts.fold_stage(
            mb,
            task,
            "route",
            "ex2",
            "fold_part",
            "fold_flag",
            FOLD_COLUMNS,
            RANK,
            "fold",
        )
    task, active = schedule.stage_task(PLAN, "red_moe")
    if active:
        conv.reduce_conv_stage(
            mb,
            task,
            "fold_part",
            "fold_flag",
            False,
            "mlp_sc",
            term,
            taps,
            conv.Ring(
                ring[0], ring[1], ring[2], schedule.chain(PLAN), RING_ROWS, D.ring_width
            ),
            D.mlp_channel0,
            D.hidden,
            FOLD_COLUMNS,
            "red_moe",
        )


def _fold_inputs(dims, rows):
    """Every rank's routes (the same on every rank) and route outputs."""
    g = torch.Generator().manual_seed(5)
    ids = torch.stack(
        [torch.randperm(dims.routed, generator=g)[:8] for _ in range(rows)]
    )
    outs = torch.randn(rows, 8, dims.hidden, generator=g).bfloat16()
    # Zero sums of either sign and exact cancellations.
    outs[:, :, :8] = -0.0
    outs[:, 1::2, 8:16] = -outs[:, 0::2, 8:16]
    return ids.int(), outs


def _fold_worker(rank, port, dims, rows, chain):
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
        plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(
            ("fold", "red_moe")
        )
        H, LOCAL = dims.hidden, dims.local_experts
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        tag = launch_tag(STEP, 0)
        ids, outs = _fold_inputs(dims, rows)
        for r in range(rows):
            route = torch.cat([ids[r], torch.ones(8).view(torch.int32)])
            fill(scratch, plan, "route", route.to(device), tag, row=r)
            for k in range(8):
                if int(ids[r, k]) // LOCAL == rank:
                    start = plan.offset("ex2") + r * plan.row_pairs + k * (H // 2)
                    w = outs[r, k].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
                    scratch[start : start + H // 2] = (tag << 32) | w.to(device)
        g = torch.Generator().manual_seed(8)
        taps = (torch.randn(H, 4, generator=g) / 2).bfloat16()
        requests = rows // chain
        ring_rows = _ring_rows(chain)
        ring = torch.randn(requests + 2, ring_rows, dims.ring_width, generator=g)
        ring = ring.bfloat16()
        # Each request its own ring slot and length.
        slots = [1 + q for q in range(requests)]
        positions = [20 + 5 * (r // chain) + r % chain for r in range(rows)]
        ring_before = ring.clone()
        ring_gpu = ring.to(device)
        term = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
        dist.barrier()
        _fold_kernel[(plan.grid,)](
            scratch,
            torch.full((1,), STEP, dtype=torch.int32, device=device),
            group.bases,
            term,
            taps.to(device),
            (
                ring_gpu,
                torch.tensor(slots, dtype=torch.int32, device=device),
                torch.tensor(positions, dtype=torch.int64, device=device),
            ),
            plan.key,
            rank,
            ring_rows,
            num_warps=NUM_WARPS,
            **extern_compile_options(),
        )
        torch.cuda.synchronize()
        # Each rank's slot-order fold of its local routes, then the rank fold.
        total = torch.zeros(rows, H)
        for source in range(WORLD):
            part = torch.zeros(rows, H)
            for r in range(rows):
                for k in range(8):
                    if int(ids[r, k]) // LOCAL == source:
                        part[r] = part[r] + outs[r, k].float()
            total = total + part.bfloat16().float()
        folded = total.bfloat16().float()
        channels = slice(dims.mlp_channel0, dims.mlp_channel0 + H)
        want = torch.empty(rows, H)
        for r in range(rows):
            p, tok = positions[r], r % chain
            acc = torch.zeros(H)
            for i in range(4):
                back = 3 - i
                if back <= tok:
                    x = folded[r - back]
                else:
                    x = ring_before[
                        slots[r // chain], (p - back) % ring_rows, channels
                    ].float()
                acc = (x.double() * taps[:, i].float().double() + acc.double()).float()
            want[r] = ((acc + folded[r]) * 0.4472135954999579).bfloat16().float()
        torch.testing.assert_close(term.float().cpu(), want, rtol=2**-7, atol=1e-6)
        # The regular path's kernels on the same inputs give the same bits:
        # each rank's route fold, the EP all-reduce's rank-ordered fold, then
        # the MLP output conv (static taps, residual, ring writes).
        flat = outs.view(rows * 8, H).to(device)
        parts = torch.stack(
            [
                _fold_routes(
                    flat,
                    None,
                    ids.to(device),
                    expert_start=source * LOCAL,
                    num_experts=LOCAL,
                    fold_into=None,
                ).view(rows, H)
                for source in range(WORLD)
            ]
        )
        # This launch's (parity 1) copy of the exchange: every rank's partial.
        got_parts = torch.stack(
            [
                words(scratch, plan, "fold_part", WORLD * H // 2, row=r, copy=1)
                for r in range(rows)
            ]
        )
        got_parts = got_parts.view(torch.int16).view(rows, WORLD, H).transpose(0, 1)
        assert torch.equal(got_parts, parts.view(torch.int16))
        reduced = rank_ordered_fold(parts, torch.empty_like(parts[0]))
        regular_ring = ring_before.to(device)
        regular_term = inkling_ring_sconv(
            reduced.view(rows, 1, H),
            taps.to(device),
            regular_ring[:, :, channels],
            torch.arange(0, rows + 1, chain, dtype=torch.int32, device=device),
            torch.arange(rows, dtype=torch.int32, device=device) // chain,
            torch.tensor(slots, dtype=torch.int32, device=device),
            torch.ones(requests, dtype=torch.bool, device=device),
            torch.tensor(
                [positions[q * chain] + chain for q in range(requests)],
                dtype=torch.int32,
                device=device,
            ),
            torch.zeros(requests, 2, dtype=torch.int32, device=device),
            torch.zeros(1, 3, H, dtype=torch.bfloat16, device=device),
            None,
            num_extends=0,
            publish_live=False,
            dynamic_weights=None,
            output_scale=(4 + 1) ** -0.5,
            token_range=None,
        )
        assert torch.equal(term.view(torch.int16), regular_term.view(torch.int16))
        assert torch.equal(ring_gpu.view(torch.int16), regular_ring.view(torch.int16))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dims,rows,chain", CASES, ids=IDS)
def test_expert_fold_exchange_folds_slots_then_ranks(dims, rows, chain):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_fold_worker, args=(port, dims, rows, chain), nprocs=WORLD, join=True)
