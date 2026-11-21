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

"""The Inkling2 MXFP8 expert stages (``quant``, ``ex1``, ``swiglu``, ``ex2``
with MXFP8 experts) of one rank against TokenSpeed's regular kernels at the
server's decode launches, bit for bit, and a torch reference of the math."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import fill, fill_words, payloads, words  # noqa: E402
from test_inkling2_experts_gfx950 import (  # noqa: E402
    _route_ids,
    _route_pairs,
    _swiglu,
)
from tokenspeed_kernel.ops.moe._triton.block_scaled import (  # noqa: E402
    _quantized_buffers,
    _routed_quantized_input,
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
    experts,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    plan as ip,
)

DIMS = ip.GEOMETRIES[0]  # Inkling 2 Small
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    quant,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag  # noqa: E402

NUM_WARPS = 8
ROWS = [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)]
STEP = 13
RANK = 2
LOCAL = 96
H = DIMS.hidden
INTER = DIMS.inter
# (geometry, rows, chain) of the expert stage tests: Inkling 2 Small's verify
# blocks of 8 tokens, the half-width geometry's of 4 (w2's K 1152 ends in a
# partial ring block).
EXPERT_CASES = [(ip.GEOMETRIES[0], rows, chain) for rows, chain in ROWS] + [
    (ip.GEOMETRIES[1], rows, chain)
    for rows, chain in [(1, 1), (4, 4), (8, 4), (16, 4), (32, 4), (64, 4)]
]
EXPERT_IDS = [f"h{d.hidden}-r{rows}-c{chain}" for d, rows, chain in EXPERT_CASES]


def _mxfp8_reference(x):
    """``bf16(v s)`` of rows ``x [R, N]`` (FP32) through TokenSpeed's MXFP8
    quantizer (``_mxfp8_chunk_codes``) and the expert GEMMs' decode."""
    groups = x.view(x.shape[0], -1, 32)
    raw = groups.abs().amax(-1) * torch.tensor(1.0 / 448.0)
    bits = raw.view(torch.int32)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
    scale = torch.exp2(exponent.double() - 127).float().clamp(min=2.0**-127)
    inverse = torch.where(exponent == 0, torch.tensor(2.0**127), 1.0 / scale)
    values = (groups * inverse[..., None]).clamp(-448.0, 448.0)
    codes = values.to(torch.float8_e4m3fn).float()
    return (codes * scale[..., None]).view(x.shape).bfloat16().float()


def _words(x):
    return x.bfloat16().contiguous().view(torch.int32)


def _assert_dot_close(got, x, w):
    """``got`` is ``bf16(x @ w.T)`` up to the FP32 accumulation's error
    (bounded by ``|x| @ |w|.T``) and one BF16 rounding."""
    want = (x.double() @ w.double().T).float()
    bound = (x.double().abs() @ w.double().abs().T).float() * 2**-20
    error = (got - want).abs()
    assert torch.all(error <= want.abs() * 2**-7 + bound), error.max()


@gluon.jit
def _quant_kernel(scratch, step, PLAN: gl.constexpr, N: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "quant")
    if active:
        quant.mxfp8_quant_stage(mb.row(task), "x_mlp", "xq", "xq_flag", N, "quant")


@pytest.mark.parametrize(
    "dims", ip.GEOMETRIES, ids=[f"h{d.hidden}" for d in ip.GEOMETRIES]
)
@pytest.mark.parametrize(
    "rows,chain", [(1, 1), (4, 4), (8, 8), (16, 8), (32, 8), (64, 8)]
)
def test_mxfp8_quant_equals_regular_quantizer(dims, rows, chain):
    """``quant`` publishes the BF16 values of the regular MXFP8 quantizer's
    codes (``_routed_quantized_input`` at the server's launch), bit for bit,
    on rows with zero groups, zeros of either sign and a wide range, at
    every geometry the plans lay out (144 or 72 groups of 32 a row)."""
    g = torch.Generator().manual_seed(30 + rows)
    H = dims.hidden
    plan = ip.local_layer_plan(dims, rows, chain, "mxfp8", False, False).only(
        ("quant",)
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    scale = torch.exp2(torch.randint(-20, 12, (rows, H // 32), generator=g).float())
    x = (torch.randn(rows, H, generator=g) * scale.repeat_interleave(32, 1)).bfloat16()
    x[:, 5::97] = -0.0
    x[:, 64:96] = -0.0
    x[:, 96:128] = 0.0
    x[:, 129] = 448.0 * 2.0**-130
    for r in range(rows):
        fill(scratch, plan, "x_mlp", _words(x[r]).cuda(), tag, row=r)
    _quant_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        plan.key,
        H,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    dequant = torch.empty_like(x).cuda()
    _routed_quantized_input(
        x.cuda(),
        torch.zeros(rows, 8, dtype=torch.int32, device="cuda"),
        dims.local_experts,
        0,
        "mxfp8",
        dequant,
    )
    want = _mxfp8_reference(x.float())
    for r in range(rows):
        assert payloads(scratch, plan, "xq_flag", 1, tag, row=r).item() == 1
        got = words(scratch, plan, "xq", H // 2, row=r).cpu()
        assert torch.equal(got.view(torch.int16), dequant[r].cpu().view(torch.int16))
        assert torch.equal(got.view(torch.bfloat16).float(), want[r])


@gluon.jit
def _experts_kernel(scratch, step, up, down, PLAN: gl.constexpr, RANK: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    routes = experts.local_routes(mb, "route", RANK)
    task, active = schedule.stage_task(PLAN, "ex1")
    if active:
        experts.up_stage(
            mb,
            task,
            experts.bank(up, False),
            routes,
            "xq",
            None,
            "xq_flag",
            "gate_up",
            "ex1",
            False,
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
            None,
            "hq_flag",
            RANK,
            "swiglu",
            False,
        )
    task, active = schedule.stage_task(PLAN, "ex2")
    if active:
        experts.down_stage(
            mb,
            task,
            experts.bank(down, False),
            routes,
            "hq",
            None,
            "hq_flag",
            "ex2",
            "ex2",
            False,
        )


def _bank(local, n, k, g):
    """A random MXFP8 bank of ``local`` experts ``[n, k]`` on the GPU (finite
    E4M3 codes, exponents around a trained layer's) and a function giving
    one expert's BF16 values."""
    codes = torch.randint(0, 256, (local, n, k), dtype=torch.uint8, generator=g)
    # 0x7F and 0xFF are E4M3 NaNs.
    codes = torch.where((codes & 0x7F) == 0x7F, codes ^ 1, codes)
    codes = codes.view(torch.float8_e4m3fn).cuda()
    exps = torch.randint(115, 124, (local, n, k // 32), dtype=torch.uint8, generator=g)
    exps = exps.cuda()

    def values(e):
        s = torch.exp2(exps[e].float() - 127).repeat_interleave(32, dim=1)
        return (codes[e].float() * s).bfloat16().float()

    return (codes, exps), values


def _regular_expert_mm(local, x, bank, ids, per_token):
    """The regular decode expert GEMM (``(16, 64, 256)`` tiles, BF16
    activations, MXFP8 weights; its masked loop where K is not a whole
    number of 256-wide blocks) on this rank's ``local`` experts' routes:
    BF16 ``[rows * 8, N]`` by route slot."""
    codes, exps = bank
    routes, counts = _routing(ids.cuda(), local, RANK * local)
    n, k = codes.shape[1], codes.shape[2]
    y = torch.zeros((ids.numel(), n), device="cuda", dtype=torch.bfloat16)
    programs = torch.cuda.get_device_properties(0).multi_processor_count * 2
    gluon_block_scaled_moe_routes_mm_gfx950[(programs,)](
        x.bfloat16().cuda(),
        exps,
        exps,
        codes,
        exps,
        exps,
        exps,
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
        False,
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
    """The regular decode SwiGLU and MXFP8 quantization of this rank's routes
    (``_swiglu_quantize_routes`` as the gfx950 expert routes launch it, with
    the BF16 copy of the quantized values): BF16 ``[rows * 8, inter]`` by
    route slot, local routes only."""
    routes = ids.numel()
    inter, local = dims.inter, dims.local_experts
    second, scale, amax = _quantized_buffers(routes, inter, False, "cuda")
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
        second_format="mxfp8",
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
    return dequant.cpu()


@pytest.mark.parametrize("dims,rows,chain", EXPERT_CASES, ids=EXPERT_IDS)
def test_mxfp8_expert_stages_equal_regular_routes(dims, rows, chain):
    g = torch.Generator().manual_seed(rows)
    plan = ip.local_layer_plan(dims, rows, chain, "mxfp8", False, False).only(
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
    x = torch.randn(rows, H, generator=g) * 4
    xq = _mxfp8_reference(x)
    for r in range(rows):
        route = torch.cat([ids[r], weights[r].view(torch.int32)])
        fill(scratch, plan, "route", route.cuda(), tag, row=r)
        fill_words(scratch, plan, "xq", _words(xq[r]).cuda(), row=r)
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
    # ex1 and ex2 run the regular expert GEMM's tiles: the same bits.
    regular_up = _regular_expert_mm(LOCAL, xq, up, ids, True)
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
    regular_down = _regular_expert_mm(LOCAL, hq_routes, down, ids, False)
    # swiglu runs the regular SwiGLU and quantizer: the same bits.
    regular_hq = _regular_swiglu(dims, regular_up, weights, ids)
    local = ((ids >= base) & (ids < base + LOCAL)).view(-1)
    got_hq = hq_routes.bfloat16()[local].view(torch.int16)
    assert torch.equal(got_hq, regular_hq[local].view(torch.int16))
    for r in range(rows):
        for k in range(8):
            e = int(ids[r, k]) - base
            if not 0 <= e < LOCAL:
                continue
            assert (
                scratch[plan.offset("hq_flag") + r * plan.row_pairs + k] >> 32
            ) == tag
            w13 = up_values(e).cpu()
            gate_up = _route_pairs(scratch, plan, "gate_up", r, k, INTER, tag)
            _assert_dot_close(gate_up, xq[r : r + 1], w13)
            assert torch.equal(gate_up, regular_up[r * 8 + k]), (r, k)
            # From the kernel's gate_up on: h, its quantization and w2.
            hq = _mxfp8_reference(_swiglu(gate_up, weights[r, k]).view(1, -1))[0]
            assert torch.equal(hq_routes[r * 8 + k], hq), (r, k)
            w2 = down_values(e).cpu()
            got = _route_pairs(scratch, plan, "ex2", r, k, H // 2, tag)
            _assert_dot_close(got, hq.view(1, -1), w2)
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
            None,
            "hq_flag",
            RANK,
            "swiglu",
            False,
        )


@pytest.mark.parametrize("dims,rows,chain", EXPERT_CASES, ids=EXPERT_IDS)
def test_mxfp8_swiglu_equals_regular_swiglu(dims, rows, chain):
    """``swiglu`` on gate / up values of a trained layer's range (both caps'
    branches, saturated sigmoids, zeros of either sign, a zero group) has the
    bits of the regular SwiGLU and MXFP8 quantizer."""
    g = torch.Generator().manual_seed(50 + rows)
    plan = ip.local_layer_plan(dims, rows, chain, "mxfp8", False, False).only(
        ("swiglu",)
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    INTER, LOCAL = dims.inter, dims.local_experts
    base = RANK * LOCAL
    ids = torch.stack(
        [torch.randperm(dims.routed, generator=g)[:8] for _ in range(rows)]
    ).int()
    ids[:, 1] = base + 7
    ids[:, 5] = base + LOCAL - 16
    weights = torch.rand(rows, 8, generator=g) + 0.5
    gate = torch.randn(rows * 8, INTER, generator=g) * 3
    up = torch.randn(rows * 8, INTER, generator=g) * 12
    gate[:, :8] = torch.tensor([0.0, -0.0, -90.0, 90.0, 2.5, -2.5, 1e-30, -1e-30])
    up[:, 8:12] = torch.tensor([0.0, -0.0, 400.0, -15.625])
    gate[:, 64:96] = 0.0
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
    want_hq = _regular_swiglu(dims, gate_up, weights, ids)
    for r in range(rows):
        got_hq = words(scratch, plan, "hq", 8 * INTER // 2, row=r).cpu()
        got_hq = got_hq.view(torch.int16).view(8, INTER)
        for k in range(8):
            if not 0 <= int(ids[r, k]) - base < LOCAL:
                continue
            route = r * 8 + k
            assert torch.equal(got_hq[k], want_hq[route].view(torch.int16)), (r, k)
            h = _swiglu(gate_up[route].float(), weights[r, k])
            hq = _mxfp8_reference(h.view(1, -1))[0]
            assert torch.equal(got_hq[k].view(torch.bfloat16).float(), hq), (r, k)
