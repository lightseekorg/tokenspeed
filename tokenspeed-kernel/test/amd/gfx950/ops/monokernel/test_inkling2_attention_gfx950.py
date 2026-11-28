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

"""The Inkling2 windowed relative attention, its merge with the output gate
and sink, the router top-k and the NVFP4 activation quantization, against
torch references of TokenSpeed's math."""

from __future__ import annotations

import math

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import fill, payloads, words  # noqa: E402
from tokenspeed_kernel.ops.attention.rmha import (  # noqa: E402
    rel_mha_decode_with_kvcache,
)
from tokenspeed_kernel.ops.attention.rmha.triton import (  # noqa: E402
    _decode_split_count,
    _rel_mha_decode_fwd,
)
from tokenspeed_kernel.ops.moe._triton.block_scaled import (  # noqa: E402
    _routed_quantized_input,
    triton_block_scaled_topk,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    attention,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    plan as ip,
)

DIMS = ip.GEOMETRIES[0]  # Inkling 2 Small
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    quant,
    route,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    Stage,
    launch_tag,
)

NUM_WARPS = 8
STEP = 9
PAGE = 16


@gluon.jit
def _attention_kernel(scratch, step, cache, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    c = attention.Cache(*cache)
    G: gl.constexpr = schedule.grid(PLAN)
    task, rounds = schedule.stage_rounds(PLAN, "attn")
    attention.attention_stage(
        mb,
        task,
        rounds,
        c,
        "q",
        "kv",
        "rel",
        "attn_part",
        "attn_lse",
        schedule.chain(PLAN),
        "attn",
    )
    task, rounds = schedule.stage_rounds(PLAN, "merge")
    for i in range(rounds):
        attention.merge_stage(
            mb, task + i * G, c, "qkvr", "attn_part", "attn_lse", "o", "o_flag", "merge"
        )


def _merged(scratch, plan, tag, row):
    """The merge's ``o`` words of ``row``, after checking every KV head's
    flag."""
    kv_heads = plan.dims.kv_heads
    flags = payloads(scratch, plan, "o_flag", kv_heads, tag, row=row)
    assert torch.equal(flags.cpu(), torch.ones(kv_heads, dtype=torch.int32)), row
    return words(scratch, plan, "o", plan.dims.q_dim // 2, row=row)


def _words(x):
    return x.contiguous().view(torch.int32)


def _attention_reference(dims, q, keys, values, rel, gate, sink, L):
    """One row of ``dims``' heads: q [Hq, 128], keys / values [L, Hkv, 128]
    by position, rel [Hq, 1024], gate [Hq * 128], sink [Hq] (FP32 of BF16).
    The split softmax of the regular path with P rounded to BF16 for the PV
    product."""
    hq = dims.q_heads
    n = min(L, 1024)
    start = L - n
    per = math.ceil(math.ceil(n / 8) / 32) * 32
    parts = []
    for s in range(8):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            continue
        idx = torch.arange(start + lo, start + hi)
        k = keys[idx].double()  # [m, Hkv, 128]
        v = values[idx].double()
        o_s = torch.empty(hq, 128, dtype=torch.float64)
        lse_s = torch.empty(hq, dtype=torch.float64)
        for h in range(hq):
            g = h // dims.group
            score = (k[:, g] @ q[h].double()) * 128**-0.5 + rel[h, L - 1 - idx].double()
            m = score.max()
            p = torch.exp(score - m)
            pb = p.float().bfloat16().double()
            o_s[h] = (pb[:, None] * v[:, g]).sum(0) / p.sum()
            lse_s[h] = m + torch.log(p.sum())
        parts.append((o_s, lse_s))
    lse = torch.stack([p[1] for p in parts])  # [S, Hq]
    top = lse.max(0).values
    w = torch.exp(lse - top)
    out = (torch.stack([p[0] for p in parts]) * w[:, :, None]).sum(0) / w.sum(0)[
        :, None
    ]
    lse_all = top + torch.log(w.sum(0))
    sg = lambda x: 1 / (1 + torch.exp(-x))  # noqa: E731
    gated = (out * sg(gate.view(hq, 128).double())) * sg(lse_all - sink.double())[
        :, None
    ]
    return gated.float().bfloat16().float().view(-1)


# The half-width geometry, whose checkpoint verifies blocks of 4 tokens.
_HALF = ip.GEOMETRIES[1]


def _geometry_cases(cases, half_cases):
    """``cases`` at every geometry, then ``half_cases`` (4-token verify
    blocks) at the half-width one, each as ``(dims, *case)``."""
    return [
        pytest.param(d, *case, id=f"{d.hidden}-{case}")
        for d in ip.GEOMETRIES
        for case in cases
    ] + [pytest.param(_HALF, *case, id=f"{_HALF.hidden}-{case}") for case in half_cases]


@pytest.mark.parametrize(
    "dims,rows,chain,first",
    _geometry_cases(
        [(1, 1, 40), (8, 8, 300), (8, 8, 1500)], [(4, 4, 300), (4, 4, 1500)]
    ),
)
def test_attention_and_merge_match_tokenspeed_math(dims, rows, chain, first):
    g = torch.Generator().manual_seed(first)
    hq, hkv = dims.q_heads, dims.kv_heads
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(
        ("attn", "merge")
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    positions = [first + r % chain for r in range(rows)]
    total = first + chain  # positions 0 .. total - 1 exist
    pages = (total + PAGE - 1) // PAGE
    table = (torch.randperm(4 * pages, generator=g)[:pages] + 1).int()
    keys = torch.randn(total, hkv, 128, generator=g).bfloat16()
    values = torch.randn(total, hkv, 128, generator=g).bfloat16()
    slots = 4 * pages * PAGE + PAGE
    k_cache = torch.randn(slots, hkv, 128, generator=g).bfloat16()
    v_cache = torch.randn(slots, hkv, 128, generator=g).bfloat16()
    launch0 = first
    for pos in range(total):
        slot = int(table[pos // PAGE]) * PAGE + pos % PAGE
        if pos < launch0:
            k_cache[slot] = keys[pos]
            v_cache[slot] = values[pos]
        # The launch's own tokens are not in the cache yet: it holds garbage.
    q = torch.randn(rows, hq, 128, generator=g).bfloat16()
    rel = torch.randn(rows, hq, 1024, generator=g).bfloat16()
    qkvr = torch.randn(rows, dims.qkvr, generator=g).bfloat16()
    for r in range(rows):
        pos = positions[r]
        fill(scratch, plan, "q", _words(q[r].view(-1)).cuda(), tag, row=r)
        kv = torch.cat([keys[pos].view(-1), values[pos].view(-1)])
        fill(scratch, plan, "kv", _words(kv).cuda(), tag, row=r)
        fill(scratch, plan, "rel", _words(rel[r].view(-1)).cuda(), tag, row=r)
        fill(scratch, plan, "qkvr", _words(qkvr[r]).cuda(), tag, row=r)
    _attention_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        (
            k_cache.cuda(),
            v_cache.cuda(),
            torch.tensor(positions, dtype=torch.int64, device="cuda"),
            table.view(1, -1).cuda(),
            pages,
            PAGE,
        ),
        plan.key,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    x = qkvr.float()
    for r in range(rows):
        got = _merged(scratch, plan, tag, r).view(torch.bfloat16).float()
        want = _attention_reference(
            dims,
            q[r].float(),
            keys.float(),
            values.float(),
            rel[r].float(),
            x[r, dims.gate_off : dims.sink_off],
            x[r, dims.sink_off : dims.sink_off + hq],
            positions[r] + 1,
        )
        torch.testing.assert_close(got.cpu(), want, rtol=2**-6, atol=2e-3)


# The served Inkling2 local layers: their KV page, the Triton decode's
# window_left and a max_seqlen_k past the window (8 KV splits).
SERVED_PAGE = 128
WINDOW_LEFT = 1023
MAX_SEQLEN_K = 262144


@pytest.mark.parametrize(
    "dims,chain,firsts",
    _geometry_cases(
        [
            (1, (40,)),
            # One verify block just inside the 1024-key window.
            (8, (1013,)),
            # Verify batches of 2, 4 and 8 requests, each of its own length
            # (within and past the window).
            (8, (300, 1500)),
            (8, (40, 1013, 1500, 4096)),
            (8, (40, 300, 1013, 1500, 4096, 77, 1024, 2048)),
        ],
        [
            (4, (1013,)),
            (4, (300, 1500)),
            (4, (40, 1013, 1500, 4096)),
            (4, (40, 300, 1013, 1500, 4096, 77, 1024, 2048)),
        ],
    ),
)
def test_attention_and_merge_equal_regular_decode(dims, chain, firsts):
    """The stages equal, bit for bit, TokenSpeed's decode of the same rows
    as served (``rel_mha_decode_with_kvcache``, ``--attention-backend
    triton``: the Triton split-KV kernels at a 128-token page, then
    ``rel_mha_gate_sink``), for every request of a verify batch."""
    requests = len(firsts)
    rows = requests * chain
    hq, hkv = dims.q_heads, dims.kv_heads
    g = torch.Generator().manual_seed(firsts[0] + rows)
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(
        ("attn", "merge")
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    positions = [firsts[r // chain] + r % chain for r in range(rows)]
    totals = [first + chain for first in firsts]
    pages = (max(totals) + SERVED_PAGE - 1) // SERVED_PAGE
    # Each request's own pages.
    table = (
        (torch.randperm(2 * requests * pages, generator=g)[: requests * pages] + 1)
        .int()
        .view(requests, pages)
    )
    shape = (2 * requests * pages + 1, SERVED_PAGE, hkv, 128)
    k_cache = torch.randn(shape, generator=g).bfloat16()
    v_cache = torch.randn(shape, generator=g).bfloat16()
    q = torch.randn(rows, hq, 128, generator=g).bfloat16()
    rel = (torch.randn(rows, hq, 1024, generator=g) * 2).bfloat16()
    qkvr = torch.randn(rows, dims.qkvr, generator=g).bfloat16()
    # The launch's own tokens: in the regular path's cache (the prologue
    # wrote them), in the monokernel's mailbox (its cache rows hold garbage).
    kv = torch.randn(rows, 2, hkv, 128, generator=g).bfloat16()
    mk_k, mk_v = k_cache.clone(), v_cache.clone()
    for i, first in enumerate(firsts):
        own = torch.arange(first, first + chain)
        launch = table[i, own // SERVED_PAGE].long()
        k_cache[launch, own % SERVED_PAGE] = kv[i * chain : (i + 1) * chain, 0]
        v_cache[launch, own % SERVED_PAGE] = kv[i * chain : (i + 1) * chain, 1]
    for r in range(rows):
        fill(scratch, plan, "q", _words(q[r].view(-1)).cuda(), tag, row=r)
        fill(scratch, plan, "kv", _words(kv[r].view(-1)).cuda(), tag, row=r)
        fill(scratch, plan, "rel", _words(rel[r].view(-1)).cuda(), tag, row=r)
        fill(scratch, plan, "qkvr", _words(qkvr[r]).cuda(), tag, row=r)
    _attention_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        (
            mk_k.view(-1, hkv, 128).cuda(),
            mk_v.view(-1, hkv, 128).cuda(),
            torch.tensor(positions, dtype=torch.int64, device="cuda"),
            table.cuda(),
            pages,
            SERVED_PAGE,
        ),
        plan.key,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    qkvr = qkvr.cuda()
    seqlens = torch.tensor(totals, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(0, rows + 1, chain, dtype=torch.int32, device="cuda")
    want = rel_mha_decode_with_kvcache(
        q=q.cuda(),
        k_cache=k_cache.cuda(),
        v_cache=v_cache.cuda(),
        page_table=table.cuda(),
        cache_seqlens=seqlens,
        max_seqlen_k=MAX_SEQLEN_K,
        rel_logits=rel.cuda(),
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=chain,
        window_left=WINDOW_LEFT,
        softmax_scale=128**-0.5,
        solution="triton",
        output_dtype=torch.bfloat16,
        return_lse=False,
        gate=qkvr[:, dims.gate_off : dims.sink_off],
        sink=qkvr[:, dims.sink_off : dims.sink_off + hq],
    )
    # The split outputs and LSEs of the regular path's first kernel, by its
    # launcher with the op's split count.
    splits = _decode_split_count(MAX_SEQLEN_K, WINDOW_LEFT)
    mid = torch.empty(rows, hq, splits, 128, device="cuda")
    mid_lse = torch.empty(rows, hq, splits, device="cuda")
    _rel_mha_decode_fwd(
        q.cuda(),
        rel.cuda(),
        k_cache.cuda().view(-1, hkv, 128),
        v_cache.cuda().view(-1, hkv, 128),
        torch.empty(rows, hq, 128, device="cuda"),
        torch.empty(rows, hq, device="cuda"),
        table.cuda(),
        seqlens,
        mid,
        mid_lse,
        splits,
        splits,
        chain,
        pages,
        SERVED_PAGE,
        WINDOW_LEFT,
        128**-0.5,
        False,
    )
    torch.cuda.synchronize()
    for r in range(rows):
        n = min(positions[r] + 1, 1024)
        per = (math.ceil(n / 8) + 31) // 32 * 32
        used = math.ceil(n / per)
        part = payloads(scratch, plan, "attn_part", 8 * hq * 128, tag, row=r)
        part = part.view(torch.float32).view(8, hq, 128)[:used].transpose(0, 1)
        assert torch.equal(part, mid[r, :, :used]), f"row {r}: split outputs"
        part_lse = payloads(scratch, plan, "attn_lse", 8 * hq, tag, row=r)
        part_lse = part_lse.view(torch.float32).view(8, hq)[:used].t()
        assert torch.equal(part_lse, mid_lse[r, :, :used]), f"row {r}: split LSEs"
    got = torch.stack([_merged(scratch, plan, tag, r) for r in range(rows)]).view(
        torch.bfloat16
    )
    mismatch = (got != want.view(rows, -1)).sum().item()
    assert mismatch == 0, f"{mismatch} of {got.numel()} values differ"


@gluon.jit
def _topk_kernel(scratch, step, bias, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "topk")
    if active:
        route.topk_stage(mb.row(task), bias, "logits", "route", "topk")


# The routing and quantization stages at every geometry the plans lay out
# (384 or 128 experts; 288 or 144 quantization groups a row). They take a
# task a row: one row, rows that fit the CTAs past the stage's first, and
# rows that wrap around the grid.
_ROUTE_DIMS = pytest.mark.parametrize(
    "dims", ip.GEOMETRIES, ids=[f"h{d.hidden}" for d in ip.GEOMETRIES]
)
_ROUTE_ROWS = [(1, 1), (8, 8), (64, 8)]


def _monokernel_topk(plan, logits, bias):
    """The ``topk`` stage's routes of FP32 ``logits [rows, E]`` (``E`` the
    plan's routed experts): int32 ``[rows, 16]``, the 8 expert ids then the
    bits of the 8 FP32 weights."""
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    for r in range(logits.shape[0]):
        fill(scratch, plan, "logits", logits[r].view(torch.int32).cuda(), tag, row=r)
    _topk_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        bias.cuda(),
        plan.key,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    return torch.stack(
        [
            payloads(scratch, plan, "route", 16, tag, row=r).cpu()
            for r in range(logits.shape[0])
        ]
    )


@_ROUTE_DIMS
@pytest.mark.parametrize("rows,chain", _ROUTE_ROWS)
def test_topk_matches_tokenspeed_routing(dims, rows, chain):
    g = torch.Generator().manual_seed(rows)
    E = dims.routed
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(("topk",))
    logits = torch.randn(rows, E, generator=g) * 3
    # Ties: equal scores pick the lower expert; the last expert can win.
    logits[:, 10] = logits[:, 20] = 5.0
    logits[:, E - 1] = 6.0
    bias = torch.randn(E, generator=g) * 0.1
    bias[10] = bias[20] = 0.0
    routes = _monokernel_topk(plan, logits, bias)
    for r in range(rows):
        got = routes[r]
        score = torch.sigmoid(logits[r]) + bias
        order = sorted(range(E), key=lambda e: (-score[e].item(), e))[:8]
        assert got[:8].tolist() == order
        sel = logits[r, order].double()
        want = 8 * torch.sigmoid(sel) / torch.sigmoid(sel).sum()
        torch.testing.assert_close(
            got[8:].view(torch.float32).double(), want, rtol=2e-6, atol=0
        )


@_ROUTE_DIMS
@pytest.mark.parametrize("rows,chain", _ROUTE_ROWS)
def test_topk_equals_regular_topk(dims, rows, chain):
    """The ids and weight bits of the regular decode top-k
    (``triton_block_scaled_topk``, which ``Inkling2MoE`` runs without a
    placement) on the same FP32 logits, near ties included."""
    E = dims.routed
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(("topk",))
    g = torch.Generator().manual_seed(17 + rows)
    # Two experts that tie exactly, apart in the expert order and from the
    # near ties below (200 and 300 of Small's 384).
    tie = (E // 2 + 8, E - 84)
    bias = torch.randn(E, generator=g) * 0.1
    bias[tie[0]] = bias[tie[1]] = 0.0
    for trial in range(16):
        logits = torch.randn(rows, E, generator=g) * (1 + trial / 2)
        # Scores a few ulps apart, and an exact tie, near the 8th place.
        logits[:, 100:110] = 2.0 + torch.arange(10) * 2.4e-7
        logits[:, tie[0]] = logits[:, tie[1]] = 2.0
        got = _monokernel_topk(plan, logits, bias)
        weights, ids = triton_block_scaled_topk(
            logits.cuda(), bias.cuda(), top_k=8, route_scale=8.0, dispatch=None
        )
        assert torch.equal(got[:, :8], ids.cpu()), trial
        assert torch.equal(got[:, 8:], weights.cpu().view(torch.int32)), trial


@gluon.jit
def _quant_kernel(scratch, step, PLAN: gl.constexpr, N: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "quant")
    if active:
        quant.quant_stage(mb.row(task), "x_mlp", "xq", "xa", "xq_flag", N, "quant")


def _e2m1(y):
    """Round to the nearest E2M1 value, ties to the even code (TokenSpeed's
    thresholds 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5), saturating at 6."""
    a = y.abs()
    v = torch.full_like(a, 0.5)
    v = torch.where(a <= 0.25, 0.0, v)
    v = torch.where(a >= 0.75, 1.0, v)
    v = torch.where(a > 1.25, 1.5, v)
    v = torch.where(a >= 1.75, 2.0, v)
    v = torch.where(a > 2.5, 3.0, v)
    v = torch.where(a >= 3.5, 4.0, v)
    v = torch.where(a > 5.0, 6.0, v)
    return torch.where(y < 0, -v, v)


def _nvfp4_reference(x):
    """``bf16(v sc)`` and amax of rows ``x [R, N]`` (FP32 values)."""
    amax = x.abs().amax(-1, keepdim=True)
    enc = torch.clamp(torch.tensor(1536.0) / amax, max=3.4028234663852886e38)
    dec = 1.0 / enc
    outer = amax * (1.0 / 1536.0)
    groups = x.view(x.shape[0], -1, 16)
    gm = groups.abs().amax(-1, keepdim=True)
    cands = []
    for target in (6.0, 4.0):
        sc = torch.clamp(gm * (enc[:, :, None] * (1.0 / target)), max=448.0)
        sc = sc.to(torch.float8_e4m3fn).float()
        inv = torch.clamp(1.0 / (sc * dec[:, :, None]), max=3.4028234663852886e38)
        v = _e2m1((groups * inv).double()).float()
        value = v * sc
        e = (groups - value * outer[:, :, None]).abs()
        while e.shape[-1] > 1:
            e = e.view(*e.shape[:-1], -1, 2).sum(-1)
        cands.append((value, e))
    best = torch.where(cands[1][1] < cands[0][1], cands[1][0], cands[0][0])
    return best.view(x.shape).bfloat16().float(), amax.view(-1)


@_ROUTE_DIMS
@pytest.mark.parametrize("rows,chain", _ROUTE_ROWS)
def test_nvfp4_quant_matches_tokenspeed(dims, rows, chain):
    g = torch.Generator().manual_seed(rows)
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False).only(
        ("quant",)
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    x = (
        torch.randn(rows, dims.hidden, generator=g)
        * torch.rand(rows, 1, generator=g)
        * 4
    ).bfloat16()
    # Zeros of either sign, a whole group of them among them.
    x[:, 5::97] = -0.0
    x[:, 16:32] = -0.0
    x[:, 40::101] = 0.0
    for r in range(rows):
        fill(scratch, plan, "x_mlp", _words(x[r]).cuda(), tag, row=r)
    _quant_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        plan.key,
        dims.hidden,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    want, amax = _nvfp4_reference(x.float())
    for r in range(rows):
        got = words(scratch, plan, "xq", dims.hidden // 2, row=r)
        got = got.view(torch.bfloat16).float().cpu()
        assert payloads(scratch, plan, "xq_flag", 1, tag, row=r).item() == 1
        got_a = payloads(scratch, plan, "xa", 1, tag, row=r).view(torch.float32).cpu()
        assert torch.equal(got_a, amax[r : r + 1])
        mismatch = (got != want[r]).sum().item()
        # The reference's division and the kernel's div_rn agree; allow no
        # mismatch.
        assert mismatch == 0, f"row {r}: {mismatch} values differ"
    # The regular quantizer of the expert input (the BF16 copy of its codes
    # that the gfx950 expert GEMM reads) gives the same bits, zero signs too.
    dequant = torch.empty_like(x).cuda()
    _, (_, _, regular_amax) = _routed_quantized_input(
        x.cuda(),
        torch.zeros(rows, 8, dtype=torch.int32, device="cuda"),
        dims.local_experts,
        0,
        "nvfp4",
        dequant,
    )
    for r in range(rows):
        got = words(scratch, plan, "xq", dims.hidden // 2, row=r).cpu()
        assert torch.equal(got.view(torch.int16), dequant[r].cpu().view(torch.int16))
        got_a = payloads(scratch, plan, "xa", 1, tag, row=r).cpu()
        assert torch.equal(got_a, regular_amax[r : r + 1].cpu().view(torch.int32))


FULL_WINDOW_LEFT = -1
# The served context length: the regular decode then runs 32 splits.
FULL_MAX_SEQLEN_K = 262120


def _full_plan(dims, rows, chain, shared):
    """The full attention layer's attention and merge stages alone:
    ``shared`` rows of a verify block a task (8 or 4), or a row a task (0)."""
    local = ip.local_layer_plan(dims, rows, chain, "nvfp4", False, False)
    stages = (
        Stage(
            "attn",
            # A task a (verify block, KV head, split), or a (row, KV head,
            # split).
            dims.kv_heads * ip.FULL_SPLITS * (rows // shared if shared else rows),
            ip.LOCAL_BASES["attn"],
            ("q", "kv", "rel"),
            ("attn_part", "attn_lse"),
        ),
        Stage(
            "merge",
            dims.kv_heads * rows,
            ip.LOCAL_BASES["merge"],
            ("attn_part", "attn_lse", "qkvr"),
            ("o", "o_flag"),
        ),
    )
    return Plan(
        grid=local.grid,
        stages=stages,
        regions=local.regions,
        rows=rows,
        chain=chain,
        external=("q", "kv", "rel", "qkvr"),
        world=local.world,
        peers=local.peers,
        flags=local.flags,
        dims=dims,
    )


@gluon.jit
def _full_attention_kernel(
    scratch, step, cache, PLAN: gl.constexpr, SHARED: gl.constexpr
):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    c = attention.Cache(*cache)
    G: gl.constexpr = schedule.grid(PLAN)
    task, rounds = schedule.stage_rounds(PLAN, "attn")
    if SHARED > 0:
        attention.shared_attention_stage(
            mb,
            task,
            rounds,
            c,
            "q",
            "kv",
            "rel",
            "attn_part",
            "attn_lse",
            ip.FULL_SPLITS,
            SHARED // 4,
            "attn",
        )
    else:
        attention.full_attention_stage(
            mb,
            task,
            rounds,
            c,
            "q",
            "kv",
            "rel",
            "attn_part",
            "attn_lse",
            schedule.chain(PLAN),
            ip.FULL_SPLITS,
            "attn",
        )
    task, rounds = schedule.stage_rounds(PLAN, "merge")
    for i in range(rounds):
        attention.full_merge_stage(
            mb,
            task + i * G,
            c,
            "qkvr",
            "attn_part",
            "attn_lse",
            "o",
            "o_flag",
            ip.FULL_SPLITS,
            "merge",
        )


def _full_case(dims, firsts, chain, seed):
    """Inputs of a verify batch whose requests start at ``firsts``."""
    requests = len(firsts)
    rows = requests * chain
    hq, hkv = dims.q_heads, dims.kv_heads
    g = torch.Generator().manual_seed(seed)
    totals = [first + chain for first in firsts]
    pages = (max(totals) + SERVED_PAGE - 1) // SERVED_PAGE
    table = (
        (torch.randperm(2 * requests * pages, generator=g)[: requests * pages] + 1)
        .int()
        .view(requests, pages)
    )
    shape = (2 * requests * pages + 1, SERVED_PAGE, hkv, 128)
    k_cache = torch.randn(shape, generator=g).bfloat16()
    v_cache = torch.randn(shape, generator=g).bfloat16()
    q = torch.randn(rows, hq, 128, generator=g).bfloat16()
    rel = (torch.randn(rows, hq, 1024, generator=g) * 2).bfloat16()
    qkvr = torch.randn(rows, dims.qkvr, generator=g).bfloat16()
    kv = torch.randn(rows, 2, hkv, 128, generator=g).bfloat16()
    mk_k, mk_v = k_cache.clone(), v_cache.clone()
    for i, first in enumerate(firsts):
        own = torch.arange(first, first + chain)
        launch = table[i, own // SERVED_PAGE].long()
        k_cache[launch, own % SERVED_PAGE] = kv[i * chain : (i + 1) * chain, 0]
        v_cache[launch, own % SERVED_PAGE] = kv[i * chain : (i + 1) * chain, 1]
    positions = [firsts[r // chain] + r % chain for r in range(rows)]
    return dict(
        rows=rows,
        pages=pages,
        table=table,
        k_cache=k_cache,
        v_cache=v_cache,
        mk_k=mk_k,
        mk_v=mk_v,
        q=q,
        rel=rel,
        qkvr=qkvr,
        kv=kv,
        positions=positions,
        totals=totals,
    )


def _full_launch(case, chain, plan, scratch, tag, shared):
    rows = case["rows"]
    for r in range(rows):
        fill(scratch, plan, "q", _words(case["q"][r].view(-1)).cuda(), tag, row=r)
        fill(scratch, plan, "kv", _words(case["kv"][r].view(-1)).cuda(), tag, row=r)
        fill(scratch, plan, "rel", _words(case["rel"][r].view(-1)).cuda(), tag, row=r)
        fill(scratch, plan, "qkvr", _words(case["qkvr"][r]).cuda(), tag, row=r)
    hkv = plan.dims.kv_heads
    args = (
        case["mk_k"].view(-1, hkv, 128).cuda(),
        case["mk_v"].view(-1, hkv, 128).cuda(),
        torch.tensor(case["positions"], dtype=torch.int64, device="cuda"),
        case["table"].cuda(),
        case["pages"],
        SERVED_PAGE,
    )
    _full_attention_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        args,
        plan.key,
        shared,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )


def _full_attention_cases(cases, half_cases):
    """``_geometry_cases`` with the rows a task takes (``shared``): a row
    (0) in every case, and in blocks of 8 also the verify block's 8 or 4."""
    geometry = [(d, *case) for d in ip.GEOMETRIES for case in cases]
    geometry += [(_HALF, *case) for case in half_cases]
    return [
        pytest.param(
            d, chain, firsts, shared, id=f"{shared}-{d.hidden}-{(chain, firsts)}"
        )
        for d, chain, firsts in geometry
        for shared in ((0, 4, 8) if chain == 8 else (0,))
    ]


@pytest.mark.parametrize(
    "dims,chain,firsts,shared",
    _full_attention_cases(
        [
            (1, (40,)),
            # One verify block across 1024 keys (32 keys a split, then 64),
            # one ending at 32 whole splits of 1024 keys.
            (8, (1020,)),
            (8, (32760,)),
            # The served context length's last verify block.
            (8, (FULL_MAX_SEQLEN_K - 8,)),
            # Verify batches of 2, 4 and 8 requests, each of its own length.
            (8, (300, 9000)),
            (8, (40, 1013, 4096, 20000)),
            (8, (40, 300, 1013, 1500, 4096, 77, 9000, 2048)),
            # Rows of a verify block whose split lengths differ (1024 keys and
            # past: 32 keys a split, then 64); a block of 8 padded rows at 0.
            (8, (1018, 2045, 50000, 3069)),
            (8, (0, 40)),
            # Lengths of several thousand keys whose block 1024 keys back holds
            # the relative logits' edge for some rows of a block but not others.
            (8, (8743, 8345)),
            (8, (9175, 8477)),
        ],
        [
            (4, (1020,)),
            (4, (FULL_MAX_SEQLEN_K - 4,)),
            (4, (300, 9000)),
            (4, (40, 1013, 4096, 20000)),
            (4, (1018, 2045, 50000, 3069)),
            (4, (0, 40)),
            (4, (40, 300, 1013, 1500, 4096, 77, 9000, 2048)),
        ],
    ),
)
def test_full_attention_and_merge_equal_regular_decode(dims, chain, firsts, shared):
    """A full attention layer's stages equal, bit for bit, the regular
    decode of the same rows (no window, 32 splits of one or more KV
    blocks, relative logits within 1024 keys), with a row a task or a
    verify block's 8 or 4 rows a task (``shared``)."""
    hq, hkv = dims.q_heads, dims.kv_heads
    case = _full_case(dims, firsts, chain, firsts[0] + len(firsts))
    rows = case["rows"]
    plan = _full_plan(dims, rows, chain, shared)
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    _full_launch(case, chain, plan, scratch, tag, shared)
    qkvr = case["qkvr"].cuda()
    seqlens = torch.tensor(case["totals"], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(0, rows + 1, chain, dtype=torch.int32, device="cuda")
    table, pages = case["table"].cuda(), case["pages"]
    k_cache, v_cache = case["k_cache"].cuda(), case["v_cache"].cuda()
    q, rel = case["q"].cuda(), case["rel"].cuda()
    want = rel_mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=table,
        cache_seqlens=seqlens,
        max_seqlen_k=FULL_MAX_SEQLEN_K,
        rel_logits=rel,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=chain,
        window_left=FULL_WINDOW_LEFT,
        softmax_scale=128**-0.5,
        solution="triton",
        output_dtype=torch.bfloat16,
        return_lse=False,
        gate=qkvr[:, dims.gate_off : dims.sink_off],
        sink=qkvr[:, dims.sink_off : dims.sink_off + hq],
    )
    splits = _decode_split_count(FULL_MAX_SEQLEN_K, FULL_WINDOW_LEFT)
    assert splits == ip.FULL_SPLITS
    mid = torch.empty(rows, hq, splits, 128, device="cuda")
    mid_lse = torch.empty(rows, hq, splits, device="cuda")
    _rel_mha_decode_fwd(
        q,
        rel,
        k_cache.view(-1, hkv, 128),
        v_cache.view(-1, hkv, 128),
        torch.empty(rows, hq, 128, device="cuda"),
        torch.empty(rows, hq, device="cuda"),
        table,
        seqlens,
        mid,
        mid_lse,
        splits,
        splits,
        chain,
        pages,
        SERVED_PAGE,
        FULL_WINDOW_LEFT,
        128**-0.5,
        False,
    )
    torch.cuda.synchronize()
    for r in range(rows):
        n = case["positions"][r] + 1
        per = (math.ceil(n / splits) + 31) // 32 * 32
        used = math.ceil(n / per)
        # An unused split publishes nothing.
        part = payloads(scratch, plan, "attn_part", used * hq * 128, tag, row=r)
        part = part.view(torch.float32).view(used, hq, 128).transpose(0, 1)
        assert torch.equal(part, mid[r, :, :used]), f"row {r}: split outputs"
        part_lse = payloads(scratch, plan, "attn_lse", used * hq, tag, row=r)
        part_lse = part_lse.view(torch.float32).view(used, hq).t()
        assert torch.equal(part_lse, mid_lse[r, :, :used]), f"row {r}: split LSEs"
    got = torch.stack([_merged(scratch, plan, tag, r) for r in range(rows)]).view(
        torch.bfloat16
    )
    mismatch = (got != want.view(rows, -1)).sum().item()
    assert mismatch == 0, f"{mismatch} of {got.numel()} values differ"
