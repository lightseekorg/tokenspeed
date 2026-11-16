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

"""The Inkling2 head prologue and depth-mix stages against the regular decode
path's kernels, bit for bit (``_rel_mha_head_norms_kernel`` through the
registered relative GQA prologue, ``_attnres_combine_kernel`` through
``attention_residual_combine``), and against torch references of TokenSpeed's
math."""

from __future__ import annotations

import pytest
import torch
from monokernel_testing import fill, payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.prologue import gqa_prologue  # noqa: E402
from tokenspeed_kernel.ops.attention.prologue.types import (  # noqa: E402
    ConvBatch,
    ConvStream,
    HeadKVCache,
    KVShortConv,
    RelativeHeadNorm,
)
from tokenspeed_kernel.ops.residual.attention import (  # noqa: E402
    attention_residual_combine,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    conv,
    head,
    mix,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    plan as ip,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag  # noqa: E402

NUM_WARPS = 8
STEP = 5
H = ip.HIDDEN
SCALE = 0.4472135954999579  # 5 ** -0.5
CACHE_SLOTS = 128  # KV cache rows of the head tests (at least the rows)
ROWS = [(1, 1), (8, 8), (16, 8), (32, 8), (64, 8)]


@gluon.jit
def _head_kernel(scratch, step, weights, cache, ring, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    G: gl.constexpr = schedule.grid(PLAN)
    task, rounds = schedule.stage_rounds(PLAN, "head")
    for i in range(rounds):
        head.head_stage(
            mb,
            task + i * G,
            head.Weights(*weights),
            head.Cache(*cache),
            conv.Ring(ring[0], ring[1], ring[2], schedule.chain(PLAN)),
            "qkvr",
            "q",
            "kv",
            "rel",
            "head",
        )


def _words(x):
    return x.contiguous().view(torch.int32)


def _clip(y, thr):
    ss = (y * y).sum(-1, keepdim=True)
    n = torch.where(ss > 0, ss.sqrt(), torch.zeros_like(ss))
    return y * torch.clamp(thr / (n + 1e-6), max=1.0)


def _short_conv(raw, taps, ring, ch0, slots, positions, chain):
    """FP32 (BF16-rounded) conv outputs of ``raw [rows, C]``."""
    rows, c = raw.shape
    out = torch.empty(rows, c)
    for r in range(rows):
        slot, p, tok = slots[r // chain], positions[r], r % chain
        acc = torch.zeros(c)
        for i in range(4):
            back = 3 - i
            if back <= tok:
                x = raw[r - back]
            elif slot >= 0 and p - back >= 0:
                x = ring[slot, (p - back) % 11, ch0 : ch0 + c].float()
            else:
                x = torch.zeros(c)
            acc = (x.double() * taps[:, i].double() + acc.double()).float()
        out[r] = ((acc + raw[r]) * SCALE).bfloat16().float()
    return out


def _regular_head(qkvr, weights, ring, slots, positions, locations, chain):
    """The regular decode path's prologue of the projection rows ``qkvr``
    (the registered relative GQA prologue: one ``_rel_mha_head_norms_kernel``
    launch): ``(q, rel_logits, k_cache, v_cache, ring)`` on the host."""
    rows = qkvr.shape[0]
    requests = rows // chain
    q_scale, rel_scale, rel_proj, k_taps, v_taps = weights
    qkvr = qkvr.cuda()
    ring = ring.cuda()
    cache = torch.zeros(2, CACHE_SLOTS, 2, 128, dtype=torch.bfloat16, device="cuda")
    checkpoint = torch.zeros(requests + 1, 3, 512, dtype=torch.bfloat16).cuda()

    def i32(x):
        return torch.as_tensor(x, dtype=torch.int32).cuda()

    batch = ConvBatch(
        cu_seqlens=i32(list(range(0, rows + 1, chain))),
        seq_idx=i32([r // chain for r in range(rows)]),
        cache_indices=i32(slots),
        has_initial_state=torch.ones(requests, dtype=torch.bool, device="cuda"),
        seq_lens=i32([positions[(r + 1) * chain - 1] + 1 for r in range(requests)]),
        block_table=torch.zeros(requests, 3, dtype=torch.int32, device="cuda"),
        num_extends=0,
        # A verify block publishes no window; the monokernel publishes none.
        publish_live=False,
    )
    k_stream, v_stream = (
        ConvStream(
            weight=taps.cuda(),
            dynamic_weights=None,
            ring=ring[:, :, 256 * i : 256 * (i + 1)],
            checkpoint=checkpoint[:, :, 256 * i : 256 * (i + 1)],
        )
        for i, taps in enumerate((k_taps, v_taps))
    )
    norm = RelativeHeadNorm(
        relative=qkvr[:, 2048:2240].view(rows, 12, 16),
        projection=rel_proj.cuda(),
        q_scale=q_scale.cuda(),
        k_scale=None,
        rel_scale=rel_scale.cuda(),
        q_threshold=22.627416997969522,
        k_threshold=22.627416997969522,
        rel_threshold=8.0,
        clip_eps=1e-6,
        value_norm_eps=1e-6,
        q_output_scale=None,
        kv_conv=KVShortConv(k_stream, v_stream, batch, SCALE),
    )
    out = gqa_prologue(
        qkvr[:, :1536].view(rows, 12, 128),
        qkvr[:, 1536:1792],
        qkvr[:, 1792:2048],
        norm=norm,
        rotary=None,
        cache=HeadKVCache(cache[0], cache[1], None, locations.long().cuda()),
        query_dtype=torch.bfloat16,
        # Decode (a verify block too) reads the K/V rows from the cache.
        return_kv=False,
        solution=None,
        override="triton_relative_gqa_prologue",
    )
    torch.cuda.synchronize()
    return tuple(x.cpu() for x in (out.q, out.rel_logits, cache[0], cache[1], ring))


def _run_head(rows, chain, g, spread):
    """One head-stage launch over random rows: ``(inputs, outputs)``. With
    ``spread`` the projection rows' magnitudes vary over ``[0.5, 4)``, so
    some clips cut and some do not."""
    plan = ip.local_layer_plan(rows, chain, "nvfp4", False, False).only(("head",))
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    qkvr = torch.randn(rows, ip.QKVR, generator=g)
    if spread:
        qkvr = qkvr * (0.5 + 3.5 * torch.rand(rows, ip.QKVR, generator=g))
    qkvr = qkvr.bfloat16()
    for r in range(rows):
        fill(scratch, plan, "qkvr", _words(qkvr[r]).cuda(), tag, row=r)
    weights = (
        (torch.rand(12, 128, generator=g) + 0.5).bfloat16(),  # q scale
        (torch.rand(12, 16, generator=g) + 0.5).bfloat16(),  # rel scale
        torch.randn(16, 1024, generator=g).bfloat16(),  # rel projection
        (torch.randn(256, 4, generator=g) / 2).bfloat16(),  # k taps
        (torch.randn(256, 4, generator=g) / 2).bfloat16(),  # v taps
    )
    k_cache = torch.zeros(CACHE_SLOTS, 2, 128, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    locations = torch.randperm(CACHE_SLOTS, generator=g)[:rows].int()
    requests = rows // chain
    ring = torch.randn(requests + 3, 11, 9728, generator=g).bfloat16()
    # Each request its own ring slot and length. The first request's oldest
    # taps fall before the sequence (zero taps).
    slots = (torch.randperm(requests + 2, generator=g)[:requests] + 1).tolist()
    positions = [2 + 9 * (r // chain) + r % chain for r in range(rows)]
    ring_gpu = ring.cuda()
    _head_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        tuple(t.cuda() for t in weights),
        (k_cache, v_cache, locations.cuda()),
        (
            ring_gpu,
            torch.tensor(slots, dtype=torch.int32, device="cuda"),
            torch.tensor(positions, dtype=torch.int64, device="cuda"),
        ),
        plan.key,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()

    def region(name, n):
        got = [payloads(scratch, plan, name, n // 2, tag, row=r) for r in range(rows)]
        return torch.stack(got).view(torch.bfloat16).cpu()

    inputs = (qkvr, weights, ring, slots, positions, locations)
    outputs = (
        region("q", 1536),
        region("rel", 12 * 1024),
        region("kv", 512).view(rows, 4, 128),
        k_cache.cpu(),
        v_cache.cpu(),
        ring_gpu.cpu(),
    )
    return inputs, outputs


@pytest.mark.parametrize("rows,chain", ROWS)
def test_head_prologue_is_the_regular_kernel(rows, chain):
    """Bit for bit the regular decode path's prologue, over enough launches
    that a different summation order or rounding shows."""
    g = torch.Generator().manual_seed(10 + rows)
    for _ in range(24):
        inputs, outputs = _run_head(rows, chain, g, True)
        qkvr, weights, ring, slots, positions, locations = inputs
        q, rel, _, k_cache, v_cache, ring_out = outputs
        reg_q, reg_rel, reg_k_cache, reg_v_cache, reg_ring = _regular_head(
            qkvr, weights, ring, slots, positions, locations, chain
        )
        assert torch.equal(q, reg_q.view(rows, -1))
        assert torch.equal(rel, reg_rel.view(rows, -1))
        assert torch.equal(k_cache, reg_k_cache)
        assert torch.equal(v_cache, reg_v_cache)
        assert torch.equal(ring_out, reg_ring)


@pytest.mark.parametrize("rows,chain", ROWS)
def test_head_prologue_matches_tokenspeed_math(rows, chain):
    g = torch.Generator().manual_seed(rows)
    inputs, outputs = _run_head(rows, chain, g, False)
    qkvr, weights, ring, slots, positions, locations = inputs
    q_scale, rel_scale, rel_proj, k_taps, v_taps = weights
    got_q, got_rel, kv, k_cache, v_cache, ring_out = outputs
    got_q, got_rel, kv = got_q.float(), got_rel.float(), kv.float()
    x = qkvr.float()
    # Q heads.
    q = x[:, :1536].view(rows, 12, 128) * q_scale.float()
    want_q = _clip(q, 22.627416997969522).bfloat16().float().view(rows, 1536)
    torch.testing.assert_close(got_q, want_q, rtol=2**-7, atol=1e-6)
    # Relative logits.
    rel = x[:, 2048:2240].view(rows, 12, 16) * rel_scale.float()
    rc = _clip(rel, 8.0).bfloat16().float()
    want_rel = (rc.double() @ rel_proj.double()).float().bfloat16().float()
    torch.testing.assert_close(got_rel, want_rel.view(rows, -1), rtol=2**-7, atol=1e-5)
    # K and V heads, through the short conv.
    kc = _short_conv(x[:, 1536:1792], k_taps.float(), ring, 0, slots, positions, chain)
    vc = _short_conv(
        x[:, 1792:2048], v_taps.float(), ring, 256, slots, positions, chain
    )
    want_k = _clip(kc.view(rows, 2, 128), 22.627416997969522).bfloat16().float()
    vc = vc.view(rows, 2, 128)
    want_v = (vc * torch.rsqrt((vc * vc).sum(-1, keepdim=True) / 128 + 1e-6)).bfloat16()
    torch.testing.assert_close(kv[:, :2], want_k, rtol=2**-7, atol=1e-6)
    torch.testing.assert_close(kv[:, 2:], want_v.float(), rtol=2**-7, atol=1e-6)
    torch.testing.assert_close(k_cache[locations].float(), kv[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(v_cache[locations].float(), kv[:, 2:], rtol=0, atol=0)
    # The ring rows of this launch's positions hold the raw k and v.
    for r in range(rows):
        got = ring_out[slots[r // chain], positions[r] % 11, :512]
        assert torch.equal(got, qkvr[r, 1536:2048])


@gluon.jit
def _mix_kernel(scratch, step, terms, cand_out, query, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "mix_attn")
    if active:
        row_terms = ()
        for t in gl.static_range(len(terms)):
            row_terms += (terms[t] + task * mix.HIDDEN,)
        mix.attention_mix_stage(
            mb.row(task),
            row_terms,
            cand_out + task * mix.HIDDEN,
            mix.Query(*query).row(task),
            "x_attn",
            "cand",
            "mix_attn",
        )


def _regular_mix(terms, w, gain, acc, m, s, store_candidate):
    """The regular decode path's mix (``attention_residual_combine``: one
    ``_attnres_combine_kernel`` launch) of one query over the set partials
    ``(m, s, acc)``: ``(out, cand)`` on the host, ``cand`` None unless
    ``store_candidate``."""
    rows = terms[0].shape[0]
    cand = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    out = attention_residual_combine(
        tuple(t.cuda() for t in terms),
        w.view(1, H).cuda(),
        (m.view(1, rows).cuda(), s.view(1, rows).cuda(), acc.view(1, rows, H).cuda()),
        candidate_out=cand if store_candidate else None,
        normalize_values=True,
        eps=1e-6,
        norm_weights=gain.view(1, H).cuda(),
        norm_eps=1e-6,
        output_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    return out[0].cpu(), cand.cpu() if store_candidate else None


def _mix_reference(cand, w, acc, m, s, gain):
    sq = (cand * cand).sum(-1, keepdim=True)
    r = torch.rsqrt(sq / H + 1e-6)
    logit = (cand * w).sum(-1, keepdim=True) * r
    m_new = torch.maximum(m, logit)
    c = torch.exp(m - m_new)
    e = torch.exp(logit - m_new)
    mixed = (acc * c + e * (cand * r)) / (s * c + e)
    rs = torch.rsqrt((mixed * mixed).sum(-1, keepdim=True) / H + 1e-6)
    return ((mixed * rs) * gain).bfloat16().float()


def _set_partials(rows, w, g):
    """FP32 set partials ``(acc, m, s)`` of two random completed blocks."""
    blocks = torch.randn(2, rows, H, generator=g)
    acc = torch.zeros(rows, H)
    m = torch.full((rows, 1), -float("inf"))
    s = torch.zeros(rows, 1)
    for src in blocks:
        r = torch.rsqrt((src * src).sum(-1, keepdim=True) / H + 1e-6)
        logit = (src * w).sum(-1, keepdim=True) * r
        m_new = torch.maximum(m, logit)
        c = torch.exp(m - m_new)
        e = torch.exp(logit - m_new)
        acc = acc * c + e * (src * r)
        s = s * c + e
        m = m_new
    return acc, m, s


def _run_attention_mix(rows, chain, terms, g):
    """One attention-mix launch over random rows: ``(inputs, (x, cand_out,
    cand))``."""
    plan = ip.local_layer_plan(rows, chain, "nvfp4", False, False).only(("mix_attn",))
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    pending = [torch.randn(rows, H, generator=g).bfloat16() for _ in range(terms)]
    w = torch.randn(H, generator=g) * H**-0.5
    gain = (torch.rand(H, generator=g) + 0.5).bfloat16()
    acc, m, s = _set_partials(rows, w, g)
    cand_out = torch.zeros(rows, H, dtype=torch.bfloat16, device="cuda")
    _mix_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        tuple(t.cuda() for t in pending),
        cand_out,
        (w.cuda(), gain.cuda(), acc.cuda(), m.view(-1).cuda(), s.view(-1).cuda()),
        plan.key,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()

    def region(name):
        got = [payloads(scratch, plan, name, H // 2, tag, row=r) for r in range(rows)]
        return torch.stack(got).view(torch.bfloat16).cpu()

    inputs = (pending, w, gain, acc, m, s)
    return inputs, (region("x_attn"), cand_out.cpu(), region("cand"))


@pytest.mark.parametrize(
    "rows,chain,terms",
    [(1, 1, 3), (8, 8, 3), (8, 8, 2), (16, 8, 3), (32, 8, 2), (64, 8, 3)],
)
def test_attention_mix_is_the_regular_kernel(rows, chain, terms):
    """Bit for bit the regular decode path's attention mix (it stores the
    rounded candidate), over enough launches that a different summation
    order or contraction shows."""
    g = torch.Generator().manual_seed(300 + rows + terms)
    for _ in range(16):
        (pending, w, gain, acc, m, s), (x, cand_out, cand) = _run_attention_mix(
            rows, chain, terms, g
        )
        reg_x, reg_cand = _regular_mix(pending, w, gain, acc, m, s, True)
        assert torch.equal(cand_out, reg_cand)
        assert torch.equal(cand, reg_cand)
        assert torch.equal(x, reg_x)


@pytest.mark.parametrize(
    "rows,chain,terms",
    [(1, 1, 3), (8, 8, 3), (8, 8, 2), (16, 8, 3), (32, 8, 2), (64, 8, 3)],
)
def test_attention_mix_matches_tokenspeed_math(rows, chain, terms):
    g = torch.Generator().manual_seed(100 + rows + terms)
    (pending, w, gain, acc, m, s), (x, cand_out, cand) = _run_attention_mix(
        rows, chain, terms, g
    )
    total = torch.zeros(rows, H)
    for t in pending:
        total = total + t.float()
    want_cand = total.bfloat16()
    assert torch.equal(cand_out, want_cand)
    assert torch.equal(cand, want_cand)
    want = _mix_reference(want_cand.float(), w, acc, m, s, gain.float())
    torch.testing.assert_close(x.float(), want, rtol=2**-7, atol=1e-6)


@gluon.jit
def _mlp_mix_kernel(scratch, step, query, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "mix_mlp")
    if active:
        mix.mlp_mix_stage(
            mb.row(task),
            mix.Query(*query).row(task),
            "cand",
            "attn_sc",
            "x_mlp",
            "set_acc",
            "set_ms",
            False,
            "mix_mlp",
        )


def _run_mlp_mix(rows, chain, g):
    """One MLP-mix launch over random rows: ``(inputs, x)``."""
    plan = ip.local_layer_plan(rows, chain, "nvfp4", False, False).only(("mix_mlp",))
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    p_term = torch.randn(rows, H, generator=g).bfloat16()
    a_term = torch.randn(rows, H, generator=g).bfloat16()
    for r in range(rows):
        fill(scratch, plan, "cand", _words(p_term[r]).cuda(), tag, row=r)
        fill(scratch, plan, "attn_sc", _words(a_term[r]).cuda(), tag, row=r)
    w = torch.randn(H, generator=g) * H**-0.5
    gain = (torch.rand(H, generator=g) + 0.5).bfloat16()
    acc = torch.randn(rows, H, generator=g)
    m = torch.randn(rows, 1, generator=g)
    s = torch.rand(rows, 1, generator=g) + 1.0
    _mlp_mix_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        (w.cuda(), gain.cuda(), acc.cuda(), m.view(-1).cuda(), s.view(-1).cuda()),
        plan.key,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    got = [payloads(scratch, plan, "x_mlp", H // 2, tag, row=r) for r in range(rows)]
    x = torch.stack(got).view(torch.bfloat16).cpu()
    return (p_term, a_term, w, gain, acc, m, s), x


@pytest.mark.parametrize("rows,chain", ROWS)
def test_mlp_mix_is_the_regular_kernel(rows, chain):
    """Bit for bit the regular decode path's MLP mix of the terms ``(P,
    A)``, over enough launches that a different summation order or
    contraction shows."""
    g = torch.Generator().manual_seed(400 + rows)
    for _ in range(16):
        (p_term, a_term, w, gain, acc, m, s), x = _run_mlp_mix(rows, chain, g)
        reg_x, _ = _regular_mix((p_term, a_term), w, gain, acc, m, s, False)
        assert torch.equal(x, reg_x)


@pytest.mark.parametrize("rows,chain", ROWS)
def test_mlp_mix_matches_tokenspeed_math(rows, chain):
    g = torch.Generator().manual_seed(200 + rows)
    (p_term, a_term, w, gain, acc, m, s), x = _run_mlp_mix(rows, chain, g)
    # cand = P + A in FP32, not rounded.
    cand = p_term.float() + a_term.float()
    want = _mix_reference(cand, w, acc, m, s, gain.float())
    torch.testing.assert_close(x.float(), want, rtol=2**-7, atol=1e-6)


FULL_SCALE = 1.0 / 3.0  # (4 * 2 + 1) ** -0.5


@gluon.jit
def _full_head_kernel(scratch, step, weights, cache, ring, tau, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    G: gl.constexpr = schedule.grid(PLAN)
    task, rounds = schedule.stage_rounds(PLAN, "head")
    for i in range(rounds):
        head.full_head_stage(
            mb,
            task + i * G,
            head.Weights(*weights),
            head.Cache(*cache),
            conv.Ring(ring[0], ring[1], ring[2], schedule.chain(PLAN)),
            tau,
            "qkvr",
            "kproj",
            "vproj",
            "q",
            "kv",
            "rel",
            "head",
        )


def _regular_full_head(
    qkvr, kproj, vproj, tau, weights, ring, slots, positions, locations, chain
):
    """The regular decode path's prologue of a full layer (dynamic K and V
    taps from their sources' projections, Q and the relative logits scaled
    by ``tau``)."""
    rows = qkvr.shape[0]
    requests = rows // chain
    q_scale, rel_scale, rel_proj, k_taps, v_taps = weights
    qkvr, kproj, vproj = qkvr.cuda(), kproj.cuda(), vproj.cuda()
    ring = ring.cuda()
    cache = torch.zeros(2, CACHE_SLOTS, 2, 128, dtype=torch.bfloat16, device="cuda")
    checkpoint = torch.zeros(requests + 1, 3, 512, dtype=torch.bfloat16).cuda()

    def i32(x):
        return torch.as_tensor(x, dtype=torch.int32).cuda()

    batch = ConvBatch(
        cu_seqlens=i32(list(range(0, rows + 1, chain))),
        seq_idx=i32([r // chain for r in range(rows)]),
        cache_indices=i32(slots),
        has_initial_state=torch.ones(requests, dtype=torch.bool, device="cuda"),
        seq_lens=i32([positions[(r + 1) * chain - 1] + 1 for r in range(requests)]),
        block_table=torch.zeros(requests, 3, dtype=torch.int32, device="cuda"),
        num_extends=0,
        publish_live=False,
    )
    k_stream, v_stream = (
        ConvStream(
            weight=taps.cuda(),
            dynamic_weights=proj[:, 256:264].view(rows, 2, 4),
            ring=ring[:, :, 256 * i : 256 * (i + 1)],
            checkpoint=checkpoint[:, :, 256 * i : 256 * (i + 1)],
        )
        for i, (taps, proj) in enumerate(((k_taps, kproj), (v_taps, vproj)))
    )
    norm = RelativeHeadNorm(
        relative=qkvr[:, 2048:2240].view(rows, 12, 16),
        projection=rel_proj.cuda(),
        q_scale=q_scale.cuda(),
        k_scale=None,
        rel_scale=rel_scale.cuda(),
        q_threshold=22.627416997969522,
        k_threshold=22.627416997969522,
        rel_threshold=8.0,
        clip_eps=1e-6,
        value_norm_eps=1e-6,
        q_output_scale=tau.cuda(),
        kv_conv=KVShortConv(k_stream, v_stream, batch, FULL_SCALE),
    )
    out = gqa_prologue(
        qkvr[:, :1536].view(rows, 12, 128),
        kproj[:, :256],
        vproj[:, :256],
        norm=norm,
        rotary=None,
        cache=HeadKVCache(cache[0], cache[1], None, locations.long().cuda()),
        query_dtype=torch.bfloat16,
        return_kv=False,
        solution=None,
        override="triton_relative_gqa_prologue",
    )
    torch.cuda.synchronize()
    return tuple(x.cpu() for x in (out.q, out.rel_logits, cache[0], cache[1], ring))


@pytest.mark.parametrize("rows,chain", ROWS)
def test_full_head_prologue_is_the_regular_kernel(rows, chain):
    """A full layer's prologue, bit for bit the regular decode path's: Q and
    the relative logits scaled by tau, K and V convolved with the taps their
    sources project."""
    g = torch.Generator().manual_seed(20 + rows)
    plan = ip.full_layer_plan(rows, chain, "nvfp4", ip.FULL_SPLITS).only(("head",))
    tag = launch_tag(STEP, 0)
    for launch in range(12):
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        spread = 0.5 + 3.5 * torch.rand(rows, ip.FULL_QKVR, generator=g)
        qkvr = (torch.randn(rows, ip.FULL_QKVR, generator=g) * spread).bfloat16()
        kproj = (torch.randn(rows, ip.KV_PROJ, generator=g) * 2).bfloat16()
        vproj = (torch.randn(rows, ip.KV_PROJ, generator=g) * 2).bfloat16()
        # tau is 1 below the scaling floor; past it, positions scale.
        tau = 1.0 + torch.rand(rows, generator=g) * (launch % 2)
        for r in range(rows):
            fill(scratch, plan, "qkvr", _words(qkvr[r]).cuda(), tag, row=r)
            fill(scratch, plan, "kproj", _words(kproj[r]).cuda(), tag, row=r)
            fill(scratch, plan, "vproj", _words(vproj[r]).cuda(), tag, row=r)
        weights = (
            (torch.rand(12, 128, generator=g) + 0.5).bfloat16(),
            (torch.rand(12, 16, generator=g) + 0.5).bfloat16(),
            torch.randn(16, 1024, generator=g).bfloat16(),
            (torch.randn(256, 4, generator=g) / 2).bfloat16(),
            (torch.randn(256, 4, generator=g) / 2).bfloat16(),
        )
        k_cache = torch.zeros(CACHE_SLOTS, 2, 128, dtype=torch.bfloat16, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        locations = torch.randperm(CACHE_SLOTS, generator=g)[:rows].int()
        requests = rows // chain
        ring = torch.randn(requests + 3, 11, 9728, generator=g).bfloat16()
        slots = (torch.randperm(requests + 2, generator=g)[:requests] + 1).tolist()
        positions = [2 + 9 * (r // chain) + r % chain for r in range(rows)]
        ring_gpu = ring.cuda()
        _full_head_kernel[(plan.grid,)](
            scratch,
            torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
            tuple(t.cuda() for t in weights),
            (k_cache, v_cache, locations.cuda()),
            (
                ring_gpu,
                torch.tensor(slots, dtype=torch.int32, device="cuda"),
                torch.tensor(positions, dtype=torch.int64, device="cuda"),
            ),
            tau.cuda(),
            plan.key,
            num_warps=NUM_WARPS,
            **extern_compile_options(),
        )
        torch.cuda.synchronize()

        def region(name, n):
            got = [
                payloads(scratch, plan, name, n // 2, tag, row=r) for r in range(rows)
            ]
            return torch.stack(got).view(torch.bfloat16).cpu()

        reg_q, reg_rel, reg_k, reg_v, reg_ring = _regular_full_head(
            qkvr, kproj, vproj, tau, weights, ring, slots, positions, locations, chain
        )
        assert torch.equal(region("q", 1536), reg_q.view(rows, -1)), launch
        assert torch.equal(region("rel", 12 * 1024), reg_rel.view(rows, -1)), launch
        assert torch.equal(k_cache.cpu(), reg_k), launch
        assert torch.equal(v_cache.cpu(), reg_v), launch
        assert torch.equal(ring_gpu.cpu(), reg_ring), launch


@gluon.jit
def _full_mix_kernel(scratch, step, terms, cand_out, queries, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "mix_attn")
    if active:
        row = task // 3
        j = task % 3
        row_terms = ()
        for t in gl.static_range(len(terms)):
            row_terms += (terms[t] + row * mix.HIDDEN,)
        if j == 0:
            q = mix.Query(*queries[0]).row(row)
        elif j == 1:
            q = mix.Query(*queries[1]).row(row)
        else:
            q = mix.Query(*queries[2]).row(row)
        mix.full_attention_mix_stage(
            mb.row(row),
            row_terms,
            cand_out + row * mix.HIDDEN,
            q,
            j,
            "cand",
            "mix_attn",
        )


@pytest.mark.parametrize(
    "rows,chain,terms",
    [(8, 8, 1), (8, 8, 3), (16, 8, 1), (32, 8, 3), (64, 8, 1), (64, 8, 3)],
)
def test_full_attention_mix_is_the_regular_kernel(rows, chain, terms):
    """A full layer's three-query mix (Q, K and V sources, each its own
    query and gain), bit for bit the regular ``attention_residual_combine``
    of the layer's 3 query rows; one pending term (the draft tap's
    ``residual_sum`` folded it) is the candidate as it is, -0.0 included."""
    g = torch.Generator().manual_seed(700 + rows + terms)
    plan = ip.full_layer_plan(rows, chain, "nvfp4", ip.FULL_SPLITS).only(("mix_attn",))
    tag = launch_tag(STEP, 0)
    for _ in range(8):
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        pending = [torch.randn(rows, H, generator=g).bfloat16() for _ in range(terms)]
        pending[0][0, :64] = -0.0
        ws = torch.randn(3, H, generator=g) * H**-0.5
        gains = (torch.rand(3, H, generator=g) + 0.5).bfloat16()
        parts = [_set_partials(rows, ws[j], g) for j in range(3)]
        cand_out = torch.zeros(rows, H, dtype=torch.bfloat16, device="cuda")
        queries = tuple(
            (
                ws[j].cuda(),
                gains[j].cuda(),
                parts[j][0].cuda(),
                parts[j][1].view(-1).cuda(),
                parts[j][2].view(-1).cuda(),
            )
            for j in range(3)
        )
        _full_mix_kernel[(plan.grid,)](
            scratch,
            torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
            tuple(t.cuda() for t in pending),
            cand_out,
            queries,
            plan.key,
            num_warps=NUM_WARPS,
            **extern_compile_options(),
        )
        torch.cuda.synchronize()
        cand = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
        want = attention_residual_combine(
            tuple(t.cuda() for t in pending),
            ws.cuda(),
            (
                torch.stack([p[1].view(-1) for p in parts]).cuda(),
                torch.stack([p[2].view(-1) for p in parts]).cuda(),
                torch.stack([p[0] for p in parts]).cuda(),
            ),
            candidate_out=cand if terms > 1 else None,
            normalize_values=True,
            eps=1e-6,
            norm_weights=gains.cuda(),
            norm_eps=1e-6,
            output_dtype=torch.bfloat16,
        ).cpu()
        for j, name in enumerate(("x_attn", "x_k", "x_v")):
            got = (
                torch.stack(
                    [
                        payloads(scratch, plan, name, H // 2, tag, row=r)
                        for r in range(rows)
                    ]
                )
                .view(torch.bfloat16)
                .cpu()
            )
            assert torch.equal(got, want[j]), name
        expected_cand = cand.cpu() if terms > 1 else pending[0]
        assert torch.equal(
            cand_out.cpu().view(torch.int16), expected_cand.view(torch.int16)
        )
