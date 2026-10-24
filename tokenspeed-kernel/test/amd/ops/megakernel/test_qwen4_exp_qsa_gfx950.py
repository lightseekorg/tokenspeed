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

"""Qwen3.8-Flash-Next QSA attention stages of the decode megakernel."""

from __future__ import annotations

import pytest
import torch
from qwen4_exp_reference import bf as _bf
from qwen4_exp_reference import cos_sin as _cos_sin
from qwen4_exp_reference import gemma as _gemma
from qwen4_exp_reference import index_norm as _index_norm
from qwen4_exp_reference import rotate
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from megakernel_testing import (  # noqa: E402
    fill,
    payloads,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.megakernel import (  # noqa: E402
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import (  # noqa: E402
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import qsa  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.layer import (  # noqa: E402
    qsa_plan,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import (  # noqa: E402
    bf16_tiles,
)

HIDDEN, HEADS, HEAD, IDX = 2560, 24, 256, 128
PAGE = 256
STEP = 7
TAG = launch_tag(STEP, 0)
PLAN = qsa_plan(False, 1, 1, "mxfp4", False).only(
    ("q1", "q2", "q3", "q4", "q4f", "q5", "q6")
)


@gluon.jit
def _qsa_kernel(scratch, step, qsa_weights, cache, trace_buf, PLAN: gl.constexpr):
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table)
    weights = qsa.Weights(*qsa_weights)
    row = qsa.Cache(*cache)
    task, active = schedule.stage_task(PLAN, "q1")
    if active:
        qsa.projection_stage(mb, task, weights, "a_", "q1")
    task, active = schedule.stage_task(PLAN, "q2")
    if active:
        qsa.prologue_stage(mb, task, 0, weights, row, "q2")
    task, active = schedule.stage_task(PLAN, "q3")
    if active:
        qsa.score_stage(mb, task, 0, row, "q3")
    task, active = schedule.stage_task(PLAN, "q4")
    if active:
        qsa.topk_stage(mb, task, row.length(), "q4")
    task, active = schedule.stage_task(PLAN, "q4f")
    if active:
        qsa.topk_final_stage(mb, task, row.length(), "q4f")
    task, active = schedule.stage_task(PLAN, "q5")
    if active:
        qsa.attention_stage(mb, task, 0, row, 1, "q5")
    task, active = schedule.stage_task(PLAN, "q6")
    if active:
        qsa.merge_stage(mb, task, "q6")
    trace.flush(table, trace_buf)


@pytest.mark.parametrize("L", [999, 2601, 2603])
def test_qsa_stages_match_tokenspeed_math(L):
    g = torch.Generator().manual_seed(L)
    dev = "cuda"
    mixed = (torch.randn(HIDDEN, generator=g) * 0.5).bfloat16()
    # qkv_proj, then index_qk_proj.
    proj_w = (torch.randn(13952, HIDDEN, generator=g) * 0.02).bfloat16().cuda()
    q_norm_w = (torch.randn(HEAD, generator=g) * 0.1).bfloat16().cuda()
    k_norm_w = (torch.randn(HEAD, generator=g) * 0.1).bfloat16().cuda()
    iq_w = (1 + 0.1 * torch.randn(IDX, generator=g)).bfloat16().cuda()
    ik_w = (1 + 0.1 * torch.randn(IDX, generator=g)).bfloat16().cuda()
    max_pos = 4096
    cos_sin = _cos_sin(max_pos)
    positions = torch.full((3, 1), L, dtype=torch.int64, device=dev)
    seq_len = torch.tensor([L + 1], dtype=torch.int32, device=dev)

    # Page tables: shuffled pages, page 0 unused (TokenSpeed's dummy page).
    n_pages = (max_pos + PAGE - 1) // PAGE
    full_table = (torch.randperm(n_pages, generator=g) + 1).int().cuda()
    slots = (n_pages + 1) * PAGE
    k_cache = (torch.randn(slots, 2, HEAD, generator=g)).bfloat16().cuda()
    v_cache = (torch.randn(slots, 2, HEAD, generator=g)).bfloat16().cuda()
    q_pages = (max_pos + 255) // 256
    qsa_table = (torch.randperm(q_pages, generator=g) + 1).int().cuda()
    compressed = (
        (torch.randn(q_pages + 1, 64, 1, IDX, generator=g) * 0.3).bfloat16().cuda()
    )
    r_pages = max_pos // 64
    recent_table = (torch.randperm(r_pages, generator=g) + 1).int().cuda()
    raw_key = (torch.randn(r_pages + 1, 4, 1, IDX, generator=g)).bfloat16().cuda()
    rope_position = torch.zeros(r_pages + 1, 3, dtype=torch.int64, device=dev)
    ring = int(recent_table[L // 64])
    rope_position[ring] = (L // 4) * 4
    raw_before = raw_key.clone()

    plan = PLAN
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device=dev)
    fill(scratch, PLAN, "a_xbf", mixed.view(torch.int32).cuda(), TAG)
    _qsa_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device=dev),
        (
            bf16_tiles(proj_w[:13312]),
            bf16_tiles(proj_w[13312:]),
            q_norm_w,
            k_norm_w,
            iq_w,
            ik_w,
        ),
        # One row: nothing uses the per-row strides.
        (
            cos_sin,
            positions,
            positions.stride(0),
            seq_len,
            full_table,
            0,
            PAGE,
            k_cache,
            v_cache,
            k_cache.stride(0),
            qsa_table,
            0,
            recent_table,
            0,
            compressed,
            compressed.stride(0),
            raw_key,
            raw_key.stride(0),
            rope_position,
            rope_position.stride(0),
            # No verify staging, draft scratch or accepted lengths: decode.
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,  # seq_offset: one row
        ),
        None,
        plan.key,
        num_warps=8,
    )
    torch.cuda.synchronize()

    # Projection.
    proj = _bf(proj_w.float() @ mixed.cuda().float())
    got = payloads(scratch, PLAN, "qp", 6976, TAG).view(torch.bfloat16).float()
    torch.testing.assert_close(got, proj, rtol=1e-2, atol=1e-2)
    proj = got  # downstream stages run on the kernel's own projections

    slot = lambda t: int(full_table[t // PAGE]) * PAGE + t % PAGE  # noqa: E731
    q = torch.stack(
        [
            _bf(rotate(_gemma(proj[h * 512 : h * 512 + 256], q_norm_w), cos_sin, L))
            for h in range(HEADS)
        ]
    )
    k_new = torch.stack(
        [
            _bf(
                rotate(
                    _gemma(proj[12288 + kh * 256 : 12544 + kh * 256], k_norm_w),
                    cos_sin,
                    L,
                )
            )
            for kh in range(2)
        ]
    )
    v_new = proj[12800:13312].view(2, HEAD)
    torch.testing.assert_close(k_cache[slot(L)].float(), k_new, rtol=1e-2, atol=1e-2)
    assert torch.equal(v_cache[slot(L)].float(), v_new)
    got_q = (
        payloads(scratch, PLAN, "qn", 3072, TAG)
        .view(torch.bfloat16)
        .float()
        .view(HEADS, HEAD)
    )
    torch.testing.assert_close(got_q, q, rtol=1e-2, atol=1e-2)

    # Indexer: ring write, compression.
    k_raw = proj[13824:13952]
    assert torch.equal(raw_key[ring, L % 4, 0].float(), k_raw)
    complete = (L + 1) // 4
    comp_ref = compressed.clone()
    if (L + 1) % 4 == 0:
        members = [raw_before[ring, (L - o) % 4, 0].float() for o in (3, 2, 1)] + [
            k_raw
        ]
        pooled = (((members[0] + members[1]) + members[2]) + members[3]) / 4.0
        start = (L // 4) * 4
        ck = _bf(rotate(_index_norm(pooled, ik_w), cos_sin, start))
        loc = int(qsa_table[L // 256]) * 256 + L % 256
        row = (loc // 256) * 64 + (loc % 256) // 4
        torch.testing.assert_close(
            compressed[row // 64, row % 64, 0].float(), ck, rtol=1e-2, atol=1e-2
        )
        comp_ref = compressed

    # Attended tokens.
    if complete > 512:
        iq = (
            payloads(scratch, PLAN, "iq", 256, TAG)
            .view(torch.bfloat16)
            .float()
            .view(4, IDX)
        )
        rows = torch.tensor(
            [int(qsa_table[b // 64]) * 64 + b % 64 for b in range(complete)], device=dev
        )
        keys = comp_ref.view(-1, IDX)[rows].float()
        score = torch.relu(keys @ iq.T).sum(-1)
        got_score = payloads(scratch, PLAN, "isc", complete, TAG).view(torch.float32)
        torch.testing.assert_close(got_score, score, rtol=1e-4, atol=1e-4)
        # Rank on the kernel's own scores: ties go to the lower block.
        order = sorted(range(complete), key=lambda b: (-float(got_score[b]), b))
        sel = sorted(order[:512])
        assert payloads(scratch, PLAN, "sel", 512, TAG).tolist() == sel
        tokens = [4 * b + i for b in sel for i in range(4)]
    else:
        tokens = list(range(4 * complete))
    tokens += list(range(4 * complete, L + 1))
    kt = torch.stack([k_cache[slot(t)].float() for t in tokens])
    vt = torch.stack([v_cache[slot(t)].float() for t in tokens])
    outs = []
    for h in range(HEADS):
        kh = h // 12
        s = (kt[:, kh] @ q[h]) / 16.0
        p = torch.softmax(s, 0)
        a = _bf(p @ vt[:, kh])
        gate = proj[h * 512 + 256 : h * 512 + 512]
        outs.append(_bf(a * torch.sigmoid(gate)))
    ref = torch.cat(outs)
    obf = payloads(scratch, PLAN, "obf", 3072, TAG).view(torch.bfloat16)
    torch.testing.assert_close(obf.float(), ref, rtol=1e-2, atol=2e-2)


TOPK_PLAN = qsa_plan(False, 1, 1, "mxfp4", False).only(("q4", "q4f"))


@gluon.jit
def _topk_kernel(scratch, step, seq_len, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "q4")
    if active:
        qsa.topk_stage(mb, task, gl.load(seq_len), "q4")
    task, active = schedule.stage_task(PLAN, "q4f")
    if active:
        qsa.topk_final_stage(mb, task, gl.load(seq_len), "q4f")


# One chunk, exactly one chunk, a second chunk of one block, and many chunks.
@pytest.mark.parametrize("blocks", [600, 8192, 8193, 20000, 65536])
def test_topk_selects_tokenspeeds_blocks(blocks):
    """The best 512 blocks (ties: the lower block) in ascending order."""
    g = torch.Generator().manual_seed(blocks)
    # Few distinct scores, so ties cross chunk boundaries.
    scores = torch.randint(0, 40, (blocks,), generator=g).float() * 0.25
    scratch = torch.zeros(TOPK_PLAN.scratch_pairs, dtype=torch.int64, device="cuda")
    start = TOPK_PLAN.offset("isc")
    words = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    scratch[start : start + blocks] = (launch_tag(STEP, 0) << 32) | words.cuda()
    seq_len = torch.tensor([blocks * 4 + 1], dtype=torch.int32, device="cuda")
    step = torch.full((1,), STEP, dtype=torch.int32, device="cuda")
    _topk_kernel[(TOPK_PLAN.grid,)](scratch, step, seq_len, TOPK_PLAN.key, num_warps=8)
    torch.cuda.synchronize()
    order = sorted(range(blocks), key=lambda b: (-scores[b].item(), b))
    pairs = scratch[TOPK_PLAN.offset("sel") : TOPK_PLAN.offset("sel") + 512].cpu()
    assert torch.all((pairs >> 32) == launch_tag(STEP, 0))
    assert (pairs & 0xFFFFFFFF).tolist() == sorted(order[:512])
