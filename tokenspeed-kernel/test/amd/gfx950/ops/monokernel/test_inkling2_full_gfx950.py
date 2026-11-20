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
"""A full attention layer's attention side (the three-query mix, the
three sources' projections, the head prologue, the attention and the merge)
as one launch on one GPU, against the regular decode path's kernels as it
runs them: ``attention_residual_combine``, the stacked FP32 projection,
the relative GQA prologue and the Triton split-KV decode."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import fill, words  # noqa: E402
from tokenspeed_kernel.ops.attention.prologue import gqa_prologue  # noqa: E402
from tokenspeed_kernel.ops.attention.prologue.types import (  # noqa: E402
    ConvBatch,
    ConvStream,
    HeadKVCache,
    KVShortConv,
    RelativeHeadNorm,
)
from tokenspeed_kernel.ops.attention.rmha import (  # noqa: E402
    rel_mha_decode_with_kvcache,
)
from tokenspeed_kernel.ops.gemm import mm  # noqa: E402
from tokenspeed_kernel.ops.model.inkling2.monokernel import (  # noqa: E402
    inkling2_full_layer_splits,
)
from tokenspeed_kernel.ops.residual.attention import (  # noqa: E402
    attention_residual_combine,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    attention,
    conv,
    head,
    mix,
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
STEP = 7
PAGE = 128  # the served KV page: the attention's bits need it
MAX_CONTEXT = 262120  # the served context length
# The attention side's stages, through the merge (the projections' K split
# groups included).
SIDE = ("mix_attn", "qkvr", "kproj", "vproj", "head", "attn", "merge")


def _side(plan):
    return tuple(s.name for s in plan.stages if s.name.split("_k")[0] in SIDE)


@gluon.jit
def _attention_side(
    scratch,
    step,
    terms,
    cand_out,
    queries,
    qkvr_w,
    kv_w,
    tau,
    weights,
    head_cache,
    attn_cache,
    ring,
    PLAN: gl.constexpr,
    SPLITS: gl.constexpr,
    SHARED: gl.constexpr,
    RING_ROWS: gl.constexpr,
):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    G: gl.constexpr = schedule.grid(PLAN)
    D: gl.constexpr = schedule.dims(PLAN)
    HIDDEN: gl.constexpr = D.hidden
    rings = conv.Ring(
        ring[0], ring[1], ring[2], schedule.chain(PLAN), RING_ROWS, D.ring_width
    )
    task, active = schedule.stage_task(PLAN, "mix_attn")
    if active:
        row = task // 3
        j = task % 3
        row_terms = ()
        for t in gl.static_range(len(terms)):
            row_terms += (terms[t] + row * HIDDEN,)
        if j == 0:
            q = mix.Query(*queries[0]).row_of(row, HIDDEN)
        elif j == 1:
            q = mix.Query(*queries[1]).row_of(row, HIDDEN)
        else:
            q = mix.Query(*queries[2]).row_of(row, HIDDEN)
        mix.full_attention_mix_stage(
            mb.row(row),
            row_terms,
            cand_out + row * HIDDEN,
            q,
            j,
            "cand",
            "mix_attn",
        )
    projection.chained_stages(
        PLAN, mb, qkvr_w, "x_attn", "qkvr", "qkvr", HIDDEN, D.qkvr_splits, False
    )
    projection.chained_stages(
        PLAN, mb, kv_w[0], "x_k", "kproj", "kproj", HIDDEN, D.qkvr_splits, False
    )
    projection.chained_stages(
        PLAN, mb, kv_w[1], "x_v", "vproj", "vproj", HIDDEN, D.qkvr_splits, False
    )
    task, rounds = schedule.stage_rounds(PLAN, "head")
    for i in range(rounds):
        head.full_head_stage(
            mb,
            task + i * G,
            head.Weights(*weights),
            head.Cache(*head_cache),
            rings,
            tau,
            "qkvr",
            "kproj",
            "vproj",
            "q",
            "kv",
            "rel",
            "head",
        )
    c = attention.Cache(*attn_cache)
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
            SPLITS,
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
            SPLITS,
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
            SPLITS,
            "merge",
        )


def _set_partials(rows, w, g):
    """FP32 set partials ``(acc, m, s)`` of two random completed blocks of
    ``w``'s width."""
    H = w.shape[0]
    acc = torch.zeros(rows, H)
    m = torch.full((rows,), -float("inf"))
    s = torch.zeros(rows)
    for _ in range(2):
        src = torch.randn(rows, H, generator=g)
        r = torch.rsqrt((src * src).sum(-1) / H + 1e-6)
        logit = (src * w).sum(-1) * r
        m_new = torch.maximum(m, logit)
        c = torch.exp(m - m_new)
        e = torch.exp(logit - m_new)
        acc = acc * c[:, None] + e[:, None] * (src * r[:, None])
        s = s * c + e
        m = m_new
    return acc, m, s


def _kv_tiles(dims, qkvr):
    """The K and V sources' projection tiles, as the op packs them: the
    source's values (K's or V's rows) and their conv tap rows."""
    kv, taps = dims.kv_dim, dims.kv_proj - dims.kv_dim
    return tuple(
        bf16_tiles(
            torch.nn.functional.pad(
                torch.cat(
                    (
                        qkvr[dims.k_off + kv * i : dims.k_off + kv * (i + 1)],
                        qkvr[dims.taps_off + taps * i : dims.taps_off + taps * (i + 1)],
                    )
                ),
                (0, 0, 0, dims.kv_proj_padded - dims.kv_proj),
            )
        )
        for i in range(2)
    )


def _ring_rows(chain):
    """Conv ring rows a slot: 3 taps back past a verify block's tokens."""
    return 3 + chain


# The cases at every geometry (8-token verify blocks), and the half-width
# geometry's checkpoint's 4-token verify blocks.
_CASES = [
    (1, (9000,)),
    (3, (300,)),
    (1, (40, 70000)),
    (3, (1013, 4096, 20000, 77)),
    (1, (40, 300, 1013, 1500, 4096, 77, 9000, 2048)),
]
_HALF_CASES = [
    (3, (1013,)),
    (1, (40, 70000)),
    (3, (1013, 4096, 20000, 77)),
    (1, (40, 300, 1013, 1500, 4096, 77, 9000, 2048)),
]


@pytest.mark.parametrize(
    "dims,chain,terms,firsts",
    [
        pytest.param(d, 8, *case, id=f"{d.hidden}-c8-{case}")
        for d in ip.GEOMETRIES
        for case in _CASES
    ]
    + [
        pytest.param(
            ip.GEOMETRIES[1], 4, *case, id=f"{ip.GEOMETRIES[1].hidden}-c4-{case}"
        )
        for case in _HALF_CASES
    ],
)
def test_full_attention_side_is_the_regular_path(dims, chain, terms, firsts):
    requests = len(firsts)
    rows = requests * chain
    H = dims.hidden
    hq, hkv, kvd = dims.q_heads, dims.kv_heads, dims.kv_dim
    taps = dims.kv_proj - kvd  # a source's conv tap rows
    splits = inkling2_full_layer_splits(MAX_CONTEXT)
    plan = ip.full_layer_plan(dims, rows, chain, "nvfp4", splits)
    plan = plan.only(_side(plan))
    tag = launch_tag(STEP, 0)
    g = torch.Generator().manual_seed(rows + terms)
    dev = "cuda"
    pending = tuple(torch.randn(rows, H, generator=g).bfloat16() for _ in range(terms))
    ws = torch.randn(3, H, generator=g) * H**-0.5
    gains = (torch.rand(3, H, generator=g) + 0.5).bfloat16()
    parts = [_set_partials(rows, ws[j], g) for j in range(3)]
    qkvr = (torch.randn(dims.full_qkvr, H, generator=g) / 32).bfloat16()
    tau = 1.0 + torch.rand(rows, generator=g) * 0.25
    weights = (
        (torch.rand(hq, 128, generator=g) + 0.5).bfloat16(),
        (torch.rand(hq, 16, generator=g) + 0.5).bfloat16(),
        torch.randn(16, 1024, generator=g).bfloat16(),
        (torch.randn(kvd, 4, generator=g) / 2).bfloat16(),
        (torch.randn(kvd, 4, generator=g) / 2).bfloat16(),
    )
    totals = [first + chain for first in firsts]
    pages = (max(totals) + PAGE - 1) // PAGE
    table = (
        (torch.randperm(2 * requests * pages, generator=g)[: requests * pages] + 1)
        .int()
        .view(requests, pages)
    )
    shape = (2 * requests * pages + 1, PAGE, hkv, 128)
    k_cache = torch.randn(shape, generator=g).bfloat16()
    v_cache = torch.randn(shape, generator=g).bfloat16()
    positions = [firsts[r // chain] + r % chain for r in range(rows)]
    locations = torch.tensor(
        [
            int(table[r // chain, p // PAGE]) * PAGE + p % PAGE
            for r, p in enumerate(positions)
        ],
        dtype=torch.int32,
    )
    ring = torch.randn(
        requests + 3, _ring_rows(chain), dims.ring_width, generator=g
    ).bfloat16()
    slots = (torch.randperm(requests + 2, generator=g)[:requests] + 1).tolist()

    # The monokernel.
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device=dev)
    cand_out = torch.zeros(rows, H, dtype=torch.bfloat16, device=dev)
    mk_k, mk_v, mk_ring = k_cache.cuda(), v_cache.cuda(), ring.cuda()
    pos = torch.tensor(positions, dtype=torch.int64, device=dev)
    queries = tuple(
        (
            ws[j].cuda(),
            gains[j].cuda(),
            parts[j][0].cuda(),
            parts[j][1].cuda(),
            parts[j][2].cuda(),
        )
        for j in range(3)
    )
    _attention_side[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device=dev),
        tuple(t.cuda() for t in pending),
        cand_out,
        queries,
        bf16_tiles(
            torch.nn.functional.pad(qkvr, (0, 0, 0, dims.qkvr_padded - dims.full_qkvr))
        ).cuda(),
        tuple(t.cuda() for t in _kv_tiles(dims, qkvr)),
        tau.cuda(),
        tuple(t.cuda() for t in weights),
        (mk_k.view(-1, hkv, 128), mk_v.view(-1, hkv, 128), locations.cuda()),
        (
            mk_k.view(-1, hkv, 128),
            mk_v.view(-1, hkv, 128),
            pos,
            table.cuda(),
            pages,
            PAGE,
        ),
        (mk_ring, torch.tensor(slots, dtype=torch.int32, device=dev), pos),
        plan.key,
        splits,
        ip.shared_attention(rows, chain),
        _ring_rows(chain),
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()

    # The regular path.
    cand = torch.empty(rows, H, dtype=torch.bfloat16, device=dev)
    x = attention_residual_combine(
        tuple(t.cuda() for t in pending),
        ws.cuda(),
        (
            torch.stack([p[1] for p in parts]).cuda(),
            torch.stack([p[2] for p in parts]).cuda(),
            torch.stack([p[0] for p in parts]).cuda(),
        ),
        candidate_out=cand if terms > 1 else None,
        normalize_values=True,
        eps=1e-6,
        norm_weights=gains.cuda(),
        norm_eps=1e-6,
        output_dtype=torch.bfloat16,
    )
    # One stacked GEMM over the three sources, rounded once.
    proj = mm(x.reshape(3 * rows, H), qkvr.cuda(), out_dtype=torch.float32).bfloat16()
    q_src, k_src, v_src = proj.view(3, rows, dims.full_qkvr).unbind(0)

    def i32(v):
        return torch.as_tensor(v, dtype=torch.int32).cuda()

    reg_k, reg_v, reg_ring = k_cache.cuda(), v_cache.cuda(), ring.cuda()
    checkpoint = torch.zeros(requests + 1, 3, 2 * kvd, dtype=torch.bfloat16).cuda()
    batch = ConvBatch(
        cu_seqlens=i32(list(range(0, rows + 1, chain))),
        seq_idx=i32([r // chain for r in range(rows)]),
        cache_indices=i32(slots),
        has_initial_state=torch.ones(requests, dtype=torch.bool, device=dev),
        seq_lens=i32(totals),
        block_table=torch.zeros(requests, 3, dtype=torch.int32, device=dev),
        num_extends=0,
        publish_live=False,
    )
    streams = (
        ConvStream(
            weight=static.cuda(),
            dynamic_weights=src[
                :, dims.taps_off + 4 * hkv * i : dims.taps_off + 4 * hkv * (i + 1)
            ].view(rows, hkv, 4),
            ring=reg_ring[:, :, kvd * i : kvd * (i + 1)],
            checkpoint=checkpoint[:, :, kvd * i : kvd * (i + 1)],
        )
        for i, (static, src) in enumerate(((weights[3], k_src), (weights[4], v_src)))
    )
    norm = RelativeHeadNorm(
        relative=q_src[:, dims.rel_off : dims.gate_off].view(rows, hq, 16),
        projection=weights[2].cuda(),
        q_scale=weights[0].cuda(),
        k_scale=None,
        rel_scale=weights[1].cuda(),
        q_threshold=22.627416997969522,
        k_threshold=22.627416997969522,
        rel_threshold=8.0,
        clip_eps=1e-6,
        value_norm_eps=1e-6,
        q_output_scale=tau.cuda(),
        kv_conv=KVShortConv(*streams, batch, 1.0 / 3.0),
    )
    pro = gqa_prologue(
        q_src[:, : dims.q_dim].view(rows, hq, 128),
        k_src[:, dims.k_off : dims.v_off],
        v_src[:, dims.v_off : dims.rel_off],
        norm=norm,
        rotary=None,
        cache=HeadKVCache(
            reg_k.view(-1, hkv, 128),
            reg_v.view(-1, hkv, 128),
            None,
            locations.long().cuda(),
        ),
        query_dtype=torch.bfloat16,
        return_kv=False,
        solution=None,
        override="triton_relative_gqa_prologue",
    )
    want = rel_mha_decode_with_kvcache(
        q=pro.q.view(rows, hq, 128),
        k_cache=reg_k,
        v_cache=reg_v,
        page_table=table.cuda(),
        cache_seqlens=i32(totals),
        max_seqlen_k=MAX_CONTEXT,
        rel_logits=pro.rel_logits.view(rows, hq, 1024),
        cu_seqlens_q=torch.arange(0, rows + 1, chain, dtype=torch.int32, device=dev),
        max_seqlen_q=chain,
        window_left=-1,
        softmax_scale=128**-0.5,
        solution="triton",
        output_dtype=torch.bfloat16,
        return_lse=False,
        gate=q_src[:, dims.gate_off : dims.sink_off],
        sink=q_src[:, dims.sink_off : dims.sink_off + hq],
    )
    torch.cuda.synchronize()
    got = torch.stack(
        [words(scratch, plan, "o", dims.q_dim // 2, row=r) for r in range(rows)]
    ).view(torch.bfloat16)
    mismatch = (got != want.view(rows, -1)).sum().item()
    assert mismatch == 0, f"{mismatch} of {got.numel()} gated outputs differ"
    assert torch.equal(mk_k, reg_k) and torch.equal(mk_v, reg_v)
    assert torch.equal(mk_ring, reg_ring)
    expected_cand = cand if terms > 1 else pending[0].cuda()
    assert torch.equal(cand_out.view(torch.int16), expected_cand.view(torch.int16))
