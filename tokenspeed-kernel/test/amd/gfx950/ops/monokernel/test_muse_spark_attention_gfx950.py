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

"""Muse Spark attention stages (head, attn, merge) of one rank, for one
decode row or the chained rows of a speculative verify."""

from __future__ import annotations

import muse_spark_reference as ref
import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import fill, payloads  # noqa: E402
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
    attention,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    qkvg as qkvg_stage,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.layer import (  # noqa: E402
    FULL_SPLITS,
    SLIDING_SPLITS,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

HEAD, HEADS, KV_HEADS, GROUP, PAGE = 64, 16, 2, 8, 64
QKVG = 2304
K_ROW, V_ROW, GATE_ROW = 1024, 1152, 1280


@gluon.jit
def attention_kernel(
    scratch,
    step,
    seq_len,
    slot,
    table,
    table_stride,
    k_cache,
    v_cache,
    cos_sin,
    proj,
    PLAN: gl.constexpr,
    SPLITS: gl.constexpr,
    WINDOW: gl.constexpr,
    ROPE: gl.constexpr,
):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "head")
    if active:
        # The launch projects its own q | k | v (the qkvg region): proj is
        # unused.
        attention.head_stage(
            mb, task, seq_len, slot, k_cache, v_cache, cos_sin, proj, ROPE, True, "head"
        )
    G: gl.constexpr = schedule.grid(PLAN)
    attn_task, attn_rounds = schedule.stage_rounds(PLAN, "attn")
    for i in range(attn_rounds):
        attention.attn_stage(
            mb,
            attn_task + i * G,
            seq_len,
            table,
            table_stride,
            k_cache,
            v_cache,
            cos_sin,
            proj,
            ROPE,
            True,
            SPLITS,
            WINDOW,
            "attn",
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        attention.merge_stage(mb, task, SPLITS, "merge")


def _plan(splits: int, rows: int, chain: int) -> Plan:
    return Plan(
        grid=256,
        stages=attention.stages(
            {"head": 0, "attn": 32, "merge": 18}, splits, rows, chain, True
        ),
        regions=qkvg_stage.REGIONS + attention.regions(splits),
        rows=rows,
        chain=chain,
        external=("qkvg", "gate"),
    )


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("requests", [1, 2])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize(
    "length,window,splits",
    [
        (6, 2048, SLIDING_SPLITS),
        (3000, 2048, SLIDING_SPLITS),
        # Fewer keys than one chunk: only split 0 has cached keys, and the
        # last split, which takes the launch's tokens, has none.
        (6, 0, FULL_SPLITS),
        (3000, 0, FULL_SPLITS),
        (9000, 0, FULL_SPLITS),
    ],
)
def test_attention_matches_the_regular_path_math(
    length, window, splits, rows, requests, kv_dtype
):
    """``requests`` requests of ``rows`` tokens each in one launch: each
    attends its own cache (its length and page table) and takes its own
    tokens' keys from the projection. An FP8 cache holds the regular path's
    unit-scale e4m3 bytes: the rows' K and V round once to it."""
    _check_attention(length, window, splits, rows, requests, kv_dtype, page_base=0)


# Long contexts, once each: the served shape (one request's verify on the
# FP8 cache, the key groups) at a long agentic context (~14 key chunks a
# split), a single decode row there (the splits without key groups), and two
# requests over several key chunks a split.
@pytest.mark.parametrize(
    "length,rows,requests,kv_dtype",
    [
        (230000, 8, 1, torch.float8_e4m3fn),
        (230000, 1, 1, torch.float8_e4m3fn),
        (50000, 8, 2, torch.bfloat16),
    ],
)
def test_long_context_attention_matches_the_regular_path_math(
    length, rows, requests, kv_dtype
):
    _check_attention(length, 0, FULL_SPLITS, rows, requests, kv_dtype, page_base=0)


# The first page whose first element lies past 2^31 elements of a cache.
FAR_PAGE = (1 << 31) // (PAGE * KV_HEADS * HEAD) + 1


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "length,window,splits",
    [(3000, 2048, SLIDING_SPLITS), (9000, 0, FULL_SPLITS)],
)
def test_attention_reaches_pages_past_2_31_cache_elements(
    length, window, splits, kv_dtype
):
    """Pages and slots at the far end of a large cache: every page of the
    launch starts past 2^31 elements, so a page or slot offset computed in
    int32 reads and writes the wrong place. Only the far end is filled."""
    _check_attention(length, window, splits, 8, 2, kv_dtype, page_base=FAR_PAGE)


def _check_attention(length, window, splits, rows, requests, kv_dtype, page_base):
    """One launch of ``requests`` requests of ``rows`` tokens against the
    regular path's math; the cache's pages are numbered from ``page_base``
    (the pages below it are allocated and never touched)."""
    rope = window != 0
    g = torch.Generator().manual_seed(length + window)
    # The second request is longer and has pages of its own.
    lengths = [length + 37 * b for b in range(requests)]
    ends = [n + rows - 1 for n in lengths]
    used = [(e + PAGE - 1) // PAGE for e in ends]
    pages = sum(used) + 3
    perm = torch.randperm(pages, generator=g) + 1
    tables, at = [], 0
    for u in used:
        tables.append(perm[at : at + u].to(torch.int32))
        at += u
    width = max(used)
    table = torch.zeros(requests, width, dtype=torch.int32)
    for b, t in enumerate(tables):
        table[b, : len(t)] = t
    k_cache = torch.randn(pages + 2, PAGE, KV_HEADS, HEAD, generator=g).to(kv_dtype)
    v_cache = torch.randn(pages + 2, PAGE, KV_HEADS, HEAD, generator=g).to(kv_dtype)
    R = requests * rows
    qkvg = torch.randn(R, QKVG, generator=g).to(torch.bfloat16)
    cs = ref.rope_table(max(ends) + 1)
    positions = torch.cat([torch.arange(n - 1, n - 1 + rows) for n in lengths])
    slots = torch.cat(
        [
            tables[b][positions[b * rows : (b + 1) * rows] // PAGE].long() * PAGE
            + positions[b * rows : (b + 1) * rows] % PAGE
            for b in range(requests)
        ]
    )

    plan = _plan(splits, R, rows)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    step = 5
    tag = launch_tag(step, 0)
    # The projection's two K halves: each half of the value (exact), so
    # their FP32 sum is the BF16 value.
    for row in range(R):
        for region, part in (
            ("qkvg", qkvg[row, :GATE_ROW]),
            ("gate", qkvg[row, GATE_ROW:]),
        ):
            half = (part.float() / 2).view(torch.int32)
            fill(scratch, plan, region, torch.cat([half, half]).cuda(), tag, row=row)
    kc, vc = (_cache_from(c, page_base) for c in (k_cache, v_cache))
    attention_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), step, dtype=torch.int32, device="cuda"),
        (positions + 1).to(torch.int32).cuda(),
        (slots + page_base * PAGE).to(torch.int32).cuda(),
        (table + page_base * (table != 0)).to(torch.int32).cuda(),
        width,
        kc,
        vc,
        cs.cuda(),
        scratch.view(torch.int32),
        plan.key,
        splits,
        window,
        rope,
        num_warps=8,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()

    for b in range(requests):
        # Reference: the regular path's math with its BF16 rounding points.
        # The cache slots of the launch's tokens hold stale values: the
        # kernel must take those keys from the projection.
        z, k, v, _ = ref.attention_z(
            qkvg[b * rows : (b + 1) * rows],
            k_cache.view(-1, KV_HEADS, HEAD),
            v_cache.view(-1, KV_HEADS, HEAD),
            tables[b],
            lengths[b],
            window,
            splits,
            cs,
        )
        for t in range(rows):
            row = b * rows + t
            got = payloads(scratch, plan, "z", HEADS * HEAD // 2, tag, row=row)
            # FP32 sums in another order, then two BF16 roundings (the
            # merge's output and the gated head norm): within two steps.
            torch.testing.assert_close(
                got.view(torch.bfloat16).cpu().float(),
                z[t].float(),
                rtol=2**-6,
                atol=1e-3,
            )
            # The cache gets the rotated k and the raw v of each row's token.
            slot = slots[row] + page_base * PAGE
            got_k = kc.view(-1, KV_HEADS, HEAD)[slot].cpu()
            got_v = vc.view(-1, KV_HEADS, HEAD)[slot].cpu()
            # FP32 norm sums in another order: within one step of k's dtype
            # (BF16 2^-7, e4m3 2^-3).
            step_k = 2**-7 if kv_dtype == torch.bfloat16 else 2**-3
            torch.testing.assert_close(
                got_k.float(), k[t].float(), rtol=step_k, atol=1e-2
            )
            assert torch.equal(got_v.view(torch.uint8), v[t].view(torch.uint8))


def _cache_from(cache: torch.Tensor, page_base: int) -> torch.Tensor:
    """``cache``'s pages on the device as pages from ``page_base`` on of a
    cache that large; the pages below stay unwritten."""
    if not page_base:
        return cache.clone().cuda()
    out = torch.empty(
        (page_base + cache.shape[0], *cache.shape[1:]), dtype=cache.dtype, device="cuda"
    )
    out[page_base:] = cache.cuda()
    return out
