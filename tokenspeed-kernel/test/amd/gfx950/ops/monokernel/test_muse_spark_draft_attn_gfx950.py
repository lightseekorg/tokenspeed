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

"""The DFlash2 draft's attention-half stages (Muse Spark) on one rank."""

from __future__ import annotations

import socket
from datetime import timedelta

import dflash2_draft_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.benchmark import muse_spark_monokernel as inputs
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import fill, payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_attn,
    draft_mlp,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

HEAD, HALF, Q_HEADS, QKV, KPARTS = 128, 64, 8, 1280, 16
EPS, THETA = 1e-5, 500000.0
SPLITS, PAGE, WINDOW_KEYS = 16, 64, 2048


@gluon.jit
def head_kernel(
    scratch,
    step,
    q_norm,
    k_norm,
    cos_sin,
    positions,
    slots,
    k_cache,
    v_cache,
    PLAN: gl.constexpr,
):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "head")
    if active:
        draft_attn.head_stage(
            mb,
            task,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            slots,
            k_cache,
            v_cache,
            "head",
        )


BASES = {
    "iscat": 0,
    "isum": 128,
    "acoef": 0,
    "aconv0": 128,
    "qkv": 0,
    "head": 160,
    "attn": 240,
    "merge": 0,
    "oproj": 64,
    "afin": 0,
}


def _plan(rows: int, world: int = 1) -> Plan:
    return Plan(
        grid=256,
        stages=draft_attn.stages(BASES, SPLITS, rows),
        regions=draft_attn.regions(SPLITS),
        rows=rows,
        chain=rows,
        world=world,
        peers=draft_attn.PEERS if world > 1 else (),
    )


def reference_head(parts, w, cos, sin, rotated: bool) -> torch.Tensor:
    """One head of one row: the K parts' sum rounded to BF16, then (q and k)
    RMSNorm and NeoX RoPE in FP32, rounded once."""
    x = parts.sum(0).to(torch.bfloat16).float()
    if not rotated:
        return x.to(torch.bfloat16)
    n = x * torch.rsqrt(x.pow(2).mean() + EPS) * w.float()
    n1, n2 = n[:HALF], n[HALF:]
    return torch.cat([n1 * cos - n2 * sin, n2 * cos + n1 * sin]).to(torch.bfloat16)


def _close_bf16(got: torch.Tensor, want: torch.Tensor) -> None:
    """Within one BF16 step: the kernel sums the K parts and rotates in
    another association than the reference."""
    torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=1e-6)


@pytest.mark.parametrize("rows", [8, 16])
def test_head_norms_rotates_and_writes_the_block_rows(rows: int):
    g = torch.Generator().manual_seed(rows)
    device = torch.device("cuda")
    plan = _plan(rows).only(("head",))
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device=device)
    table = inputs.draft_rope_table(70000)
    q_norm = (1 + 0.1 * torch.randn(HEAD, generator=g)).to(torch.bfloat16)
    k_norm = (1 + 0.1 * torch.randn(HEAD, generator=g)).to(torch.bfloat16)
    # The block of a request at 52 000 tokens: positions past 2^15 and slots
    # past 2^31 / 128 keep the int64 cache offsets honest.
    positions = (52000 + torch.arange(rows) % 8 + 8 * (torch.arange(rows) // 8)).to(
        torch.int32
    )
    num_slots = (1 << 24) + 64
    slots = torch.randperm(64, generator=g)[:rows].to(torch.int64) + (1 << 24)
    k_cache = torch.zeros(num_slots, 1, HEAD, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros_like(k_cache)
    step = 5
    parts = torch.randn(rows, KPARTS, QKV, generator=g) * 0.3
    for row in range(rows):
        fill(
            scratch,
            plan,
            "qpart",
            parts[row].reshape(-1).view(torch.int32).to(device),
            launch_tag(step, 0),
            row=row,
        )
    head_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), step, dtype=torch.int32, device=device),
        q_norm.to(device),
        k_norm.to(device),
        table.to(device),
        positions.to(device),
        slots.to(device),
        k_cache,
        v_cache,
        plan.key,
        num_warps=8,
    )
    torch.cuda.synchronize()
    tag = launch_tag(step, 0)
    for row in range(rows):
        cos, sin = table[positions[row], :HALF], table[positions[row], HALF:]
        p = parts[row].view(KPARTS, QKV)
        q = payloads(scratch, plan, "qh", Q_HEADS * HALF, tag, row=row)
        q = q.view(torch.bfloat16).cpu().view(Q_HEADS, HEAD)
        for h in range(Q_HEADS):
            want = reference_head(
                p[:, h * HEAD : (h + 1) * HEAD], q_norm, cos, sin, True
            )
            _close_bf16(q[h], want)
        want_k = reference_head(
            p[:, Q_HEADS * HEAD : (Q_HEADS + 1) * HEAD], k_norm, cos, sin, True
        )
        want_v = reference_head(p[:, (Q_HEADS + 1) * HEAD :], k_norm, cos, sin, False)
        kb = payloads(scratch, plan, "kb", HALF, tag, row=row).view(torch.bfloat16)
        vb = payloads(scratch, plan, "vb", HALF, tag, row=row).view(torch.bfloat16)
        _close_bf16(kb.cpu(), want_k)
        _close_bf16(vb.cpu(), want_v)
        # The cache rows are the mailbox rows, at the row's slot.
        assert torch.equal(k_cache[slots[row], 0].cpu(), kb.cpu())
        assert torch.equal(v_cache[slots[row], 0].cpu(), vb.cpu())


@gluon.jit
def attn_kernel(
    scratch,
    step,
    table,
    seq_len,
    k_cache,
    v_cache,
    PLAN: gl.constexpr,
    SPLITS_: gl.constexpr,
):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "attn")
    if active:
        # One request: no other page table row (stride 0).
        draft_attn.attn_stage(
            mb, task, table, 0, seq_len, k_cache, v_cache, SPLITS_, "attn"
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        draft_attn.merge_stage(mb, task, SPLITS_, "merge")


# The first page whose first element lies past 2^31 elements of a cache.
FAR_PAGE = (1 << 31) // (PAGE * HEAD) + 1


@pytest.mark.parametrize(
    "prefix, page_base", [(52000, 0), (900, 0), (0, 0), (52000, FAR_PAGE)]
)
def test_attention_takes_the_window_and_the_whole_block_for_every_row(
    prefix: int, page_base: int
):
    """``page_base`` numbers the request's pages from there: past 2^31
    cache elements, a page offset computed in int32 reads the wrong keys.
    The pages below it are allocated and never touched."""
    rows = 8
    g = torch.Generator().manual_seed(prefix + 1)
    device = torch.device("cuda")
    plan = _plan(rows).only(("attn", "merge"))
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device=device)
    end = prefix + rows
    # A shuffled page table over a pool larger than the request.
    pages = (end + PAGE - 1) // PAGE
    table = torch.randperm(4 * pages + 8, generator=g)[: pages + 1].to(torch.int32)
    num_slots = (4 * pages + 8) * PAGE
    k_cache = torch.zeros(num_slots, 1, HEAD, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    ctx_k = torch.randn(max(prefix, 1), HEAD, generator=g).to(torch.bfloat16)
    ctx_v = torch.randn(max(prefix, 1), HEAD, generator=g).to(torch.bfloat16)
    tok = torch.arange(prefix)
    slot = table[tok // PAGE].long() * PAGE + tok % PAGE
    k_cache[slot, 0] = ctx_k[:prefix]
    v_cache[slot, 0] = ctx_v[:prefix]
    q = torch.randn(rows, Q_HEADS, HEAD, generator=g).to(torch.bfloat16)
    kb = torch.randn(rows, HEAD, generator=g).to(torch.bfloat16)
    vb = torch.randn(rows, HEAD, generator=g).to(torch.bfloat16)
    step = 9
    tag = launch_tag(step, 0)
    for row in range(rows):
        fill(
            scratch,
            plan,
            "qh",
            q[row].reshape(-1).view(torch.int32).to(device),
            tag,
            row=row,
        )
        fill(scratch, plan, "kb", kb[row].view(torch.int32).to(device), tag, row=row)
        fill(scratch, plan, "vb", vb[row].view(torch.int32).to(device), tag, row=row)
    attn_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), step, dtype=torch.int32, device=device),
        (table + page_base).to(device),
        torch.full((1,), end, dtype=torch.int32, device=device),
        _far(k_cache, page_base * PAGE, device),
        _far(v_cache, page_base * PAGE, device),
        plan.key,
        SPLITS,
        num_warps=8,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    # Every row sees the same keys: [max(end - 2048, 0), end), the block last.
    first = max(end - WINDOW_KEYS, 0)
    keys = torch.cat([ctx_k[first:prefix], kb]).float()
    values = torch.cat([ctx_v[first:prefix], vb]).float()
    for row in range(rows):
        ao = payloads(scratch, plan, "ao", Q_HEADS * HALF, tag, row=row)
        got = ao.view(torch.bfloat16).cpu().view(Q_HEADS, HEAD).double()
        # The PV product takes the probabilities in BF16, as the target
        # layer's attention does: the bound covers that and the output's
        # rounding.
        want, bound = ref.draft_attention(q[row], keys, values)
        err = (got - want).abs()
        assert torch.all(err <= bound), float((err / bound).max())


def _far(cache: torch.Tensor, first: int, device) -> torch.Tensor:
    """``cache`` on the device as the slots from ``first`` on of a cache that
    large; the slots below stay unwritten."""
    if not first:
        return cache.to(device)
    out = torch.empty(
        (first + cache.shape[0], *cache.shape[1:]), dtype=cache.dtype, device=device
    )
    out[first:] = cache.to(device)
    return out


WORLD, HIDDEN, OWNED = 8, ref.HIDDEN, 1024


@gluon.jit
def attn_half_kernel(
    scratch,
    step,
    bases,
    x,
    residual,
    residual_out,
    gain,
    coef_w,
    base,
    qkv_w,
    q_norm,
    k_norm,
    cos_sin,
    positions,
    slots,
    table,
    seq_len,
    k_cache,
    v_cache,
    o_w,
    out,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
    SPLITS_: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    H: gl.constexpr = draft_mlp.HIDDEN
    task, active = schedule.stage_task(PLAN, "iscat")
    if active:
        draft_mlp.scat_stage(mb, task, x, RANK, "ared", "iscat")
    task, active = schedule.stage_task(PLAN, "isum1")
    if active:
        draft_mlp.sum_norm_stages(
            mb,
            task,
            residual,
            residual_out,
            gain,
            RANK,
            "ared",
            "ass",
            "ahin",
            "isum1",
            "inorm",
            True,
        )
    task, active = schedule.stage_task(PLAN, "acoef")
    if active:
        draft_mlp.proj_task(
            mb,
            task,
            coef_w,
            "ahin",
            "acpart",
            draft_mlp.COEFS,
            H,
            draft_mlp.COEF_ROWS,
            draft_mlp.COEF_KS,
            "acoef",
        )
    task, active = schedule.stage_task(PLAN, "aconv0")
    if active:
        draft_mlp.conv0_stage(mb, task, base, "ahin", "acpart", "ahc", "aconv0")
    task, active = schedule.stage_task(PLAN, "qkv")
    if active:
        draft_mlp.proj_task(
            mb, task, qkv_w, "ahc", "qpart", draft_attn.QKV, H, 128, 512, "qkv"
        )
    task, active = schedule.stage_task(PLAN, "head")
    if active:
        draft_attn.head_stage(
            mb,
            task,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            slots,
            k_cache,
            v_cache,
            "head",
        )
    task, active = schedule.stage_task(PLAN, "attn")
    if active:
        # One request: no other page table row (stride 0).
        draft_attn.attn_stage(
            mb, task, table, 0, seq_len, k_cache, v_cache, SPLITS_, "attn"
        )
    task, active = schedule.stage_task(PLAN, "merge")
    if active:
        draft_attn.merge_stage(mb, task, SPLITS_, "merge")
    task, active = schedule.stage_task(PLAN, "oproj")
    if active:
        draft_mlp.proj_task(mb, task, o_w, "ao", "opart", H, 1024, 128, 512, "oproj")
    task, active = schedule.stage_task(PLAN, "afin")
    if active:
        draft_mlp.fin_stage(
            mb,
            task,
            base,
            out,
            "opart",
            draft_attn.OPROJ_KPARTS,
            "acpart",
            draft_attn.AFIN_COLS,
            "afin",
        )


def _close_chained(got: torch.Tensor, want: torch.Tensor) -> None:
    """Within two BF16 steps: the reference projects with one FP32 product,
    the kernel sums 16 K parts, so the projection can round one step apart
    before the norm and RoPE carry it on."""
    torch.testing.assert_close(got.float(), want.float(), rtol=2**-6, atol=1e-6)


def _half_worker(rank: int, port: int, prefix: int) -> None:
    rows = 8
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
        plan = _plan(rows, WORLD)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        parts, residual, gain, coef_w, base, q_norm, k_norm = inputs.draft_attn_shared(
            prefix, rows
        )
        qkv_w, o_w, ctx_k, ctx_v = inputs.draft_attn_rank(prefix, rank, rows)
        end = prefix + rows
        pages = (end + PAGE - 1) // PAGE
        g = torch.Generator().manual_seed(prefix)
        table = torch.randperm(4 * pages + 8, generator=g)[: pages + 1].to(torch.int32)
        k_cache = torch.zeros((4 * pages + 8) * PAGE, 1, HEAD, dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        tok = torch.arange(prefix)
        cslot = table[tok // PAGE].long() * PAGE + tok % PAGE
        k_cache[cslot, 0] = ctx_k[:prefix]
        v_cache[cslot, 0] = ctx_v[:prefix]
        btok = prefix + torch.arange(rows)
        slots = table[btok // PAGE].long() * PAGE + btok % PAGE
        positions = btok.to(torch.int32)
        table_d, k_d, v_d = table.to(device), k_cache.to(device), v_cache.to(device)
        cos_sin = inputs.draft_rope_table(end + 8)
        residual_out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        step = 21
        attn_half_kernel[(plan.grid,)](
            scratch,
            torch.full((1,), step, dtype=torch.int32, device=device),
            group.bases,
            parts[rank].to(device),
            residual.to(device),
            residual_out,
            gain[own].float().to(device),
            bf16_tiles(coef_w).to(device),
            base.to(device),
            bf16_tiles(qkv_w).to(device),
            q_norm.to(device),
            k_norm.to(device),
            cos_sin.to(device),
            positions.to(device),
            slots.to(device),
            table_d,
            torch.full((1,), end, dtype=torch.int32, device=device),
            k_d,
            v_d,
            bf16_tiles(o_w).to(device),
            out,
            plan.key,
            rank,
            SPLITS,
            num_warps=8,
            **extern_compile_options(),
        )
        torch.cuda.synchronize()
        tag = launch_tag(step, 0)

        def words(name, width, copy=0):
            return torch.stack(
                [
                    payloads(scratch, plan, name, width, tag, row=r, copy=copy).cpu()
                    for r in range(rows)
                ]
            )

        def bf(name, width, copy=0):
            return words(name, width, copy).view(torch.bfloat16)

        # Each stage on the kernel's own input, as in the MLP half's test.
        want_h, want_res = ref.tp_sum_norm(parts, residual, gain, round_tp_sum=True)
        assert torch.equal(residual_out[:, own].cpu(), want_res[:, own])
        h = bf("ahin", HIDDEN // 2, copy=step & 1)
        ref.assert_bf16_close(h, want_h)
        cpart = words("acpart", 2 * ref.COEFFICIENTS).view(torch.float32)
        dl = (cpart[:, : ref.COEFFICIENTS] + cpart[:, ref.COEFFICIENTS :]).to(
            torch.bfloat16
        )
        ref.assert_bf16_close(dl, (h.float() @ coef_w.float().T).to(torch.bfloat16))
        delta = dl.view(rows, 2, ref.TAPS, ref.GROUPS)
        hc = bf("ahc", HIDDEN // 2)
        ref.assert_bf16_close(hc, ref.grouped_conv(h, delta[:, 0], base[0]))
        proj = hc.float() @ qkv_w.float().T  # [rows, 1280]
        q = bf("qh", Q_HEADS * HALF).view(rows, Q_HEADS, HEAD)
        kb = bf("kb", HALF)
        vb = bf("vb", HALF)
        for r in range(rows):
            cos, sin = cos_sin[positions[r], :HALF], cos_sin[positions[r], HALF:]
            for hh in range(Q_HEADS):
                _close_chained(
                    q[r, hh],
                    reference_head(
                        proj[r, None, hh * HEAD : (hh + 1) * HEAD],
                        q_norm,
                        cos,
                        sin,
                        True,
                    ),
                )
            _close_chained(
                kb[r], reference_head(proj[r, None, 1024:1152], k_norm, cos, sin, True)
            )
            _close_chained(
                vb[r], reference_head(proj[r, None, 1152:], k_norm, cos, sin, False)
            )
            assert torch.equal(k_d[slots[r], 0].cpu(), kb[r])
            assert torch.equal(v_d[slots[r], 0].cpu(), vb[r])
        first = max(end - WINDOW_KEYS, 0)
        keys = torch.cat([ctx_k[first:prefix], kb]).float()
        values = torch.cat([ctx_v[first:prefix], vb]).float()
        ao = bf("ao", Q_HEADS * HALF).view(rows, Q_HEADS, HEAD)
        for r in range(rows):
            want, bound = ref.draft_attention(q[r], keys, values)
            err = (ao[r].double() - want).abs()
            assert torch.all(err <= bound), float((err / bound).max())
        partial = (ao.reshape(rows, -1).float() @ o_w.float().T).to(torch.bfloat16)
        ref.assert_bf16_close(
            out.cpu(), ref.grouped_conv(partial, delta[:, 1], base[1])
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("prefix", [3000])
def test_draft_attention_half_matches_the_reference_stage_by_stage(prefix: int):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_half_worker, args=(port, prefix), nprocs=WORLD, join=True)
