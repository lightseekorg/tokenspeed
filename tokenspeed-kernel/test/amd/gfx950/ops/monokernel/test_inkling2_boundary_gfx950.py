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

"""The stages of an Inkling2 boundary local layer (the first of an
attention-residual block) against the regular decode path's kernels, bit
for bit: the new block set's partials sweep (``attention_residual_partials``:
``_attnres_partial_kernel``), the MLP mix over them (``attention_residual_
combine`` of the one term ``A``), and layer 0's attention input
(``rmsnorm_precision`` of the embedding: ``_rmsnorm_precision_kernel``),
at every geometry the plans lay out (``plan.GEOMETRIES``): the regular
kernels' tiles follow the row width, and so do the stages'."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import fill, payloads  # noqa: E402
from test_inkling2_head_gfx950 import (  # noqa: E402
    _assert_writes_only,
    _regular_mix,
    _words,
)
from tokenspeed_kernel.ops.layernorm.precision import (  # noqa: E402
    rmsnorm_precision,
)
from tokenspeed_kernel.ops.residual.attention import (  # noqa: E402
    attention_residual_partials,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    mix,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (  # noqa: E402
    plan as ip,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag  # noqa: E402

NUM_WARPS = 8
STEP = 17
Q = gl.constexpr(ip.SET_QUERIES)
ROWS = [(1, 1), (4, 4), (8, 8), (16, 8), (32, 8), (64, 8)]
# Each geometry's tiles: Small's 4608-wide rows split 4096 + 512, the
# half-width geometry's 2304 2048 + 256 (a masked second tile).
GEOMETRIES = pytest.mark.parametrize(
    "dims", ip.GEOMETRIES, ids=[f"h{d.hidden}" for d in ip.GEOMETRIES]
)


@gluon.jit
def _sweep_kernel(scratch, step, sweep, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    G: gl.constexpr = schedule.grid(PLAN)
    sweeps = mix.Sweep(*sweep)
    task, rounds = schedule.stage_rounds(PLAN, "sweep")
    for i in range(rounds):
        mix.sweep_stage(
            mb, task + i * G, sweeps, "cand", "set_acc", "set_ms", 1, 0, "sweep"
        )
    task, rounds = schedule.stage_rounds(PLAN, "sweep_rest")
    for i in range(rounds):
        mix.sweep_stage(
            mb,
            task + i * G,
            sweeps,
            "cand",
            "set_acc",
            "set_ms",
            Q - 1,
            1,
            "sweep_rest",
        )


@GEOMETRIES
@pytest.mark.parametrize("stored", [1, 3, 7])
@pytest.mark.parametrize("rows,chain", ROWS)
def test_sweep_is_the_regular_kernel(dims, rows, chain, stored):
    """The new block set's partials over ``stored`` blocks and this launch's
    candidate (from its mailbox) equal the regular partials kernel's over
    the ``stored + 1`` blocks, bit for bit; the MLP query's are also
    published for the MLP mix."""
    g = torch.Generator().manual_seed(500 + rows + stored)
    H = dims.hidden
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", True, False).only(
        ("sweep", "sweep_rest")
    )
    tag = launch_tag(STEP, 0)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    blocks = (torch.randn(stored + 1, rows, H, generator=g) * 3).bfloat16().cuda()
    blocks[:, :, 5::97] = 0.0
    weights = (torch.randn(Q, H, generator=g) * H**-0.5).cuda()
    for r in range(rows):
        fill(scratch, plan, "cand", _words(blocks[stored, r].cpu()).cuda(), tag, row=r)
    m = torch.zeros(Q, rows, device="cuda")
    s = torch.zeros(Q, rows, device="cuda")
    acc = torch.zeros(Q, rows, H, device="cuda")
    # The kernel reads the stored blocks only; the candidate's slot is junk.
    stored_blocks = blocks.clone()
    stored_blocks[stored] = 7.0
    before = scratch.clone()
    _sweep_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
        (
            stored_blocks,
            torch.tensor([stored], dtype=torch.int32, device="cuda"),
            weights,
            m,
            s,
            acc,
        ),
        plan.key,
        num_warps=NUM_WARPS,
        **extern_compile_options(),
    )
    torch.cuda.synchronize()
    want_m, want_s, want_acc = attention_residual_partials(
        blocks, weights, normalize_values=True, eps=1e-6
    )
    assert torch.equal(m, want_m)
    assert torch.equal(s, want_s)
    assert torch.equal(acc, want_acc)
    _assert_writes_only(before, scratch, plan, ("set_acc", "set_ms"))
    for r in range(rows):
        got = payloads(scratch, plan, "set_acc", H, tag, row=r).view(torch.float32)
        assert torch.equal(got.cpu(), want_acc[0, r].cpu())
        ms = payloads(scratch, plan, "set_ms", 2, tag, row=r).view(torch.float32)
        assert torch.equal(ms.cpu(), torch.stack([want_m[0, r], want_s[0, r]]).cpu())


@gluon.jit
def _boundary_mlp_mix_kernel(scratch, step, query, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "mix_mlp")
    if active:
        mix.mlp_mix_stage(
            mb.row(task),
            mix.Query(*query).row_of(task, schedule.dims(PLAN).hidden),
            "cand",
            "attn_sc",
            "x_mlp",
            "set_acc",
            "set_ms",
            True,
            "mix_mlp",
        )


@GEOMETRIES
@pytest.mark.parametrize("rows,chain", ROWS)
def test_boundary_mlp_mix_is_the_regular_kernel(dims, rows, chain):
    """A boundary layer's MLP mix of the one term ``A`` over the new set's
    partial (from the sweep's mailbox regions) is bit for bit the regular
    path's, over enough launches that a different order or contraction
    shows."""
    g = torch.Generator().manual_seed(600 + rows)
    H = dims.hidden
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", True, False).only(
        ("mix_mlp",)
    )
    tag = launch_tag(STEP, 0)
    for _ in range(8):
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        a_term = torch.randn(rows, H, generator=g).bfloat16()
        w = torch.randn(H, generator=g) * H**-0.5
        gain = (torch.rand(H, generator=g) + 0.5).bfloat16()
        acc = torch.randn(rows, H, generator=g)
        m = torch.randn(rows, 1, generator=g)
        s = torch.rand(rows, 1, generator=g) + 1.0
        for r in range(rows):
            fill(scratch, plan, "attn_sc", _words(a_term[r]).cuda(), tag, row=r)
            fill(scratch, plan, "set_acc", acc[r].view(torch.int32).cuda(), tag, row=r)
            ms = torch.cat([m[r], s[r]]).view(torch.int32).cuda()
            fill(scratch, plan, "set_ms", ms, tag, row=r)
        # The query's own partial fields are not read: the sweep's are.
        junk = torch.full((rows, H), float("nan"))
        before = scratch.clone()
        _boundary_mlp_mix_kernel[(plan.grid,)](
            scratch,
            torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
            (
                w.cuda(),
                gain.cuda(),
                junk.cuda(),
                junk[:, 0].contiguous().cuda(),
                junk[:, 0].contiguous().cuda(),
            ),
            plan.key,
            num_warps=NUM_WARPS,
            **extern_compile_options(),
        )
        torch.cuda.synchronize()
        _assert_writes_only(before, scratch, plan, ("x_mlp",))
        got = [
            payloads(scratch, plan, "x_mlp", H // 2, tag, row=r) for r in range(rows)
        ]
        x = torch.stack(got).view(torch.bfloat16).cpu()
        reg_x, _ = _regular_mix((a_term,), w, gain, acc, m, s, False)
        assert torch.equal(x, reg_x)


@gluon.jit
def _norm_kernel(scratch, step, terms, cand_out, gain, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    H: gl.constexpr = schedule.dims(PLAN).hidden
    row, active = schedule.stage_task(PLAN, "norm_attn")
    if active:
        mix.attention_norm_stage(
            mb.row(row),
            terms + row * H,
            cand_out + row * H,
            gain,
            "x_attn",
            "cand",
            "norm_attn",
        )


@GEOMETRIES
@pytest.mark.parametrize("rows,chain", ROWS)
def test_layer0_attention_norm_is_the_regular_kernel(dims, rows, chain):
    """Layer 0's attention input is the regular path's ``attn_norm`` of the
    embedding (``rmsnorm_precision``, no mix) bit for bit, over enough
    launches that a different reduction order or contraction shows; its
    block (``cand_out`` and the sweep's ``cand``) is the embedding as is,
    signed zeros included. (The regular launch takes 8 warps from 4096
    columns, 4 below.)"""
    g = torch.Generator().manual_seed(700 + rows)
    H = dims.hidden
    plan = ip.local_layer_plan(dims, rows, chain, "nvfp4", True, True).only(
        ("norm_attn",)
    )
    tag = launch_tag(STEP, 0)
    for _ in range(8):
        scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
        embedding = (torch.randn(rows, H, generator=g) * 2).bfloat16()
        embedding[:, 3::101] = -0.0
        gain = (torch.rand(H, generator=g) + 0.5).bfloat16()
        cand_out = torch.full((rows, H), float("nan"), device="cuda").bfloat16()
        before = scratch.clone()
        _norm_kernel[(plan.grid,)](
            scratch,
            torch.full((1,), STEP, dtype=torch.int32, device="cuda"),
            embedding.cuda(),
            cand_out,
            gain.cuda(),
            plan.key,
            num_warps=NUM_WARPS,
            **extern_compile_options(),
        )
        torch.cuda.synchronize()
        want = rmsnorm_precision(
            embedding.cuda(),
            gain.cuda(),
            1e-6,
            torch.bfloat16,
            mean_by_reciprocal=False,
        ).cpu()
        _assert_writes_only(before, scratch, plan, ("x_attn", "cand"))
        x = torch.stack(
            [payloads(scratch, plan, "x_attn", H // 2, tag, row=r) for r in range(rows)]
        )
        assert torch.equal(
            x.view(torch.bfloat16).cpu().view(torch.int16), want.view(torch.int16)
        )
        bits = embedding.view(torch.int16)
        assert torch.equal(cand_out.cpu().view(torch.int16), bits)
        cand = torch.stack(
            [payloads(scratch, plan, "cand", H // 2, tag, row=r) for r in range(rows)]
        )
        assert torch.equal(cand.cpu().view(torch.int16).view(rows, H), bits)
