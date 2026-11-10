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

"""The slot table by wave ballots (a pick a lane, or two past 64 picks)
matches :func:`moe_experts.slots`."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_experts,
)


@gluon.jit
def _store_slots(base, owner, slot, count, table, NP: gl.constexpr, RP: gl.constexpr):
    q = gl.arange(0, NP, owner.type.layout)
    gl.store(base + q, owner)
    gl.store(base + NP + q, slot)
    gl.store(base + 2 * NP + q * 0, q * 0 + count, mask=q == 0)
    tl: gl.constexpr = table.type.layout
    tp = gl.arange(0, NP, gl.SliceLayout(1, tl))[:, None]
    tr = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    gl.store(base + 3 * NP + tp * RP + tr, table)


@gluon.jit
def _slots_kernel(ids_ptr, out_ptr, RP: gl.constexpr):
    """Stores slots() then the ballot slots, each as owner, slot, count, table."""
    NP: gl.constexpr = RP * 8
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    ids = gl.load(ids_ptr + gl.arange(0, NP, lanes))
    _, owner, slot, count, table = moe_experts.slots(ids)
    _store_slots(out_ptr, owner, slot, count, table, NP, RP)
    if NP <= 64:
        _, owner, slot, count, table = moe_experts._ballot_slots(ids, RP)
    else:
        # Two picks a lane past 64 picks, as the slot table stage loads them.
        two: gl.constexpr = gl.BlockedLayout([2], [64], [gl.num_warps()], [0])
        ids2 = gl.load(ids_ptr + gl.arange(0, NP, two))
        _, owner, slot, count, table = moe_experts._ballot_slots2(ids2, RP)
    _store_slots(out_ptr + (3 * NP + NP * RP + 1), owner, slot, count, table, NP, RP)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16])
def test_ballot_slots_match_slots(rows: int):
    rp = triton_next_power_of_2(rows)
    g = torch.Generator().manual_seed(rows)
    for trial in range(24):
        # Each row's 8 distinct picks, from a small pool so rows share
        # experts (the last trials from all 256, up to 128 slots); the slot
        # table stage repeats the last row up to rp.
        pool = 256 if trial >= 20 else 8 + trial * (2 if rows > 8 else 1)
        picks = [torch.randperm(pool, generator=g)[:8] for _ in range(rows)]
        picks += [picks[-1]] * (rp - rows)
        ids = torch.stack(picks).flatten().to(torch.int32).cuda()
        size = 3 * rp * 8 + rp * 8 * rp + 1
        out = torch.full((2 * size,), -7, dtype=torch.int32, device="cuda")
        _slots_kernel[(1,)](ids, out, rp, num_warps=8, **extern_compile_options())
        torch.cuda.synchronize()
        assert torch.equal(out[:size].cpu(), out[size:].cpu()), (rows, trial)


def triton_next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()
