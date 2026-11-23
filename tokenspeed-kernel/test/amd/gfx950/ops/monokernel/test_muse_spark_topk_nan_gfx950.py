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

"""The Muse Spark top-8 of one wave on NaN and -inf choices: 8 distinct
experts of the layer, the highest choices first (NaN and -inf rank lowest,
ties to the lowest expert)."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_route,
)

EXPERTS = gl.constexpr(256)
WAVES = 8


@gluon.jit
def _top8_kernel(choice_ptr, score_ptr, ids_ptr):
    """topk_stage's top-8 of one row's choices: the choice mapping, then the
    top-8 asm, as topk_stage runs them."""
    nw: gl.constexpr = gl.num_warps()
    wl: gl.constexpr = gl.BlockedLayout([EXPERTS // 64], [64], [nw], [0])
    e = gl.arange(0, EXPERTS, wl)
    choice = moe_route._finite_choice(gl.load(choice_ptr + e))
    score = gl.load(score_ptr + e)
    one: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    slot = gl.arange(0, 64 * nw, one)
    vt: gl.constexpr = slot.type.with_element_ty(gl.float32)
    res = gl.inline_asm(
        moe_route._top8_asm,
        moe_route._TOP8_CONSTRAINTS,
        [choice, score],
        [slot.type, vt, choice.type, vt] + [gl.int64] * 5 + [gl.int32] * 10,
        is_pure=True,
    )
    gl.store(ids_ptr + slot, res[0], mask=slot < 8)


@pytest.mark.parametrize(
    "case", ["all-nan", "nan-at-max", "one-nan", "few-finite", "all-inf"]
)
def test_the_top8_picks_8_distinct_experts_on_nan_and_inf_choices(case: str):
    device = torch.device("cuda:0")
    g = torch.Generator(device=device).manual_seed(0)
    choice = torch.rand(int(EXPERTS), generator=g, device=device)
    if case == "all-nan":
        choice.fill_(float("nan"))
    elif case == "nan-at-max":
        choice[[5, 6]] = 3.0
        choice[[2, 129]] = float("nan")
    elif case == "one-nan":
        choice[37] = float("nan")
    elif case == "few-finite":
        choice.fill_(float("-inf"))
        choice[[5, 77, 140]] = 1.0
    else:
        choice.fill_(float("-inf"))
    score = torch.rand(int(EXPERTS), generator=g, device=device)
    ids = torch.full((64 * WAVES,), -2, dtype=torch.int32, device=device)
    _top8_kernel[(1,)](choice, score, ids, num_warps=WAVES)
    # NaN and -inf rank below every finite choice; ties go to the lowest expert.
    rank = torch.where(torch.isfinite(choice), choice, float("-inf")).cpu()
    want = sorted(range(int(EXPERTS)), key=lambda x: (-rank[x].item(), x))[:8]
    assert ids[:8].cpu().tolist() == want
