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


"""CPU test of the gfx950 MXFP4 MoE stream's closed-form plan
(``moe_stream_plan``): every expert piece is planned once, whole experts stay
on one XCD, the full workgroups' bytes balance, and the tail workgroups get
only gate/up groups."""

from collections import Counter

import pytest

pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

from moe_stream_helpers import RECORDED_ROUTINGS, random_routings  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_stream import (  # noqa: E402
    GROUP_BYTES,
    GROUPS_PER_EXPERT,
    TAIL_SHARE,
    UNIT_BYTES,
    UNITS_PER_EXPERT,
    moe_stream_plan,
)

XCDS = 8
# Slots of the workgroups that get only gate/up groups.
TAIL_SLOTS = {0, *range(16, 31)}

ROUTINGS = [
    pytest.param(RECORDED_ROUTINGS[rows][step], id=f"recorded-r{rows}-s{step}")
    for rows in (8, 16)
    for step in range(4)
] + [
    pytest.param(routing, id=f"random-r{rows}-{i}")
    for rows in (8, 16)
    for i, routing in enumerate(random_routings(rows, 5, rows))
]
# The measured split and an even one (a tail workgroup worth a full one).
SHARES = [TAIL_SHARE, (1, 1)]


def _item_bytes(item):
    return GROUP_BYTES if item[0] == "ug" else UNIT_BYTES


@pytest.mark.parametrize("share", SHARES, ids=str)
@pytest.mark.parametrize("ids", ROUTINGS)
def test_plan_covers_every_piece_once(ids, share):
    experts = {e for row in ids for e in row}
    plan = moe_stream_plan(ids, tail_num=share[0], tail_den=share[1])
    seen = Counter((kind, e, i) for items in plan for kind, e, i in items)
    expected = {("ug", e, g) for e in experts for g in range(GROUPS_PER_EXPERT)}
    expected |= {("dn", e, u) for e in experts for u in range(UNITS_PER_EXPERT)}
    assert set(seen) == expected
    assert max(seen.values()) == 1


@pytest.mark.parametrize("ids", ROUTINGS)
def test_plan_keeps_whole_experts_on_one_xcd(ids):
    plan = moe_stream_plan(ids, tail_num=TAIL_SHARE[0], tail_den=TAIL_SHARE[1])
    xcds = {}
    for cta, items in enumerate(plan):
        for _kind, e, _ in items:
            xcds.setdefault(e, set()).add(cta % XCDS)
    split = [e for e, where in xcds.items() if len(where) > 1]
    # Only the E % 8 leftover experts are split over the XCDs.
    assert len(split) == len(xcds) % XCDS


@pytest.mark.parametrize("share", SHARES, ids=str)
@pytest.mark.parametrize("rows,step", [(r, s) for r in (8, 16) for s in range(4)])
def test_plan_balances_bytes_and_keeps_tail_workgroups_on_gate_up(rows, step, share):
    plan = moe_stream_plan(
        RECORDED_ROUTINGS[rows][step], tail_num=share[0], tail_den=share[1]
    )
    full = [
        sum(map(_item_bytes, items))
        for c, items in enumerate(plan)
        if c // XCDS not in TAIL_SLOTS
    ]
    assert max(full) / (sum(full) / len(full)) <= 1.03
    for c, items in enumerate(plan):
        if c // XCDS in TAIL_SLOTS:
            assert all(kind == "ug" for kind, _, _ in items)
