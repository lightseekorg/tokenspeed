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

"""CPU test of the monokernel MoE stream's closed-form plan on recorded
routings.

Runs the host form of the plan (``moe_stream_plan``) and a fluid timeline
model: every CU streams its list in order; the active CUs share
``min(peak, active * PER_CU_GBS)``; a down unit cannot finish before its
expert's last gate/up group plus a hand-off. Checks coverage, XCD locality,
byte balance, and that the workgroups that also run post-MoE roles finish
their MoE share early enough to be drained before those roles poll.
"""

from collections import Counter

import pytest
import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    pack_cdna4_mxfp4_scale_lanes,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.moe_stream import (
    GROUP_BYTES,
    GROUPS_PER_EXPERT,
    UNIT_BYTES,
    UNITS_PER_EXPERT,
    moe_stream_plan,
)

# Recorded verify routings of Muse Spark at EP1: 4 decode steps each at 8 and 16
# rows (top-8 global expert ids per row).
RECORDED_ROUTINGS = {
    8: [
        [
            [250, 218, 56, 39, 61, 239, 76, 78],
            [61, 125, 56, 78, 187, 110, 228, 76],
            [73, 150, 56, 218, 61, 239, 76, 146],
            [150, 191, 125, 39, 218, 56, 37, 78],
            [218, 239, 126, 250, 165, 178, 150, 205],
            [61, 125, 56, 110, 141, 90, 78, 39],
            [37, 190, 25, 4, 78, 208, 81, 134],
            [37, 78, 110, 134, 133, 224, 151, 39],
        ],
        [
            [206, 232, 32, 239, 152, 58, 254, 10],
            [232, 32, 26, 127, 206, 21, 248, 37],
            [52, 45, 245, 226, 159, 162, 10, 137],
            [231, 32, 156, 45, 225, 235, 76, 10],
            [206, 44, 6, 150, 32, 99, 176, 101],
            [16, 239, 83, 100, 165, 1, 223, 152],
            [150, 206, 6, 99, 159, 165, 82, 44],
            [83, 150, 239, 222, 1, 180, 62, 32],
        ],
        [
            [38, 227, 241, 168, 214, 211, 167, 82],
            [227, 82, 211, 38, 176, 250, 241, 45],
            [168, 250, 227, 241, 38, 122, 82, 211],
            [168, 167, 38, 241, 92, 250, 123, 82],
            [255, 92, 168, 167, 153, 97, 202, 14],
            [227, 11, 89, 45, 147, 211, 98, 235],
            [13, 41, 176, 174, 144, 249, 91, 134],
            [83, 10, 13, 90, 215, 226, 155, 174],
        ],
        [
            [77, 20, 149, 144, 132, 68, 12, 116],
            [68, 190, 77, 114, 20, 120, 149, 144],
            [90, 225, 20, 190, 77, 68, 132, 149],
            [114, 68, 190, 120, 93, 166, 87, 130],
            [190, 35, 76, 255, 205, 51, 157, 120],
            [228, 96, 20, 77, 149, 151, 193, 144],
            [68, 114, 190, 130, 120, 77, 52, 87],
            [190, 68, 52, 114, 90, 28, 145, 132],
        ],
    ],
    16: [
        [
            [96, 224, 123, 73, 179, 118, 72, 95],
            [70, 24, 72, 14, 123, 107, 79, 229],
            [96, 14, 203, 27, 30, 104, 77, 70],
            [96, 203, 104, 27, 200, 72, 123, 77],
            [96, 27, 104, 203, 30, 77, 190, 223],
            [157, 102, 213, 42, 96, 223, 203, 255],
            [43, 72, 104, 27, 200, 223, 77, 203],
            [4, 27, 223, 72, 200, 46, 118, 203],
            [27, 4, 223, 200, 79, 70, 30, 171],
            [102, 104, 223, 79, 46, 190, 114, 16],
            [223, 27, 102, 203, 79, 84, 21, 175],
            [104, 43, 223, 203, 179, 210, 200, 143],
            [118, 223, 203, 102, 46, 2, 96, 50],
            [102, 104, 43, 223, 20, 203, 114, 179],
            [102, 43, 200, 27, 223, 104, 20, 179],
            [43, 104, 179, 223, 200, 102, 138, 70],
        ],
        [
            [177, 238, 245, 45, 251, 18, 15, 211],
            [238, 138, 180, 230, 29, 45, 15, 76],
            [177, 138, 180, 88, 35, 106, 204, 15],
            [177, 202, 59, 200, 138, 45, 127, 18],
            [177, 138, 139, 180, 88, 45, 106, 5],
            [177, 238, 42, 96, 45, 245, 180, 148],
            [177, 238, 245, 45, 42, 15, 35, 146],
            [177, 138, 45, 224, 139, 211, 148, 15],
            [176, 18, 146, 141, 247, 211, 180, 110],
            [176, 141, 146, 247, 133, 53, 45, 211],
            [66, 54, 205, 176, 146, 244, 46, 147],
            [176, 66, 146, 110, 141, 54, 105, 199],
            [176, 133, 148, 146, 211, 53, 184, 202],
            [77, 66, 146, 176, 211, 251, 54, 127],
            [0, 191, 46, 176, 205, 200, 146, 254],
            [205, 176, 146, 211, 147, 50, 241, 254],
        ],
        [
            [99, 230, 110, 137, 39, 31, 211, 34],
            [249, 126, 15, 31, 211, 236, 230, 6],
            [126, 31, 211, 54, 15, 8, 249, 120],
            [249, 126, 169, 15, 31, 222, 131, 54],
            [31, 174, 120, 8, 47, 126, 57, 54],
            [31, 9, 211, 186, 152, 15, 151, 137],
            [223, 107, 22, 92, 211, 39, 137, 99],
            [17, 236, 99, 31, 174, 103, 92, 145],
            [185, 92, 25, 228, 147, 179, 165, 43],
            [185, 92, 23, 147, 72, 37, 179, 45],
            [17, 146, 185, 92, 235, 100, 3, 245],
            [67, 164, 254, 165, 92, 38, 185, 84],
            [36, 44, 92, 246, 31, 38, 185, 87],
            [35, 92, 228, 164, 235, 165, 87, 38],
            [67, 164, 92, 61, 228, 179, 95, 185],
            [17, 253, 185, 92, 179, 146, 38, 67],
        ],
        [
            [243, 163, 254, 10, 127, 59, 62, 250],
            [21, 184, 10, 251, 160, 14, 243, 136],
            [160, 6, 10, 21, 59, 166, 254, 136],
            [21, 136, 252, 107, 132, 160, 6, 81],
            [6, 136, 71, 252, 14, 170, 160, 81],
            [254, 10, 195, 62, 231, 109, 181, 59],
            [227, 252, 14, 165, 10, 6, 88, 67],
            [6, 111, 160, 149, 5, 21, 252, 44],
            [210, 245, 81, 7, 207, 194, 29, 242],
            [183, 243, 233, 190, 81, 210, 24, 7],
            [232, 4, 183, 64, 7, 238, 81, 111],
            [231, 54, 171, 162, 217, 24, 4, 180],
            [4, 86, 40, 24, 183, 81, 171, 235],
            [183, 31, 4, 40, 86, 233, 24, 205],
            [82, 111, 25, 24, 86, 81, 112, 29],
            [73, 25, 31, 209, 69, 165, 183, 112],
        ],
    ],
}

XCDS = 8
SLOTS = 32
TAIL_SLOTS = {0, *range(16, 31)}
PER_CU_GBS = 5.9e3 / 128  # model input: GB/s per CU when 128 CUs stream alone
HANDOFF_US = 0.7  # gate/up -> down hand-off on one XCD
ENTRY_US = 1.2
TAIL_US = 0.6

CASES = [(rows, step) for rows in (8, 16) for step in range(4)]


def _item_bytes(item):
    return GROUP_BYTES if item[0] == "ug" else UNIT_BYTES


def _fluid(plan, peak_tbs, dt_us=0.02):
    """Finish time (us) of each workgroup's list, its launch entry and its
    post-MoE work included."""
    peak = peak_tbs * 1e6  # bytes per us
    cap = PER_CU_GBS * 1e3
    lists = [list(items) for items in plan]
    pos = [0] * len(lists)
    left = [_item_bytes(items[0]) if items else 0.0 for items in lists]
    groups_left = Counter(e for items in lists for kind, e, _ in items if kind == "ug")
    ready = {}
    finish = [0.0] * len(lists)
    t = 0.0
    while any(p < len(items) for p, items in zip(pos, lists)):
        active = [
            c
            for c, items in enumerate(lists)
            if pos[c] < len(items)
            and (items[pos[c]][0] == "ug" or ready.get(items[pos[c]][1], 1e30) <= t)
        ]
        rate = min(peak, len(active) * cap) / max(len(active), 1) * dt_us
        for c in active:
            left[c] -= rate
            if left[c] > 0:
                continue
            kind, expert, _ = lists[c][pos[c]]
            if kind == "ug":
                groups_left[expert] -= 1
                if groups_left[expert] == 0:
                    ready[expert] = t + HANDOFF_US
            pos[c] += 1
            if pos[c] < len(lists[c]):
                left[c] += _item_bytes(lists[c][pos[c]])
            else:
                finish[c] = t
        t += dt_us
    return [ENTRY_US + f for f in finish]


@pytest.mark.parametrize("rows,step", CASES)
def test_plan_covers_every_piece_once(rows, step):
    ids = RECORDED_ROUTINGS[rows][step]
    experts = {e for row in ids for e in row}
    plan = moe_stream_plan(ids)
    seen = Counter((kind, e, i) for items in plan for kind, e, i in items)
    expected = {("ug", e, g) for e in experts for g in range(GROUPS_PER_EXPERT)}
    expected |= {("dn", e, u) for e in experts for u in range(UNITS_PER_EXPERT)}
    assert set(seen) == expected
    assert max(seen.values()) == 1


@pytest.mark.parametrize("rows,step", CASES)
def test_plan_keeps_whole_experts_on_one_xcd(rows, step):
    ids = RECORDED_ROUTINGS[rows][step]
    plan = moe_stream_plan(ids)
    xcds = {}
    for cta, items in enumerate(plan):
        for _kind, e, _ in items:
            xcds.setdefault(e, set()).add(cta % XCDS)
    split = [e for e, where in xcds.items() if len(where) > 1]
    # Only the E % 8 leftover experts are split over the XCDs.
    assert len(split) == len(xcds) % XCDS


@pytest.mark.parametrize("rows,step", CASES)
def test_plan_balances_bytes_and_drains_tail_roles_early(rows, step):
    plan = moe_stream_plan(RECORDED_ROUTINGS[rows][step])
    full = [
        sum(map(_item_bytes, items))
        for c, items in enumerate(plan)
        if c // XCDS not in TAIL_SLOTS
    ]
    assert max(full) / (sum(full) / len(full)) <= 1.03
    for c, items in enumerate(plan):
        if c // XCDS in TAIL_SLOTS:
            assert all(kind == "ug" for kind, _, _ in items)
    for peak in (5.5, 6.5):
        finish = _fluid(plan, peak)
        span = max(finish) + TAIL_US
        tail = max(f for c, f in enumerate(finish) if c // XCDS in TAIL_SLOTS)
        assert tail <= 0.70 * span, (peak, tail, span)


def test_lane_major_scales_follow_the_mfma_lanes():
    generator = torch.Generator().manual_seed(7)
    scale = torch.randint(0, 256, (3, 64, 32), dtype=torch.uint8, generator=generator)
    packed = pack_cdna4_mxfp4_scale_lanes(scale)
    assert packed.shape == (3, 4, 64, 8) and packed.is_contiguous()
    # Lane 16 q + n of strip s holds column 16 s + n, K group q, at step t.
    strip, lane, step = torch.meshgrid(
        torch.arange(4), torch.arange(64), torch.arange(8), indexing="ij"
    )
    expected = scale[:, 16 * strip + lane % 16, 4 * step + lane // 16]
    assert torch.equal(packed, expected)
