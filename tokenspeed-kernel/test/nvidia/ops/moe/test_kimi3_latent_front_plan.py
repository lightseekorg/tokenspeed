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

"""Host plan rules of the vendored Kimi-K3 front, checked without a GPU.

The expected values are those FlashInfer v0.7.3's planner
(``flashinfer.experimental.kimi_k3_latent_moe.cake_backend.prefill_front_plan``)
produces for the same rows, so a drift in the port shows up here.
"""

import pytest
from tokenspeed_kernel.thirdparty.cuda import kimi3_latent_front as front


@pytest.mark.parametrize(
    "rows,sm_count,expected",
    [
        # 16 row tiles, 8 pair rows: 528 pair tiles over 76 resident clusters leave 72, more
        # than half a wave, so the plan stays whole; 152-SM parts stream evict_first up to
        # eight pair rows.
        (
            2048,
            152,
            dict(
                m_tiles=16,
                num_items=528,
                full_items=528,
                sk_ipc=112,
                sk_max_seg=1,
                sk_total=0,
                sk_tiles=0,
                evict_first=True,
                grid=1056,
            ),
        ),
        # One more row rounds the tile count up to the next pair (18 tiles, 9 pair rows) and
        # leaves the evict_first bound.
        (
            2049,
            152,
            dict(
                m_tiles=18,
                num_items=594,
                full_items=594,
                sk_ipc=112,
                sk_max_seg=1,
                sk_total=0,
                sk_tiles=0,
                evict_first=False,
                grid=1188,
            ),
        ),
        # 1024 rows, 4 pair rows: 264 pair tiles are three full waves plus 36, whose two K
        # halves fit one wave (72 items), so the trailing wave is split and its tiles share
        # the FP32 partial workspace.
        (
            1024,
            152,
            dict(
                m_tiles=8,
                num_items=300,
                full_items=228,
                sk_ipc=56,
                sk_max_seg=2,
                sk_total=4032,
                sk_tiles=36,
                evict_first=True,
                grid=600,
            ),
        ),
        # 148-SM parts: 74 resident clusters and evict_first only up to two pair rows.
        (
            8192,
            148,
            dict(
                m_tiles=64,
                num_items=2112,
                full_items=2112,
                sk_ipc=112,
                sk_max_seg=1,
                sk_total=0,
                sk_tiles=0,
                evict_first=False,
                grid=4224,
            ),
        ),
        # Above 16384 rows the same rules apply: 16385 rows round up to 130 tiles (65 pair
        # rows), and 32768 rows are 256 tiles; both are whole plans without evict_first.
        (
            16385,
            152,
            dict(
                m_tiles=130,
                num_items=4290,
                full_items=4290,
                sk_ipc=112,
                sk_max_seg=1,
                sk_total=0,
                sk_tiles=0,
                evict_first=False,
                grid=8580,
            ),
        ),
        (
            32768,
            152,
            dict(
                m_tiles=256,
                num_items=8448,
                full_items=8448,
                sk_ipc=112,
                sk_max_seg=1,
                sk_total=0,
                sk_tiles=0,
                evict_first=False,
                grid=16896,
            ),
        ),
    ],
)
def test_front_plan_matches_flashinfer_rules(rows, sm_count, expected):
    plan = front.front_plan(rows, sm_count)
    assert {key: plan[key] for key in expected} == expected
    assert plan["cluster_tiles"] == (plan["m_tiles"] // 2) * 66
    assert front.front_kernel_key(plan) == (
        "front:i6144e1" if expected["evict_first"] else "front:i6144"
    )
    slots = expected["sk_tiles"] * expected["sk_max_seg"] * 2
    assert front.partial_ws_numel(plan) == slots * 128 * 256
    assert front.arrival_counter_words(plan) == max(2, expected["sk_tiles"] * 2)


def test_front_plan_is_a_private_copy():
    plan = front.front_plan(4096, 152)
    plan["grid"] = 0
    assert front.front_plan(4096, 152)["grid"] == 32 * 66


@pytest.mark.parametrize("rows", [0, 1, 128, 512])
def test_front_plan_refuses_rows_below_the_vendored_programs(rows):
    with pytest.raises(ValueError, match="at least 513 rows"):
        front.front_plan(rows, 152)


def test_front_plan_refuses_unqualified_sm_counts():
    with pytest.raises(ValueError, match="validated for"):
        front.front_plan(4096, 132)
