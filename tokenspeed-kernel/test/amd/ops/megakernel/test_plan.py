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

"""Megakernel plans: the deadlock rule, region layout, keys and slices."""

from __future__ import annotations

import json

import pytest
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Plan, Stage, placement


def test_plan_rejects_a_stage_reading_a_later_stage():
    with pytest.raises(ValueError, match="before any earlier stage"):
        Plan(
            grid=256,
            stages=(
                Stage("consume", 4, 0, reads=("x",)),
                Stage("produce", 4, 8, writes=("x",)),
            ),
            regions=(("x", 16),),
            rows=1,
            chain=1,
        )


def test_plan_regions_start_on_their_own_line():
    plan = Plan(
        grid=256,
        stages=(Stage("produce", 4, 0, writes=("a", "b")),),
        regions=(("a", 33), ("b", 1)),
        rows=2,
        chain=1,
    )
    assert plan.offset("a") == 0
    assert plan.offset("b") == 64
    assert plan.row_pairs == 96
    assert plan.scratch_pairs == 192


def test_plan_key_is_json_text_that_tells_plans_apart():
    def plan(base):
        return Plan(
            grid=256,
            stages=(Stage("produce", 4, base, writes=("a",)),),
            regions=(("a", 8),),
            rows=1,
            chain=1,
        )

    key = plan(0).key
    # Triton's compile hooks serialize compile-time arguments to JSON.
    assert json.loads(json.dumps(key)) == key
    assert key.plan.stage("produce").base == 0
    assert key == plan(0).key
    assert key != plan(1).key


def test_plan_rejects_a_base_outside_the_grid_or_no_tasks():
    with pytest.raises(ValueError, match="do not fit"):
        Plan(grid=256, stages=(Stage("s", 4, 256),), regions=(), rows=1, chain=1)
    with pytest.raises(ValueError, match="do not fit"):
        Plan(grid=256, stages=(Stage("s", 0, 0),), regions=(), rows=1, chain=1)


def test_plan_rows_are_whole_requests():
    assert Plan(grid=8, stages=(), regions=(), rows=4, chain=2).chain == 2
    with pytest.raises(ValueError, match="requests of 3 tokens"):
        Plan(grid=8, stages=(), regions=(), rows=4, chain=3)
    with pytest.raises(ValueError, match="requests of 0 tokens"):
        Plan(grid=8, stages=(), regions=(), rows=4, chain=0)


def test_plan_row_copies_of_regions_follow_row_zero():
    plan = Plan(
        grid=8,
        stages=(Stage("s", 1, 0, writes=("a",)),),
        regions=(("a", 40),),
        rows=3,
        chain=1,
    )
    # The plan pads one row of regions to whole 256-byte lines. Rows follow it.
    assert plan.row_pairs == 64
    assert plan.scratch_pairs == 3 * 64


def _chain_plan() -> Plan:
    return Plan(
        grid=8,
        stages=(
            Stage("a", 1, 0, writes=("x",)),
            Stage("b", 1, 1, reads=("x",), writes=("y",)),
            Stage("c", 1, 2, reads=("x", "y"), writes=("z",)),
        ),
        regions=(("x", 4), ("y", 4), ("z", 4)),
        rows=2,
        chain=1,
    )


def test_plan_slice_takes_the_inputs_of_dropped_stages_from_the_host():
    full = _chain_plan()
    part = full.only(("c",))
    assert [stage.name for stage in part.stages] == ["c"]
    assert part.external == ("x", "y")
    # The slice keeps the full plan's placement and mailbox layout.
    assert part.stage("c") == full.stage("c")
    assert part.offset("z") == full.offset("z")
    assert part.scratch_pairs == full.scratch_pairs
    assert full.only(("b", "c")).external == ("x",)


def test_plan_slice_rejects_unknown_stages():
    with pytest.raises(KeyError, match="no stages"):
        _chain_plan().only(("c", "d"))


def test_placement_must_name_the_module_stages_in_order():
    assert placement({"b1": 28, "b3": 30}, ("b1", "b3")) == (28, 30)
    with pytest.raises(ValueError, match="must place"):
        placement({"b3": 30, "b1": 28}, ("b1", "b3"))
    with pytest.raises(ValueError, match="must place"):
        placement({"b1": 28}, ("b1", "b3"))
