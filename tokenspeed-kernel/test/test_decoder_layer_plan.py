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

"""The fused-layer plan checker on the Muse Spark role table (CPU only)."""

import dataclasses

import pytest
from tokenspeed_kernel.ops.monokernel import PlanKey
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark import plan_variants
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.plan import muse_plan
from tokenspeed_kernel.ops.monokernel._gluon.plan import (
    Clear,
    Load,
    PlanError,
    Publish,
    Region,
    Wait,
    check_plan,
)

_KEY = PlanKey(1, "sliding", "sliding")


def _edit(plan, cta, edit):
    """``plan`` with workgroup ``cta``'s steps replaced by ``edit(steps)``."""
    programs = dict(plan.programs)
    programs[cta] = tuple(edit(list(plan.programs[cta])))
    return dataclasses.replace(plan, programs=programs)


@pytest.mark.parametrize("key", plan_variants(), ids=str)
def test_every_served_variant_is_accepted(key):
    """12 variants: 8 / 16 rows x sliding / full x next sliding / full /
    regular; each contains the all-to-all H1 hop."""
    assert check_plan(muse_plan(key)) > 0


def test_a_merge_task_on_a_weight_holding_slot_is_rejected():
    """At full layers SND workgroups hold their o_proj and router weights
    after their first attention item; a merge task placed there would poll
    the records behind those loads (poll-behind-own-loads)."""
    plan = muse_plan(PlanKey(1, "full", "sliding"))
    merge = [x for x in plan.programs[(0, 24)] if x.role == "MERGE"]
    assert merge

    def move_out(steps):
        return [x for x in steps if x.role != "MERGE"]

    def move_in(steps):
        at = next(
            i for i, x in enumerate(steps) if x.role == "SND" and isinstance(x, Clear)
        )
        return steps[:at] + merge + steps[at:]

    plan = _edit(_edit(plan, (0, 24), move_out), (0, 5), move_in)
    with pytest.raises(
        PlanError, match="MERGE on \\(0, 5\\) polls .*poll-behind-own-loads"
    ):
        check_plan(plan)


def test_the_v01_role_map_with_56_own4_is_rejected():
    """v0.1 placed OWN4 on 7 slots per XCD; the role table needs 8."""
    plan = muse_plan(_KEY)
    for xcd in range(8):
        plan = _edit(
            plan, (xcd, 30), lambda s: [x for x in s if x.role not in ("OWN4", "FIN")]
        )
    with pytest.raises(PlanError, match="role OWN4"):
        check_plan(plan)


def test_a_variant_without_h1_is_rejected():
    plan = muse_plan(_KEY)
    for cta, steps in plan.programs.items():
        plan = _edit(
            plan,
            cta,
            lambda s: [
                x
                for x in s
                if not (isinstance(x, Publish) and x.keys[0].startswith("H1.rs"))
            ],
        )
    with pytest.raises(PlanError, match="no all-to-all hop"):
        check_plan(plan)


def test_a_vector_poll_behind_its_own_loads_is_rejected():
    """RTR polls the AG words behind its router weights: allowed only as
    issue-wait-compute."""

    def drop_flag(steps):
        return [
            (
                dataclasses.replace(x, queued="")
                if isinstance(x, Wait) and x.role == "RTR"
                else x
            )
            for x in steps
        ]

    with pytest.raises(PlanError, match="poll-behind-own-loads"):
        check_plan(_edit(muse_plan(_KEY), (3, 5), drop_flag))


def test_router_weights_at_entry_would_queue_the_owner_poll():
    """OWN1 polls the reduce-scatter words with no loads in flight: the
    router's tiles go out after it. Issued at entry (before the push), they
    would sit in front of that poll."""

    def router_at_entry(steps):
        router = next(x for x in steps if isinstance(x, Load) and x.role == "RTR")
        steps = [x for x in steps if x is not router]
        return [steps[0], router] + steps[1:]

    plan = muse_plan(_KEY)
    assert check_plan(plan) > 0
    with pytest.raises(PlanError, match="OWN1 on \\(3, 0\\) polls .*poll-behind"):
        check_plan(_edit(plan, (3, 0), router_at_entry))


def test_tail_workgroups_push_every_h5_before_waiting_on_one():
    """At 16 rows a tail workgroup completes two routes: both H5 pushes go
    out before its first H5 wait, so the second push does not wait for the
    first row's whole H5 hop."""
    plan = muse_plan(PlanKey(2, "sliding", "sliding"))
    for xcd in range(8):
        steps = plan.programs[(xcd, 24)]
        pushes = [
            i
            for i, x in enumerate(steps)
            if isinstance(x, Publish) and x.keys[0].startswith("H5")
        ]
        waits = [
            i for i, x in enumerate(steps) if isinstance(x, Wait) and x.role == "FIN"
        ]
        assert len(pushes) == 2 and len(waits) == 2
        assert max(pushes) < min(waits)
    assert check_plan(plan) > 0


def test_a_wait_cycle_is_rejected():
    """Each owner waits for the next owner's AG words before publishing its
    own: a ring of waits that resident workgroups cannot leave."""
    plan = muse_plan(_KEY)
    for xcd in range(8):

        def ring(steps, xcd=xcd):
            at = next(
                i
                for i, x in enumerate(steps)
                if x.role == "OWN1" and isinstance(x, Publish)
            )
            wait = Wait("OWN1", "all", (f"H1.ag[{(xcd + 1) % 8}]",), "vector", "landed")
            return steps[:at] + [wait] + steps[at:]

        plan = _edit(plan, (xcd, 0), ring)
    with pytest.raises(PlanError, match="deadlock"):
        check_plan(plan)


def test_a_clear_after_the_rs_push_is_rejected():
    """Clears must drain before the all-to-all push that lets a peer reach
    the next launch."""

    def late(steps):
        clear = next(x for x in steps if isinstance(x, Clear))
        steps.remove(clear)
        at = next(
            i for i, x in enumerate(steps) if x.role == "SND" and isinstance(x, Publish)
        )
        return steps[: at + 1] + [clear] + steps[at + 1 :]

    with pytest.raises(PlanError, match="after its H1.rs publish"):
        check_plan(_edit(muse_plan(_KEY), (2, 7), late))


def test_a_region_missing_a_clearer_is_rejected():
    def no_clear(steps):
        return [x for x in steps if not isinstance(x, Clear)]

    with pytest.raises(PlanError, match="is cleared by 127"):
        check_plan(_edit(muse_plan(_KEY), (4, 9), no_clear))


def test_a_key_with_two_publishers_is_rejected():
    def twice(steps):
        return steps + [Publish("TOPK", "all", ("H3[0]",))]

    with pytest.raises(PlanError, match="published twice"):
        check_plan(_edit(muse_plan(_KEY), (7, 24), twice))


def test_regions_must_span_the_rows():
    plan = muse_plan(PlanKey(2, "sliding", "sliding"))
    short = tuple(Region(r.name, r.copies, 8) for r in plan.regions)
    with pytest.raises(PlanError, match="spans 8 rows"):
        check_plan(dataclasses.replace(plan, regions=short))
