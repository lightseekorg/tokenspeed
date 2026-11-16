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

"""Stuck-poll record decoding on the host: grouping a stall's records, and
naming a CTA's tasks when no stage that reads the region runs there."""

from __future__ import annotations

import pytest
import torch

# The CPU-side helpers come from the AMD kernel package, which other vendors
# do not install.
pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel_amd.ops.gfx950.monokernel import stuck  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.mailbox import (  # noqa: E402
    STUCK_WORDS,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    Stage,
    launch_tag,
)

SCRATCH = 1 << 20
GRID = 8
WAVES = 8


def _plan() -> Plan:
    # Stage a (CTAs 0-1) feeds b (CTAs 2-5) through "x"; b feeds c (CTAs
    # 6-7) through "y".
    return Plan(
        grid=GRID,
        stages=(
            Stage("a", 2, 0, writes=("x",)),
            Stage("b", 4, 2, reads=("x",), writes=("y",)),
            Stage("c", 2, 6, reads=("y",)),
        ),
        regions=(("x", 64), ("y", 64)),
        rows=2,
        chain=1,
    )


def _buffer(plan: Plan, waits) -> torch.Tensor:
    """A record buffer of ``waits``: (cta, wave, region, row, pair, first check)."""
    t = torch.zeros(GRID * WAVES + 1, STUCK_WORDS, dtype=torch.int64)
    t[-1, :5] = torch.tensor([stuck.MAGIC, SCRATCH, 0, 1, WAVES])
    for cta, wave, region, row, pair, start in waits:
        slot = cta * WAVES + wave
        at = plan.offset(region) + row * plan.row_pairs + pair
        t[slot, 0] = launch_tag(5, 3)
        t[slot, 2] = SCRATCH + 8 * at
        t[slot, 3] = start
        t[slot, 4] = start + 60 * stuck.TICKS_PER_SECOND
    return t


def test_summary_groups_a_wait_and_leads_with_the_earliest_stage():
    plan = _plan()
    # c checks "y" first, but it waits on b, which waits on "x" (a's): b's
    # group leads, the rows of one wait share a line.
    t = _buffer(
        plan,
        [
            (6, 0, "y", 0, 1, 100),
            (7, 0, "y", 1, 1, 101),
            (2, 0, "x", 0, 3, 200),
            (3, 0, "x", 1, 5, 201),
        ],
    )
    lines = stuck.summarize(t, lambda layer: [plan] if layer == 3 else [])
    assert len(lines) == 4
    assert lines[0].startswith(
        "rank 0 CTA 2 wave 0: stage b waited for region x row 0 pair 3"
    )
    assert lines[1].startswith(
        "  2 waves on 2 CTAs in all (2-3) on rows 0-1, pairs 3, 5"
    )
    assert lines[2].startswith(
        "rank 0 CTA 6 wave 0: stage c waited for region y row 0 pair 1"
    )
    assert lines[3].startswith("  2 waves on 2 CTAs in all (6-7) on rows 0-1, pairs 1")


def test_a_wait_with_no_reading_stage_names_the_cta_tasks():
    plan = _plan()
    # CTA 0 runs only a, which reads nothing: a record there decoded under
    # this plan (another launch's layout, say) names the CTA's tasks.
    t = _buffer(plan, [(0, 0, "y", 1, 2, 100)])
    (line,) = stuck.summarize(t, lambda layer: [plan])
    assert line.startswith(
        "rank 0 CTA 0 wave 0: stage ? (no stage that reads y runs on CTA 0; its tasks "
        f"under the plan of 2 rows, chain 1, {plan.row_pairs} pairs a row: a#0) "
        "waited for region y row 1 pair 2"
    )


def test_two_waves_of_one_cta_decode_distinct_pairs_and_use_cta_for_stage(tmp_path):
    plan = _plan()
    t = _buffer(plan, [(2, 1, "x", 0, 3, 200), (2, 7, "x", 1, 5, 201)])
    path = tmp_path / "waves.bin"
    path.write_bytes(t.numpy().tobytes())
    t = stuck.read_file(str(path))
    assert stuck.header(t)["waves"] == WAVES
    recs = stuck.records(t)
    assert [(r["cta"], r["wave"]) for r in recs] == [(2, 1), (2, 7)]
    assert [stuck.locate(plan, (r["address"] - SCRATCH) // 8) for r in recs] == [
        ("x", 0, 3, 0),
        ("x", 1, 5, 0),
    ]
    lines = stuck.decode(t, lambda layer: [plan])
    assert "CTA 2 wave 1: stage b waited for region x row 0 pair 3" in lines[0]
    assert "CTA 2 wave 7: stage b waited for region x row 1 pair 5" in lines[1]
    summary = stuck.summarize(t, lambda layer: [plan])
    assert summary[1].startswith("  2 waves on 1 CTAs in all (2) on rows 0-1")


@pytest.mark.parametrize("waves", [-1, 0, 3])
def test_invalid_header_wave_count_is_rejected(waves):
    t = _buffer(_plan(), [])
    t[-1, 4] = waves
    with pytest.raises(ValueError, match="wave count"):
        stuck.records(t)
