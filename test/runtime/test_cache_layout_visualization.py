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

"""Startup layout reports over real packed plans, without device allocations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.visualization import (
    format_cache_layout,
)

register_cuda_ci(
    est_time=1,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
)


def _layout():
    history = CacheGroupSpec(
        "full_attention",
        "full_history",
        rows_per_page=16,
        entry_stride_tokens=1,
        replayable=False,
    )
    state = CacheGroupSpec("linear_attention_0", "state", replayable=False)
    return pack(
        (
            (
                history,
                (
                    CacheFieldSpec("layer.0.key", "unit.0", (16,), "uint8"),
                    CacheFieldSpec(
                        "layer.0.value",
                        "unit.1",
                        (31,),
                        "uint8",
                        exact_page_stride=False,
                    ),
                ),
            ),
            (
                state,
                (
                    CacheFieldSpec(
                        "layer.1.a", "unit.0", (3,), "uint8", exact_page_stride=False
                    ),
                    CacheFieldSpec(
                        "layer.1.b", "unit.0", (1,), "float32", exact_page_stride=False
                    ),
                ),
            ),
        ),
        prefix_granularity=16,
        cache_blocks_per_lcm_block={"full_attention": 1, "linear_attention_0": 2},
        alignment=16,
        max_padding_fraction=float("inf"),
    )


def _table_cells(report):
    """Read the displayed rows and columns, ignoring alignment whitespace."""
    rows = []
    row = []
    for line in report.splitlines():
        if line.startswith("|"):
            row.append([cell.strip() for cell in line.split("|")[1:-1]])
        elif line.startswith("+") and row:
            rows.append(list(zip(*row, strict=True)))
            row = []
    return {row[0][0]: [list(filter(None, cell)) for cell in row] for row in rows}


def test_group_columns_and_plane_rows_show_field_offsets_and_packing():
    plan = _layout().bind(3)
    report = format_cache_layout(plan)
    rows = _table_cells(report)

    assert list(rows) == ["plane", "unit.0", "unit.1", "TOTAL / parent"]
    assert rows["plane"] == [
        ["plane"],
        ["full_attention", "packing=1", "usable_blocks=3"],
        ["linear_attention_0", "packing=2", "usable_blocks=6"],
    ]
    assert rows["unit.0"][1] == [
        "stride=16 B",
        "layer.0.key",
        "[0, 16): 16 B",
        "uint8 (16,)",
        "payload=16 B/parent",
        "padding=0 B/parent",
    ]
    assert rows["unit.0"][2] == [
        "stride=8 B",
        "layer.1.a",
        "[0, 3): 3 B",
        "uint8 (3,)",
        "padding [3, 4): 1 B",
        "layer.1.b",
        "[4, 8): 4 B",
        "float32 (1,)",
        "payload=14 B/parent",
        "padding=2 B/parent",
    ]
    assert "padding [31, 32): 1 B" in rows["unit.1"][1]
    assert rows["unit.1"][2] == ["hole=32 B/parent"]
    # The plane arrays include one reserved parent, even though usable block
    # counts exclude it. Field offsets above are relative to a child stride.
    assert rows["unit.1"][0] == ["unit.1", "parent=32 B", "arena_offset=64 B"]
    assert "usable_parents=3, reserved_null_parents=1" in report
    assert "logical_parent=48 B, resident_parent=48 B, layout_arena=192 B" in report


def test_parent_totals_count_shared_stride_padding_once():
    rows = _table_cells(format_cache_layout(_layout().bind(3)))

    assert rows["TOTAL / parent"] == [
        ["TOTAL / parent", "48 B"],
        [
            "payload=47 B",
            "padding=1 B",
            "hole=0 B",
            "unused=1 B",
            "utilization=97.92%",
        ],
        [
            "payload=14 B",
            "padding=2 B",
            "hole=32 B",
            "unused=34 B",
            "utilization=29.17%",
        ],
    ]


def test_pp_reports_only_resident_planes_and_preserves_global_packing():
    plan = _layout().bind(3).narrow_to_layers(1, 2)
    report = format_cache_layout(plan)
    rows = _table_cells(report)

    assert list(rows) == ["plane", "unit.0", "TOTAL / parent"]
    assert rows["unit.0"][1] == ["hole=16 B/parent"]
    assert "packing=2" in rows["plane"][2]
    assert rows["TOTAL / parent"][2] == [
        "payload=14 B",
        "padding=2 B",
        "hole=0 B",
        "unused=2 B",
        "utilization=87.50%",
    ]
    assert "logical_parent=48 B, resident_parent=16 B, layout_arena=64 B" in report
    assert "layer.0." not in report


def test_pp_retains_leading_and_trailing_padding_in_a_shared_stride():
    state = CacheGroupSpec("state", "state", replayable=False)
    layout = pack(
        (
            (
                state,
                (
                    CacheFieldSpec(
                        "layer.0.a", "shared", (3,), "uint8", exact_page_stride=False
                    ),
                    CacheFieldSpec(
                        "layer.1.b", "shared", (1,), "float32", exact_page_stride=False
                    ),
                ),
            ),
        ),
        prefix_granularity=16,
        cache_blocks_per_lcm_block={"state": 1},
        alignment=16,
        max_padding_fraction=float("inf"),
    )
    plan = layout.bind(3).narrow_to_layers(1, 2)
    rows = _table_cells(format_cache_layout(plan))

    assert rows["shared"][1] == [
        "stride=16 B",
        "padding [0, 4): 4 B",
        "layer.1.b",
        "[4, 8): 4 B",
        "float32 (1,)",
        "padding [8, 16): 8 B",
        "payload=4 B/parent",
        "padding=12 B/parent",
    ]


@pytest.mark.parametrize("parents", [1, 1_000_000_000])
def test_report_does_not_expand_parent_or_packing_counts(parents):
    layout = _layout()
    report = format_cache_layout(layout.bind(parents))
    rows = _table_cells(report)

    assert f"usable_blocks={parents * 2:,}" in rows["plane"][2]
    assert f"layout_arena={(parents + 1) * 48:,} B" in report
    assert len(report.splitlines()) == len(
        format_cache_layout(layout.bind(1)).splitlines()
    )


def test_empty_resident_window_reports_zero_bytes():
    report = format_cache_layout(_layout().bind(3).narrow_to_layers(2, 3))
    rows = _table_cells(report)

    assert "logical_parent=48 B, resident_parent=0 B, layout_arena=0 B" in report
    assert list(rows) == ["plane", "TOTAL / parent"]
    for cell in rows["TOTAL / parent"][1:]:
        assert cell == [
            "payload=0 B",
            "padding=0 B",
            "hole=0 B",
            "unused=0 B",
            "utilization=0.00%",
        ]
