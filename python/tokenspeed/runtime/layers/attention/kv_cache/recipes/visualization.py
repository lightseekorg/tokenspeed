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

"""Text visualization of the bound cache layout, without allocating tensors."""

from __future__ import annotations

from collections import defaultdict

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldLayout,
    CacheMemoryPlan,
)


def _table(rows: list[list[list[str]]]) -> str:
    """Render multiline cells without terminal control sequences."""
    widths = [
        max(len(line) for row in rows for line in row[column])
        for column in range(len(rows[0]))
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [border]
    for row in rows:
        for index in range(max(map(len, row))):
            cells = [
                (cell[index] if index < len(cell) else "").ljust(width)
                for cell, width in zip(row, widths, strict=True)
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append(border)
    return "\n".join(lines)


def _stride_lines(fields: list[CacheFieldLayout]) -> list[str]:
    stride = fields[0].page_stride_bytes
    lines = [f"stride={stride:,} B"]
    offset = 0
    for field in sorted(fields, key=lambda field: field.field_offset_bytes):
        start = field.field_offset_bytes
        if start > offset:
            lines.append(f"padding [{offset:,}, {start:,}): {start - offset:,} B")
        end = start + field.payload_bytes
        lines.extend(
            (
                field.field_id,
                f"  [{start:,}, {end:,}): {field.payload_bytes:,} B",
                f"  {field.dtype} {field.shape}",
            )
        )
        offset = end
    if offset < stride:
        lines.append(f"padding [{offset:,}, {stride:,}): {stride - offset:,} B")
    return lines


def format_cache_layout(plan: CacheMemoryPlan) -> str:
    """Return an ASCII matrix with group names as columns and planes as rows.

    Args:
        plan: The bound memory plan, narrowed to resident layers when using PP.

    Returns:
        A log-ready table of physical packing, field ranges, shapes, dtypes,
        stride padding and unused planes. Byte totals describe a fully packed
        parent bound to each group independently, not live cache occupancy.
        Arena sizes include the reserved null parent; block counts exclude it.
    """
    fields_by_plane_group: dict[tuple[str, str], list[CacheFieldLayout]] = defaultdict(
        list
    )
    for field in plan.fields:
        fields_by_plane_group[field.plane_id, field.group_id].append(field)

    rows = [
        [["plane"]]
        + [
            [
                group.group_id,
                f"packing={group.cache_blocks_per_lcm_block}",
                f"usable_blocks={group.page_count - 1:,}",
            ]
            for group in plan.groups
        ]
    ]
    payloads = dict.fromkeys((group.group_id for group in plan.groups), 0)
    paddings = payloads.copy()
    holes = payloads.copy()
    for plane in plan.planes:
        row = [
            [
                plane.plane_id,
                f"parent={plane.bytes_per_lcm_block:,} B",
                f"arena_offset={plane.arena_offset_bytes:,} B",
            ]
        ]
        for group in plan.groups:
            fields = fields_by_plane_group[plane.plane_id, group.group_id]
            if not fields:
                row.append([f"hole={plane.bytes_per_lcm_block:,} B/parent"])
                holes[group.group_id] += plane.bytes_per_lcm_block
                continue
            # Several fields can share a stride. Count its padding once, not
            # once per field, including alignment gaps between the fields.
            payload = (
                sum(field.payload_bytes for field in fields)
                * group.cache_blocks_per_lcm_block
            )
            padding = plane.bytes_per_lcm_block - payload
            payloads[group.group_id] += payload
            paddings[group.group_id] += padding
            row.append(
                _stride_lines(fields)
                + [f"payload={payload:,} B/parent", f"padding={padding:,} B/parent"]
            )
        rows.append(row)

    resident_parent_bytes = sum(plane.bytes_per_lcm_block for plane in plan.planes)
    totals = [["TOTAL / parent", f"{resident_parent_bytes:,} B"]]
    for group in plan.groups:
        group_id = group.group_id
        payload = payloads[group_id]
        padding = paddings[group_id]
        hole = holes[group_id]
        utilization = payload / resident_parent_bytes if resident_parent_bytes else 0
        totals.append(
            [
                f"payload={payload:,} B",
                f"padding={padding:,} B",
                f"hole={hole:,} B",
                f"unused={padding + hole:,} B",
                f"utilization={utilization:.2%}",
            ]
        )
    rows.append(totals)
    return "\n".join(
        (
            f"prefix_granularity={plan.prefix_granularity:,} tokens, "
            f"usable_parents={plan.num_lcm_blocks:,}, reserved_null_parents=1",
            f"logical_parent={plan.lcm_block_bytes:,} B, "
            f"resident_parent={resident_parent_bytes:,} B, "
            f"layout_arena={plan.arena_bytes:,} B ({plan.arena_bytes / (1 << 30):.3f} GiB)",
            "Groups are alternative bindings of the same parent; do not sum columns.",
            "Packing and block counts are rank-local physical counts.",
            "Ranges [start, end) are bytes within one child stride, repeated packing times.",
            "Padding is unused space in an occupied plane; a hole is an unused plane.",
            _table(rows),
        )
    )
