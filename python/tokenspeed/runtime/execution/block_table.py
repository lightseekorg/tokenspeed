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

"""Selection helpers for scheduler-exported per-group block tables."""

from __future__ import annotations


def select_block_table(forward_op, group_id: str | None = None):
    tables = forward_op.block_tables_arrays()
    if group_id is not None:
        if group_id not in tables:
            raise ValueError(
                f"history_group_id {group_id!r} is missing from block tables "
                f"{sorted(tables)}"
            )
        return tables[group_id]
    if len(tables) != 1:
        raise ValueError(
            "history_group_id is required when the scheduler exports "
            f"{len(tables)} block tables"
        )
    return next(iter(tables.values()))


def unpadded_block_table_row(table, row_index: int) -> list[int]:
    row = list(table[row_index])
    if -1 in row:
        row = row[: row.index(-1)]
    return row
