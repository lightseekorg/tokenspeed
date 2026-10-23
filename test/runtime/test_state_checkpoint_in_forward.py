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

"""Resolving a forward's in-forward State checkpoints and the scan's plan.

CPU-only: the rows that write a checkpoint, the packed position after each,
the receiving block of every State group, and where a chunked scan stores
them -- plus the refusals that keep the scheduler from publishing a block the
runtime never wrote.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (  # noqa: E402
    in_forward_state_checkpoints,
    scan_checkpoint_plan,
)

_GROUPS = ("state0", "state1")


def _tables(rows: list[list[int]]) -> dict[str, torch.Tensor]:
    """Both State groups' tables, the second group's blocks offset by 100."""
    table = torch.tensor(rows, dtype=torch.int32)
    return {"state0": table, "state1": torch.where(table > 0, table + 100, table)}


def test_checkpoint_rows_positions_and_blocks():
    # Three extend rows (300, 600 and 40 new tokens) and one decode row; the
    # second and the first write checkpoints 512 and 256 tokens in.
    tables = _tables([[3, 4, 7], [5, 6, 8], [9, 10, -1], [11, 12, -1]])
    checkpoints = in_forward_state_checkpoints(
        torch.tensor([256, 512, 0], dtype=torch.int32),
        torch.tensor([300, 600, 40], dtype=torch.int32),
        3,
        tables,
        _GROUPS,
    )
    assert checkpoints is not None
    assert checkpoints.rows.tolist() == [0, 1]
    assert checkpoints.offsets.tolist() == [256, 512]
    assert checkpoints.token_index.tolist() == [256, 300 + 512]
    assert checkpoints.blocks_by_group["state0"].tolist() == [7, 8]
    assert checkpoints.blocks_by_group["state1"].tolist() == [107, 108]


def test_no_checkpoint_resolves_to_none():
    tables = _tables([[3, 4], [5, 6]])
    lens = torch.zeros(2, dtype=torch.int32)
    seq = torch.tensor([300, 1], dtype=torch.int32)
    assert in_forward_state_checkpoints(lens, seq, 1, tables, _GROUPS) is None
    assert in_forward_state_checkpoints(None, seq, 1, tables, _GROUPS) is None


@pytest.mark.parametrize("lens", [None, torch.zeros(1, dtype=torch.int32)])
def test_checkpoint_blocks_without_positions_are_refused(lens):
    # The scheduler appends a third entry only for a row it gave a
    # checkpoint: a forward path that lost the positions would leave the
    # block unwritten while the scheduler publishes it.
    tables = _tables([[3, 4, 7]])
    with pytest.raises(RuntimeError, match="checkpoint"):
        in_forward_state_checkpoints(
            lens, torch.tensor([300], dtype=torch.int32), 1, tables, _GROUPS
        )


def test_a_checkpoint_needs_its_block_and_must_lie_inside_its_chunk():
    seq = torch.tensor([300], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="lack"):
        in_forward_state_checkpoints(
            torch.tensor([256], dtype=torch.int32), seq, 1, _tables([[3, 4]]), _GROUPS
        )
    with pytest.raises(
        RuntimeError,
        match=r"at or past the end of its chunk \(offset 300, chunk 300 tokens\)",
    ):
        in_forward_state_checkpoints(
            torch.tensor([300], dtype=torch.int32),
            seq,
            1,
            _tables([[3, 4, 7]]),
            _GROUPS,
        )


def test_the_scan_stores_at_the_offsets_gcd():
    # Offsets 512 and 768 share the interval 256: the 600-, 1000- and
    # 1-token sequences store 2, 3 and 0 states, and the checkpoints are the
    # second of the first sequence's and the third of the second's.
    tables = _tables([[3, 4, 7], [5, 6, 8], [9, 10, -1]])
    seq = torch.tensor([600, 1000, 1], dtype=torch.int32)
    checkpoints = in_forward_state_checkpoints(
        torch.tensor([512, 768], dtype=torch.int32), seq, 2, tables, _GROUPS
    )
    interval, cu_starts, rows = scan_checkpoint_plan(checkpoints, seq)
    assert interval == 256
    assert cu_starts.tolist() == [0, 2, 5, 5]
    assert rows.tolist() == [1, 4]


def test_a_single_checkpoint_is_the_scans_only_state():
    # One checkpoint 5120 tokens into an 8192-token chunk: the interval is
    # the offset itself, so the scan stores exactly that one state.
    tables = _tables([[3, 4, 7]])
    seq = torch.tensor([8192], dtype=torch.int32)
    checkpoints = in_forward_state_checkpoints(
        torch.tensor([5120], dtype=torch.int32), seq, 1, tables, _GROUPS
    )
    interval, cu_starts, rows = scan_checkpoint_plan(checkpoints, seq)
    assert (interval, cu_starts.tolist(), rows.tolist()) == (5120, [0, 1], [0])
