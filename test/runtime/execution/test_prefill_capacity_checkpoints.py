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

"""Fixed in-forward State checkpoint slots of capacity-shaped prefill metadata.

A captured scan stores a state at every page; each forward only rewrites, at
stable addresses, which stored state goes to which checkpoint block. These run
on CPU tensors: the slots are plain tensors and one packed upload.
"""

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (  # noqa: E402
    in_forward_state_checkpoints,
)
from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (  # noqa: E402
    CapacityCheckpointLayout,
    CapacityPrefillMetadata,
    PrefillCapacity,
    _checkpoint_slots,
    _refresh_checkpoint_slots,
)

PAGE = 64


def _layout(rows, every_n_tokens=PAGE, groups=("state",)):
    return CapacityCheckpointLayout(
        every_n_tokens=every_n_tokens,
        states=torch.empty(rows, 2, 4, 4),
        group_ids=groups,
    )


def _shape(token_capacity, num_sequences, layout):
    """A retained shape's metadata as far as its checkpoint slots go."""
    capacity = PrefillCapacity(token_capacity, num_sequences)
    checkpoints, scan, rows = _checkpoint_slots(capacity, layout, "cpu")
    return CapacityPrefillMetadata(
        query_start_loc=None,
        scan_query_start_loc=None,
        extend_seq_lens_cpu=torch.ones(num_sequences, dtype=torch.int32),
        state_checkpoints=checkpoints,
        scan_state_checkpoints=scan,
        scan_checkpoint_rows=rows,
        capacity=capacity,
    )


def _live(lengths, offsets, group="state"):
    """The scheduler's checkpoints: a third State-table block per checkpoint row."""
    table = torch.tensor(
        [[0, 1 + row, 20 + row] for row in range(len(lengths))], dtype=torch.int32
    )
    return in_forward_state_checkpoints(
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int32),
        len(lengths),
        {group: table},
        (group,),
    )


def _refresh(target, lengths, live):
    # As the capacity refresh leaves them: live lengths, then one scanned
    # row per unused request slot.
    target.extend_seq_lens_cpu[: len(lengths)] = torch.tensor(lengths)
    target.extend_seq_lens_cpu[len(lengths) :] = 1
    _refresh_checkpoint_slots(target, live)


def test_live_checkpoints_take_their_page_rows_and_unused_slots_write_nothing():
    target = _shape(512, 4, _layout(512 // PAGE))
    slots = target.state_checkpoints
    tensors = (
        slots.token_index,
        slots.blocks_by_group["state"],
        target.scan_state_checkpoints.cu_starts,
        target.scan_checkpoint_rows,
    )
    addresses = [tensor.data_ptr() for tensor in tensors]

    live = _live([300, 129, 70], [256, 128, 0])
    _refresh(target, [300, 129, 70], live)
    # The scan stores length // page states per sequence, none for the
    # unused slot's one row; a checkpoint takes the row of its offset.
    assert target.scan_state_checkpoints.cu_starts.tolist() == [0, 4, 6, 7, 7]
    assert target.scan_checkpoint_rows.tolist() == [3, 5, 0, 0]
    assert target.state_checkpoints.token_index.tolist() == [256, 428, 512, 512]
    assert target.state_checkpoints.blocks_by_group["state"].tolist() == [
        20,
        21,
        -1,
        -1,
    ]
    assert target.state_checkpoints.rows.tolist() == [0, 1]
    assert target.state_checkpoints.offsets.tolist() == [256, 128]

    # A later forward without checkpoints clears every slot in place.
    _refresh(target, [448], None)
    assert target.scan_state_checkpoints.cu_starts.tolist() == [0, 7, 7, 7, 7]
    assert target.scan_checkpoint_rows.tolist() == [0, 0, 0, 0]
    assert target.state_checkpoints.token_index.tolist() == [512] * 4
    assert target.state_checkpoints.blocks_by_group["state"].tolist() == [-1] * 4
    assert target.state_checkpoints.rows.numel() == 0
    slots = target.state_checkpoints
    assert addresses == [
        tensor.data_ptr()
        for tensor in (
            slots.token_index,
            slots.blocks_by_group["state"],
            target.scan_state_checkpoints.cu_starts,
            target.scan_checkpoint_rows,
        )
    ]


def test_every_request_slot_can_hold_a_checkpoint():
    target = _shape(256, 2, _layout(256 // PAGE))
    _refresh(target, [128, 128], _live([128, 128], [64, 64]))
    assert target.scan_state_checkpoints.cu_starts.tolist() == [0, 2, 4]
    assert target.scan_checkpoint_rows.tolist() == [0, 2]
    assert target.state_checkpoints.token_index.tolist() == [64, 192]
    assert target.state_checkpoints.blocks_by_group["state"].tolist() == [20, 21]


def test_a_checkpoint_off_the_captured_interval_is_refused():
    target = _shape(512, 2, _layout(512 // 128, every_n_tokens=128))
    with pytest.raises(ValueError, match="128-token scan interval"):
        _refresh(target, [300], _live([300], [64]))


def test_checkpoints_of_another_state_group_are_refused():
    target = _shape(512, 2, _layout(512 // PAGE))
    with pytest.raises(RuntimeError, match="state groups changed"):
        _refresh(target, [300], _live([300], [256], group="other"))


def test_the_scan_scratch_must_hold_the_shapes_states():
    with pytest.raises(ValueError, match="fewer states"):
        _shape(512, 1, _layout(512 // PAGE - 1))
    # A shape shorter than one page still keeps one row for unused slots.
    with pytest.raises(ValueError, match="fewer states"):
        _shape(32, 1, _layout(0))
    _shape(32, 1, _layout(1))
