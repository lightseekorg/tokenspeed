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

"""Request-token history: control-plane seeds, executor state, batch layout."""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.engine.scheduler_utils import (
    request_history_seeds_for_forward,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import RequestHistorySeeds


class _ForwardOp(SimpleNamespace):
    def num_extends(self) -> int:
        return len(self.extend_prefix_lens)


_STATES = {
    "a": SimpleNamespace(prompt_input_ids=[10, 11, 12, 13], output_ids=[20, 21]),
    "b": SimpleNamespace(prompt_input_ids=[30, 31], output_ids=[]),
}


def test_seeds_cover_prompt_and_output_prefixes() -> None:
    op = _ForwardOp(
        request_ids=["a", "b"],
        request_pool_indices=[1, 2],
        extend_prefix_lens=[5, 2],
    )
    assert request_history_seeds_for_forward(op, _STATES) == RequestHistorySeeds(
        slots=(1, 2),
        prefix_lengths=(5, 2),
        tokens=((10, 11, 12, 13, 20), (30, 31)),
    )


def test_seeds_skip_fresh_extends_and_decodes() -> None:
    fresh = _ForwardOp(
        request_ids=["a"], request_pool_indices=[1], extend_prefix_lens=[0]
    )
    assert request_history_seeds_for_forward(fresh, _STATES) is None
    # The decode row "b" follows the extend rows and never resumes a prefix.
    mixed = _ForwardOp(
        request_ids=["a", "b"], request_pool_indices=[1, 2], extend_prefix_lens=[0]
    )
    assert request_history_seeds_for_forward(mixed, _STATES) is None


def test_seed_prefix_must_exist() -> None:
    op = _ForwardOp(request_ids=["b"], request_pool_indices=[2], extend_prefix_lens=[3])
    with pytest.raises(ValueError, match="exceeds the physical tokens"):
        request_history_seeds_for_forward(op, _STATES)
    with pytest.raises(ValueError, match="equal lengths"):
        RequestHistorySeeds(slots=(1, 2), prefix_lengths=(0,), tokens=((),))
    with pytest.raises(ValueError, match="3 tokens for a 4-token prefix"):
        RequestHistorySeeds(slots=(1,), prefix_lengths=(4,), tokens=((1, 2, 3),))


def _runtime_states(capacity: int) -> RuntimeStates:
    states = RuntimeStates(
        req_pool_size=2, vocab_size=32, output_length=1, device="cpu"
    )
    states.init_request_token_history(capacity)
    return states


def test_runtime_states_seed_and_view_history() -> None:
    states = _runtime_states(8)
    assert states.has_request_token_history
    # One row per slot plus the graph-padding row.
    assert tuple(states.request_token_history_ids.shape) == (3, 8)
    states.seed_request_token_history(
        RequestHistorySeeds(
            slots=(1, 0), prefix_lengths=(5, 2), tokens=((4, 5, 6, 7, 8), (9, 3))
        )
    )
    assert states.request_token_history_ids[1, :5].tolist() == [4, 5, 6, 7, 8]
    assert states.request_token_history_ids[0, :2].tolist() == [9, 3]

    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    mask = torch.tensor([True, False])
    view = states.request_token_history_view(
        req_pool_indices=req_pool_indices,
        input_start_offsets=offsets,
        active_request_mask=mask,
    )
    assert view.history_token_ids is states.request_token_history_ids
    assert view.committed_lengths is states.valid_cache_lengths
    assert view.req_pool_indices is req_pool_indices
    assert view.input_start_offsets is offsets
    assert view.active_request_mask is mask


def test_runtime_states_reject_bad_seeds_and_disabled_history() -> None:
    states = _runtime_states(4)
    with pytest.raises(ValueError, match="slot 2 is out of range"):
        states.seed_request_token_history(
            RequestHistorySeeds(slots=(2,), prefix_lengths=(1,), tokens=((1,),))
        )
    with pytest.raises(ValueError, match="exceeds capacity 4"):
        states.seed_request_token_history(
            RequestHistorySeeds(slots=(0,), prefix_lengths=(5,), tokens=((1,) * 5,))
        )

    disabled = _runtime_states(0)
    assert not disabled.has_request_token_history
    with pytest.raises(RuntimeError, match="not enabled"):
        disabled.seed_request_token_history(
            RequestHistorySeeds(slots=(0,), prefix_lengths=(0,), tokens=((),))
        )
    with pytest.raises(ValueError, match="non-negative"):
        disabled.init_request_token_history(-1)


def _input_buffers() -> InputBuffers:
    return InputBuffers(
        max_bs=4, max_num_tokens=32, state_write_padding_pool_index=2, device="cpu"
    )


def test_layout_packs_extend_widths_then_decode_widths() -> None:
    buffers = _input_buffers()
    buffers.input_lengths_buf[:3] = torch.tensor([2, 1, 1], dtype=torch.int32)
    buffers.prepare_request_token_history_inputs(
        batch_size=3, num_extends=1, decode_width=8
    )
    assert buffers.input_start_offsets_buf[:4].tolist() == [0, 2, 10, 18]
    assert buffers.active_request_mask_buf[:3].tolist() == [True, True, True]
    with pytest.raises(ValueError, match="batch sizes"):
        buffers.prepare_request_token_history_inputs(
            batch_size=5, num_extends=1, decode_width=1
        )


def test_graph_layout_masks_padding_rows() -> None:
    buffers = _input_buffers()
    buffers.prepare_request_token_history_graph_inputs(
        active_bs=2, padded_bs=4, decode_width=2
    )
    assert buffers.input_start_offsets_buf.tolist() == [0, 2, 4, 6, 8]
    assert buffers.active_request_mask_buf.tolist() == [True, True, False, False]
    # Capture marks every row inactive so it never appends to live history.
    buffers.prepare_request_token_history_graph_inputs(
        active_bs=0, padded_bs=4, decode_width=2
    )
    assert not buffers.active_request_mask_buf.any()


def _draft_states() -> RuntimeStates:
    states = RuntimeStates(
        req_pool_size=4, vocab_size=100, device="cpu", output_length=1
    )
    states.init_request_token_history(16)
    return states


def test_draft_table_is_separate_and_frontier_is_explicit() -> None:
    states = _draft_states()
    states.init_draft_request_token_history(16)
    lengths = torch.zeros(5, dtype=torch.int32)
    lengths[1] = 3
    view = states.draft_request_token_history_view(
        req_pool_indices=torch.tensor([1]),
        input_start_offsets=torch.tensor([0, 1], dtype=torch.int32),
        active_request_mask=torch.tensor([True]),
        committed_lengths=lengths,
    )
    assert view.history_token_ids is not states.request_token_history_ids
    assert view.committed_lengths is lengths


def test_seeding_shifts_the_draft_stream_by_one() -> None:
    states = _draft_states()
    states.init_draft_request_token_history(16)
    states.seed_request_token_history(
        RequestHistorySeeds(slots=(1,), prefix_lengths=(4,), tokens=((7, 8, 9, 10),))
    )
    assert states.request_token_history_ids[1, :4].tolist() == [7, 8, 9, 10]
    assert states.draft_request_token_history_ids[1, :3].tolist() == [8, 9, 10]


def test_draft_table_requires_capacity_and_enablement() -> None:
    states = _draft_states()
    with pytest.raises(ValueError):
        states.init_draft_request_token_history(0)
    with pytest.raises(RuntimeError):
        states.draft_request_token_history_view(
            req_pool_indices=torch.tensor([0]),
            input_start_offsets=torch.tensor([0, 1], dtype=torch.int32),
            active_request_mask=torch.tensor([True]),
            committed_lengths=torch.zeros(5, dtype=torch.int32),
        )
