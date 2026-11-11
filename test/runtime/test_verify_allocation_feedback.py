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

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41

from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import DeepseekV41DSpark

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


def make_drafter():
    # Model initialization requires weights and GPUs. Exercise the real
    # request-state and allocation methods with their existing tensor state.
    drafter = object.__new__(DeepseekV41DSpark)
    drafter.verify_feedback = None
    drafter.confidence_scores = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32, device="cuda"
    )
    drafter._confidence_request_ids = ["a", "b", "c"]
    return drafter


def check_allocation(drafter, slots, history, expected):
    device = drafter.confidence_scores.device
    history = torch.tensor(history, dtype=torch.float32, device=device)
    expected = torch.tensor(expected, dtype=torch.float32, device=device)
    slots = torch.tensor(slots, dtype=torch.int64, device=device)
    bs, width = history.shape

    def outputs(q):
        return dict(
            rows_out=torch.empty(q, dtype=torch.int64, device=device),
            indptr_out=torch.empty(bs + 1, dtype=torch.int64, device=device),
            counts_out=torch.empty(bs, dtype=torch.int64, device=device),
            valid_out=torch.empty((), dtype=torch.bool, device=device),
        )

    original = history.clone()
    for q in range(bs, bs * (width + 1) + 1):
        actual, reference = outputs(q), outputs(q)
        dsv41.verify_layout(
            history,
            query_capacity=q,
            latest_scores=drafter.confidence_scores,
            request_slots=slots,
            **actual
        )
        dsv41.verify_layout(expected, query_capacity=q, **reference)
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], atol=0, rtol=0)
    torch.testing.assert_close(history, original, atol=0, rtol=0)


def test_allocation_uses_latest_scores_in_request_order():
    drafter = make_drafter()
    check_allocation(drafter, [2, 0], [[0, 0], [0, 0]], [[5, 6], [1, 2]])


def test_reused_slot_and_extend_do_not_leak_previous_predictions():
    drafter = make_drafter()
    drafter.prepare_request_state(["new", "b"], [0, 1], num_extends=0)
    check_allocation(drafter, [0, 1], [[-1, -2], [-3, -4]], [[-1, -2], [3, 4]])
    drafter.prepare_request_state(["b"], [1], num_extends=1)
    check_allocation(drafter, [1], [[-1, -2]], [[-1, -2]])


def test_nonfinite_prediction_falls_back_as_a_whole_row():
    drafter = make_drafter()
    drafter.confidence_scores[0, 1] = float("nan")
    drafter.confidence_scores[2, 0] = float("inf")
    check_allocation(
        drafter, [2, 1, 0], [[-1, -2], [-3, -4], [-5, -6]], [[-1, -2], [3, 4], [-5, -6]]
    )
