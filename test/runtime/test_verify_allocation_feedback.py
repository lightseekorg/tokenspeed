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

import torch

from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import DeepseekV41DSpark


def make_drafter():
    # Model initialization requires weights and GPUs. Exercise the real
    # request-state and allocation methods with their existing tensor state.
    drafter = object.__new__(DeepseekV41DSpark)
    drafter.verify_feedback = None
    drafter.confidence_scores = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
    )
    drafter._confidence_request_ids = ["a", "b", "c"]
    return drafter


def test_allocation_uses_latest_scores_in_request_order():
    drafter = make_drafter()
    history = torch.zeros((2, 2))
    result = drafter._allocation_scores(torch.tensor([2, 0]), history)
    torch.testing.assert_close(result, torch.tensor([[5.0, 6.0], [1.0, 2.0]]))
    torch.testing.assert_close(history, torch.zeros((2, 2)))


def test_reused_slot_and_extend_do_not_leak_previous_predictions():
    drafter = make_drafter()
    drafter.prepare_request_state(["new", "b"], [0, 1], num_extends=0)
    history = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    result = drafter._allocation_scores(torch.tensor([0, 1]), history)
    torch.testing.assert_close(result, torch.tensor([[-1.0, -2.0], [3.0, 4.0]]))
    drafter.prepare_request_state(["b"], [1], num_extends=1)
    result = drafter._allocation_scores(torch.tensor([1]), history[:1])
    torch.testing.assert_close(result, history[:1])


def test_nonfinite_prediction_falls_back_as_a_whole_row():
    drafter = make_drafter()
    drafter.confidence_scores[0, 1] = float("nan")
    drafter.confidence_scores[2, 0] = float("inf")
    history = torch.tensor([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])
    result = drafter._allocation_scores(torch.tensor([2, 1, 0]), history)
    torch.testing.assert_close(
        result, torch.tensor([[-1.0, -2.0], [3.0, 4.0], [-5.0, -6.0]])
    )
