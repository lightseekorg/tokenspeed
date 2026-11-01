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

"""The force-single epilogue writes the sampler's caller-owned length slice."""

import pytest
import torch

from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_executor import ModelExecutor


@pytest.mark.parametrize("force", [[False, False, False], [False, True, False]])
def test_force_single_verify_preserves_the_output_buffer(force):
    executor = ModelExecutor.__new__(ModelExecutor)
    inputs = InputBuffers.__new__(InputBuffers)
    inputs.force_single_token_verify_buf = torch.tensor([True, *force, True])
    executor.input_buffers = inputs
    storage = torch.tensor([99, 4, 2, 3, 99], dtype=torch.int32)
    lengths = storage[1:4]

    returned = executor._apply_force_single_token_verify(lengths, 1, 3, [-1, 7, -1])

    expected = [4, 1 if force[1] else 2, 3]
    assert returned.tolist() == expected
    assert storage.tolist() == [99, *expected, 99]
    # A later stop/state update must modify the buffer the sampler owns.
    returned[0] = 0
    assert lengths.tolist() == [0, *expected[1:]]


def test_no_explicit_decode_ids_keeps_the_existing_output_slice():
    executor = ModelExecutor.__new__(ModelExecutor)
    lengths = torch.tensor([3, 2], dtype=torch.int32)
    returned = executor._apply_force_single_token_verify(lengths, 0, 2, None)
    returned[1] = 0
    assert lengths.tolist() == [3, 0]
