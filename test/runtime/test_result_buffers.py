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

import gc
import weakref

import pytest
import torch

from tokenspeed.runtime.engine.result_buffers import completed_result, result_buffer
from tokenspeed.runtime.execution.types import ModelExecutionResult


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_result_buffer_preserves_values_shape_dtype_and_storage(dtype):
    tensor = torch.arange(12, dtype=dtype, device="cpu").reshape(3, 4)[:, ::2]
    view = result_buffer(tensor)
    assert view.shape == (3, 2)
    assert view.itemsize == tensor.element_size()
    assert view.tolist() == [[0, 2], [4, 6], [8, 10]]
    # Exporting is a view. The Rust query copies it while the result owns the
    # storage, before its reuse lease can be released.
    tensor[0, 0] = 27
    assert view.tolist()[0][0] == 27
    del tensor
    gc.collect()
    assert view.tolist() == [[27, 2], [4, 6], [8, 10]]
    owner = weakref.ref(view)
    del view
    gc.collect()
    assert owner() is None


def test_missing_and_empty_result_buffers():
    assert result_buffer(None) is None
    assert (
        result_buffer(torch.empty(0, dtype=torch.float32, device="cpu")).tolist() == []
    )


def test_result_export_rejects_a_tensor_that_is_not_on_the_cpu():
    tensor = torch.empty(2, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="already be on the CPU"):
        result_buffer(tensor)


class _CopyEvent:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


def test_completed_result_exports_every_column_after_the_copy_lands():
    event = _CopyEvent()
    result = ModelExecutionResult(
        output_tokens=torch.tensor([7], dtype=torch.int32, device="cpu"),
        output_lengths=torch.tensor([1], dtype=torch.int32, device="cpu"),
        output_logprobs=torch.tensor([-0.5], dtype=torch.float32, device="cpu"),
        output_nan_flags=torch.tensor([0], dtype=torch.int32, device="cpu"),
        next_input_ids=torch.tensor([[1, 2]], dtype=torch.int32, device="cpu"),
        input_token_logprobs=torch.tensor(
            [-0.25, -1.0], dtype=torch.float32, device="cpu"
        ),
        input_top_logprobs_val=torch.tensor(
            [[-0.25, -2.0], [-1.0, -3.0]], dtype=torch.float32, device="cpu"
        ),
        input_top_logprobs_idx=torch.tensor(
            [[70000, 1], [70001, 2]], dtype=torch.int64, device="cpu"
        ),
        output_top_logprobs_val=torch.tensor(
            [[-0.5, -2.0]], dtype=torch.float32, device="cpu"
        ),
        output_top_logprobs_idx=torch.tensor([[7, 8]], dtype=torch.int64, device="cpu"),
        output_token_cumulative_probs=torch.tensor(
            [0.875], dtype=torch.float32, device="cpu"
        ),
        copy_event=event,
    )
    result.sync()
    exported = completed_result(
        result, serving_compiles=(3, 0.25), grammar_terminated_at=[2]
    )
    assert event.waits == 1
    assert exported[:2] == ([7], [1])
    assert exported[2].tolist() == [-0.5]
    assert exported[3:7] == ([0], [[1, 2]], (3, 0.25), [2])
    assert exported[7][0].tolist() == [-0.25, -1.0]
    assert exported[7][1][0].tolist() == [[-0.25, -2.0], [-1.0, -3.0]]
    assert exported[7][1][1].tolist() == [[70000, 1], [70001, 2]]
    assert exported[7][2][0].tolist() == [[-0.5, -2.0]]
    assert exported[7][2][1].tolist() == [[7, 8]]
    del result
    gc.collect()
    assert exported[8].tolist() == [0.875]


def test_completed_result_keeps_absent_columns_absent():
    result = ModelExecutionResult(
        output_tokens=torch.tensor([7], dtype=torch.int32, device="cpu"),
        output_lengths=torch.tensor([1], dtype=torch.int32, device="cpu"),
    )
    exported = completed_result(
        result, serving_compiles=None, grammar_terminated_at=None
    )
    assert exported == ([7], [1], None, None, None, None, None, None, None)
