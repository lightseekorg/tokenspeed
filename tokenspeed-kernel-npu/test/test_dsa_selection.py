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

"""Selection constants keep capture storage and leave dynamic metadata live."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


def _inputs(rows):
    torch.manual_seed(421)
    return (
        torch.randn(rows, 64, 128, dtype=torch.bfloat16, device="npu") * 0.1,
        torch.randn(3, 128, 1, 128, dtype=torch.bfloat16, device="npu") * 0.1,
        torch.randn(rows, 64, dtype=torch.bfloat16, device="npu") * 0.1,
        torch.tensor([rows // 2, rows], dtype=torch.int32, device="npu"),
        torch.tensor([48, 64], dtype=torch.int32, device="npu"),
        torch.tensor([[1], [2]], dtype=torch.int32, device="npu"),
    )


def _uncached(kernels, inputs, initial, local):
    q, k, weights, ends, lengths, table = inputs
    output, _ = kernels._index(
        q,
        k,
        weights,
        actual_seq_lengths_query=ends,
        actual_seq_lengths_key=lengths,
        block_table=table,
        init_tensor=torch.full_like(ends, initial),
        local_tensor=torch.full(
            (q.shape[0],), local, dtype=torch.int32, device=q.device
        ),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
        pre_tokens=9223372036854775807,
        next_tokens=9223372036854775807,
        return_value=False,
    )
    return output


def _equal_selection(actual, expected):
    # The native indexer does not promise order within an identical selection.
    assert torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values)


def test_selection_windows_keep_values_and_storage_across_shapes():
    kernels = AscendDSAKernels()
    first = _inputs(8)
    out, _ = kernels.index(*first, 2048, 8, 16)
    _equal_selection(out, _uncached(kernels, first, 8, 16))
    initial, local = next(iter(kernels._selection_windows.values()))
    addresses = initial.data_ptr(), local.data_ptr()
    for inputs, init_value, local_value in [
        (first, 8, 16),
        (_inputs(16), 12, 20),
        (first, 12, 20),
        (first, 8, 16),
    ]:
        actual, _ = kernels.index(*inputs, 2048, init_value, local_value)
        _equal_selection(actual, _uncached(kernels, inputs, init_value, local_value))
    assert (initial.data_ptr(), local.data_ptr()) == addresses
    assert torch.equal(initial.cpu(), torch.full((2,), 8, dtype=torch.int32))
    assert torch.equal(local.cpu(), torch.full((8,), 16, dtype=torch.int32))
    assert len(kernels._selection_windows) == 3


def test_selection_graph_updates_lengths_after_another_shape_is_cached():
    kernels = AscendDSAKernels()
    inputs = _inputs(8)
    for _ in range(3):
        kernels.index(*inputs, 2048, 8, 16)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, _ = kernels.index(*inputs, 2048, 8, 16)
    kernels.index(*_inputs(16), 2048, 12, 20)
    for values in ([48, 64], [32, 40], [60, 72]):
        inputs[4].copy_(torch.tensor(values, dtype=torch.int32, device="npu"))
        graph.replay()
        _equal_selection(actual, _uncached(kernels, inputs, 8, 16))
