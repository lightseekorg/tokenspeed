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

"""Contracts for graph-safe grouped KDA state copy-on-write."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton import KdaGroupedStateCopyDescriptor


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA test")


def _component(pages: int, layer: int) -> torch.Tensor:
    # Dense sequence-major logical [feature, tap] view plus outer row padding.
    storage = torch.arange(pages * 32, dtype=torch.float32, device="cuda").to(
        torch.bfloat16
    )
    storage.add_(layer)
    return torch.as_strided(storage, (pages, 8, 3), (32, 1, 8))


def _recurrent_component(pages: int, layer: int) -> torch.Tensor:
    value = torch.arange(pages * 8, dtype=torch.float32, device="cuda")
    return value.view(pages, 2, 4).add_(layer)


def _fixture():
    conv = tuple(_component(10, layer) for layer in range(6))
    recurrent = tuple(_recurrent_component(10, layer) for layer in range(6))
    descriptor = KdaGroupedStateCopyDescriptor.build(
        conv, recurrent, [0, 0, 1, 1, 2, 2]
    )
    return (conv, recurrent), descriptor


def _expected(
    before: tuple[torch.Tensor, ...],
    groups: list[int],
    reads: tuple[torch.Tensor, ...],
    writes: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    result = tuple(value.clone() for value in before)
    read_cpu = [value.cpu().tolist() for value in reads]
    write_cpu = [value.cpu().tolist() for value in writes]
    for layer, group in enumerate(groups):
        for src, dst in zip(read_cpu[group], write_cpu[group], strict=True):
            if src == dst or dst < 0:
                continue
            if src < 0:
                result[layer][dst].zero_()
            else:
                result[layer][dst].copy_(before[layer][src])
    return result


@pytest.mark.parametrize("case", ["nonboundary", "mixed", "all_boundary", "padding"])
def test_grouped_state_copy_cases(case: str) -> None:
    _require_cuda()
    component_sets, descriptor = _fixture()
    before = tuple(
        tuple(value.clone() for value in components) for components in component_sets
    )
    same = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    reads = tuple(same.clone() for _ in range(3))
    writes = tuple(same.clone() for _ in range(3))
    if case == "mixed":
        writes[0][1] = 6
        writes[1][3] = 7
    elif case == "all_boundary":
        writes = tuple(value + 4 for value in reads)
    elif case == "padding":
        reads[0][0], writes[0][0] = -1, 6
        reads[1][1], writes[1][1] = -1, -1
        reads[2][2], writes[2][2] = 3, -1
    expected = tuple(
        _expected(values, [0, 0, 1, 1, 2, 2], reads, writes) for values in before
    )
    descriptor.copy(reads, writes, batch_size=4)
    for components, references in zip(component_sets, expected, strict=True):
        for actual, reference in zip(components, references, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_grouped_state_copy_descriptor_alias_and_lifetime() -> None:
    _require_cuda()
    (conv, recurrent), descriptor = _fixture()
    for table, components in (
        (descriptor.conv, conv),
        (descriptor.recurrent, recurrent),
    ):
        assert all(
            actual is expected
            for actual, expected in zip(table.components, components, strict=True)
        )
        assert table.addresses.tolist() == [value.data_ptr() for value in components]
    assert descriptor.conv.row_bytes == 48
    assert descriptor.conv.row_strides.tolist() == [8] * len(conv)
    assert descriptor.recurrent.row_bytes == 32
    assert descriptor.recurrent.row_strides.tolist() == [4] * len(recurrent)


def test_grouped_state_copy_cuda_graph_dynamic_indices() -> None:
    _require_cuda()
    component_sets, descriptor = _fixture()
    reads = tuple(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda") for _ in range(3)
    )
    writes = tuple(value.clone() for value in reads)
    descriptor.copy(reads, writes, batch_size=4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        descriptor.copy(reads, writes, batch_size=4)
    before = tuple(
        tuple(value.clone() for value in components) for components in component_sets
    )
    writes[0][0] = 6
    writes[2][3] = 7
    expected = tuple(
        _expected(values, [0, 0, 1, 1, 2, 2], reads, writes) for values in before
    )
    graph.replay()
    torch.cuda.synchronize()
    for components, references in zip(component_sets, expected, strict=True):
        for actual, reference in zip(components, references, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_grouped_state_copy_rejects_non_dense_inner_layout() -> None:
    _require_cuda()
    storage = torch.empty(200, dtype=torch.float32, device="cuda")
    component = torch.as_strided(storage, (4, 2, 3), (40, 10, 2))
    recurrent = torch.empty((4, 2, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="physically dense"):
        KdaGroupedStateCopyDescriptor.build([component], [recurrent], [0])
