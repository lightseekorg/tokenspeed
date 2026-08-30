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
from tokenspeed_kernel.ops.kvcache.triton import GroupedStateCopyDescriptor


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA test")


def _component(pages: int, layer: int) -> torch.Tensor:
    # Dense sequence-major logical [feature, tap] view plus outer row padding.
    storage = torch.arange(pages * 24, dtype=torch.float32, device="cuda").to(
        torch.bfloat16
    )
    storage.add_(layer)
    return torch.as_strided(storage, (pages, 6, 3), (24, 1, 6))


def _fixture():
    components = tuple(_component(10, layer) for layer in range(6))
    descriptor = GroupedStateCopyDescriptor.build(components, [0, 0, 1, 1, 2, 2])
    return components, descriptor


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
    components, descriptor = _fixture()
    before = tuple(value.clone() for value in components)
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
    expected = _expected(before, [0, 0, 1, 1, 2, 2], reads, writes)
    descriptor.copy(reads, writes, batch_size=4)
    for actual, reference in zip(components, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_grouped_state_copy_descriptor_alias_and_lifetime() -> None:
    _require_cuda()
    components, descriptor = _fixture()
    assert all(
        actual is expected
        for actual, expected in zip(descriptor.components, components, strict=True)
    )
    assert descriptor.row_bytes == 36
    assert descriptor.addresses.tolist() == [value.data_ptr() for value in components]
    assert descriptor.row_strides.tolist() == [12] * len(components)


def test_grouped_state_copy_cuda_graph_dynamic_indices() -> None:
    _require_cuda()
    components, descriptor = _fixture()
    reads = tuple(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda") for _ in range(3)
    )
    writes = tuple(value.clone() for value in reads)
    descriptor.copy(reads, writes, batch_size=4)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        descriptor.copy(reads, writes, batch_size=4)
    before = tuple(value.clone() for value in components)
    writes[0][0] = 6
    writes[2][3] = 7
    expected = _expected(before, [0, 0, 1, 1, 2, 2], reads, writes)
    graph.replay()
    torch.cuda.synchronize()
    for actual, reference in zip(components, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_grouped_state_copy_rejects_non_dense_inner_layout() -> None:
    _require_cuda()
    storage = torch.empty(200, dtype=torch.float32, device="cuda")
    component = torch.as_strided(storage, (4, 2, 3), (40, 10, 2))
    with pytest.raises(ValueError, match="physically dense"):
        GroupedStateCopyDescriptor.build([component], [0])
