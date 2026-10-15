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

"""Indexer rotation agrees with CANN and an independent FP32 formula."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import (
    _composite_interleave_rope,
    interleave_rope,
)


@pytest.mark.parametrize("rows", [1, 8, 64, 65])
@pytest.mark.parametrize("heads", [1, 8, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_rope_matches_cann(rows, heads, dtype, strided):
    torch.npu.set_device(0)
    torch.manual_seed(421)
    shape = (rows, 128) if heads == 1 else (rows, heads, 128)
    base = torch.randn(
        (*shape[:-1], 256 if strided else 128), dtype=dtype, device="npu"
    )
    x = base[..., :128]
    table = torch.randn(8192, 64, dtype=dtype, device="npu")
    positions = (torch.arange(rows, device="npu") * 97) % 8192
    reference = _composite_interleave_rope(x, positions, table, rope_dim=64)
    actual = interleave_rope(x, positions, table, rope_dim=64)
    assert torch.equal(actual, reference)
    # Independently check the pair mapping and each FP32 multiply/add on CPU.
    source = x.cpu().float()
    cos, sin = table.cpu().float()[positions.cpu()].chunk(2, dim=-1)
    if x.ndim == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    even, odd = source[..., :64:2], source[..., 1:64:2]
    expected = torch.cat(
        (even * cos - odd * sin, odd * cos + even * sin, source[..., 64:]),
        dim=-1,
    ).to(dtype)
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(actual[..., 64:], x[..., 64:])


def test_rope_graph_reads_updated_positions():
    x = torch.randn(64, 64, 128, dtype=torch.bfloat16, device="npu")
    table = torch.randn(8192, 64, dtype=torch.bfloat16, device="npu")
    positions = torch.arange(64, device="npu")
    for _ in range(3):
        interleave_rope(x, positions, table, rope_dim=64)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = interleave_rope(x, positions, table, rope_dim=64)
    for offset in (0, 127, 4096):
        positions.copy_(torch.arange(64, device="npu") + offset)
        graph.replay()
        reference = _composite_interleave_rope(x, positions, table, rope_dim=64)
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("heads", [3, 65])
def test_rope_non_power_of_two_heads_and_strides(heads):
    x = torch.randn(8, heads, 256, dtype=torch.bfloat16, device="npu")[..., ::2]
    table = torch.randn(8192, 128, dtype=torch.bfloat16, device="npu")[:, ::2]
    positions = torch.arange(16, device="npu")[::2]
    expected = _composite_interleave_rope(x, positions, table, rope_dim=64)
    assert torch.equal(interleave_rope(x, positions, table, rope_dim=64), expected)


def test_rope_preserves_empty_and_shaped_positions():
    table = torch.randn(256, 64, dtype=torch.bfloat16, device="npu")
    for rows in (0, 8):
        x = torch.randn(rows, 8, 128, dtype=torch.bfloat16, device="npu")
        positions = torch.arange(rows, device="npu").reshape(rows, 1)
        expected = _composite_interleave_rope(x, positions, table, rope_dim=64)
        assert torch.equal(interleave_rope(x, positions, table, rope_dim=64), expected)
