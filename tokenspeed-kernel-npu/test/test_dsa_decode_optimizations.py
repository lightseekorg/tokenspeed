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

"""DSA layout, RoPE and cache-write contracts on a visible Ascend device."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import (
    AscendDSAKernels,
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
    assert torch.equal(interleave_rope(x, positions, table, rope_dim=64), reference)


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


@pytest.mark.parametrize("rows,heads", [(1, 8), (64, 8), (64, 32), (64, 64)])
def test_value_projection_layout(rows, heads):
    x = torch.randn(heads, rows, 512, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(heads, 512, 128, dtype=torch.bfloat16, device="npu")
    reference = torch.bmm(x, weight).transpose(0, 1).contiguous().flatten(1)
    actual = AscendDSAKernels.project_value(x, weight)
    error = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert error < 1e-3
    assert actual.shape == reference.shape
    assert actual.is_contiguous()


def test_index_scatter_preserves_invalid_and_unwritten_slots():
    kernels = AscendDSAKernels()
    rows = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="npu")
    cache = torch.full((3, 128, 1, 128), 7, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([-1, 0, 127, 128, 383, 384], dtype=torch.int32, device="npu")
    reference = cache.cpu()
    reference[1, 0] = rows[3].cpu()
    reference[2, 127] = rows[4].cpu()
    kernels.scatter(rows, cache, slots)
    assert torch.equal(cache.cpu(), reference)


@pytest.mark.parametrize("heads", [8, 32, 64])
def test_packed_sfad_matches_split_query(heads):
    kernels = AscendDSAKernels()
    rows = 8
    q = torch.randn(rows, heads, 576, dtype=torch.bfloat16, device="npu") * 0.1
    cache = torch.randn(3, 128, 1, 576, dtype=torch.bfloat16, device="npu") * 0.1
    indices = torch.full((rows, 1, 2048), -1, dtype=torch.int32, device="npu")
    indices[:, :, :32] = torch.arange(32, dtype=torch.int32, device="npu")
    chunks = torch.full((rows,), 16, dtype=torch.int32, device="npu")
    ends = torch.arange(1, rows + 1, dtype=torch.int32, device="npu")
    lengths = torch.full((rows,), 32, dtype=torch.int32, device="npu")
    table = torch.ones(rows, 1, dtype=torch.int32, device="npu")
    padded = q.repeat(1, 32 // heads, 1) if heads < 32 else q
    reference = kernels.attention(
        padded[..., :512].contiguous(),
        padded[..., 512:].contiguous(),
        cache,
        indices,
        chunks,
        ends,
        lengths,
        table,
        192**-0.5,
    )[:, :heads]
    for major in (False, True):
        actual = kernels.attention_packed(
            q,
            cache,
            indices,
            chunks,
            ends,
            lengths,
            table,
            192**-0.5,
            head_major_output=major,
        )
        expected = (
            reference.transpose(0, 1).contiguous() if major else reference.flatten(1)
        )
        assert torch.equal(actual, expected)


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
