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

"""Bitwise and dynamic-graph contracts for V4.1 AI CPU removal."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_attention import rope_inplace


@pytest.fixture(scope="module", autouse=True)
def device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)


def assert_bits(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    assert actual.dtype == expected.dtype
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    # Compare signed zero as well; NaN payloads are not part of the contract.
    assert torch.equal(actual.view(integer)[~nan], expected.view(integer)[~nan])


def old_rope(values, positions, table):
    width = table.shape[1]
    angles = table[positions.clamp_min(0).long()]
    shape = (positions.numel(),) + (1,) * (values.ndim - 2) + (width // 2,)
    cosine, sine = (x.reshape(shape) for x in angles.chunk(2, -1))
    tail = values[..., -width:].float()
    even, odd = tail[..., ::2], tail[..., 1::2]
    re = even * cosine - odd * sine
    ro = odd * cosine + even * sine
    values[..., -width::2].copy_(re)
    values[..., -width + 1 :: 2].copy_(ro)
    return values


def view(storage, layout):
    if layout == "kv":
        return storage[:, 0, :]
    if layout == "padded_heads":
        return storage[:, 1:5, :]
    if layout == "strided_channels":
        return storage[:, 1:5, ::2]
    return storage


def rope_inputs(batch, dtype, layout):
    torch.manual_seed(731 + batch)
    base = torch.randn(batch, 8, 512, dtype=dtype)
    angles = torch.randn(1024, 32)
    table = torch.cat((angles.cos(), angles.sin()), -1)
    positions = torch.arange(batch, dtype=torch.int64).mul(37).remainder(1024)
    if batch:
        positions[0] = -1
        positions[-1] = 1023
    return base, positions, table


@pytest.mark.parametrize("batch", [0, 1, 32, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["kv", "dense", "padded_heads", "strided_channels"])
def test_rope_matches_original_and_cpu(batch, dtype, layout):
    base, positions, table = rope_inputs(batch, dtype, layout)
    actual, original = base.npu(), base.npu()
    expected = base.clone()
    target = view(actual, layout)
    assert rope_inplace(target, positions.npu(), table.npu()) is target
    old_rope(view(original, layout), positions.npu(), table.npu())
    old_rope(view(expected, layout), positions, table)
    assert_bits(actual, original)
    assert_bits(actual, expected)  # Includes untouched heads and non-RoPE channels.


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["kv", "padded_heads", "strided_channels"])
def test_rope_dynamic_graph_replay(batch, dtype, layout):
    base, positions, table = rope_inputs(batch, dtype, layout)
    actual, original = base.npu(), base.npu()
    pos_storage = torch.zeros(batch * 2, device="npu", dtype=torch.int64)
    pos = pos_storage[::2]
    pos.copy_(positions)
    table_npu = table.npu()
    with torch.inference_mode():
        rope_inplace(view(actual, layout), pos, table_npu)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            rope_inplace(view(actual, layout), pos, table_npu)
        for step in range(3):
            base = (base.float() * -1.375).to(dtype)
            actual.copy_(base)
            original.copy_(base)
            pos.copy_((positions + step * 131).clamp_min(0).remainder(1024))
            table_npu.copy_(table * (-1 if step % 2 else 1))
            graph.replay()
            old_rope(view(original, layout), pos, table_npu)
            torch.npu.synchronize()
            assert_bits(actual, original)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_special_values(dtype):
    base = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0, 2.0**-14],
        dtype=dtype,
    ).repeat(2, 64)
    table = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float32).repeat(4, 16)
    pos = torch.tensor([0, 3], device="npu")
    actual, original = base.npu(), base.npu()
    rope_inplace(actual, pos, table.npu())
    old_rope(original, pos, table.npu())
    assert_bits(actual, original)


@pytest.mark.parametrize("rows,heads", [(24, 32), (48, 16), (96, 8), (192, 4)])
def test_rope_decode_tiles_mutable_strides(rows, heads):
    torch.manual_seed(756)
    values = torch.randn(rows, heads + 2, 512, device="npu", dtype=torch.bfloat16)[
        :, :heads
    ]
    positions_store = torch.empty(rows * 2, device="npu", dtype=torch.int64)
    positions = positions_store[::2]
    angles = torch.randn(128, 32, device="npu")
    table = torch.cat((angles.cos(), angles.sin()), dim=1)
    positions.fill_(0)
    rope_inplace(values, positions, table)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        rope_inplace(values, positions, table)
    for step in range(3):
        fresh = torch.randn(rows, heads, 512, dtype=torch.bfloat16)
        ids = torch.randint(-1, 128, (rows,), dtype=torch.int64)
        expected = old_rope(fresh.clone(), ids, table.cpu())
        values.copy_(fresh)
        positions.copy_(ids)
        graph.replay()
        torch.npu.synchronize()
        assert_bits(values, expected)
