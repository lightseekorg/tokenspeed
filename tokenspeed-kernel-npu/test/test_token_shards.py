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

"""Token shard packing: exact copies, padding, strides, and live replay."""

import pytest
import torch
from tokenspeed_kernel_npu.ops.token_shards import pad_token_shards

torch.npu.set_device(0)
torch.set_num_threads(4)


def reference(x, counts):
    out = x.new_zeros((len(counts) * max(counts), x.shape[1]))
    start = 0
    for rank, count in enumerate(counts):
        out[rank * max(counts) : rank * max(counts) + count] = x[start : start + count]
        start += count
    return out


@pytest.mark.parametrize(
    "counts",
    [[0, 0], [3], [1] * 6 + [0] * 2 + [1] * 6 + [0] * 2, [0, 2, 1], [24, 23, 0, 19]],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int32])
@pytest.mark.parametrize("strided", [False, True])
def test_pack_graph_exact(counts, dtype, strided):
    h = 513 if strided else 5120
    backing = torch.randint(
        -32, 32, (sum(counts) * 2, h * 2), device="npu", dtype=torch.int32
    ).to(dtype)
    x = backing[::2, ::2] if strided else backing[: sum(counts), :h].contiguous()
    pad_token_shards(x, counts)
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=torch.npu.Stream(device=torch.npu.current_device())):
        out = pad_token_shards(x, counts)
    for iteration in range(3):
        x.copy_(
            torch.randint(-32, 32, x.shape, device="npu", dtype=torch.int32).to(dtype)
        )
        before = backing.cpu().clone()
        g.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)
        torch.testing.assert_close(
            out.cpu(), reference(x.cpu(), counts), rtol=0, atol=0
        )


def test_invalid_layout():
    x = torch.ones(3, 8, device="npu")
    for counts in ([], [-1, 4], [2, 2]):
        with pytest.raises(ValueError):
            pad_token_shards(x, counts)


def test_float_bits_and_unaligned_source():
    counts = [2, 0, 1, 2]
    bits = torch.tensor(
        [0, -2147483648, 2139095040, -8388608, 2143363909], dtype=torch.int32
    )
    words = bits.repeat((5 * 64 + 4) // 5)[: 5 * 64].reshape(5, 64)
    x = words.npu().view(torch.float32)
    out = pad_token_shards(x, counts)
    expected = reference(words, counts)
    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)
    backing = torch.arange(5 * 64 + 1, device="npu", dtype=torch.int32)
    unaligned = backing[1:].reshape(5, 64)
    result = pad_token_shards(unaligned, counts)
    torch.testing.assert_close(
        result.cpu(), reference(unaligned.cpu(), counts), rtol=0, atol=0
    )


def test_more_than_sixteen_shards():
    counts = [1, 0] * 9
    x = torch.arange(9 * 64, device="npu", dtype=torch.int32).reshape(9, 64)
    torch.testing.assert_close(
        pad_token_shards(x, counts).cpu(), reference(x.cpu(), counts), rtol=0, atol=0
    )


@pytest.mark.parametrize("shards", [1, 2, 4])
@pytest.mark.parametrize("rows", [0, 1, 6, 32, 192])
def test_sum_shards_preserves_fp32_until_final_cast(shards, rows):
    from tokenspeed_kernel_npu.ops.token_shards import sum_token_shards

    x = torch.randn(shards * rows, 1024, device="npu", dtype=torch.float32)
    sum_token_shards(x, shards)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph, stream=torch.npu.Stream(device=torch.npu.current_device())
    ):
        actual = sum_token_shards(x, shards)
    for scale in (1.0, 1e-7, 128.0):
        x.copy_(torch.randn_like(x) * scale)
        before = x.cpu()
        graph.replay()
        torch.npu.synchronize()
        expected = before.reshape(shards, rows, 1024)[0].clone()
        for part in before.reshape(shards, rows, 1024)[1:]:
            expected += part
        torch.testing.assert_close(actual.cpu(), expected.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(x.cpu(), before, rtol=0, atol=0)


def test_sum_shards_cancellation_and_nonfinite_values():
    from tokenspeed_kernel_npu.ops.token_shards import sum_token_shards

    # Early BF16 rounding would lose the small residual after cancellation.
    x = torch.tensor(
        [[1.003, float("inf"), float("nan"), 0.0], [-1.0, -float("inf"), 2.0, -0.0]],
        device="npu",
    )
    actual = sum_token_shards(x, 2)
    expected = (x.cpu()[0:1] + x.cpu()[1:2]).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)


def test_sum_shards_rejects_invalid_layout():
    from tokenspeed_kernel_npu.ops.token_shards import sum_token_shards

    for tensor, shards in [
        (torch.ones(3, 8, device="npu"), 2),
        (torch.ones(2, 8, device="npu"), 0),
        (torch.ones(2, 8, device="npu").bfloat16(), 2),
        (torch.ones(2, 16, device="npu")[:, ::2], 2),
    ]:
        with pytest.raises(ValueError):
            sum_token_shards(tensor, shards)
