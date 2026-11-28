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

"""Ascend interleaved indexer RoPE matches the float32 definition."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import interleave_rope

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend RoPE tests require an NPU"
)


def _cos_sin_cache(positions, rope_dim, *, dtype, device="npu"):
    max_position = int(positions.max()) + 1
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    freqs = torch.einsum(
        "i,j->ij", torch.arange(max_position, dtype=torch.float32), inv_freq
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device=device, dtype=dtype)


def _reference(x, cos_sin_cache, positions, rope_dim):
    cos_sin = cos_sin_cache.index_select(0, positions.flatten()).float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary = x[..., :rope_dim]
    first, second = rotary[..., ::2].float(), rotary[..., 1::2].float()
    rotated = torch.stack(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).flatten(-2)
    tail = x[..., rope_dim:].float()
    return torch.cat((rotated, tail), dim=-1).reshape(x.shape).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_interleave_rope_zero_sin_keeps_values(dtype):
    torch.npu.set_device(0)
    torch.manual_seed(11)
    rope_dim = 64
    positions = torch.arange(4, device="npu")
    cos_sin_cache = torch.zeros((4, rope_dim), dtype=dtype, device="npu")
    cos_sin_cache[:, : rope_dim // 2] = 1
    x = torch.randn(4, 3, 128, dtype=dtype, device="npu")

    actual = interleave_rope(x, positions, cos_sin_cache, rope_dim=rope_dim)

    torch.testing.assert_close(actual, x, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(("head_dim", "rope_dim"), [(128, 64), (128, 128), (64, 64)])
@torch.inference_mode()
def test_interleave_rope_matches_interleaved_reference(dtype, head_dim, rope_dim):
    torch.npu.set_device(0)
    torch.manual_seed(173)
    positions = torch.tensor([0, 7, 3, 15, 2, 9], dtype=torch.int64, device="npu")
    cos_sin_cache = _cos_sin_cache(positions, rope_dim, dtype=dtype)
    x = torch.randn(6, 4, head_dim, dtype=dtype, device="npu")
    expected = _reference(x, cos_sin_cache, positions, rope_dim)

    actual = interleave_rope(x, positions, cos_sin_cache, rope_dim=rope_dim)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_interleave_rope_accepts_two_dimensional_positions():
    torch.npu.set_device(0)
    torch.manual_seed(41)
    rope_dim = 64
    positions = torch.tensor([[0], [4], [11], [2]], dtype=torch.int64, device="npu")
    cos_sin_cache = _cos_sin_cache(positions, rope_dim, dtype=torch.bfloat16)
    x = torch.randn(4, 3, 128, dtype=torch.bfloat16, device="npu")
    expected = _reference(x, cos_sin_cache, positions, rope_dim)

    actual = interleave_rope(x, positions, cos_sin_cache, rope_dim=rope_dim)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_interleave_rope_casts_float32_cache(dtype):
    torch.npu.set_device(0)
    torch.manual_seed(29)
    rope_dim = 64
    positions = torch.tensor([0, 5, 1, 12], dtype=torch.int64, device="npu")
    cos_sin_cache = _cos_sin_cache(positions, rope_dim, dtype=torch.float32)
    x = torch.randn(4, 2, 128, dtype=dtype, device="npu")
    expected = _reference(x, cos_sin_cache, positions, rope_dim)

    actual = interleave_rope(x, positions, cos_sin_cache, rope_dim=rope_dim)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_interleave_rope_graph_replay():
    torch.npu.set_device(0)
    torch.manual_seed(37)
    rope_dim = 64
    positions = torch.tensor([0, 3, 9], dtype=torch.int64, device="npu")
    cos_sin_cache = _cos_sin_cache(positions, rope_dim, dtype=torch.float32)
    x = torch.randn(3, 2, 128, dtype=torch.bfloat16, device="npu")

    def run():
        return interleave_rope(x, positions, cos_sin_cache, rope_dim=rope_dim)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    for scale in (0.5, -2.0):
        x.mul_(scale)
        graph.replay()
        torch.npu.synchronize()
        expected = _reference(x, cos_sin_cache, positions, rope_dim)
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    graph.reset()
