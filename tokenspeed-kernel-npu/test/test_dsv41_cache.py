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

"""Ascend fused KV bytes against an independent CPU nearest-value oracle."""

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU device
from tokenspeed_kernel.numerics.torch.attention.dsv41 import torch_dsv41_cache_quantize
from tokenspeed_kernel_npu.ops import dsv41_cache as op


@pytest.fixture(scope="module", autouse=True)
def device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)


@pytest.mark.parametrize("fmt", ["swa", "global"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_codec_rounding_boundaries(fmt, dtype):
    torch.manual_seed(41)
    dim, group, _, width = op.LAYOUTS[fmt]
    rows = torch.randn(37, dim)
    rows[0].zero_()
    rows[1].fill_(2.0**-20)
    rows[2].fill_(6.375 if fmt == "global" else 6)
    rows[3].fill_(1.875 * (448 if fmt == "swa" else 6))
    rows[4].reshape(-1, group)[:, 0] = -0.0
    rows[5].fill_(6 * 2.0**-126)
    rows[6].fill_(2.0**-127)
    rows[7].fill_(-(2.0**-127))
    rows = rows.to(dtype)
    expected_bytes, expected = torch_dsv41_cache_quantize(rows, fmt)
    actual = op.cache_pack(rows.npu(), fmt, None)
    torch.testing.assert_close(actual.cpu(), expected_bytes, rtol=0, atol=0)
    for out_dtype in (torch.float32, torch.bfloat16):
        out = torch.empty((37, dim), dtype=out_dtype, device="npu")
        op.cache_unpack(actual, fmt, out)
        torch.testing.assert_close(out.cpu(), expected.to(out_dtype), rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["swa", "global"])
def test_qdq_resident_cache_and_graph_replay(fmt):
    torch.manual_seed(412)
    storage = torch.full((8, 64, 1024), -7.0, dtype=torch.bfloat16, device="npu")
    cache = storage[::2, :, ::2]
    x = torch.randn(32, 1024, device="npu", dtype=torch.bfloat16)[:, ::2]
    slots = torch.arange(64, device="npu")[::2]
    slots[-2:] = torch.tensor([-1, 256], device="npu")
    selections = torch.arange(32769, device="npu") % 258 - 1
    mode = fmt + "_bf16_qdq"

    def run():
        op.cache_scatter(x, cache, slots, mode)
        return op.cache_gather(cache, selections, mode, None)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        out = run()
    for step in range(3):
        storage.fill_(-7)
        x.normal_()
        slots[:30].copy_(torch.arange(30, device="npu") + step * 64)
        graph.replay()
        torch.npu.synchronize()
        _, decoded = torch_dsv41_cache_quantize(x.cpu(), fmt)
        expected = torch.full(storage.shape, -7.0, dtype=torch.bfloat16)
        expected_cache = expected[::2, :, ::2]
        ids = slots.cpu()
        valid = (ids >= 0) & (ids < 256)
        expected_cache[ids[valid] // 64, ids[valid] % 64] = decoded.bfloat16()[valid]
        torch.testing.assert_close(storage.cpu(), expected, atol=0, rtol=0)
        pick = selections.cpu()
        valid = (pick >= 0) & (pick < 256)
        pick = pick.clamp(0, 255)
        want = expected_cache[pick // 64, pick % 64].masked_fill(~valid[:, None], 0)
        torch.testing.assert_close(out.cpu(), want, atol=0, rtol=0)


@pytest.mark.parametrize("fmt", ["swa", "global", "index"])
def test_resident_packed_cache_is_rejected(fmt):
    x = torch.empty((0, 512), device="npu", dtype=torch.bfloat16)
    cache = torch.empty((0, 64, 288), device="npu", dtype=torch.uint8)
    slots = torch.empty(0, device="npu", dtype=torch.int64)
    with pytest.raises(ValueError, match="BF16"):
        op.cache_scatter(x, cache, slots, fmt)
    with pytest.raises(ValueError, match="BF16"):
        op.cache_gather(cache, slots, fmt, None)
