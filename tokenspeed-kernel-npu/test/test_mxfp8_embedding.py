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

"""Fused Engram gather preserves E4M3/E8M0 decoding and graph refresh."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.embedding import mxfp8_embedding
from tokenspeed_kernel_npu.ops.dsv41_engram_decode import fp8_e8m0_rows


def reference(weight, scales, ids, start, end):
    local = (ids >= start) & (ids < end)
    if not weight.numel():
        return torch.zeros((*ids.shape, weight.shape[1]), dtype=torch.bfloat16)
    idx = (ids.cpu().long() - start).clamp(0, weight.shape[0] - 1)
    codes = weight.cpu().contiguous().view(torch.float8_e4m3fn).float()
    scale = scales.cpu().contiguous().view(torch.float8_e8m0fnu).float()
    value = (codes[idx] * scale[idx].repeat_interleave(32, -1)).bfloat16()
    return value.masked_fill(~local.cpu().unsqueeze(-1), 0)


@pytest.mark.parametrize(
    "dim,capacity,count,dtype",
    [
        (32, 0, 7, torch.int64),
        (32, 259, 0, torch.int32),
        (32, 259, 1, torch.int32),
        (64, 259, 193, torch.int64),
        (256, 259, 4608, torch.int64),
        (512, 259, 53, torch.int64),
    ],
)
def test_fused_gather_native_graph(dim, capacity, count, dtype):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(911)
    # Padded rows exercise non-contiguous leading strides, with unit columns.
    wc = torch.randint(0, 256, (capacity, dim + 32), dtype=torch.uint8)
    sc = torch.randint(0, 256, (capacity, dim // 32 + 1), dtype=torch.uint8)
    w = wc.npu()[:, :dim]
    s = sc.npu()[:, : dim // 32]
    start = 20000000 if dtype == torch.int32 else 2**40
    end = start + max(0, capacity - 3)
    ids = torch.empty(count, device="npu", dtype=dtype)

    def update(step):
        ids.copy_(
            torch.arange(count, device="npu", dtype=dtype) % (capacity + 5) + start - 2
        )
        if step == 1:
            ids.fill_(start - 1)
        elif step == 2:
            ids.fill_(end)
        elif step == 3:
            w.copy_(torch.randint(0, 256, w.shape, device="npu", dtype=torch.uint8))
            s.copy_(torch.randint(0, 256, s.shape, device="npu", dtype=torch.uint8))

    def run():
        return mxfp8_embedding(w, s, ids, start, end)

    update(0)
    y = run()
    torch.npu.synchronize()
    torch.testing.assert_close(
        y.cpu(), reference(w, s, ids, start, end), atol=0, rtol=0, equal_nan=True
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        y = run()
    for step in range(4):
        update(step)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            y.cpu(), reference(w, s, ids, start, end), atol=0, rtol=0, equal_nan=True
        )


def test_all_codes_and_scales_match_existing_decoder():
    torch.npu.set_device(0)
    # Every E4M3 byte paired with every E8M0 byte, including subnormals and NaNs.
    w = torch.arange(256, dtype=torch.uint8).repeat(256, 1).npu()
    s = torch.arange(256, dtype=torch.uint8)[:, None].expand(256, 8).contiguous().npu()
    ids = torch.arange(256, device="npu", dtype=torch.int64)
    actual = mxfp8_embedding(w, s, ids, 0, 256)
    expected = fp8_e8m0_rows(w, s)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(
        actual.cpu(), reference(w, s, ids, 0, 256), atol=0, rtol=0, equal_nan=True
    )
