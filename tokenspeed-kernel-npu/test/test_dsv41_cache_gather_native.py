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


"""Byte-exact copy, invalid metadata, large-address and replay contracts."""

import os

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops import dsv41_cache_gather_native as native
from tokenspeed_kernel_npu.ops.dsv41_bf16_cache import cache_gather_bf16


def setup_module():
    torch.npu.set_device(int(os.environ.get("V41_TEST_DEVICE", "0")))
    torch.set_num_threads(4)


def equal_bits(actual, expected):
    assert torch.equal(
        actual.cpu().contiguous().view(torch.uint8),
        expected.cpu().contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize(
    "n,padded,index_dtype,out_dtype",
    [
        (0, False, torch.int32, torch.bfloat16),
        (1, False, torch.int64, torch.float32),
        (63, True, torch.int32, torch.bfloat16),
        (65, True, torch.int64, torch.float32),
        (4097, False, torch.int32, torch.bfloat16),
        (24576, True, torch.int64, torch.float32),
        (98304, False, torch.int64, torch.bfloat16),
    ],
)
def test_copy_and_dynamic_replay(n, padded, index_dtype, out_dtype):
    torch.manual_seed(92)
    storage = torch.randn(
        20, 64, 544 if padded else 512, dtype=torch.bfloat16, device="npu"
    )
    cache = storage[:, :, :512]
    indices = torch.randint(-10, 1290, (n,), dtype=index_dtype, device="npu")
    out = torch.empty((n, 512), dtype=out_dtype, device="npu")

    def check():
        ids = indices.cpu().long()
        expected = cache.cpu().reshape(-1, 512)[ids.clamp(0, 1279)].to(out_dtype)
        expected[(ids < 0) | (ids >= 1280)] = 0
        equal_bits(out, expected)

    native.gather_rows(cache, indices, out)
    torch.npu.synchronize()
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        native.gather_rows(cache, indices, out)
    for _ in range(3):
        storage.copy_(torch.randn_like(storage))
        indices.copy_(torch.randint(-10, 1290, (n,), dtype=index_dtype, device="npu"))
        graph.replay()
        torch.npu.synchronize()
        check()
    if n >= 4096:
        equal_bits(cache_gather_bf16(cache, indices), out.bfloat16())


def test_all_bf16_bits():
    raw = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(2, 64, 512)
    )
    cache = raw.npu()
    ids = torch.arange(4097, dtype=torch.int64, device="npu") % 128
    actual = cache_gather_bf16(cache, ids)
    equal_bits(actual, raw.reshape(128, 512)[ids.cpu()])


def test_addresses_above_two_gib():
    cache = torch.empty((40000, 64, 512), dtype=torch.bfloat16, device="npu")
    chosen = torch.tensor([0, 2097151, 2097152, 2097153, 2559999], dtype=torch.int64)
    rows = torch.arange(5 * 512, dtype=torch.float32).reshape(5, 512).bfloat16()
    cache.view(-1, 512).index_copy_(0, chosen.npu(), rows.npu())
    ids = chosen.repeat(1025).npu()
    out = cache_gather_bf16(cache, ids)
    equal_bits(out, rows.repeat(1025, 1))


def test_strided_columns_fallback():
    cache = torch.randn((8, 64, 1024), dtype=torch.bfloat16, device="npu")[:, :, ::2]
    ids = torch.arange(4097, dtype=torch.int32, device="npu") % 512
    equal_bits(
        cache_gather_bf16(cache, ids), cache.cpu().reshape(-1, 512)[ids.cpu().long()]
    )
