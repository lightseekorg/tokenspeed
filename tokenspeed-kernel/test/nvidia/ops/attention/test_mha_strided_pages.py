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

"""Consecutive K/V fields retain page strides on eager and graph paths."""

import pytest
import torch
from utils import assert_no_triton_compile, is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache
from tokenspeed_kernel.ops.attention.prologue import HeadKVCache, gqa_prologue
from tokenspeed_kernel.ops.attention.prologue.triton import _gqa_prologue_kernel
from tokenspeed_kernel.ops.kvcache.triton import _store_kv_cache_kernel, store_kv_cache


def _fields(heads, dim):
    rows = 128
    payload = rows * heads * dim
    backing = torch.full(
        (3, 2 * payload + 256), -3, dtype=torch.bfloat16, device="cuda"
    )
    shape = (3, rows, heads, dim)
    stride = (backing.stride(0), heads * dim, dim, 1)
    key = backing.as_strided(shape, stride)
    value = backing.as_strided(shape, stride, payload)
    return backing, key, value


def _run(call, graph, before_replay=None):
    if not graph:
        return call()
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        output = call()
    if before_replay is not None:
        before_replay()
    capture.replay()
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("writer", ["scatter", "prologue"])
def test_paged_kv_writes_preserve_sibling_field_and_padding(graph, writer):
    backing, key, value = _fields(2, 64)
    slots = torch.tensor([128, 255, 256, 383], dtype=torch.int64, device="cuda")
    k = torch.randn(4, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    if writer == "scatter":
        call = lambda: store_kv_cache(k, v, key, value, slots, enable_pdl=False)
    else:
        q = torch.randn(4, 4, 64, dtype=torch.bfloat16, device="cuda")
        target = HeadKVCache(key, value, None, slots)
        call = lambda: gqa_prologue(
            q,
            k.flatten(1),
            v.flatten(1),
            norm=None,
            rotary=None,
            cache=target,
            query_dtype=q.dtype,
            return_kv=False,
            solution="triton",
            override=None,
        )
    old_slots = slots.clone()

    def change_write_rows():
        slots.copy_(torch.tensor([129, 254, 257, 382], device="cuda"))
        k.add_(0.5)
        v.mul_(2)
        backing.fill_(-3)

    _run(call, graph, change_write_rows)
    if graph:
        assert torch.all(key[old_slots // 128, old_slots % 128] == -3)
        assert torch.all(value[old_slots // 128, old_slots % 128] == -3)
    torch.testing.assert_close(key[slots // 128, slots % 128], k, atol=0, rtol=0)
    torch.testing.assert_close(value[slots // 128, slots % 128], v, atol=0, rtol=0)
    assert torch.all(backing[:, -256:] == -3)
    assert torch.all(key[0] == -3)
    assert torch.all(value[0] == -3)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "rows, heads, length, return_lse, window_left, sink_tokens",
    [
        (1, 2, 384, False, 63, 16),
        (1, 4, 384, False, 63, 16),
        (4, 4, 384, False, 63, 16),
        (4, 4, 48, True, 63, 16),
        (1, 4, 384, True, 63, 16),
        (3, 2, 12, False, 7, 4),
        (3, 2, 12, True, 7, 4),
    ],
)
def test_decode_reads_strided_pages_and_sink_window_union(
    graph, rows, heads, length, return_lse, window_left, sink_tokens
):
    _, key, value = _fields(2, 64)
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    q = torch.randn(rows, heads, 64, dtype=torch.bfloat16, device="cuda")
    table = torch.tensor([[1, 0, 2]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    call = lambda: mha_decode_with_kvcache(
        q=q,
        k_cache=key,
        v_cache=value,
        page_table=table,
        cache_seqlens=lengths,
        max_seqlen_k=length,
        max_seqlen_q=rows,
        window_left=window_left,
        window_sink=sink_tokens,
        return_lse=return_lse,
        solution="triton",
    )

    def change_decode_rows():
        table.copy_(torch.tensor([[2, 0, 1]], dtype=torch.int32, device="cuda"))
        lengths.sub_(4)
        q.mul_(0.5)
        key.add_(0.25)
        value.sub_(0.5)

    result = _run(call, graph, change_decode_rows)
    length = int(lengths[0].item())
    output, lse = result if return_lse else (result, None)
    keys = (
        key[table[0].long()].flatten(0, 1).float().repeat_interleave(heads // 2, dim=1)
    )
    values = (
        value[table[0].long()]
        .flatten(0, 1)
        .float()
        .repeat_interleave(heads // 2, dim=1)
    )
    expected, expected_lse = [], []
    for row in range(rows):
        end = length - rows + row + 1
        positions = torch.arange(end, device="cuda")
        indices = positions[
            (positions < sink_tokens) | (positions >= end - window_left - 1)
        ]
        scores = torch.einsum("hd,thd->ht", q[row].float(), keys[indices]) / 8
        expected_lse.append(scores.logsumexp(-1))
        expected.append(torch.einsum("ht,thd->hd", scores.softmax(-1), values[indices]))
    torch.testing.assert_close(
        output.float(), torch.stack(expected), atol=2e-2, rtol=2e-2
    )
    if lse is not None:
        torch.testing.assert_close(lse, torch.stack(expected_lse), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("writer", ["scatter", "prologue"])
def test_paged_kv_offsets_past_int32(writer):
    # Both factors fit int32; the last page's element offset does not.
    # Allocate address space but only touch the last page, not the 4 GiB gap.
    rows, heads, dim = 128, 2, 64
    payload = rows * heads * dim
    page_stride = (1 << 30) + 256
    backing = torch.empty(
        2 * page_stride + 2 * payload, dtype=torch.bfloat16, device="cuda"
    )
    shape = (3, rows, heads, dim)
    strides = (page_stride, heads * dim, dim, 1)
    key = backing.as_strided(shape, strides)
    value = backing.as_strided(shape, strides, payload)
    slots = torch.arange(256, 288, device="cuda", dtype=torch.int64)
    k = torch.randn(32, heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn_like(k)
    if writer == "scatter":
        store_kv_cache(k, v, key, value, slots, enable_pdl=False)
    else:
        gqa_prologue(
            q,
            k.flatten(1),
            v.flatten(1),
            norm=None,
            rotary=None,
            cache=HeadKVCache(key, value, None, slots),
            query_dtype=q.dtype,
            return_kv=False,
            solution="triton",
            override=None,
        )
    torch.testing.assert_close(key[2, :32], k, atol=0, rtol=0)
    torch.testing.assert_close(value[2, :32], v, atol=0, rtol=0)
    out = mha_decode_with_kvcache(
        q=q[-1:],
        k_cache=key,
        v_cache=value,
        page_table=torch.tensor([[2]], dtype=torch.int32, device="cuda"),
        cache_seqlens=torch.tensor([32], dtype=torch.int32, device="cuda"),
        max_seqlen_k=32,
        max_seqlen_q=1,
        window_left=-1,
        window_sink=0,
        return_lse=False,
        solution="triton",
    )
    scores = torch.einsum("hd,thd->ht", q[-1].float(), k.float()) / 8
    expected = torch.einsum("ht,thd->hd", scores.softmax(-1), v.float())
    torch.testing.assert_close(out[0].float(), expected, atol=1e-2, rtol=1e-2)


def test_paged_writes_reuse_compilation_across_token_counts():
    _, key, value = _fields(2, 64)
    slots = torch.arange(128, 160, dtype=torch.int64, device="cuda")
    q = torch.randn(32, 4, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(32, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)

    def write(count):
        store_kv_cache(
            k[:count], v[:count], key, value, slots[:count], enable_pdl=False
        )
        gqa_prologue(
            q[:count],
            k[:count].flatten(1),
            v[:count].flatten(1),
            norm=None,
            rotary=None,
            cache=HeadKVCache(key, value, None, slots[:count]),
            query_dtype=q.dtype,
            return_kv=False,
            solution="triton",
            override=None,
        )

    for count in (2, 16):
        write(count)
    with assert_no_triton_compile(_store_kv_cache_kernel, _gqa_prologue_kernel):
        for count in (3, 17, 31):
            write(count)
    torch.testing.assert_close(key[1, :31], k[:31], atol=0, rtol=0)
    torch.testing.assert_close(value[1, :31], v[:31], atol=0, rtol=0)
