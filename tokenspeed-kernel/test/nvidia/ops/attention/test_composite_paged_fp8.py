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

"""Composite FP8 writes preserve paged field strides and graph-updated slots."""

from __future__ import annotations

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from attention_prologue_helpers import bytes_equal, qkv, split
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    gqa_prologue,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_composite_paged_fp8_preserves_other_fields_and_replays_slots(
    dtype, cuda_graph
):
    """Padded K/V fields and flat rows receive the same FP8 bytes in place.

    The extra input rows are graph padding. Slots cross every page boundary,
    and replay changes both the packed inputs and destinations in place.
    """
    pages, page_size, page_padding = 4, 128, 5
    tokens, hq, hkv, dim = 11, 4, 2, 128
    fp8, poison = torch.float8_e4m3fn, 0x5A
    packed = qkv(tokens, hq, hkv, dim, seed=305, dtype=dtype)
    q, k, v = split(packed, hq, hkv, dim)
    backing = torch.full(
        (pages, 3, page_size + page_padding, hkv, dim),
        poison,
        dtype=torch.uint8,
        device="cuda",
    )
    # A sibling field and page-tail padding separate the K/V payloads.
    k_cache = backing[:, 0, :page_size].view(fp8)
    v_cache = backing[:, 2, :page_size].view(fp8)
    flat_k, flat_v = (
        torch.full(
            (pages * page_size, hkv, dim), poison, dtype=torch.uint8, device="cuda"
        ).view(fp8)
        for _ in range(2)
    )
    initial_slots = [0, 127, 128, 129, 255, 256, 258, 383, 511]
    updated_slots = [1, 126, 130, 254, 257, 259, 382, 384, 510]
    loc = torch.tensor(initial_slots, device="cuda", dtype=torch.int64)
    paged_target = HeadKVCache(k_cache, v_cache, None, loc)
    flat_target = HeadKVCache(flat_k, flat_v, None, loc)

    def run():
        return tuple(
            gqa_prologue(
                q,
                k,
                v,
                norm=None,
                rotary=None,
                cache=target,
                query_dtype=fp8,
                return_kv=True,
                solution="composite",
                override=None,
            )
            for target in (paged_target, flat_target)
        )

    outputs = run()
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run()

    for destinations in (initial_slots, updated_slots):
        packed.mul_(0.5)
        loc.copy_(torch.tensor(destinations, device="cuda", dtype=loc.dtype))
        backing.fill_(poison)
        flat_k.view(torch.uint8).fill_(poison)
        flat_v.view(torch.uint8).fill_(poison)
        if cuda_graph:
            graph.replay()
        else:
            outputs = run()
        torch.cuda.synchronize()

        expected = torch.full_like(backing, poison)
        expected[loc // page_size, 0, loc % page_size] = (
            k[: loc.numel()].reshape(-1, hkv, dim).to(fp8).view(torch.uint8)
        )
        expected[loc // page_size, 2, loc % page_size] = (
            v[: loc.numel()].reshape(-1, hkv, dim).to(fp8).view(torch.uint8)
        )
        # Check all payloads, untouched rows, sibling fields and page padding.
        assert torch.equal(backing, expected)
        assert bytes_equal(k_cache.reshape(-1, hkv, dim), flat_k)
        assert bytes_equal(v_cache.reshape(-1, hkv, dim), flat_v)
        for out in outputs:
            assert out.q.dtype is fp8
            assert bytes_equal(out.q, q.to(fp8))
            assert bytes_equal(out.k, k)
            assert bytes_equal(out.v, v)
