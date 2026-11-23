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

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv4.triton import triton_dsv4_prefill


@pytest.fixture
def sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120/SM121")
    pytest.importorskip("flashinfer.mla")
    return torch.device("cuda:0")


def packed(rows, fmt, width):
    cache = torch.zeros(
        ((rows.shape[0] + 63) // 64, 64, width), dtype=torch.uint8, device=rows.device
    )
    dsv41.cache_scatter(
        rows, cache, torch.arange(rows.shape[0], device=rows.device), fmt
    )
    return cache


@pytest.mark.parametrize("tokens,heads", [(1, 1), (17, 8), (128, 32), (1024, 32)])
@pytest.mark.parametrize("with_global", [False, True])
@pytest.mark.parametrize("with_sink", [False, True])
def test_paged_prefill_joint_softmax_and_refreshed_selection(
    sm120, tokens, heads, with_global, with_sink
):
    torch.manual_seed(241)
    q = torch.randn(tokens, heads, 512, device=sm120, dtype=torch.bfloat16)
    raw_swa = torch.randn(263, 512, device=sm120, dtype=torch.bfloat16)
    swa = dsv41.cache_unpack(dsv41.cache_pack(raw_swa, "swa", None), "swa", None)
    prefix, current = swa[:127], swa[127:]
    g = packed(torch.randn(1024, 512, device=sm120, dtype=q.dtype), "global", 288)
    original_g = g.clone()
    gd = dsv41.cache_gather(g, torch.arange(1024, device=sm120), "global", None)
    kv = torch.cat((swa, gd)) if with_global else swa
    si = torch.randint(0, 263, (tokens, 128), device=sm120, dtype=torch.int32)
    gi = torch.randint(0, 1024, (tokens, 512), device=sm120, dtype=torch.int32)
    gl = torch.full((tokens,), 512, device=sm120, dtype=torch.int32)
    sink = torch.randn(heads, device=sm120) if with_sink else None
    out = torch.empty_like(q)
    lse = torch.empty((tokens, heads), device=sm120)
    reference = torch.empty_like(q)
    ref_lse = torch.empty_like(lse)
    for phase in range(2):
        if phase:
            q.normal_()
            si.random_(0, 263)
            gi.random_(0, 1024)
            gl.random_(0, 513)
        si[0].fill_(-1)
        gi[0].fill_(-1)
        gl[0] = 0
        if tokens > 1:
            si[1, 1::3] = -1
            gi[1, 2::5] = -1
        indices = si
        if with_global:
            mask = (torch.arange(512, device=sm120)[None, :] < gl[:, None]) & (gi >= 0)
            indices = torch.cat((si, (gi + 263).masked_fill(~mask, -1)), dim=1)
        lengths = torch.full(
            (tokens,), indices.shape[1], device=sm120, dtype=torch.int32
        )
        triton_dsv4_prefill(
            q, kv, indices, lengths, sink, 512**-0.5, reference, ref_lse
        )
        actual = dsv41.try_paged_prefill(
            q,
            prefix,
            current,
            si,
            g if with_global else None,
            gi if with_global else None,
            gl if with_global else None,
            sink,
            512**-0.5,
            out,
            lse,
        )
        assert actual is out
        torch.testing.assert_close(out, reference, atol=0.002, rtol=0.02)
        torch.testing.assert_close(lse, ref_lse, atol=0.003, rtol=0.001)
        torch.testing.assert_close(g, original_g, atol=0, rtol=0)
        torch.testing.assert_close(swa, torch.cat((prefix, current)), atol=0, rtol=0)
        assert torch.count_nonzero(out[0]) == 0


def test_repacking_protected_swa_preserves_decoded_values(sm120):
    torch.manual_seed(512)
    rows = torch.randn(263, 512, device=sm120, dtype=torch.bfloat16)
    rows[0].zero_()
    rows[1].mul_(2**-15)
    rows[2].mul_(2**12)
    encoded = packed(rows, "swa", 528)
    decoded = dsv41.cache_gather(encoded, torch.arange(263, device=sm120), "swa", None)
    repacked = packed(decoded, "swa", 528)
    reread = dsv41.cache_gather(repacked, torch.arange(263, device=sm120), "swa", None)
    torch.testing.assert_close(decoded, reread, atol=0, rtol=0)
