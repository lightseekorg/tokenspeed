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

from __future__ import annotations

import pytest
import torch
from utils import is_cdna5

if not is_cdna5():
    pytest.skip("GFX1250 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.dsv4 import dsv4_decode


def make_cache(pages, page_size, padding):
    nope = torch.randn(pages, page_size, 448, device="cuda").to(torch.float8_e4m3fn)
    rope = torch.randn(pages, page_size, 64, device="cuda", dtype=torch.bfloat16)
    exponents = torch.randint(
        125, 130, (pages, page_size, 8), device="cuda", dtype=torch.uint8
    )
    storage = torch.zeros(
        (pages, page_size * 584 + padding), device="cuda", dtype=torch.uint8
    )
    cache = storage[:, : page_size * 584]
    payload = torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
    cache[:, : page_size * 576] = payload.reshape(pages, -1)
    cache[:, page_size * 576 :] = exponents.reshape(pages, -1)
    scales = torch.exp2(exponents[..., :7].float() - 127).repeat_interleave(64, dim=-1)
    reference = torch.cat(((nope.float() * scales).to(torch.bfloat16), rope), dim=-1)
    return cache, reference.reshape(-1, 512)


def make_inputs(tokens, heads, width, extra_width, page_size, padding):
    torch.manual_seed(74)
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    cache, kv = make_cache(7, page_size, padding)
    extra_cache, extra_kv = make_cache(5, page_size, padding)
    slots = torch.randint(
        0, kv.shape[0], (tokens, width), device="cuda", dtype=torch.int32
    )
    extra_slots = torch.randint(
        0, extra_kv.shape[0], (tokens, extra_width), device="cuda", dtype=torch.int32
    )
    lens = torch.full((tokens,), width, device="cuda", dtype=torch.int32)
    extra_lens = torch.full((tokens,), extra_width, device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    args = dict(
        q=q,
        swa_kv_cache=cache,
        swa_slots=slots,
        swa_lens=lens,
        swa_page_size=page_size,
        attn_sink=sink,
        softmax_scale=512**-0.5,
        extra_kv_cache=extra_cache if extra_width else None,
        extra_slots=extra_slots if extra_width else None,
        extra_lens=extra_lens if extra_width else None,
        extra_page_size=page_size if extra_width else None,
        out=torch.empty_like(q),
    )
    return args, kv, extra_kv


def reference(args, kv, extra_kv):
    result = torch.zeros_like(args["q"])
    for token in range(result.shape[0]):
        selected = []
        for prefix, values in (("swa", kv), ("extra", extra_kv)):
            slots = args[f"{prefix}_slots"]
            if slots is None:
                continue
            length = max(0, min(int(args[f"{prefix}_lens"][token]), slots.shape[1]))
            indices = slots[token, :length].long()
            indices = indices[(indices >= 0) & (indices < values.shape[0])]
            selected.append(values[indices].float())
        keys = torch.cat(selected)
        scores = args["q"][token].float() @ keys.T * args["softmax_scale"]
        probabilities = torch.cat(
            (scores, args["attn_sink"].float()[:, None]), dim=-1
        ).softmax(-1)[:, :-1]
        result[token] = probabilities @ keys
    return result


def run(args, solution):
    return dsv4_decode(**args, override=None, solution=solution, return_lse=False)


@pytest.mark.parametrize(
    "tokens,heads,width,extra_width",
    [
        (1, 16, 128, 1024),
        (3, 32, 128, 1024),
        (7, 64, 128, 512),
        (17, 128, 128, 0),
        (64, 64, 128, 512),
        (256, 64, 128, 512),
        (128, 96, 128, 1024),
        (2, 8, 65, 33),
        (2, 17, 31, 0),
        (1, 16, 0, 0),
    ],
)
@pytest.mark.parametrize("page_size,padding", [(64, 0), (16, 64), (64, 2)])
def test_decode(tokens, heads, width, extra_width, page_size, padding):
    args, kv, extra_kv = make_inputs(
        tokens, heads, width, extra_width, page_size, padding
    )
    if width:
        args["swa_slots"][:, ::7] = -1
        args["swa_slots"][:, 1::13] = kv.shape[0]
        if tokens > 1:
            args["swa_lens"][-1] = 0
    if extra_width:
        args["extra_slots"][:, ::11] = -1
        args["extra_slots"][:, 1::17] = extra_kv.shape[0] + 1
        if tokens > 1:
            args["extra_lens"][-1] = -1
    if tokens > 2:
        args["swa_slots"][1].fill_(-1)
        if extra_width:
            args["extra_slots"][1].fill_(-1)
    if padding:
        args["attn_sink"] = args["attn_sink"].bfloat16()
    expected = reference(args, kv, extra_kv)
    output = run(args, "gluon")
    assert output.data_ptr() == args["out"].data_ptr()
    torch.testing.assert_close(output, expected, atol=0.015, rtol=0.015)
    # Automatic dispatch must also select a working GFX1250 implementation.
    torch.testing.assert_close(run(args, None), expected, atol=0.015, rtol=0.015)


@pytest.mark.parametrize("tokens,heads", [(3, 32), (256, 64)])
def test_graph_replay_refreshes_lengths_and_slots(tokens, heads):
    args, kv, extra_kv = make_inputs(tokens, heads, 128, 1024, 64, 64)
    run(args, "gluon")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(args, "gluon")
    for length in (0, 1, 65, 128):
        args["swa_lens"].fill_(length)
        args["extra_lens"].fill_(min(length * 3, 1024))
        args["swa_slots"].copy_(args["swa_slots"].roll(1, dims=1))
        args["extra_slots"][:, ::3] = -1
        args["q"].mul_(0.9)
        args["out"].fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(
            args["out"], reference(args, kv, extra_kv), atol=0.015, rtol=0.015
        )


def test_validation():
    args, _, _ = make_inputs(1, 16, 128, 0, 64, 0)
    args["out"] = args["q"]
    with pytest.raises(ValueError, match="alias"):
        run(args, "gluon")
    args["out"] = None
    args["softmax_scale"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run(args, "gluon")


def test_independent_page_sizes_and_allocated_output():
    args, kv, _ = make_inputs(2, 32, 128, 33, 64, 0)
    args["extra_kv_cache"], extra_kv = make_cache(20, 16, 64)
    args["extra_page_size"] = 16
    args["extra_slots"].remainder_(extra_kv.shape[0])
    args["swa_lens"].fill_(129)
    args["extra_lens"].fill_(34)
    args["out"] = None
    torch.testing.assert_close(
        run(args, "gluon"), reference(args, kv, extra_kv), atol=0.015, rtol=0.015
    )


def test_single_partition_fuses_sink_and_output():
    args, kv, extra_kv = make_inputs(256, 64, 65, 33, 64, 0)
    args["swa_lens"][-1] = 0
    args["extra_lens"][-1] = 0
    output = run(args, "gluon")
    torch.testing.assert_close(
        output, reference(args, kv, extra_kv), atol=0.015, rtol=0.015
    )


@pytest.mark.parametrize("page_stride", [2**31, 2**31 + 2])
def test_cache_offsets_beyond_signed_int32(page_stride):
    args, _, extra_kv = make_inputs(1, 32, 64, 0, 64, 0)
    compact, kv = make_cache(2, 64, 0)
    # Touch only two pages, with the second page starting beyond 2 GiB.
    cache = torch.empty_strided(
        (2, 64 * 584), (page_stride, 1), device="cuda", dtype=torch.uint8
    )
    cache.copy_(compact)
    args["swa_kv_cache"] = cache
    args["swa_slots"].copy_(
        torch.arange(64, 128, device="cuda", dtype=torch.int32)[None, :]
    )
    torch.testing.assert_close(
        run(args, "gluon"), reference(args, kv, extra_kv), atol=0.015, rtol=0.015
    )
