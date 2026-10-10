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

"""GFX950/GFX1250 fused two-reader DeepSeek V4.1 selected-attention checks."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile

pytest.importorskip("tokenspeed_triton")
pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")


def _selected_name() -> str:
    kernel = select_kernel(
        "attention",
        "dsv41_selected_attention",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"flashmla_eligible": False},
    )
    return kernel.name


def test_fused_gluon_is_selected_on_supported_amd():
    platform = current_platform()
    if platform.is_cdna4:
        assert _selected_name() == "gluon_dsv41_selected_attention_gfx950"
    elif platform.is_cdna5:
        assert _selected_name() == "gluon_dsv41_selected_attention_gfx1250"
    else:
        pytest.skip("AMD fused selected attention")


def _make_cache(x, fmt):
    width = {"swa": 528, "global": 288}[fmt]
    cache = torch.zeros(
        ((x.shape[0] + 63) // 64, 64, width), dtype=torch.uint8, device=x.device
    )
    dsv41.cache_scatter(x, cache, torch.arange(x.shape[0], device=x.device), fmt)
    return cache


@pytest.mark.parametrize("heads", [1, 7, 16])
@pytest.mark.parametrize("with_global", [False, True])
def test_fused_selected_attention_matches_gather_softmax(heads, with_global):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("AMD fused selected attention")
    torch.manual_seed(43)
    device = torch.device("cuda:0")
    q = torch.randn((5, heads, 512), dtype=torch.bfloat16, device=device) * 2
    swa = _make_cache(
        torch.randn((130, 512), dtype=torch.bfloat16, device=device), "swa"
    )
    glob = _make_cache(
        torch.randn((65, 512), dtype=torch.bfloat16, device=device), "global"
    )
    swa_slots = torch.tensor(
        [
            [63, 64, 129, -1],
            [-1, -1, -1, -1],
            [192, -2, 63, 0],
            [0, 0, 1, 2],
            [1, 2, 3, 4],
        ],
        device=device,
    )
    global_slots = torch.tensor(
        [[0, 64, -1], [-1, -1, -1], [128, 0, 1], [0, 1, 2], [2, 3, 4]],
        device=device,
    )
    swa_lens = torch.tensor([3, 4, 3, 4, 0], dtype=torch.int32, device=device)
    global_lens = torch.tensor([2, 3, 2, 3, 0], dtype=torch.int32, device=device)
    sink = torch.linspace(-10, 10, heads, dtype=torch.float32, device=device)
    parts, masks = [], []
    for cache, slots, lens, fmt in [(swa, swa_slots, swa_lens, "swa")] + (
        [(glob, global_slots, global_lens, "global")] if with_global else []
    ):
        parts.append(dsv41.cache_gather(cache, slots, fmt, None).float())
        masks.append(
            (torch.arange(slots.shape[1], device=device) < lens[:, None])
            & (slots >= 0)
            & (slots < cache.shape[0] * 64)
        )
    kv, valid = torch.cat(parts, dim=1), torch.cat(masks, dim=1)
    logits = torch.bmm(q.float(), kv.transpose(1, 2)) * 512**-0.5
    logits.masked_fill_(~valid[:, None], -torch.inf)
    logits = torch.cat((logits, sink[None, :, None].expand(5, -1, 1)), dim=-1)
    expected = torch.bmm(logits.softmax(dim=-1)[..., :-1], kv).bfloat16()
    out = dsv41.selected_attention(
        q,
        swa,
        swa_slots,
        swa_lens,
        glob if with_global else None,
        global_slots if with_global else None,
        global_lens if with_global else None,
        sink,
        512**-0.5,
        None,
        5,
        None,
        None,
        None,
    )
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.004)
    assert torch.count_nonzero(out[[1, 4]]).item() == 0


def _arena_cache(rows, fmt, page_pad):
    """Page-planar cache view with an arena-style padded page stride."""
    width = {"swa": 528, "global": 288}[fmt]
    pages = (rows.shape[0] + 63) // 64
    stride = 64 * width + page_pad
    raw = torch.zeros(pages * stride, dtype=torch.uint8, device=rows.device)
    cache = raw.as_strided((pages, 64, width), (stride, width, 1))
    dsv41.cache_scatter(
        rows, cache, torch.arange(rows.shape[0], device=rows.device), fmt
    )
    return cache


def _gather_softmax(q, sink, segments):
    parts, masks = [], []
    for cache, slots, lens, fmt in segments:
        parts.append(dsv41.cache_gather(cache, slots, fmt, None).float())
        masks.append(
            (torch.arange(slots.shape[1], device=q.device) < lens[:, None])
            & (slots >= 0)
            & (slots < cache.shape[0] * 64)
        )
    kv, valid = torch.cat(parts, dim=1), torch.cat(masks, dim=1)
    logits = torch.bmm(q.float(), kv.transpose(1, 2)) * 512**-0.5
    logits.masked_fill_(~valid[:, None], -torch.inf)
    sinks = sink[None, :, None].expand(q.shape[0], -1, 1)
    probs = torch.cat((logits, sinks), dim=-1).softmax(dim=-1)[..., :-1]
    return torch.bmm(probs, kv)


def _decode_inputs(tokens, device, seed):
    """Decode-shaped selections: 128 SWA and 512 global slots per query row."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    swa = _arena_cache(
        torch.randn((256, 512), generator=generator).bfloat16().to(device),
        "swa",
        512,
    )
    glob = _arena_cache(
        torch.randn((1024, 512), generator=generator).bfloat16().to(device),
        "global",
        256,
    )
    swa_slots = torch.randint(-2, 256, (tokens, 128), generator=generator)
    global_slots = torch.randint(-1, 1100, (tokens, 512), generator=generator)
    swa_lens = torch.randint(0, 129, (tokens,), generator=generator)
    global_lens = torch.randint(0, 513, (tokens,), generator=generator)
    swa_lens[:2] = 128
    global_lens[:2] = 512
    return [
        (swa, swa_slots.to(device), swa_lens.int().to(device), "swa"),
        (glob, global_slots.to(device), global_lens.int().to(device), "global"),
    ]


def _padded_query(tokens, real_heads, heads, device, seed):
    """rope_pad_query-style queries: zero padded heads with -inf sinks."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.zeros((tokens, heads, 512), dtype=torch.bfloat16)
    q[:, :real_heads] = torch.randn((tokens, real_heads, 512), generator=generator)
    sink = torch.full((heads,), -torch.inf)
    sink[:real_heads] = torch.linspace(-3, 3, real_heads)
    return q.to(device), sink.to(device)


@pytest.mark.parametrize("tokens", [6, 192])
@pytest.mark.parametrize("real_heads", [16, 64])
def test_fused_selected_attention_decode_shapes(tokens, real_heads):
    """Split (6 rows) and single-workgroup (192 rows) paths at decode widths.

    Padded zero-query heads take the row-sum shortcut and must still match the
    softmax over their (all-zero) scores.
    """
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("AMD fused selected attention")
    device = torch.device("cuda:0")
    segments = _decode_inputs(tokens, device, 7)
    q, sink = _padded_query(tokens, real_heads, 64, device, 11)
    expected = _gather_softmax(q, sink, segments).bfloat16()
    (swa, swa_slots, swa_lens, _), (glob, global_slots, global_lens, _) = segments
    out = dsv41.selected_attention(
        q,
        swa,
        swa_slots,
        swa_lens,
        glob,
        global_slots,
        global_lens,
        sink,
        512**-0.5,
        None,
        256,
        None,
        None,
        None,
    )
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.004)


def test_fused_selected_attention_gfx950_no_recompile_across_rows():
    """Row count changes the grid and split count, never a compile key."""
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 kernel")
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsv41 import selected

    device = torch.device("cuda:0")
    segments = _decode_inputs(192, device, 3)
    (swa, swa_slots, swa_lens, _), (glob, global_slots, global_lens, _) = segments
    q, sink = _padded_query(192, 16, 64, device, 5)

    def run(tokens):
        selected.launch_gluon_dsv41_selected_attention_gfx950(
            q[:tokens],
            swa,
            swa_slots[:tokens],
            swa_lens[:tokens],
            glob,
            global_slots[:tokens],
            global_lens[:tokens],
            sink,
            512**-0.5,
            None,
            256,
            None,
            None,
            None,
        )

    for tokens in (1, 192):
        run(tokens)
    with assert_no_triton_compile(
        selected.gluon_dsv41_selected_attention_gfx950,
        selected.gluon_dsv41_selected_attention_reduce_gfx950,
    ):
        for tokens in (2, 5, 33, 96, 150, 191):
            run(tokens)
    torch.cuda.synchronize()
