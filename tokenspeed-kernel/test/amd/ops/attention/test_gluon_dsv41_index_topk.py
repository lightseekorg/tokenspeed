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

"""GFX950/GFX1250 gluon DeepSeek V4.1 CSA2 indexer checks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv41 import (
    torch_dsv41_cache_quantize,
)
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import gluon as gluon_backend
from tokenspeed_kernel.ops.attention.dsv41 import triton as triton_backend
from tokenspeed_kernel.ops.attention.dsv41._gluon import indexer as gluon_indexer
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile

pytest.importorskip("tokenspeed_triton")
pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

_OPS = Path(__file__).resolve().parents[3] / "ops"
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

from test_attention_dsv41_index_scan import (  # noqa: E402
    run_index_scan_graph_oracle,
    run_index_topk_full_and_reindex,
)


def _index_name() -> str:
    kernel = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={
            "native_indexer": False,
            "index_heads": 32,
            "index_k_format": "mxfp4",
            "index_shards": 1,
        },
        solution="gluon",
    )
    return kernel.name


def test_gluon_index_topk_is_selected_on_supported_amd():
    platform = current_platform()
    if platform.is_cdna4:
        assert _index_name() == "gluon_dsv41_index_topk_gfx950"
    elif platform.is_cdna5:
        assert _index_name() == "gluon_dsv41_index_topk_gfx1250"
    else:
        pytest.skip("AMD gluon DSV4.1 indexer")


@pytest.mark.parametrize(
    ("width", "expected"),
    [(32768, 256), (131072, 64), (1048576, 8)],
)
def test_gluon_index_topk_query_tile_bounds_logits(width, expected):
    assert gluon_indexer._score_query_tile(256, width) == expected
    assert expected * width * torch.float32.itemsize <= 32 << 20


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return torch.device("cuda:0")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_index_scan_graph_oracle_gluon(device, dtype, require, monkeypatch):
    # Force multiple score-query tiles so graph replay covers the bounded path.
    monkeypatch.setattr(gluon_indexer, "_LOGITS_BUDGET_BYTES", 256)
    run_index_scan_graph_oracle(
        device, 1, dtype, None, require, "gluon", rtol=0.1, atol=0.1
    )


@pytest.mark.parametrize("heads", [1, 8, 32])
@pytest.mark.parametrize("candidate_topk", [0, 17])
@pytest.mark.parametrize("topk", [65, 512])
def test_index_topk_full_and_reindex_gluon(
    device, heads, candidate_topk, topk, require
):
    run_index_topk_full_and_reindex(
        device, "gluon", heads, candidate_topk, topk, require, match_triton=False
    )


def test_index_topk_gluon_full_above_32k(device, require, monkeypatch):
    from tokenspeed_kernel.ops.attention.dsv41 import triton as implementation

    require("attention", "dsv41_index_topk", "gluon", torch.bfloat16, "x")
    pages = 513
    rows = pages * 64
    q = torch.ones((1, 1, 128), dtype=torch.bfloat16, device=device)
    weights = torch.ones((1, 1), dtype=torch.bfloat16, device=device)
    keys = torch.zeros((rows, 128), dtype=torch.bfloat16, device=device)
    keys[-8:] = 1
    cache = torch.zeros((pages, 64, 68), dtype=torch.uint8, device=device)
    dsv41.cache_scatter(keys, cache, torch.arange(rows, device=device), "index")
    table = torch.arange(pages, dtype=torch.int32, device=device)[None]
    visible = torch.tensor([rows], dtype=torch.int32, device=device)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("long Full unexpectedly used portable Triton")

    monkeypatch.setattr(implementation, "index_topk", unexpected_fallback)
    selected, lengths, blocks, block_lengths = dsv41.index_topk(
        q,
        weights,
        cache,
        table,
        visible,
        None,
        8,
        0,
        8,
        256,
        256,
        None,
        None,
        solution="gluon",
    )

    torch.testing.assert_close(
        selected.cpu(),
        torch.arange(rows - 8, rows, dtype=torch.int32)[None],
        rtol=0,
        atol=0,
    )
    assert lengths.item() == 8
    assert blocks.shape == (1, 0)
    assert block_lengths.item() == 0


def test_index_topk_large_head_count_auto_fallback(device, require):
    run_index_topk_full_and_reindex(
        device, None, 64, 17, 65, require, match_triton=True
    )


@pytest.mark.parametrize(
    ("arch", "dtype", "width", "metric", "kernel_name"),
    [
        ("gfx950", torch.uint8, 64, "flops4", "gluon_dsv41_index_topk_gfx950"),
        ("gfx1250", torch.uint8, 64, "flops8", "gluon_dsv41_index_topk_gfx1250"),
    ],
)
def test_indexer_launch_metadata(arch, dtype, width, metric, kernel_name):
    module = importlib.import_module(
        f"tokenspeed_kernel_amd.ops.{arch}.attention.dsv41.indexer"
    )
    args = {
        "q": torch.empty((2, 32, width), dtype=dtype),
        "logits": torch.empty((2, 512), dtype=torch.float32),
        "NUM_HEADS": 32,
    }
    metadata = module._index_launch_metadata(
        (2, 2, 1), SimpleNamespace(name="index_score"), args
    )
    assert metadata["name"] == "index_score"
    assert metadata[metric] == 2 * 2 * 32 * 512 * 128
    assert metadata["bytes"] > args["logits"].numel() * 4
    assert getattr(module, kernel_name).launch_metadata is module._index_launch_metadata


@pytest.mark.parametrize(
    ("arch", "entry_name", "kernel_name", "tile_n"),
    [
        ("gfx950", "dsv41_index_logits_gfx950", "gluon_dsv41_index_topk_gfx950", 32),
        ("gfx1250", "dsv41_index_logits_gfx1250", "gluon_dsv41_index_topk_gfx1250", 64),
    ],
)
@pytest.mark.parametrize(
    ("requested_chunk", "score_chunk"),
    [(8, 8), (64, 64), (512, 256)],
)
def test_indexer_score_chunk_bound(
    monkeypatch,
    arch,
    entry_name,
    kernel_name,
    tile_n,
    requested_chunk,
    score_chunk,
):
    hardware_chunk = (score_chunk + tile_n - 1) // tile_n * tile_n
    module = importlib.import_module(
        f"tokenspeed_kernel_amd.ops.{arch}.attention.dsv41.indexer"
    )
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs["SCORE_CHUNK"], kwargs["CHUNK_N"]))

            return launch

    monkeypatch.setattr(module, kernel_name, FakeKernel())
    tokens, width = 2, 257
    weights = torch.empty((tokens, 32), dtype=torch.float32)
    cache = torch.empty((5, 64 * 68), dtype=torch.uint8)
    table = torch.empty((tokens, 5), dtype=torch.int32)
    visible = torch.empty((tokens,), dtype=torch.int32)
    logits = torch.empty((tokens, width), dtype=torch.float32)
    getattr(module, entry_name)(
        torch.empty((tokens, 32, 64), dtype=torch.uint8),
        torch.empty((tokens, 32), dtype=torch.int32),
        weights,
        cache,
        table,
        visible,
        None,
        logits,
        requested_chunk,
    )

    assert launches == [
        (
            (tokens, (width + score_chunk - 1) // score_chunk),
            score_chunk,
            hardware_chunk,
        )
    ]


# A 40-row score chunk masks part of each CTA's 64-row tile, and in Full mode
# CTAs span page boundaries.
@pytest.mark.parametrize("reindex", [False, True])
@pytest.mark.parametrize("score_chunk", [40, 256])
def test_gfx1250_index_logits_match_reference(device, reindex, score_chunk):
    if not current_platform().is_cdna5:
        pytest.skip("GFX1250 indexer logits")
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41.indexer import (
        dsv41_index_logits_gfx1250,
    )

    g = torch.Generator(device=device).manual_seed(251)
    keys = torch.randn(512, 128, device=device, dtype=torch.bfloat16, generator=g)
    keys *= 0.1
    _, key_reference = torch_dsv41_cache_quantize(keys, "index")
    # Padding between pages gives a page stride wider than the 64 * 68 payload.
    storage = torch.zeros(8, 64 * 68 + 16, device=device, dtype=torch.uint8)
    cache = storage[:, : 64 * 68].view(8, 64, 68)
    dsv41.cache_scatter(keys, cache, torch.arange(512, device=device), "index")
    q = torch.randn(2, 32, 128, device=device, dtype=torch.bfloat16, generator=g)
    q *= 0.1
    weights = torch.randn(2, 32, device=device, generator=g)
    table = torch.tensor(
        [[3, 1, -1, 2], [7, 5, 4, 6]], device=device, dtype=torch.int32
    )
    visible = torch.tensor([193, 221], device=device, dtype=torch.int32)
    candidates = (
        torch.tensor(
            [[3, 0, -1, 20, 31], [27, 4, 2, 1, -1]], device=device, dtype=torch.int32
        )
        if reindex
        else None
    )
    width = 40 if reindex else 256
    out = torch.full((2, width), -torch.inf, device=device)
    values, scales, packed_weights = gluon_indexer._pack_index_q(q, weights)
    dsv41_index_logits_gfx1250(
        values,
        scales,
        packed_weights,
        cache.flatten(1),
        table,
        visible,
        candidates,
        out,
        score_chunk,
    )

    _, q_reference = torch_dsv41_cache_quantize(q.reshape(-1, 128), "index")
    q_reference = q_reference.reshape(2, 32, 128)
    expected = torch.full((2, width), -torch.inf)
    for row in range(2):
        logical = torch.arange(width)
        if reindex:
            blocks = candidates[row].cpu()[logical // 8].long()
            logical = torch.where(blocks >= 0, blocks * 8 + logical % 8, -1)
        pages = table[row].cpu()[logical.clamp(min=0) // 64].long()
        valid = (logical >= 0) & (logical < int(visible[row])) & (pages >= 0)
        slots = pages[valid] * 64 + logical[valid] % 64
        dots = q_reference[row] @ key_reference[slots].float().T
        expected[row, valid] = (dots.relu() * weights[row].cpu()[:, None]).sum(0)
    torch.testing.assert_close(out.cpu(), expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize("missing_page", [-1, 3])
def test_index_topk_missing_latest_page(device, require, solution, missing_page):
    require("attention", "dsv41_index_topk", solution, torch.bfloat16, "x")
    q = torch.ones((3, 32, 128), dtype=torch.bfloat16, device=device)
    weights = -torch.ones((3, 32), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((3, 64, 68), dtype=torch.uint8, device=device)
    dsv41.cache_scatter(
        torch.ones((192, 128), dtype=torch.bfloat16, device=device),
        cache,
        torch.arange(192, device=device),
        "index",
    )
    table = torch.tensor(
        [[0, missing_page, 2, missing_page], [missing_page] * 4, [0, 1, 2, 0]],
        dtype=torch.int32,
        device=device,
    )
    visible = torch.tensor([256, 256, 0], dtype=torch.int32, device=device)
    rows, row_lens, blocks, block_lens = dsv41.index_topk(
        q, weights, cache, table, visible, None, 512, 32, 8, 2, 64, None, None, solution
    )
    expected_rows = torch.cat(
        [torch.arange(64, device=device), torch.arange(128, 192, device=device)]
    ).to(torch.int32)
    expected_blocks = torch.cat(
        [torch.arange(8, device=device), torch.arange(16, 24, device=device)]
    ).to(torch.int32)
    torch.testing.assert_close(rows[0, :128], expected_rows)
    torch.testing.assert_close(blocks[0, :16], expected_blocks)
    assert row_lens.tolist() == [128, 0, 0]
    assert block_lens.tolist() == [16, 0, 0]
    assert (rows[0, 128:] == -1).all() and (rows[1:] == -1).all()
    assert (blocks[0, 16:] == -1).all() and (blocks[1:] == -1).all()


@pytest.mark.parametrize("solution", ["triton", "gluon"])
@pytest.mark.parametrize(("length", "expected_block"), [(256, 31), (320, 31)])
def test_index_topk_latest_block_outside_table(
    device, require, solution, length, expected_block
):
    require("attention", "dsv41_index_topk", solution, torch.bfloat16, "x")
    q = torch.ones((1, 32, 128), dtype=torch.bfloat16, device=device)
    weights = -torch.ones((1, 32), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((3, 64, 68), dtype=torch.uint8, device=device)
    keys = (
        torch.arange(1, 193, dtype=torch.bfloat16, device=device)[:, None]
        .expand(-1, 128)
        .contiguous()
    )
    dsv41.cache_scatter(keys, cache, torch.arange(192, device=device), "index")
    table = torch.tensor([[0, 1, 2, 2]], dtype=torch.int32, device=device)
    visible = torch.tensor([length], dtype=torch.int32, device=device)
    output = dsv41.index_topk(
        q, weights, cache, table, visible, None, 16, 1, 8, 1, 64, None, None, solution
    )
    assert output[2].item() == expected_block
    assert output[3].item() == 1
    # Visibility is capped to table capacity without modifying the caller's input.
    assert visible.item() == length


# A gap between arena pages leaves a page stride that is not a 16-byte multiple.
@pytest.mark.parametrize(
    ("byte_stride", "page_gap"),
    [(1, 0), (2, 0), (1, 13)],
    ids=["arena", "byte-strided", "unaligned-pages"],
)
@pytest.mark.parametrize("reindex", [False, True])
def test_index_topk_preserves_arena_page_stride(
    device, require, monkeypatch, reindex, byte_stride, page_gap
):
    require("attention", "dsv41_index_topk", "gluon", torch.bfloat16, "x")
    q = torch.ones((1, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.ones((1, 32), dtype=torch.bfloat16, device=device)
    page_bytes = 2 * 64 * 68 * byte_stride
    storage = torch.zeros((4, page_bytes + page_gap), dtype=torch.uint8, device=device)
    arena = storage[:, :page_bytes].view(4, 2, 64, 68 * byte_stride)
    cache = arena[:, 0, :, ::byte_stride]
    assert not cache.is_contiguous()
    dsv41.cache_scatter(
        torch.ones((256, 128), dtype=torch.bfloat16, device=device),
        cache,
        torch.arange(256, device=device),
        "index",
    )
    table = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)
    visible = torch.tensor([320], dtype=torch.int32, device=device)
    candidates = (
        torch.tensor([[0, 16, 32, -1]], dtype=torch.int32, device=device)
        if reindex
        else None
    )
    candidate_topk = 0 if reindex else 32

    def run(pages):
        return dsv41.index_topk(
            q,
            weights,
            pages,
            table,
            visible,
            candidates,
            512,
            candidate_topk,
            8,
            1,
            64,
            None,
            None,
            "gluon",
        )

    expected = run(cache.contiguous())
    name = (
        "launch_gfx950_logits"
        if current_platform().is_cdna4
        else "launch_gfx1250_logits"
    )
    launch = getattr(gluon_backend, name)
    calls = []

    def check_page_view(
        q, w, cache_2d, table, visible, candidates, logits, score_chunk_size
    ):
        assert cache_2d.stride(1) == 1
        if byte_stride == 1 and not page_gap:
            assert cache_2d.data_ptr() == cache.data_ptr()
            assert cache_2d.stride(0) == cache.stride(0)
        else:
            assert cache_2d.is_contiguous()
            assert cache_2d.data_ptr() != cache.data_ptr()
        calls.append((cache_2d.shape, score_chunk_size))
        return launch(
            q,
            w,
            cache_2d,
            table,
            visible,
            candidates,
            logits,
            score_chunk_size,
        )

    monkeypatch.setattr(gluon_backend, name, check_page_view)
    actual = run(cache)
    assert calls == [((4, 64 * 68), 64)]
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert actual[1].item() == (16 if reindex else 256)


@pytest.mark.parametrize("rows", [0, 6, 12, 24, 48, 96, 192, 257])
@pytest.mark.parametrize("heads", [1, 4, 32])
@pytest.mark.parametrize("strided", [False, True])
def test_index_query_planes_match_codec_and_padding(device, rows, heads, strided):
    torch.manual_seed(333)
    storage = torch.randn(rows, heads * 2, 128, dtype=torch.bfloat16, device=device)
    ws = torch.randn(rows, heads * 2, dtype=torch.bfloat16, device=device)
    q = storage[:, ::2] if strided else storage[:, :heads].contiguous()
    w = ws[:, ::2] if strided else ws[:, :heads].contiguous()
    packed = triton_backend.cache_pack(
        q.contiguous().reshape(rows * heads, 128), "index", None
    )
    expected_values = packed[:, :64].reshape(rows, heads, 64).contiguous()
    expected_scales = (
        packed[:, 64:68].contiguous().view(torch.int32).reshape(rows, heads)
    )
    if heads < 32:
        expected_values = torch.nn.functional.pad(
            expected_values, (0, 0, 0, 32 - heads)
        )
        expected_scales = torch.nn.functional.pad(expected_scales, (0, 32 - heads))
    expected_weights = torch.nn.functional.pad(w.float(), (0, 32 - heads))
    actual = gluon_indexer._pack_index_q(q, w)
    for got, expected in zip(
        actual, (expected_values, expected_scales, expected_weights)
    ):
        assert got.is_contiguous()
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_index_query_planes_large_inner_stride(device):
    stride = 1 << 25
    elements = 191 * stride + 1
    required_bytes = elements * torch.bfloat16.itemsize
    if torch.cuda.mem_get_info(device)[0] < required_bytes + (256 << 20):
        pytest.skip("requires space for a large-stride query view")
    storage = torch.empty(elements, dtype=torch.bfloat16, device=device)
    # Keep wrapped int32 reads in bounds so a regression fails by value.
    storage.as_strided((192,), (stride,)).fill_(-17)
    q = storage.as_strided((1, 1, 128), (0, 0, stride), storage_offset=64 * stride)
    q.copy_(torch.arange(128, dtype=torch.bfloat16, device=device).view(1, 1, 128))
    w = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    expected = gluon_indexer._pack_index_q(q.contiguous(), w)
    actual = gluon_indexer._pack_index_q(q, w)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_index_query_planes_do_not_specialize_on_rows_or_row_strides(device):
    q = torch.randn(257, 32, 128, dtype=torch.bfloat16, device=device)
    w = torch.randn(257, 32, dtype=torch.bfloat16, device=device)
    gluon_indexer._pack_index_q(q[:6], w[:6])
    with assert_no_triton_compile(gluon_indexer._pack_index_queries_padded):
        for rows in (12, 24, 48, 96, 192, 257):
            gluon_indexer._pack_index_q(q[:rows], w[:rows])
        for padding in (1, 7, 31):
            padded_q = torch.randn(12, 32 * 128 + padding, dtype=q.dtype, device=device)
            padded_w = torch.randn(12, 32 + padding, dtype=w.dtype, device=device)
            gluon_indexer._pack_index_q(
                padded_q[:, : 32 * 128].view(12, 32, 128), padded_w[:, :32]
            )
