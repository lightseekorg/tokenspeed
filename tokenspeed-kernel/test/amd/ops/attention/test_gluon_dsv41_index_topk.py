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
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import gluon as gluon_backend
from tokenspeed_kernel.ops.attention.dsv41._gluon import indexer as gluon_indexer
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

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


@pytest.mark.parametrize("solution", ["triton", "gluon"])
def test_index_topk_rejects_invalid_page_table_shape(device, require, solution):
    require("attention", "dsv41_index_topk", solution, torch.bfloat16, "x")
    with pytest.raises(ValueError, match="page_table must be"):
        dsv41.index_topk(
            torch.ones((1, 32, 128), dtype=torch.bfloat16, device=device),
            torch.ones((1, 32), dtype=torch.bfloat16, device=device),
            torch.zeros((1, 64, 68), dtype=torch.uint8, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
            None,
            16,
            0,
            8,
            1,
            64,
            None,
            None,
            solution,
        )


@pytest.mark.parametrize(
    ("arch", "dtype", "width", "metric", "kernel_name"),
    [
        ("gfx950", torch.uint8, 64, "flops4", "gluon_dsv41_index_topk_gfx950"),
        ("gfx1250", torch.bfloat16, 128, "flops16", "gluon_dsv41_index_topk_gfx1250"),
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
    ("arch", "entry_name", "kernel_name"),
    [
        ("gfx950", "dsv41_index_logits_gfx950", "gluon_dsv41_index_topk_gfx950"),
        ("gfx1250", "dsv41_index_logits_gfx1250", "gluon_dsv41_index_topk_gfx1250"),
    ],
)
@pytest.mark.parametrize(
    ("requested_chunk", "score_chunk", "hardware_chunk"),
    [(8, 8, 32), (64, 64, 64), (512, 256, 256)],
)
def test_indexer_score_chunk_bound(
    monkeypatch,
    arch,
    entry_name,
    kernel_name,
    requested_chunk,
    score_chunk,
    hardware_chunk,
):
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
    args = (
        (
            torch.empty((tokens, 32, 64), dtype=torch.uint8),
            torch.empty((tokens, 32), dtype=torch.int32),
            weights,
        )
        if arch == "gfx950"
        else (torch.empty((tokens, 32, 128), dtype=torch.bfloat16), weights)
    )

    getattr(module, entry_name)(
        *args,
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


@pytest.mark.parametrize("byte_stride", [1, 2])
@pytest.mark.parametrize("reindex", [False, True])
def test_index_topk_preserves_arena_page_stride(
    device, require, monkeypatch, reindex, byte_stride
):
    require("attention", "dsv41_index_topk", "gluon", torch.bfloat16, "x")
    q = torch.ones((1, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.ones((1, 32), dtype=torch.bfloat16, device=device)
    arena = torch.zeros((4, 2, 64, 68 * byte_stride), dtype=torch.uint8, device=device)
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
        if byte_stride == 1:
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
