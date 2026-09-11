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
from kimi3_reference import dequantize_mxfp4
from tokenspeed_kernel.ops.attention.dsv4 import (
    dsv4_decode_topk,
    dsv4_plan,
    dsv4_prefill_topk,
)
from tokenspeed_kernel.ops.attention.dsv4._triton.indexer import _indexer_logits
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectionObjective, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def test_indexer_selection(
    mi350_platform: PlatformInfo, mi450_platform: PlatformInfo
) -> None:
    signature = format_signature(
        q=dense_tensor_format(torch.uint8),
        weights=dense_tensor_format(torch.float32),
        index_k_cache=dense_tensor_format(torch.uint8),
    )
    for platform in (mi350_platform, mi450_platform):
        for mode in ("prefill", "decode"):
            kernel = select_kernel(
                "attention",
                f"dsv4_{mode}_topk",
                signature,
                traits={
                    "index_heads": 64,
                    "head_dim": 128,
                    "topk": 512,
                    "page_size": 64,
                    "index_k_format": "mxfp4",
                },
                features=None,
                platform=platform,
                objective=SelectionObjective.DEFAULT,
                solution=None,
                override=None,
            )
            specialized = f"gluon_dsv4_{mode}_topk_mxfp4_gfx950"
            expected = (
                specialized
                if platform.is_cdna4
                and KernelRegistry.get().get_by_name(specialized) is not None
                else f"triton_dsv4_{mode}_topk_mxfp4"
            )
            assert kernel.name == expected


def _inputs(tokens: int, heads: int, pages: int):
    generator = torch.Generator(device="cuda").manual_seed(41)
    q = torch.randint(
        0,
        256,
        (tokens, heads, 64),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    scales = torch.randint(
        123,
        128,
        (tokens, heads, 4),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    weights = torch.randn(
        (tokens, heads), device="cuda", dtype=torch.float32, generator=generator
    )
    keys = torch.randint(
        0, 256, (pages, 64, 64), device="cuda", dtype=torch.uint8, generator=generator
    )
    key_scales = torch.randint(
        123, 128, (pages, 64, 4), device="cuda", dtype=torch.uint8, generator=generator
    )
    storage = torch.full((pages, 64 * 68 + 128), 255, device="cuda", dtype=torch.uint8)
    storage[:, : 64 * 64] = keys.reshape(pages, -1)
    storage[:, 64 * 64 : 64 * 68] = key_scales.reshape(pages, -1)
    cache = storage[:, : 64 * 68]
    q_pair = (q, scales.view(torch.int32).squeeze(-1))
    return (
        q_pair,
        weights,
        cache,
        dequantize_mxfp4(q, scales, group_size=32),
        dequantize_mxfp4(keys, key_scales, group_size=32),
    )


def _reference(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    tables: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.full((q.shape[0], tables.shape[1] * 64), -torch.inf, device="cuda")
    for token in range(q.shape[0]):
        count = int(lengths[token])
        if count:
            k = keys[tables[token].long()].reshape(-1, 128)[:count]
            logits[token, :count] = (
                (q[token].float() @ k.float().T).relu() * weights[token, :, None]
            ).sum(0)
    # Stable sorting fixes the lower-index tie contract independently of the kernel.
    order = logits.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    out = torch.full((q.shape[0], topk), -1, device="cuda", dtype=torch.int32)
    out[:, : order.shape[1]] = torch.where(
        logits.gather(1, order).isfinite(), order, -1
    ).to(torch.int32)
    return logits, out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("heads,topk", [(32, 512), (64, 1024), (64, 2048)])
def test_decode_logits_and_topk(heads: int, topk: int) -> None:
    q, weights, cache, q_ref, keys = _inputs(3, heads, 40)
    table_storage = torch.full((3, 40), -1, device="cuda", dtype=torch.int32)
    tables = table_storage[:, :36]
    tables.copy_(
        torch.stack((torch.arange(1, 37), torch.arange(36, 0, -1), torch.arange(2, 38)))
    )
    lengths = torch.tensor([2237, 65, 0], device="cuda", dtype=torch.int32)
    expected_logits, expected = _reference(q_ref, weights, keys, tables, lengths, topk)
    logits, _ = _indexer_logits(
        q,
        weights,
        cache,
        lengths,
        tables,
        page_size=64,
        max_candidates=tables.shape[1] * 64,
        cu_seq_lens=None,
        starts=None,
    )
    torch.testing.assert_close(logits, expected_logits, atol=2e-3, rtol=2e-5)
    out = torch.full((4, topk), 77, device="cuda", dtype=torch.int32)
    plan = dsv4_plan(
        page_size=64,
        seq_lens_2d=lengths[:, None],
        out=None,
        override=None,
        solution="triton",
    )
    actual = dsv4_decode_topk(
        q,
        weights,
        cache,
        lengths[:, None],
        tables,
        page_size=64,
        topk=topk,
        max_context_len=tables.shape[1] * 64,
        plan=plan,
        index_k_format="mxfp4",
        block_table_base_offsets=None,
        out=out,
        persistent_topk_workspace=None,
        override=None,
        solution=None,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert (out[-1] == 77).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("compact", [False, True])
def test_prefill_ragged_candidates(compact: bool) -> None:
    q, weights, cache, q_ref, keys = _inputs(4, 32, 30)
    tables = torch.stack((torch.arange(1, 14), torch.arange(25, 12, -1))).to(
        device="cuda", dtype=torch.int32
    )
    cu = torch.tensor([0, 800, 1570], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 3, 800, 800], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([700, 60, 513, 0], device="cuda", dtype=torch.int32)
    ends = starts + lengths
    bases = torch.tensor([7, 19], device="cuda", dtype=torch.int32) if compact else None
    actual, gathered = dsv4_prefill_topk(
        q,
        weights,
        cache,
        tables,
        cu,
        starts,
        ends,
        lengths,
        page_size=64,
        topk=512,
        max_seqlen_k=775,
        index_k_format="mxfp4",
        block_table_base_offsets=bases,
        gathered_k=None,
        gather_workspace=None,
        out=None,
        override=None,
        solution="triton",
    )
    assert gathered is None
    for token, request in enumerate((0, 0, 1, 1)):
        count = int(lengths[token])
        offset = int(starts[token] - cu[request])
        selected_keys = keys[tables[request].long()].reshape(-1, 128)[
            offset : offset + count
        ]
        scores = (
            (q_ref[token].float() @ selected_keys.float().T).relu()
            * weights[token, :, None]
        ).sum(0)
        expected = scores.argsort(descending=True, stable=True)[:512].to(torch.int32)
        if compact:
            expected += bases[request] * 64 + offset
        torch.testing.assert_close(actual[token, : min(count, 512)], expected)
        assert (actual[token, min(count, 512) :] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_decode_graph_refresh_and_padding() -> None:
    q, weights, cache, q_ref, keys = _inputs(2, 32, 16)
    tables = torch.arange(1, 13, device="cuda", dtype=torch.int32).repeat(2, 1)
    lengths = torch.tensor([[720], [0]], device="cuda", dtype=torch.int32)
    bases = torch.tensor([4, 20], device="cuda", dtype=torch.int32)
    out = torch.empty((2, 512), device="cuda", dtype=torch.int32)
    with torch.inference_mode():
        plan = dsv4_plan(
            page_size=64,
            seq_lens_2d=lengths,
            out=None,
            override=None,
            solution="triton",
        )

    def run():
        refreshed = dsv4_plan(
            page_size=64,
            seq_lens_2d=lengths,
            out=plan,
            override=None,
            solution="triton",
        )
        assert refreshed.data_ptr() == plan.data_ptr()
        return dsv4_decode_topk(
            q,
            weights,
            cache,
            lengths,
            tables,
            page_size=64,
            topk=512,
            max_context_len=768,
            plan=plan,
            index_k_format="mxfp4",
            block_table_base_offsets=bases,
            out=out,
            persistent_topk_workspace=None,
            override=None,
            solution="triton",
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    lengths.copy_(torch.tensor([[17], [641]], device="cuda", dtype=torch.int32))
    tables.copy_(tables.flip(1))
    bases.add_(3)
    graph.replay()
    _, expected = _reference(q_ref, weights, keys, tables, lengths.reshape(-1), 512)
    expected = torch.where(expected >= 0, expected + bases[:, None] * 64, -1)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(plan, lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_empty_indexer_and_invalid_pages() -> None:
    q, weights, cache, _, _ = _inputs(2, 32, 1)
    tables = torch.tensor([[-1], [1]], device="cuda", dtype=torch.int32)
    lengths = torch.full((2, 1), 64, device="cuda", dtype=torch.int32)
    for max_context in (0, 64):
        actual = dsv4_decode_topk(
            q,
            weights,
            cache,
            lengths,
            tables,
            page_size=64,
            topk=512,
            max_context_len=max_context,
            plan=None,
            index_k_format="mxfp4",
            block_table_base_offsets=None,
            out=None,
            persistent_topk_workspace=None,
            override=None,
            solution="triton",
        )
        assert (actual == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_zero_token_indexer() -> None:
    q, weights, cache, _, _ = _inputs(0, 32, 1)
    tables = torch.empty((0, 1), device="cuda", dtype=torch.int32)
    lengths = torch.empty((0,), device="cuda", dtype=torch.int32)
    bases = torch.empty((0,), device="cuda", dtype=torch.int64)
    actual = dsv4_decode_topk(
        q,
        weights,
        cache,
        lengths[:, None],
        tables,
        page_size=64,
        topk=512,
        max_context_len=64,
        plan=None,
        index_k_format="mxfp4",
        block_table_base_offsets=bases,
        out=None,
        persistent_topk_workspace=None,
        override=None,
        solution="triton",
    )
    assert actual.shape == (0, 512)
    assert actual.dtype == torch.int32
    actual, gathered = dsv4_prefill_topk(
        q,
        weights,
        cache,
        tables,
        torch.zeros((1,), device="cuda", dtype=torch.int32),
        lengths,
        lengths,
        lengths,
        page_size=64,
        topk=512,
        max_seqlen_k=64,
        index_k_format="mxfp4",
        block_table_base_offsets=bases,
        gathered_k=None,
        gather_workspace=None,
        out=None,
        override=None,
        solution="triton",
    )
    assert actual.shape == (0, 512)
    assert actual.dtype == torch.int32
    assert gathered is None
