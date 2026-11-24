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

"""Real GPU equivalence for SM120 indexer launch geometry."""

import os

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41

pytestmark = pytest.mark.skipif(
    os.environ.get("TOKENSPEED_ENABLE_SM120_PREFILL_INDEXER") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120 and explicit Prefill indexer opt-in",
)


@pytest.mark.parametrize(
    "tokens,history,weight_dtype,candidate_topk",
    [
        (257, 40960, torch.bfloat16, 0),
        (256, 8192, torch.float32, 0),
        (128, 1024, torch.bfloat16, 17),
    ],
)
def test_full_scan_preserves_selected_ids_and_lengths(
    tokens, history, weight_dtype, candidate_topk
):
    torch.manual_seed(271)
    device = "cuda"
    q = torch.randn(tokens, 32, 128, device=device, dtype=torch.bfloat16)
    w = torch.randn(tokens, 32, device=device, dtype=weight_dtype)
    keys = torch.randn(history, 128, device=device, dtype=torch.bfloat16)
    pages = history // 64 + 1
    # Reverse physical pages and leave the sentinel page untouched.
    cache = torch.zeros(pages, 64, 68, device=device, dtype=torch.uint8)
    table = torch.arange(pages - 1, 0, -1, device=device, dtype=torch.int32)
    slots = table.long().repeat_interleave(64) * 64
    slots += torch.arange(history, device=device) % 64
    dsv41.cache_scatter(keys, cache, slots, "index")
    table = table[None, :].expand(tokens, -1)
    visible = torch.full((tokens,), history, device=device, dtype=torch.int32)
    visible -= torch.arange(tokens - 1, -1, -1, device=device, dtype=torch.int32)
    visible[0] = 0
    args = (
        q,
        w,
        cache,
        table,
        visible,
        None,
        512,
        candidate_topk,
        8,
        256,
        256,
        None,
        None,
    )
    reference = dsv41.index_topk(*args, solution="triton")
    from tokenspeed_kernel.thirdparty.flashinfer.topk import has_exact_topk

    solutions = ["triton_sm120", None]
    if torch.cuda.get_device_capability() == (12, 0) and has_exact_topk():
        solutions.append("flashinfer_sm120")
    for solution in solutions:
        actual = dsv41.index_topk(*args, solution=solution)
        for got, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
    if candidate_topk:
        reindex_args = (
            q,
            w,
            cache,
            table,
            visible,
            reference[2],
            512,
            0,
            8,
            256,
            256,
            None,
            None,
        )
        expected = dsv41.index_topk(*reindex_args, solution="triton")
        actual = dsv41.index_topk(*reindex_args, solution="triton_sm120")
        for got, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ties", [False, True])
def test_dense_full_selection_graph_refreshes_metadata(weight_dtype, ties):
    from tokenspeed_kernel.thirdparty.flashinfer.topk import has_exact_topk

    if torch.cuda.get_device_capability() != (12, 0) or not has_exact_topk():
        pytest.skip("optional exact Full selector requires SM120 and FlashInfer TopK")
    torch.manual_seed(31)
    tokens, history = 256, 8192
    cache = torch.zeros(history // 64 + 1, 64, 68, device="cuda", dtype=torch.uint8)
    dsv41.cache_scatter(
        torch.randn(history, 128, device="cuda", dtype=torch.bfloat16),
        cache,
        torch.arange(64, history + 64, device="cuda"),
        "index",
    )
    table = torch.arange(1, history // 64 + 1, device="cuda", dtype=torch.int32)
    table = table[None].expand(tokens, -1).clone()
    table[3::7, 2] = -1
    q = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, 32, device="cuda", dtype=weight_dtype)
    if ties:
        weights.zero_()
    visible = torch.full((tokens,), history, device="cuda", dtype=torch.int32)
    args = (q, weights, cache, table, visible, None, 512, 0, 8, 256, 256, None, None)
    dsv41.index_topk(*args, solution="flashinfer_sm120")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dsv41.index_topk(*args, solution="flashinfer_sm120")
    for shift in (3, 5):
        q.copy_(torch.randn_like(q))
        if not ties:
            weights.copy_(torch.randn_like(weights))
        visible.copy_(torch.arange(tokens, device="cuda", dtype=torch.int32) + shift)
        visible[:7] = torch.tensor([0, 1, 17, 511, 512, 513, history], device="cuda")
        graph.replay()
        expected = dsv41.index_topk(*args, solution="triton")
        for got, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)


def test_dense_full_table_width_does_not_recompile():
    from tokenspeed_kernel.ops.attention.dsv41 import triton as impl
    from tokenspeed_kernel.thirdparty.flashinfer.topk import has_exact_topk
    from utils import assert_no_triton_compile

    if torch.cuda.get_device_capability() != (12, 0) or not has_exact_topk():
        pytest.skip("optional exact Full selector requires SM120 and FlashInfer TopK")
    q = torch.randn(256, 32, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 32, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(129, 64, 68, device="cuda", dtype=torch.uint8)
    dsv41.cache_scatter(
        torch.randn(8192, 128, device="cuda", dtype=torch.bfloat16),
        cache,
        torch.arange(64, 8256, device="cuda"),
        "index",
    )
    visible = torch.full((256,), 8192, device="cuda", dtype=torch.int32)

    def run(cols):
        pages = torch.full((256, cols), -1, device="cuda", dtype=torch.int32)
        pages[:, :128] = torch.arange(1, 129, device="cuda", dtype=torch.int32)
        return dsv41.index_topk(
            q,
            w,
            cache,
            pages,
            visible,
            None,
            512,
            0,
            8,
            256,
            256,
            None,
            None,
            solution="flashinfer_sm120",
        )

    expected = run(128)
    run(129)
    with assert_no_triton_compile(impl._index_scan_kernel):
        for cols in (144, 131, 263, 1024):
            for got, wanted in zip(run(cols), expected, strict=True):
                torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.skipif(
    os.environ.get("TOKENSPEED_ENABLE_SM120_NATIVE_FP4_INDEXER") == "1",
    reason="checks the BF16 default with native FP4 disabled",
)
def test_default_selector_keeps_bf16_when_native_fp4_is_disabled():
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature
    from tokenspeed_kernel.thirdparty.flashinfer.topk import has_exact_topk

    if not has_exact_topk():
        pytest.skip("requires the exact FlashInfer selector")
    q = torch.zeros((2, 32, 128), dtype=torch.bfloat16, device="cuda")
    weights = torch.ones((2, 32), dtype=torch.float32, device="cuda")
    cache = torch.zeros((1, 64, 68), dtype=torch.uint8, device="cuda")
    table = torch.zeros((1, 1), dtype=torch.int32, device="cuda").expand(2, -1)
    visible = torch.full((2,), 64, dtype=torch.int32, device="cuda")
    selected = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(q.dtype)),
        features=None,
        platform=None,
        traits=dsv41._index_topk_traits(32, True, cache, None),
        solution=None,
        override=None,
    )
    assert selected.name == "flashinfer_sm120_dsv41_index_topk"
    args = (q, weights, cache, table, visible, None, 512, 0, 8, 256, 256, None, None)
    automatic = dsv41.index_topk(*args, solution=None)
    explicit = dsv41.index_topk(*args, solution="flashinfer_sm120")
    for actual, expected in zip(automatic, explicit, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
