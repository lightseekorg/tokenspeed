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

"""FP8 QSA graph refresh across target and draft row geometries on Rubin."""

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform().arch_version != ArchVersion(10, 7),
    reason="Rubin FP8 QSA head-tile regression",
)


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    ("rows", "query_width", "large_offsets", "kv_heads"),
    [
        # Below one output tile per SM: sixteen, four and two KV splits.
        (4, 1, False, 2),
        (8, 1, False, 2),
        (32, 1, False, 2),
        (32, 1, False, 1),
        (96, 1, False, 2),
        (128, 4, False, 2),
        (384, 4, False, 2),
        (128, 4, True, 2),
        (384, 4, False, 1),
    ],
)
def test_fp8_target_and_draft_graphs_refresh_without_compilation(
    rows, query_width, large_offsets, kv_heads, enable_pdl
):
    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.ops.attention.qsa import qsa_sparse_attention
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse
    from tokenspeed_kernel.workspace import capture_session
    from utils import move_pages_past_int32

    torch.manual_seed(443 + rows)
    query = torch.randn((rows, kv_heads * 12, 256), device="cuda", dtype=torch.bfloat16)
    keys = (torch.randn((4096, kv_heads, 256), device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    values = (torch.randn((4096, kv_heads, 256), device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    keys[0].fill_(float("nan"))
    values[0].fill_(float("nan"))
    selected = torch.randint(1, 4096, (rows, 2051), device="cuda", dtype=torch.int32)
    selected[:, 9::17] = 0
    selected[::4, -3:] = -1
    selected[-query_width:] = -1
    if large_offsets:
        (keys, values), far = move_pages_past_int32((keys, values), selected)
        selected = torch.where(selected > 0, far, selected)
        assert int(selected[selected > 0].min()) * keys[0].numel() > 2**31
        keys[1].zero_()
        values[1].zero_()
    qsa_sparse.warmup(query, keys, values, max_rows=384, enable_pdl=enable_pdl)
    compiled = dict(qsa_sparse._COMPILED_KERNELS)
    previous = pdl_enabled()
    pdl_enabled(enable_pdl)
    try:

        def run():
            return qsa_sparse_attention(
                query,
                keys,
                values,
                selected,
                scale=1 / 16,
                max_seqlen_q=query_width,
                metadata_capacity_rows=None,
                k_scale=0.5,
                v_scale=0.25,
                override=None,
                solution="cute_dsl",
                sparse_metadata=None,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with capture_session(graph), torch.cuda.graph(graph):
            output = run()
        for factor in (0.75, -0.5):
            query.mul_(factor)
            selected[:, 1000:2048] = -1
            values[-4095:].copy_(
                (torch.randn_like(values[-4095:], dtype=torch.float32) * 0.25).to(
                    values.dtype
                )
            )
            graph.replay()
            expected = torch_qsa_selected_attention(
                query, keys, values, selected, 1 / 16, 0.5, 0.25
            )
            torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)
            torch.testing.assert_close(run(), output, atol=0, rtol=0)
            assert torch.count_nonzero(output[-query_width:]) == 0
        assert qsa_sparse._COMPILED_KERNELS == compiled
    finally:
        pdl_enabled(previous)


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "rows"),
    [
        (24, 2, 4),
        (24, 2, 8),
        (24, 2, 32),
        (12, 1, 8),
        (12, 1, 32),
        (5, 1, 8),
        (9, 1, 8),
        (16, 1, 8),
    ],
)
def test_small_fp8_grids_with_padded_heads_match_reference(q_heads, kv_heads, rows):
    """Odd and padded head counts preserve masking and paired softmax math;
    random prefixes put data in every split."""
    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    assert rows * kv_heads < sm_count
    torch.manual_seed(5 + rows)
    keys = torch.randn((65536, kv_heads, 256), device="cuda").to(torch.float8_e4m3fn)
    values = torch.randn((65536, kv_heads, 256), device="cuda").to(torch.float8_e4m3fn)
    if kv_heads == 1:
        # A lane shares one address calculation across two 16-byte copies;
        # the cache still only needs 16-byte base alignment.
        def offset_cache(cache):
            backing = torch.empty(
                cache.numel() + 16, device=cache.device, dtype=cache.dtype
            )
            shifted = backing[16:].view(cache.shape)
            shifted.copy_(cache)
            assert shifted.data_ptr() % 32 == 16
            return shifted

        keys, values = offset_cache(keys), offset_cache(values)
    keys[0].fill_(float("nan"))
    values[0].fill_(float("nan"))
    query = torch.randn((rows, q_heads, 256), device="cuda", dtype=torch.bfloat16) * 2
    # Zero logits keep the running maximum unchanged after the first tile.
    query[0].zero_()
    selected = torch.randint(1, 65536, (rows, 2051), device="cuda", dtype=torch.int32)
    selected[:, 13::19] = 0
    selected[::3, 1500:2048] = -1
    output = qsa_sparse.kernel(
        query,
        keys,
        values,
        selected,
        scale=1 / 16,
        max_seqlen_q=1,
        k_scale=0.5,
        v_scale=1.0,
        enable_pdl=False,
    )
    expected = torch_qsa_selected_attention(
        query, keys, values, selected, 1 / 16, 0.5, 1.0
    )
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_large_fp8_grids_refresh_across_split_boundary(enable_pdl):
    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse
    from tokenspeed_kernel.workspace import capture_session
    from utils import assert_no_cute_compile

    torch.manual_seed(821)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    def kv_splits(rows):
        return qsa_sparse._configuration(rows, 24, 2, 1, False, sm_count, 0)[2]

    # Past 128 rows the wave model drops from two KV splits to one where a
    # second split would add a CTA wave; replay graphs on both sides of it.
    boundary = next(r for r in range(129, 16384) if kv_splits(r) != kv_splits(r - 1))
    query = torch.randn((16384, 24, 256), device="cuda", dtype=torch.bfloat16)
    keys = (torch.randn((4096, 2, 256), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    values = (torch.randn((4096, 2, 256), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    keys[0].fill_(float("nan"))
    values[0].fill_(float("nan"))
    slots = torch.randint(1, 4096, (16384, 2051), device="cuda", dtype=torch.int32)
    slots[:, 13::19] = 0
    slots[::4, 1024:2048] = -1
    qsa_sparse.warmup(query, keys, values, max_rows=16384, enable_pdl=enable_pdl)

    def forward(q, selected):
        return qsa_sparse.kernel(
            q,
            keys,
            values,
            selected,
            scale=1 / 16,
            max_seqlen_q=1,
            k_scale=0.5,
            v_scale=0.25,
            enable_pdl=enable_pdl,
        )

    with assert_no_cute_compile():
        for rows in (boundary - 1, boundary, boundary + 1, 16384):
            q, selected = query[:rows], slots[:rows]
            forward(q, selected)
            graph = torch.cuda.CUDAGraph()
            with capture_session(graph), torch.cuda.graph(graph):
                output = forward(q, selected)
            sample = torch.tensor([0, 1, rows // 2, rows - 2, rows - 1], device="cuda")
            saved_last = selected[-1].clone()
            for empty_last in (True, False):
                selected[-1].copy_(
                    torch.full_like(saved_last, -1) if empty_last else saved_last
                )
                q.mul_(-0.75)
                graph.replay()
                expected = torch_qsa_selected_attention(
                    q[sample], keys, values, selected[sample], 1 / 16, 0.5, 0.25
                )
                torch.testing.assert_close(
                    output[sample], expected, atol=2e-3, rtol=2e-2
                )
                if empty_last:
                    assert torch.count_nonzero(output[-1]) == 0
                torch.testing.assert_close(forward(q, selected), output, atol=0, rtol=0)


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("rows", [8, 32, 96, 128])
def test_fp8_short_prefixes_and_private_tails_refresh_in_graph(rows, enable_pdl):
    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse
    from utils import assert_no_cute_compile

    torch.manual_seed(1841)
    query = torch.randn(rows, 24, 256, device="cuda", dtype=torch.bfloat16)
    keys, values = (
        (torch.randn(4096, 2, 256, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    keys[0].fill_(float("nan"))
    values[0].fill_(float("nan"))
    selected = torch.full((rows, 2051), -1, device="cuda", dtype=torch.int32)
    candidates = torch.randint(
        1, 4096, selected.shape, device="cuda", dtype=torch.int32
    )
    sample = [0, 1, 2, 3, rows // 2, rows - 1]
    qsa_sparse.warmup(query, keys, values, max_rows=rows, enable_pdl=enable_pdl)

    def run():
        return qsa_sparse.kernel(
            query,
            keys,
            values,
            selected,
            scale=1 / 16,
            max_seqlen_q=4,
            k_scale=0.5,
            v_scale=0.25,
            enable_pdl=enable_pdl,
        )

    graph = torch.cuda.CUDAGraph()
    with assert_no_cute_compile(), torch.cuda.graph(graph):
        output = run()
    # A one-tile prefix leaves fewer copy groups than the configured lookahead.
    # Tail-only and empty rows must still drain safely before shared storage reuse.
    with assert_no_cute_compile():
        for count in (0, 1, 127, 128, 129, 255, 256, 257, 383, 384, 385, 2048):
            selected.fill_(-1)
            selected[:, :count] = candidates[:, :count]
            selected[:, -3:] = candidates[:, -3:]
            for row in range(3):
                selected[row::4, 2048 + row :] = -1
            selected[0] = -1
            query.mul_(-0.75)
            values.copy_(
                (torch.randn_like(values, dtype=torch.float32) * 0.25).to(values.dtype)
            )
            values[0].fill_(float("nan"))
            # Repeated replays exercise ring wraparound and reuse of the
            # completed gather storage by the output epilogue.
            for _ in range(8):
                graph.replay()
            expected = torch_qsa_selected_attention(
                query[sample], keys, values, selected[sample], 1 / 16, 0.5, 0.25
            )
            torch.testing.assert_close(output[sample], expected, atol=1e-3, rtol=1e-2)
            assert torch.count_nonzero(output[0]) == 0
            torch.testing.assert_close(run(), output, atol=0, rtol=0)


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("query_rows_per_cta", [1, 2])
def test_fp8_pipeline_retirement_and_reuse_across_graph_replays(
    enable_pdl, query_rows_per_cta
):
    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse
    from utils import assert_no_cute_compile

    torch.manual_seed(3351)
    query = torch.randn(3, 24, 256, device="cuda", dtype=torch.bfloat16)
    keys, values = (
        (torch.randn(4096, 2, 256, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    keys[0].fill_(float("nan"))
    values[0].fill_(float("nan"))
    selected = torch.randint(1, 4096, (3, 2051), device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    prefix_shared = selected[:, 0]
    # Cover single-row retirement and two-row reuse of the CTA's barriers.
    # The odd row count also gives the two-row tactic an absent second row.
    # Exercise both directly; the wave policy favors one row on this grid.
    compiled = qsa_sparse._compile_kernel(
        query,
        query,
        keys,
        values,
        selected,
        output,
        kv_splits=1,
        enable_pdl=enable_pdl,
        query_rows_per_cta=query_rows_per_cta,
        bf16_smem_slots=1,
        bf16_async_ready=False,
        head_tile=16,
        query_group=1,
        prefix_shared=prefix_shared,
        distinct_bf16_smem_slots=1,
    )

    def run():
        compiled(
            query,
            query,
            keys.view(torch.uint8),
            values.view(torch.uint8),
            selected,
            prefix_shared,
            output,
            1 / 16,
            0.5,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with assert_no_cute_compile(), torch.cuda.graph(graph):
        run()
    with assert_no_cute_compile():
        for live_row in (0, 1, 2, 0):
            selected.fill_(-1)
            selected[live_row].random_(1, 4096)
            selected[live_row, 513:2048] = -1
            query.mul_(-0.75)
            for _ in range(16):
                graph.replay()
            expected = torch_qsa_selected_attention(
                query, keys, values, selected, 1 / 16, 1.0, 0.5
            )
            torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)
            assert (
                torch.count_nonzero(output[torch.arange(3, device="cuda") != live_row])
                == 0
            )
            saved = output.clone()
            run()
            torch.testing.assert_close(output, saved, atol=0, rtol=0)


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(("k_scale", "v_scale"), [(1.0, 1.0), (0.5, 1.25)])
def test_fp8_copy_pipelines_preserve_identical_head_bits(enable_pdl, k_scale, v_scale):
    """Duplicated KV heads agree across the three/four-deep copy pipelines."""
    from tokenspeed_kernel.thirdparty.cute_dsl import qsa_sparse
    from utils import assert_no_cute_compile

    torch.manual_seed(947)
    query = torch.randn(384, 12, 256, device="cuda", dtype=torch.bfloat16)
    keys, values = [
        (torch.randn(4096, 1, 256, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        for _ in range(2)
    ]
    keys[0] = float("nan")
    values[0] = float("nan")
    selected = torch.randint(1, 4096, (384, 2051), device="cuda", dtype=torch.int32)
    selected[:, ::17] = -1
    selected[-4:] = -1
    inputs = [
        (query, keys, values),
        tuple(tensor.repeat(1, 2, 1) for tensor in (query, keys, values)),
    ]

    def run(q, k, v):
        return qsa_sparse.kernel(
            q,
            k,
            v,
            selected,
            scale=1 / 16,
            max_seqlen_q=4,
            k_scale=k_scale,
            v_scale=v_scale,
            enable_pdl=enable_pdl,
        )

    graphs, outputs = [], []
    for tensors in inputs:
        run(*tensors)
        graph = torch.cuda.CUDAGraph()
        with assert_no_cute_compile(), torch.cuda.graph(graph):
            output = run(*tensors)
        graphs.append(graph)
        outputs.append(output)
    for _ in range(3):
        query.mul_(0.75)
        for tensor in (keys, values):
            raw = tensor[1:].view(torch.uint8)
            raw.copy_(raw.roll(1, dims=0))
        selected.copy_(selected.roll(1, dims=0))
        for src, dst in zip(inputs[0], inputs[1], strict=True):
            dst.copy_(src.repeat(1, 2, 1))
        for graph in graphs:
            graph.replay()
        for heads in (slice(0, 12), slice(12, 24)):
            assert torch.equal(
                outputs[0].view(torch.int16),
                outputs[1][:, heads].contiguous().view(torch.int16),
            )
