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

"""Kernels launched with batch-shaped arguments must not recompile per shape.

Each test warms a kernel on shapes covering the integer specialization classes
Triton still keys on for runtime scalars (divisible by 16 or not), then sweeps
the dimension that follows the batch -- a token or row count, a table or
score width, a split count -- inside ``assert_no_triton_compile``, and checks
the results against a reference or against an equivalent narrower launch.
"""

import pytest
import torch
import torch.nn.functional as F
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEVICE = "cuda"


def test_packed_qkv_complex_rotary_token_count():
    from tokenspeed_kernel.ops.attention.mha._triton import qkv_rotary

    heads, dim = 2, 64

    def run(tokens):
        qkv = torch.randn(tokens, 3 * heads * dim, device=DEVICE, dtype=torch.bfloat16)
        angles = torch.randn(tokens, dim // 2, device=DEVICE)
        freqs = torch.polar(torch.ones_like(angles), angles)
        q, k, v = qkv_rotary.packed_qkv_complex_rotary(
            qkv, heads * dim, heads * dim, heads, dim, freqs, copy_v=True
        )
        pairs = qkv.float().view(tokens, 3, heads, dim // 2, 2)
        rotated = torch.view_as_real(
            torch.view_as_complex(pairs[:, :2].contiguous()) * freqs[:, None, None]
        ).flatten(-2)
        torch.testing.assert_close(q.float(), rotated[:, 0], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.float(), rotated[:, 1], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v, qkv.view(tokens, 3, heads, dim)[:, 2])

    run(32)
    run(33)
    with assert_no_triton_compile(qkv_rotary._packed_qkv_complex_rotary_kernel):
        for tokens in (48, 97, 130, 1483):
            run(tokens)


def test_packed_qkv_neox_rotary_token_count():
    from tokenspeed_kernel.ops.attention.mha._triton import qkv_rotary

    heads, dim = 2, 64

    def run(tokens):
        qkv = torch.randn(tokens, 3 * heads * dim, device=DEVICE, dtype=torch.bfloat16)
        angles = torch.randn(tokens, dim // 2, device=DEVICE).repeat(1, 2)
        cos, sin = angles.cos(), angles.sin()
        q, k, _ = qkv_rotary.packed_qkv_neox_rotary(
            qkv, heads * dim, heads * dim, heads, dim, cos, sin
        )
        x = qkv.float().view(tokens, 3, heads, dim)[:, :2]
        half = x.chunk(2, dim=-1)
        rotated = torch.cat((-half[1], half[0]), dim=-1)
        expected = x * cos[:, None, None] + rotated * sin[:, None, None]
        torch.testing.assert_close(q.float(), expected[:, 0], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.float(), expected[:, 1], rtol=1e-2, atol=1e-2)

    run(32)
    run(33)
    with assert_no_triton_compile(qkv_rotary._packed_qkv_neox_rotary_kernel):
        for tokens in (48, 97, 130, 1483):
            run(tokens)


def test_hadamard_row_count():
    from tokenspeed_kernel.ops.transform import triton as transform

    hadamard = torch.ones(1, 1, device=DEVICE)
    for _ in range(7):
        hadamard = torch.cat(
            (torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1))
        )

    def run(rows):
        x = torch.randn(rows, 128, device=DEVICE)
        out = transform.triton_hadamard_transform_128(x, scale=0.125)
        torch.testing.assert_close(out, x @ hadamard * 0.125, rtol=1e-5, atol=1e-4)

    run(32)
    run(33)
    with assert_no_triton_compile(transform._hadamard_128_kernel):
        for rows in (48, 97, 130, 1483):
            run(rows)


def test_accumulate_counts_token_count():
    from tokenspeed_kernel.ops.sampling.triton import penalties

    vocab = 1000

    def run(tokens):
        counts = torch.zeros(4, vocab, dtype=torch.int32, device=DEVICE)
        pool_idx = torch.randint(0, 4, (tokens,), dtype=torch.int32, device=DEVICE)
        ids = torch.randint(-3, vocab + 3, (tokens,), dtype=torch.int64, device=DEVICE)
        weights = torch.randint(-2, 3, (tokens,), dtype=torch.int32, device=DEVICE)
        penalties.accumulate_counts_inplace(counts, pool_idx, ids, weights)
        valid = (ids >= 0) & (ids < vocab)
        expected = torch.zeros_like(counts).index_put_(
            (pool_idx[valid].long(), ids[valid]), weights[valid], accumulate=True
        )
        torch.testing.assert_close(counts, expected, rtol=0, atol=0)

    run(32)
    run(33)
    with assert_no_triton_compile(penalties._accumulate_counts_inplace_kernel):
        for tokens in (48, 97, 130, 1483):
            run(tokens)


def test_l2norm_row_count():
    from tokenspeed_kernel.ops.attention.gdn._triton import l2norm

    def run(tokens):
        x = torch.randn(tokens, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        out = l2norm.l2norm_fwd(x, 1e-6)
        expected = x.float() * torch.rsqrt(x.float().square().sum(-1, True) + 1e-6)
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

    run(8)
    run(9)
    with assert_no_triton_compile(l2norm.l2norm_fwd_kernel):
        for tokens in (12, 25, 33, 371):
            run(tokens)


def test_mxfp4_mm_row_count():
    from tokenspeed_kernel.ops.gemm import triton as gemm

    n, k = 64, 256
    a = torch.randint(0, 256, (1500, k // 2), dtype=torch.uint8, device=DEVICE)
    a_scales = torch.randint(124, 130, (1500, k // 32), dtype=torch.uint8)
    b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=DEVICE)
    b_scales = torch.randint(124, 130, (n, k // 32), dtype=torch.uint8, device=DEVICE)
    a_scales = a_scales.to(DEVICE)

    def run(rows):
        return gemm.triton_mm_mxfp4(
            a[:rows], b, a_scales[:rows], b_scales, torch.bfloat16
        )

    full = run(1500)
    run(32)
    with assert_no_triton_compile(gemm._mxfp4_mm_kernel):
        for rows in (48, 97, 130, 1483):
            # Rows are independent: a shorter batch is a prefix of the longer.
            torch.testing.assert_close(run(rows), full[:rows], rtol=0, atol=0)


def _compact_reference(table, requests, causal, page):
    rows, cols = requests.numel(), table.shape[1]
    pages = torch.zeros(rows, cols, dtype=torch.int32)
    positions = torch.zeros_like(pages)
    lengths = torch.zeros(rows, 1, dtype=torch.int32)
    for row, (req, length) in enumerate(zip(requests.tolist(), causal.tolist())):
        if not 0 <= req < table.shape[0]:
            continue
        valid = [
            col
            for col, value in enumerate(table[req].tolist())
            if value > 0 and col * page < length
        ]
        pages[row, : len(valid)] = table[req, valid].cpu()
        positions[row, : len(valid)] = torch.tensor(valid, dtype=torch.int32)
        lengths[row] = sum(min(page, length - col * page) for col in valid)
    return pages, positions, lengths


def test_dsa_index_candidates_table_width():
    from tokenspeed_kernel.ops.attention.dsa._triton import index_candidates as ic

    page, topk = 16, 24
    base = torch.tensor(
        [[3, 0, 5, 7, 2, 9], [4, 6, -1, 8, 1, 11], [12, 13, 14, 15, 16, 17]],
        dtype=torch.int32,
        device=DEVICE,
    )
    requests = torch.tensor([0, 2, -1, 1, 2], dtype=torch.int32, device=DEVICE)
    causal = torch.tensor([90, 40, 50, 96, 7], dtype=torch.int32, device=DEVICE)

    def run(cols, rows):
        table = torch.zeros(rows, cols, dtype=torch.int32, device=DEVICE)
        table[: base.shape[0], : base.shape[1]] = base
        pages, positions, lengths = ic.compact_index_pages(
            table, requests, causal, page
        )
        want = _compact_reference(table.cpu(), requests.cpu(), causal.cpu(), page)
        for got, expected in zip((pages, positions, lengths), want, strict=True):
            torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)

        width = cols * page
        logits = torch.randn(requests.numel(), width, device=DEVICE)
        logits[0, 5] = float("nan")
        expected = logits.clone()
        ic.mask_index_scores(logits, positions, lengths, causal, page, 3, 5)
        col = torch.arange(width, device=DEVICE)
        valid = col[None] < lengths
        logical = positions.gather(
            1, (col // page)[None].expand_as(valid).clamp(max=cols - 1)
        )
        logical = logical * page + col % page
        forced = (logical < 3) | (logical >= causal[:, None] - 5)
        expected = torch.where(torch.isfinite(expected), expected, -float("inf"))
        expected = torch.where(valid & forced, float("inf"), expected)
        expected = torch.where(valid, expected, -float("inf"))
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

        offsets = torch.randint(-1, width, (requests.numel(), topk), device=DEVICE)
        logical_out, scores = ic.gather_index_candidates(
            offsets.int(), logits, positions, page
        )
        safe = offsets.clamp(min=0)
        score = torch.where(offsets >= 0, logits.gather(1, safe), -float("inf"))
        page_ids = positions.gather(1, (safe // page).clamp(max=cols - 1))
        live = (offsets >= 0) & (score > -float("inf"))
        torch.testing.assert_close(
            logical_out,
            torch.where(live, page_ids * page + safe % page, -1).int(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            scores, torch.where(live, score, -float("inf")), rtol=0, atol=0
        )

    # _compact buckets the column range to a power of two; stay in one bucket.
    run(16, 3)
    run(9, 4)
    with assert_no_triton_compile(ic._compact, ic._mask_scores, ic._gather_candidates):
        for cols, rows in ((10, 3), (11, 5), (13, 6), (15, 3)):
            run(cols, rows)


def test_dsa_mark_forced_logits_width():
    from tokenspeed_kernel.ops.attention.dsa._triton import topk as dsa_topk

    def run(cols):
        logits = torch.randn(5, cols, device=DEVICE)
        causal = torch.tensor([0, 3, cols, cols - 2, 9], device=DEVICE).clamp(max=cols)
        expected = logits.clone()
        dsa_topk.mark_forced_initial_local_logits(
            logits, causal, initial_tokens=2, local_tokens=4
        )
        col = torch.arange(cols, device=DEVICE)[None]
        initial_end = causal[:, None].clamp(max=2)
        local_start = torch.maximum(initial_end, causal[:, None] - 4)
        forced = (col < initial_end) | ((col >= local_start) & (col < causal[:, None]))
        expected[forced] = float("inf")
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    run(256)
    run(300)
    with assert_no_triton_compile(dsa_topk._mark_forced_initial_local_logits_kernel):
        for cols in (320, 517, 1030, 4099):
            run(cols)


def test_dsa_local_topk_to_global_slots_table_width():
    from tokenspeed_kernel.ops.attention.dsa._triton import topk as dsa_topk

    block_size, topk = 64, 16

    def run(cols):
        table = torch.randint(0, 1000, (3, cols), dtype=torch.int32, device=DEVICE)
        offsets = torch.randint(
            -1, cols * block_size + 64, (3, topk), dtype=torch.int32, device=DEVICE
        )
        slots, lens = dsa_topk.local_topk_to_global_slots(
            local_topk_offsets=offsets, block_table=table, block_size=block_size
        )
        # Without seq_lens every column of the table is visible.
        valid = (offsets >= 0) & (offsets < cols * block_size)
        safe = torch.where(valid, offsets, 0).long()
        expected = table.gather(1, safe // block_size) * block_size + safe % block_size
        torch.testing.assert_close(
            slots.long(), torch.where(valid, expected, -1).long(), rtol=0, atol=0
        )
        torch.testing.assert_close(lens, valid.sum(1).int(), rtol=0, atol=0)

    run(16)
    run(3)
    kernel = dsa_topk._local_topk_to_global_slots_kernel
    with assert_no_triton_compile(kernel):
        for cols in (5, 7, 13, 37):
            run(cols)
    # A single column specializes to a compile-time 1; the value must still
    # flow through the sequence-length bound.
    run(1)


def test_dsv4_gather_indexer_mxfp4_row_count():
    from tokenspeed_kernel.ops.attention.dsv4 import triton as dsv4

    values, scales = (
        dsv4.DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
        dsv4.DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
    )
    block_size, pages = 64, 32
    cache = torch.randint(
        0, 256, (pages, block_size * (values + scales)), dtype=torch.uint8
    ).to(DEVICE)
    slots = torch.randperm(pages * block_size, device=DEVICE)[:1500].to(torch.int32)
    slots[::7] = -1

    def run(rows):
        values_out = torch.empty(rows, values, dtype=torch.uint8, device=DEVICE)
        scales_out = torch.empty(rows, scales, dtype=torch.uint8, device=DEVICE)
        dsv4.dsv4_gather_indexer_mxfp4_cache(
            cache_2d=cache,
            slot_mapping=slots[:rows],
            values_out=values_out,
            scales_out=scales_out,
            block_size=block_size,
        )
        live = slots[:rows] >= 0
        page, pos = (slots[:rows] // block_size).long(), (slots[:rows] % block_size)
        value_cols = pos[:, None].long() * values + torch.arange(values, device=DEVICE)
        scale_cols = (
            block_size * values
            + pos[:, None].long() * scales
            + torch.arange(scales, device=DEVICE)
        )
        rows_bytes = cache[page.clamp(min=0)]
        torch.testing.assert_close(
            values_out[live], rows_bytes.gather(1, value_cols)[live], rtol=0, atol=0
        )
        torch.testing.assert_close(
            scales_out[live], rows_bytes.gather(1, scale_cols)[live], rtol=0, atol=0
        )

    run(32)
    run(33)
    with assert_no_triton_compile(dsv4._dsv4_gather_indexer_mxfp4_cache_kernel):
        for rows in (48, 97, 130, 1483):
            run(rows)


def test_dsv4_indexer_decode_metadata_table_geometry():
    from tokenspeed_kernel.ops.attention.dsv4 import triton as dsv4

    base = torch.tensor(
        [[4, 7, 1, -1], [2, 9, 3, 5], [6, -1, -1, -1]], dtype=torch.int32
    ).to(DEVICE)
    positions = torch.tensor([37, 250, 9, 511, 64], device=DEVICE)
    requests = torch.tensor([0, 1, 2, 1, 0], dtype=torch.int32, device=DEVICE)

    def run(rows, cols, max_blocks):
        table = torch.full((rows, cols), -1, dtype=torch.int32, device=DEVICE)
        table[:3, :4] = base
        context = torch.empty(5, dtype=torch.int32, device=DEVICE)
        tables = torch.empty(5, max_blocks, dtype=torch.int32, device=DEVICE)
        dsv4.dsv4_indexer_decode_metadata_compute(
            positions=positions,
            token_to_req_indices=requests,
            block_table=table,
            cache_block_size=16,
            compress_ratio=4,
            max_blocks=max_blocks,
            out_context_lens=context,
            out_block_tables=tables,
        )
        return context, tables[:, :4]

    expected = run(3, 4, 16)
    # The column loop is bucketed to a power of two; stay in the 32 bucket.
    run(5, 9, 32)
    run(4, 6, 20)
    kernel = dsv4._dsv4_indexer_decode_metadata_kernel
    with assert_no_triton_compile(kernel):
        # Extra request rows and -1 padding columns change nothing.
        for rows, cols, max_blocks in ((7, 11, 24), (9, 37, 27), (6, 13, 31)):
            for got, want in zip(run(rows, cols, max_blocks), expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_dsv41_index_topk_table_width():
    from test_attention_dsv41_index_scan import _inputs
    from tokenspeed_kernel.ops.attention import dsv41
    from tokenspeed_kernel.ops.attention.dsv41 import triton as implementation

    q, w, _, cache, table, visible = _inputs(DEVICE, torch.bfloat16)
    visible.copy_(torch.tensor([65, 513, 0], device=DEVICE))

    def run(cols):
        pages = torch.full((3, 2 * cols), -1, dtype=torch.int32, device=DEVICE)
        pages = pages[:, ::2]
        pages[:, :128].copy_(table)
        return dsv41.index_topk(
            q,
            w,
            cache,
            pages,
            visible,
            None,
            65,
            17,
            8,
            2,
            64,
            None,
            None,
            solution="triton",
        )

    # The table row stride is twice the width, so both keep one divisibility
    # class per width class.
    expected = run(128)
    run(129)
    with assert_no_triton_compile(implementation._index_scan_kernel):
        for cols in (144, 131, 263, 1024):
            for got, want in zip(run(cols), expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_dsv41_page_table_geometry():
    """The dsv41 address kernels take the page table's rows, columns and
    strides as runtime scalars. Rows follow the batch and columns the longest
    request, so a lone request (one row, which Triton otherwise folds into a
    constant) or a width crossing a multiple of 16 must not recompile."""
    from tokenspeed_kernel.ops.attention.dsv41 import triton as dsv41
    from tokenspeed_kernel.ops.attention.mla._triton import page_table

    base = torch.tensor([3, 1, 7, 2], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([0, 5, 63, 64, 130, 255, 256, -1], device=DEVICE)
    requests = torch.zeros(8, dtype=torch.int32, device=DEVICE)
    requests[-1] = -1
    selected = (torch.arange(48, device=DEVICE, dtype=torch.int32) * 7 % 70).view(8, 6)
    n = positions.numel()

    def run(rows, cols, col_stride):
        table = torch.full(
            (rows, cols * col_stride), -1, dtype=torch.int32, device=DEVICE
        )
        table = table[:, ::col_stride]
        table[0, :4] = base
        window = (
            torch.empty(n, dtype=torch.int64, device=DEVICE),
            torch.empty((n, 128), dtype=torch.int32, device=DEVICE),
            torch.empty(n, dtype=torch.int32, device=DEVICE),
        )
        dsv41.decode_window(positions, requests, *window, table, 8)
        compressor = tuple(
            torch.empty(n, dtype=dtype, device=DEVICE)
            for dtype in (
                torch.bool,
                torch.int64,
                torch.int32,
                torch.int64,
                torch.int64,
                torch.int64,
            )
        )
        dsv41.compressor_metadata(positions, requests, table, 8, *compressor)
        selection, lengths = dsv41.selection_table(positions, requests, table, 4)
        assert (selection[:, 4:] == -1).all()
        return (
            page_table.bounded_group_slots(positions, requests, table, 64, 1, 1, 8),
            dsv41.global_slots(selected, positions, requests, table, 4, 8),
            selection[:, :4],
            lengths,
            *window,
            *compressor,
        )

    expected = run(3, 8, 1)
    # Pages 3, 1, 7 hold columns 0-2; column 4 is padding and resolves to -1.
    assert expected[0][[1, 3, 4, 6]].tolist() == [197, 64, 450, -1]
    with assert_no_triton_compile(
        page_table._group_slots_kernel,
        dsv41._global_slots_kernel,
        dsv41._selection_table_kernel,
        dsv41._decode_window_kernel,
        dsv41._compressor_metadata,
    ):
        for rows, cols, col_stride in (
            (1, 4, 1),
            (1, 16, 1),
            (2, 33, 1),
            (16, 1024, 1),
            (17, 129, 2),
            (1, 6, 2),
        ):
            for got, want in zip(run(rows, cols, col_stride), expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_dsv41_cache_pack_row_count():
    from tokenspeed_kernel.ops.attention.dsv41 import triton as dsv41

    # Packing index queries gives each row its own slot, so the slot bound is
    # the row count. Rows are independent: a shorter batch is a prefix.
    x = torch.randn(1500, 128, device=DEVICE, dtype=torch.bfloat16)
    full_packed = dsv41.cache_pack(x, "index", None)
    full_unpacked = dsv41.cache_unpack(full_packed, "index", None)

    def run(rows):
        packed = dsv41.cache_pack(x[:rows], "index", None)
        unpacked = dsv41.cache_unpack(full_packed[:rows], "index", None)
        torch.testing.assert_close(packed, full_packed[:rows], rtol=0, atol=0)
        torch.testing.assert_close(unpacked, full_unpacked[:rows], rtol=0, atol=0)

    run(32)
    run(33)
    with assert_no_triton_compile(dsv41._pack_kernel, dsv41._gather_kernel):
        for rows in (48, 97, 130, 1483):
            run(rows)


def test_kda_prepare_capacity_scan_token_and_sequence_counts():
    from tokenspeed_kernel.ops.attention.kda._triton import prefill_scan_inputs as scan

    heads, dim = 2, 128

    def run(case):
        lengths, capacity = case
        live, sequences = sum(lengths), len(lengths)

        def randn(*shape):
            return torch.randn(1, capacity, *shape, device=DEVICE).bfloat16()

        q, k, v, gate = (randn(heads, dim) for _ in range(4))
        beta = randn(heads)
        bounds = F.pad(torch.tensor(lengths).cumsum(0), (1, 0)).int().to(DEVICE)
        oq, ok, ov, og, ob, chunks, chunk_rows = scan.prepare_capacity_scan(
            q, k, v, gate, beta, bounds, inputs_packed=False
        )
        rows = torch.arange(capacity, device=DEVICE) < live
        for got, x in ((oq, q), (ok, k), (ov, v), (og, gate.float())):
            torch.testing.assert_close(got, x * rows[None, :, None, None])
        torch.testing.assert_close(ob, beta * rows[None, :, None])
        counts = [triton_cdiv(n, 16) for n in lengths]
        expected = [s for s, c in enumerate(counts) for _ in range(c)]
        expected += [sequences - 1] * (chunk_rows.numel() - len(expected))
        assert chunks.tolist() == [0, *torch.tensor(counts).cumsum(0).tolist()]
        assert chunk_rows.tolist() == expected

    def key(case):
        lengths, capacity = case
        chunk_count = triton_cdiv(capacity, 16) + len(lengths) - 1
        counts = (capacity, len(lengths), chunk_count)
        return (
            *map(int_specialization_class, counts),
            1 << (len(lengths) - 1).bit_length(),
        )

    # Token capacity and live sequences both follow the batch.
    sweep = (
        ((20, 7, 33), 64),
        ((1, 2, 3, 4), 130),
        ((50, 60, 70, 5, 9, 11), 300),
        ((100, 3, 3, 3, 3), 381),
        ((8, 8, 8, 8, 8, 8, 8), 264),
        ((500, 17), 590),
    )
    pool = [
        ((max(1, capacity // (2 * n)),) * n, capacity)
        for n in range(1, 9)
        for capacity in (48, 49, 96, 97, 160, 161, 400, 401)
    ]
    warm_specialization_classes(run, key, sweep, pool)
    with assert_no_triton_compile(scan._prepare_capacity_scan_kernel):
        for case in sweep:
            run(case)


def test_causal_conv1d_capacity_metadata_counts():
    from tokenspeed_kernel.ops.attention.gdn._triton import causal_conv1d_metadata

    block_m = 8

    def run(case):
        lengths, capacity = case
        bounds = F.pad(torch.tensor(lengths).cumsum(0), (1, 0)).int().to(DEVICE)
        metadata = causal_conv1d_metadata.build_causal_conv1d_capacity_metadata(
            bounds, capacity, block_m
        )
        requests, offsets = [], []
        for request, length in enumerate(lengths):
            count = triton_cdiv(length, block_m)
            requests += [request] * count
            offsets += list(range(count))
        pad = metadata.batch_indices.numel() - len(requests)
        assert metadata.batch_indices.tolist() == requests + [-1] * pad
        assert metadata.chunk_offsets.tolist() == offsets + [0] * pad

    def key(case):
        lengths, capacity = case
        chunks = triton_cdiv(capacity, block_m) + len(lengths) - 1
        # The offsets map starts ``chunks`` int32s into one allocation, so its
        # 16-byte pointer alignment is a specialization class too.
        return (
            int_specialization_class(chunks),
            int_specialization_class(len(lengths)),
            chunks % 4 == 0,
        )

    # Token capacity and live sequences both follow the batch.
    sweep = (
        ((20, 7, 33), 64),
        ((1, 2, 3, 4), 130),
        ((50, 60, 70, 5, 9, 11), 300),
        ((100, 3, 3, 3, 3), 381),
        ((500, 17), 590),
    )
    pool = [
        ((max(1, capacity // (2 * n)),) * n, capacity)
        for n in range(1, 9)
        for capacity in (48, 49, 96, 97, 160, 161, 400, 401)
    ]
    warm_specialization_classes(run, key, sweep, pool)
    with assert_no_triton_compile(causal_conv1d_metadata._refresh_conv_capacity_kernel):
        for case in sweep:
            run(case)


def test_dp_sampling_kernels_bucket_size():
    from tokenspeed_kernel.ops.communication import triton as comm

    n, vocab = 4, 96

    def peers(tensor):
        return torch.tensor([tensor.data_ptr()], dtype=torch.int64, device=DEVICE)

    def run(reqs):
        logits = torch.randn(reqs * n, vocab, device=DEVICE, dtype=torch.bfloat16)
        received = torch.zeros_like(logits)
        comm._dp_sampling_swap_kernel[(reqs * n * triton_cdiv(vocab, 128),)](
            logits, peers(received), reqs, n, vocab, vocab, 0, 1, 128, 0
        )
        # One rank owns every request and the whole vocabulary: a copy.
        torch.testing.assert_close(received, logits, rtol=0, atol=0)

        predict = torch.randint(0, vocab, (reqs, n), dtype=torch.int32, device=DEVICE)
        accept = torch.randint(0, n, (reqs, n), dtype=torch.int32, device=DEVICE)
        length = torch.randint(1, n, (reqs,), dtype=torch.int32, device=DEVICE)
        outputs = [torch.zeros_like(x) for x in (predict, accept, length)]
        comm._dp_sampling_gather_kernel[(reqs,)](
            predict, accept, length, *map(peers, outputs), reqs, n, 0, 1, 4
        )
        for got, want in zip(outputs, (predict, accept, length), strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    run(16)
    run(3)
    with assert_no_triton_compile(
        comm._dp_sampling_swap_kernel, comm._dp_sampling_gather_kernel
    ):
        for reqs in (5, 6, 7, 9, 12):
            run(reqs)


def triton_cdiv(a, b):
    return (a + b - 1) // b


def test_merge_prefill_checkpoint_outputs_token_counts():
    from tokenspeed_kernel.ops.attention._triton import (
        prefill_state_checkpoints as ckpt,
    )

    def run(body_tokens, tail_tokens, extent):
        body = torch.randn(body_tokens, 3, 4, device=DEVICE).transpose(-1, -2)
        tail = torch.randn(tail_tokens, 3, 4, device=DEVICE).transpose(-1, -2)
        order = torch.randperm(extent, device=DEVICE)
        body_indices = order[:body_tokens].clone()
        tail_indices = order[body_tokens : body_tokens + tail_tokens].clone()
        body_indices[::5] = -1
        expected = torch.zeros(extent, 4, 3, device=DEVICE)
        for source, indices in ((body, body_indices), (tail, tail_indices)):
            live = indices >= 0
            expected[indices[live]] = source[live]
        merged = ckpt.merge_prefill_checkpoint_outputs(
            body, tail, body_indices, tail_indices, 0, extent, None
        )
        torch.testing.assert_close(merged, expected, rtol=0, atol=0)

        sources = torch.full((extent,), -1, dtype=torch.int64, device=DEVICE)
        concat = torch.cat((body_indices, tail_indices))
        live = concat >= 0
        sources[concat[live]] = torch.arange(concat.numel(), device=DEVICE)[live]
        gathered = ckpt.merge_prefill_checkpoint_outputs(
            body, tail, body_indices, tail_indices, 0, extent, sources
        )
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)

    run(16, 16, 40)
    run(9, 7, 21)
    with assert_no_triton_compile(
        ckpt._scatter_checkpoint_output_kernel, ckpt._gather_checkpoint_output_kernel
    ):
        for body_tokens, tail_tokens, extent in (
            (11, 5, 19),
            (37, 13, 60),
            (70, 3, 90),
        ):
            run(body_tokens, tail_tokens, extent)


def test_mla_decode_table_width_and_max_seqlen():
    from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache
    from tokenspeed_kernel.ops.attention.mla._triton import decode

    batch, heads, rank, rope, page = 3, 16, 512, 64, 64
    q = torch.randn(batch, 1, heads, rank + rope, device=DEVICE, dtype=torch.bfloat16)
    kv_cache = torch.randn(
        batch * 3, page, 1, rank + rope, device=DEVICE, dtype=torch.bfloat16
    )
    seqlens = torch.tensor([65, 64, 129], dtype=torch.int32, device=DEVICE)
    base = torch.arange(batch * 3, dtype=torch.int32, device=DEVICE).view(batch, 3)

    def run(cols, max_seqlen_k):
        table = torch.zeros(batch, cols, dtype=torch.int32, device=DEVICE)
        table[:, :3] = base
        return mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=table,
            cache_seqlens=seqlens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=128,
            kv_lora_rank=rank,
            qk_rope_head_dim=rope,
            softmax_scale=0.07,
            solution="triton",
        )

    expected = run(3, 129)
    run(16, 1024)
    with assert_no_triton_compile(decode._mla_decode_kernel):
        for cols, max_seqlen_k in ((5, 300), (7, 131), (19, 4097), (40, 1500)):
            torch.testing.assert_close(
                run(cols, max_seqlen_k), expected, rtol=0, atol=0
            )


def _paged_kv(kv_lens, heads, dim, page):
    pages = [(length + page - 1) // page for length in kv_lens]
    k_cache = torch.randn(sum(pages) + 1, page, heads, dim, device=DEVICE)
    v_cache = torch.randn_like(k_cache)
    table = torch.zeros(len(kv_lens), max(pages), dtype=torch.int32, device=DEVICE)
    start = 1
    for row, count in enumerate(pages):
        table[row, :count] = torch.arange(start, start + count, device=DEVICE)
        start += count
    return k_cache.bfloat16(), v_cache.bfloat16(), table


def _widen(table, cols):
    wide = torch.zeros(table.shape[0], cols, dtype=table.dtype, device=DEVICE)
    wide[:, : table.shape[1]] = table
    return wide


def test_mha_decode_and_extend_table_width():
    from tokenspeed_kernel.ops.attention.mha import (
        mha_decode_with_kvcache,
        mha_extend_with_kvcache,
    )
    from tokenspeed_kernel.ops.attention.mha._triton import decode, prefill

    kv_lens = [64, 130, 17, 191]
    k_cache, v_cache, table = _paged_kv(kv_lens, 2, 64, 64)
    seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    # 8 query heads over 2 KV heads take the grouped decode kernel, 2 over 2
    # the per-head one.
    q_grouped = torch.randn(4, 8, 64, device=DEVICE, dtype=torch.bfloat16)
    q_per_head = torch.randn(4, 2, 64, device=DEVICE, dtype=torch.bfloat16)
    q_lens = torch.tensor([1, 3, 2, 4], dtype=torch.int32, device=DEVICE)
    q_extend = torch.randn(10, 8, 64, device=DEVICE, dtype=torch.bfloat16)
    cu_q = F.pad(q_lens.cumsum(0, dtype=torch.int32), (1, 0))
    cu_kv = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))

    def run(cols):
        wide = _widen(table, cols)
        decoded = [
            mha_decode_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=wide,
                cache_seqlens=seqlens,
                max_seqlen_q=1,
                max_seqlen_k=256,
                solution="triton",
            )
            for q in (q_grouped, q_per_head)
        ]
        extended = mha_extend_with_kvcache(
            q=q_extend,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=wide,
            cache_seqlens=seqlens,
            max_seqlen_q=4,
            max_seqlen_k=191,
            solution="triton",
        )
        return *decoded, extended

    expected = run(table.shape[1])
    run(16)
    with assert_no_triton_compile(
        decode._fwd_kernel_stage1,
        decode._fwd_grouped_kernel_stage1,
        prefill._fwd_kernel,
    ):
        for cols in (5, 7, 19, 40):
            for got, want in zip(run(cols), expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_rel_mha_decode_table_width_and_split_count():
    from rel_mha_reference import HEAD_DIM, NUM_KV_HEADS, NUM_Q_HEADS, build_paged
    from tokenspeed_kernel.ops.attention.rmha import (
        rel_mha_decode_with_kvcache,
    )
    from tokenspeed_kernel.ops.attention.rmha import triton as rmha

    kv_lens = [70, 300, 17]
    k_cache, v_cache, table, _, _ = build_paged(kv_lens, DEVICE, 64)
    q = torch.randn(3, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    rel_logits = torch.randn(3, NUM_Q_HEADS, 64, device=DEVICE, dtype=torch.bfloat16)
    seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    cu_q = torch.arange(4, dtype=torch.int32, device=DEVICE)

    def run(cols, max_seqlen_k):
        # All query heads share the 2 KV heads (grouped kernel); the first 2
        # alone map one to one (per-head kernel).
        return [
            rel_mha_decode_with_kvcache(
                q=q[:, :heads],
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=_widen(table, cols),
                cache_seqlens=seqlens,
                max_seqlen_k=max_seqlen_k,
                rel_logits=rel_logits[:, :heads],
                cu_seqlens_q=cu_q,
                max_seqlen_q=1,
                softmax_scale=1.0 / HEAD_DIM,
                solution="triton",
            )
            for heads in (NUM_Q_HEADS, NUM_KV_HEADS)
        ]

    expected = run(table.shape[1], 300)
    run(16, 2 * 2048)
    with assert_no_triton_compile(
        rmha._rel_mha_decode_stage1_kernel,
        rmha._rel_mha_decode_grouped_stage1_kernel,
        rmha._rel_mha_decode_stage2_kernel,
    ):
        # The longest context picks 3, 5 and 11 KV splits here.
        for cols, max_seqlen_k in ((7, 5000), (19, 9000), (40, 21000)):
            for got, want in zip(run(cols, max_seqlen_k), expected, strict=True):
                torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


def test_rel_mha_extend_table_width():
    from rel_mha_reference import HEAD_DIM, NUM_Q_HEADS, build_paged
    from tokenspeed_kernel.ops.attention.rmha import rel_mha_extend_with_kvcache
    from tokenspeed_kernel.ops.attention.rmha import triton as rmha

    q_lens, kv_lens = [3, 5, 2], [70, 300, 17]
    k_cache, v_cache, table, _, _ = build_paged(kv_lens, DEVICE, 64)
    q = torch.randn(10, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    rel_logits = torch.randn(10, NUM_Q_HEADS, 64, device=DEVICE, dtype=torch.bfloat16)
    seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    cu_q = F.pad(
        torch.tensor(q_lens, device=DEVICE).cumsum(0, dtype=torch.int32), (1, 0)
    )
    cu_kv = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))

    def run(cols):
        return rel_mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=_widen(table, cols),
            cache_seqlens=seqlens,
            max_seqlen_q=5,
            max_seqlen_k=300,
            rel_logits=rel_logits,
            softmax_scale=1.0 / HEAD_DIM,
            solution="triton",
        )

    expected = run(table.shape[1])
    run(16)
    with assert_no_triton_compile(rmha._rel_mha_prefill_kernel):
        for cols in (7, 19, 40):
            torch.testing.assert_close(run(cols), expected, rtol=0, atol=0)


def test_kpool_prefill_chunk_scores_table_width():
    from test_kpool_select import _DIM, _KV_PAGE, _PAGE, _POOL, _setup
    from tokenspeed_kernel.ops.attention.kpool import kpool_prefill_topk
    from tokenspeed_kernel.ops.attention.kpool._triton import score

    # Four query rows at the end of each request; both see more pools than
    # topk, so they are scored in chunked windows.
    seq_lens = [900, 1300]
    q, cache, _, weights, _, index_table, kv_table = _setup(
        seq_lens, q_len_per_req=4, seed=53
    )
    positions = torch.cat(
        [torch.arange(n - 4, n, dtype=torch.int32, device=DEVICE) for n in seq_lens]
    )
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEVICE)

    def run(extra_cols):
        return kpool_prefill_topk(
            q,
            cache,
            weights,
            positions,
            query_start_loc,
            _widen(index_table, index_table.shape[1] + extra_cols),
            kv_table,
            pool_size=_POOL,
            page_size=_PAGE,
            kv_page_size=_KV_PAGE,
            topk_pools=64,
            softmax_scale=_DIM**-0.5,
            apply_relu=True,
            chunk_pools=128,
        )

    expected = run(0)
    run(10)
    with assert_no_triton_compile(score._kpool_score_prefill_chunk_kernel):
        for extra_cols in (1, 5, 13, 42):
            for got, want in zip(run(extra_cols), expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_kpool_dense_scores_width():
    from test_kpool_select import _DIM, _PAGE, _POOL, _setup
    from tokenspeed_kernel.ops.attention.kpool._triton import score
    from tokenspeed_kernel.ops.attention.kpool.triton import score_kpool_dense

    q, cache, _, weights, seq_lens, index_table, _ = _setup([2051, 4101], seed=47)
    req_ids = torch.arange(q.shape[0], dtype=torch.int32, device=DEVICE)
    num_pools = int((seq_lens // _POOL).max())

    def run(extra_pools, extra_cols):
        return score_kpool_dense(
            q,
            cache,
            weights,
            seq_lens,
            req_ids,
            _widen(index_table, index_table.shape[1] + extra_cols),
            pool_size=_POOL,
            page_size=_PAGE,
            softmax_scale=_DIM**-0.5,
            apply_relu=True,
            max_num_pools=num_pools + extra_pools,
        )

    expected = run(0, 0)
    run(16, 16)
    with assert_no_triton_compile(score._kpool_score_dense_mma_kernel):
        for extra_pools, extra_cols in ((3, 1), (21, 5), (130, 9)):
            got = run(extra_pools, extra_cols)
            torch.testing.assert_close(got[:, :num_pools], expected, rtol=0, atol=0)
            assert torch.all(got[:, num_pools:] == -float("inf"))


def test_compact_dcp_pages_table_width():
    from tokenspeed_kernel.ops.kvcache import triton_cache_placement as placement

    lengths = torch.tensor([300, 1, 129, 0], dtype=torch.int32)

    def run(cols):
        table = torch.randint(1, 40, (4, cols), dtype=torch.int32)
        outputs = []
        for device in ("cpu", DEVICE):
            out = torch.empty(4, cols, dtype=torch.int32, device=device)
            local = torch.empty(4, dtype=torch.int32, device=device)
            placement.compact_dcp_pages(
                table.to(device),
                lengths.to(device),
                page_size=64,
                block_granularity=128,
                virtual_block_count=20,
                degree=2,
                rank=1,
                out=out,
                local_lengths=local,
            )
            outputs.append((out.cpu(), local.cpu()))
        for got, want in zip(outputs[1], outputs[0], strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    # BLOCK buckets the column range to a power of two; stay in one bucket.
    run(16)
    run(9)
    with assert_no_triton_compile(placement._compact_owned_pages):
        for cols in (10, 11, 13, 15):
            run(cols)


def test_mhc_mixes_split_count():
    from test_mhc_prefill import _reference
    from tokenspeed_kernel.ops.residual import triton as residual

    hidden = 4096
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    x = torch.randn(1024, 4, hidden, device=DEVICE, generator=generator)
    x = x.bfloat16()
    fn = torch.randn(24, 4 * hidden, device=DEVICE, generator=generator) * 0.01
    scale = torch.tensor([0.7, 1.1, 0.5], device=DEVICE)
    base = torch.randn(24, device=DEVICE, generator=generator)

    def run(tokens):
        args = (x[:tokens], fn, scale, base, 1e-6, 1e-5, 3)
        pre, post, comb = residual.triton_mhc_mixes(*args)
        _, post_ref, comb_ref = _reference(*args)
        torch.testing.assert_close(post, post_ref.squeeze(-1), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(comb, comb_ref, rtol=1e-3, atol=1e-3)
        return pre

    def key(tokens):
        splits = residual.compute_mhc_num_splits(
            x.device, 64, 4 * hidden, triton_cdiv(tokens, 64)
        )
        config = residual._mhc_prenorm_gemm_launch_config(
            tokens, 4 * hidden, 24, splits
        )
        return (
            int_specialization_class(tokens),
            int_specialization_class(splits),
            config,
        )

    # Each batch size picks its own split count from the SM count.
    sweep = (128, 192, 320, 448, 1024)
    first = run(64)
    warm_specialization_classes(run, key, sweep, range(64, 1025, 64))
    with assert_no_triton_compile(
        residual._mhc_prenorm_gemm_triton_kernel, residual._mhc_pre_mix_hc4_kernel
    ):
        for tokens in sweep:
            pre = run(tokens)
            torch.testing.assert_close(pre[:64], first, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("sm_count", "hidden", "tokens", "expected"),
    [
        # MI355: the GFX950 pre-reduce-apply kernel needs hidden_size // 64.
        (256, 7168, 64, 112),
        (256, 4096, 64, 64),
        # H20 and B200: the SM fill is rounded down before the K-tile cap.
        (78, 7168, 64, 64),
        (148, 7168, 64, 112),
        (148, 7168, 128, 64),
        (148, 7168, 1024, 8),
    ],
)
def test_mhc_split_count_rounds_the_sm_fill_before_the_k_cap(
    monkeypatch, sm_count, hidden, tokens, expected
):
    from types import SimpleNamespace

    from tokenspeed_kernel.ops.residual import triton as residual

    props = SimpleNamespace(multi_processor_count=sm_count)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    splits = residual.compute_mhc_num_splits.__wrapped__(
        torch.device(DEVICE), 64, 4 * hidden, triton_cdiv(tokens, 64)
    )
    assert splits == expected


def test_mhc_pre_split_count():
    from test_mhc_prefill import _reference
    from tokenspeed_kernel.ops.residual import triton as residual

    hidden = 4096
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    x = torch.randn(256, 4, hidden, device=DEVICE, generator=generator).bfloat16()
    fn = torch.randn(24, 4 * hidden, device=DEVICE, generator=generator) * 0.01
    scale = torch.tensor([0.7, 1.1, 0.5], device=DEVICE)
    base = torch.randn(24, device=DEVICE, generator=generator)

    def run(tokens):
        # Up to 256 tokens take the split-K GEMM and the generic mix kernel.
        args = (x[:tokens], fn, scale, base, 1e-6, 1e-5, 3)
        actual = residual.triton_mhc_pre(*args, norm_weight=None, norm_eps=None)
        for got, want in zip(actual, _reference(*args), strict=True):
            torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)

    def key(tokens):
        splits = residual.compute_mhc_num_splits(
            x.device, 64, 4 * hidden, triton_cdiv(tokens, 64)
        )
        config = residual._mhc_prenorm_gemm_launch_config(
            tokens, 4 * hidden, 24, splits
        )
        return (
            int_specialization_class(tokens),
            int_specialization_class(splits),
            config,
        )

    # Below 256 SMs, two to four token tiles pick fewer splits than one.
    sweep = (65, 130, 200, 256)
    warm_specialization_classes(run, key, sweep, range(1, 257))
    with assert_no_triton_compile(
        residual._mhc_prenorm_gemm_triton_kernel, residual._mhc_pre_mix_triton_kernel
    ):
        for tokens in sweep:
            run(tokens)


def test_mhc_hc4_coefficients_split_count():
    from tokenspeed_kernel.ops.residual import triton as residual

    hidden, tokens = 4096, 37
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    mul = torch.randn(tokens, 24, device=DEVICE, generator=generator)
    sqrsum = torch.rand(tokens, device=DEVICE, generator=generator) * 4 * hidden
    scale = torch.tensor([0.7, 1.1, 0.5], device=DEVICE)
    base = torch.randn(24, device=DEVICE, generator=generator)

    def run(n_splits):
        # The whole projection sits in split 0 and the rest add exact zeros,
        # so every split count must reproduce the single-split result.
        gemm_mul = torch.zeros(n_splits, tokens, 24, device=DEVICE)
        gemm_sqrsum = torch.zeros(n_splits, tokens, device=DEVICE)
        gemm_mul[0], gemm_sqrsum[0] = mul, sqrsum
        pre = torch.empty(tokens, 4, device=DEVICE)
        post = torch.empty(tokens, 4, device=DEVICE)
        comb = torch.empty(tokens, 16, device=DEVICE)
        common = dict(hidden_size=hidden, rms_eps=1e-6, hc_eps=1e-5)
        common.update(n_splits=n_splits, num_tokens=tokens)
        residual.mhc_pre_only_hc4(gemm_mul, gemm_sqrsum, scale, base, pre, **common)
        residual.mhc_post_comb_hc4(
            gemm_mul, gemm_sqrsum, scale, base, post, comb, sinkhorn_iters=3, **common
        )
        return pre, post, comb

    expected = run(1)
    run(16)
    run(3)
    with assert_no_triton_compile(
        residual._mhc_pre_only_hc4_kernel, residual._mhc_post_comb_hc4_kernel
    ):
        # 112 is the split count of hidden_size 7168 on GFX950.
        for n_splits in (5, 7, 12, 64, 112):
            for got, want in zip(run(n_splits), expected, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_qrita_top_k_top_p_row_count():
    from tokenspeed_kernel.ops.sampling.triton import topk_topp

    vocab, pools = 1025, 4
    table = torch.tensor(
        topk_topp._QRITA_PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device=DEVICE
    )

    def run(rows):
        logits = torch.randn(rows, vocab, device=DEVICE) * 2.0
        # Top-k of one leaves only the argmax to sample.
        got = topk_topp.gumbel_sample_top_k_top_p_qrita_from_pools(
            logits,
            torch.arange(rows, dtype=torch.int32, device=DEVICE) % pools,
            torch.ones(pools, device=DEVICE),
            torch.ones(pools, dtype=torch.int32, device=DEVICE),
            torch.ones(pools, device=DEVICE),
            torch.arange(pools, dtype=torch.int64, device=DEVICE),
            torch.zeros(pools, dtype=torch.int64, device=DEVICE),
            torch.empty(pools, vocab, dtype=torch.float32, device=DEVICE),
            table,
            torch.empty(rows, dtype=torch.int32, device=DEVICE),
            num_programs=pools,
        )
        torch.testing.assert_close(got, logits.argmax(-1).int(), rtol=0, atol=0)

    for rows in (1, 16, 3):
        run(rows)
    with assert_no_triton_compile(topk_topp._top_k_top_p_qrita_gumbel_kernel):
        for rows in (5, 6, 7, 9, 12, 130, 131):
            run(rows)


def test_marlin_deepep_pack_global_token_count():
    from tokenspeed_kernel.ops.moe.marlin import deepep_layout

    experts, recv_m, hidden, top_k, block_m = 4, 24, 256, 2, 16
    counts = torch.tensor([5, 0, 17, 3], dtype=torch.int32, device=DEVICE)
    recv_x = torch.randn(experts, recv_m, hidden, device=DEVICE, dtype=torch.bfloat16)

    def run(tokens):
        packed, sorted_ids, _, _, offsets = deepep_layout.pack_recv_rows(
            recv_x, counts, tokens, top_k, block_m
        )
        capacity = deepep_layout.compact_row_capacity(
            tokens, top_k, experts, recv_m, block_m
        )
        assert packed.shape[0] == capacity
        for expert, count in enumerate(counts.tolist()):
            start = int(offsets[expert])
            live = packed[start : start + count]
            torch.testing.assert_close(live, recv_x[expert, :count], rtol=0, atol=0)
            padding = sorted_ids[start + count : start + -(-count // block_m) * block_m]
            assert (padding == capacity).all()
        return capacity

    run(13)
    with assert_no_triton_compile(
        deepep_layout._layout_kernel, deepep_layout._pack_kernel
    ):
        capacities = {run(tokens) for tokens in range(14, 41)}
    # The sweep must cross several capacities, each a compile while it was constexpr.
    assert len(capacities) >= 4
