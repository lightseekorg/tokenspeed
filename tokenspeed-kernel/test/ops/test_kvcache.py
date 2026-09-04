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
from tokenspeed_kernel.ops.kvcache.triton import (
    copy_state_rows,
    fused_fp8_set_kv_buffer,
    index_k_block_split_scatter,
    state_verify_commit_rows,
    transfer_kv_all_layer,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)


@pytest.mark.parametrize("tokens", [1, 4, 32])
def test_fused_fp8_set_kv_buffer_matches_qsa_store(device: str, tokens: int) -> None:
    torch.manual_seed(tokens)
    page_size, num_slots = 16, 64
    # Match Qwen4-Exp TP4: Q/Gate precede one 256-wide K and V in the GEMM
    # output, so both inputs are strided views rather than contiguous tensors.
    qkv = torch.randn((tokens, 3584), device=device, dtype=torch.bfloat16)
    k = qkv[:, 3072:3328].view(tokens, 1, 256)
    v = qkv[:, 3328:3584].view(tokens, 1, 256)
    assert k.stride(-1) == v.stride(-1) == 1
    if tokens > 1:
        assert k.stride(0) == qkv.stride(0) == v.stride(0)
    cache_locs = torch.randperm(num_slots, device=device)[:tokens].to(torch.int32)
    k_cache = torch.zeros((num_slots, 1, 256), device=device, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)
    expected_k = torch.zeros_like(k_cache)
    expected_v = torch.zeros_like(v_cache)
    expected_k[cache_locs.to(torch.long)] = k.to(torch.float8_e4m3fn)
    expected_v[cache_locs.to(torch.long)] = v.to(torch.float8_e4m3fn)

    fused_fp8_set_kv_buffer(
        k,
        v,
        k_cache,
        v_cache,
        cache_locs,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    assert torch.equal(k_cache.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(v_cache.view(torch.uint8), expected_v.view(torch.uint8))


@pytest.mark.parametrize(
    "src_row_dtype,dst_row_dtype",
    [
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int64, torch.int64),
    ],
)
def test_copy_state_rows_accepts_32_and_64_bit_row_ids(
    device: str, src_row_dtype: torch.dtype, dst_row_dtype: torch.dtype
) -> None:
    num_layers = 2
    row_i32 = 5
    row_stride_i32 = 8
    rows_per_layer = 3
    src_slabs = [
        torch.arange(
            layer * 1_000,
            layer * 1_000 + 5 * row_stride_i32,
            device=device,
            dtype=torch.int32,
        ).reshape(5, row_stride_i32)
        for layer in range(num_layers)
    ]
    dst_slabs = [
        torch.full(
            (6, row_stride_i32),
            -1,
            device=device,
            dtype=torch.int32,
        )
        for _ in range(num_layers)
    ]
    src_rows = torch.tensor([4, -1, 1, 0, 3, 2], device=device, dtype=src_row_dtype)
    # Rows 1 and 4 carry a negative destination: the kernel must skip those
    # stores entirely, leaving dst rows 2 and 3 at their sentinel.
    dst_rows = torch.tensor([0, -1, 4, 1, -1, 5], device=device, dtype=dst_row_dtype)
    src_addresses = torch.tensor(
        [slab.data_ptr() for slab in src_slabs], device=device, dtype=torch.uint64
    )
    dst_addresses = torch.tensor(
        [slab.data_ptr() for slab in dst_slabs], device=device, dtype=torch.uint64
    )
    row_strides = torch.full(
        (num_layers,), row_stride_i32, device=device, dtype=torch.int64
    )

    copy_state_rows(
        src_addresses,
        dst_addresses,
        src_rows,
        dst_rows,
        row_bytes=row_i32 * 4,
        src_row_strides=row_strides,
        dst_row_strides=row_strides,
    )
    torch.cuda.synchronize()

    expected = [torch.full_like(slab, -1) for slab in dst_slabs]
    for layer in range(num_layers):
        for row in range(rows_per_layer):
            work_index = layer * rows_per_layer + row
            dst_row = int(dst_rows[work_index])
            src_row = int(src_rows[work_index])
            if dst_row < 0:
                continue
            if src_row < 0:
                expected[layer][dst_row, :row_i32] = 0
            else:
                expected[layer][dst_row, :row_i32] = src_slabs[layer][src_row, :row_i32]

    for actual, reference in zip(dst_slabs, expected, strict=True):
        assert torch.equal(actual, reference)


def test_copy_state_rows_masks_null_destination_pages(device: str) -> None:
    """A layer whose destination rows are all null must be left untouched.

    This is the contract PLE's batched post-verify commit depends on: cache
    page id 0 is the null page, the caller maps it to row -1, and the whole
    layer's slab must survive bit-identically while its peers in the same
    launch still commit.
    """

    num_layers = 3
    rows_per_layer = 4
    row_i32 = 6
    row_stride_i32 = 9
    null_layer = 1
    src_slabs = [
        torch.arange(
            layer * 100,
            layer * 100 + rows_per_layer * row_stride_i32,
            device=device,
            dtype=torch.int32,
        ).reshape(rows_per_layer, row_stride_i32)
        for layer in range(num_layers)
    ]
    dst_slabs = [
        torch.full(
            (rows_per_layer, row_stride_i32), -7, device=device, dtype=torch.int32
        )
        for _ in range(num_layers)
    ]
    before = [slab.clone() for slab in dst_slabs]
    # Every request of the null layer resolves page 0, i.e. destination -1.
    dst_rows = torch.tensor(
        [
            -1 if layer == null_layer else (row + 1) % rows_per_layer
            for layer in range(num_layers)
            for row in range(rows_per_layer)
        ],
        device=device,
        dtype=torch.int64,
    )
    src_rows = torch.arange(
        num_layers * rows_per_layer, device=device, dtype=torch.int64
    )
    src_rows = src_rows % rows_per_layer
    row_strides = torch.full(
        (num_layers,), row_stride_i32, device=device, dtype=torch.int64
    )

    copy_state_rows(
        torch.tensor(
            [slab.data_ptr() for slab in src_slabs], device=device, dtype=torch.uint64
        ),
        torch.tensor(
            [slab.data_ptr() for slab in dst_slabs], device=device, dtype=torch.uint64
        ),
        src_rows,
        dst_rows,
        row_bytes=row_i32 * 4,
        src_row_strides=row_strides,
        dst_row_strides=row_strides,
    )
    torch.cuda.synchronize()

    assert torch.equal(dst_slabs[null_layer], before[null_layer])
    for layer in range(num_layers):
        if layer == null_layer:
            continue
        expected = before[layer].clone()
        for row in range(rows_per_layer):
            expected[(row + 1) % rows_per_layer, :row_i32] = src_slabs[layer][
                row, :row_i32
            ]
        assert torch.equal(dst_slabs[layer], expected)


def test_copy_state_rows_commits_verified_state(device: str) -> None:
    batch_size, draft_tokens, num_layers = 4, 3, 3
    page_size, num_pages = 4, 48
    conv_words, ssm_words = 7, 1100
    scratch_rows = batch_size * (draft_tokens + 1)

    conv_scratch = [
        (
            torch.arange(
                scratch_rows * conv_words, device=device, dtype=torch.int32
            ).view(scratch_rows, conv_words)
            + layer * 100_000
        )
        for layer in range(num_layers)
    ]
    ssm_scratch = [
        (
            torch.arange(
                scratch_rows * ssm_words, device=device, dtype=torch.int32
            ).view(scratch_rows, ssm_words)
            + layer * 1_000_000
        )
        for layer in range(num_layers)
    ]
    conv_committed = [
        torch.full((num_pages, conv_words), -1, device=device, dtype=torch.int32)
        for _ in range(num_layers)
    ]
    ssm_committed = [
        torch.full((num_pages, ssm_words), -1, device=device, dtype=torch.int32)
        for _ in range(num_layers)
    ]

    def pointer_table(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(
            [tensor.data_ptr() for tensor in tensors],
            device=device,
            dtype=torch.uint64,
        )

    def stride_table(tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(
            [tensor.stride(0) for tensor in tensors],
            device=device,
            dtype=torch.int64,
        )

    tables = (
        torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            device=device,
            dtype=torch.int32,
        ),
        torch.tensor(
            [
                [21, 22, 23, 24],
                [25, 26, 27, 28],
                [29, 30, 31, 32],
                [33, 34, 35, 36],
            ],
            device=device,
            dtype=torch.int32,
        ),
    )
    group_sel = torch.tensor([0, 1, 0], device=device, dtype=torch.int64)
    committed = torch.tensor([3, 4, 7, 8], device=device, dtype=torch.int64)
    accepted = torch.tensor([1, 2, 3, 9], device=device, dtype=torch.int32)

    expected_conv = [tensor.clone() for tensor in conv_committed]
    expected_ssm = [tensor.clone() for tensor in ssm_committed]
    accepted_ref = accepted.to(torch.int64).clamp(1, draft_tokens)
    for layer in range(num_layers):
        table = tables[int(group_sel[layer])]
        for request in range(batch_size):
            src_row = request * (draft_tokens + 1) + int(accepted_ref[request])
            slot = (
                int(committed[request]) + int(accepted_ref[request]) - 1
            ) // page_size
            slot = min(max(slot, 0), table.shape[1] - 1)
            dst_row = max(int(table[request, slot]), 0)
            expected_conv[layer][dst_row] = conv_scratch[layer][src_row]
            expected_ssm[layer][dst_row] = ssm_scratch[layer][src_row]

    accepted_rows = accepted.clamp(1, draft_tokens)
    src_rows = (
        torch.arange(batch_size, device=device, dtype=torch.int32) * (draft_tokens + 1)
        + accepted_rows
    ).repeat(num_layers)
    slots = torch.div(
        committed + accepted_rows.to(torch.int64) - 1,
        page_size,
        rounding_mode="floor",
    ).clamp(0, tables[0].shape[1] - 1)
    destination_by_group = torch.stack(
        [table.gather(1, slots[:, None]).squeeze(1) for table in tables]
    )
    dst_rows = destination_by_group.index_select(0, group_sel).reshape(-1)

    copy_state_rows(
        pointer_table(conv_scratch),
        pointer_table(conv_committed),
        src_rows,
        dst_rows,
        row_bytes=conv_words * 4,
        src_row_strides=stride_table(conv_scratch),
        dst_row_strides=stride_table(conv_committed),
    )
    copy_state_rows(
        pointer_table(ssm_scratch),
        pointer_table(ssm_committed),
        src_rows,
        dst_rows,
        row_bytes=ssm_words * 4,
        src_row_strides=stride_table(ssm_scratch),
        dst_row_strides=stride_table(ssm_committed),
    )
    torch.cuda.synchronize()

    for actual, expected in zip(conv_committed, expected_conv):
        assert torch.equal(actual, expected)
    for actual, expected in zip(ssm_committed, expected_ssm):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("row_dtype", [torch.int32, torch.int64])
def test_state_verify_commit_rows_matches_torch(
    device: str, row_dtype: torch.dtype
) -> None:
    num_layers, batch_size, verify_width = 3, 4, 3
    # 9 exceeds verify_width and must clamp down; 0 must clamp up to 1.
    accepted = torch.tensor([1, 0, 3, 9], device=device, dtype=torch.int32)
    # Page 0 is the null page; -1 stands in for a pad slot a caller did not
    # clamp, and must map to -1 exactly like the null page does.
    pages = torch.tensor([5, 0, 7, -1], device=device, dtype=torch.int64)
    src_rows = torch.empty(num_layers * batch_size, device=device, dtype=row_dtype)
    dst_rows = torch.empty(num_layers * batch_size, device=device, dtype=row_dtype)

    state_verify_commit_rows(
        accepted,
        pages,
        src_rows,
        dst_rows,
        verify_width=verify_width,
        num_layers=num_layers,
    )
    torch.cuda.synchronize()

    clamped = accepted.to(torch.int64).clamp(1, verify_width)
    expected_src = (
        torch.arange(batch_size, device=device, dtype=torch.int64) * (verify_width + 1)
        + clamped
    ).repeat(num_layers)
    expected_dst = torch.where(
        pages > 0, pages.to(torch.int64), torch.full_like(pages, -1)
    ).repeat(num_layers)
    assert torch.equal(src_rows.to(torch.int64), expected_src)
    assert torch.equal(dst_rows.to(torch.int64), expected_dst)


def test_state_verify_commit_rows_single_layer_matches_tiled_prefix(
    device: str,
) -> None:
    """The shared-field launch passes num_layers=1 and slices the first block.

    A single-layer result must equal the first ``batch_size`` entries of the
    tiled one, otherwise the caller's ``rows[:bs]`` view addresses a different
    layer's tile.
    """

    num_layers, batch_size, verify_width = 4, 3, 2
    accepted = torch.tensor([1, 2, 5], device=device, dtype=torch.int32)
    pages = torch.tensor([2, 0, 9], device=device, dtype=torch.int64)
    tiled_src = torch.empty(num_layers * batch_size, device=device, dtype=torch.int64)
    tiled_dst = torch.empty(num_layers * batch_size, device=device, dtype=torch.int64)
    single_src = torch.empty(batch_size, device=device, dtype=torch.int64)
    single_dst = torch.empty(batch_size, device=device, dtype=torch.int64)

    state_verify_commit_rows(
        accepted,
        pages,
        tiled_src,
        tiled_dst,
        verify_width=verify_width,
        num_layers=num_layers,
    )
    state_verify_commit_rows(
        accepted,
        pages,
        single_src,
        single_dst,
        verify_width=verify_width,
        num_layers=1,
    )
    torch.cuda.synchronize()

    assert torch.equal(single_src, tiled_src[:batch_size])
    assert torch.equal(single_dst, tiled_dst[:batch_size])
    for layer in range(num_layers):
        tile = slice(layer * batch_size, (layer + 1) * batch_size)
        assert torch.equal(tiled_src[tile], single_src)
        assert torch.equal(tiled_dst[tile], single_dst)


def test_state_verify_commit_rows_rejects_bad_args(device: str) -> None:
    accepted = torch.tensor([1, 2], device=device, dtype=torch.int32)
    pages = torch.tensor([3, 4], device=device, dtype=torch.int64)
    src = torch.empty(2, device=device, dtype=torch.int64)
    dst = torch.empty(2, device=device, dtype=torch.int64)
    kwargs = {"verify_width": 2, "num_layers": 1}

    with pytest.raises(ValueError, match="one page id per request"):
        state_verify_commit_rows(accepted, pages[:1], src, dst, **kwargs)
    with pytest.raises(ValueError, match="num_layers \\* batch_size"):
        state_verify_commit_rows(accepted, pages, src[:1], dst, **kwargs)
    with pytest.raises(ValueError, match="num_layers \\* batch_size"):
        state_verify_commit_rows(accepted, pages, src, dst[:1], **kwargs)
    with pytest.raises(ValueError, match="verify_width"):
        state_verify_commit_rows(
            accepted, pages, src, dst, verify_width=0, num_layers=1
        )
    with pytest.raises(ValueError, match="num_layers"):
        state_verify_commit_rows(
            accepted, pages, src, dst, verify_width=2, num_layers=0
        )
    with pytest.raises(ValueError, match="torch.int32 or torch.int64"):
        state_verify_commit_rows(
            accepted,
            pages,
            torch.empty(2, device=device, dtype=torch.float32),
            dst,
            **kwargs,
        )


def test_state_verify_commit_rows_empty_batch_is_noop(device: str) -> None:
    accepted = torch.empty(0, device=device, dtype=torch.int32)
    pages = torch.empty(0, device=device, dtype=torch.int64)
    src = torch.empty(0, device=device, dtype=torch.int64)
    dst = torch.empty(0, device=device, dtype=torch.int64)

    state_verify_commit_rows(accepted, pages, src, dst, verify_width=2, num_layers=3)
    torch.cuda.synchronize()
    assert src.numel() == 0 and dst.numel() == 0


def test_transfer_kv_per_layer(device: str) -> None:
    num_slots = 6
    num_heads = 8
    head_dim = 128
    element_dim = num_heads * head_dim

    k_cache_dst = torch.zeros(
        num_slots, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v_cache_dst = torch.zeros_like(k_cache_dst)

    k_cache_src = torch.arange(
        num_slots * num_heads * head_dim,
        device=device,
        dtype=torch.float16,
    ).reshape(num_slots, num_heads, head_dim)
    v_cache_src = torch.arange(
        10_000,
        10_000 + num_slots * num_heads * head_dim,
        device=device,
        dtype=torch.float16,
    ).reshape(num_slots, num_heads, head_dim)

    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)

    expected_k = k_cache_dst.clone()
    expected_v = v_cache_dst.clone()
    expected_k[indices_dst.to(torch.int64)] = k_cache_src[indices_src.to(torch.int64)]
    expected_v[indices_dst.to(torch.int64)] = v_cache_src[indices_src.to(torch.int64)]

    transfer_kv_per_layer(
        src_k=k_cache_src,
        dst_k=k_cache_dst,
        src_v=v_cache_src,
        dst_v=v_cache_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=element_dim * k_cache_src.element_size(),
    )

    torch.cuda.synchronize()

    assert torch.equal(k_cache_dst, expected_k)
    assert torch.equal(v_cache_dst, expected_v)


def test_transfer_kv_all_layer(device: str) -> None:
    num_layers = 3
    num_slots = 6
    num_heads = 8
    head_dim = 128

    k_layers_dst = [
        torch.zeros(num_slots, num_heads, head_dim, device=device, dtype=torch.float16)
        for _ in range(num_layers)
    ]
    v_layers_dst = [torch.zeros_like(k_layers_dst[0]) for _ in range(num_layers)]
    k_layers_src = [
        torch.arange(
            layer_idx * num_slots * num_heads * head_dim,
            (layer_idx + 1) * num_slots * num_heads * head_dim,
            device=device,
            dtype=torch.float16,
        ).reshape(num_slots, num_heads, head_dim)
        for layer_idx in range(num_layers)
    ]
    v_layers_src = [
        torch.arange(
            20_000 + layer_idx * num_slots * num_heads * head_dim,
            20_000 + (layer_idx + 1) * num_slots * num_heads * head_dim,
            device=device,
            dtype=torch.float16,
        ).reshape(num_slots, num_heads, head_dim)
        for layer_idx in range(num_layers)
    ]

    k_ptr_dst = torch.tensor(
        [layer.data_ptr() for layer in k_layers_dst], device=device, dtype=torch.uint64
    )
    v_ptr_dst = torch.tensor(
        [layer.data_ptr() for layer in v_layers_dst], device=device, dtype=torch.uint64
    )
    k_ptr_src = torch.tensor(
        [layer.data_ptr() for layer in k_layers_src], device=device, dtype=torch.uint64
    )
    v_ptr_src = torch.tensor(
        [layer.data_ptr() for layer in v_layers_src], device=device, dtype=torch.uint64
    )
    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)
    slot_stride_bytes = k_layers_dst[0].stride(0) * k_layers_dst[0].element_size()

    expected_k = [layer.clone() for layer in k_layers_dst]
    expected_v = [layer.clone() for layer in v_layers_dst]
    for layer_idx in range(num_layers):
        expected_k[layer_idx][indices_dst.to(torch.int64)] = k_layers_src[layer_idx][
            indices_src.to(torch.int64)
        ]
        expected_v[layer_idx][indices_dst.to(torch.int64)] = v_layers_src[layer_idx][
            indices_src.to(torch.int64)
        ]

    transfer_kv_all_layer(
        src_k_layers=k_ptr_src,
        dst_k_layers=k_ptr_dst,
        src_v_layers=v_ptr_src,
        dst_v_layers=v_ptr_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=slot_stride_bytes,
        num_layers=num_layers,
    )

    torch.cuda.synchronize()

    for layer_idx in range(num_layers):
        assert torch.equal(k_layers_dst[layer_idx], expected_k[layer_idx])
        assert torch.equal(v_layers_dst[layer_idx], expected_v[layer_idx])


def test_transfer_kv_per_layer_mla(device: str) -> None:
    num_slots = 6
    kv_cache_dim = 576

    cache_dst = torch.zeros(
        num_slots, 1, kv_cache_dim, device=device, dtype=torch.float16
    )
    cache_src = torch.arange(
        num_slots * kv_cache_dim,
        device=device,
        dtype=torch.float16,
    ).reshape(num_slots, 1, kv_cache_dim)
    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)

    expected = cache_dst.clone()
    expected[indices_dst.to(torch.int64)] = cache_src[indices_src.to(torch.int64)]

    transfer_kv_per_layer_mla(
        src=cache_src,
        dst=cache_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=kv_cache_dim * cache_src.element_size(),
    )

    torch.cuda.synchronize()

    assert torch.equal(cache_dst, expected)


def test_transfer_kv_all_layer_mla(device: str) -> None:
    num_layers = 3
    num_slots = 6
    kv_cache_dim = 576

    layers_dst = [
        torch.zeros(num_slots, 1, kv_cache_dim, device=device, dtype=torch.float16)
        for _ in range(num_layers)
    ]
    layers_src = [
        torch.arange(
            layer_idx * num_slots * kv_cache_dim,
            (layer_idx + 1) * num_slots * kv_cache_dim,
            device=device,
            dtype=torch.float16,
        ).reshape(num_slots, 1, kv_cache_dim)
        for layer_idx in range(num_layers)
    ]
    ptr_dst = torch.tensor(
        [layer.data_ptr() for layer in layers_dst], device=device, dtype=torch.uint64
    )
    ptr_src = torch.tensor(
        [layer.data_ptr() for layer in layers_src], device=device, dtype=torch.uint64
    )
    indices_dst = torch.tensor([1, 4], device=device, dtype=torch.int32)
    indices_src = torch.tensor([0, 5], device=device, dtype=torch.int32)
    slot_stride_bytes = layers_dst[0].stride(0) * layers_dst[0].element_size()

    expected = [layer.clone() for layer in layers_dst]
    for layer_idx in range(num_layers):
        expected[layer_idx][indices_dst.to(torch.int64)] = layers_src[layer_idx][
            indices_src.to(torch.int64)
        ]

    transfer_kv_all_layer_mla(
        src_layers=ptr_src,
        dst_layers=ptr_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=slot_stride_bytes,
        num_layers=num_layers,
    )

    torch.cuda.synchronize()

    for layer_idx in range(num_layers):
        assert torch.equal(layers_dst[layer_idx], expected[layer_idx])


# index_k_block_split_scatter (GLM-5 DSA index-K cache write)


def _index_k_block_views(buf, num_pages, page_size, head_dim, num_groups):
    row = head_dim + num_groups * 4
    page_bytes = page_size * row
    flat = buf.reshape(-1)
    fp8_view = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, page_size, head_dim),
        (page_bytes, head_dim, 1),
    )
    scale_view = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, page_size, num_groups),
        (page_bytes // 4, num_groups, 1),
        (page_size * head_dim) // 4,
    )
    return fp8_view, scale_view


@pytest.mark.parametrize(
    "head_dim,group_size",
    [
        (128, 128),  # NG=1
        (128, 64),  # NG=2
        (256, 128),  # NG=2
        (384, 128),  # NG=3: non-power-of-2 head_dim and NG
        (384, 64),  # NG=6: non-power-of-2 NG
    ],
)
@pytest.mark.parametrize("tokens", [1, 7, 16, 64])
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_index_k_block_split_scatter_matches_index_put(
    device: str, head_dim: int, group_size: int, tokens: int, loc_dtype: torch.dtype
) -> None:
    torch.manual_seed(head_dim + group_size + tokens)
    page_size, num_pages = 64, 32
    num_slots = num_pages * page_size
    ng = head_dim // group_size
    row = head_dim + ng * 4

    k_fp8 = torch.randn(tokens, head_dim, device=device).to(torch.float8_e4m3fn)
    k_scale = torch.rand(tokens, ng, device=device, dtype=torch.float32) + 0.1
    loc = torch.randperm(num_slots, device=device)[:tokens].to(loc_dtype)
    page, slot = loc.long() // page_size, loc.long() % page_size

    buf_ref = torch.zeros(num_slots, row, dtype=torch.uint8, device=device)
    buf_k = torch.zeros(num_slots, row, dtype=torch.uint8, device=device)

    fp8_view, scale_view = _index_k_block_views(
        buf_ref, num_pages, page_size, head_dim, ng
    )
    fp8_view[page, slot] = k_fp8.view(-1, head_dim)
    scale_view[page, slot] = k_scale.view(-1, ng)

    index_k_block_split_scatter(
        buf_k,
        k_fp8,
        k_scale,
        loc,
        page_size=page_size,
        head_dim=head_dim,
        group_size=group_size,
    )
    torch.cuda.synchronize()
    assert torch.equal(buf_ref, buf_k)


def test_index_k_block_split_scatter_empty_is_noop(device: str) -> None:
    buf = torch.zeros(64, 132, dtype=torch.uint8, device=device)
    empty_fp8 = torch.empty(0, 128, device=device, dtype=torch.float8_e4m3fn)
    empty_scale = torch.empty(0, 1, device=device, dtype=torch.float32)
    empty_loc = torch.empty(0, dtype=torch.int64, device=device)
    index_k_block_split_scatter(
        buf,
        empty_fp8,
        empty_scale,
        empty_loc,
        page_size=64,
        head_dim=128,
        group_size=128,
    )
    assert torch.count_nonzero(buf) == 0
