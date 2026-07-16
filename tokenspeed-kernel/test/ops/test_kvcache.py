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
    index_k_block_split_scatter,
    transfer_kv_all_layer,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)


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


# index_k_block_split_scatter (GLM-5 DSA index-K cache write, 组C)


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
def test_index_k_block_split_scatter_matches_index_put(
    device: str, head_dim: int, group_size: int, tokens: int
) -> None:
    torch.manual_seed(head_dim + group_size + tokens)
    page_size, num_pages = 64, 32
    num_slots = num_pages * page_size
    ng = head_dim // group_size
    row = head_dim + ng * 4

    k_fp8 = torch.randn(tokens, head_dim, device=device).to(torch.float8_e4m3fn)
    k_scale = torch.rand(tokens, ng, device=device, dtype=torch.float32) + 0.1
    loc = torch.randperm(num_slots, device=device)[:tokens].to(torch.int64)
    page, slot = loc // page_size, loc % page_size

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
        page,
        slot,
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
    empty_idx = torch.empty(0, dtype=torch.int64, device=device)
    index_k_block_split_scatter(
        buf,
        empty_fp8,
        empty_scale,
        empty_idx,
        empty_idx,
        page_size=64,
        head_dim=128,
        group_size=128,
    )
    assert torch.count_nonzero(buf) == 0
