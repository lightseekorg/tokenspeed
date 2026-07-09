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

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
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
