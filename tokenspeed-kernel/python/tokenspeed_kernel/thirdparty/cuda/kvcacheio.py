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

import functools
from pathlib import Path
from typing import List

import torch
from tokenspeed_kernel.platform import current_platform


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_kvcacheio_module():
    import tvm_ffi

    so_path = _objs_dir() / "kvcacheio" / "kvcacheio.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel kvcacheio library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


_is_amd = current_platform().is_amd


def _indices_to_host_list(indices: torch.Tensor) -> List[int]:
    indices_i64 = indices.to(torch.int64)
    if indices_i64.device.type != "cpu":
        indices_i64 = indices_i64.cpu()
    return indices_i64.tolist()


def _check_direct_copy_args(
    src_indices: torch.Tensor, dst_indices: torch.Tensor, page_size: int
) -> None:
    if src_indices.numel() != dst_indices.numel():
        raise ValueError("Source and destination indices must have the same length")
    if page_size <= 0:
        raise ValueError("Page size must be positive")
    if src_indices.numel() % page_size != 0:
        raise ValueError("Source indices size must be divisible by page size")


def _transfer_page_direct(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_page_index: int,
    dst_page_index: int,
    page_size: int,
) -> None:
    dst_buffer[dst_page_index : dst_page_index + page_size].copy_(
        src_buffer[src_page_index : src_page_index + page_size],
        non_blocking=True,
    )


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_per_layer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_per_layer_pf_lf(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_per_layer_ph_lf(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_all_layer(
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_all_layer_lf_pf(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_all_layer_lf_ph(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    if len(src_layers) != len(dst_layers):
        raise ValueError(
            "Source and destination layers must have the same number of layers"
        )
    _check_direct_copy_args(src_indices, dst_indices, page_size)

    src_indices_host = _indices_to_host_list(src_indices)
    dst_indices_host = _indices_to_host_list(dst_indices)

    start_index = 0
    end_index = 0
    num_indices = len(src_indices_host)

    for i in range(num_indices):
        if i < num_indices - 1:
            src_diff = src_indices_host[i + 1] - src_indices_host[i]
            dst_diff = dst_indices_host[i + 1] - dst_indices_host[i]
            if src_diff == 1 and dst_diff == 1:
                continue
            end_index = i + 1
        else:
            end_index = num_indices

        src_index = src_indices_host[start_index]
        dst_index = dst_indices_host[start_index]
        num_tokens = end_index - start_index

        for src_layer, dst_layer in zip(src_layers, dst_layers):
            _transfer_page_direct(
                src_layer, dst_layer, src_index, dst_index, num_tokens
            )

        start_index = end_index


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_per_layer_mla(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_per_layer_mla_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_all_layer_mla(
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_amd else 32,
):
    _load_kvcacheio_module().transfer_kv_all_layer_mla_lf_pf(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


__all__ = [
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
]
