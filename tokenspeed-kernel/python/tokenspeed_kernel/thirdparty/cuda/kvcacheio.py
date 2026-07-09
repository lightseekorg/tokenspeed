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
from dataclasses import dataclass
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


@functools.cache
def _load_transfer_kv_direct_func():
    try:
        return getattr(_load_kvcacheio_module(), "transfer_kv_direct_ptr_table")
    except (AttributeError, ModuleNotFoundError):
        return None
    except RuntimeError as exc:
        if "kvcacheio library not found" in str(exc):
            return None
        raise


@functools.cache
def _load_transfer_kv_direct_scatter_h2d_func():
    try:
        return getattr(
            _load_kvcacheio_module(),
            "transfer_kv_direct_ptr_table_scatter_h2d",
        )
    except (AttributeError, ModuleNotFoundError):
        return None
    except RuntimeError as exc:
        if "kvcacheio library not found" in str(exc):
            return None
        raise


@dataclass(frozen=True)
class DirectH2DScatterResult:
    used: bool
    buckets: int = 0
    kernel_launches: int = 0
    fallback_reason: str = ""


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


def _has_cuda_layer(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
) -> bool:
    return any(layer.device.type == "cuda" for layer in src_layers) or any(
        layer.device.type == "cuda" for layer in dst_layers
    )


def _leading_stride_bytes(tensor: torch.Tensor) -> int:
    if tensor.dim() < 1:
        raise ValueError("Direct KV transfer buffers must have at least one dimension")
    return int(tensor.stride(0) * tensor.element_size())


def _h2d_scatter_device(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
) -> tuple[torch.device | None, str]:
    if _is_amd:
        return None, "amd_fallback"
    if not src_layers:
        return None, "empty_layers"
    if any(layer.device.type != "cpu" for layer in src_layers):
        return None, "src_not_cpu"
    dst_device = dst_layers[0].device
    if dst_device.type != "cuda":
        return None, "dst_not_cuda"
    if any(layer.device != dst_device for layer in dst_layers):
        return None, "mixed_dst_devices"
    return dst_device, ""


def _to_i64_contiguous_on_device(
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    if indices.device != device:
        indices = indices.to(device, non_blocking=True)
    if not indices.is_contiguous():
        indices = indices.contiguous()
    return indices


def _record_cuda_metadata_stream(*tensors: torch.Tensor) -> None:
    cuda_tensors = [tensor for tensor in tensors if tensor.device.type == "cuda"]
    if not cuda_tensors:
        return
    stream = torch.cuda.current_stream(cuda_tensors[0].device)
    for tensor in cuda_tensors:
        tensor.record_stream(stream)


def _transfer_kv_direct_cpp(
    direct_func,
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> bool:
    buckets: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for src_layer, dst_layer in zip(src_layers, dst_layers):
        src_item_size = _leading_stride_bytes(src_layer)
        dst_item_size = _leading_stride_bytes(dst_layer)
        if src_item_size != dst_item_size:
            return False
        src_bucket, dst_bucket = buckets.setdefault(src_item_size, ([], []))
        src_bucket.append(src_layer)
        dst_bucket.append(dst_layer)

    platform = current_platform()
    for item_size, (bucket_src_layers, bucket_dst_layers) in buckets.items():
        src_ptrs = torch.tensor(
            [platform.device_visible_data_ptr(layer) for layer in bucket_src_layers],
            dtype=torch.uint64,
            device="cpu",
        )
        dst_ptrs = torch.tensor(
            [platform.device_visible_data_ptr(layer) for layer in bucket_dst_layers],
            dtype=torch.uint64,
            device="cpu",
        )
        direct_func(
            src_ptrs,
            dst_ptrs,
            src_indices,
            dst_indices,
            item_size,
            page_size,
        )
    return True


def transfer_kv_direct_h2d_scatter(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
    *,
    effective_copy_calls: int | None = None,
    min_copy_calls: int = 4096,
) -> DirectH2DScatterResult:
    """Try a pointer-table H2D scatter kernel for fragmented host loadback.

    Returns ``used=False`` when the transfer should fall back to
    ``transfer_kv_direct``. This helper is intentionally H2D-only: source
    layers must be CPU tensors already registered or pinned for GPU access, and
    destination layers must be CUDA tensors on a single device.
    """

    if len(src_layers) != len(dst_layers):
        raise ValueError(
            "Source and destination layers must have the same number of layers"
        )
    _check_direct_copy_args(src_indices, dst_indices, page_size)

    if effective_copy_calls is not None and effective_copy_calls < min_copy_calls:
        return DirectH2DScatterResult(
            used=False,
            fallback_reason="below_threshold",
        )

    scatter_func = _load_transfer_kv_direct_scatter_h2d_func()
    if scatter_func is None:
        return DirectH2DScatterResult(
            used=False,
            fallback_reason="symbol_missing",
        )

    device, reason = _h2d_scatter_device(src_layers, dst_layers)
    if device is None:
        return DirectH2DScatterResult(used=False, fallback_reason=reason)

    buckets: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for src_layer, dst_layer in zip(src_layers, dst_layers):
        src_item_size = _leading_stride_bytes(src_layer)
        dst_item_size = _leading_stride_bytes(dst_layer)
        if src_item_size != dst_item_size:
            return DirectH2DScatterResult(
                used=False,
                fallback_reason="item_size_mismatch",
                buckets=len(buckets),
            )
        src_bucket, dst_bucket = buckets.setdefault(src_item_size, ([], []))
        src_bucket.append(src_layer)
        dst_bucket.append(dst_layer)

    src_indices_device = _to_i64_contiguous_on_device(src_indices, device)
    dst_indices_device = _to_i64_contiguous_on_device(dst_indices, device)
    platform = current_platform()
    kernel_launches = 0
    for item_size, (bucket_src_layers, bucket_dst_layers) in buckets.items():
        src_ptrs = torch.tensor(
            [platform.device_visible_data_ptr(layer) for layer in bucket_src_layers],
            dtype=torch.uint64,
            device=device,
        )
        dst_ptrs = torch.tensor(
            [platform.device_visible_data_ptr(layer) for layer in bucket_dst_layers],
            dtype=torch.uint64,
            device=device,
        )
        scatter_func(
            src_ptrs,
            dst_ptrs,
            src_indices_device,
            dst_indices_device,
            item_size,
            page_size,
        )
        _record_cuda_metadata_stream(
            src_ptrs,
            dst_ptrs,
            src_indices_device,
            dst_indices_device,
        )
        kernel_launches += 1

    return DirectH2DScatterResult(
        used=True,
        buckets=len(buckets),
        kernel_launches=kernel_launches,
    )


def _transfer_kv_direct_python(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
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

    direct_func = (
        _load_transfer_kv_direct_func()
        if _has_cuda_layer(
            src_layers,
            dst_layers,
        )
        else None
    )
    if direct_func is not None:
        if _transfer_kv_direct_cpp(
            direct_func,
            src_layers,
            dst_layers,
            src_indices,
            dst_indices,
            page_size,
        ):
            return

    _transfer_kv_direct_python(
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        page_size,
    )


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
    "DirectH2DScatterResult",
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_direct_h2d_scatter",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
]
