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
from bisect import bisect_left
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
        return getattr(_load_kvcacheio_module(), "transfer_kv_direct")
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
class _DirectH2DScatterResult:
    used: bool
    buckets: int = 0
    kernel_launches: int = 0
    fallback_reason: str = ""


@dataclass(frozen=True)
class _DirectH2DScatterBucket:
    item_size: int
    entry_ids: tuple[int, ...]
    src_ptrs_host: torch.Tensor
    dst_ptrs_host: torch.Tensor
    src_ptrs_device: torch.Tensor
    dst_ptrs_device: torch.Tensor


@dataclass(frozen=True)
class DirectH2DScatterPlan:
    """Immutable H2D pointer tables prepared for one device stream."""

    device: torch.device
    stream_id: int
    buckets: tuple[_DirectH2DScatterBucket, ...]


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


def _has_cuda_layer(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
) -> bool:
    return any(
        layer.device.type == "cuda"
        for layers in (src_layers, dst_layers)
        for layer in layers
    )


def _leading_stride_bytes(tensor: torch.Tensor) -> int:
    if tensor.dim() < 1:
        raise ValueError("Direct KV transfer buffers must have at least one dimension")
    return int(tensor.stride(0) * tensor.element_size())


def _group_direct_layers(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    entry_ids,
):
    grouped: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for entry_id, src, dst in zip(entry_ids, src_layers, dst_layers):
        item_size = _leading_stride_bytes(src)
        if item_size != _leading_stride_bytes(dst):
            return None
        grouped.setdefault(item_size, []).append((int(entry_id), src, dst))
    return grouped


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


def _current_stream_id(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.current_stream(device).cuda_stream)


def _pinned_pointer_tensor(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=torch.uint64,
        device="cpu",
        pin_memory=device.type == "cuda" and torch.cuda.is_available(),
    )


def _scatter_fallback(reason: str) -> _DirectH2DScatterResult:
    return _DirectH2DScatterResult(False, fallback_reason=reason)


def prepare_kv_direct_h2d_scatter_plan(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    entry_ids: List[int],
) -> tuple[DirectH2DScatterPlan | None, str]:
    """Prepare stream-affine pointer tables for repeated H2D page scatters.

    Args:
        src_layers: GPU-visible CPU tensors whose leading dimension indexes
            complete cache-page records.
        dst_layers: CUDA tensors paired with ``src_layers``. All tensors must
            reside on one device, and each pair must have the same leading
            stride in bytes.
        entry_ids: Logical entry, typically layer, for every tensor pair. A
            prepared launch selects entries using a half-open ID range.

    Returns:
        A ``(plan, fallback_reason)`` pair. On success, ``plan`` contains
        pointer tables bucketed by page-record size and ``fallback_reason`` is
        empty. If scatter is unavailable or incompatible, ``plan`` is ``None``
        and ``fallback_reason`` describes why the caller should use direct
        copies.

    Raises:
        ValueError: If the three input sequences have different lengths.

    The plan captures tensor addresses, device, and current CUDA stream.
    Callers must rebuild it after any of those properties change.
    """

    if len(src_layers) != len(dst_layers) or len(src_layers) != len(entry_ids):
        raise ValueError(
            "Source layers, destination layers, and entry ids must have equal lengths"
        )
    if _load_transfer_kv_direct_scatter_h2d_func() is None:
        return None, "symbol_missing"

    device, reason = _h2d_scatter_device(src_layers, dst_layers)
    if device is None:
        return None, reason

    grouped = _group_direct_layers(src_layers, dst_layers, entry_ids)
    if grouped is None:
        return None, "item_size_mismatch"

    platform = current_platform()
    buckets: list[_DirectH2DScatterBucket] = []
    for item_size, entries in grouped.items():
        entries.sort(key=lambda entry: entry[0])
        bucket_entry_ids = tuple(entry_id for entry_id, _, _ in entries)
        src_ptrs_host = _pinned_pointer_tensor(
            [platform.device_visible_data_ptr(src) for _, src, _ in entries],
            device,
        )
        dst_ptrs_host = _pinned_pointer_tensor(
            [platform.device_visible_data_ptr(dst) for _, _, dst in entries],
            device,
        )
        buckets.append(
            _DirectH2DScatterBucket(
                item_size=item_size,
                entry_ids=bucket_entry_ids,
                src_ptrs_host=src_ptrs_host,
                dst_ptrs_host=dst_ptrs_host,
                src_ptrs_device=src_ptrs_host.to(device, non_blocking=True),
                dst_ptrs_device=dst_ptrs_host.to(device, non_blocking=True),
            )
        )

    return (
        DirectH2DScatterPlan(
            device=device,
            stream_id=_current_stream_id(device),
            buckets=tuple(buckets),
        ),
        "",
    )


def transfer_kv_direct_h2d_scatter_prepared(
    plan: DirectH2DScatterPlan,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    entry_begin: int,
    entry_end: int,
) -> _DirectH2DScatterResult:
    """Scatter complete cache-page records using a prepared H2D plan.

    Args:
        plan: Pointer tables returned by
            :func:`prepare_kv_direct_h2d_scatter_plan`.
        src_indices: Source physical-page IDs. Each index addresses one
            complete leading-dimension record in every selected source tensor.
        dst_indices: Destination physical-page IDs paired one-to-one with
            ``src_indices``.
        entry_begin: Inclusive logical entry ID to launch.
        entry_end: Exclusive logical entry ID to launch.

    Returns:
        A result with ``used=True`` when the request was handled by the scatter
        path, together with the number of selected size buckets and kernel
        launches. If the prepared stream no longer matches or the native
        symbol is unavailable, ``used`` is false and ``fallback_reason`` tells
        the caller to use direct copies.

    Raises:
        ValueError: If source and destination index counts differ.

    The launch must run on the CUDA stream captured by ``plan``. Index
    conversions, if needed, are submitted on that stream.
    """

    if src_indices.numel() != dst_indices.numel():
        raise ValueError("Source and destination indices must have the same length")
    if _current_stream_id(plan.device) != plan.stream_id:
        return _scatter_fallback("stream_mismatch")

    scatter_func = _load_transfer_kv_direct_scatter_h2d_func()
    if scatter_func is None:
        return _scatter_fallback("symbol_missing")

    src_indices_device = _to_i64_contiguous_on_device(src_indices, plan.device)
    dst_indices_device = _to_i64_contiguous_on_device(dst_indices, plan.device)
    kernel_launches = 0
    for bucket in plan.buckets:
        first = bisect_left(bucket.entry_ids, entry_begin)
        last = bisect_left(bucket.entry_ids, entry_end)
        if first == last:
            continue
        src_ptrs = bucket.src_ptrs_device[first:last]
        dst_ptrs = bucket.dst_ptrs_device[first:last]
        scatter_func(
            src_ptrs,
            dst_ptrs,
            src_indices_device,
            dst_indices_device,
            bucket.item_size,
        )
        _record_cuda_metadata_stream(
            src_ptrs,
            dst_ptrs,
            src_indices_device,
            dst_indices_device,
        )
        kernel_launches += 1

    return _DirectH2DScatterResult(True, kernel_launches, kernel_launches)


def _transfer_kv_direct_cpp(
    direct_func,
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> bool:
    buckets = _group_direct_layers(src_layers, dst_layers, range(len(src_layers)))
    if buckets is None:
        return False

    platform = current_platform()
    for item_size, entries in buckets.items():
        src_ptrs = _pinned_pointer_tensor(
            [platform.device_visible_data_ptr(src) for _, src, _ in entries],
            torch.device("cpu"),
        )
        dst_ptrs = _pinned_pointer_tensor(
            [platform.device_visible_data_ptr(dst) for _, _, dst in entries],
            torch.device("cpu"),
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
    num_indices = len(src_indices_host)

    for end_index in range(1, num_indices + 1):
        if (
            end_index < num_indices
            and src_indices_host[end_index] - src_indices_host[end_index - 1] == 1
            and dst_indices_host[end_index] - dst_indices_host[end_index - 1] == 1
        ):
            continue

        src_index = src_indices_host[start_index]
        dst_index = dst_indices_host[start_index]
        num_tokens = end_index - start_index

        for src_layer, dst_layer in zip(src_layers, dst_layers):
            dst_layer[dst_index : dst_index + num_tokens].copy_(
                src_layer[src_index : src_index + num_tokens],
                non_blocking=True,
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
    "DirectH2DScatterPlan",
    "prepare_kv_direct_h2d_scatter_plan",
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_direct_h2d_scatter_prepared",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
]
