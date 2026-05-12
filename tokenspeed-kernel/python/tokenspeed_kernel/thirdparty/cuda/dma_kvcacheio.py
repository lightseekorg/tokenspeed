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

"""DMA batched memcpy for KV cache transfers via ``cuMemcpyBatchAsync``.

One driver call dispatches all (K/V × layer × page) copies onto the
GPU's copy engines, so the transfer does not consume SMs or L2 and
cannot slow concurrent compute (allreduce, MoE bmm). Requires CUDA
12.8+ (the v13 entry point dropped the failIdx out-param; we go through
``cuda.bindings.driver`` so the binding picks the right ABI).
``cuMemcpyBatchAsync`` rejects the legacy NULL stream, so callers must
issue on an explicit stream.
"""

from __future__ import annotations

import numpy as np
import torch
from tokenspeed_kernel.platform import current_platform

_driver = None
_attrs_singleton = None
_resolve_attempted = False


def _try_resolve():
    global _driver, _attrs_singleton, _resolve_attempted
    if _resolve_attempted:
        return _driver
    _resolve_attempted = True
    if not current_platform().is_nvidia:
        return None
    try:
        from cuda.bindings import driver as drv

        attrs = drv.CUmemcpyAttributes()
        attrs.srcAccessOrder = (
            drv.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        )
        _driver = drv
        _attrs_singleton = attrs
    except Exception:
        _driver = None
    return _driver


def is_available() -> bool:
    """True iff ``cuMemcpyBatchAsync`` is resolvable on the current platform."""
    return _try_resolve() is not None


def batch_async_load_kv_pages(
    host_k_layers: list,
    host_v_layers: list,
    device_k_layers: list,
    device_v_layers: list,
    src_page_token_offsets: torch.Tensor,
    dst_page_token_offsets: torch.Tensor,
    page_bytes: int,
    token_stride_bytes: int,
    stream: torch.cuda.Stream,
) -> None:
    """Single ``cuMemcpyBatchAsync`` call for all (K/V × layer × page) blocks.

    ``*_page_token_offsets`` are 1-D CPU int tensors of per-block source /
    destination token offsets; each block copies ``page_bytes`` bytes at
    offset ``offset * token_stride_bytes`` from the per-layer base
    pointer. The layer-list arguments are read host-side via
    ``Tensor.data_ptr()`` — no syncs.
    """
    drv = _try_resolve()
    if drv is None:
        raise RuntimeError("cuMemcpyBatchAsync not available; install cuda-python")

    n_pages = int(src_page_token_offsets.numel())
    if n_pages == 0:
        return
    assert int(dst_page_token_offsets.numel()) == n_pages
    n_layers = len(host_k_layers)
    assert (
        n_layers == len(host_v_layers) == len(device_k_layers) == len(device_v_layers)
    )
    assert src_page_token_offsets.is_cpu and dst_page_token_offsets.is_cpu

    src_off_np = src_page_token_offsets.numpy().astype(np.uint64, copy=False)
    dst_off_np = dst_page_token_offsets.numpy().astype(np.uint64, copy=False)
    stride = np.uint64(token_stride_bytes)
    src_byte_off = src_off_np * stride
    dst_byte_off = dst_off_np * stride

    src_k_bases = np.fromiter(
        (t.data_ptr() for t in host_k_layers), dtype=np.uint64, count=n_layers
    )
    src_v_bases = np.fromiter(
        (t.data_ptr() for t in host_v_layers), dtype=np.uint64, count=n_layers
    )
    dst_k_bases = np.fromiter(
        (t.data_ptr() for t in device_k_layers), dtype=np.uint64, count=n_layers
    )
    dst_v_bases = np.fromiter(
        (t.data_ptr() for t in device_v_layers), dtype=np.uint64, count=n_layers
    )

    src_k = (src_k_bases[:, None] + src_byte_off[None, :]).ravel()
    src_v = (src_v_bases[:, None] + src_byte_off[None, :]).ravel()
    dst_k = (dst_k_bases[:, None] + dst_byte_off[None, :]).ravel()
    dst_v = (dst_v_bases[:, None] + dst_byte_off[None, :]).ravel()

    src_all = np.concatenate([src_k, src_v])
    dst_all = np.concatenate([dst_k, dst_v])
    sz_all = np.full(src_all.shape[0], page_bytes, dtype=np.uint64)
    total = int(src_all.shape[0])

    src_list = [drv.CUdeviceptr(int(p)) for p in src_all]
    dst_list = [drv.CUdeviceptr(int(p)) for p in dst_all]
    sz_list = [int(s) for s in sz_all]

    (err,) = drv.cuMemcpyBatchAsync(
        dst_list,
        src_list,
        sz_list,
        total,
        [_attrs_singleton],
        [0],
        1,
        stream.cuda_stream,
    )
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"cuMemcpyBatchAsync failed: err={err} "
            f"total={total} n_layers={n_layers} n_pages={n_pages}"
        )
