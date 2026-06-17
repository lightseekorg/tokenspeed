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

from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter
from tokenspeed.runtime.cache.transfer.types import CacheKind


class KVCachePool:
    kind = CacheKind.KV

    def __init__(
        self,
        device_pool,
        host_pool,
        io_backend: str,
        layer_num: int,
        draft_device_pool=None,
        draft_host_pool=None,
        draft_layer_num: int = 0,
    ):
        self.device_pool = device_pool
        self.host_pool = host_pool
        self.io_backend = io_backend
        self.layer_num = layer_num
        self.draft_device_pool = draft_device_pool
        self.draft_host_pool = draft_host_pool
        self.draft_layer_num = draft_layer_num
        self._counter = LayerDoneCounter(max(layer_num, draft_layer_num, 1))
        device_pool.register_layer_transfer_counter(self._counter)

    @property
    def device(self):
        return self.device_pool.device

    @property
    def host_layout(self) -> str:
        return self.host_pool.layout

    def page_size(self) -> int:
        return self.host_pool.page_size

    def num_layers(self) -> int:
        return max(self.layer_num, self.draft_layer_num)

    def supports_layerwise_loadback(self) -> bool:
        return True

    def get_layer_done_counter(self) -> LayerDoneCounter:
        return self._counter

    def local_layer_idx(self, global_layer_id: int) -> int:
        return global_layer_id

    def writeback(
        self,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        block_quota: int | None = None,
    ) -> None:
        import logging as _logging
        _log = _logging.getLogger(__name__)

        # Sync BEFORE base writeback to flush any residual async error
        # from previous draft writeback on this stream.
        torch.cuda.synchronize()

        # Compare host vs device pool strides to detect mismatch
        dev_pool = self.device_pool
        dev_inner = getattr(dev_pool, 'inner', dev_pool)
        dev_stride = getattr(dev_inner, 'data_strides', None)
        dev_stride_val = dev_stride[0].item() if dev_stride is not None and dev_stride.numel() > 0 else '?'
        host_stride = getattr(self.host_pool, 'token_stride_size', '?')
        dev_head_num = getattr(dev_inner, 'head_num', '?')
        dev_head_dim = getattr(dev_inner, 'head_dim', '?')
        dev_store_dtype = getattr(dev_inner, 'store_dtype', '?')
        dev_dtype = getattr(dev_inner, 'dtype', '?')
        # k_buffer[0] shape for device pool
        dev_k0_shape = '?'
        dev_k0_stride_bytes = '?'
        if hasattr(dev_inner, 'k_buffer') and len(dev_inner.k_buffer) > 0:
            dev_k0_shape = list(dev_inner.k_buffer[0].shape)
            dev_k0_stride_bytes = dev_inner.k_buffer[0].stride()[0] * dev_inner.k_buffer[0].element_size()

        # Collect host & device pointer info for alignment check
        host_k_ptrs = getattr(self.host_pool, 'k_data_ptrs', None)
        host_v_ptrs = getattr(self.host_pool, 'v_data_ptrs', None)
        dev_k_ptrs = getattr(dev_inner, 'k_data_ptrs', None)
        dev_v_ptrs = getattr(dev_inner, 'v_data_ptrs', None)

        # Re-compute V pointers from refs to detect corruption
        host_v_refs = getattr(self.host_pool, 'v_data_refs', None)
        host_k_refs = getattr(self.host_pool, 'k_data_refs', None)
        recomputed_v = '?'
        recomputed_k = '?'
        if host_v_refs is not None and len(host_v_refs) > 0:
            recomputed_v = [host_v_refs[0].data_ptr(), host_v_refs[-1].data_ptr()]
        if host_k_refs is not None and len(host_k_refs) > 0:
            recomputed_k = [host_k_refs[0].data_ptr(), host_k_refs[-1].data_ptr()]
        # Also get raw kv_buffer data_ptr
        host_kv_buffer = getattr(self.host_pool, 'kv_buffer', None)
        kv_buffer_ptr = '?'
        kv_buffer_shape = '?'
        if host_kv_buffer is not None:
            kv_buffer_ptr = hex(host_kv_buffer.data_ptr())
            kv_buffer_shape = list(host_kv_buffer.shape)

        def _ptr_info(ptrs, label):
            if ptrs is None:
                return f"{label}=None"
            vals = ptrs.tolist()
            aligns = [v % 4 for v in vals]
            nulls = [i for i, v in enumerate(vals) if v == 0]
            bad_align = [i for i, a in enumerate(aligns) if a != 0]
            return (
                f"{label}: count={len(vals)} "
                f"first=0x{vals[0]:x} last=0x{vals[-1]:x} "
                f"null_layers={nulls} bad_align_layers={bad_align}"
            )

        _log.warning(
            "[DEBUG-KVStore] pre-sync OK. Starting base writeback. "
            "host_stride=%s dev_stride=%s MATCH=%s "
            "host(head_num=%s head_dim=%s dtype=%s) "
            "dev(head_num=%s head_dim=%s dtype=%s store_dtype=%s "
            "k_buf0_shape=%s k_buf0_stride_bytes=%s) "
            "layer_num(host=%s dev=%s) "
            "size(host=%s dev=%s) "
            "src_indices.numel=%d dst_indices.numel=%d "
            "src_max=%s dst_max=%s "
            "| %s | %s | %s | %s "
            "| kv_buffer(ptr=%s shape=%s) "
            "recomputed_k=[0x%x,0x%x] recomputed_v=[0x%x,0x%x]",
            host_stride, dev_stride_val,
            host_stride == dev_stride_val if isinstance(dev_stride_val, int) else '?',
            getattr(self.host_pool, 'head_num', '?'),
            getattr(self.host_pool, 'head_dim', '?'),
            getattr(self.host_pool, 'dtype', '?'),
            dev_head_num, dev_head_dim, dev_dtype, dev_store_dtype,
            dev_k0_shape, dev_k0_stride_bytes,
            getattr(self.host_pool, 'layer_num', '?'),
            getattr(dev_inner, 'layer_num', '?'),
            getattr(self.host_pool, 'size', '?'),
            getattr(dev_inner, 'size', '?'),
            src_indices.numel(), dst_indices.numel(),
            src_indices.max().item() if src_indices.numel() > 0 else -1,
            dst_indices.max().item() if dst_indices.numel() > 0 else -1,
            _ptr_info(host_k_ptrs, 'host_k'),
            _ptr_info(host_v_ptrs, 'host_v'),
            _ptr_info(dev_k_ptrs, 'dev_k'),
            _ptr_info(dev_v_ptrs, 'dev_v'),
            kv_buffer_ptr, kv_buffer_shape,
            recomputed_k[0] if isinstance(recomputed_k, list) else 0,
            recomputed_k[1] if isinstance(recomputed_k, list) else 0,
            recomputed_v[0] if isinstance(recomputed_v, list) else 0,
            recomputed_v[1] if isinstance(recomputed_v, list) else 0,
        )

        # WORKAROUND: rebuild k/v_data_ptrs before writeback to recover from
        # GPU memory corruption (the small 120B tensors get overwritten by
        # other GPU operations).
        def _rebuild_ptrs(pool):
            if pool is None:
                return
            k_refs = getattr(pool, 'k_data_refs', None)
            v_refs = getattr(pool, 'v_data_refs', None)
            if k_refs is None or v_refs is None:
                return
            from tokenspeed_kernel.platform import current_platform
            _platform = current_platform()
            dev = pool.k_data_ptrs.device
            pool.k_data_ptrs = torch.tensor(
                [_platform.device_visible_data_ptr(x) for x in k_refs],
                dtype=torch.uint64, device=dev,
            )
            pool.v_data_ptrs = torch.tensor(
                [_platform.device_visible_data_ptr(x) for x in v_refs],
                dtype=torch.uint64, device=dev,
            )

        _rebuild_ptrs(self.host_pool)
        _rebuild_ptrs(self.draft_host_pool)
        torch.cuda.synchronize()
        _log.warning(
            "[DEBUG-KVStore] rebuilt host_v=0x%x..0x%x",
            self.host_pool.v_data_ptrs[0].item(),
            self.host_pool.v_data_ptrs[-1].item(),
        )

        self.host_pool.backup_from_device_all_layer(
            self.device_pool,
            dst_indices,
            src_indices,
            self.io_backend,
            block_quota=block_quota,
        )
        torch.cuda.synchronize()
        _log.warning("[DEBUG-KVStore] base writeback OK.")

        if self.draft_host_pool is not None:
            _log.warning(
                "[DEBUG-KVStore] Starting draft writeback. "
                "draft_host_pool=%s draft_device_pool=%s "
                "layer_num=%s token_stride_size=%s "
                "head_num=%s head_dim=%s dtype=%s "
                "draft_device_pool.layer_num=%s "
                "draft_host_pool.size=%s draft_device_pool.size=%s",
                type(self.draft_host_pool).__name__,
                type(self.draft_device_pool).__name__,
                getattr(self.draft_host_pool, 'layer_num', '?'),
                getattr(self.draft_host_pool, 'token_stride_size', '?'),
                getattr(self.draft_host_pool, 'head_num', '?'),
                getattr(self.draft_host_pool, 'head_dim', '?'),
                getattr(self.draft_host_pool, 'dtype', '?'),
                getattr(self.draft_device_pool, 'layer_num', '?'),
                getattr(self.draft_host_pool, 'size', '?'),
                getattr(self.draft_device_pool, 'size', '?'),
            )
            self.draft_host_pool.backup_from_device_all_layer(
                self.draft_device_pool,
                dst_indices,
                src_indices,
                self.io_backend,
                block_quota=block_quota,
            )
            torch.cuda.synchronize()
            _log.warning("[DEBUG-KVStore] draft writeback OK.")

    def loadback(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor, layer_idx: int
    ) -> None:
        if layer_idx < self.layer_num:
            self.host_pool.load_to_device_per_layer(
                self.device_pool,
                src_indices,
                dst_indices,
                layer_idx,
                self.io_backend,
            )
        if self.draft_host_pool is not None and layer_idx < self.draft_layer_num:
            self.draft_host_pool.load_to_device_per_layer(
                self.draft_device_pool,
                src_indices,
                dst_indices,
                layer_idx,
                self.io_backend,
            )

    def alloc_host(self, n: int):
        return self.host_pool.alloc(n)

    def free_host(self, indices: torch.Tensor) -> None:
        self.host_pool.free(indices)

    def host_available(self) -> int:
        return self.host_pool.available_size()
