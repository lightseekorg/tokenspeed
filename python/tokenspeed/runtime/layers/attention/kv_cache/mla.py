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

import numpy as np
import torch

from tokenspeed.runtime.cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.pdl import pdl_enabled
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


class MLATokenToKVPool(BaseTokenToKVPool):
    """Compute interface for MLA latent KV pools.

    Holds the MLA read/write kernels (``set_mla_kv_buffer`` /
    ``get_mla_kv_buffer`` and the key/value buffer views) shared by every MLA
    pool. Storage is owned by subclasses: all MLA models run on the shared LCM
    arena (:class:`LcmMLATokenToKVPool`), so this class does not allocate its
    own per-layer tensors and is never instantiated directly. ``_create_buffers``
    is the storage hook a subclass must implement.
    """

    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        quant_method: str,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        enable_kv_cache_copy: bool = False,
        enable_alt_stream: bool = True,
        max_scheduled_tokens: int = 0,
    ):
        super().__init__(
            size, dtype, device, max_batch_size, max_context_len, page_size, rank
        )
        self.model_dtype = model_dtype
        self.quant_method = quant_method

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self._create_buffers()
        self._compute_buffer_data_ptrs()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream()
            if torch.cuda.is_available() and enable_alt_stream
            else None
        )
        self._kv_copy_config = None

    def _create_buffers(self) -> None:
        """Allocate ``self.kv_buffer``; implemented by the storage subclass."""
        raise NotImplementedError(
            "MLATokenToKVPool is a compute base; a storage subclass "
            "(LcmMLATokenToKVPool) must implement _create_buffers"
        )

    def _compute_buffer_data_ptrs(self) -> None:
        """Cache device pointers and per-token strides for every KV buffer.

        Drives the tiled KV copy kernel. State-layer slots (``None``) are
        skipped.
        """
        all_buffers = [buffer for buffer in self.kv_buffer if buffer is not None]
        self.data_ptrs = torch.tensor(
            [buf.data_ptr() for buf in all_buffers],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [np.prod(buf.shape[1:]) * buf.dtype.itemsize for buf in all_buffers],
            device=self.device,
        )

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        buffer = self.kv_buffer[layer_id]
        if buffer is None:
            raise ValueError(f"layer {layer_id} is a KDA state layer")
        if self.store_dtype != self.dtype:
            return buffer.view(self.dtype)
        return buffer

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        buffer = self.kv_buffer[layer_id]
        if buffer is None:
            raise ValueError(f"layer {layer_id} is a KDA state layer")
        if self.store_dtype != self.dtype:
            return buffer[..., : self.kv_lora_rank].view(self.dtype)
        return buffer[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        self.kv_buffer[layer.layer_id][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        sanitize: bool = False,
    ):
        layer_id = layer.layer_id
        if self.store_dtype != self.dtype:
            # Bitwise-viewed pool: pre-cast and re-view for the raw word copy.
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)
        # else: the write kernel casts to the buffer dtype on store.

        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id],
            loc,
            cache_k_nope,
            cache_k_rope,
            enable_pdl=pdl_enabled(),
            sanitize=sanitize,
        )

    def get_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ):
        layer_id = layer.layer_id
        dst_dtype = dst_dtype or self.dtype

        kv_buffer = self.get_key_buffer(layer_id)
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(
            kv_buffer, loc, cache_k_nope, cache_k_rope, enable_pdl=pdl_enabled()
        )
        return cache_k_nope, cache_k_rope
