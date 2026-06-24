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
from tokenspeed.runtime.layers.attention.kv_cache.utils import (
    copy_all_layer_kv_cache_tiled,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.pdl import pdl_enabled
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)

GB = 1024 * 1024 * 1024


def _get_tensor_size_bytes(t: torch.Tensor | list[torch.Tensor]):
    if isinstance(t, list):
        return sum(_get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


class MLATokenToKVPool(BaseTokenToKVPool):
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

        self.memory_saver_adapter = memory_saver_adapter = (
            TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        )
        self.page_size_bytes = self._get_page_size_bytes()

        with memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            # The padded page 0 is used for writing dummy outputs from padded tokens.
            if self.quant_method == "per_token_head":
                self.kv_buffer = [
                    (
                        torch.zeros(
                            (self.size + self.page_size, 1, kv_lora_rank),
                            dtype=self.store_dtype,
                            device=device,
                        ),
                        torch.zeros(
                            (self.size + self.page_size, 1, 1),
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.zeros(
                            (self.size + self.page_size, 1, qk_rope_head_dim),
                            dtype=self.model_dtype,
                            device=device,
                        ),
                    )
                    for _ in range(layer_num)
                ]
            else:
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]

        # Calculate data pointers and strides for all buffers
        all_buffers = []
        if self.quant_method == "per_token_head":
            # kv_buffer is a list of tuples (k_lora_cache, k_scale_cache, k_rope_cache)
            for layer_buffers in self.kv_buffer:
                # Each layer has 3 tensors
                all_buffers.extend(layer_buffers)
        else:
            # kv_buffer is a list of single tensors
            all_buffers = self.kv_buffer

        self.data_ptrs = torch.tensor(
            [buf.data_ptr() for buf in all_buffers],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [np.prod(buf.shape[1:]) * buf.dtype.itemsize for buf in all_buffers],
            device=self.device,
        )

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream()
            if torch.cuda.is_available() and enable_alt_stream
            else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

    def _get_page_size_bytes(self):
        if self.quant_method == "per_token_head":
            dim_size_bytes = (
                self.kv_lora_rank * torch._utils._element_size(self.dtype)
                + self.qk_rope_head_dim * torch._utils._element_size(self.model_dtype)
                + 1 * torch._utils._element_size(torch.float32)
            )
        else:
            dim_size_bytes = (
                self.kv_lora_rank + self.qk_rope_head_dim
            ) * torch._utils._element_size(self.dtype)
        return self.page_size * self.layer_num * dim_size_bytes

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
        }

        dummy_loc = torch.zeros(1, dtype=torch.int32, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            1,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if self._kv_copy_config is None:
            # Native implementation for MLA
            if tgt_loc.numel() == 0:
                return

            tgt_loc_flat = tgt_loc.view(-1).long()
            src_loc_flat = src_loc.view(-1).long()

            if self.quant_method == "per_token_head":
                # kv_buffer is a list of tuples
                for layer_buffers in self.kv_buffer:
                    # Each layer has 3 tensors: k_lora_cache, k_scale_cache, k_rope_cache
                    for buf in layer_buffers:
                        buf[tgt_loc_flat] = buf[src_loc_flat]
            else:
                # kv_buffer is a list of single tensors
                for buf in self.kv_buffer:
                    buf[tgt_loc_flat] = buf[src_loc_flat]
        else:
            grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                tgt_loc.numel(),
                tgt_loc.numel(),
                BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
                num_warps=self._kv_copy_config["num_warps"],
                num_stages=2,
            )

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += _get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        if self.quant_method == "per_token_head":
            kv_data_ptrs = [
                sub_tuple[i].data_ptr()
                for i in range(3)
                for sub_tuple in self.kv_buffer
            ]
            kv_data_lens = [
                sub_tuple[i].nbytes for i in range(3) for sub_tuple in self.kv_buffer
            ]
            kv_item_lens = [
                sub_tuple[i][0].nbytes * self.page_size
                for i in range(3)
                for sub_tuple in self.kv_buffer
            ]
        else:
            # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
            kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
            kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
            kv_item_lens = [
                self.kv_buffer[i][0].nbytes * self.page_size
                for i in range(self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        if self.quant_method == "per_token_head":
            return [
                [start_idx + i * self.layer_num + layer_id for i in range(3)]
                for layer_id in range(self.layer_num)
            ]
        else:
            return [[start_idx + layer_id] for layer_id in range(self.layer_num)]

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        if self.quant_method == "per_token_head":
            return self.kv_buffer[layer_id]
        elif self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id].view(self.dtype)
        else:
            return self.kv_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        if self.quant_method == "per_token_head":
            return self.kv_buffer[layer_id][:2]
        elif self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        else:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank]

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
        layer_id = layer.layer_id
        if self.quant_method == "per_token_head":
            k_lora = cache_k[..., : self.kv_lora_rank].float()
            k_rope = cache_k[..., self.kv_lora_rank :].float()
            scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
            k_lora = (k_lora / scale).to(torch.float8_e4m3fn)
            k_rope = (k_rope / scale).to(self.model_dtype)
            self.kv_buffer[layer_id][0][loc] = k_lora.view(self.store_dtype)
            self.kv_buffer[layer_id][1][loc] = scale
            self.kv_buffer[layer_id][2][loc] = k_rope
        else:
            self.kv_buffer[layer_id][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if self.quant_method == "per_token_head":
            k_lora = cache_k_nope.float()
            k_rope = cache_k_rope.float()
            scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
            k_lora = (k_lora / scale).to(torch.float8_e4m3fn)
            k_rope = (k_rope / scale).to(self.model_dtype)
            self.kv_buffer[layer_id][0][loc] = k_lora.view(self.store_dtype)
            self.kv_buffer[layer_id][1][loc] = scale
            self.kv_buffer[layer_id][2][loc] = k_rope
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id],
                loc,
                cache_k_nope,
                cache_k_rope,
                enable_pdl=pdl_enabled(),
            )

    def get_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ):
        layer_id = layer.layer_id
        dst_dtype = dst_dtype or self.dtype

        if self.quant_method == "per_token_head":
            k_lora_cache, k_scale_cache, k_rope_cache = self.kv_buffer[layer_id]
            k_lora = k_lora_cache[loc].view(self.dtype).float()
            k_scale = k_scale_cache[loc]
            k_rope = k_rope_cache[loc].float()
            cache_k_nope = (k_lora * k_scale).to(dst_dtype).contiguous()
            cache_k_rope = (k_rope * k_scale).to(dst_dtype).contiguous()
            return cache_k_nope, cache_k_rope

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

    def get_cpu_copy(self, token_indices: list[int]) -> torch.Tensor:
        torch.cuda.synchronize()
        kv_cache_cpu = []
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(token_indices), self.offload_chunk_page_num):
                chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                if self.quant_method == "per_token_head":
                    kv_cache_cpu[-1].append(
                        [
                            buffer[chunk_indices].to("cpu", non_blocking=True)
                            for buffer in self.kv_buffer[layer_id]
                        ]
                    )
                else:
                    kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                        "cpu", non_blocking=True
                    )
                    kv_cache_cpu[-1].append([kv_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu: torch.Tensor, token_indices: list[int]
    ) -> None:
        torch.cuda.synchronize()
        for layer_id in range(self.layer_num):
            for i in range(0, len(token_indices), self.offload_chunk_page_num):
                chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                if self.quant_method == "per_token_head":
                    for j in range(3):
                        t = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][j]
                        assert t.shape[0] == len(chunk_indices)
                        self.kv_buffer[layer_id][j][chunk_indices] = t.to(
                            self.kv_buffer[0][0].device, non_blocking=True
                        )
                else:
                    kv_cpu = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][0]
                    assert kv_cpu.shape[0] == len(
                        chunk_indices
                    ), f"kv_cpu.shape[0] {kv_cpu.shape[0]} != len(chunk_indices) {len(chunk_indices)}"
                    kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                    self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()
