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
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
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


class MLATokenToKVPool(CachePool):
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
        page_size: int,
        rank: int,
        *,
        memory_plan: CacheMemoryPlan,
        paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...] = (),
        token_capacity: int | None = None,
        layer_group_ids: tuple[str, ...] = (),
        pd_disaggregation_enabled: bool = False,
    ):
        super().__init__(
            size,
            dtype,
            device,
            page_size,
            rank,
            memory_plan,
            paged_cache_group_specs=paged_cache_group_specs,
            token_capacity=token_capacity,
            pd_disaggregation_enabled=pd_disaggregation_enabled,
        )
        self.model_dtype = model_dtype
        self.quant_method = quant_method

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        # Physical group id per layer, from the cache recipe
        # (CachePoolSpec.layer_group_ids) — the single source the scheduler
        # groups are published from.
        self.layer_cache_group_ids = tuple(layer_group_ids)
        if len(self.layer_cache_group_ids) != layer_num:
            raise ValueError(
                f"layer_group_ids has {len(self.layer_cache_group_ids)} "
                f"entries but the pool has {layer_num} layers; the cache "
                "recipe must supply one group id per layer "
                "(CachePoolSpec.layer_group_ids)"
            )
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self._create_buffers()

    def _create_buffers(self) -> None:
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            # The padded page 0 is used for writing dummy outputs from padded tokens.
            if self.quant_method == "per_token_head":
                self.kv_buffer = [
                    (
                        self.field(
                            f"layer.{layer_id}.latent_kv", self.store_dtype
                        ).view(-1, 1, self.kv_lora_rank),
                        self.field(
                            f"layer.{layer_id}.latent_scale", torch.float32
                        ).view(-1, 1, 1),
                        self.field(f"layer.{layer_id}.rope_k", self.model_dtype).view(
                            -1, 1, self.qk_rope_head_dim
                        ),
                    )
                    for layer_id in range(self.layer_num)
                ]
            else:
                self.kv_buffer = [
                    self.field(f"layer.{layer_id}.latent_kv", self.store_dtype).view(
                        -1, 1, self.kv_cache_dim
                    )
                    for layer_id in range(self.layer_num)
                ]

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
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        buffer = self.kv_buffer[layer_id]
        if buffer is None:
            raise ValueError(f"layer {layer_id} is a KDA state layer")
        if self.quant_method == "per_token_head":
            return buffer
        elif self.store_dtype != self.dtype:
            return buffer.view(self.dtype)
        else:
            return buffer

    def get_value_buffer(self, layer_id: int):
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        buffer = self.kv_buffer[layer_id]
        if buffer is None:
            raise ValueError(f"layer {layer_id} is a KDA state layer")
        if self.quant_method == "per_token_head":
            return buffer[:2]
        elif self.store_dtype != self.dtype:
            return buffer[..., : self.kv_lora_rank].view(self.dtype)
        else:
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
        sanitize: bool = False,
    ):
        layer_id = layer.layer_id
        if self.quant_method == "per_token_head":
            # Preserve the writer's sanitization contract for the quantized
            # fallback. The BF16 path below folds this work into Triton.
            if sanitize:
                cache_k_nope = torch.nan_to_num(cache_k_nope)
                cache_k_rope = torch.nan_to_num(cache_k_rope)
            k_lora = cache_k_nope.float()
            k_rope = cache_k_rope.float()
            scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
            k_lora = (k_lora / scale).to(torch.float8_e4m3fn)
            k_rope = (k_rope / scale).to(self.model_dtype)
            self.kv_buffer[layer_id][0][loc] = k_lora.view(self.store_dtype)
            self.kv_buffer[layer_id][1][loc] = scale
            self.kv_buffer[layer_id][2][loc] = k_rope
        else:
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
