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

"""DSA + KDA cache views for GLM-5.3-Flash over one recipe-planned arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from typing_extensions import override

from tokenspeed.runtime.cache.utils import set_mla_kv_buffer_triton
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
    Glm53FlashPoolOptions,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils.pdl import pdl_enabled


@dataclass
class _KPoolTailWorkspace:
    options: Glm53FlashPoolOptions
    storage: torch.Tensor
    row_by_layer: dict[int, int]


class HybridGlm53FlashTokenToKVPool(HybridKDATokenToKVPool):
    """KDA/DSA pages with pooled-index fields and a request-local KPool tail."""

    def __init__(self, *args, pool_options: Glm53FlashPoolOptions, **kwargs):
        self.index_head_dim = pool_options.index_head_dim
        super().__init__(*args, **kwargs)
        self._kpool_tail_workspace = self._bind_kpool_tail_workspace(pool_options)

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **HybridKDATokenToKVPool.layer_plane_bindings,
        "index_k": "_index_k",
    }

    def _bind_kpool_tail_workspace(
        self, options: Glm53FlashPoolOptions
    ) -> _KPoolTailWorkspace:
        """Allocate the model-private ring once and share it with draft views."""
        attribute = "_glm53_flash_kpool_tail_workspace"
        workspace = getattr(self.arena, attribute, None)
        if workspace is None:
            with self.arena.memory_saver_adapter.region(
                tag="kv_cache", enable_cpu_backup=False
            ):
                storage = torch.zeros(
                    (
                        len(options.dsa_layer_ids),
                        2,
                        options.num_request_slots,
                        options.tail_width,
                        options.index_head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.arena.device,
                )
            workspace = _KPoolTailWorkspace(
                options=options,
                storage=storage,
                row_by_layer={
                    layer_id: row for row, layer_id in enumerate(options.dsa_layer_ids)
                },
            )
            setattr(self.arena, attribute, workspace)
        elif workspace.options != options:
            raise ValueError(
                "GLM-5.3-Flash cache views disagree on KPool tail geometry"
            )
        return workspace

    def has_index_k_buffer(self) -> bool:
        return any(buffer is not None for buffer in getattr(self, "_index_k", ()))

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        try:
            buffer = self._index_k[layer_id]
        except (AttributeError, IndexError) as exc:
            raise ValueError(f"layer {layer_id} has no DSA index cache") from exc
        if buffer is None:
            raise ValueError(f"layer {layer_id} has no DSA index cache")
        return buffer

    def get_kpool_buffers(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_k = self.get_index_k_buffer(layer_id)
        try:
            row = self._kpool_tail_workspace.row_by_layer[
                self._field_layer_id(layer_id)
            ]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no KPool tail cache") from exc
        storage = self._kpool_tail_workspace.storage
        return index_k, storage[row, 0], storage[row, 1]

    def kpool_tail_workspace_bytes(self) -> int:
        return self._kpool_tail_workspace.storage.nbytes

    @torch.no_grad()
    @override
    def clear_kv_buffers(self) -> None:
        super().clear_kv_buffers()
        self._kpool_tail_workspace.storage.zero_()

    def index_k_block_views(
        self, buf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        page_size = buf.shape[1]
        num_groups = self.index_head_dim // 128
        scale_bytes = torch._utils._element_size(torch.float32)
        page_stride = buf.stride(0)
        values = torch.as_strided(
            buf,
            (buf.shape[0], page_size, self.index_head_dim),
            (page_stride, self.index_head_dim, 1),
        ).view(torch.float8_e4m3fn)
        scales = torch.as_strided(
            buf,
            (buf.shape[0], page_size, num_groups * scale_bytes),
            (page_stride, num_groups * scale_bytes, 1),
            buf.storage_offset() + page_size * self.index_head_dim,
        ).view(torch.float32)
        return values, scales

    @override
    def set_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        sanitize: bool = False,
    ) -> None:
        if self.qk_rope_head_dim != 0:
            super().set_mla_kv_buffer(
                layer,
                loc,
                cache_k_nope,
                cache_k_rope,
                sanitize=sanitize,
            )
            return

        kv_buffer = self.kv_buffer[layer.layer_id]
        if sanitize:
            float_maxes = [
                torch.finfo(tensor.dtype).max
                for tensor in (cache_k_nope, self.get_key_buffer(layer.layer_id))
                if tensor.dtype.is_floating_point
            ]
            max_finite = min(float_maxes)
            cache_k_nope = torch.nan_to_num(
                cache_k_nope.float(),
                nan=0.0,
                posinf=max_finite,
                neginf=-max_finite,
            )
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
            kv_buffer = kv_buffer.view(self.dtype)
        elif cache_k_nope.dtype != kv_buffer.dtype:
            cache_k_nope = cache_k_nope.to(kv_buffer.dtype)
        set_mla_kv_buffer_triton(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
            enable_pdl=pdl_enabled(),
            sanitize=False,
        )

    def get_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qk_rope_head_dim != 0:
            return super().get_mla_kv_buffer(layer, loc, dst_dtype=dst_dtype)

        dst_dtype = dst_dtype or self.dtype
        kv_buffer = self.kv_buffer[layer.layer_id]
        index = loc if loc.dtype == torch.int64 else loc.to(torch.int64)
        cache_k_nope = kv_buffer.index_select(0, index)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.dtype)
        if cache_k_nope.dtype != dst_dtype:
            cache_k_nope = cache_k_nope.to(dst_dtype)
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, 0), dtype=dst_dtype, device=kv_buffer.device
        )
        return cache_k_nope, cache_k_rope
