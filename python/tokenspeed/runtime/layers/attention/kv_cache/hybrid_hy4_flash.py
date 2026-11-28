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

"""Hybrid DSA + SWA-MLA cache pool and compressed-indexer tail state for Hy4-flash."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from tokenspeed_kernel.ops.kvcache.index_k_scale import index_k_scale_dtype
from tokenspeed_kernel.ops.kvcache.triton import get_mla_kv_buffer_triton
from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.hy4_flash import (
    HY4_FLASH_LOGICAL_BLOCK_TOKENS,
    Hy4FlashPoolOptions,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class _Hy4KPoolTailWorkspace:
    """Arena-shared request-local tail buffers for incomplete 4-token index groups."""

    options: Hy4FlashPoolOptions
    token_keys: torch.Tensor
    group_pos0: torch.Tensor
    row_by_layer: dict[int, int]


class HybridHy4FlashTokenToKVPool(MLATokenToKVPool):
    """Hy4-flash KV pool spanning 12 DSA layers (latent KV + 4:1 index-K) and 30 SWA layers."""

    latent_write_sanitizes: ClassVar[bool] = False
    index_k_format: ClassVar[str] = "fp8_scaled"

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MLATokenToKVPool.layer_plane_bindings,
        "index_k": "_index_k",
    }

    def __init__(
        self,
        arena: CacheArena,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        quant_method: str,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        rank: int,
        *,
        pool_options: Hy4FlashPoolOptions,
        layer_types: tuple[str, ...],
        field_layer_offset: int = 0,
    ) -> None:
        self._layer_types = tuple(layer_types)
        self.index_head_dim = pool_options.index_head_dim
        self.index_kpool = pool_options.index_kpool
        self._index_k_page_size = HY4_FLASH_LOGICAL_BLOCK_TOKENS // self.index_kpool
        self._index_k: list[torch.Tensor | None] = []
        super().__init__(
            arena=arena,
            model_dtype=model_dtype,
            dtype=dtype,
            quant_method=quant_method,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            rank=rank,
            field_layer_offset=field_layer_offset,
        )
        self._hy4_kpool_tail_workspace = self._bind_hy4_kpool_tail_workspace(
            pool_options
        )

    @property
    def index_k_page_size(self) -> int:
        return self._index_k_page_size

    def _bind_hy4_kpool_tail_workspace(
        self, options: Hy4FlashPoolOptions
    ) -> _Hy4KPoolTailWorkspace:
        attribute = "_hy4_flash_kpool_tail_workspace"
        workspace = getattr(self.arena, attribute, None)
        if workspace is None:
            with self.arena.memory_saver_adapter.region(
                tag="kv_cache", enable_cpu_backup=False
            ):
                token_keys = torch.zeros(
                    (
                        len(options.dsa_layer_ids),
                        options.num_request_slots,
                        options.tail_width,
                        options.index_head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.arena.device,
                )
                group_pos0 = torch.zeros(
                    (options.num_request_slots, 3),
                    dtype=torch.int64,
                    device=self.arena.device,
                )
            workspace = _Hy4KPoolTailWorkspace(
                options=options,
                token_keys=token_keys,
                group_pos0=group_pos0,
                row_by_layer={
                    layer_id: row for row, layer_id in enumerate(options.dsa_layer_ids)
                },
            )
            setattr(self.arena, attribute, workspace)
        elif workspace.options != options:
            raise ValueError(
                "Hy4-flash cache views disagree on compressed-indexer tail geometry"
            )
        return workspace

    @override
    def get_kv_size_bytes(self) -> int:
        return self.arena.buffer.nbytes

    @override
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)[..., : -self.qk_rope_head_dim]

    @override
    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        del cache_v
        self.set_mla_kv_buffer(
            layer,
            loc,
            cache_k[..., : -self.qk_rope_head_dim],
            cache_k[..., -self.qk_rope_head_dim :],
            write_mask=None,
        )

    @override
    def get_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.get_key_buffer(layer.layer_id)
        latent_dim = cache.shape[-1] - self.qk_rope_head_dim
        out_dtype = dst_dtype or self.dtype
        if not cache.is_cuda:
            if cache.ndim == 4:
                page_size = cache.shape[1]
                slots = loc.to(torch.int64)
                gathered = cache[slots // page_size, slots % page_size]
            else:
                gathered = cache[loc.to(torch.int64)]
            if gathered.dtype != out_dtype:
                gathered = gathered.to(out_dtype)
            nope, rope = gathered.split((latent_dim, self.qk_rope_head_dim), dim=-1)
            return nope.contiguous(), rope.contiguous()

        output = torch.empty(
            (loc.numel(), 1, cache.shape[-1]),
            device=cache.device,
            dtype=out_dtype,
        )
        nope, rope = output.split((latent_dim, self.qk_rope_head_dim), dim=-1)
        get_mla_kv_buffer_triton(cache, loc, nope, rope)
        return nope, rope

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        try:
            buffer = self._index_k[layer_id]
        except IndexError as exc:
            raise ValueError(
                f"layer {layer_id} has no Hy4-flash index-K cache"
            ) from exc
        if buffer is None:
            raise ValueError(f"SWA layer {layer_id} has no Hy4-flash index-K cache")
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return buffer

    def index_k_block_views(
        self, buf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(values [pages, rows, head_dim] fp8, scales [pages, rows, groups] scale_dtype)``."""
        rows_per_page = buf.shape[1]
        num_groups = self.index_head_dim // 128
        scale_bytes = torch._utils._element_size(torch.float32)
        page_stride = buf.stride(0)
        values = torch.as_strided(
            buf,
            (buf.shape[0], rows_per_page, self.index_head_dim),
            (page_stride, self.index_head_dim, 1),
        ).view(torch.float8_e4m3fn)
        scales = torch.as_strided(
            buf,
            (buf.shape[0], rows_per_page, num_groups * scale_bytes),
            (page_stride, num_groups * scale_bytes, 1),
            buf.storage_offset() + rows_per_page * self.index_head_dim,
        ).view(index_k_scale_dtype())
        return values, scales

    def get_hy4_kpool_buffers(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(index_k_buf, layer_tail_token_keys, group_pos0)`` for a DSA layer."""
        index_k = self.get_index_k_buffer(layer_id)
        try:
            row = self._hy4_kpool_tail_workspace.row_by_layer[
                self._field_layer_id(layer_id)
            ]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer_id} has no Hy4-flash compressed-indexer tail cache"
            ) from exc
        ws = self._hy4_kpool_tail_workspace
        return index_k, ws.token_keys[row], ws.group_pos0

    def kpool_tail_slot_rows(self, slot: int) -> list[torch.Tensor]:
        """Export one request slot's compressed-indexer tail state for retraction."""
        ws = self._hy4_kpool_tail_workspace
        return [ws.token_keys[:, slot], ws.group_pos0[slot]]

    @torch.no_grad()
    @override
    def clear_kv_buffers(self) -> None:
        super().clear_kv_buffers()
        self._hy4_kpool_tail_workspace.token_keys.zero_()
        self._hy4_kpool_tail_workspace.group_pos0.zero_()


__all__ = ["HybridHy4FlashTokenToKVPool", "_Hy4KPoolTailWorkspace"]
