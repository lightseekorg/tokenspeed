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

"""Paged BF16 latent and page-planar FP8 index views for dots3-note."""

from typing import ClassVar

import torch
from tokenspeed_kernel.ops.attention.prologue import LatentKVCache
from tokenspeed_kernel.ops.kvcache.triton import (
    get_mla_kv_buffer_triton,
    index_k_block_split_scatter,
    set_mla_kv_buffer_triton,
)

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dots3_note import Dots3NoteAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec
from tokenspeed.runtime.layers.paged_attention import PagedAttention


def create_dots3_note_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    """Bind a target or MTP layer window to the shared dots3 arena.

    Args:
        spec: Cache recipe's layer view.
        config: Dots3 target or native MTP attention config.
        arena: Allocation shared by both views.
        num_layers: Number of local attention layers.
        rank: Cache view's distributed rank.
        field_layer_offset: First layer in the merged cache plan.

    Returns:
        A pool exposing only this view's latent and index fields.
    """
    if config.component(Dots3NoteAttnConfig) is None:
        raise TypeError(f"cache family {spec.family!r} requires Dots3NoteAttnConfig")
    return Dots3NoteCachePool(
        arena,
        layer_num=num_layers,
        rank=rank,
        field_layer_offset=field_layer_offset,
    )


class Dots3NoteCachePool(CachePool):
    layer_plane_bindings: ClassVar[dict[str, str]] = {"latent_kv": "_latent_kv"}

    def __init__(
        self,
        arena: CacheArena,
        *,
        layer_num: int,
        rank: int,
        field_layer_offset: int,
    ):
        super().__init__(
            arena, torch.bfloat16, rank, field_layer_offset=field_layer_offset
        )
        self.layer_num = layer_num
        self._latent_kv: list[torch.Tensor] = []
        self._index_k: dict[int, torch.Tensor] = {}
        field_ids = arena.field_ids()
        for layer_id in range(layer_num):
            field_layer = self._field_layer_id(layer_id)
            kv = arena.field_pages(f"layer.{field_layer}.latent_kv")
            if kv.dtype != torch.bfloat16 or tuple(kv.shape[1:]) not in (
                (64, 1, 576),
                (32, 1, 1088),
            ):
                raise ValueError(
                    f"Unsupported dots3 latent geometry at layer {layer_id}"
                )
            self._latent_kv.append(kv)
            index_id = f"layer.{field_layer}.index_k"
            if kv.shape[1] == 64:
                index = arena.field_pages(index_id)
                if index.dtype != torch.uint8 or tuple(index.shape[1:]) != (64, 132):
                    raise ValueError(
                        f"Unsupported dots3 index geometry at layer {layer_id}"
                    )
                if (
                    arena.plan.field(index_id).group_id
                    != arena.plan.field(f"layer.{field_layer}.latent_kv").group_id
                ):
                    raise ValueError("Dots3 index and latent KV must share one group")
                # Collapse within each page only; the interleaved page stride stays.
                self._index_k[layer_id] = index.view(index.shape[0], 64 * 132)
            elif index_id in field_ids:
                raise ValueError(f"SWA layer {layer_id} cannot own an index field")

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._field_layer_id(layer_id)
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self._latent_kv[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)[..., :-64]

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.get_key_buffer(layer_id)
        return kv, kv[..., :-64]

    def kv_write_target(
        self, layer_id: int, slots: torch.Tensor, write_mask: torch.Tensor | None
    ) -> LatentKVCache:
        """Expose the original paged arena view, never a flattened cache copy."""
        return LatentKVCache(
            kv_cache=self.get_key_buffer(layer_id),
            sanitize=False,
            slots=slots,
            write_mask=write_mask,
        )

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "dots3 stores latent KV, not independent K/V; use set_mla_kv_buffer"
        )

    def set_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        *,
        write_mask: torch.Tensor | None = None,
    ) -> None:
        """Scatter BF16 latent/RoPE rows, skipping masked rows; None writes all."""
        kv = self.get_key_buffer(layer.layer_id)
        if cache_k_nope.shape[-1] != kv.shape[-1] - 64 or cache_k_rope.shape[-1] != 64:
            raise ValueError("Dots3 latent/RoPE widths do not match this layer's cache")
        set_mla_kv_buffer_triton(
            kv, loc, cache_k_nope, cache_k_rope, sanitize=False, write_mask=write_mask
        )

    def get_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather selected group-local slots, returning latent and RoPE tensors."""
        kv = self.get_key_buffer(layer.layer_id)
        dtype = self.dtype if dst_dtype is None else dst_dtype
        latent = torch.empty(
            (loc.numel(), 1, kv.shape[-1] - 64), dtype=dtype, device=kv.device
        )
        rope = torch.empty((loc.numel(), 1, 64), dtype=dtype, device=kv.device)
        get_mla_kv_buffer_triton(kv, loc, latent, rope)
        return latent, rope

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        """Return Full-layer ``[pages, 8448]`` bytes, retaining the page stride."""
        self._field_layer_id(layer_id)
        if layer_id not in self._index_k:
            raise ValueError(f"SWA layer {layer_id} has no index cache")
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self._index_k[layer_id]

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_scale: torch.Tensor,
    ) -> None:
        """Store the model's power-of-two FP8 codec; do not quantize a second time."""
        if index_k.dtype != torch.float8_e4m3fn or index_scale.dtype != torch.float32:
            raise ValueError("Dots3 index writes require E4M3FN keys and FP32 scales")
        index_k_block_split_scatter(
            self.get_index_k_buffer(layer_id),
            index_k,
            index_scale,
            loc,
            page_size=64,
            head_dim=128,
            group_size=128,
            write_mask=None,
        )
