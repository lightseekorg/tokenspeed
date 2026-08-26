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

from typing import ClassVar

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    quantize_store_kv_mxfp8,
    store_kv_cache,
    store_sf_interleaved,
)

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    MXFP8_KV_SCALE_TILE_TOKENS,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


GB = 1024 * 1024 * 1024


class MHATokenToKVPool(CachePool):
    def __init__(
        self,
        arena: CacheArena,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        rank: int,
        *,
        layer_types: tuple[str, ...] = (),
        layer_kv_head_counts: tuple[int, ...] | None = None,
        kv_alloc_head_count: int | None = None,
        field_layer_offset: int = 0,
    ):
        super().__init__(
            arena,
            dtype,
            rank,
            field_layer_offset=field_layer_offset,
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        # Fewer-head layers reinterpret the allocation width as more rows.
        self._layer_kv_head_counts = (
            tuple(int(h) for h in layer_kv_head_counts)
            if layer_kv_head_counts
            else None
        )
        # Pre-TP head count ``head_num`` is the per-rank shard of — the view
        # normalization base. Falling back to max(counts) is only correct
        # when the pool's own layers include an alloc-width layer: an
        # all-narrow pool (Inkling MTP draft with full-attention-only
        # depths, alloc'd at the config max) would silently collapse the
        # reinterpretation and write narrow rows into wide strides (the
        # #65/1.82-accept GQA-stride corruption).
        self._kv_alloc_head_count = (
            int(kv_alloc_head_count) if kv_alloc_head_count else None
        )
        self._layer_types = tuple(layer_types or ())
        self._bind_layer_planes()

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            "KV Cache is allocated. K size: %.2f GB, V size: %.2f GB.",
            k_size / GB,
            v_size / GB,
        )

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        "k": "k_buffer",
        "v": "v_buffer",
    }

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        # Different layer views may share an address through the memory plan.
        # Count each physical region once, independent of Python identity.
        k_caches = {t.data_ptr(): t for t in self.k_buffer if t is not None}
        v_caches = {t.data_ptr(): t for t in self.v_buffer if t is not None}
        k_size_bytes = sum(t.nbytes for t in k_caches.values())
        v_size_bytes = sum(t.nbytes for t in v_caches.values())
        return k_size_bytes, v_size_bytes

    def _layer_row_view(self, buf: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Per-layer token-row view over one byte-uniform cache field.

        Fields are allocated ``(rows, head_num, head_dim)`` at the max head
        count; a layer serving fewer heads reinterprets the same bytes as
        ``rows * (head_num / heads_l)`` rows of ``heads_l`` heads (full
        layers: 2x the token rows per slot — the zero-padding contract).
        """
        if self._layer_kv_head_counts is None:
            return buf
        heads_l = self._layer_heads_per_rank(layer_id)
        if heads_l == self.head_num:
            return buf
        return buf.reshape(-1, heads_l, self.head_dim)

    def _layer_heads_per_rank(self, layer_id: int) -> int:
        counts = self._layer_kv_head_counts
        if counts is None:
            # Uniform head counts (hetero off): every layer serves head_num.
            return self.head_num
        served = counts[layer_id]
        # head_num is the per-rank shard of the ALLOCATION width (the config
        # max); scale the pre-TP served count proportionally. max(counts) is
        # only a valid stand-in when some layer serves the alloc width — an
        # all-narrow pool must still reinterpret every layer.
        alloc = self._kv_alloc_head_count or max(counts)
        return max(1, self.head_num * served // alloc)

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        buf = self.k_buffer[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV buffer")
        if self.store_dtype != self.dtype:
            buf = buf.view(self.dtype)
        return self._layer_row_view(buf, layer_id)

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        buf = self.v_buffer[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV buffer")
        if self.store_dtype != self.dtype:
            buf = buf.view(self.dtype)
        return self._layer_row_view(buf, layer_id)

    def get_value_buffer(self, layer_id: int):
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self._get_value_buffer(layer_id)

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
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)
        # Locs are in per-layer view rows: the store must target the same view get_key_buffer serves
        store_kv_cache(
            cache_k,
            cache_v,
            self._layer_row_view(self.k_buffer[layer_id], layer_id),
            self._layer_row_view(self.v_buffer[layer_id], layer_id),
            loc,
        )


class MHATokenToKVPoolMXFP8(MHATokenToKVPool):
    """MHA KV pool storing MXFP8 block-scaled FP8 (data + UE8M0 scales).

    Data buffers hold float8_e4m3fn; scale buffers hold one float8_e8m0fnu
    per 32 elements of head_dim. ``set_kv_buffer`` expects PRE-QUANTIZED
    K/V plus per-token scale tensors (producer: ``quantize_mxfp8``); the
    bf16 per-tensor-scale paths of the base class do not apply.

    Scale layout follows what the FA4 blockscaled kernel consumes:
    page_size 128 stores scales interleaved in the BlockScaledBasicChunk
    atom ([num_pages, heads, 32, 4, 4], written via
    ``store_sf_interleaved``); any other page size stores them flat
    ([slots, heads, head_dim // 32]).

    When the plan aliases fields, data and scale views remain layer-local
    Python objects while addressing the same physical bytes.
    """

    MXFP8_SCALE_BLOCK_SIZE = 32

    # Scale planes stay page-major: the blockscaled kernels read them in the
    # interleaved layout the plan already gives them.
    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MHATokenToKVPool.layer_plane_bindings,
        "k_scale": "k_scale_buffer",
        "v_scale": "v_scale_buffer",
    }

    def _bind_layer_planes(self) -> None:
        if self.head_dim % self.MXFP8_SCALE_BLOCK_SIZE:
            raise ValueError("MXFP8 head_dim must be divisible by 32")
        # These writes go through dtype-aware kernels, so the fp8 view the
        # plan hands out is the one to keep for input reinterpretation too.
        self.store_dtype = torch.float8_e4m3fn
        super()._bind_layer_planes()

    def _layer_page_tokens(self, layer_id: int) -> int:
        """Tokens represented by one page id for this layer."""
        # Byte-uniform slots factor one id through the layer's head count.
        heads_l = self._layer_heads_per_rank(layer_id)
        return self.arena.kv_page_size * self.head_num // heads_l

    def _layer_scale_view(self, buf: torch.Tensor, layer_id: int) -> torch.Tensor:
        """(num_ids, heads_l, k_l, 32, 4, 4) view over a layer's SF slots
        (the paged interleaved layout the blockscaled kernels consume)."""
        heads_l = self._layer_heads_per_rank(layer_id)
        k_l = self._layer_page_tokens(layer_id) // MXFP8_KV_SCALE_TILE_TOKENS
        sf_dim = self.head_dim // self.MXFP8_SCALE_BLOCK_SIZE
        return buf.view(buf.shape[0], heads_l, k_l, 32, sf_dim, sf_dim)

    def get_kv_scale_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(k_scale, v_scale) buffers for the blockscaled attention kernel.

        Returns the per-layer scale views consumed by the attention kernel.
        """
        k_sf = self.k_scale_buffer[layer_id]
        v_sf = self.v_scale_buffer[layer_id]
        if self._layer_kv_head_counts is not None:
            return (
                self._layer_scale_view(k_sf, layer_id),
                self._layer_scale_view(v_sf, layer_id),
            )
        return k_sf, v_sf

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        layer_id_override: int | None = None,
    ):
        assert (
            cache_k.dtype == self.store_dtype
        ), "MXFP8 pool expects pre-quantized fp8 K (see quantize_mxfp8)"
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 pool requires per-token e8m0 scale tensors"
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        # Byte views: triton can't mask-fill fp8; locs are per-layer view rows (target the served view)
        store_kv_cache(
            cache_k.view(torch.uint8),
            cache_v.view(torch.uint8),
            self._layer_row_view(self.k_buffer[layer_id], layer_id).view(torch.uint8),
            self._layer_row_view(self.v_buffer[layer_id], layer_id).view(torch.uint8),
            loc,
        )
        if self._layer_kv_head_counts is not None:
            page_tokens = self._layer_page_tokens(layer_id)
            store_sf_interleaved(
                k_scale,
                self.k_scale_buffer[layer_id],
                loc,
                page_size=page_tokens,
            )
            store_sf_interleaved(
                v_scale,
                self.v_scale_buffer[layer_id],
                loc,
                page_size=page_tokens,
            )
        elif self.arena.kv_page_size == MXFP8_KV_SCALE_TILE_TOKENS:
            store_sf_interleaved(k_scale, self.k_scale_buffer[layer_id], loc)
            store_sf_interleaved(v_scale, self.v_scale_buffer[layer_id], loc)
        else:
            self.k_scale_buffer[layer_id][loc] = k_scale
            self.v_scale_buffer[layer_id][loc] = v_scale

    def quantize_and_set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id_override: int | None = None,
    ) -> bool:
        """Fused per-token quantize + data store + SF scatter (one launch).

        Bit-identical to quantize_mxfp8 + set_kv_buffer (parity-tested) and
        keeps the store inside the PDL chain. Returns False when the layout
        has no interleaved-SF path (page not a 128 multiple) — the caller
        falls back to the split path.
        """
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        if self._layer_kv_head_counts is not None:
            page_tokens = self._layer_page_tokens(layer_id)
        elif self.arena.kv_page_size == MXFP8_KV_SCALE_TILE_TOKENS:
            page_tokens = MXFP8_KV_SCALE_TILE_TOKENS
        else:
            return False
        if self.head_dim != 128:
            return False
        quantize_store_kv_mxfp8(
            k,
            v,
            self._layer_row_view(self.k_buffer[layer_id], layer_id),
            self._layer_row_view(self.v_buffer[layer_id], layer_id),
            self.k_scale_buffer[layer_id],
            self.v_scale_buffer[layer_id],
            loc,
            page_tokens=page_tokens,
        )
        return True

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        for sf in {
            buffer.data_ptr(): buffer for buffer in self.k_scale_buffer
        }.values():
            k_size += sf.numel() * sf.element_size()
        for sf in {
            buffer.data_ptr(): buffer for buffer in self.v_scale_buffer
        }.values():
            v_size += sf.numel() * sf.element_size()
        return k_size, v_size
