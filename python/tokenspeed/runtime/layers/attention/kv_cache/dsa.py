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
from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

from tokenspeed.runtime.layers.attention.configs.dsa import (
    index_k_plane_dtype,
    index_k_row_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.mla import (
    MLATokenToKVPool,
    _get_tensor_size_bytes,
)

_INDEX_K_FP8_GROUP_SIZE = 128


def split_index_k_rows(
    packed: torch.Tensor, *, index_head_dim: int, index_k_format: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """View packed index-K rows as the keys (and scales) of their format.

    The inverse of :meth:`DSATokenToKVPool.gather_index_k_rows`'s packing:
    no copy, the views share ``packed``'s storage. The pair is what
    ``dsa_prefill_topk`` takes as rows in workspace-row order:
    ``index_k_fp8`` / ``index_k_scale`` for ``fp8_scaled``, ``index_k_bf16``
    (scale ``None``) for ``bf16``.

    Args:
        packed: ``[rows, index_k_row_bytes(index_head_dim, index_k_format)]``
            uint8 rows.
        index_head_dim: Elements of one key.
        index_k_format: The plane's format (``configs/dsa.py``
            ``INDEX_K_FORMATS``).

    Returns:
        ``fp8_scaled``: ``[rows, index_head_dim]`` uint8 FP8 keys and
        ``[rows, groups]`` fp32 scales. ``bf16``: ``[rows, index_head_dim]``
        bf16 keys and ``None``.
    """
    row_bytes = index_k_row_bytes(index_head_dim, index_k_format)
    if packed.dim() != 2 or packed.dtype != torch.uint8:
        raise ValueError(f"packed index-K rows are 2-D uint8, got {packed.dtype}")
    if packed.shape[1] != row_bytes:
        raise ValueError(
            f"packed {index_k_format} index-K rows are {packed.shape[1]} bytes "
            f"wide, not {row_bytes}"
        )
    if index_k_format == "fp8_scaled":
        return (
            packed[:, :index_head_dim],
            packed[:, index_head_dim:].view(torch.float32),
        )
    return packed.view(index_k_plane_dtype(index_k_format)), None


class DSATokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        *args,
        index_head_dim: int,
        **kwargs,
    ):
        self.index_head_dim = int(index_head_dim)
        super().__init__(*args, **kwargs)

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MLATokenToKVPool.layer_plane_bindings,
        "index_k": "index_k_buffer",
    }

    def get_kv_size_bytes(self):
        return super().get_kv_size_bytes() + _get_tensor_size_bytes(self.index_k_buffer)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self.index_k_buffer[layer_id]

    def gather_index_k_rows(
        self, layer_id: int, slots: torch.Tensor, *, index_k_format: str
    ) -> torch.Tensor:
        """Read index-K rows out of the plane as packed bytes, one row per slot.

        The read side of :meth:`set_index_k_buffer`, for the rows a
        query-context-parallel history gather contributes. The caller names
        the format it packs rows in (``DSAConfig.index_k_format``) and the
        plane must be that format's dtype (``index_k_plane_dtype``). An
        ``fp8_scaled`` plane stores every page as its ``page_size`` FP8 rows
        followed by their fp32 scales (``index_k_block_split_scatter``), and
        each row comes back as its FP8 bytes followed by its scale bytes so
        one gather moves both; a ``bf16`` plane's row is its key's bytes.
        :func:`split_index_k_rows` views the packed rows apart again.

        Args:
            layer_id: The indexer layer whose plane to read.
            slots: ``[rows]`` int64 local cache slots.
            index_k_format: The format to pack rows in, one of
                ``configs/dsa.py`` ``INDEX_K_FORMATS``.

        Returns:
            ``[rows, index_k_row_bytes(index_head_dim, index_k_format)]`` uint8
            rows in ``slots`` order.
        """
        buf = self.get_index_k_buffer(layer_id)
        plane_dtype = index_k_plane_dtype(index_k_format)
        if buf.dtype != plane_dtype:
            raise ValueError(
                f"layer {layer_id} index-K plane is {buf.dtype}, not the "
                f"{plane_dtype} of index_k_format={index_k_format!r}"
            )
        head_dim = self.index_head_dim
        if index_k_format != "fp8_scaled":
            # A plain plane: the row is the key's bytes.
            return buf.reshape(-1, head_dim)[slots].view(torch.uint8)
        page_size = int(self.arena.kv_page_size)
        row_bytes = index_k_row_bytes(head_dim, index_k_format)
        scale_bytes = row_bytes - head_dim
        page_bytes = page_size * row_bytes
        num_pages = buf.numel() // page_bytes
        pages = buf.reshape(-1)[: num_pages * page_bytes].view(num_pages, page_bytes)
        fp8 = pages[:, : page_size * head_dim].view(num_pages, page_size, head_dim)
        scale = pages[:, page_size * head_dim :].view(num_pages, page_size, scale_bytes)
        page = torch.div(slots, page_size, rounding_mode="floor")
        offset = slots - page * page_size
        return torch.cat((fp8[page, offset], scale[page, offset]), dim=1)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        *,
        write_mask: torch.Tensor | None,
    ) -> None:
        """Write one layer's index keys into its plane, in the plane's format.

        The plane's dtype is its format (``configs/dsa.py`` INDEX_K_FORMATS):
        a uint8 plane takes FP8 keys with per-128 fp32 scales, a bf16 plane
        takes the keys unquantized. Nothing is converted between formats.
        """
        if index_k.dtype != self.model_dtype:
            index_k = index_k.to(self.model_dtype)
        index_k = index_k.view(-1, self.index_head_dim)
        buf = self.index_k_buffer[layer_id]
        if buf.dtype == torch.bfloat16:
            rows = index_k.to(buf.dtype)
            slots = loc.to(torch.int64)
            if write_mask is not None:
                # Rows this rank does not own keep what the plane holds; the
                # gather keeps the shapes static for graph capture.
                rows = torch.where(write_mask.unsqueeze(1), rows, buf[slots])
            buf[slots] = rows
            return
        if buf.dtype != torch.uint8:
            raise TypeError(
                f"index-K plane dtype {buf.dtype} has no write path: uint8 "
                "(fp8_scaled) or bfloat16 (bf16)"
            )
        index_k_fp8, index_k_scale = quantize_fp8_with_scale(
            index_k,
            granularity="token_group",
            group_size=_INDEX_K_FP8_GROUP_SIZE,
            scale_encoding="float32",
        )

        # Fused scatter; (page, slot_in_page) is derived from loc in-kernel.
        index_k_block_split_scatter(
            buf,
            index_k_fp8,
            index_k_scale,
            loc,
            page_size=self.arena.kv_page_size,
            head_dim=self.index_head_dim,
            group_size=_INDEX_K_FP8_GROUP_SIZE,
            write_mask=write_mask,
        )
