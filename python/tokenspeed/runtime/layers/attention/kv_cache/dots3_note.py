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

"""Model-local views of the two differently shaped MLA history groups."""

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    get_mla_kv_buffer_triton,
    index_k_block_split_scatter,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8

from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool
from tokenspeed.runtime.layers.paged_attention import PagedAttention


class Dots3NoteTokenToKVPool(DSATokenToKVPool):
    def get_kv_size_bytes(self) -> int:
        return self.arena.buffer.nbytes

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)[..., : -self.qk_rope_head_dim]

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        self.set_mla_kv_buffer(
            layer,
            loc,
            cache_k[..., : -self.qk_rope_head_dim],
            cache_k[..., -self.qk_rope_head_dim :],
            write_mask=None,
        )

    def get_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self.get_key_buffer(layer.layer_id)
        latent = cache.shape[-1] - self.qk_rope_head_dim
        output = torch.empty(
            (loc.numel(), 1, cache.shape[-1]),
            device=cache.device,
            dtype=dst_dtype or self.dtype,
        )
        nope, rope = output.split((latent, self.qk_rope_head_dim), dim=-1)
        get_mla_kv_buffer_triton(cache, loc, nope, rope)
        return nope, rope

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        buffer = super().get_index_k_buffer(layer_id)
        if buffer is None:
            raise ValueError(f"SWA layer {layer_id} has no index-K cache")
        # The scoring kernels accept a page matrix with an explicit byte stride.
        return buffer.view(buffer.shape[0], -1)

    def gather_index_k_rows(
        self, layer_id: int, slots: torch.Tensor, *, index_k_format: str
    ) -> torch.Tensor:
        if index_k_format != "fp8_scaled":
            raise ValueError("dots3-note uses scaled FP8 index-K")
        pages = self.get_index_k_buffer(layer_id)
        page_size = self.index_k_buffer[layer_id].shape[1]
        split = page_size * self.index_head_dim
        keys = pages[:, :split].view(-1, page_size, self.index_head_dim)
        scales = pages[:, split:].view(-1, page_size, 4)
        page, offset = slots // page_size, slots % page_size
        return torch.cat((keys[page, offset], scales[page, offset]), dim=-1)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        *,
        write_mask: torch.Tensor | None,
    ) -> None:
        keys, scales = quantize_fp8(
            index_k.to(self.model_dtype).view(-1, self.index_head_dim),
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        index_k_block_split_scatter(
            self.get_index_k_buffer(layer_id),
            keys,
            scales,
            loc,
            page_size=self.index_k_buffer[layer_id].shape[1],
            head_dim=self.index_head_dim,
            group_size=128,
            write_mask=write_mask,
        )
