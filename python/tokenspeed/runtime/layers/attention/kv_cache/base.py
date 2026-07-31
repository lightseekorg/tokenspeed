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

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.cache.flat_host_mirror import HostMirrorFamily
from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter

logger = get_colorful_logger(__name__)


class BaseTokenToKVPool:
    """A memory pool that maps a token location to its kv cache data."""

    paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...] = ()
    paged_cache_group_page_counts: dict[str, int] = {}
    supports_hierarchical_kv_cache: bool = True
    # Flat-cache pools that alias recurrent-state bytes and KV in one slab must
    # zero physical pages on reuse to avoid poisoned tails. Pure-attention
    # pools do not alias state, so reused pages need no sanitization.
    flat_kv_requires_page_zeroing: bool = False

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
    ):
        self.dtype = dtype
        self.rank = rank
        self.size = size
        self.page_size = page_size
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            #  Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.offload_chunk_page_num = 1024
        self.token_slot_refs = None

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None
        logger.info(
            f"Initialized token to kv pool with size {size}, dtype {dtype}, device {device}, page size {page_size}, rank {rank}"
        )

    @classmethod
    def cell_size(self) -> int:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def set_token_slot_refs(self, token_slot_refs: torch.Tensor):
        self.token_slot_refs = token_slot_refs

    def bind_paged_cache_scheduler(self, scheduler: object) -> None:
        """Optional hook for model-specific paged-cache diagnostics."""
        return None

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        """Zero the KV buffers in place.

        Used by sleep/wake: after resume_memory_occupation re-maps the KV region
        its pages hold garbage, so zero them. Subclasses store buffers under
        different attributes (``k_buffer``/``v_buffer`` for MHA, ``kv_buffer`` —
        possibly tuples — for MLA); introspect the known names so every pool is
        covered without per-class overrides. For non-quantized KV this is
        belt-and-suspenders (paging overwrites); for FP8 KV it removes garbage.
        """
        attrs = (
            "k_buffer",
            "v_buffer",
            "kv_buffer",
            # DeepSeek V4 pool buffer names.
            "swa_kv_buffer",
            "compressed_kv_buffer",
            "compressor_state_buffer",
            "indexer_kv_buffer",
            "indexer_state_buffer",
        )
        for attr in attrs:
            for entry in getattr(self, attr, None) or []:
                items = entry if isinstance(entry, (tuple, list)) else (entry,)
                for t in items:
                    if torch.is_tensor(t):
                        t.zero_()

    def maybe_log_paged_cache_group_pages(self) -> None:
        return None

    def host_mirror_families(self) -> list[HostMirrorFamily]:
        """Device tensor families the flat host tier (kvstore L2) mirrors.

        ``FlatHostMirror`` copies whole pages byte-blind, so a pool only has
        to declare which tensors carry a layer's page bytes; the mirror
        deduplicates aliased slabs and fences a layer on the LAST family
        listing it. The default describes the K/V pair plus GDN state slabs;
        pools with another layout override (MLA's fused latent, DSA's packed
        index-K).

        Returns:
            Families in fence order, or an empty list when this pool cannot
            be mirrored -- ``flat_host_tier_unsupported_reason`` then
            downgrades L2 instead of building a partial mirror.
        """
        k_buffer = getattr(self, "k_buffer", None)
        v_buffer = getattr(self, "v_buffer", None)
        if k_buffer is None or v_buffer is None:
            return []
        families = [
            HostMirrorFamily.per_layer(k_buffer, self.page_size),
            HostMirrorFamily.per_layer(v_buffer, self.page_size),
        ]
        if getattr(self, "state_slabs", None):
            # State slabs are page-indexed: one snapshot row per page id,
            # bound to their layers by the pool's get_state_buffers.
            layer_states = []
            for layer_id in range(len(k_buffer)):
                try:
                    layer_states.append(tuple(self.get_state_buffers(layer_id)))
                except ValueError:
                    layer_states.append(())  # not a state layer
            families.append(HostMirrorFamily(tuple(layer_states), 1))
        return families

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def get_cpu_copy(self, page_indices: list[int]) -> torch.Tensor:
        raise NotImplementedError()

    def load_cpu_copy(
        self, kv_cache_cpu: torch.Tensor, page_indices: list[int]
    ) -> None:
        raise NotImplementedError()

    @property
    def prefix_cache_required_group_ids(self) -> tuple[str, ...] | None:
        """None means adjunct disabled; subclasses return required group ids."""
        return None

    # Buffer metadata used by prefill/decode disaggregation.
    def get_contiguous_buf_infos(self):
        raise NotImplementedError()

    def get_contiguous_buf_unit_lens(self):
        return [1] * len(self.get_contiguous_buf_infos()[2])

    # Layerwise buffer offsets used by prefill/decode disaggregation.
    def get_layerwise_buf_info_offsets(self, start_idx=0):
        raise NotImplementedError()
