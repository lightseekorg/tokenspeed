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

from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter

logger = get_colorful_logger(__name__)


class BaseTokenToKVPool:
    """A memory pool that maps a token location to its kv cache data."""

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
            "Initialized token to kv pool with size %s, dtype %s, device %s, page size %s, rank %s",
            size,
            dtype,
            device,
            page_size,
            rank,
        )

    @classmethod
    def cell_size(self) -> int:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def set_token_slot_refs(self, token_slot_refs: torch.Tensor):
        self.token_slot_refs = token_slot_refs

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

    # Layerwise buffer offsets used by prefill/decode disaggregation.
    def get_layerwise_buf_info_offsets(self, start_idx=0):
        raise NotImplementedError()
