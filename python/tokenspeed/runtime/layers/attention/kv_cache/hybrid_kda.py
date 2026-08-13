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

"""Paged latent-KV and KDA-state cache."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_LAYER_TYPES,
)


class HybridKDATokenToKVPool(MLATokenToKVPool):
    """MLA compute interface whose latent KV and KDA state share one buffer."""

    def __init__(
        self,
        *,
        memory_plan: CacheMemoryPlan,
        layer_group_ids: tuple[str, ...],
        layer_types: tuple[str, ...],
        state_field_dtypes: Mapping[str, torch.dtype] | None = None,
        **kwargs,
    ):
        self._layer_types = tuple(layer_types)
        group_ids = tuple(layer_group_ids)
        self._group_ids_by_layer = dict(enumerate(group_ids))
        self._state_field_dtypes = dict(state_field_dtypes or {})
        self._state_buffers_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.paged_cache_requires_page_zeroing = True

        layer_num = kwargs["layer_num"]
        if len(self._layer_types) != layer_num:
            raise ValueError("cache layer types must cover every model layer")
        if len(group_ids) != layer_num:
            raise ValueError("cache group ids must cover every model layer")

        super().__init__(
            memory_plan=memory_plan,
            layer_group_ids=group_ids,
            **kwargs,
        )

    def _create_buffers(self) -> None:
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            self._bind_buffers()

    def _bind_buffers(self) -> None:
        if self.quant_method == "per_token_head":
            raise ValueError("KDA cache does not support per-token-head KV")
        if self.plan.prefix_granularity != self.page_size:
            raise ValueError(
                f"cache plan P={self.plan.prefix_granularity} does not match "
                f"pool page_size={self.page_size}"
            )
        max_packing = max(
            group.cache_blocks_per_lcm_block for group in self.plan.groups
        )
        expected_size = self.plan.num_lcm_blocks * max_packing * self.page_size
        if self.size != expected_size:
            raise ValueError(
                f"cache pool size {self.size} does not match child capacity "
                f"{expected_size}"
            )
        self.kv_buffer = [None] * self.layer_num
        for layer_id, label in enumerate(self._layer_types):
            if label in STATE_LAYER_TYPES:
                conv_id = f"layer.{layer_id}.conv_state"
                recurrent_id = f"layer.{layer_id}.recurrent_state"
                try:
                    conv_dtype = self._state_field_dtypes[conv_id]
                    recurrent_dtype = self._state_field_dtypes[recurrent_id]
                except KeyError as exc:
                    raise ValueError(
                        f"cache state field {exc.args[0]!r} has no dtype"
                    ) from exc
                conv = self.field(
                    conv_id,
                    conv_dtype,
                )
                recurrent = self.field(
                    recurrent_id,
                    recurrent_dtype,
                )
                self._state_buffers_by_layer[layer_id] = (conv, recurrent)
                continue
            latent = self.field(f"layer.{layer_id}.latent_kv", self.store_dtype)
            page_elements = int(np.prod(latent.shape[1:]))
            if latent.stride(0) != page_elements:
                raise ValueError(
                    f"layer {layer_id} latent pages have padding between pages"
                )
            self.kv_buffer[layer_id] = latent.view(-1, 1, self.kv_cache_dim)

    @property
    def num_lcm_blocks(self) -> int:
        return self.plan.num_lcm_blocks

    @property
    def state_slabs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(self._state_buffers_by_layer.values())

    def group_id_for_layer(self, layer_id: int) -> str:
        try:
            return self._group_ids_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no cache group") from exc

    def get_component(self, layer_id: int, component_name: str) -> torch.Tensor:
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        if component_name == "latent_kv":
            buffer = self.kv_buffer[layer_id]
            if buffer is None:
                raise ValueError(f"layer {layer_id} has no MLA latent cache")
            return self.field(f"layer.{layer_id}.latent_kv", self.store_dtype)
        try:
            conv, recurrent = self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no KDA state") from exc
        if component_name == "conv_state":
            return conv
        if component_name == "recurrent_state":
            return recurrent
        raise ValueError(f"unknown KDA component {component_name!r}")

    def get_state_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no KDA state") from exc

    def zero_new_blocks(self, new_page_ids: dict[str, list[int]]) -> None:
        if new_page_ids:
            self.zero_blocks(new_page_ids)

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        assert self.buffer is not None
        self.buffer.zero_()

    def get_kv_size_bytes(self):
        assert self.buffer is not None
        return self.buffer.nbytes
