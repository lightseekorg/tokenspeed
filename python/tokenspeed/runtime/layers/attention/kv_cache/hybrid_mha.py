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

"""Paged MHA-history and recurrent-state cache."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from tokenspeed.runtime.layers.attention.kv_cache.mha import (
    MHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_LAYER_TYPES,
)


class HybridMHATokenToKVPool(MHATokenToKVPool):
    """MHA compute interface whose history and state share one buffer."""

    def __init__(
        self,
        *,
        memory_plan: CacheMemoryPlan,
        layer_group_ids: tuple[str, ...],
        state_field_dtypes: Mapping[str, torch.dtype] | None = None,
        **kwargs,
    ):
        group_ids = tuple(layer_group_ids)
        self._group_ids_by_layer = dict(enumerate(group_ids))
        self._state_field_dtypes = dict(state_field_dtypes or {})
        layer_types = tuple(kwargs.get("layer_types", ()))
        self._state_layer_ids = tuple(
            layer_id
            for layer_id, label in enumerate(layer_types)
            if label in STATE_LAYER_TYPES
        )
        self._state_buffers_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.paged_cache_requires_page_zeroing = True

        if len(group_ids) != kwargs["layer_num"]:
            raise ValueError("cache group ids must cover every model layer")

        super().__init__(memory_plan=memory_plan, layer_group_ids=group_ids, **kwargs)

    def _create_buffers(self) -> None:
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            self._bind_buffers()

    def _bind_buffers(self) -> None:
        if self.plan.logical_block_tokens != self.page_size:
            raise ValueError(
                f"cache plan P={self.plan.logical_block_tokens} does not match pool "
                f"page_size={self.page_size}"
            )
        max_packing = max(
            group.cache_blocks_per_lcm_block for group in self.plan.groups
        )
        expected_size = self.plan.num_lcm_blocks * max_packing * self.page_size
        if self.size != expected_size:
            raise ValueError(
                f"cache pool size {self.size} does not match plan child capacity "
                f"{expected_size}"
            )

        self.k_buffer = [None] * self.layer_num
        self.v_buffer = [None] * self.layer_num
        for layer_id, label in enumerate(self._layer_types):
            if label in STATE_LAYER_TYPES:
                conv_id = f"layer.{layer_id}.conv"
                ssm_id = f"layer.{layer_id}.ssm"
                try:
                    conv_dtype = self._state_field_dtypes[conv_id]
                    ssm_dtype = self._state_field_dtypes[ssm_id]
                except KeyError as exc:
                    raise ValueError(
                        f"cache state field {exc.args[0]!r} has no dtype"
                    ) from exc
                conv = self.field(conv_id, conv_dtype)
                ssm = self.field(ssm_id, ssm_dtype)
                if ssm.stride(0) != int(np.prod(ssm.shape[1:])):
                    raise ValueError(
                        f"state plane for layer {layer_id} has padding "
                        "between pages; the GDN decode ABI requires contiguous "
                        "state page rows"
                    )
                self._state_buffers_by_layer[layer_id] = (conv, ssm)
                continue

            k_pages = self.field(f"layer.{layer_id}.k", self.store_dtype)
            v_pages = self.field(f"layer.{layer_id}.v", self.store_dtype)
            contiguous_page_elements = int(np.prod(k_pages.shape[1:]))
            if (
                k_pages.stride(0) != contiguous_page_elements
                or v_pages.stride(0) != contiguous_page_elements
            ):
                raise ValueError(
                    f"history plane for layer {layer_id} has padding "
                    "between child pages; the attention ABI requires "
                    "contiguous flattened page rows"
                )
            self.k_buffer[layer_id] = k_pages.view(-1, self.head_num, self.head_dim)
            self.v_buffer[layer_id] = v_pages.view(-1, self.head_num, self.head_dim)

    @property
    def num_lcm_blocks(self) -> int:
        return self.plan.num_lcm_blocks

    @property
    def state_slabs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            self._state_buffers_by_layer[layer_id] for layer_id in self._state_layer_ids
        ]

    @property
    def supports_disaggregation(self) -> bool:
        return self._pd_disaggregation_enabled

    def get_pd_cache_contract(self):
        if not self.supports_disaggregation:
            raise RuntimeError("paged cache PD requires an enabled state MHA pool")
        return self.pd_contract(self.paged_cache_group_specs)

    def group_id_for_layer(self, layer_id: int) -> str:
        try:
            return self._group_ids_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no cache group") from exc

    def get_state_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id not in self._state_layer_ids:
            raise ValueError(f"layer {layer_id} is not a state layer")
        try:
            return self._state_buffers_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no bound state fields") from exc

    def get_component(self, layer_id: int, component_name: str) -> torch.Tensor:
        conv, recurrent = self.get_state_buffers(layer_id)
        if component_name == "conv_state":
            return conv
        if component_name == "recurrent_state":
            return recurrent
        raise ValueError(f"unknown state component {component_name!r}")

    def zero_new_pages(self, new_page_ids: dict[str, list[int]]) -> None:
        if new_page_ids:
            self.zero_blocks(new_page_ids)

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        assert self.buffer is not None
        self.buffer.zero_()

    def get_contiguous_buf_infos(self):
        raise RuntimeError("state MHA transfer uses get_pd_cache_contract()")


class HybridMHATokenToKVPoolMXFP8(
    HybridMHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
):
    def _create_buffers(self) -> None:
        if self.head_dim % self.MXFP8_SCALE_BLOCK_SIZE != 0:
            raise ValueError("MXFP8 head_dim must be divisible by 32")
        self.store_dtype = torch.float8_e4m3fn
        super()._create_buffers()
        self.k_scale_buffer = [
            self.field(f"layer.{layer_id}.k_scale", torch.float8_e8m0fnu)
            for layer_id in range(self.layer_num)
        ]
        self.v_scale_buffer = [
            self.field(f"layer.{layer_id}.v_scale", torch.float8_e8m0fnu)
            for layer_id in range(self.layer_num)
        ]

    def _layer_page_tokens(self, layer_id: int) -> int:
        return self.page_size
