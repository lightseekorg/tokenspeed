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

"""MLA interfaces backed by one LCM cache arena."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch

from tokenspeed.runtime.configs import paged_cache_spec
from tokenspeed.runtime.configs.cache_runtime import (
    PagedCacheRuntimeContract,
)
from tokenspeed.runtime.configs.lcm_memory_plan import LcmMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.lcm import LcmCachePool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool


class LcmMLATokenToKVPool(MLATokenToKVPool):
    """MLA compute interface whose latent KV and KDA state share one arena."""

    def __init__(
        self,
        *,
        memory_plan: LcmMemoryPlan,
        layer_group_ids: tuple[str, ...],
        layer_types: tuple[str, ...],
        max_scheduled_tokens: int = 0,
        pd_disaggregation_enabled: bool = False,
        state_field_dtypes: Mapping[str, torch.dtype] | None = None,
        token_capacity: int | None = None,
        **kwargs,
    ):
        self._lcm_memory_plan = memory_plan
        self._layer_types = tuple(layer_types)
        self.layer_cache_group_ids = tuple(layer_group_ids)
        self._group_ids_by_layer = dict(enumerate(self.layer_cache_group_ids))
        self._token_capacity = (
            token_capacity if token_capacity is not None else kwargs["size"]
        )
        self._pd_disaggregation_enabled = pd_disaggregation_enabled
        self._state_field_dtypes = dict(state_field_dtypes or {})
        self.lcm_pool: LcmCachePool | None = None
        self._state_buffers_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.paged_cache_requires_page_zeroing = True

        layer_num = kwargs["layer_num"]
        if len(self._layer_types) != layer_num:
            raise ValueError("LCM layer types must cover every model layer")
        if len(self.layer_cache_group_ids) != layer_num:
            raise ValueError("LCM cache group ids must cover every model layer")

        max_live_requests = kwargs["max_batch_size"]
        max_context_len = kwargs["max_context_len"]
        super().__init__(**kwargs)
        published = paged_cache_spec.publish_paged_cache_groups(
            layer_types=self._layer_types,
            group_ids=self.layer_cache_group_ids,
            sliding_window_tokens=None,
            page_size=self.page_size,
            cache_blocks_per_lcm_block={
                group.group_id: group.cache_blocks_per_lcm_block
                for group in memory_plan.groups
            },
            max_live_requests=max_live_requests,
            max_scheduled_tokens=max_scheduled_tokens,
            max_total_tokens=self.size,
            max_context_len=max_context_len,
        )
        if published is None:
            raise RuntimeError("MLA LCM cache requires cache-group scheduling")
        specs, counts = published
        if self._pd_disaggregation_enabled:
            specs = [
                replace(
                    spec,
                    transfer_policy=(
                        "latest_snapshot" if spec.family == "state" else "full_suffix"
                    ),
                )
                for spec in specs
            ]
        counts.update(
            {group.group_id: group.page_count for group in memory_plan.groups}
        )
        self.paged_cache_group_specs = tuple(specs)
        self.paged_cache_group_page_counts = counts
        self.runtime_contract = PagedCacheRuntimeContract(
            block_size=self.page_size,
            num_lcm_blocks=memory_plan.num_lcm_blocks,
            token_capacity=self._token_capacity,
            group_specs=self.paged_cache_group_specs,
            group_page_counts=self.paged_cache_group_page_counts,
        )

    def _create_buffers(self) -> None:
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            self._create_lcm_buffers()

    def _create_lcm_buffers(self) -> None:
        plan = self._lcm_memory_plan
        if self.quant_method == "per_token_head":
            raise ValueError("MLA LCM cache does not support per-token-head KV")
        if plan.logical_block_tokens != self.page_size:
            raise ValueError(
                f"LCM plan P={plan.logical_block_tokens} does not match "
                f"pool page_size={self.page_size}"
            )
        max_packing = max(group.cache_blocks_per_lcm_block for group in plan.groups)
        expected_size = plan.num_lcm_blocks * max_packing * self.page_size
        if self.size != expected_size:
            raise ValueError(
                f"LCM pool size {self.size} does not match child capacity "
                f"{expected_size}"
            )
        self.lcm_pool = LcmCachePool(plan, self.device)
        self.kv_buffer = [None] * self.layer_num
        for layer_id, label in enumerate(self._layer_types):
            if label in paged_cache_spec.STATE_LAYER_TYPES:
                conv_id = f"layer.{layer_id}.conv_state"
                recurrent_id = f"layer.{layer_id}.recurrent_state"
                try:
                    conv_dtype = self._state_field_dtypes[conv_id]
                    recurrent_dtype = self._state_field_dtypes[recurrent_id]
                except KeyError as exc:
                    raise ValueError(
                        f"LCM state field {exc.args[0]!r} has no dtype"
                    ) from exc
                conv = self.lcm_pool.field(
                    conv_id,
                    conv_dtype,
                )
                recurrent = self.lcm_pool.field(
                    recurrent_id,
                    recurrent_dtype,
                )
                self._state_buffers_by_layer[layer_id] = (conv, recurrent)
                continue
            latent = self.lcm_pool.field(
                f"layer.{layer_id}.latent_kv", self.store_dtype
            )
            page_elements = int(np.prod(latent.shape[1:]))
            if latent.stride(0) != page_elements:
                raise ValueError(
                    f"layer {layer_id} latent pages have padding between pages"
                )
            self.kv_buffer[layer_id] = latent.view(-1, 1, self.kv_cache_dim)
        self.supports_hierarchical_kv_cache = False

    @property
    def num_lcm_blocks(self) -> int:
        return self._lcm_memory_plan.num_lcm_blocks

    @property
    def state_slabs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(self._state_buffers_by_layer.values())

    @property
    def supports_disaggregation(self) -> bool:
        return self._pd_disaggregation_enabled

    def get_pd_cache_contract(self):
        if not self.supports_disaggregation or self.lcm_pool is None:
            raise RuntimeError("Paged cache PD requires an enabled MLA LCM pool")
        return self.lcm_pool.pd_contract(self.paged_cache_group_specs)

    def cache_transfer_layout(self):
        if self.lcm_pool is None:
            raise RuntimeError("LCM backing is not allocated")
        consumers = tuple(
            tuple(
                field.field_id
                for field in self.lcm_pool.plan.fields
                if field.field_id.startswith(f"layer.{layer_id}.")
            )
            for layer_id in range(self.layer_num)
        )
        return self.lcm_pool.transfer_layout(consumers)

    def group_id_for_layer(self, layer_id: int) -> str:
        try:
            return self._group_ids_by_layer[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} has no cache group") from exc

    def get_component(self, layer_id: int, component_name: str) -> torch.Tensor:
        if component_name == "latent_kv":
            buffer = self.kv_buffer[layer_id]
            if buffer is None or self.lcm_pool is None:
                raise ValueError(f"layer {layer_id} has no MLA latent cache")
            return self.lcm_pool.field(f"layer.{layer_id}.latent_kv", self.store_dtype)
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

    def zero_new_pages(self, new_page_ids: dict[str, list[int]]) -> None:
        if new_page_ids:
            assert self.lcm_pool is not None
            self.lcm_pool.zero_pages(new_page_ids)

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        assert self.lcm_pool is not None
        self.lcm_pool.backing.zero_()

    def get_kv_size_bytes(self):
        assert self.lcm_pool is not None
        return self.lcm_pool.backing.nbytes

    def get_contiguous_buf_infos(self):
        raise RuntimeError("MLA LCM transfer uses get_pd_cache_contract()")

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        raise RuntimeError("MLA LCM transfer uses get_pd_cache_contract()")

    def get_cpu_copy(self, token_indices: list[int]) -> torch.Tensor:
        raise RuntimeError("MLA LCM cache does not use hierarchical copy")

    def load_cpu_copy(
        self, kv_cache_cpu: torch.Tensor, token_indices: list[int]
    ) -> None:
        raise RuntimeError("MLA LCM cache does not use hierarchical copy")
