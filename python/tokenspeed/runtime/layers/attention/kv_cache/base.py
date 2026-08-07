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
from tokenspeed_kernel.ops.kvcache.triton import zero_byte_segments

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    PagedCacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.page_table import expand_page_table
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter

logger = get_colorful_logger(__name__)


class CachePool:
    """Own page-backed cache memory and expose backend-specific views."""

    # Pools that alias recurrent-state bytes and KV in one buffer must
    # zero physical pages on reuse to avoid poisoned tails. Pure-attention
    # pools do not alias state, so reused pages need no sanitization.
    paged_cache_requires_page_zeroing: bool = False

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        page_size: int,
        rank: int,
        memory_plan: CacheMemoryPlan,
        *,
        paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...] = (),
        token_capacity: int | None = None,
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
        self.plan = memory_plan
        # The cache recipe is the single source of the scheduler group specs
        # (CachePoolSpec.paged_cache_group_specs); the pool aligns their
        # physical fields with the memory plan and publishes the runtime
        # contract from the pair. Pools constructed without specs (tests)
        # publish no contract.
        self.runtime_contract: PagedCacheRuntimeContract | None = None
        self.paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...] = ()
        self.paged_cache_group_page_counts: dict[str, int] = {}
        if paged_cache_group_specs:
            self._publish_runtime_contract(
                paged_cache_group_specs,
                token_capacity if token_capacity is not None else size,
            )
        # Allocate lazily when the first field is bound. Concrete pools do
        # that inside their memory-saver region, so the shared buffer keeps
        # the same sleep/wake lifetime as the legacy per-buffer allocations.
        self.buffer: torch.Tensor | None = None
        self._fields: dict[str, torch.Tensor] = {}

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None
        logger.info(
            f"Initialized token to kv pool with size {size}, dtype {dtype}, device {device}, page size {page_size}, rank {rank}"
        )

    def _publish_runtime_contract(
        self,
        group_specs: tuple[PagedCacheGroupSpec, ...],
        token_capacity: int,
    ) -> None:
        """Align recipe group specs with the memory plan and publish the
        scheduler contract. The plan is the source of truth for per-group
        packing and page counts, so every spec group must be planned."""
        from dataclasses import replace

        plan_groups = {group.group_id: group for group in self.plan.groups}
        aligned = []
        counts: dict[str, int] = {}
        for spec in group_specs:
            if spec.group_id in counts:
                raise ValueError(
                    f"cache group {spec.group_id!r} is published more than once"
                )
            group = plan_groups.get(spec.group_id)
            if group is None:
                raise ValueError(
                    f"cache group {spec.group_id!r} has no planned fields; "
                    "every published group must appear in the memory plan"
                )
            aligned.append(
                replace(
                    spec,
                    cache_blocks_per_lcm_block=group.cache_blocks_per_lcm_block,
                )
            )
            counts[spec.group_id] = group.page_count
        self.paged_cache_group_specs = tuple(aligned)
        self.paged_cache_group_page_counts = counts
        self.runtime_contract = PagedCacheRuntimeContract(
            block_size=self.page_size,
            num_lcm_blocks=self.plan.num_lcm_blocks,
            token_capacity=token_capacity,
            group_specs=self.paged_cache_group_specs,
            group_page_counts=counts,
        )

    def field(self, field_id: str, dtype: torch.dtype) -> torch.Tensor:
        """Return one typed field view into the shared cache buffer."""
        buffer = self._ensure_buffer()
        view = self._fields.get(field_id)
        if view is not None:
            if view.dtype != dtype:
                raise ValueError(
                    f"cache field {field_id!r} is already bound as {view.dtype}"
                )
            return view
        try:
            field = self.plan.field(field_id)
        except KeyError as exc:
            raise ValueError(f"cache field {field_id!r} is not planned") from exc
        if torch.empty((), dtype=dtype).element_size() != field.element_size:
            raise ValueError(f"field {field_id!r}: dtype itemsize does not match plan")
        group = self.plan.group(field.group_id)
        element_strides = []
        stride = 1
        for extent in reversed(field.shape):
            element_strides.append(stride)
            stride *= extent
        view = buffer.view(dtype).as_strided(
            (group.page_count, *field.shape),
            (
                field.page_stride_bytes // field.element_size,
                *reversed(element_strides),
            ),
            self._field_block_byte_offset(field_id, 0) // field.element_size,
        )
        self._fields[field_id] = view
        return view

    def expand_block_table(
        self,
        group_id: str | None,
        block_table: torch.Tensor,
        *,
        kernel_block_tokens: int,
        max_kernel_blocks: int | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map scheduler CacheBlock IDs to the blocks consumed by a kernel."""
        logical_block_tokens = self._scheduler_block_tokens(group_id)
        return expand_page_table(
            block_table,
            logical_page_size=logical_block_tokens,
            kernel_page_size=kernel_block_tokens,
            max_kernel_pages=max_kernel_blocks,
            out=out,
        )

    def _scheduler_block_tokens(self, group_id: str | None) -> int:
        if group_id is None:
            history_specs = tuple(
                spec
                for spec in self.paged_cache_group_specs
                if spec.family == "history"
            )
            if len(history_specs) != 1:
                raise ValueError(
                    "cache pool must publish exactly one history group when "
                    "the backend does not name a group"
                )
            return history_specs[0].cache_block_tokens
        for spec in self.paged_cache_group_specs:
            if spec.group_id == group_id:
                return spec.cache_block_tokens
        self.plan.group(group_id)
        return self.plan.logical_block_tokens

    def zero_blocks(self, block_ids_by_group: dict[str, list[int]]) -> None:
        """Clear selected CacheBlocks without interpreting their field types."""
        buffer = self._ensure_buffer()
        segments = [
            segment
            for group_id, block_ids in block_ids_by_group.items()
            for segment in self._block_byte_segments(group_id, block_ids)
        ]
        if segments:
            zero_byte_segments(buffer, segments)

    def pd_contract(self, group_specs):
        buffer = self._ensure_buffer()
        from tokenspeed.runtime.pd.cache_protocol import build_lcm_pd_cache_contract

        missing = [
            field.field_id
            for field in self.plan.fields
            if field.field_id not in self._fields
        ]
        if missing:
            raise RuntimeError(f"cache fields have no runtime dtype: {missing}")
        field_dtypes = {
            field_id: str(view.dtype).removeprefix("torch.")
            for field_id, view in self._fields.items()
        }
        return build_lcm_pd_cache_contract(
            plan=self.plan,
            buffer=buffer,
            group_specs=group_specs,
            field_dtypes=field_dtypes,
        )

    def _ensure_buffer(self) -> torch.Tensor:
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.plan.arena_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
        return self.buffer

    def _field_block_byte_offset(self, field_id: str, block_id: int) -> int:
        field = self.plan.field(field_id)
        group = self.plan.group(field.group_id)
        if block_id < 0 or block_id >= group.page_count:
            raise IndexError(
                f"block_id {block_id} outside [0, {group.page_count}) for "
                f"group {group.group_id!r}"
            )
        plane = self.plan.plane(field.plane_id)
        return (
            plane.arena_offset_bytes
            + plane.bytes_per_lcm_block
            - field.page_stride_bytes
            + block_id * field.page_stride_bytes
            + field.field_offset_bytes
        )

    def _block_byte_segments(
        self, group_id: str, block_ids: list[int]
    ) -> list[tuple[int, int]]:
        self.plan.group(group_id)
        fields = [field for field in self.plan.fields if field.group_id == group_id]
        return [
            (
                self._field_block_byte_offset(field.field_id, block_id),
                field.payload_bytes,
            )
            for block_id in block_ids
            for field in fields
        ]

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def bind_paged_cache_scheduler(self, scheduler: object) -> None:
        """Optional hook for model-specific paged-cache diagnostics."""

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        """Zero the shared cache buffer after sleep/wake remaps its storage."""
        if self.buffer is not None:
            self.buffer.zero_()

    def maybe_log_paged_cache_group_pages(self) -> None:
        return None

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

    # Buffer metadata used by prefill/decode disaggregation.
    def get_contiguous_buf_infos(self):
        raise NotImplementedError()

    def get_contiguous_buf_unit_lens(self):
        return [1] * len(self.get_contiguous_buf_infos()[2])

    # Layerwise buffer offsets used by prefill/decode disaggregation.
    def get_layerwise_buf_info_offsets(self, start_idx=0):
        raise NotImplementedError()
