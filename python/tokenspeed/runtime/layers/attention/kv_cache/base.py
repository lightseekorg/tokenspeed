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

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.kvcache.triton import zero_byte_ranges

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    PagedCacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker

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
        prefix_granularity: int,
        rank: int,
        memory_plan: CacheMemoryPlan,
        *,
        paged_cache_group_specs: tuple[PagedCacheGroupSpec, ...] = (),
        token_capacity: int | None = None,
        backing_pool: CachePool | None = None,
        field_layer_offset: int = 0,
    ):
        self.dtype = dtype
        self.rank = rank
        self.size = size
        self.prefix_granularity = prefix_granularity
        # KV arena page span for paged consumers. Pinned 1:1 to the identity
        # grain at construction — this assignment is the single point of the
        # prefix-page <-> KV-page convention; geometry math must read this,
        # never prefix_granularity.
        self.kv_page_size = prefix_granularity
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            #  Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.plan = memory_plan
        self._field_layer_offset = int(field_layer_offset)
        if self._field_layer_offset < 0:
            raise ValueError("field_layer_offset must be non-negative")
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
        # the same sleep/wake lifetime as the former per-buffer allocations.
        #
        # A heterogeneous draft view (for example, an MHA Eagle3 head over an
        # MLA target) binds its own field family but must not allocate another
        # arena. Construction is deliberately target-first: the draft aliases
        # the target's already-registered buffer and field registry. The
        # registry also supplies the runtime dtypes for the PD contract.
        self._backing_pool = backing_pool
        if backing_pool is None:
            self.buffer: torch.Tensor | None = None
            self._fields: dict[str, torch.Tensor] = {}
        else:
            if backing_pool.plan != memory_plan:
                raise ValueError("a cache view must share its backing pool's plan")
            if backing_pool.buffer is None:
                raise ValueError(
                    "the backing cache pool must bind its fields before a view"
                )
            if paged_cache_group_specs:
                raise ValueError(
                    "a cache view must inherit, not republish, the runtime contract"
                )
            self.buffer = backing_pool.buffer
            self._fields = backing_pool._fields
            self.runtime_contract = backing_pool.runtime_contract
            self.paged_cache_group_specs = backing_pool.paged_cache_group_specs
            self.paged_cache_group_page_counts = (
                backing_pool.paged_cache_group_page_counts
            )

        # default state for optional layer-wise transfer control
        self.layerwise_load_tracker = None
        logger.info(
            f"Initialized token to kv pool with size {size}, dtype {dtype}, device {device}, prefix granularity {prefix_granularity}, rank {rank}"
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
            # The identity axis comes from the plan, never read back out of
            # pool state. The pool's prefix_granularity scalar is pinned to
            # it at construction; per-group CacheBlock spans live in the
            # group specs as block_granularity.
            prefix_granularity=self.plan.prefix_granularity,
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

    def zero_blocks(self, block_ids_by_group: dict[str, list[int]]) -> None:
        """Clear selected CacheBlocks without interpreting their field types."""
        buffer = self._ensure_buffer()
        segments = [
            segment
            for group_id, block_ids in block_ids_by_group.items()
            for segment in self._block_byte_segments(group_id, block_ids)
        ]
        if segments:
            zero_byte_ranges(buffer, segments)

    @property
    def supports_disaggregation(self) -> bool:
        """Whether this pool exposes complete inputs for cache transfer."""
        return bool(self.paged_cache_group_specs) and all(
            spec.transfer_policy is not None for spec in self.paged_cache_group_specs
        )

    def _ensure_buffer(self) -> torch.Tensor:
        if self._backing_pool is not None:
            buffer = self._backing_pool.buffer
            if buffer is None:
                raise RuntimeError("the backing cache pool released its buffer")
            self.buffer = buffer
            return buffer
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.plan.arena_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
        return self.buffer

    def _field_block_byte_offset(self, field_id: str, block_id: int) -> int:
        return self.plan.field_page_byte_offset(field_id, block_id)

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

    def register_layerwise_load_tracker(
        self, layerwise_load_tracker: LayerwiseLoadTracker
    ) -> None:
        self.layerwise_load_tracker = layerwise_load_tracker

    def bind_paged_cache_scheduler(self, scheduler: object) -> None:
        """Optional hook for model-specific paged-cache diagnostics."""

    def cache_transfer_layout(self):
        """Return the transfer layout consumed by this compute view."""
        from tokenspeed.runtime.cache.transfer.layout import (
            select_layer_fields,
        )

        try:
            field_ids, consumers = select_layer_fields(
                self.plan.fields,
                first_layer=self._field_layer_offset,
                num_layers=self.layer_num,
            )
        except (AttributeError, IndexError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        return self._build_cache_transfer_layout(field_ids, consumers)

    def cache_contract_binding(self) -> tuple[torch.Tensor, dict[str, str]]:
        """Return the arena and field dtypes bound to the cache contract."""
        if not self.supports_disaggregation:
            raise RuntimeError(
                "cache pool has no transfer policy in its runtime contract"
            )
        field_ids = {field.field_id for field in self.plan.fields}
        missing = sorted(field_ids - set(self._fields))
        if missing:
            raise RuntimeError(f"cache fields have no runtime dtype: {missing}")
        return self._ensure_buffer(), {
            field_id: str(self._fields[field_id].dtype).removeprefix("torch.")
            for field_id in field_ids
        }

    def _build_cache_transfer_layout(self, field_ids, consumers):
        from tokenspeed.runtime.cache.transfer.layout import layout_from_lcm_plan

        missing = sorted(set(field_ids) - set(self._fields))
        if missing:
            raise RuntimeError(f"cache fields have no runtime dtype: {missing}")
        local_group_ids = {
            field.group_id for field in self.plan.fields if field.field_id in field_ids
        }
        scheduler_group_ids = tuple(
            spec.group_id
            for spec in self.paged_cache_group_specs
            if spec.group_id in local_group_ids
        )
        return layout_from_lcm_plan(
            self.plan,
            self._ensure_buffer(),
            consumers=consumers,
            group_ids=scheduler_group_ids or None,
            field_ids=field_ids,
        )

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        """Zero the shared cache buffer after sleep/wake remaps its storage."""
        # The event loop visits both target and draft pools. A draft view owns
        # no allocation; the target clears their shared arena exactly once.
        if self._backing_pool is not None:
            return
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


class LayerMappedKVPool:
    """Wraps a KV pool to map the caller's layer IDs to inner pool indices.

    Two callers, one mechanism — a dict from the id a model layer carries to
    the index its plane occupies in the wrapped pool:

    - Hybrid models: layers carry global sparse ids (e.g. 3, 7, 11) while the
      inner pool holds compact full-attention planes (0, 1, 2); the map is
      ``{global_id: pool_idx}`` (the default built from ``layer_ids``).
    - Draft views of the ONE merged pool: draft layers carry LOCAL ids
      (0..n-1) while their planes are the continuation range
      ``num_target_layers..``; the map is ``{local: global}`` (pass
      ``layer_map`` explicitly).

    Deliberately not an ``MLATokenToKVPool`` subclass: ``models/utils.py``'s
    fused MLA write gate uses ``isinstance`` to keep off this wrapper,
    because ``set_mla_kv_buffer`` here defaults to ``sanitize=True`` (see that
    method) and the fused write has no sanitize path. Losing this property --
    by subclassing, or by the gate switching to duck-typing -- reopens the
    padded-row NaN hazard that default exists to close.
    """

    def __init__(
        self,
        inner_pool,
        full_attention_layer_ids: list[int],
        *,
        layer_map: dict[int, int] | None = None,
    ):
        self.inner = inner_pool
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_map = (
            dict(layer_map)
            if layer_map is not None
            else {
                global_id: pool_idx
                for pool_idx, global_id in enumerate(full_attention_layer_ids)
            }
        )
        # Expose the identity grain for the scheduler and the KV page span
        # for paged consumers.
        self.prefix_granularity = getattr(inner_pool, "prefix_granularity", 1)
        self.kv_page_size = getattr(inner_pool, "kv_page_size", 1)

    def _map(self, layer_id: int) -> int:
        return self.layer_map.get(layer_id, layer_id)

    @contextmanager
    def _mapped(self, layer):
        """Temporarily remap ``layer.layer_id`` to its inner-pool slot."""
        orig = layer.layer_id
        layer.layer_id = self._map(orig)
        try:
            yield
        finally:
            layer.layer_id = orig

    def set_kv_buffer(
        self,
        layer,
        out_cache_loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ):
        with self._mapped(layer):
            self.inner.set_kv_buffer(layer, out_cache_loc, k, v, k_scale, v_scale)

    def get_kv_buffer(self, layer_id: int):
        return self.inner.get_kv_buffer(self._map(layer_id))

    def get_key_buffer(self, layer_id: int):
        return self.inner.get_key_buffer(self._map(layer_id))

    def get_value_buffer(self, layer_id: int):
        return self.inner.get_value_buffer(self._map(layer_id))

    # MLA pools index their per-layer kv_buffer by ``layer.layer_id`` directly.
    # In a hybrid model the inner MLA pool only holds the full-attention layers,
    # so the global id must be mapped to its pool slot first (mirrors
    # ``set_kv_buffer``). Reached via the DeepseekV3-style MLA chunked-prefill
    # path (Kimi-K3).
    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope, sanitize=True):
        # Prefill breakable-graph padding contract: the dummy-batch capture (and
        # bucket-padding rows) whose ``out_cache_loc`` is the reserved
        # ``dummy_kv_slot`` can carry NaN into this fp8 KV write. The paged MLA
        # decode kernel reads that shared dummy slot through the zero-padded
        # block-table entries and computes ``q·k`` BEFORE applying the causal
        # mask, so the NaN survives it (``NaN + -inf = NaN``) and poisons a live
        # row's softmax -> NaN logits -> token 0. Eager prefill leaves the dummy
        # slot finite (``q·0`` masks cleanly), which is why the bug only appears
        # with the prefill graph on. Ask the cache writer to sanitize in-kernel
        # so real rows stay bitwise unchanged without allocating two temporary
        # tensors or launching two separate nan_to_num kernels.
        with self._mapped(layer):
            self.inner.set_mla_kv_buffer(
                layer,
                loc,
                cache_k_nope,
                cache_k_rope,
                sanitize=sanitize,
            )

    def get_mla_kv_buffer(self, layer, loc, dst_dtype=None):
        with self._mapped(layer):
            return self.inner.get_mla_kv_buffer(layer, loc, dst_dtype)

    # DSA/MSA index-key planes are layer-indexed like the KV planes; a draft
    # view must map its local layer ids onto the continuation range here too
    # (an unmapped pass-through would read/write the target's planes).
    def get_index_k_buffer(self, layer_id: int):
        return self.inner.get_index_k_buffer(self._map(layer_id))

    def set_index_k_buffer(self, layer_id: int, loc, index_k):
        self.inner.set_index_k_buffer(self._map(layer_id), loc, index_k)

    def gather_index_k(self, layer_id: int, slots):
        return self.inner.gather_index_k(self._map(layer_id), slots)

    def kvconv_checkpoint_buffers(self, layer_id: int):
        return self.inner.kvconv_checkpoint_buffers(self._map(layer_id))

    def hiddenconv_checkpoint_buffer(self, layer_id: int, component: str):
        return self.inner.hiddenconv_checkpoint_buffer(self._map(layer_id), component)

    def __getattr__(self, name):
        return getattr(self.inner, name)
