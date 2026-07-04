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

import numpy as np
import torch
from tokenspeed_kernel.ops.kvcache.triton import store_kv_cache

from tokenspeed.runtime.configs.paged_cache_spec import hybrid_slab_group_size
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.utils import (
    copy_all_layer_kv_cache_tiled,
    move_kv_cache_native,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import debug_timing, get_colorful_logger
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


GB = 1024 * 1024 * 1024


class MHATokenToKVPool(BaseTokenToKVPool):
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        layer_types: tuple[str, ...] = (),
        sliding_window_tokens: int | None = None,
        max_scheduled_tokens: int = 0,
        speculative_enabled: bool = False,
        kvstore_enabled: bool = False,
        pd_disaggregation_enabled: bool = False,
        enable_kv_cache_copy: bool = False,
        enable_alt_stream: bool = True,
    ):
        super().__init__(
            size, dtype, device, max_batch_size, max_context_len, page_size, rank
        )

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        # Decide the buffer layout (M12 hybrid slab vs legacy per-layer)
        # BEFORE sizing: _get_page_size_bytes (next line) and
        # _create_buffers both key off _slab_group_size and must agree.
        # hybrid_slab_group_size is also what the registry's KV memory
        # profile consumes (_kv_profile_layer_divisor) -- one predicate,
        # two consumers, so sizing and layout can never diverge.
        self._layer_types = tuple(layer_types or ())
        self._kvstore_enabled = kvstore_enabled
        self._pd_disaggregation_enabled = pd_disaggregation_enabled
        self._slab_group_size = hybrid_slab_group_size(
            self._layer_types, speculative_enabled=speculative_enabled
        )
        self.page_size_bytes = self._get_page_size_bytes()
        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream()
            if torch.cuda.is_available() and enable_alt_stream
            else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            "KV Cache is allocated. K size: %.2f GB, V size: %.2f GB.",
            k_size / GB,
            v_size / GB,
        )

        from tokenspeed.runtime.configs.paged_cache_spec import (
            compute_paged_cache_group_page_counts,
            group_specs_from_layer_types,
            scheduler_ext_flat_kvcache,
        )

        # Publish per-group specs so the C++ scheduler is configured multi-group
        # for hybrid full/SWA models (gpt-oss). Empty layer_types -> single
        # full-history group, which the flat (TOKENSPEED_FLAT_KVCACHE) scheduler
        # build needs to allocate pages for plain non-hybrid MHA models.
        #
        # Rule: publish groups iff (a) the scheduler ext is flat-built AND
        # (b) speculative decoding is off. Publication is THE upstream signal:
        # the C++ scheduler config, the CUDA-graph flat capture path, and the
        # flat_block_tables bridge all key off it, so this single gate keeps
        # every downstream consumer consistent.
        # (a) Only a flat-built ext ever populates flat_block_tables; a
        #     radix-built ext (default) never does (MaybeFillFlatBlockTables is
        #     a no-op), so publishing against it would make CUDA-graph capture
        #     bind the flat per-group buffers to tables that never arrive and
        #     the replay guard (backends/mha.py) raise on the first graph
        #     decode. Older exts without the FLAT_KVCACHE attribute report
        #     False — the correct radix-safe default.
        # (b) Flat page tables do not support spec-expanded metadata
        #     (backends/mha.py asserts on flat_cache_group_ids with
        #     spec_num_tokens > 1), and non-empty groups turn off the overlap
        #     scheduler under spec decode
        #     (scheduler_utils.should_use_overlap_schedule), so spec runs keep
        #     the pre-group behavior: no specs, no flat capture kwarg, overlap
        #     schedule unaffected. TODO(flat+spec): publish under spec decode
        #     once the flat path supports spec-expanded metadata.
        if speculative_enabled or not scheduler_ext_flat_kvcache():
            self.paged_cache_group_specs = ()
            self.paged_cache_group_page_counts = {}
        else:
            effective_layer_types = layer_types or ("full_attention",)
            self.paged_cache_group_specs = tuple(
                group_specs_from_layer_types(
                    layer_types=effective_layer_types,
                    sliding_window_tokens=sliding_window_tokens,
                    page_size=page_size,
                )
            )
            # Per-group page budgets, required by pool_to_paged_cache_groups.
            # Mirrors DeepseekV4TokenToKVPool: size=total tokens,
            # max_batch_size=live reqs.
            self.paged_cache_group_page_counts = compute_paged_cache_group_page_counts(
                self.paged_cache_group_specs,
                max_live_requests=max_batch_size,
                max_scheduled_tokens=max(0, int(max_scheduled_tokens)),
                max_total_tokens=size,
                max_context_len=max_context_len,
            )

    def _get_page_size_bytes(self):
        # Under the hybrid slab layout (M12) paired layers share K/V
        # slabs, so a page only carries one group's layers worth of bytes
        # -- this is the byte-level capacity win the layout exists for.
        return (
            2
            * self.page_size
            * (self._slab_group_size or self.layer_num)
            * self.head_num
            * self.head_dim
            * torch._utils._element_size(self.dtype)
        )

    def _slab_pair_index(self) -> list[int]:
        """Map layer_id -> slab index for the hybrid slab layout.

        Groups form in FIRST-APPEARANCE order of layer_types -- the same
        ordering group_specs_from_layer_types uses, so slab i
        deterministically pairs group-A's i-th layer with group-B's i-th
        layer -- and the i-th layer of every group binds slab i.
        """
        assert self._slab_group_size is not None
        assert len(self._layer_types) == self.layer_num, (
            f"hybrid slab layout: layer_types has {len(self._layer_types)} "
            f"entries but layer_num={self.layer_num}"
        )
        occurrence: dict[str, int] = {}
        pair_index: list[int] = []
        for label in self._layer_types:
            idx = occurrence.get(label, 0)
            occurrence[label] = idx + 1
            pair_index.append(idx)
        # Pairing completeness: every group contributes exactly one layer
        # to each slab (equal group sizes are guaranteed by the predicate;
        # this pins the mapping itself).
        assert all(
            count == self._slab_group_size for count in occurrence.values()
        ), f"hybrid slab layout: uneven groups {occurrence!r}"
        return pair_index

    def _check_slab_guards(self):
        """Refuse features whose per-layer buffer assumptions break under
        the slab layout (paired layers alias the SAME tensor)."""
        if self._kvstore_enabled:
            raise RuntimeError(
                "hybrid slab KV layout is incompatible with the kvstore L2 "
                "cache: host offload copies KV per layer (get_cpu_copy / "
                "get_flat_data / transfer), and paired layers alias the "
                "same slab, so per-layer copies would double-count bytes "
                "and misattribute page ownership. Disable the kvstore "
                "(--enable-kvstore off) or use a radix-built "
                "tokenspeed_scheduler extension, which keeps the legacy "
                "per-layer layout."
            )
        if self._pd_disaggregation_enabled:
            raise RuntimeError(
                "hybrid slab KV layout is incompatible with PD "
                "disaggregation: KV transfer registers per-layer buffer "
                "pointers (get_contiguous_buf_infos), and paired layers "
                "alias the same slab, so per-layer transfers would send "
                "the same bytes twice and clobber the peer's pairing. Set "
                "disaggregation_mode='null' or use a radix-built "
                "tokenspeed_scheduler extension, which keeps the legacy "
                "per-layer layout."
            )

    def _create_buffers(self):
        with self.memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer.
            # The padded page 0 is used for writing dummy outputs from padded tokens.
            # Zero-init: attention kernels may read block_table entries beyond the
            # valid seq_len (pointing at page 0), so the slots must be finite to
            # keep softmax well-defined. Under the slab layout paired layers
            # share one dummy page 0 per slab -- writes there are discarded
            # either way, so the contract is unchanged.
            logger.info(
                "_create_buffers self.size=%r, self.page_size=%r, self.head_num=%r, self.head_dim=%r, self.layer_num=%r",
                self.size,
                self.page_size,
                self.head_num,
                self.head_dim,
                self.layer_num,
            )

            def _alloc():
                return torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )

            if self._slab_group_size is not None:
                # M12 hybrid slab layout: allocate group_size K/V slab
                # pairs and bind the i-th layer of EVERY group to slab i
                # (same tensor object, vLLM-style aliasing). Safety
                # invariant (do not re-derive here): the flat scheduler's
                # single BlockPool guarantees a page id is owned by at
                # most one group at a time, so paired layers' live rows
                # never overlap; the M11 write-within-group-table probes
                # cover it at runtime.
                self._check_slab_guards()
                pair_index = self._slab_pair_index()
                k_slabs = [_alloc() for _ in range(self._slab_group_size)]
                v_slabs = [_alloc() for _ in range(self._slab_group_size)]
                self.k_buffer = [
                    k_slabs[pair_index[layer_id]]
                    for layer_id in range(self.layer_num)
                ]
                self.v_buffer = [
                    v_slabs[pair_index[layer_id]]
                    for layer_id in range(self.layer_num)
                ]
                # Per-layer host (L2) copies would alias shared slabs, so
                # opt out of the hierarchical cache surface: event_loop
                # builds a MemoryExecutor for retraction offload even when
                # the kvstore flag is off, and this attribute is what
                # gates it (DeepseekV4TokenToKVPool precedent).
                self.supports_hierarchical_kv_cache = False
                # move_kv_cache broadcasts one (tgt, src) over data_ptrs;
                # duplicated slab entries would just re-copy the same rows
                # (idempotent, wasteful). It has NO callers anywhere today
                # (grep before wiring it up under the slab layout).
                logger.info(
                    "KV layout: hybrid slab (%d slabs x %d rows; paired "
                    "layers share storage; M12)",
                    self._slab_group_size,
                    self.size + self.page_size,
                )
            else:
                self.k_buffer = [_alloc() for _ in range(self.layer_num)]
                self.v_buffer = [_alloc() for _ in range(self.layer_num)]
                logger.info(
                    "KV layout: per-layer (%d buffers; hybrid slab "
                    "inactive: predicate returned None -- radix ext, "
                    "spec decode, or non-uniform/single-group "
                    "layer_types)",
                    self.layer_num,
                )
            self.k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.k_buffer],
                dtype=torch.uint64,
                device=self.device,
            )
            self.v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.v_buffer],
                dtype=torch.uint64,
                device=self.device,
            )
            self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
            self.data_strides = torch.tensor(
                [
                    np.prod(x.shape[1:]) * x.dtype.itemsize
                    for x in self.k_buffer + self.v_buffer
                ],
                device=self.device,
            )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        if hasattr(self, "k_data_ptrs"):
            del self.k_data_ptrs
        if hasattr(self, "v_data_ptrs"):
            del self.v_data_ptrs
        if hasattr(self, "data_ptrs"):
            del self.data_ptrs
        if hasattr(self, "data_strides"):
            del self.data_strides

    def _init_kv_copy_and_warmup(self):
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
        }

        dummy_loc = torch.zeros(1, dtype=torch.int32, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            1,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if self._kv_copy_config is None:
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
        else:
            grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                tgt_loc.numel(),
                tgt_loc.numel(),
                BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
                num_warps=self._kv_copy_config["num_warps"],
                num_stages=2,
            )

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        # Dedup by tensor identity: under the hybrid slab layout k_buffer
        # holds layer_num references to group_size slabs, and allocated
        # bytes must not be double-counted (legacy layout: no-op).
        k_size_bytes = 0
        for k_cache in {id(t): t for t in self.k_buffer}.values():
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in {id(t): t for t in self.v_buffer}.values():
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr() for i in range(self.layer_num)
        ] + [self._get_value_buffer(i).data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes for i in range(self.layer_num)
        ] + [self._get_value_buffer(i).nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_contiguous_buf_unit_lens(self):
        key_units = [
            self._get_key_buffer(i)[0, 0].nbytes for i in range(self.layer_num)
        ]
        value_units = [
            self._get_value_buffer(i)[0, 0].nbytes for i in range(self.layer_num)
        ]
        return key_units + value_units

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        return [
            [start_idx + i * self.layer_num + layer_id for i in range(2)]
            for layer_id in range(self.layer_num)
        ]

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), self.offload_chunk_page_num):
                chunk_indices = indices[i : i + self.offload_chunk_page_num]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.cuda.synchronize()
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), self.offload_chunk_page_num):
                chunk_indices = indices[i : i + self.offload_chunk_page_num]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][0],
                    kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        flatten = torch.stack(
            [
                torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
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
        store_kv_cache(
            cache_k, cache_v, self.k_buffer[layer_id], self.v_buffer[layer_id], loc
        )
