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

from collections import Counter

import numpy as np
import torch
from tokenspeed_kernel.ops.kvcache.triton import store_kv_cache

from tokenspeed.runtime.configs import paged_cache_spec
from tokenspeed.runtime.configs.flat_memory_plan import occurrence_index
from tokenspeed.runtime.configs.paged_cache_spec import hybrid_slab_group_size
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.flat_state_slabs import (
    FlatStateSlabs,
)
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
        sliding_window_tokens: int | tuple[int | None, ...] | None = None,
        max_scheduled_tokens: int = 0,
        pd_disaggregation_enabled: bool = False,
        enable_kv_cache_copy: bool = False,
        enable_alt_stream: bool = True,
        conv_state_shape: tuple[int, ...] | None = None,
        temporal_state_shape: tuple[int, ...] | None = None,
        conv_dtype: torch.dtype | None = None,
        ssm_dtype: torch.dtype | None = None,
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
        self._layer_types = tuple(layer_types or ())
        self._pd_disaggregation_enabled = pd_disaggregation_enabled
        self._slab_group_size = hybrid_slab_group_size(
            self._layer_types,
            sliding_window_tokens=sliding_window_tokens,
        )
        # GDN/mamba2 recurrent state slabs live under this same pool object
        # (one page-id space with the KV pages), but their bookkeeping is
        # owned by FlatStateSlabs. Constructing it here runs the
        # equalization pre-check (same trigger, same ValueError) before any
        # buffer allocation; slabs themselves are allocated in
        # _create_buffers inside the memory-saver region.
        self._state = FlatStateSlabs(
            layer_types=self._layer_types,
            conv_state_shape=conv_state_shape,
            temporal_state_shape=temporal_state_shape,
            conv_dtype=conv_dtype,
            ssm_dtype=ssm_dtype,
            default_dtype=dtype,
            page_size=self.page_size,
            size=self.size,
            kv_bytes_per_slot=2 * head_num * head_dim * self.store_dtype.itemsize,
        )
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

        # Publication rule lives in paged_cache_spec.publish_paged_cache_groups
        # (module-attr call so tests can patch the flat-ext probe at call time).
        published = paged_cache_spec.publish_paged_cache_groups(
            layer_types=self._layer_types,
            sliding_window_tokens=sliding_window_tokens,
            page_size=page_size,
            max_live_requests=max_batch_size,
            max_scheduled_tokens=max_scheduled_tokens,
            max_total_tokens=size,
            max_context_len=max_context_len,
        )
        if published is None:
            self.paged_cache_group_specs = ()
            self.paged_cache_group_page_counts = {}
        else:
            specs, counts = published
            self.paged_cache_group_specs = tuple(specs)
            self.paged_cache_group_page_counts = counts
        # Slab aliasing is only safe under the single-BlockPool ownership the
        # published groups configure.
        assert self._slab_group_size is None or self.paged_cache_group_specs

    def _slab_pair_index(self) -> list[int]:
        """Map layer_id -> slab index: the i-th layer of every group binds
        slab i (first-appearance order, as in group_specs_from_layer_types).
        """
        assert self._slab_group_size is not None
        assert len(self._layer_types) == self.layer_num, (
            f"hybrid slab layout: layer_types has {len(self._layer_types)} "
            f"entries but layer_num={self.layer_num}"
        )
        counts = Counter(self._layer_types)
        assert all(
            count == self._slab_group_size for count in counts.values()
        ), f"hybrid slab layout: uneven groups {dict(counts)!r}"
        return occurrence_index(self._layer_types)

    def _check_slab_guards(self):
        """Refuse features whose per-layer buffer assumptions break when
        paired layers alias the same slab tensor."""
        # kvstore is allowed (spec §6 revision): the flat L2 tier mirrors
        # whole slabs byte-blind, so per-slab copies are group-safe.
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
        # Tag as "kv_cache", no CPU backup: KV is discarded on sleep and rebuilt
        # after wake (paging overwrites; clear_kv_buffers zeros the remapped pages).
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            # Page 0 is the zero-initialized dummy page: padded tokens write
            # there, and kernels may read it past valid seq_len, so its slots
            # must stay finite to keep softmax well-defined.
            def _alloc():
                return torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )

            # State-layer bookkeeping lives in FlatStateSlabs. The KV skip
            # set below (which layers carry None KV) and the state-slab
            # allocation are gated by the SAME flat-GDN predicate -- the plan
            # sizing (registry) charges exactly full-layer KV + state rows,
            # so the two decisions must never diverge. state_layer_ids is
            # empty unless the gate is on, so non-flat profiles keep full KV.
            flat_state_layers = set(self._state.state_layer_ids)
            if self._state.is_active:
                # Gates event_loop's retraction offload: state layers carry no
                # per-layer KV, so the radix offload executor (and its host
                # pool, sized for ALL layers) cannot represent this pool.
                self.supports_hierarchical_kv_cache = False

            if self._slab_group_size is not None:
                # Paired layers alias the same slab tensor; live rows never
                # overlap (page-ownership contract in hybrid_slab_group_size).
                self._check_slab_guards()
                pair_index = self._slab_pair_index()
                k_slabs = [_alloc() for _ in range(self._slab_group_size)]
                v_slabs = [_alloc() for _ in range(self._slab_group_size)]
                self.k_buffer = [
                    k_slabs[pair_index[layer_id]] for layer_id in range(self.layer_num)
                ]
                self.v_buffer = [
                    v_slabs[pair_index[layer_id]] for layer_id in range(self.layer_num)
                ]
                # Gates event_loop's retraction offload (built even with the
                # kvstore off): per-layer host copies would alias shared slabs.
                self.supports_hierarchical_kv_cache = False
                logger.info(
                    "KV layout: hybrid slab (%d slabs x %d rows; paired "
                    "layers share storage; M12)",
                    self._slab_group_size,
                    self.size + self.page_size,
                )
            else:
                # The hybrid-slab branch above never sees state labels
                # (hybrid_slab_group_size excludes them), so the skip set
                # only applies here.
                self.k_buffer = [
                    None if layer_id in flat_state_layers else _alloc()
                    for layer_id in range(self.layer_num)
                ]
                self.v_buffer = [
                    None if layer_id in flat_state_layers else _alloc()
                    for layer_id in range(self.layer_num)
                ]
                if flat_state_layers:
                    logger.info(
                        "KV layout: per-layer (%d of %d layers carry KV "
                        "buffers; state layers carry none)",
                        self.layer_num - len(flat_state_layers),
                        self.layer_num,
                    )
                else:
                    logger.info(
                        "KV layout: per-layer (%d buffers; hybrid slab "
                        "inactive: predicate returned None -- radix ext "
                        "or non-uniform/single-group layer_types)",
                        self.layer_num,
                    )
            # Pointer/stride tables carry the REAL tensors only: _kv_copy
            # launches one block per data_ptrs entry (grid = numel), so a
            # placeholder entry for a skipped state layer would be
            # dereferenced.
            real_k = [x for x in self.k_buffer if x is not None]
            real_v = [x for x in self.v_buffer if x is not None]
            self.k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in real_k],
                dtype=torch.uint64,
                device=self.device,
            )
            self.v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in real_v],
                dtype=torch.uint64,
                device=self.device,
            )
            self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
            self.data_strides = torch.tensor(
                [np.prod(x.shape[1:]) * x.dtype.itemsize for x in real_k + real_v],
                device=self.device,
            )

            # State slabs (GDN/mamba2 conv+ssm rows) share this pool's
            # memory-saver region so they follow the KV discard-on-sleep
            # policy. FlatStateSlabs.allocate is a no-op (leaving
            # state_slabs == []) unless the flat-GDN gate is on.
            self._state.allocate(self.device)

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
        # Slab layout: data_ptrs holds duplicated slab entries, so this
        # broadcast re-copies rows. No callers today; re-check before wiring.
        if self._kv_copy_config is None:
            # Real tensors only: flat GDN state layers carry None slots.
            move_kv_cache_native(
                [x for x in self.k_buffer if x is not None],
                [x for x in self.v_buffer if x is not None],
                tgt_loc,
                src_loc,
            )
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
        # Dedup by tensor identity: the slab layout aliases layers to shared
        # slabs, and allocated bytes must not be double-counted. None slots
        # (flat GDN state layers carry no KV) are skipped.
        k_size_bytes = 0
        for k_cache in {id(t): t for t in self.k_buffer if t is not None}.values():
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in {id(t): t for t in self.v_buffer if t is not None}.values():
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        if any(x is None for x in self.k_buffer):
            raise ValueError(
                "flat GDN layout has no per-layer KV on state layers; "
                "PD disaggregation unsupported: KV transfer registers "
                "per-layer buffer pointers, and state layers carry only "
                "state slabs. Set disaggregation_mode='null' or use a "
                "radix-built tokenspeed_scheduler extension, which keeps "
                "the full per-layer KV layout."
            )
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
        buf = self.k_buffer[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV buffer")
        if self.store_dtype != self.dtype:
            return buf.view(self.dtype)
        return buf

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        buf = self.v_buffer[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV buffer")
        if self.store_dtype != self.dtype:
            return buf.view(self.dtype)
        return buf

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    @property
    def state_slabs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """(conv, ssm) state slab pairs; [] when no state slabs are active.

        Forwarding property: FlatStateSlabs owns the slabs, but the flat
        host mirror and hybrid-linear-attn backend probe pool.state_slabs
        directly (getattr), so keep the attribute on the pool."""
        return self._state.state_slabs

    def get_state_buffers(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(conv, ssm) state slab pair for a state layer; the n-th state
        layer (within-state-label occurrence order, the slab pairing order)
        binds pair n. Raises ValueError for non-state layers."""
        return self._state.get_state_buffers(layer_id)

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
