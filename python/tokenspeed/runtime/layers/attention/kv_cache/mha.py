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

import os
from collections import Counter

import numpy as np
import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    quantize_store_kv_mxfp8,
    store_kv_cache,
    store_sf_interleaved,
)

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
from tokenspeed.runtime.utils.pdl import pdl_enabled
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
        extra_paged_groups: tuple = (),
        slot_tokens: int | None = None,
        group_page_sizes: dict | None = None,
        layer_kv_head_counts: tuple[int, ...] | None = None,
        kv_alloc_head_count: int | None = None,
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
        # Heterogeneous KV (#647): one pool id = one byte-uniform slot;
        # each group factorizes it as (block tokens x per-layer heads) —
        # full 256 tok x half heads == swa 128 tok x full heads, zero pad.
        # Per-layer views absorb the geometry; _slot_tokens tracks the
        # LARGEST block for dummy-slot sizing only.
        self._slot_tokens = int(slot_tokens or page_size)
        # Hiddenconv column dtype: fp8-e5m2 direct cast; INKLING_FP8_SCONV=0
        # restores bf16.
        self.conv_col_dtype = (
            torch.bfloat16
            if os.environ.get("INKLING_FP8_SCONV", "1") == "0"
            else torch.float8_e5m2
        )
        self._group_page_sizes = dict(group_page_sizes or {})
        # Per-layer KV heads: slabs alloc at the max; fewer heads = same bytes viewed as more rows
        self._layer_kv_head_counts = (
            tuple(int(h) for h in layer_kv_head_counts)
            if layer_kv_head_counts
            else None
        )
        # Pre-TP head count ``head_num`` is the per-rank shard of — the view
        # normalization base. Falling back to max(counts) is only correct
        # when the pool's own layers include an alloc-width layer: an
        # all-narrow pool (Inkling MTP draft with full-attention-only
        # depths, alloc'd at the config max) would silently collapse the
        # reinterpretation and write narrow rows into wide strides (the
        # #65/1.82-accept GQA-stride corruption).
        self._kv_alloc_head_count = (
            int(kv_alloc_head_count) if kv_alloc_head_count else None
        )
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
            page_sizes=self._group_page_sizes or None,
            extra_groups=extra_paged_groups,
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
        With unequal groups, slabs past a smaller group's count are
        single-layer (occurrence indices never exceed the group's count).
        """
        assert self._slab_group_size is not None
        assert len(self._layer_types) == self.layer_num, (
            f"hybrid slab layout: layer_types has {len(self._layer_types)} "
            f"entries but layer_num={self.layer_num}"
        )
        counts = Counter(self._layer_types)
        assert max(counts.values()) == self._slab_group_size, (
            f"hybrid slab layout: groups {dict(counts)!r} inconsistent with "
            f"slab count {self._slab_group_size}"
        )
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
        # Hetero head counts reinterpret the byte-uniform slab as MORE rows
        # for fewer-head layers (rows * head_num / heads_l), and the backend
        # then views those rows at the layer group's page size. Every such
        # view must divide: an unaligned profiled size otherwise surfaces as
        # an opaque view() error deep inside the first forward. Seen in the
        # wild 2026-07-14: residual memory on one
        # rank shifted the memory-dependent profile to size % 128 == 64 and
        # the full-attention 2x reinterpretation could no longer be paged at
        # 256 rows. Fail fast with the actual remainders instead.
        if self._layer_kv_head_counts:
            rows = self.size + self._slot_tokens
            for layer_id in range(self.layer_num):
                heads_l = self._layer_heads_per_rank(layer_id)
                view_rows = rows * self.head_num // heads_l
                page = self._group_page_sizes.get(
                    self._layer_types[layer_id] if self._layer_types else "",
                    self.page_size,
                )
                if view_rows % page:
                    raise RuntimeError(
                        f"hybrid slab size {self.size} (+{self._slot_tokens} slot "
                        f"rows) is not page-aligned for layer {layer_id}: the "
                        f"{heads_l}-head reinterpretation yields {view_rows} rows, "
                        f"not divisible by its group page size {page}. The KV "
                        "profile must floor max_total_num_tokens to the layout's "
                        "alignment (memory-dependent sizing can land off-grid — "
                        "check for residual allocations on a rank)."
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
                    (self.size + self._slot_tokens, self.head_num, self.head_dim),
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
                    "KV layout: hybrid slab (%d slabs x %d rows; slot %d; "
                    "%d ids = %d full-tok / %d swa-tok capacity; paired "
                    "layers share storage; M12)",
                    self._slab_group_size,
                    self.size + self._slot_tokens,
                    self._slot_tokens,
                    self.size // self.page_size,
                    (self.size // self.page_size)
                    * max(self._group_page_sizes.values(), default=self.page_size),
                    (self.size // self.page_size) * self.page_size,
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

    def _layer_row_view(self, buf: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Per-layer token-row view over the byte-uniform slab.

        Slabs are allocated ``(rows, head_num, head_dim)`` at the MAX head
        count; a layer serving fewer heads reinterprets the same bytes as
        ``rows * (head_num / heads_l)`` rows of ``heads_l`` heads (full
        layers: 2x the token rows per slot — the zero-padding contract).
        """
        if self._layer_kv_head_counts is None:
            return buf
        heads_l = self._layer_heads_per_rank(layer_id)
        if heads_l == self.head_num:
            return buf
        return buf.reshape(-1, heads_l, self.head_dim)

    def _layer_heads_per_rank(self, layer_id: int) -> int:
        counts = self._layer_kv_head_counts
        if counts is None:
            # Uniform head counts (hetero off): every layer serves head_num.
            return self.head_num
        served = counts[layer_id]
        # head_num is the per-rank shard of the ALLOCATION width (the config
        # max); scale the pre-TP served count proportionally. max(counts) is
        # only a valid stand-in when some layer serves the alloc width — an
        # all-narrow pool must still reinterpret every layer.
        alloc = self._kv_alloc_head_count or max(counts)
        return max(1, self.head_num * served // alloc)

    def conv_slot_view(
        self,
        layer_id: int,
        kind: str,
        block_tokens: int,
        ch: int,
        col_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """One conv-column view ``(num_ids, block_tokens, ch)`` over the
        layer's K (``kind="k"``) or V (``kind="v"``) slab slots.

        Pool id j owns slot j (page_size tokens of KV rows); the view
        takes the slot's leading ``block_tokens * ch`` elements, so slot
        slack keeps the slot stride and the view stays zero-copy (the
        paged conv kernels take independent slot/row strides). Ids are
        the shared global currency; each slot has a single owner group
        because the conv groups mirror the attention groups the slab
        pairing is built from.
        """
        bufs = self.k_buffer if kind == "k" else self.v_buffer
        buf = bufs[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} has no KV slab")
        # Columns use conv_col_dtype regardless of slab dtype (an fp8 slab holds half the bf16 elems)
        if col_dtype is None:
            col_dtype = self.conv_col_dtype
        slot_bytes = self.page_size * self.head_num * self.head_dim * buf.element_size()
        conv_bytes = block_tokens * ch * col_dtype.itemsize
        assert (
            conv_bytes <= slot_bytes
        ), f"conv block {block_tokens}x{ch} ({col_dtype}) exceeds slot {slot_bytes} bytes"
        num_ids = buf.numel() * buf.element_size() // slot_bytes
        flat = buf.view(torch.uint8).reshape(num_ids, slot_bytes)
        return flat[:, :conv_bytes].view(col_dtype).view(num_ids, block_tokens, ch)

    def kvconv_slot_views_for_layer(
        self, layer_id: int, block_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(K, V) conv-input column views: K-conv columns in the K slot,
        V-conv in the V slot, at the layer's served kv width per rank —
        byte-exact for swa layers, the leading half for hetero full
        layers."""
        if self._layer_kv_head_counts is not None:
            ch = self._layer_heads_per_rank(layer_id) * self.head_dim
        else:
            ch = self.head_num * self.head_dim
        # kvconv stays bf16: these columns alias attention-owned pages, so fp8 saves nothing
        return (
            self.conv_slot_view(layer_id, "k", block_tokens, ch, torch.bfloat16),
            self.conv_slot_view(layer_id, "v", block_tokens, ch, torch.bfloat16),
        )

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        buf = self.k_buffer[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV buffer")
        if self.store_dtype != self.dtype:
            buf = buf.view(self.dtype)
        return self._layer_row_view(buf, layer_id)

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
            buf = buf.view(self.dtype)
        return self._layer_row_view(buf, layer_id)

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
        # Locs are in per-layer view rows: the store must target the same view get_key_buffer serves
        store_kv_cache(
            cache_k,
            cache_v,
            self._layer_row_view(self.k_buffer[layer_id], layer_id),
            self._layer_row_view(self.v_buffer[layer_id], layer_id),
            loc,
            enable_pdl=pdl_enabled(),
        )


class MHATokenToKVPoolMXFP8(MHATokenToKVPool):
    """MHA KV pool storing MXFP8 block-scaled FP8 (data + UE8M0 scales).

    Data buffers hold float8_e4m3fn; scale buffers hold one float8_e8m0fnu
    per 32 elements of head_dim. ``set_kv_buffer`` expects PRE-QUANTIZED
    K/V plus per-token scale tensors (producer: ``quantize_mxfp8``); the
    bf16 per-tensor-scale paths of the base class do not apply.

    Scale layout follows what the FA4 blockscaled kernel consumes:
    page_size 128 stores scales interleaved in the BlockScaledBasicChunk
    atom ([num_pages, heads, 32, 4, 4], written via
    ``store_sf_interleaved``); any other page size stores them flat
    ([slots, heads, head_dim // 32]).

    Hybrid-slab mode (flat ext, Inkling): fp8 data rides the base class's
    byte-uniform slabs unchanged, and every block id additionally owns one
    SF slot per K/V in parallel scale slabs with the same slab pairing
    (slot_bytes / 32 e8m0 each — byte-uniform like the data slots). A
    layer views its SF slots as (num_ids, heads_l, k_l, 32, 4, 4), the
    per-in-page-128-chunk atom order the tsmha blockscaled TMA consumes
    (k_l = the layer's page tokens / 128; hetero full layers have half
    the heads and twice the chunks of swa, same bytes).
    """

    MXFP8_SCALE_BLOCK_SIZE = 32

    def _create_buffers(self):
        assert self.head_dim % self.MXFP8_SCALE_BLOCK_SIZE == 0
        self.store_dtype = torch.float8_e4m3fn
        if self._slab_group_size is not None:
            super()._create_buffers()
            with self.memory_saver_adapter.region(
                tag="kv_cache", enable_cpu_backup=False
            ):
                self._create_scale_slabs()
            return
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            m = self.size + self.page_size
            n, k = self.head_num, self.head_dim
            self.k_buffer = [
                torch.zeros((m, n, k), dtype=self.store_dtype, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros((m, n, k), dtype=self.store_dtype, device=self.device)
                for _ in range(self.layer_num)
            ]
            sf_dim = k // self.MXFP8_SCALE_BLOCK_SIZE
            if self.page_size == 128:
                sf_shape = (m // self.page_size, n, 32, sf_dim, sf_dim)
            else:
                sf_shape = (m, n, sf_dim)
            self.k_scale_buffer = [
                torch.zeros(sf_shape, dtype=torch.float8_e8m0fnu, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.v_scale_buffer = [
                torch.zeros(sf_shape, dtype=torch.float8_e8m0fnu, device=self.device)
                for _ in range(self.layer_num)
            ]
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

    def _create_scale_slabs(self):
        # One SF slot per block id and K/V side; paired layers alias SF slabs exactly like data slabs
        slot_sf = (
            self.page_size * self.head_num * self.head_dim
        ) // self.MXFP8_SCALE_BLOCK_SIZE
        num_ids = (self.size + self._slot_tokens) // self.page_size
        pair_index = self._slab_pair_index()

        def _alloc():
            return torch.zeros(
                num_ids, slot_sf, dtype=torch.float8_e8m0fnu, device=self.device
            )

        k_sf = [_alloc() for _ in range(self._slab_group_size)]
        v_sf = [_alloc() for _ in range(self._slab_group_size)]
        self.k_scale_buffer = [
            k_sf[pair_index[layer_id]] for layer_id in range(self.layer_num)
        ]
        self.v_scale_buffer = [
            v_sf[pair_index[layer_id]] for layer_id in range(self.layer_num)
        ]

    def _layer_page_tokens(self, layer_id: int) -> int:
        """Tokens per page for one layer under byte-uniform slots (the
        slot's fixed bytes factorized through the layer's head count)."""
        heads_l = self._layer_heads_per_rank(layer_id)
        return self.page_size * self.head_num // heads_l

    def _layer_scale_view(self, buf: torch.Tensor, layer_id: int) -> torch.Tensor:
        """(num_ids, heads_l, k_l, 32, 4, 4) view over a layer's SF slots
        (the paged interleaved layout the blockscaled kernels consume)."""
        heads_l = self._layer_heads_per_rank(layer_id)
        k_l = self._layer_page_tokens(layer_id) // 128
        sf_dim = self.head_dim // self.MXFP8_SCALE_BLOCK_SIZE
        return buf.view(buf.shape[0], heads_l, k_l, 32, sf_dim, sf_dim)

    def _get_page_size_bytes(self):
        # fp8 data + e8m0 scales (1 per 32 elements of head_dim).
        per_elem = 1 + 1 / self.MXFP8_SCALE_BLOCK_SIZE
        return int(
            2
            * self.page_size
            * self.layer_num
            * self.head_num
            * self.head_dim
            * per_elem
        )

    def _clear_buffers(self):
        super()._clear_buffers()
        if hasattr(self, "k_scale_buffer"):
            del self.k_scale_buffer
        if hasattr(self, "v_scale_buffer"):
            del self.v_scale_buffer

    def get_kv_scale_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(k_scale, v_scale) buffers for the blockscaled attention kernel.

        Legacy per-layer mode returns the raw wheel-shaped buffers; slab
        mode returns the layer's (num_ids, heads_l, k_l, 32, 4, 4) views.
        """
        k_sf = self.k_scale_buffer[layer_id]
        v_sf = self.v_scale_buffer[layer_id]
        if self._slab_group_size is not None:
            return (
                self._layer_scale_view(k_sf, layer_id),
                self._layer_scale_view(v_sf, layer_id),
            )
        return k_sf, v_sf

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        layer_id_override: int = None,
    ):
        assert (
            cache_k.dtype == self.store_dtype
        ), "MXFP8 pool expects pre-quantized fp8 K (see quantize_mxfp8)"
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 pool requires per-token e8m0 scale tensors"
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        # Byte views: triton can't mask-fill fp8; locs are per-layer view rows (target the served view)
        store_kv_cache(
            cache_k.view(torch.uint8),
            cache_v.view(torch.uint8),
            self._layer_row_view(self.k_buffer[layer_id], layer_id).view(torch.uint8),
            self._layer_row_view(self.v_buffer[layer_id], layer_id).view(torch.uint8),
            loc,
            enable_pdl=pdl_enabled(),
        )
        if self._slab_group_size is not None:
            page_tokens = self._layer_page_tokens(layer_id)
            store_sf_interleaved(
                k_scale,
                self.k_scale_buffer[layer_id],
                loc,
                page_size=page_tokens,
                enable_pdl=pdl_enabled(),
            )
            store_sf_interleaved(
                v_scale,
                self.v_scale_buffer[layer_id],
                loc,
                page_size=page_tokens,
                enable_pdl=pdl_enabled(),
            )
        elif self.page_size == 128:
            store_sf_interleaved(
                k_scale, self.k_scale_buffer[layer_id], loc, enable_pdl=pdl_enabled()
            )
            store_sf_interleaved(
                v_scale, self.v_scale_buffer[layer_id], loc, enable_pdl=pdl_enabled()
            )
        else:
            self.k_scale_buffer[layer_id][loc] = k_scale
            self.v_scale_buffer[layer_id][loc] = v_scale

    def quantize_and_set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id_override: int = None,
    ) -> bool:
        """Fused per-token quantize + data store + SF scatter (one launch).

        Bit-identical to quantize_mxfp8 + set_kv_buffer (parity-tested) and
        keeps the store inside the PDL chain. Returns False when the layout
        has no interleaved-SF path (page not a 128 multiple) — the caller
        falls back to the split path.
        """
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        if self._slab_group_size is not None:
            page_tokens = self._layer_page_tokens(layer_id)
        elif self.page_size == 128:
            page_tokens = 128
        else:
            return False
        if self.head_dim != 128:
            return False
        quantize_store_kv_mxfp8(
            k,
            v,
            self._layer_row_view(self.k_buffer[layer_id], layer_id),
            self._layer_row_view(self.v_buffer[layer_id], layer_id),
            self.k_scale_buffer[layer_id],
            self.v_scale_buffer[layer_id],
            loc,
            page_tokens=page_tokens,
            enable_pdl=pdl_enabled(),
        )
        return True

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        for sf in self.k_scale_buffer:
            k_size += sf.numel() * sf.element_size()
        for sf in self.v_scale_buffer:
            v_size += sf.numel() * sf.element_size()
        return k_size, v_size
