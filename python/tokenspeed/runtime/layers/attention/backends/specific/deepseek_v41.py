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

"""V4.1 FlatKV model-facing baseline.

The model owns projection, RMSNorm, RoPE and inverse output RoPE. This backend
owns logical-position resolution, all cache writes, and sparse selection. All
request history is in the four LCM groups; the only cross-layer scratch is the
current forward's SparseTopKShare. Memory-only writes need not have query rows.

Request indices below are BATCH TABLE ROWS, not request-pool slots. Obtain the
ordinary full-query inputs from query_metadata(mode); CED may explicitly subset
those positions. A source must select every query needed by its Reuse consumers.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import DeepseekV41Config
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_COMPRESSOR_TAIL_GROUP_ID,
    V41_GLOBAL_R1_GROUP_ID,
    V41_GLOBAL_R2_GROUP_ID,
    V41_GROUP_GEOMETRY,
    V41_SWA_GROUP_ID,
    v41_table_widths,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v41 import (
    DeepseekV41CachePool,
)
from tokenspeed.runtime.layers.attention.page_table import group_slot_mapping_from_raw
from tokenspeed.runtime.layers.attention.registry import register_backend


@dataclass
class V41Metadata:
    block_tables: dict[str, torch.Tensor]
    positions: torch.Tensor
    request_indices: torch.Tensor
    request_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    num_extends: int
    swa_write_slots: torch.Tensor


@dataclass
class V41Candidates:
    positions: torch.Tensor
    request_indices: torch.Tensor
    block_ids: torch.Tensor
    lengths: torch.Tensor


@dataclass
class V41Selection:
    owner: int
    source: int
    positions: torch.Tensor
    request_indices: torch.Tensor
    logical_rows: torch.Tensor
    lengths: torch.Tensor
    candidates: V41Candidates | None


class DeepseekV41AttentionBackend(AttentionBackend):
    # ponytail: host-sized tiled scans and masked tail writes synchronize;
    # enable graphs only after replacing them with fixed-shape native kernels.
    cuda_graph_support = CudaGraphSupport(decode_graph=False, prefill_graph=False)
    supports_layer_sliding_window = True

    def __init__(self, config: AttnConfig, spec: DeepseekV41Config) -> None:
        super().__init__(config, spec)
        if config.is_draft or self.spec_num_tokens != 1:
            raise NotImplementedError(
                "V4.1 FlatKV baseline supports target-only decoding"
            )
        if config.kernel_page_size not in (None, 64):
            raise ValueError("V4.1 SWA/global readers require 64-row pages")
        if config.prefix_granularity <= 0 or config.prefix_granularity % 128:
            raise ValueError(
                "V4.1 prefix granularity must be a positive multiple of 128"
            )
        if (
            spec.head_dim != 512
            or spec.sliding_window_tokens != 128
            or spec.candidate_block_size != 8
        ):
            raise ValueError(
                "V4.1 baseline requires head_dim=512, window=128, candidate blocks=8"
            )
        if not (1 <= spec.index_topk <= 512 and 1 <= spec.candidate_topk <= 2048):
            raise ValueError(
                "V4.1 selection sizes exceed the budgeted Top-512/2048 capacities"
            )
        self.spec = spec
        self.context_len = config.context_len
        self.forward_metadata: V41Metadata | None = None
        self.forward_prefill_metadata: V41Metadata | None = None
        self.forward_decode_metadata: V41Metadata | None = None
        self._decode_views_by_bs: dict[int, V41Metadata] = {}
        self._decode_buffers: V41Metadata | None = None
        self._max_decode_bs = 0

    def set_cache_pool(self, cache_pool: DeepseekV41CachePool) -> None:
        if not isinstance(cache_pool, DeepseekV41CachePool):
            raise TypeError("V4.1 backend requires DeepseekV41CachePool")
        specs = {s.group_id: s for s in cache_pool.arena.cache_group_specs}
        for gid, (rows, stride) in V41_GROUP_GEOMETRY.items():
            if gid not in specs or (
                specs[gid].rows_per_page,
                specs[gid].entry_stride_tokens,
            ) != (rows, stride):
                raise ValueError(f"V4.1 pool is missing the {gid} row geometry")
        super().set_cache_pool(cache_pool)

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        if self._decode_buffers is not None:
            if max_bs != self._max_decode_bs:
                raise RuntimeError(
                    "V4.1 decode capacity cannot change after initialization"
                )
            return
        self._max_decode_bs = max_bs
        horizon = (1 + int(kwargs.get("overlap_schedule_depth", 0))) * int(
            kwargs.get("max_tokens_per_req", 1)
        )
        self._decode_buffers = V41Metadata(
            block_tables={
                gid: torch.zeros((max_bs, width), dtype=torch.int32, device=self.device)
                for gid, width in v41_table_widths(self.context_len, horizon).items()
            },
            positions=torch.full((max_bs,), -1, dtype=torch.int64, device=self.device),
            request_indices=torch.arange(max_bs, dtype=torch.int64, device=self.device),
            request_pool_indices=torch.full(
                (max_bs,), -1, dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            num_extends=0,
            swa_write_slots=torch.full(
                (max_bs,), -1, dtype=torch.int64, device=self.device
            ),
        )

    def _decode_view(self, bs: int) -> V41Metadata:
        if self._decode_buffers is None or not 0 <= bs <= self._max_decode_bs:
            raise ValueError("V4.1 decode batch exceeds initialized capacity")
        if bs not in self._decode_views_by_bs:
            b = self._decode_buffers
            self._decode_views_by_bs[bs] = V41Metadata(
                {gid: t[:bs] for gid, t in b.block_tables.items()},
                b.positions[:bs],
                b.request_indices[:bs],
                b.request_pool_indices[:bs],
                b.seq_lens[:bs],
                0,
                b.swa_write_slots[:bs],
            )
        return self._decode_views_by_bs[bs]

    def _check_tables(self, block_tables: Mapping[str, torch.Tensor], bs: int) -> None:
        for gid in V41_GROUP_GEOMETRY:
            if gid not in block_tables:
                raise ValueError(f"V4.1 missing cache block table: {gid}")
            t = block_tables[gid]
            if t.ndim != 2 or t.shape[0] < bs or t.dtype != torch.int32:
                raise ValueError(f"V4.1 {gid} requires int32 [>= batch, columns] table")

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_tables: Mapping[str, torch.Tensor],
        for_graph_replay: bool,
        **kwargs,
    ) -> None:
        # The target runner omits num_extends on pure decode; mixed/draft
        # refresh callers can name the leading extend window explicitly.
        num_extends = kwargs.pop("num_extends", 0)
        del for_graph_replay, kwargs
        if not 0 <= num_extends <= actual_bs <= bs:
            raise ValueError("V4.1 invalid live/decode batch sizes")
        self._check_tables(block_tables, actual_bs)
        meta = self._decode_view(bs)
        for gid, dest in meta.block_tables.items():
            src = block_tables[gid]
            if src.shape[1] > dest.shape[1]:
                raise ValueError(
                    f"V4.1 {gid} table exceeds initialized context capacity"
                )
            dest.zero_()
            dest[:actual_bs, : src.shape[1]].copy_(src[:actual_bs])
        meta.seq_lens.zero_()
        meta.seq_lens[:actual_bs].copy_(seq_lens[:actual_bs])
        meta.positions.copy_(meta.seq_lens.to(torch.int64) - 1)
        meta.positions[:num_extends].fill_(-1)
        meta.request_pool_indices.fill_(-1)
        meta.request_pool_indices[:actual_bs].copy_(req_pool_indices[:actual_bs])
        meta.num_extends = num_extends
        self.forward_metadata = self.forward_decode_metadata = meta
        meta.swa_write_slots.copy_(
            self.cache_slots(
                V41_SWA_GROUP_ID,
                meta.positions,
                meta.request_indices,
                ForwardMode.DECODE,
            )
        )
        if num_extends:
            self.sparse_topk.decode = None
        else:
            self.forward_prefill_metadata = None
            self.sparse_topk.clear()

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ) -> None:
        del extend_prefix_lens_cpu, extend_with_prefix, kwargs
        if forward_mode.is_decode():
            raise ValueError("V4.1 decode metadata must use refresh_decode_metadata")
        self._check_tables(block_tables, bs)
        counts = [int(n) for n in extend_seq_lens_cpu[:num_extends].tolist()] + [1] * (
            bs - num_extends
        )
        total = sum(counts)
        if total > self.spec.max_query_tokens:
            raise ValueError("V4.1 forward exceeds budgeted query workspace")
        lengths = torch.tensor(counts, dtype=torch.int64, device=self.device)
        requests = torch.repeat_interleave(
            torch.arange(bs, device=self.device), lengths, output_size=total
        )
        starts = lengths.cumsum(0) - lengths
        prefix = torch.cat(
            (extend_prefix_lens[:num_extends], seq_lens[num_extends:bs] - 1)
        ).to(torch.int64)
        positions = (
            torch.arange(total, device=self.device)
            - starts[requests]
            + prefix[requests]
        )
        tables = {
            gid: t[:bs] for gid, t in block_tables.items() if gid in V41_GROUP_GEOMETRY
        }
        meta = V41Metadata(
            tables,
            positions,
            requests,
            req_pool_indices[:bs],
            seq_lens[:bs],
            num_extends,
            torch.empty(total, dtype=torch.int64, device=self.device),
        )
        self.forward_metadata = meta
        meta.swa_write_slots.copy_(
            self.cache_slots(V41_SWA_GROUP_ID, positions, requests, ForwardMode.MIXED)
        )
        n = sum(counts[:num_extends])
        self.forward_prefill_metadata = V41Metadata(
            tables,
            positions[:n],
            requests[:n],
            meta.request_pool_indices,
            meta.seq_lens,
            num_extends,
            meta.swa_write_slots[:n],
        )
        self.forward_decode_metadata = V41Metadata(
            tables,
            positions[n:],
            requests[n:],
            meta.request_pool_indices,
            meta.seq_lens,
            num_extends,
            meta.swa_write_slots[n:],
        )
        self.sparse_topk.clear()

    def query_metadata(self, forward_mode: ForwardMode) -> V41Metadata:
        """Return token positions/request table rows and current per-group tables.

        EXTEND and DECODE select explicit windows; MIXED returns the combined
        request-major token span. Never interpret request_pool_indices as rows.
        """
        meta = (
            self.forward_decode_metadata
            if forward_mode.is_decode()
            else (
                self.forward_prefill_metadata
                if forward_mode.is_extend()
                else self.forward_metadata
            )
        )
        if meta is None:
            raise RuntimeError("V4.1 metadata has not been prepared for this mode")
        return meta

    def _owner_group(self, owner: int) -> str:
        if (
            not 0 <= owner < len(self.spec.kv_owners)
            or self.spec.kv_owners[owner] != owner
        ):
            raise ValueError(f"V4.1 layer {owner} is not a global KV owner")
        return (
            V41_GLOBAL_R2_GROUP_ID
            if self.spec.compress_ratios[owner] == 2
            else V41_GLOBAL_R1_GROUP_ID
        )

    def cache_slots(
        self,
        group_id: str,
        positions: torch.Tensor,
        request_indices: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> torch.Tensor:
        """Resolve raw positions to field-relative slots; null/invalid rows are -1.

        Positions and request_indices have equal shape (including selection
        matrices). The caller names the address domain, never a physical page.
        """
        if positions.shape != request_indices.shape:
            raise ValueError("positions and request_indices must have equal shape")
        rows, stride = V41_GROUP_GEOMETRY[group_id]
        meta = self.query_metadata(forward_mode)
        slots = group_slot_mapping_from_raw(
            positions.reshape(-1),
            request_indices.reshape(-1),
            meta.block_tables[group_id],
            rows,
            stride,
        ).reshape(positions.shape)
        valid = (positions >= 0) & (slots >= rows)
        if self.cache_pool is not None:
            valid &= (
                slots < self.cache_pool.arena.cache_group_page_counts[group_id] * rows
            )
        return slots.masked_fill(~valid, -1)

    def write_locations(self, layer, forward_mode: ForwardMode) -> torch.Tensor:
        return self.query_metadata(forward_mode).swa_write_slots

    def global_read_slots(
        self,
        owner: int,
        logical_rows: torch.Tensor,
        positions: torch.Tensor,
        request_indices: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> torch.Tensor:
        """Map request-local global row IDs to owner's slots with per-query causality."""
        gid = self._owner_group(owner)
        ratio = self.spec.compress_ratios[owner]
        visible = (positions + 1) // ratio
        valid = (logical_rows >= 0) & (logical_rows < visible[:, None])
        raw = (logical_rows * ratio).masked_fill(~valid, -1)
        return self.cache_slots(
            gid, raw, request_indices[:, None].expand_as(raw), forward_mode
        ).to(torch.int32)

    def _lookup_rows(
        self, source_positions, source_requests, positions, requests
    ) -> torch.Tensor:
        if source_positions.numel() == 0:
            return torch.full_like(positions, -1, dtype=torch.int64)
        grain = self.context_len + 1
        keys, order = (
            source_requests.to(torch.int64) * grain + source_positions
        ).sort()
        return self._lookup_sorted(keys, order, positions, requests)

    def _lookup_sorted(self, keys, order, positions, requests) -> torch.Tensor:
        wanted = requests.to(torch.int64) * (self.context_len + 1) + positions
        index = torch.searchsorted(keys, wanted.contiguous()).clamp_max(
            keys.numel() - 1
        )
        valid = (keys[index] == wanted) & (positions >= 0) & (requests >= 0)
        return order[index].masked_fill(~valid, -1)

    def read_compressor_tail(self, owner, positions, request_indices, forward_mode):
        """Return FP32 [rows, 2, 512] content/score history; missing rows fail."""
        if self.spec.compress_ratios[owner] != 2:
            raise ValueError("Only ratio-2 owners have compressor tails")
        self._owner_group(owner)
        slots = self.cache_slots(
            V41_COMPRESSOR_TAIL_GROUP_ID, positions, request_indices, forward_mode
        )
        if bool((slots < 0).any()):
            raise RuntimeError(
                "V4.1 required compressor tail is absent; request needs recovery"
            )
        return self.cache_pool.compressor_tail(owner)[slots // 2, slots % 2]

    def write_compressor_tail(
        self, owner, content, scores, positions, request_indices, forward_mode
    ) -> None:
        """Store FP32 projected inputs only at LCM-retained token rows.

        Released rows in a prefill are skipped; completed pairs use this
        forward's projection tensors rather than requiring expired tail pages.
        """
        self._owner_group(owner)
        tail = self.cache_pool.compressor_tail(owner)
        if content.shape != (positions.numel(), 512) or scores.shape != content.shape:
            raise ValueError("compressor content/scores must be [tokens, 512]")
        slots = self.cache_slots(
            V41_COMPRESSOR_TAIL_GROUP_ID, positions, request_indices, forward_mode
        )
        live = slots >= 0
        tail[slots[live] // 2, slots[live] % 2, 0] = content[live].float()
        tail[slots[live] // 2, slots[live] % 2, 1] = scores[live].float()

    def compress(
        self, owner, content, scores, positions, request_indices, forward_mode
    ):
        """Pool a ratio-2 owner's projected inputs, without norm or RoPE.

        Inputs: FP32 [T,512] content/scores and absolute token positions/table
        rows. Returns (FP32 pooled rows, pair-start RoPE positions, requests).
        Only completed odd-position pairs are returned. The caller casts to
        activation dtype BEFORE RMSNorm, then derives index K before main RoPE.
        """
        if content.dtype != torch.float32 or scores.dtype != torch.float32:
            raise ValueError("V4.1 ratio-2 projection/pooling requires FP32 inputs")
        if self.spec.compress_ratios[owner] != 2:
            raise ValueError("compress() is only for ratio-2 owners")
        end = torch.where((positions >= 0) & (positions % 2 == 1))[0]
        pair_positions, pair_requests = positions[end] - 1, request_indices[end]
        previous = self._lookup_rows(
            positions, request_indices, pair_positions, pair_requests
        )
        missing = previous < 0
        # Read before any writes: a scheduler-reused tail page must not destroy
        # the odd-prefix input needed by the first completed pair in this chunk.
        prior = torch.stack(
            (content[previous.clamp_min(0)], scores[previous.clamp_min(0)]), dim=1
        )
        if bool(missing.any()):
            prior[missing] = self.read_compressor_tail(
                owner, pair_positions[missing], pair_requests[missing], forward_mode
            )
        weights = torch.stack((prior[:, 1], scores[end]), dim=1).softmax(dim=1)
        pooled = weights[:, 0] * prior[:, 0] + weights[:, 1] * content[end]
        self.write_compressor_tail(
            owner, content, scores, positions, request_indices, forward_mode
        )
        return pooled, pair_positions, pair_requests

    def write_global(
        self, owner, main_kv, index_k, positions, request_indices, forward_mode
    ) -> None:
        """Store both post-RoPE owner fields, including memory-only positions.

        main_kv is BF16/FP32 [rows,512], index_k [rows,128]. Positions name
        global row STARTS (j*ratio), not pair ends; ratio-1 uses token positions.
        Both writes are enqueued on the model stream before returning; later
        attention on that stream observes both fields, without a host sync.
        """
        from tokenspeed_kernel.ops.attention import dsv41

        gid = self._owner_group(owner)
        ratio = self.spec.compress_ratios[owner]
        if bool(((positions >= 0) & (positions % ratio != 0)).any()):
            raise ValueError(
                "global write positions must name compression-group starts"
            )
        if main_kv.shape != (positions.numel(), 512) or index_k.shape != (
            positions.numel(),
            128,
        ):
            raise ValueError(
                "global main/index rows must have matching positions and dimensions"
            )
        slots = self.cache_slots(gid, positions, request_indices, forward_mode)
        dsv41.cache_scatter(main_kv, self.cache_pool.global_kv(owner), slots, "global")
        dsv41.cache_scatter(index_k, self.cache_pool.index_k(owner), slots, "index")

    def _selection_rows(self, record, positions, requests):
        index = self._lookup_rows(
            record.positions, record.request_indices, positions, requests
        )
        if bool(((index < 0) & (positions >= 0)).any()):
            raise RuntimeError(
                "V4.1 source did not select a consumer's request/absolute query position"
            )
        return index

    def select_global(
        self,
        layer_id,
        index_q,
        index_weights,
        positions,
        request_indices,
        forward_mode,
        index_process_group,
    ):
        """Select/reuse request-local global rows; return (physical slots, lengths).

        Index Q is post-RoPE BF16 [T,Hindex,128]; weights are already scaled as
        in the model reference. Supply both only at an index source, otherwise
        None. A sharded indexer must supply its head-reduction process group;
        replicated indexer heads supply None. Candidates survive Reindex updates.
        """
        from tokenspeed_kernel.ops.attention import dsv41

        if (
            positions.numel() > self.spec.max_query_tokens
            or positions.shape != request_indices.shape
        ):
            raise ValueError(
                "V4.1 selection queries exceed or disagree with the workspace budget"
            )
        owner, source = self.spec.kv_owners[layer_id], self.spec.index_sources[layer_id]
        if owner < 0:
            if index_q is not None or index_weights is not None:
                raise ValueError("SWA-only layers have no indexer")
            return None, None
        if forward_mode.is_mixed():
            raise ValueError(
                "select_global requires the explicit EXTEND or DECODE query window"
            )
        share = self.sparse_topk
        record = share.decode if forward_mode.is_decode() else share.prefill
        if layer_id != source:
            if index_q is not None or index_weights is not None:
                raise ValueError("Reuse layers must not supply index projections")
            if record is None or record.owner != owner or record.source != source:
                raise RuntimeError(
                    "V4.1 Reuse has no compatible selection from this forward"
                )
            index = self._selection_rows(record, positions, request_indices)
            rows = record.logical_rows[index.clamp_min(0)].masked_fill(
                index[:, None] < 0, -1
            )
            lens = record.lengths[index.clamp_min(0)].masked_fill(index < 0, 0)
            return (
                self.global_read_slots(
                    owner, rows, positions, request_indices, forward_mode
                ),
                lens,
            )
        if index_q is None or index_weights is None:
            raise ValueError("V4.1 index sources require index_q and index_weights")
        candidates = (
            record.candidates if record is not None and record.owner == owner else None
        )
        reindex = source != owner
        if reindex and candidates is None:
            raise RuntimeError("V4.1 Reindex is missing candidate-source selections")
        candidate_index = (
            self._selection_rows(candidates, positions, request_indices)
            if reindex
            else None
        )
        produce_candidates = layer_id == self.spec.candidate_source
        n = positions.numel()
        rows = torch.full(
            (n, self.spec.index_topk), -1, dtype=torch.int32, device=self.device
        )
        lens = torch.zeros(n, dtype=torch.int32, device=self.device)
        blocks = torch.full(
            (n, self.spec.candidate_topk if produce_candidates else 0),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        block_lens = torch.zeros(n, dtype=torch.int32, device=self.device)
        ratio = self.spec.compress_ratios[owner]
        table = self.query_metadata(forward_mode).block_tables[self._owner_group(owner)]
        # Query tiling bounds both score scratch and replicated table rows. Do
        # not index_select the table for the entire prefill (T * context pages).
        for start in range(0, n, 8):
            stop = min(start + 8, n)
            visible = (
                ((positions[start:stop] + 1) // ratio).clamp_min(0).to(torch.int32)
            )
            width = min(table.shape[1], (int(visible.max().item()) + 63) // 64)
            req = request_indices[start:stop]
            live_req = (req >= 0) & (req < table.shape[0])
            tile_table = table[req.clamp(0, table.shape[0] - 1), :width].clone()
            tile_table.masked_fill_((tile_table <= 0) | ~live_req[:, None], -1)
            cb = None
            if reindex:
                ci = candidate_index[start:stop]
                candidate_count = int(candidates.lengths[ci.clamp_min(0)].max().item())
                cb = candidates.block_ids[
                    ci.clamp_min(0), :candidate_count
                ].masked_fill(ci[:, None] < 0, -1)
            dsv41.index_topk(
                index_q[start:stop],
                index_weights[start:stop],
                self.cache_pool.index_k(owner),
                tile_table,
                visible,
                cb,
                self.spec.index_topk,
                self.spec.candidate_topk if produce_candidates else 0,
                self.spec.candidate_block_size,
                8,
                256,
                index_process_group,
                (
                    rows[start:stop],
                    lens[start:stop],
                    blocks[start:stop],
                    block_lens[start:stop],
                ),
            )
        if produce_candidates:
            candidates = V41Candidates(
                positions.clone(), request_indices.clone(), blocks, block_lens
            )
        record = V41Selection(
            owner,
            source,
            positions.clone(),
            request_indices.clone(),
            rows,
            lens,
            candidates,
        )
        if forward_mode.is_decode():
            share.decode = record
        else:
            share.prefill = record
        return (
            self.global_read_slots(
                owner, rows, positions, request_indices, forward_mode
            ),
            lens,
        )

    def forward_v41(
        self,
        q,
        swa_kv,
        *,
        layer_id: int,
        positions,
        request_indices,
        forward_mode: ForwardMode,
        index_q,
        index_weights,
        attn_sink,
        softmax_scale: float,
        index_process_group,
    ) -> torch.Tensor:
        """Joint SWA/global attention; returns BF16 [T,Hlocal,512], BEFORE inverse RoPE.

        q and swa_kv are post-RoPE [T,Hlocal,512] / [T,512]. Each SWA input
        corresponds to the explicit query positions/requests. Global memory is
        written separately by write_global; index inputs follow select_global.
        Cache SWA writes happen AFTER all queries read the prefix, and only into
        LCM-retained pages. CED callers must supply each layer's required SWA
        activation range, not only the final output position.
        """
        from tokenspeed_kernel.ops.attention import dsv41

        n = positions.numel()
        if (
            n > self.spec.max_query_tokens
            or q.shape[0] != n
            or swa_kv.shape != (n, 512)
            or request_indices.shape != positions.shape
        ):
            raise ValueError(
                "V4.1 query/SWA shapes exceed or disagree with the query budget"
            )
        if not 0 <= layer_id < len(self.spec.kv_owners):
            raise ValueError("V4.1 layer_id is outside the backbone")
        if forward_mode.is_mixed():
            boundary = self.query_metadata(forward_mode).num_extends
            out = torch.empty_like(q)
            for mode, mask in (
                (ForwardMode.EXTEND, request_indices < boundary),
                (ForwardMode.DECODE, request_indices >= boundary),
            ):
                out[mask] = self.forward_v41(
                    q[mask],
                    swa_kv[mask],
                    layer_id=layer_id,
                    positions=positions[mask],
                    request_indices=request_indices[mask],
                    forward_mode=mode,
                    index_q=index_q[mask] if index_q is not None else None,
                    index_weights=(
                        index_weights[mask] if index_weights is not None else None
                    ),
                    attn_sink=attn_sink,
                    softmax_scale=softmax_scale,
                    index_process_group=index_process_group,
                )
            return out
        if n == 0:
            return torch.empty_like(q)
        global_slots, global_lens = self.select_global(
            layer_id,
            index_q,
            index_weights,
            positions,
            request_indices,
            forward_mode,
            index_process_group,
        )
        owner = self.spec.kv_owners[layer_id]
        global_cache = self.cache_pool.global_kv(owner) if owner >= 0 else None
        cache = self.cache_pool.swa(layer_id)
        packed = dsv41.cache_pack(swa_kv, "swa", None)
        current_keys, current_order = (
            request_indices.to(torch.int64) * (self.context_len + 1) + positions
        ).sort()
        out = torch.empty_like(q)
        for start in range(0, n, 8):
            stop = min(start + 8, n)
            pos = positions[start:stop, None] - torch.arange(
                127, -1, -1, device=self.device
            )
            req = request_indices[start:stop, None].expand_as(pos)
            slots = self.cache_slots(V41_SWA_GROUP_ID, pos, req, forward_mode)
            current = self._lookup_sorted(current_keys, current_order, pos, req)
            valid = (pos >= 0) & ((slots >= 0) | (current >= 0))
            required = (pos >= 0) & (positions[start:stop, None] >= 0)
            if bool((required & ~valid).any()):
                raise RuntimeError(
                    "V4.1 SWA prefix is missing; supply the dependency tail or recover the request"
                )
            # Gather packed bytes for at most eight windows. Current rows stay
            # transient until all old prefix reads finish, even with reused pages.
            selected = cache[slots.clamp_min(0) // 64, slots.clamp_min(0) % 64]
            selected = torch.where(
                (current >= 0)[..., None], packed[current.clamp_min(0)], selected
            )
            swa_view = selected.reshape(-1, 64, 528)
            local_slots = torch.arange(
                (stop - start) * 128, dtype=torch.int32, device=self.device
            ).view(stop - start, 128)
            local_slots.masked_fill_(~valid, -1)
            dsv41.selected_attention(
                q[start:stop],
                swa_view,
                local_slots,
                torch.full((stop - start,), 128, dtype=torch.int32, device=self.device),
                global_cache,
                global_slots[start:stop] if global_slots is not None else None,
                global_lens[start:stop] if global_lens is not None else None,
                attn_sink,
                softmax_scale,
                out[start:stop],
                8,
            )
        locations = self.cache_slots(
            V41_SWA_GROUP_ID, positions, request_indices, forward_mode
        )
        dsv41.cache_scatter(swa_kv, cache, locations, "swa")
        return out

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "V4.1 model layers call forward_v41 with explicit positions"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "V4.1 model layers call forward_v41 with explicit positions"
        )


register_backend("deepseek_v41", {AttentionArch.MLA}, DeepseekV41AttentionBackend)
