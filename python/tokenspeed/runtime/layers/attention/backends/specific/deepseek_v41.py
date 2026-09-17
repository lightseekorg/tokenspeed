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
request history is in the four LCM groups; cross-layer scratch contains only
the current forward's selection and immutable query-address plans. Memory-only
writes need not have query rows. Decode/verify uses a fixed request-major token
window, including graph padding. Rejected suffix bytes stay invisible until
overwritten at their absolute positions; no backend-private rollback is needed.

Request indices below are BATCH TABLE ROWS, not request-pool slots. Obtain the
ordinary full-query inputs from query_metadata(mode); the CED decoder layers use
decoder_view() instead, the per-request tail of the same rows. A source must
select every query needed by its Reuse consumers. Compressor projections cover
the full canonical query window: request spans have consecutive positions, so
pair metadata is prepared once by direct row lookup.

Bounded replay: the SWA and compressor-tail groups are never prefix-cached. A
prefix hit re-feeds the cached prefix's last window (``extend_replay_lens``);
those rows regenerate SWA rows and tails into private pages, see SWA keys only
from the replay start, and never write the global/index rows the hit already
holds (``V41Metadata.global_write_floor``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

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
    V41_PREFILL_QUERY_TILE,
    V41_SWA_GROUP_ID,
    v41_table_widths,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v41 import (
    DeepseekV41CachePool,
)
from tokenspeed.runtime.layers.attention.registry import register_backend


class V41CompressorPlan(NamedTuple):
    """Current-forward pair indices and addresses shared by all ratio-2 owners.

    Contains no projected inputs or request history. Decode storage is allocated
    at max_decode_bs * verify_width and refreshed in place before execution.
    """

    active: torch.Tensor
    pair_positions: torch.Tensor
    pair_requests: torch.Tensor
    previous: torch.Tensor
    read_slots: torch.Tensor
    write_slots: torch.Tensor

    @classmethod
    def allocate(cls, n: int, device: torch.device):
        return cls(
            torch.empty(n, dtype=torch.bool, device=device),
            *(torch.empty(n, dtype=torch.int64, device=device) for _ in range(5)),
        )

    def window(self, start: int, stop: int):
        """Slice canonical rows, rebasing predecessors without changing the parent.

        Prefix views retain every pointer for decode capture/replay. A mixed
        suffix owns only its rebased predecessor indices; addresses remain views.
        """
        plan = V41CompressorPlan(*(value[start:stop] for value in self))
        if start:
            plan = plan._replace(previous=(plan.previous - start).clamp_min(-1))
        return plan


@dataclass
class V41Metadata:
    block_tables: dict[str, torch.Tensor]
    positions: torch.Tensor
    request_indices: torch.Tensor
    request_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    num_extends: int
    swa_write_slots: torch.Tensor
    swa_read_slots: torch.Tensor | None
    swa_read_lens: torch.Tensor | None
    compressor: V41CompressorPlan
    # [bs] first position whose global/index rows this forward may write; rows
    # below it (a prefix hit's replay window) are read-only. None on decode.
    global_write_floor: torch.Tensor | None


class V41PrefillSpan(NamedTuple):
    """One request's consecutive query rows in a canonical prefill window."""

    request: int
    offset: int
    prefix: int
    count: int
    # First SWA key position a query may see. Ordinary rows look back 127
    # positions into their own retained rows; replayed rows and CED decoder
    # rows start at their window (bounded replay truncation).
    swa_prefix_begin: int


class V41RowPlan(NamedTuple):
    """Rows one layer's attention receives and rows it attends.

    ``source`` describes the hidden rows the layer is called with; a KV owner
    writes its global rows for all of them. ``query`` describes the rows the
    layer attends and returns; ``keep_rows`` gathers them out of the source
    rows, None when the two are the same window. Only the first CED decoder
    layer narrows: it projects the decoder's global KV from every encoder row
    and then attends the per-request tail only.
    """

    source: V41Metadata
    query: V41Metadata
    keep_rows: torch.Tensor | None


class V41DecoderView(NamedTuple):
    """Rows the CED decoder layers run on: each extend request's tail, then decode rows.

    ``keep_rows`` gathers them from the full query window (None: identity);
    ``logits_rows`` gathers the sampled rows from the view (None: identity).
    """

    metadata: V41Metadata
    prefill: V41Metadata | None
    spans: tuple[V41PrefillSpan, ...]
    keep_rows: torch.Tensor | None
    logits_rows: torch.Tensor | None


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
    physical_slots: torch.Tensor


@dataclass
class V41PrefillRequestPlan:
    request: int
    rows: slice | torch.Tensor
    last_position: int
    prefix_slots: torch.Tensor
    swa_indices: torch.Tensor
    history_slots: dict[int, torch.Tensor]


@dataclass
class V41SWAQueryPlan:
    positions: torch.Tensor
    request_indices: torch.Tensor
    requests: tuple[V41PrefillRequestPlan, ...]


class DeepseekV41AttentionBackend(AttentionBackend):
    # Decode uses fixed-capacity rows and refresh-time history validation. The
    # CED decoder runs on a per-request tail of the prefill rows, so a prefill
    # forward changes its row count mid-way and cannot be captured as one
    # token-shaped graph.
    cuda_graph_support = CudaGraphSupport(decode_graph=True, prefill_graph=False)
    supports_layer_sliding_window = True

    def __init__(self, config: AttnConfig, spec: DeepseekV41Config) -> None:
        super().__init__(config, spec)
        if config.is_draft:
            raise NotImplementedError("V4.1 FlatKV implements target attention only")
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
        self._decode_history_status: torch.Tensor | None = None
        self._max_decode_bs = 0
        self._swa_plans: dict[tuple[ForwardMode, int], V41SWAQueryPlan] = {}
        self._prefill_spans: tuple[V41PrefillSpan, ...] = ()
        self._decoder_view: V41DecoderView | None = None
        self._decode_schedule_keepalive: list[object] = []
        self._prepared_selections: dict[tuple, tuple] = {}

    def validate_cache_pool(self, cache_pool: CachePool) -> None:
        super().validate_cache_pool(cache_pool)
        if not isinstance(cache_pool, DeepseekV41CachePool):
            raise TypeError("V4.1 backend requires DeepseekV41CachePool")
        specs = {s.group_id: s for s in cache_pool.arena.cache_group_specs}
        for gid, (rows, stride) in V41_GROUP_GEOMETRY.items():
            if gid not in specs or (
                specs[gid].rows_per_page,
                specs[gid].entry_stride_tokens,
            ) != (rows, stride):
                raise ValueError(f"V4.1 pool is missing the {gid} row geometry")
        # The write mask and SWA truncation assume these two groups are private
        # to the request: regenerated by replay over their whole retention
        # window, never shared through a hit.
        for gid in (V41_SWA_GROUP_ID, V41_COMPRESSOR_TAIL_GROUP_ID):
            if specs[gid].replay_window_tokens != specs[gid].sliding_window_tokens:
                raise ValueError(
                    f"V4.1 {gid} must declare replay_window_tokens equal to its "
                    "sliding_window_tokens"
                )

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self.forward_metadata = None
        self.forward_prefill_metadata = None
        self.forward_decode_metadata = None
        self._decode_views_by_bs.clear()
        self._decode_buffers = None
        self._decode_history_status = None
        self._max_decode_bs = 0
        self._swa_plans.clear()
        self._prefill_spans = ()
        self._decoder_view = None
        self._decode_schedule_keepalive.clear()
        self._prepared_selections.clear()

    def init_cuda_graph_state(
        self,
        max_bs: int,
        *,
        max_tokens_per_req: int,
        overlap_schedule_depth: int,
        **kwargs,
    ) -> None:
        """Allocate request tables and packed token views for eager and graph verify."""
        del kwargs
        if max_tokens_per_req != self.spec_num_tokens:
            raise ValueError(
                "V4.1 decode capacity must use the configured verify width"
            )
        max_tokens = max_bs * self.spec_num_tokens
        if max_tokens > self.spec.max_query_tokens:
            raise ValueError("V4.1 decode capacity exceeds budgeted query workspace")
        if self._decode_buffers is not None:
            if max_bs != self._max_decode_bs:
                raise RuntimeError(
                    "V4.1 decode capacity cannot change after initialization"
                )
            return
        self._max_decode_bs = max_bs
        self._decode_history_status = torch.empty(
            max_tokens, dtype=torch.int32, device=self.device
        )
        horizon = (1 + overlap_schedule_depth) * max_tokens_per_req
        self._decode_buffers = V41Metadata(
            block_tables={
                gid: torch.zeros((max_bs, width), dtype=torch.int32, device=self.device)
                for gid, width in v41_table_widths(self.context_len, horizon).items()
            },
            positions=torch.full(
                (max_tokens,), -1, dtype=torch.int64, device=self.device
            ),
            request_indices=torch.full(
                (max_tokens,), -1, dtype=torch.int64, device=self.device
            ),
            request_pool_indices=torch.full(
                (max_bs,), -1, dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            num_extends=0,
            swa_write_slots=torch.full(
                (max_tokens,), -1, dtype=torch.int64, device=self.device
            ),
            swa_read_slots=torch.full(
                (max_tokens, 128), -1, dtype=torch.int32, device=self.device
            ),
            swa_read_lens=torch.zeros(
                max_tokens, dtype=torch.int32, device=self.device
            ),
            compressor=V41CompressorPlan.allocate(max_tokens, self.device),
            global_write_floor=None,
        )

    def _decode_view(self, bs: int) -> V41Metadata:
        if self._decode_buffers is None or not 0 <= bs <= self._max_decode_bs:
            raise ValueError("V4.1 decode batch exceeds initialized capacity")
        if bs not in self._decode_views_by_bs:
            b = self._decode_buffers
            num_tokens = bs * self.spec_num_tokens
            self._decode_views_by_bs[bs] = V41Metadata(
                {gid: t[:bs] for gid, t in b.block_tables.items()},
                b.positions[:num_tokens],
                b.request_indices[:num_tokens],
                b.request_pool_indices[:bs],
                b.seq_lens[:bs],
                0,
                b.swa_write_slots[:num_tokens],
                b.swa_read_slots[:num_tokens],
                b.swa_read_lens[:num_tokens],
                b.compressor.window(0, num_tokens),
                None,
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
        # The configured width owns the shape even for callers that supply only
        # request metadata. The runner's explicit packed count must agree.
        if "num_tokens" in kwargs and kwargs["num_tokens"] != bs * self.spec_num_tokens:
            raise ValueError("V4.1 num_tokens must equal batch size times verify width")
        del for_graph_replay, kwargs
        self._prepared_selections.clear()
        if not forward_mode.is_decode_or_idle():
            raise ValueError("V4.1 refresh requires decode or idle mode")
        if not 0 <= num_extends <= actual_bs <= bs:
            raise ValueError("V4.1 invalid live/decode batch sizes")
        self._check_tables(block_tables, actual_bs)
        if num_extends:
            for key in [k for k in self._swa_plans if k[0] is ForwardMode.DECODE]:
                del self._swa_plans[key]
        else:
            self._swa_plans.clear()
        meta = self._decode_view(bs)
        from tokenspeed_kernel.ops.attention.mla._triton.page_table import (
            copy_page_table,
        )

        for gid, dest in meta.block_tables.items():
            src = block_tables[gid]
            if src.shape[1] > dest.shape[1]:
                raise ValueError(
                    f"V4.1 {gid} table exceeds initialized context capacity"
                )
            copy_page_table(src, dest, actual_bs)
        meta.num_extends = num_extends
        self.forward_metadata = self.forward_decode_metadata = meta
        from tokenspeed_kernel.ops.attention.dsv41 import decode_rows

        decode_rows(
            seq_lens,
            req_pool_indices,
            meta.positions,
            meta.request_indices,
            meta.seq_lens,
            meta.request_pool_indices,
            actual_bs,
            num_extends,
            self.spec_num_tokens,
        )
        self._prepare_compressor(meta)
        self._refresh_decode_window(meta, actual_bs)
        if num_extends:
            self.sparse_topk.decode = None
        else:
            self.forward_prefill_metadata = None
            # Decode rows all reach the decoder: an identity view, no tensors.
            self._decoder_view = V41DecoderView(meta, None, (), None, None)
            self.sparse_topk.clear()

    def _prepare_compressor(self, metadata: V41Metadata) -> None:
        from tokenspeed_kernel.ops.attention.dsv41 import compressor_metadata

        gid = V41_COMPRESSOR_TAIL_GROUP_ID
        compressor_metadata(
            metadata.positions,
            metadata.request_indices,
            metadata.block_tables[gid],
            self.cache_pool.arena.cache_group_page_counts[gid],
            *metadata.compressor,
        )

    def window_slots(
        self, group_id: str, positions: torch.Tensor, request_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve each decode query's 128-row window in ``group_id`` to slots.

        Args:
            group_id: A group with the SWA row geometry.
            positions: Absolute query positions, one per decode token row.
            request_indices: Batch rows of the decode metadata, same shape.

        Returns:
            ``[n, 128]`` int32 slots of positions ``p-127..p`` (``-1`` before
            the sequence start) and the ``[n]`` int32 count of live rows.
        """
        wanted = (positions - 127).clamp_min(0)[:, None] + torch.arange(
            128, device=self.device
        )
        wanted = wanted.masked_fill(wanted > positions[:, None], -1)
        slots = self.cache_slots(
            group_id,
            wanted,
            request_indices[:, None].expand_as(wanted),
            ForwardMode.DECODE,
        ).to(torch.int32)
        return slots, (positions + 1).clamp(0, 128).to(torch.int32)

    def _refresh_decode_window(self, metadata, actual_bs):
        """Build query addresses and check dependencies before any layer runs.

        Eager, mixed, capture setup and live graph replay use the same producer.
        All persistent outputs are refreshed in place; history remains LCM-owned.
        """
        from tokenspeed_kernel.ops.attention.dsv41 import decode_window

        n = metadata.positions.numel()
        if not n:
            return
        assert self._decode_history_status is not None
        status = self._decode_history_status[:n]
        counts = self.cache_pool.arena.cache_group_page_counts
        decode_window(
            metadata.positions,
            metadata.request_indices,
            metadata.swa_write_slots,
            metadata.swa_read_slots,
            metadata.swa_read_lens,
            status,
            metadata.block_tables[V41_SWA_GROUP_ID],
            metadata.block_tables[V41_COMPRESSOR_TAIL_GROUP_ID],
            counts[V41_SWA_GROUP_ID],
            counts[V41_COMPRESSOR_TAIL_GROUP_ID],
        )
        # The fused producer reports only span-start errors. Observe all token
        # rows: request starts need not have a uniform stride in mixed batches.
        # Padding, leading extends and internal pairs contribute no errors.
        errors = status.cpu().tolist() if actual_bs else []
        if any(error & 1 for error in errors):
            raise RuntimeError(
                "V4.1 SWA prefix is missing; supply the dependency tail or recover the request"
            )
        if any(error & 2 for error in errors):
            raise RuntimeError(
                "V4.1 required compressor tail is absent; request needs recovery"
            )

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
        extend_replay_lens_cpu: torch.Tensor,
        extend_prompt_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ) -> None:
        del extend_with_prefix
        if forward_mode.is_decode():
            raise ValueError("V4.1 decode metadata must use refresh_decode_metadata")
        self._check_tables(block_tables, bs)
        self._swa_plans.clear()
        self._prepared_selections.clear()
        if not 0 <= num_extends <= bs:
            raise ValueError("V4.1 invalid extend/decode batch sizes")
        width = self.spec_num_tokens
        window = self.spec.sliding_window_tokens
        counts = [int(n) for n in extend_seq_lens_cpu[:num_extends].tolist()] + [
            width
        ] * (bs - num_extends)
        total = sum(counts)
        if "num_tokens" in kwargs and kwargs["num_tokens"] != total:
            raise ValueError("V4.1 num_tokens disagrees with extend/verify lengths")
        prefixes = [int(n) for n in extend_prefix_lens_cpu[:num_extends].tolist()]
        replays = [int(n) for n in extend_replay_lens_cpu[:num_extends].tolist()]
        prompt_lens = [int(n) for n in extend_prompt_lens_cpu[:num_extends].tolist()]
        offset = 0
        spans = []
        floors = []
        completes = []
        for request, (prefix, count, replay, prompt_len) in enumerate(
            zip(prefixes, counts[:num_extends], replays, prompt_lens, strict=True)
        ):
            if not 0 <= replay <= count or prefix + count > prompt_len:
                raise ValueError("V4.1 replay/prompt lengths disagree with the extend")
            # Replayed rows regenerate the SWA window from its first position;
            # an ordinary chunk continues from its own retained rows.
            swa_prefix_begin = prefix if replay else max(0, prefix - (window - 1))
            spans.append(
                V41PrefillSpan(request, offset, prefix, count, swa_prefix_begin)
            )
            floors.append(prefix + replay)
            completes.append(prefix + count == prompt_len)
            offset += count
        self._prefill_spans = tuple(spans)
        if total > self.spec.max_query_tokens:
            raise ValueError("V4.1 forward exceeds budgeted query workspace")
        lengths = torch.tensor(counts, dtype=torch.int64, device=self.device)
        requests = torch.repeat_interleave(
            torch.arange(bs, device=self.device), lengths, output_size=total
        )
        starts = lengths.cumsum(0) - lengths
        prefix = torch.cat(
            (
                extend_prefix_lens[:num_extends],
                seq_lens[num_extends:bs].clamp_min(width) - width,
            )
        ).to(torch.int64)
        positions = (
            torch.arange(total, device=self.device)
            - starts[requests]
            + prefix[requests]
        )
        seq_lens = torch.cat(
            (seq_lens[:num_extends], seq_lens[num_extends:bs].clamp_min(width))
        )
        tables = {
            gid: t[:bs] for gid, t in block_tables.items() if gid in V41_GROUP_GEOMETRY
        }
        global_write_floor = torch.tensor(
            floors + [0] * (bs - num_extends), dtype=torch.int64, device=self.device
        )
        meta = V41Metadata(
            tables,
            positions,
            requests,
            req_pool_indices[:bs],
            seq_lens[:bs],
            num_extends,
            torch.empty(total, dtype=torch.int64, device=self.device),
            None,
            None,
            V41CompressorPlan.allocate(total, self.device),
            global_write_floor,
        )
        self._prepare_compressor(meta)
        plan = meta.compressor
        if bool((plan.active & (plan.previous < 0) & (plan.read_slots < 0)).any()):
            raise RuntimeError(
                "V4.1 required compressor tail is absent; request needs recovery"
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
            None,
            None,
            meta.compressor.window(0, n),
            global_write_floor,
        )
        self.forward_decode_metadata = V41Metadata(
            tables,
            positions[n:],
            requests[n:],
            meta.request_pool_indices,
            meta.seq_lens,
            num_extends,
            meta.swa_write_slots[n:],
            torch.empty((total - n, 128), dtype=torch.int32, device=self.device),
            torch.empty(total - n, dtype=torch.int32, device=self.device),
            meta.compressor.window(n, total),
            None,
        )
        self._refresh_decode_window(self.forward_decode_metadata, bs - num_extends)
        self._decoder_view = self._build_decoder_view(
            meta, completes, window, forward_mode
        )
        self.sparse_topk.clear()

    def _build_decoder_view(
        self,
        meta: V41Metadata,
        completes: list[bool],
        window: int,
        forward_mode: ForwardMode,
    ) -> V41DecoderView:
        """Select the rows the CED decoder layers run on.

        The decoder attends the prompt's last window only, so a chunk that ends
        its prompt keeps its last ``window`` rows and any other chunk keeps one
        row (its logits are discarded, and one row per request keeps the
        sampler's row contract). Decode rows are all kept. The scheduler never
        leaves a final chunk shorter than the window, so a kept tail is the
        prompt's last window unless the whole prompt is shorter.
        """
        n = self.forward_prefill_metadata.positions.numel()
        total = meta.positions.numel()
        keeps = [
            min(window, span.count) if done else min(1, span.count)
            for span, done in zip(self._prefill_spans, completes, strict=True)
        ]
        if all(keep == span.count for keep, span in zip(keeps, self._prefill_spans)):
            # Identity: the decoder sees exactly the forward's query window.
            return V41DecoderView(
                self.query_metadata(forward_mode),
                self.forward_prefill_metadata,
                self._prefill_spans,
                None,
                None,
            )
        rows = []
        spans = []
        logits_rows = []
        view_offset = 0
        for span, keep in zip(self._prefill_spans, keeps, strict=True):
            begin = span.offset + span.count - keep
            rows.extend(range(begin, span.offset + span.count))
            if keep:
                first_position = span.prefix + span.count - keep
                spans.append(
                    V41PrefillSpan(
                        span.request, view_offset, first_position, keep, first_position
                    )
                )
                logits_rows.append(view_offset + keep - 1)
            view_offset += keep
        k = view_offset
        rows.extend(range(n, total))
        logits_rows.extend(range(k, k + total - n))
        keep_rows = torch.tensor(rows, dtype=torch.int64, device=self.device)
        positions = meta.positions[keep_rows]
        requests = meta.request_indices[keep_rows]
        write_slots = meta.swa_write_slots[keep_rows]
        view = V41Metadata(
            meta.block_tables,
            positions,
            requests,
            meta.request_pool_indices,
            meta.seq_lens,
            meta.num_extends,
            write_slots,
            None,
            None,
            meta.compressor.window(0, 0),
            meta.global_write_floor,
        )
        # Without decode rows the view is its own prefill window: consumers
        # hand back the same tensors, and _window recognizes them by identity.
        prefill = (
            view
            if k == len(rows)
            else V41Metadata(
                meta.block_tables,
                positions[:k],
                requests[:k],
                meta.request_pool_indices,
                meta.seq_lens,
                meta.num_extends,
                write_slots[:k],
                None,
                None,
                meta.compressor.window(0, 0),
                meta.global_write_floor,
            )
        )
        return V41DecoderView(
            view,
            prefill,
            tuple(spans),
            keep_rows,
            torch.tensor(logits_rows, dtype=torch.int64, device=self.device),
        )

    def decoder_view(self) -> V41DecoderView:
        """Rows for the CED decoder layers of the current forward (see V41DecoderView)."""
        if self._decoder_view is None:
            raise RuntimeError("V4.1 metadata has not been prepared for this forward")
        return self._decoder_view

    def _window(self, positions, requests, forward_mode):
        """Canonical window backing (positions, requests), or None for a free subset.

        Canonical windows are the full query metadata of the mode and the CED
        decoder view's prefill rows; both carry host spans and write slots, so
        planning them needs no device snapshot.
        """
        meta = self.query_metadata(forward_mode)
        if positions is meta.positions and requests is meta.request_indices:
            return meta, self._prefill_spans
        view = self._decoder_view
        if (
            view is not None
            and view.prefill is not None
            and positions is view.prefill.positions
            and requests is view.prefill.request_indices
        ):
            return view.prefill, view.spans
        return None

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
        from tokenspeed_kernel.ops.attention.mla._triton.page_table import (
            bounded_group_slots,
        )

        return bounded_group_slots(
            positions,
            request_indices,
            meta.block_tables[group_id],
            rows,
            stride,
            1,
            self.cache_pool.arena.cache_group_page_counts[group_id],
        )

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
        from tokenspeed_kernel.ops.attention.dsv41 import global_slots

        return global_slots(
            logical_rows,
            positions,
            request_indices,
            self.query_metadata(forward_mode).block_tables[gid],
            ratio,
            self.cache_pool.arena.cache_group_page_counts[gid],
        )

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
        if content.is_cuda:
            from tokenspeed_kernel.ops.attention import dsv41

            dsv41.compressor_tail_scatter(content, scores, tail, slots)
        else:
            live = slots >= 0
            tail[slots[live] // 2, slots[live] % 2, 0] = content[live].float()
            tail[slots[live] // 2, slots[live] % 2, 1] = scores[live].float()

    def compress(
        self,
        owner,
        content,
        scores,
        forward_mode,
        norm_weight,
        norm_eps,
    ):
        """Pool projected inputs in query_metadata(forward_mode) row order.

        Inputs: FP32 [T,512] content/scores for the complete canonical query
        window, plus optional RMSNorm weight and epsilon. Reordered/subset rows
        must not be passed to this entry point. Pairing and tail addresses come
        from the current forward's shared compressor plan; all request history
        stays in LCM. Metadata preparation validates dependencies before layers.

        Returns pooled rows and pair-start positions/requests, all with capacity
        T on eager and graph paths. Inactive outputs are zero with coordinates
        -1. With norm_weight, output is BF16 and pooling is rounded to BF16 BEFORE
        RMSNorm; otherwise output is FP32. RoPE remains model-owned.
        """
        meta = self.query_metadata(forward_mode)
        positions = meta.positions
        active, pair_positions, pair_requests, previous, slots, write_slots = (
            meta.compressor
        )
        if content.dtype != torch.float32 or scores.dtype != torch.float32:
            raise ValueError("V4.1 ratio-2 projection/pooling requires FP32 inputs")
        if self.spec.compress_ratios[owner] != 2:
            raise ValueError("compress() is only for ratio-2 owners")
        self._owner_group(owner)
        if content.shape != (positions.numel(), 512) or scores.shape != content.shape:
            raise ValueError("compressor content/scores must be [tokens, 512]")
        # Read before any writes: a scheduler-reused tail page must not destroy
        # the odd-prefix input needed by the first completed pair in this chunk.
        tail = self.cache_pool.compressor_tail(owner)
        if content.is_cuda:
            from tokenspeed_kernel.ops.attention import dsv41

            pooled = dsv41.compressor_pool(
                content,
                scores,
                previous,
                tail,
                slots,
                active,
                None,
                norm_weight,
                norm_eps,
            )
        else:
            missing = active & (previous < 0)
            pooled = torch.empty_like(content)
            for start in range(0, positions.numel(), 8):
                stop = min(start + 8, positions.numel())
                slot = slots[start:stop].clamp_min(0)
                history = tail[slot // 2, slot % 2]
                prior = previous[start:stop].clamp_min(0)
                prior_content = torch.where(
                    missing[start:stop, None], history[:, 0], content[prior]
                )
                prior_scores = torch.where(
                    missing[start:stop, None], history[:, 1], scores[prior]
                )
                weights = torch.stack(
                    (prior_scores, scores[start:stop]), dim=1
                ).softmax(1)
                pooled[start:stop] = (
                    weights[:, 0] * prior_content + weights[:, 1] * content[start:stop]
                ).masked_fill(~active[start:stop, None], 0)
        if not content.is_cuda and norm_weight is not None:
            normalized = pooled.to(torch.bfloat16).float()
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(-1, keepdim=True) + norm_eps
            )
            pooled = (normalized * norm_weight.float()).to(torch.bfloat16)
        if content.is_cuda:
            dsv41.compressor_tail_scatter(content, scores, tail, write_slots)
        else:
            live = write_slots >= 0
            tail[write_slots[live] // 2, write_slots[live] % 2, 0] = content[live]
            tail[write_slots[live] // 2, write_slots[live] % 2, 1] = scores[live]
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
        if not forward_mode.is_decode() and bool(
            ((positions >= 0) & (positions % ratio != 0)).any()
        ):
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
        floor = self.query_metadata(forward_mode).global_write_floor
        if floor is not None:
            # A replayed row's global rows already sit in the shared pages the
            # prefix hit claimed: skip every row whose last covered position
            # lies below the request's write floor (a negative slot is skipped
            # by the scatter kernels).
            below = (positions >= 0) & (
                positions + (ratio - 1) < floor[request_indices.clamp_min(0)]
            )
            slots = slots.masked_fill(below, -1)
        dsv41.cache_scatter(main_kv, self.cache_pool.global_kv(owner), slots, "global")
        dsv41.cache_scatter(index_k, self.cache_pool.index_k(owner), slots, "index")

    def _selection_rows(self, record, positions, requests, forward_mode):
        if positions is record.positions and requests is record.request_indices:
            return torch.arange(positions.numel(), device=positions.device).masked_fill(
                (positions < 0) | (requests < 0), -1
            )
        index = self._lookup_rows(
            record.positions, record.request_indices, positions, requests
        )
        # A full decode source covers the refresh-validated request window,
        # including reordered/subset consumers. Noncanonical source windows
        # retain the explicit coverage check used by arbitrary-chunk prefill.
        meta = self.query_metadata(forward_mode)
        full_decode_source = (
            forward_mode.is_decode()
            and record.positions is meta.positions
            and record.request_indices is meta.request_indices
        )
        if not full_decode_source and bool(((index < 0) & (positions >= 0)).any()):
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
        Decode graph Full sources use the refresh's full query window; consumers
        may reorder/subset it. A subset Reindex and its Reuse consumers share the
        same window tensors. Other source windows retain eager coverage validation.
        Full scans device-visible history in fixed partitions; Reindex scans
        shape-bounded candidates. TP gathers small Q/weight tiles once, not
        scores per history tile. No full-history query/score matrix is materialized.
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
            if (
                positions is record.positions
                and request_indices is record.request_indices
            ):
                return record.physical_slots, record.lengths
            index = self._selection_rows(
                record, positions, request_indices, forward_mode
            )
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
        same_candidate_rows = reindex and (
            positions is candidates.positions
            and request_indices is candidates.request_indices
        )
        candidate_index = (
            self._selection_rows(candidates, positions, request_indices, forward_mode)
            if reindex and not same_candidate_rows
            else None
        )
        produce_candidates = layer_id == self.spec.candidate_source
        metadata = self.query_metadata(forward_mode)
        table = metadata.block_tables[self._owner_group(owner)]
        ratio = self.spec.compress_ratios[owner]
        window = (
            self._window(positions, request_indices, forward_mode)
            if forward_mode.is_extend()
            else None
        )
        spans = window[1] if window is not None else ()
        if window is not None:
            visible_bound = max(
                ((span.prefix + span.count) // ratio for span in spans),
                default=0,
            )
            table = table[:, : max(1, (visible_bound + 63) // 64)]
        index_cache = self.cache_pool.index_k(owner)
        # Full-history residency bounds logical pages. Trim replicated table
        # scratch; the kernel separately bounds work by device-visible lengths.
        table = table[:, : max(0, index_cache.shape[0] - 1)]
        candidate_capacity = min(self.spec.candidate_topk, table.shape[1] * 8)
        n = positions.numel()
        rows = torch.full(
            (n, self.spec.index_topk), -1, dtype=torch.int32, device=self.device
        )
        lens = torch.zeros(n, dtype=torch.int32, device=self.device)
        blocks = torch.full(
            (n, candidate_capacity if produce_candidates else 0),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        block_lens = torch.zeros(n, dtype=torch.int32, device=self.device)
        # Query tiling bounds both score scratch and replicated table rows. Do
        # not index_select the table for the entire prefill (T * context pages).
        canonical_prefill = window is not None
        query_tile = 1024 if canonical_prefill else 256
        windows = (
            [
                (span.offset, span.offset + span.count, span)
                for span in spans
                if span.count
            ]
            if canonical_prefill
            else [
                (start, min(start + query_tile, n), None)
                for start in range(0, n, query_tile)
            ]
        )
        for start, stop, span in windows:
            if span is not None:
                visible = (
                    ((positions[start:stop] + 1) // ratio).clamp_min(0).to(torch.int32)
                )
                width = max(1, ((span.prefix + span.count) // ratio + 63) // 64)
                row = table[span.request : span.request + 1, :width]
                row = row.masked_fill(row <= 0, -1)
                tile_table = row.expand(stop - start, -1)
            else:
                from tokenspeed_kernel.ops.attention.dsv41 import selection_table

                tile_table, visible = selection_table(
                    positions[start:stop], request_indices[start:stop], table, ratio
                )
            cb = None
            if reindex:
                if same_candidate_rows:
                    cb = candidates.block_ids[start:stop]
                else:
                    ci = candidate_index[start:stop]
                    cb = candidates.block_ids[ci.clamp_min(0)].masked_fill(
                        ci[:, None] < 0, -1
                    )
            dsv41.index_topk(
                index_q[start:stop],
                index_weights[start:stop],
                index_cache,
                tile_table,
                visible,
                cb,
                self.spec.index_topk,
                candidate_capacity if produce_candidates else 0,
                self.spec.candidate_block_size,
                query_tile,
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
            candidates = V41Candidates(positions, request_indices, blocks, block_lens)
        slots = self.global_read_slots(
            owner, rows, positions, request_indices, forward_mode
        )
        record = V41Selection(
            owner,
            source,
            positions,
            request_indices,
            rows,
            lens,
            candidates,
            slots,
        )
        if forward_mode.is_decode():
            share.decode = record
        else:
            share.prefill = record
        return slots, lens

    def prepare_global_selection(
        self,
        layer_id,
        index_q,
        index_weights,
        positions,
        requests,
        forward_mode,
        index_process_group,
    ):
        """Prepare source selection on the caller's stream for this one forward."""
        result = self.select_global(
            layer_id,
            index_q,
            index_weights,
            positions,
            requests,
            forward_mode,
            index_process_group,
        )
        key = (layer_id, forward_mode, id(positions), id(requests))
        self._prepared_selections[key] = (positions, requests, result)
        return result

    def _swa_query_plan(self, positions, requests, forward_mode) -> V41SWAQueryPlan:
        """Validate one prefill query window and retain only address scratch.

        Canonical spans reuse the scheduler's CPU lengths. Arbitrary reordered
        or subset windows take one host metadata snapshot. Strong references
        distinguish equal-sized windows; only the latest plan per mode is held.
        Every metadata refresh invalidates these plans. No KV payload is cached.
        """
        key = (forward_mode, id(positions))
        plan = self._swa_plans.get(key)
        if (
            plan is not None
            and plan.positions is positions
            and plan.request_indices is requests
        ):
            return plan
        window = self._window(positions, requests, forward_mode)
        groups = []
        if window is not None:
            for span in window[1]:
                if span.count:
                    groups.append(
                        (
                            span.request,
                            slice(span.offset, span.offset + span.count),
                            span.prefix,
                            span.count,
                            span.swa_prefix_begin,
                        )
                    )
        else:
            # This is prefill-only control metadata. All layer consumers share
            # the resulting plan; decode/capture never reads device values here.
            snapshot = torch.stack((requests, positions), dim=-1).cpu().tolist()
            grouped = {}
            for row, (request, position) in enumerate(snapshot):
                if request >= 0 and position >= 0:
                    grouped.setdefault(request, []).append((row, position))
            for request, entries in grouped.items():
                rows = torch.tensor([row for row, _ in entries], device=self.device)
                groups.append(
                    (request, rows, [position for _, position in entries], None, 0)
                )
        plans = []
        for request, rows, prefix, count, swa_prefix_begin in groups:
            current_positions = positions[rows]
            if count is not None:
                # Keys before swa_prefix_begin are outside the window: a replayed
                # or CED decoder row sees only [max(begin, i - 127), i].
                prefix_positions = torch.arange(
                    swa_prefix_begin, prefix, device=self.device
                )
                last_position = prefix + count - 1
            else:
                current = set(prefix)
                needed = set()
                for position in current:
                    needed.update(range(max(0, position - 127), position + 1))
                prefix_positions = torch.tensor(
                    sorted(needed - current), dtype=torch.int64, device=self.device
                )
                last_position = max(prefix)
            prefix_slots = self.cache_slots(
                V41_SWA_GROUP_ID,
                prefix_positions,
                torch.full_like(prefix_positions, request),
                forward_mode,
            )
            if prefix_slots.numel() and bool((prefix_slots < 0).any()):
                raise RuntimeError(
                    "V4.1 SWA prefix is missing; supply the dependency tail or recover the request"
                )
            workspace_positions = torch.cat((prefix_positions, current_positions))
            keys, order = workspace_positions.sort()
            wanted = current_positions[:, None] - torch.arange(
                127, -1, -1, device=self.device
            )
            lookup = torch.searchsorted(keys, wanted.contiguous()).clamp_max(
                keys.numel() - 1
            )
            indices = (
                order[lookup]
                .masked_fill((wanted < 0) | (keys[lookup] != wanted), -1)
                .to(torch.int32)
            )
            plans.append(
                V41PrefillRequestPlan(
                    request, rows, last_position, prefix_slots, indices, {}
                )
            )
        plan = V41SWAQueryPlan(positions, requests, tuple(plans))
        self._swa_plans[key] = plan
        return plan

    def _prefill_global_history(self, plan, owner, forward_mode):
        ratio = self.spec.compress_ratios[owner]
        if ratio not in plan.history_slots:
            positions = (
                torch.arange((plan.last_position + 1) // ratio, device=self.device)
                * ratio
            )
            plan.history_slots[ratio] = self.cache_slots(
                self._owner_group(owner),
                positions,
                torch.full_like(positions, plan.request),
                forward_mode,
            )
        return plan.history_slots[ratio]

    def _decode_schedule(self):
        from tokenspeed_kernel.ops.attention import dsv41

        # Each call owns a fresh scheduler producer. Reusing initialized warmup
        # metadata during capture would omit its length-dependent GPU work.
        schedule = dsv41.new_attention_schedule()
        if schedule is not None and torch.cuda.is_current_stream_capturing():
            self._decode_schedule_keepalive.append(schedule)
        return schedule

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
        swa_rope_cache,
    ) -> torch.Tensor:
        """Joint SWA/global attention; returns BF16 [T,Hlocal,512], BEFORE inverse RoPE.

        q and swa_kv are post-RoPE [T,Hlocal,512] / [T,512]. Each SWA input
        corresponds to the explicit query positions/requests. Global memory is
        written separately by write_global; index inputs follow select_global.
        Prefill SWA writes follow all old-prefix reads, using only LCM-retained
        pages. Decode publishes the current row before its paged read; it cannot
        alias the preceding 127 logical rows. CED callers supply the required SWA
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
            meta = self.query_metadata(forward_mode)
            view = self._decoder_view
            # The full window and the decoder view are both canonical: their
            # extend rows carry host spans (the view's prefill sub-window) and
            # their decode rows are the same refreshed decode window.
            if positions is meta.positions and request_indices is meta.request_indices:
                extend_window = self.forward_prefill_metadata
            elif (
                view is not None
                and positions is view.metadata.positions
                and request_indices is view.metadata.request_indices
            ):
                extend_window = view.prefill
            else:
                extend_window = None
            canonical = extend_window is not None
            boundary = meta.num_extends
            extend_tokens = extend_window.positions.numel() if canonical else 0
            out = torch.empty_like(q)
            for mode, rows in (
                (
                    ForwardMode.EXTEND,
                    (
                        slice(0, extend_tokens)
                        if canonical
                        else request_indices < boundary
                    ),
                ),
                (
                    ForwardMode.DECODE,
                    (
                        slice(extend_tokens, n)
                        if canonical
                        else request_indices >= boundary
                    ),
                ),
            ):
                window = (
                    extend_window if mode.is_extend() else self.query_metadata(mode)
                )
                out[rows] = self.forward_v41(
                    q[rows],
                    swa_kv[rows],
                    layer_id=layer_id,
                    positions=window.positions if canonical else positions[rows],
                    request_indices=(
                        window.request_indices if canonical else request_indices[rows]
                    ),
                    forward_mode=mode,
                    index_q=index_q[rows] if index_q is not None else None,
                    index_weights=(
                        index_weights[rows] if index_weights is not None else None
                    ),
                    attn_sink=attn_sink,
                    softmax_scale=softmax_scale,
                    index_process_group=index_process_group,
                    swa_rope_cache=swa_rope_cache,
                )
            return out
        if n == 0:
            return torch.empty_like(q)
        prepared = self._prepared_selections.pop(
            (layer_id, forward_mode, id(positions), id(request_indices)), None
        )
        if prepared is None:
            global_slots, global_lens = self.select_global(
                layer_id,
                index_q,
                index_weights,
                positions,
                request_indices,
                forward_mode,
                index_process_group,
            )
        else:
            global_slots, global_lens = prepared[2]
        owner = self.spec.kv_owners[layer_id]
        global_cache = self.cache_pool.global_kv(owner) if owner >= 0 else None
        cache = self.cache_pool.swa(layer_id)
        metadata = self.query_metadata(forward_mode)
        window = self._window(positions, request_indices, forward_mode)
        canonical = window is not None
        locations = (
            window[0].swa_write_slots
            if canonical
            else self.cache_slots(
                V41_SWA_GROUP_ID, positions, request_indices, forward_mode
            )
        )
        if forward_mode.is_decode():
            # The current position cannot alias one of its previous 127 logical
            # rows. Ordinary retained-page writes therefore precede paged decode.
            if swa_rope_cache is None:
                dsv41.cache_scatter(swa_kv, cache, locations, "swa")
            else:
                dsv41.swa_rope_scatter(
                    swa_kv, positions, swa_rope_cache, cache, locations, None
                )
            swa_slots, swa_lens = (
                (metadata.swa_read_slots, metadata.swa_read_lens)
                if canonical
                else self.window_slots(V41_SWA_GROUP_ID, positions, request_indices)
            )
            return dsv41.selected_attention(
                q,
                cache,
                swa_slots,
                swa_lens,
                global_cache,
                global_slots,
                global_lens,
                attn_sink,
                softmax_scale,
                None,
                256,
                self._decode_schedule(),
                None,
                None,
            )
        plan = self._swa_query_plan(positions, request_indices, forward_mode)
        # Quantize current rows exactly as cache storage, without publishing them
        # until every request has read its old sliding prefix.
        prefixes = [
            dsv41.cache_gather(cache, request.prefix_slots, "swa", None)
            for request in plan.requests
        ]
        if swa_rope_cache is None:
            current = dsv41.cache_unpack(
                dsv41.cache_pack(swa_kv, "swa", None), "swa", None
            )
        else:
            current = torch.empty_like(swa_kv)
            dsv41.swa_rope_scatter(
                swa_kv, positions, swa_rope_cache, cache, locations, current
            )
        logical_rows = None
        if owner >= 0:
            record = self.sparse_topk.prefill
            if (
                record.positions is positions
                and record.request_indices is request_indices
            ):
                logical_rows = record.logical_rows
            else:
                selected = self._selection_rows(
                    record, positions, request_indices, forward_mode
                )
                logical_rows = record.logical_rows[selected.clamp_min(0)].masked_fill(
                    selected[:, None] < 0, -1
                )
        out = torch.zeros_like(q)
        for request_plan, prefix in zip(plan.requests, prefixes, strict=True):
            rows = request_plan.rows
            prefix_count = request_plan.prefix_slots.numel()
            current_rows = current[rows]
            swa_count = prefix_count + current_rows.shape[0]
            history_slots = (
                self._prefill_global_history(request_plan, owner, forward_mode)
                if owner >= 0
                else None
            )
            history_count = history_slots.numel() if history_slots is not None else 0
            workspace = current.new_empty((swa_count + history_count, 512))
            workspace[:prefix_count].copy_(prefix)
            workspace[prefix_count:swa_count].copy_(current_rows)
            indices = request_plan.swa_indices
            if owner >= 0:
                dsv41.cache_gather(
                    global_cache, history_slots, "global", workspace[swa_count:]
                )
                selected = logical_rows[rows]
                compressed_indices = torch.where(
                    (selected >= 0) & (global_slots[rows] >= 0),
                    selected + swa_count,
                    -1,
                )
                indices = torch.cat((indices, compressed_indices), dim=-1).to(
                    torch.int32
                )
            # Advanced request subsets do not provide a writable output view;
            # copy their compact result back explicitly after native execution.
            output = out[rows] if isinstance(rows, slice) else None
            result = dsv41.selected_attention(
                q[rows].contiguous(),
                cache,
                None,
                None,
                global_cache,
                None,
                None,
                attn_sink,
                softmax_scale,
                output,
                V41_PREFILL_QUERY_TILE,
                None,
                workspace[:, None, :],
                indices.contiguous(),
            )
            if output is None:
                out[rows] = result
            del workspace, current_rows, indices
        if swa_rope_cache is None:
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
