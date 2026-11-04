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

"""Distributed sparse selection over sharded Index-K storage."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.dsa import (
    DSAIndexCandidatePages,
    dsa_index_candidates,
    dsa_index_candidates_accepts_pages,
    dsa_merge_candidates,
    plan_dsa_index_candidates,
)
from tokenspeed_kernel.ops.attention.dsa._triton.candidate_exchange import (
    pack_index_candidates,
)

from tokenspeed.runtime.distributed.comm_ops import (
    all_gather,
    all_gather_single,
    all_to_all_single,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)

# Some leaves offset logits rows in int32: a tile's padded logits stay < 2**31.
_MAX_TILE_LOGITS = 1 << 31
_LOGITS_ROW_PADDING = 256


def _tile_rows(
    rows: int, columns: int, page_size: int, max_logits_bytes: int | None
) -> int:
    """Query rows per candidate tile: the logits budget, else every row,
    never more than 32-bit logits offsets allow."""
    width = columns * page_size
    limit = max(1, _MAX_TILE_LOGITS // (width + _LOGITS_ROW_PADDING))
    if max_logits_bytes is not None and max_logits_bytes > 0:
        limit = min(limit, max(1, max_logits_bytes // (width * 4)))
    return min(rows, limit)


# Per candidate: an 8-byte gathered record plus a 4-byte FP32 merge score.
_EXCHANGE_BYTES_PER_CANDIDATE = 12
# Row-owner: 8-byte records sent and received, FP32 score, int32 selection, any degree.
_ROW_OWNER_EXCHANGE_BYTES_PER_CANDIDATE = 24
# The int32 bits of FP32 -inf (0xFF800000): the score of a padding candidate.
_NEG_INF_BITS = -(1 << 23)


def _exchange_rows(
    rows: int,
    tile_rows: int,
    degree: int,
    topk: int,
    max_logits_bytes: int | None,
    *,
    row_owner: bool = False,
) -> int:
    """Query rows per candidate exchange: whole tiles whose exchange buffers
    fit the logits budget, at least one tile; every row without a budget.

    An exchange that merges every row on every rank holds ``degree * topk *
    12`` bytes per row, a ``row_owner`` exchange ``topk * 24``.
    """
    if max_logits_bytes is None or max_logits_bytes <= 0:
        return rows
    if row_owner:
        row_bytes = topk * _ROW_OWNER_EXCHANGE_BYTES_PER_CANDIDATE
    else:
        row_bytes = degree * topk * _EXCHANGE_BYTES_PER_CANDIDATE
    fitting = max_logits_bytes // row_bytes
    return max(1, fitting // tile_rows) * tile_rows


def _owned_tile_rows(
    rows: int,
    columns: int,
    full_tile_rows: int,
    page_size: int,
    degree: int,
    topk: int,
    max_logits_bytes: int | None,
    *,
    row_owner: bool,
) -> int:
    """Query rows per candidate tile over pages planned ``columns`` wide.

    Narrower logits fit more rows per tile, but an exchange joins whole
    tiles, so a tile is also held to the rows one exchange of the same kind
    (``row_owner`` or not, see :func:`_exchange_rows`) fits, or to the
    table-width tile where that is larger: no exchange grows past the
    table-width plan's."""
    tile_rows = _tile_rows(rows, columns, page_size, max_logits_bytes)
    exchange_rows = _exchange_rows(
        rows, 1, degree, topk, max_logits_bytes, row_owner=row_owner
    )
    return min(tile_rows, max(full_tile_rows, exchange_rows))


def _exchanges_packed_records(offsets: torch.Tensor, scores: torch.Tensor) -> bool:
    return (
        offsets.is_cuda
        and offsets.dtype == torch.int32
        and scores.dtype == torch.float32
    )


def merge_index_candidates(
    offsets: torch.Tensor,
    scores: torch.Tensor,
    *,
    topk: int,
    group: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge disjoint local candidates with one global Top-K selection.

    Communicates only K candidate positions and scores per query and rank.
    Int32 positions and FP32 scores share one all-gather as interleaved int32
    records; scores are reinterpreted bitwise, including mandatory +inf.
    Returns global logical positions and valid counts, identical on every rank.
    Positive infinity marks mandatory candidates; negative infinity is padding.
    Equal-score candidates have no prescribed logical-position order.
    """
    if offsets.shape != scores.shape or offsets.ndim != 2 or topk <= 0:
        raise ValueError("Candidate offsets/scores must be matching matrices")
    if _exchanges_packed_records(offsets, scores):
        packed = pack_index_candidates(offsets, scores)
        if len(group) > 1:
            records = torch.empty(
                (len(group), *packed.shape), dtype=packed.dtype, device=packed.device
            )
            all_gather_single(records.flatten(), packed.flatten(), group)
        else:
            records = packed.unsqueeze(0)
        return dsa_merge_candidates(records, topk=topk, solution=None)
    if len(group) > 1:
        if offsets.dtype == torch.int32 and scores.dtype == torch.float32:
            # Numeric conversion would lose large positions or score bits.
            # Stack also packs strided producer views into contiguous records.
            packed = torch.stack((offsets, scores.view(torch.int32)), dim=-1)
            gathered = all_gather(packed.flatten(1), group, dim=-1)
            candidates = gathered.reshape(
                offsets.shape[0], offsets.shape[1] * len(group), packed.shape[-1]
            )
            offsets, score_bits = candidates.unbind(dim=-1)
            scores = score_bits.view(torch.float32)
        else:
            # Keep the existing input domain for callers using other dtypes.
            offsets = all_gather(offsets.contiguous(), group, dim=-1)
            scores = all_gather(scores.contiguous(), group, dim=-1)
    valid = (offsets >= 0) & ~torch.isnan(scores) & (scores > -float("inf"))
    scores = torch.where(valid, scores, -float("inf"))
    selected_scores, order = torch.topk(scores, topk, dim=-1)
    result = offsets.gather(1, order)
    valid = selected_scores > -float("inf")
    result = torch.where(valid, result, -1)
    return result, valid.sum(dim=-1, dtype=torch.int32)


def merge_index_candidates_row_owner(
    records: torch.Tensor,
    rows: int,
    *,
    topk: int,
    group: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge disjoint local candidates like :func:`merge_index_candidates`,
    each query row on one rank of ``group`` instead of on every rank.

    Rank ``d`` owns the ``d``-th equal slice of the rows. One all-to-all sends
    every slice's candidates to its owner, the owner merges its rows over
    every rank's candidates with the kernel the all-gather path uses, and two
    all-gathers (positions, then counts) return every row's selection to
    every rank. A row's merge reads the same rank-major candidates as on the
    all-gather path, so the selection is the same up to the order of equal
    scores, which neither path prescribes. Every rank must call with the same
    ``rows`` and record shape.

    Args:
        records: This rank's packed candidates
            (:func:`pack_index_candidates`), contiguous int32
            ``[padded_rows, candidates, 2]``: ``rows`` query rows, then
            padding rows of invalid candidates (-1, -inf) up to the next
            multiple of the group's size.
        rows: Query rows before the padding.
        topk: Candidates selected per row.
        group: The KVP group's ranks.

    Returns:
        Global logical positions ``[rows, topk]`` (padding -1) and valid
        counts ``[rows]``, int32 views of buffers with ``padded_rows`` rows,
        identical on every rank.
    """
    degree = len(group)
    if (
        records.ndim != 3
        or records.shape[2] != 2
        or records.dtype != torch.int32
        or not records.is_contiguous()
        or topk <= 0
        or records.shape[0] % degree
        or not records.shape[0] - degree < rows <= records.shape[0]
    ):
        raise ValueError(
            "Row-owner candidates must be contiguous int32 [rows, candidates, 2] "
            "records padded to the next multiple of the group size"
        )
    padded = records.shape[0]
    received = torch.empty(
        (degree, padded // degree, *records.shape[1:]),
        dtype=records.dtype,
        device=records.device,
    )
    # Equal splits: rank d gets slice d, laid out source-rank-major as the merge reads.
    all_to_all_single(received.flatten(), records.flatten(), group)
    positions, counts = dsa_merge_candidates(received, topk=topk, solution=None)
    all_positions = positions.new_empty((padded, topk))
    all_counts = counts.new_empty((padded,))
    all_gather_single(all_positions, positions, group)
    all_gather_single(all_counts, counts, group)
    return all_positions[:rows], all_counts[:rows]


@dataclass(frozen=True)
class DCPIndexMetadata:
    """Layer-invariant inputs of :func:`select_dsa_topk` for one forward's rows.

    Every indexer layer of a forward scores the same rows against the same
    page table, so the local table, the row geometry and the compact owned
    pages are built once and read by every layer. ``pages`` is ``None`` when
    the candidate leaf compacts per call, or when per-tile compaction bounds
    memory (prefill). All tensors have fixed shapes, so a captured build
    refreshes them from the live page table and lengths on replay.

    ``columns`` is the owned width the rows are tiled by: ``pages``, when
    planned, are that wide, and otherwise every tile plans its own pages at
    it. ``None`` tiles by the table's width (decode's static width, or a leaf
    that compacts the whole table per call).
    """

    page_table: torch.Tensor
    local_page_table: torch.Tensor
    query_requests: torch.Tensor
    causal_lens: torch.Tensor
    pages: DSAIndexCandidatePages | None
    placement: CachePlacement
    columns: int | None = None


# Above this, plan pages per tile: forward-wide pages stay live across MoE peaks.
_MAX_FORWARD_PAGES_BYTES = 256 << 20
_PAGES_BYTES_PER_COLUMN = 8


def _local_index_table(
    page_table: torch.Tensor, placement: CachePlacement, page_size: int
) -> torch.Tensor:
    slots, owned = resolve_cache_slots(page_table.long() * page_size, placement)
    return torch.where(owned, slots // page_size, -1).to(torch.int32)


def _owned_columns(
    columns: int | None,
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    local_table: torch.Tensor,
    page_size: int,
) -> int | None:
    """The owned width to tile by, or ``None`` for the table's width.

    A row never owns more pages than the table has columns, so a wider bound
    is clamped to the table. A leaf that does not score planned pages sizes
    its logits by the whole table, so it keeps the table's width.
    """
    if columns is None or not dsa_index_candidates_accepts_pages(
        q, weights, index_k_cache, page_size=page_size
    ):
        return None
    if columns <= 0:
        raise ValueError(f"owned index columns must be positive, got {columns}")
    return min(columns, local_table.shape[1])


def build_dcp_index_metadata(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_requests: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    placement: CachePlacement,
    page_size: int,
    columns: int | None = None,
) -> DCPIndexMetadata:
    """Resolve owned pages once for every indexer layer of a forward.

    ``q``, ``weights`` and ``index_k_cache`` select the candidate leaf the
    pages are planned for; any indexer layer's tensors serve every layer.

    ``columns`` (a prefill's) bounds the pages any row owns on any rank
    (``placement.max_owned_index_pages``) and plans the pages that wide, so
    tiles cover proportionally more rows. Pages that would exceed
    ``_MAX_FORWARD_PAGES_BYTES`` are not planned here; :func:`select_dsa_topk`
    plans each tile's at the same width. ``None`` plans the table's width
    (decode's static width).
    """
    local_table = _local_index_table(page_table, placement, page_size)
    query_requests = query_requests.contiguous()
    causal_lens = causal_lens.contiguous()
    columns = _owned_columns(columns, q, weights, index_k_cache, local_table, page_size)
    pages = None
    if (
        columns is None
        or q.shape[0] * columns * _PAGES_BYTES_PER_COLUMN <= _MAX_FORWARD_PAGES_BYTES
    ):
        pages = plan_dsa_index_candidates(
            q,
            weights,
            index_k_cache,
            local_table,
            query_requests,
            causal_lens,
            page_size=page_size,
            columns=columns,
        )
    return DCPIndexMetadata(
        page_table, local_table, query_requests, causal_lens, pages, placement, columns
    )


def select_dsa_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_requests: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    placement: CachePlacement,
    page_size: int,
    topk: int,
    softmax_scale: float,
    initial_tokens: int,
    local_tokens: int,
    max_logits_bytes: int | None,
    metadata: DCPIndexMetadata | None = None,
    row_owner: bool = False,
    columns: int | None = None,
    q_scales: torch.Tensor | None = None,
    query_group_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return request-relative global Top-K from local Index-K pages.

    Metadata has global positions throughout. Only cache addresses are local;
    global Top-K precedes the attention-side ownership mask and LSE reduction.
    Query tiling targets logical FP32 logits for both prefill and decode;
    kernel padding and selection scratch require additional memory.
    A ``max_logits_bytes`` of ``None`` or ``0`` scores all rows in one tile,
    up to the 2**31 logits elements that 32-bit row offsets address, and
    exchanges every row's local candidates once for one global merge. A
    positive budget also bounds an exchange: consecutive whole tiles join one
    exchange while its gathered records and merge scores
    (``degree * topk * 12`` bytes per row) fit, and a tile that alone exceeds
    the budget is exchanged by itself.
    ``metadata`` from :func:`build_dcp_index_metadata` for these rows replaces
    the per-call table resolution; without it, tiles compact their own rows.
    ``row_owner`` merges each row on one rank of a multi-rank group
    (:func:`merge_index_candidates_row_owner`) instead of on every rank, with
    the same selection; its exchange holds ``topk * 24`` bytes per row. It
    needs the int32 positions and FP32 scores CUDA candidate leaves emit.
    ``columns`` tiles a call without ``metadata`` by that owned width, each
    tile planning its pages at it, as ``build_dcp_index_metadata``'s
    ``columns`` does (metadata carries its own). An owned-width tile also
    stays within the rows one exchange of the call's kind (``row_owner`` or
    not) fits, or the table-width tile where that is larger, so no exchange
    outgrows the table-width plan's.
    """
    if q.shape[0] == 0:
        return (
            torch.empty((0, topk), device=q.device, dtype=torch.int32),
            torch.empty(0, device=q.device, dtype=torch.int32),
        )
    if metadata is None:
        local_table = _local_index_table(page_table, placement, page_size)
        metadata = DCPIndexMetadata(
            page_table,
            local_table,
            query_requests,
            causal_lens,
            None,
            placement,
            _owned_columns(columns, q, weights, index_k_cache, local_table, page_size),
        )
    elif columns is not None:
        raise ValueError("DCP index metadata carries its own columns")
    elif (
        metadata.placement != placement
        or metadata.page_table.shape != page_table.shape
        or metadata.query_requests.shape != (q.shape[0],)
    ):
        raise ValueError(
            "DCP index metadata was built for another placement, table or rows"
        )
    rows, table_columns = q.shape[0], metadata.local_page_table.shape[1]
    degree = len(placement.group)
    # One rank merges its rows alone either way.
    row_owner = row_owner and degree > 1
    tile_rows = _tile_rows(rows, table_columns, page_size, max_logits_bytes)
    if metadata.columns is not None:
        tile_rows = _owned_tile_rows(
            rows,
            metadata.columns,
            tile_rows,
            page_size,
            degree,
            topk,
            max_logits_bytes,
            row_owner=row_owner,
        )
    exchange_rows = _exchange_rows(
        rows, tile_rows, degree, topk, max_logits_bytes, row_owner=row_owner
    )

    def tile_pages(start: int, end: int) -> DSAIndexCandidatePages | None:
        if metadata.pages is not None:
            return metadata.pages.rows(start, end)
        if metadata.columns is None:
            return None
        # Plan this tile's pages, or the leaf would size its logits by the whole table.
        pages = plan_dsa_index_candidates(
            q[start:end],
            weights[start:end],
            index_k_cache,
            metadata.local_page_table,
            metadata.query_requests[start:end],
            metadata.causal_lens[start:end],
            page_size=page_size,
            columns=metadata.columns,
        )
        if pages is None:
            raise RuntimeError(
                "owned-width index tiles need a candidate leaf that scores "
                "planned pages"
            )
        return pages

    def score_tile(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return dsa_index_candidates(
            q[start:end],
            weights[start:end],
            index_k_cache,
            metadata.local_page_table,
            metadata.query_requests[start:end],
            metadata.causal_lens[start:end],
            page_size=page_size,
            topk=topk,
            softmax_scale=softmax_scale,
            initial_tokens=initial_tokens,
            local_tokens=local_tokens,
            solution=None,
            pages=tile_pages(start, end),
            **(
                {
                    "q_scales": q_scales[start:end],
                    "query_group_size": (
                        query_group_size
                        if start % query_group_size == 0 and end % query_group_size == 0
                        else 1
                    ),
                }
                if q_scales is not None
                else {}
            ),
        )

    def exchange(first: int, last: int) -> tuple[torch.Tensor, torch.Tensor]:
        offsets, scores = _join_tiles(
            score_tile, first, last, tile_rows, pad_to=degree if row_owner else 1
        )
        if not row_owner:
            return merge_index_candidates(
                offsets, scores, topk=topk, group=placement.group
            )
        if not _exchanges_packed_records(offsets, scores):
            raise ValueError(
                "The row-owner candidate merge exchanges CUDA int32 positions "
                "and FP32 scores"
            )
        return merge_index_candidates_row_owner(
            pack_index_candidates(offsets, scores),
            last - first,
            topk=topk,
            group=placement.group,
        )

    if exchange_rows >= rows:
        return exchange(0, rows)
    indices = lengths = None
    for first in range(0, rows, exchange_rows):
        last = min(rows, first + exchange_rows)
        part, part_lengths = exchange(first, last)
        if indices is None:
            indices = part.new_empty((rows, *part.shape[1:]))
            lengths = part_lengths.new_empty((rows,))
        indices[first:last].copy_(part)
        lengths[first:last].copy_(part_lengths)
    return indices, lengths


def _join_tiles(
    score_tile, first: int, last: int, tile_rows: int, *, pad_to: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local candidates of rows ``[first, last)``, scored ``tile_rows`` at a
    time and joined for one exchange.

    Packed candidates (see :func:`_exchanges_packed_records`) come back with
    their rows rounded up to a multiple of ``pad_to``, the padding rows
    holding invalid candidates (-1, -inf); other candidates are not padded.
    """
    rows = last - first
    padded = -(-rows // pad_to) * pad_to
    if rows <= tile_rows:
        offsets, scores = score_tile(first, last)
        if padded == rows or not _exchanges_packed_records(offsets, scores):
            return offsets, scores
        # One tile whose rows do not split evenly: copy it into padded records.
        records = _joined_records(rows, padded, offsets)
        records[:rows].copy_(pack_index_candidates(offsets, scores))
        return records[..., 0], records[..., 1].view(torch.float32)
    records, parts = None, []
    for start in range(first, last, tile_rows):
        end = min(last, start + tile_rows)
        offsets, scores = score_tile(start, end)
        if records is None and not parts and _exchanges_packed_records(offsets, scores):
            records = _joined_records(rows, padded, offsets)
        if records is not None:
            # Copy each tile in and free it before scoring the next; packing is free.
            records[start - first : end - first].copy_(
                pack_index_candidates(offsets, scores)
            )
        else:
            parts.append((offsets, scores))
    if records is not None:
        return records[..., 0], records[..., 1].view(torch.float32)
    return (
        torch.cat([part[0] for part in parts]),
        torch.cat([part[1] for part in parts]),
    )


def _joined_records(rows: int, padded: int, like: torch.Tensor) -> torch.Tensor:
    """Int32 ``[padded, candidates, 2]`` records for ``like``'s candidates;
    rows past ``rows`` hold invalid candidates (-1, -inf), the rest is unset."""
    records = torch.empty(
        (padded, like.shape[1], 2), dtype=torch.int32, device=like.device
    )
    if padded > rows:
        records[rows:, :, 0].fill_(-1)
        records[rows:, :, 1].fill_(_NEG_INF_BITS)
    return records
