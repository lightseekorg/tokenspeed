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

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import chain

import numpy as np

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.pd.cache_protocol import (
    CachePDBlockManifest,
    CachePDLayerwiseBlockSelection,
    CacheTransferContract,
)
from tokenspeed.runtime.pd.mooncake.entities import (
    KVArgs,
    KVArgsRegisterInfo,
)
from tokenspeed.runtime.pd.mooncake.pack import (
    PackedCopy,
    PageFieldCopies,
)
from tokenspeed.runtime.pd.transfer_plan import (
    CachePageOwnerFilter,
    CacheTransferFragment,
    CacheTransferPlanner,
    local_transfer_pages,
    validate_rank_owner_filters,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)

logger = get_colorful_logger(__name__)


def _strided_page_copies(
    fragment: CacheTransferFragment,
    *,
    src_base: int,
    dst_base: int,
    src_page_stride: int,
    dst_page_stride: int,
    src_pages: np.ndarray,
    dst_pages: np.ndarray,
) -> Iterator[PackedCopy | tuple[int, int, int]]:
    """Per-page copies of a fragment with several strided rows per page.

    A contiguous destination packs the rows into one ``PackedCopy`` (or one
    plain span when the source is contiguous too); otherwise every row is its
    own descriptor. ``src_base``/``dst_base`` already include the fragment's
    byte offsets within the field.
    """
    width = fragment.bytes_per_row
    rows = fragment.rows_per_page
    src_pitch = fragment.src_row_stride_bytes
    dst_pitch = fragment.dst_row_stride_bytes
    for src_page, dst_page in zip(src_pages.tolist(), dst_pages.tolist(), strict=True):
        src_page_addr = src_base + src_page * src_page_stride
        dst_page_addr = dst_base + dst_page * dst_page_stride
        if dst_pitch == width:
            if src_pitch == width:
                yield (src_page_addr, dst_page_addr, width * rows)
                continue
            yield PackedCopy(
                src=src_page_addr,
                dst=dst_page_addr,
                width=width,
                src_pitch=src_pitch,
                rows=rows,
            )
            continue
        for row in range(rows):
            yield (
                src_page_addr + row * src_pitch,
                dst_page_addr + row * dst_pitch,
                width,
            )


def cache_transfer_blocks(
    kv_args: KVArgs,
    *,
    dst_ptr: int,
    dst_tp_rank: int,
    src_block_manifest: CachePDBlockManifest | None,
    dst_block_manifest: CachePDBlockManifest,
    transfer_fragments: tuple[CacheTransferFragment, ...] = (),
    owner_filters: Mapping[str, CachePageOwnerFilter | None],
    dst_cache_layout: CacheTransferContract,
    block_selection: CachePDLayerwiseBlockSelection | None = None,
    step_field_ids: Sequence[frozenset[str] | None],
) -> Iterator[Iterable[PageFieldCopies | PackedCopy | tuple[int, int, int]]]:
    """Yield, per entry of ``step_field_ids``, the copies that move this
    rank's part of one request's blocks for those fields (None: every
    field); an empty list when none of them is on this route.

    The blocks are translated to pages and every field's copy geometry is
    resolved once, at the first ``next``; each entry then only picks its
    fields' rows. A layerwise sender thus pays the translation once per
    Decode destination, not once per producer interval: this runs on the
    transfer thread while holding the GIL the forward thread needs.

    Each cache group gives one pages x fields grid (``PageFieldCopies``)
    holding every copy with one contiguous span per page -- a whole field
    on the empty-fragment route, or a fragment whose rows collapsed into
    one -- so a sharded or pipeline-staged source still goes through the
    page-gathered WRITE. Only a fragment with several strided rows per
    page is emitted per page, lazily, so its row copies stream into the
    bounded descriptor batches of ``_transfer_data``.

    Args:
        kv_args: This rank's cache layout and arena base.
        owner_filters: The registration route's decision for every
            KVP-sharded group of the layout (``validate_rank_owner_filters``
            checked it): the filter selecting the manifest blocks this
            rank owns, copied from their local pages to the destination
            blocks at the same manifest positions, or None when the route
            carries no block of the group and nothing is sent for it.
            Replicated groups are copied whole; their scheduler IDs are
            already local pages.
    """
    layout = kv_args.cache_layout

    cache_fragments = tuple(transfer_fragments)
    local_segments = {
        (segment.group_id, segment.field_id): segment for segment in layout.plan.fields
    }
    peer_segments = {
        (segment.group_id, segment.field_id): segment
        for segment in dst_cache_layout.plan.fields
    }
    fragments_by_group: dict[str, list[CacheTransferFragment]] = defaultdict(list)
    for fragment in cache_fragments:
        fragments_by_group[fragment.group_id].append(fragment)

    # Resolve block ids directly from the validated manifest/selection. This
    # leaves one authoritative representation of the request's block map.
    group_transfers: list[tuple[CacheGroupSpec, np.ndarray, np.ndarray]] = []
    source_groups = (
        block_selection.groups
        if block_selection is not None
        else src_block_manifest.groups
    )
    for group_spec, src_group, dst_group in zip(
        layout.group_specs, source_groups, dst_block_manifest.groups, strict=True
    ):
        source_block_ids = (
            src_group.source_block_ids
            if block_selection is not None
            else src_group.block_ids
        )
        destination_block_ids = (
            tuple(
                dst_group.block_ids[position]
                for position in src_group.destination_positions
            )
            if block_selection is not None
            else dst_group.block_ids
        )
        if group_spec.shard_count > 1:
            owner_filter = owner_filters[group_spec.group_id]
            if owner_filter is None:
                continue
            source_owner_rank = owner_filter.owner_rank
        else:
            source_owner_rank = 0
        # KVP subgroups are aligned runs of consecutive attention-TP
        # ranks, so the destination's owner coordinate is its TP rank
        # modulo its group's shard count (1 for a replicated group).
        src_pages, dst_pages = local_transfer_pages(
            source_block_ids,
            destination_block_ids,
            group_id=group_spec.group_id,
            source_layout=layout,
            destination_layout=dst_cache_layout,
            source_owner_rank=source_owner_rank,
            destination_owner_rank=dst_tp_rank
            % dst_cache_layout.shard_count(group_spec.group_id),
        )
        if src_pages.size:
            group_transfers.append(
                (
                    group_spec,
                    src_pages,
                    dst_pages,
                )
            )

    # Each field's copies on this route: (group transfer index, (src base,
    # src page stride, dst base, dst page stride, length), the fragment if
    # it has several strided rows per page).
    src_ptr = kv_args.kv_data_ptr
    field_rows: dict[
        str,
        list[
            tuple[
                int,
                tuple[int, int, int, int, int],
                CacheTransferFragment | None,
            ]
        ],
    ] = defaultdict(list)
    for index, (group_spec, _, _) in enumerate(group_transfers):
        if cache_fragments:
            for fragment in fragments_by_group.get(group_spec.group_id, ()):
                key = (fragment.group_id, fragment.field_id)
                field_rows[fragment.field_id].append(
                    (
                        index,
                        (
                            layout.field_address(src_ptr, fragment.field_id)
                            + fragment.src_byte_offset,
                            local_segments[key].page_stride_bytes,
                            dst_cache_layout.field_address(dst_ptr, fragment.field_id)
                            + fragment.dst_byte_offset,
                            peer_segments[key].page_stride_bytes,
                            fragment.bytes_per_row,
                        ),
                        fragment if fragment.rows_per_page > 1 else None,
                    )
                )
        else:
            for src_segment in layout.fields_for_group(group_spec.group_id):
                dst_segment = peer_segments[(group_spec.group_id, src_segment.field_id)]
                field_rows[src_segment.field_id].append(
                    (
                        index,
                        (
                            layout.field_address(src_ptr, src_segment.field_id),
                            src_segment.page_stride_bytes,
                            dst_cache_layout.field_address(
                                dst_ptr, dst_segment.field_id
                            ),
                            dst_segment.page_stride_bytes,
                            src_segment.payload_bytes,
                        ),
                        None,
                    )
                )

    for field_ids in step_field_ids:
        span_rows: dict[int, list[tuple[int, int, int, int, int]]] = {}
        strided_rows = []
        for field_id in (
            field_rows if field_ids is None else sorted(field_ids & field_rows.keys())
        ):
            for index, row, strided_fragment in field_rows[field_id]:
                if strided_fragment is None:
                    span_rows.setdefault(index, []).append(row)
                else:
                    strided_rows.append((index, row, strided_fragment))
        copies = [
            PageFieldCopies(
                group_transfers[index][1],
                group_transfers[index][2],
                np.asarray(rows, dtype=np.int64).reshape(-1, 5),
            )
            for index, rows in sorted(span_rows.items())
        ]
        if not strided_rows:
            yield copies
            continue
        # Never empty, which the layerwise sender's skip relies on: every
        # transfer group has pages, so each row gives at least one copy.
        yield chain(
            copies,
            (
                copy
                for index, row, strided_fragment in strided_rows
                for copy in _strided_page_copies(
                    strided_fragment,
                    src_base=row[0],
                    dst_base=row[2],
                    src_page_stride=row[1],
                    dst_page_stride=row[3],
                    src_pages=group_transfers[index][1],
                    dst_pages=group_transfers[index][2],
                )
            ),
        )


def prepare_decode_registration(
    kv_args: KVArgs, topology, registration: KVArgsRegisterInfo
) -> KVArgsRegisterInfo:
    """Plan the route a Decode rank's registration asks of this Prefill rank.

    Args:
        kv_args: This rank's cache layout and stage placement.
        topology: This rank's ``PDParallelTopology``.
        registration: The Decode rank's parsed registration.

    Returns:
        ``registration`` with its fragments, owner filters and expected
        Decode ranks.

    Raises:
        ValueError: The registration targets the wrong Prefill rank, or its
            layout does not fit this one.
    """
    # Plan against the LOGICAL layout (full model): peer validation and
    # fragment geometry must match what Decode sees on the wire. The
    # field filter below keeps only fields this stage's physical arena
    # actually holds, so local addressing never touches a dropped plane.
    layout = getattr(kv_args, "wire_layout", None) or kv_args.cache_layout
    peer_layout = registration.peer_cache_layout
    prefill_tp_size = topology.tp_size
    local_tp_rank = topology.tp_rank
    planner = CacheTransferPlanner(
        prefill_tp_size=prefill_tp_size,
        decode_tp_size=registration.decode_tp_size,
        prefill_layout=layout,
        decode_layout=peer_layout,
        # Construction declares residency; transfer never infers model layers.
        prefill_field_ids=frozenset(kv_args.cache_fields_by_stage[topology.pp_rank]),
    )
    route = planner.plan_for_decode_rank(registration.decode_tp_rank)
    expected_decode_ranks = planner.decode_ranks_by_prefill_rank[local_tp_rank]
    if local_tp_rank in route.target_prefill_ranks:
        # A KVP-sharded source is never idle: every rank of the chosen
        # subgroup serves the decode rank with the blocks it owns.
        fragments = route.fragments_by_prefill_rank[local_tp_rank]
        owner_filters = route.owner_filters_by_prefill_rank[local_tp_rank]
        is_dummy = False
    elif registration.decode_tp_rank == 0 and not expected_decode_ranks:
        fragments = ()
        # A rendezvous-only rank sends nothing of any group.
        owner_filters = dict.fromkeys(planner.sharded_group_ids)
        is_dummy = True
        expected_decode_ranks = frozenset({0})
    else:
        raise ValueError(
            "CachePD Decode registration targets the wrong Prefill TP rank"
        )
    # The sender applies the route's owner-filter decisions as given, so
    # they are checked against the local cache groups once, here.
    validate_rank_owner_filters(
        group_specs=kv_args.cache_layout.group_specs,
        fragments=fragments,
        owner_filters=owner_filters,
    )
    return replace(
        registration,
        transfer_fragments=fragments,
        transfer_owner_filters=owner_filters,
        is_dummy=is_dummy,
        expected_decode_ranks=expected_decode_ranks,
    )
