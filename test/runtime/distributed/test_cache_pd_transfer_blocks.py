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

"""CPU-only tests for the PD transfer planner (``pd/mooncake/prefill.py``,
``pd/transfer_plan.py``): the copies one request's blocks take between a
Prefill and a Decode rank's caches, which the Rust transfer peer writes."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from test.runtime.cache_pd_test_utils import block_manifest as make_block_manifest
from test.runtime.cache_pd_test_utils import group as make_group  # noqa: E402
from test.runtime.cache_pd_test_utils import layout as make_layout
from test.runtime.cache_pd_test_utils import operation as make_operation
from test.runtime.cache_pd_test_utils import segment as make_segment

from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    CacheContractError,
    CachePDBlockManifest,
    CacheTransferContract,
    build_cache_block_manifest,
    build_cache_layerwise_block_selection,
    validate_cache_manifest,
)
from tokenspeed.runtime.pd.mooncake.entities import (  # noqa: E402
    KVArgs,
    KVArgsRegisterInfo,
)
from tokenspeed.runtime.pd.mooncake.pack import (  # noqa: E402
    PageFieldCopies,
    coalesce_transfer_blocks,
    flatten_transfer_blocks,
)
from tokenspeed.runtime.pd.mooncake.prefill import cache_transfer_blocks  # noqa: E402
from tokenspeed.runtime.pd.topology import PDParallelTopology  # noqa: E402
from tokenspeed.runtime.pd.transfer_plan import (  # noqa: E402
    CacheTransferFragment,
    CacheTransferPlanner,
    UnsupportedPDLayoutError,
)


def _kv_args(layout: CacheTransferContract, kv_data_ptr: int) -> KVArgs:
    return KVArgs(
        engine_rank=0,
        kv_data_ptr=kv_data_ptr,
        ib_device="",
        gpu_id=0,
        cache_layout=layout,
        cache_fields_by_stage=(tuple(field.field_id for field in layout.plan.fields),),
    )


def _transfer_spans(
    layout: CacheTransferContract,
    src_ptr: int,
    dst_ptr: int,
    transfer_fragments: tuple[CacheTransferFragment, ...],
    *,
    src_block_manifest: CachePDBlockManifest,
    dst_block_manifest: CachePDBlockManifest,
    dst_cache_layout: CacheTransferContract,
) -> list[tuple[int, int, int]]:
    """Every (source, destination, length) span of one whole-request
    transfer, in the order the planner gives them."""
    (copies,) = cache_transfer_blocks(
        _kv_args(layout, src_ptr),
        dst_tp_rank=0,
        dst_ptr=dst_ptr,
        src_block_manifest=src_block_manifest,
        dst_block_manifest=dst_block_manifest,
        transfer_fragments=transfer_fragments,
        owner_filters={},
        dst_cache_layout=dst_cache_layout,
        step_field_ids=(None,),
    )
    spans = []
    for item in copies:
        if isinstance(item, PageFieldCopies):
            spans.extend(
                _expand_page_fields(item.src_pages, item.dst_pages, item.fields)
            )
        else:
            spans.append(tuple(item))
    return spans


def _topology(
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    dp_size: int = 1,
    dp_rank: int = 0,
    world_size: int | None = None,
    global_rank: int = 0,
) -> PDParallelTopology:
    return PDParallelTopology(
        tp_size=tp_size,
        tp_rank=tp_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        world_size=world_size or tp_size * dp_size,
        global_rank=global_rank,
    )


def _layout(
    *,
    capacity: int = 16,
    physical_page_bytes: int = 32,
    page_stride_bytes: int = 32,
    history_offset: int = 0,
    state_offset: int = 16,
) -> CacheTransferContract:
    return make_layout(
        make_group(
            "history",
            make_segment(
                "layer.0.kv",
                dtype="bfloat16",
                shape=(8,),
                offset=history_offset,
                stride=page_stride_bytes,
            ),
        ),
        make_group(
            "state",
            make_segment(
                "layer.1.state",
                dtype="bfloat16",
                shape=(8,),
                offset=state_offset,
                stride=page_stride_bytes,
            ),
            attn_kind="state",
        ),
        capacity=capacity,
        page_bytes=physical_page_bytes,
    )


def _typed_layout(
    *,
    local_heads: int,
    global_heads: int,
) -> CacheTransferContract:
    payload_bytes = 2 * local_heads * 2 * 2
    return make_layout(
        make_group(
            "history",
            make_segment(
                "layer.0.k",
                dtype="bfloat16",
                shape=(2, local_heads, 2),
                axis=1,
                extent=global_heads,
            ),
        ),
        capacity=5,
        page_bytes=payload_bytes,
    )


def _single_group_block_manifest(
    group_id: str, block_ids: tuple[int, ...]
) -> CachePDBlockManifest:
    return make_block_manifest((group_id, block_ids))


def _registration_frames(
    layout: CacheTransferContract,
    *,
    pointer: int = 0x1000,
    decode_tp_size: int = 1,
    decode_tp_rank: int = 0,
) -> list[bytes]:
    return [
        b"None",
        b"127.0.0.1",
        b"9000",
        b"session",
        np.asarray([pointer], dtype=np.uint64).tobytes(),
        layout.to_wire_bytes(),
        str(decode_tp_size).encode("ascii"),
        str(decode_tp_rank).encode("ascii"),
    ]


def _registration(
    layout: CacheTransferContract,
    *,
    rank: int = 0,
    decode_tp_size: int = 1,
    session: str | None = None,
    endpoint: str | None = None,
    pointer: int = 0x2000,
    expected_decode_ranks=(),
) -> KVArgsRegisterInfo:
    return KVArgsRegisterInfo(
        endpoint=endpoint or f"decode-{rank}",
        dst_port=9000 + rank,
        mooncake_session_id=session or f"session-{rank}",
        dst_kv_ptr=pointer,
        peer_cache_layout=layout,
        decode_tp_size=decode_tp_size,
        decode_tp_rank=rank,
        expected_decode_ranks=frozenset(expected_decode_ranks),
    )


def _expand_page_fields(src_pages, dst_pages, fields) -> list[tuple[int, int, int]]:
    """Reference expansion of a PageFieldCopies grid, field-major."""
    return [
        (
            int(src_base + page * src_stride),
            int(dst_base + peer * dst_stride),
            int(length),
        )
        for src_base, src_stride, dst_base, dst_stride, length in fields.tolist()
        for page, peer in zip(src_pages.tolist(), dst_pages.tolist(), strict=True)
    ]


def test_cache_factory_exposes_only_typed_arena() -> None:
    import torch

    from tokenspeed.runtime.pd.factory import get_kv_args

    layout = _layout()
    buffer = torch.zeros(layout.plan.arena_bytes, dtype=torch.uint8)
    pool = SimpleNamespace(
        arena=SimpleNamespace(
            supports_disaggregation=True,
            plan=layout.plan,
            cache_group_specs=layout.group_specs,
            contract_binding=lambda: buffer,
            field_address=lambda name: (
                buffer.data_ptr() + layout.plan.field_page_byte_offset(name, 0)
            ),
            registration_regions=lambda: ((buffer.data_ptr(), buffer.nbytes),),
        ),
    )

    kv_args = get_kv_args(
        0,
        0,
        "mlx5_0",
        pool,
        draft_model_config=None,
        cache_fields_by_stage=(tuple(field.field_id for field in layout.plan.fields),),
        producer_fields_by_step=tuple(
            (field.field_id,) for field in layout.plan.fields
        ),
        logical_plan=None,
        model_config=SimpleNamespace(
            num_attention_layers=2,
            num_key_value_heads=1,
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(),
        ),
    )

    assert kv_args.engine_rank == 0
    assert kv_args.kv_data_ptr == buffer.data_ptr()
    assert kv_args.ib_device == "mlx5_0"
    assert kv_args.gpu_id == 0
    assert kv_args.cache_layout.plan == layout.plan
    assert kv_args.cache_layout.group_specs == layout.group_specs
    assert kv_args.cache_layout.transfer_schema == layout.transfer_schema
    for field in layout.plan.fields:
        assert kv_args.cache_layout.field_address(
            buffer.data_ptr(), field.field_id
        ) == buffer.data_ptr() + layout.plan.field_page_byte_offset(field.field_id, 0)
    assert kv_args.cache_producer_schedule.fields_by_step == (
        ("layer.0.kv",),
        ("layer.1.state",),
    )


def test_a_strided_fragment_copies_each_row() -> None:
    source_segment = make_segment(
        "layer.0.k",
        dtype="bfloat16",
        shape=(2, 2, 2),
        stride=32,
        axis=1,
        extent=4,
    )
    destination_segment = make_segment(
        "layer.0.k",
        dtype="bfloat16",
        shape=(2, 4, 2),
        stride=64,
        axis=1,
        extent=4,
    )

    def one_field_layout(field):
        return make_layout(make_group("history", field), capacity=5, page_bytes=64)

    source_layout = one_field_layout(source_segment)
    destination_layout = one_field_layout(destination_segment)
    source_manifest = _single_group_block_manifest("history", (1,))
    destination_manifest = _single_group_block_manifest("history", (2,))
    fragment = CacheTransferFragment(
        group_id="history",
        field_id="layer.0.k",
        src_byte_offset=0,
        dst_byte_offset=8,
        src_row_stride_bytes=8,
        dst_row_stride_bytes=16,
        bytes_per_row=8,
        rows_per_page=2,
    )
    assert _transfer_spans(
        source_layout,
        0x1000,
        0x2000,
        (fragment,),
        src_block_manifest=source_manifest,
        dst_block_manifest=destination_manifest,
        dst_cache_layout=destination_layout,
    ) == [(0x1020, 0x2088, 8), (0x1028, 0x2098, 8)]


@pytest.mark.parametrize("physical_addresses", [False, True])
def test_transfer_blocks_for_many_pages_match_the_per_page_geometry(
    physical_addresses,
) -> None:
    # Two fields and hundreds of pages: the generator resolves each field's
    # geometry once and expands pages in bulk, so check it against the plain
    # per-page formula in both the whole-field and the fragment path.
    def two_field_layout(capacity: int):
        return make_layout(
            make_group(
                "history",
                make_segment("layer.0.k", dtype="bfloat16", shape=(2, 4, 2)),
                make_segment("layer.1.k", dtype="bfloat16", shape=(2, 4, 2)),
            ),
            capacity=capacity,
            page_bytes=64,
        )

    source_layout = two_field_layout(700)
    destination_layout = two_field_layout(900)
    if physical_addresses:
        # Mixed arenas advertise each field's landing address independently
        # of the logical plane offsets and the endpoint's registration base.
        source_layout = replace(
            source_layout,
            field_addresses={
                field.field_id: 0x100000 * (i + 1)
                for i, field in enumerate(source_layout.plan.fields)
            },
        )
        destination_layout = replace(
            destination_layout,
            field_addresses={
                field.field_id: 0x1000000 * (i + 1)
                for i, field in enumerate(destination_layout.plan.fields)
            },
        )
    pages = 300
    src_pages = tuple(range(1, 2 * pages, 2))
    dst_pages = tuple(range(899, 899 - 2 * pages, -2))
    source_manifest = _single_group_block_manifest("history", src_pages)
    destination_manifest = _single_group_block_manifest("history", dst_pages)

    def field_pages(layout, ptr, field_id, page_ids):
        segment = next(f for f in layout.plan.fields if f.field_id == field_id)
        return [
            layout.field_address(ptr, field_id) + page * segment.page_stride_bytes
            for page in page_ids
        ], segment.payload_bytes

    kv_args = _kv_args(source_layout, 0x1000)
    fields = tuple(f.field_id for f in source_layout.fields_for_group("history"))
    expected = []
    for field_id in fields:
        src, size = field_pages(source_layout, 0x1000, field_id, src_pages)
        dst, _ = field_pages(destination_layout, 0x2000, field_id, dst_pages)
        expected.extend(zip(src, dst, [size] * pages, strict=True))
    blocks = next(
        cache_transfer_blocks(
            kv_args,
            dst_tp_rank=0,
            dst_ptr=0x2000,
            src_block_manifest=source_manifest,
            dst_block_manifest=destination_manifest,
            owner_filters={},
            dst_cache_layout=destination_layout,
            step_field_ids=(None,),
        )
    )
    # One pages x fields item for the group, no per-page Python objects.
    (item,) = blocks
    assert isinstance(item, PageFieldCopies)
    assert item.fields.shape == (2, 5) and len(item) == 2 * pages
    assert _expand_page_fields(item.src_pages, item.dst_pages, item.fields) == expected
    with pytest.raises(TypeError):
        list(flatten_transfer_blocks(blocks))

    # One request, two steps: each step moves only its own field.
    steps = cache_transfer_blocks(
        kv_args,
        dst_tp_rank=0,
        dst_ptr=0x2000,
        src_block_manifest=source_manifest,
        dst_block_manifest=destination_manifest,
        owner_filters={},
        dst_cache_layout=destination_layout,
        step_field_ids=(frozenset(fields[:1]), frozenset(fields[1:])),
    )
    assert [
        _expand_page_fields(item.src_pages, item.dst_pages, item.fields)
        for (item,) in steps
    ] == [expected[:pages], expected[pages:]]

    fragment = CacheTransferFragment(
        group_id="history",
        field_id=fields[1],
        src_byte_offset=4,
        dst_byte_offset=8,
        src_row_stride_bytes=16,
        dst_row_stride_bytes=16,
        bytes_per_row=8,
        rows_per_page=2,
    )
    src, _ = field_pages(source_layout, 0x1000, fields[1], src_pages)
    dst, _ = field_pages(destination_layout, 0x2000, fields[1], dst_pages)
    fragment_blocks = list(
        next(
            cache_transfer_blocks(
                kv_args,
                dst_tp_rank=0,
                dst_ptr=0x2000,
                src_block_manifest=source_manifest,
                dst_block_manifest=destination_manifest,
                transfer_fragments=(fragment,),
                owner_filters={},
                dst_cache_layout=destination_layout,
                step_field_ids=(None,),
            )
        )
    )
    assert fragment_blocks == [
        (s + 4 + row * 16, d + 8 + row * 16, 8)
        for s, d in zip(src, dst, strict=True)
        for row in range(2)
    ]


def test_page_field_copies_validates_its_grid() -> None:
    item = PageFieldCopies(
        np.asarray([1, 5], dtype=np.int64),
        np.asarray([9, 8], dtype=np.int64),
        np.asarray(
            [[1000, 64, 5000, 128, 16], [2000, 4096, 3000, 4096, 3000]], dtype=np.int64
        ),
    )
    assert len(item) == 4
    with pytest.raises(TypeError):
        PageFieldCopies(item.src_pages, item.dst_pages, item.fields[:, :4].copy())
    with pytest.raises(ValueError):
        PageFieldCopies(item.src_pages, item.dst_pages[:1], item.fields)
    # Neither the packer nor the SGE flattener may see one.
    with pytest.raises(TypeError):
        list(flatten_transfer_blocks([item]))


def test_flatten_packed_copy_emits_contiguous_dest_rows() -> None:
    from tokenspeed.runtime.pd.mooncake.pack import PackedCopy, flatten_transfer_blocks

    copy = PackedCopy(src=0x10, dst=0x20, width=8, src_pitch=16, rows=2)
    flattened = flatten_transfer_blocks([copy, (1, 2, 3)])
    assert not isinstance(flattened, list)
    assert list(flattened) == [
        (0x10, 0x20, 8),
        (0x20, 0x28, 8),
        (1, 2, 3),
    ]


def test_copies_land_at_the_destination_page_zero_offsets() -> None:
    source_layout = _layout(capacity=8)
    destination_layout = _layout(
        capacity=8,
        physical_page_bytes=64,
        page_stride_bytes=64,
        history_offset=8,
        state_offset=40,
    )
    source_manifest = make_block_manifest(
        ("history", (1, 2)), ("state", (4,)), prompt=4
    )
    destination_manifest = make_block_manifest(
        ("history", (5, 6)), ("state", (3,)), prompt=4
    )
    assert _transfer_spans(
        source_layout,
        0x10000,
        0x20000,
        (),
        src_block_manifest=source_manifest,
        dst_block_manifest=destination_manifest,
        dst_cache_layout=destination_layout,
    ) == [
        (0x10000 + 1 * 32, 0x20000 + 8 + 5 * 64, 16),
        (0x10000 + 2 * 32, 0x20000 + 8 + 6 * 64, 16),
        (0x10000 + 16 + 4 * 32, 0x20000 + 40 + 3 * 64, 16),
    ]


@pytest.mark.parametrize(
    "blocks, expected",
    [
        ([], []),
        ([(10, 100, 4), (14, 104, 8), (22, 112, 2)], [(10, 100, 14)]),
        ([(10, 100, 4), (15, 104, 4)], [(10, 100, 4), (15, 104, 4)]),
        ([(10, 100, 4), (14, 105, 4)], [(10, 100, 4), (14, 105, 4)]),
        ([(14, 104, 4), (10, 100, 4)], [(14, 104, 4), (10, 100, 4)]),
        ([(10, 100, 4), (12, 102, 4)], [(10, 100, 4), (12, 102, 4)]),
    ],
)
def test_coalesce_transfer_blocks_preserves_byte_mapping(blocks, expected):
    from tokenspeed.runtime.pd.mooncake.pack import coalesce_transfer_blocks

    result = list(coalesce_transfer_blocks(iter(blocks)))
    assert result == expected

    def byte_mapping(entries):
        return [
            (src + i, dst + i) for src, dst, length in entries for i in range(length)
        ]

    assert byte_mapping(result) == byte_mapping(blocks)


def test_coalesce_transfer_blocks_after_pack_expansion():
    from tokenspeed.runtime.pd.mooncake.pack import (
        PackedCopy,
        coalesce_transfer_blocks,
        flatten_transfer_blocks,
    )

    blocks = [PackedCopy(src=100, dst=200, width=4, src_pitch=8, rows=2), (112, 208, 4)]
    assert list(coalesce_transfer_blocks(flatten_transfer_blocks(blocks))) == [
        (100, 200, 4),
        (108, 204, 8),
    ]


def _kvp_layout(degree):
    contract = make_layout(
        make_group("kv", make_segment("layer.0.kv", shape=(2, 8), stride=32)),
        make_group(
            "index",
            make_segment("layer.0.index", shape=(2, 12), stride=32, offset=1024),
        ),
        make_group(
            "swa", make_segment("layer.1.swa", shape=(2, 4), stride=32, offset=2048)
        ),
        page_bytes=256,
    )
    return replace(
        contract,
        group_specs=tuple(
            replace(spec, shard_count=degree if spec.group_id != "swa" else 1)
            for spec in contract.group_specs
        ),
    )


def _kvp_planner(source, destination, source_tp, destination_tp):
    return CacheTransferPlanner(
        prefill_tp_size=source_tp,
        decode_tp_size=destination_tp,
        prefill_layout=source,
        decode_layout=destination,
        prefill_field_ids=None,
    )


@pytest.mark.parametrize(
    "source_tp,destination_tp,source_degree,destination_degree",
    [
        (p, d, ps, ds)
        for p, d in ((4, 4), (4, 2), (2, 4), (8, 4))
        for ps, ds in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (4, 4))
        if p % ps == 0 and d % ds == 0
    ],
)
@pytest.mark.parametrize("layerwise", [False, True])
def test_kvp_pd_copies_logical_history_between_independently_allocated_pages(
    source_tp, destination_tp, source_degree, destination_degree, layerwise
):
    source, destination = _kvp_layout(source_degree), _kvp_layout(destination_degree)
    planner = _kvp_planner(source, destination, source_tp, destination_tp)
    # Include IDs beyond the local physical-page bound, reversed ownership,
    # independent Index-K allocation, a prefix hit, and a partial final page.
    source_tables, destination_tables = {}, {}
    for index, spec in enumerate(source.group_specs):
        degree = spec.shard_count
        source_tables[spec.group_id] = np.array(
            [[1, 2, 8 * degree + 1 + index, 3, 7]], dtype=np.int32
        )
        destination_tables[spec.group_id] = np.array(
            [[1, 4, 6, 8 * destination.shard_count(spec.group_id) + 2 + index, 2]],
            dtype=np.int32,
        )
    source_op, destination_op = make_operation(source_tables), make_operation(
        destination_tables
    )
    src_manifest = build_cache_block_manifest(
        source_op,
        layout=source,
        request_row=0,
        prefix_len=2,
        prompt_len=9,
        state_prefix_cached=False,
    )
    dst_manifest = build_cache_block_manifest(
        destination_op,
        layout=destination,
        request_row=0,
        prefix_len=2,
        prompt_len=9,
        state_prefix_cached=False,
    )
    validate_cache_manifest(src_manifest, layout=source, peer="source")
    validate_cache_manifest(dst_manifest, layout=destination, peer="destination")
    src_buffers = [
        np.full(source.plan.arena_bytes, 173, np.uint8) for _ in range(source_tp)
    ]
    dst_buffers = [
        np.full(destination.plan.arena_bytes, 211, np.uint8)
        for _ in range(destination_tp)
    ]
    expected = [buffer.copy() for buffer in dst_buffers]
    for contract, manifest, buffers in (
        (source, src_manifest, src_buffers),
        (destination, dst_manifest, expected),
    ):
        for group_index, blocks in enumerate(manifest.groups):
            degree = contract.shard_count(blocks.group_id)
            for position, virtual in enumerate(blocks.block_ids):
                page, owner = divmod(virtual - 1, degree)
                for rank, buffer in enumerate(buffers):
                    if rank % degree != owner:
                        continue
                    for field in contract.fields_for_group(blocks.group_id):
                        offset = contract.plan.field_page_byte_offset(
                            field.field_id, page + 1
                        )
                        buffer[offset : offset + field.payload_bytes] = (
                            np.arange(field.payload_bytes, dtype=np.uint8)
                            + position
                            + group_index * 32
                        )
    chunks = [(2, 5), (5, 9)] if layerwise else [None]
    for destination_rank, buffer in enumerate(dst_buffers):
        writes = np.zeros(buffer.shape, dtype=np.uint8)
        route = planner.plan_for_decode_rank(destination_rank)
        for source_rank, fragments in route.fragments_by_prefill_rank.items():
            assert destination_rank in planner.decode_ranks_by_prefill_rank[source_rank]
            kv_args = _kv_args(source, src_buffers[source_rank].ctypes.data)
            for chunk in chunks:
                selection = (
                    None
                    if chunk is None
                    else build_cache_layerwise_block_selection(
                        source_op,
                        layout=source,
                        request_row=0,
                        prefix_len=2,
                        prompt_len=9,
                        chunk_start=chunk[0],
                        chunk_end=chunk[1],
                        state_prefix_cached=False,
                    )
                )
                (copies,) = cache_transfer_blocks(
                    kv_args,
                    dst_ptr=buffer.ctypes.data,
                    dst_tp_rank=destination_rank,
                    src_block_manifest=src_manifest if selection is None else None,
                    dst_block_manifest=dst_manifest,
                    transfer_fragments=fragments,
                    owner_filters=route.owner_filters_by_prefill_rank[source_rank],
                    dst_cache_layout=destination,
                    block_selection=selection,
                    step_field_ids=(None,),
                )
                copies = coalesce_transfer_blocks(
                    block
                    for item in copies
                    for block in (
                        _expand_page_fields(item.src_pages, item.dst_pages, item.fields)
                        if isinstance(item, PageFieldCopies)
                        else [item]
                    )
                )
                for source_address, destination_address, nbytes in copies:
                    assert (
                        src_buffers[source_rank].ctypes.data
                        <= source_address
                        < src_buffers[source_rank].ctypes.data
                        + src_buffers[source_rank].nbytes
                    )
                    assert (
                        buffer.ctypes.data
                        <= destination_address
                        < buffer.ctypes.data + buffer.nbytes
                    )
                    offset = destination_address - buffer.ctypes.data
                    writes[offset : offset + nbytes] += 1
                    ctypes.memmove(destination_address, source_address, nbytes)
        np.testing.assert_array_equal(buffer, expected[destination_rank])
        np.testing.assert_array_equal(writes, expected[destination_rank] != 211)


def test_kvp_pd_equal_tp_contacts_all_source_owners():
    source, destination = _kvp_layout(4), _kvp_layout(4)
    planner = _kvp_planner(source, destination, 4, 4)
    for rank in range(4):
        assert planner.plan_for_decode_rank(rank).target_prefill_ranks == (0, 1, 2, 3)
    assert all(
        ranks == frozenset(range(4))
        for ranks in planner.decode_ranks_by_prefill_rank.values()
    )


def test_kvp_pd_rejects_incompatible_owner_topology():
    with pytest.raises(UnsupportedPDLayoutError, match="divide"):
        _kvp_planner(_kvp_layout(4), _kvp_layout(1), 2, 4)


def test_kvp_pd_rejects_head_partitioned_sharded_fields():
    contract = make_layout(
        make_group("kv", make_segment("layer.0.kv", shape=(2, 2), axis=1, extent=4))
    )
    contract = replace(
        contract, group_specs=(replace(contract.group_specs[0], shard_count=2),)
    )
    with pytest.raises(UnsupportedPDLayoutError, match="head-partitioned"):
        _kvp_planner(contract, contract, 2, 2)


def test_kvp_pd_manifest_preserves_virtual_ids_and_checks_virtual_bounds():
    contract = _kvp_layout(4)
    restored = CacheTransferContract.from_wire_bytes(contract.to_wire_bytes())
    assert restored == contract
    tables = {
        spec.group_id: np.array(
            [[restored.virtual_block_count(spec.group_id) - 1]], dtype=np.int32
        )
        for spec in restored.group_specs
    }
    manifest = build_cache_block_manifest(
        make_operation(tables),
        layout=restored,
        request_row=0,
        prefix_len=0,
        prompt_len=1,
        state_prefix_cached=False,
    )
    validate_cache_manifest(manifest, layout=restored, peer="decode")
    assert manifest.groups[0].block_ids[0] >= restored.plan.group("kv").page_count
    tables["kv"][0, 0] += 1
    with pytest.raises(CacheContractError, match="invalid block ID"):
        build_cache_block_manifest(
            make_operation(tables),
            layout=restored,
            request_row=0,
            prefix_len=0,
            prompt_len=1,
            state_prefix_cached=False,
        )


def test_registration_validates_owner_filter_decisions_once() -> None:
    from tokenspeed.runtime.pd.transfer_plan import (
        CachePageOwnerFilter,
        validate_rank_owner_filters,
    )

    group_specs = make_layout(
        make_group(
            "history",
            make_segment("layer.0.kv", dtype="bfloat16", shape=(8,), stride=32),
            shard_count=2,
        ),
        make_group(
            "state",
            make_segment("layer.1.state", dtype="bfloat16", shape=(8,), stride=32),
        ),
        capacity=16,
        page_bytes=64,
    ).group_specs
    history = CacheTransferFragment(
        group_id="history",
        field_id="layer.0.kv",
        src_byte_offset=0,
        dst_byte_offset=0,
        src_row_stride_bytes=16,
        dst_row_stride_bytes=16,
        bytes_per_row=16,
        rows_per_page=1,
    )
    state = replace(history, group_id="state", field_id="layer.1.state")

    def check(fragments, owner_filters):
        validate_rank_owner_filters(
            group_specs=group_specs, fragments=fragments, owner_filters=owner_filters
        )

    check((history, state), {"history": CachePageOwnerFilter(1, 2)})
    check((state,), {"history": None})
    with pytest.raises(ValueError, match="no owner-filter decision"):
        check((history,), {})
    with pytest.raises(ValueError, match="disagrees with its shard count"):
        check((history,), {"history": CachePageOwnerFilter(0, 4)})
    with pytest.raises(ValueError, match="but no owner filter"):
        check((history,), {"history": None})
    with pytest.raises(ValueError, match="carries no fragment of it"):
        check((state,), {"history": CachePageOwnerFilter(0, 2)})
    with pytest.raises(ValueError, match="which is not sharded"):
        check((state,), {"history": None, "state": CachePageOwnerFilter(0, 2)})


def test_bootstrap_layout_excludes_rank_local_field_addresses():
    from dataclasses import replace

    from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVBootstrapServer
    from tokenspeed.runtime.pd.mooncake.entities import KVArgs

    layout = _layout()
    names = tuple(f.field_id for f in layout.plan.fields)
    bootstrap = object.__new__(MooncakeKVBootstrapServer)
    bootstrap.prefill_cache_layout_wire = None
    bootstrap.prefill_cache_fields_by_stage = None
    wires = []
    for rank in range(2):
        bound = replace(
            layout,
            field_addresses={
                name: 0x1000 * (rank + 1) + i * 128 for i, name in enumerate(names)
            },
        )
        args = KVArgs(
            engine_rank=rank,
            kv_data_ptr=0x1000,
            ib_device="",
            gpu_id=rank,
            cache_layout=bound,
            cache_fields_by_stage=(names,),
        )
        wire = args.wire_layout.to_wire_bytes().decode("ascii")
        wires.append(wire)
        bootstrap._ingest_put_extra(
            {
                "cache_layout": wire,
                "cache_fields_by_stage": [list(names)],
                "pp_size": 1,
            }
        )
        # Endpoint registration still retains this rank's actual landing addresses.
        assert args.cache_layout.field_addresses == bound.field_addresses
        assert args.wire_layout.field_addresses is None
    assert wires[0] == wires[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
