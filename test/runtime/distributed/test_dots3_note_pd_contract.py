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

"""Exact Dots3-note CachePD geometry and descriptor-driven CPU byte copies.

No RDMA, CUDA allocation, kernels, model execution, or unequal-TP model claim.
Only the copy test allocates slabs; length/TP coverage is metadata-only.
"""

from contextlib import nullcontext
from test.ci_system.ci_register import register_cuda_ci
from test.runtime.cache.test_dots3_note_recipe import (
    LAYERS,
    MTP_PARENT_BYTES,
    PACKING,
    PARENT_BYTES,
    _layout,
    configure_mtp,
)
from test.runtime.cache.test_dots3_note_recipe import inputs as inputs
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.dots3_note import Dots3NoteAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.dots3_note import (
    Dots3NoteRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
    CacheLayerOwnership,
    cache_field_placement,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
    build_cache_block_manifest,
    build_cache_fields_by_producer_step,
    build_cache_layerwise_block_selection,
    build_cache_transfer_contract,
    build_cache_transfer_schema,
)
from tokenspeed.runtime.pd.mooncake.pack import PageFieldCopies
from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill
from tokenspeed.runtime.pd.transfer_plan import CacheTransferPlanner

# CPU buffers only; full runtime imports may query a visible GPU.
register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.fixture(params=[1, 2, 4])
def geometry(inputs, request):
    args = inputs["server_args"]
    args.disaggregation_mode = "prefill"
    args.attn_tp_size = args.mapping.attn.tp_size = 4
    inputs["attn_config"] = Dots3NoteAttnConfig.generate(
        args, inputs["model_config"], is_draft=False
    )
    if request.param > 1:
        configure_mtp(inputs, width=request.param)
    recipe = Dots3NoteRecipe(**inputs)
    layout = _layout(recipe)
    recipe.check_layout(layout)
    plan = layout.bind(1)
    schema = build_cache_transfer_schema(
        plan,
        model_config=inputs["model_config"],
        draft_model_config=inputs["draft_model_config"],
    )
    owner = CacheLayerOwnership(
        recipe.num_target_layers, recipe.num_draft_layers, (0, 46)
    )
    _, schedules = cache_field_placement(plan, (owner,))
    schedule = build_cache_fields_by_producer_step(
        plan, producer_fields_by_step=schedules[0]
    )
    return layout, tuple(spec for spec, _ in recipe.groups()), schema, schedule


def _request(geometry, prompt, *, first_parent, reverse):
    layout, specs, schema, _ = geometry
    tables, occupied = {}, set()
    parent = first_parent
    for spec in specs:
        gid, rows = spec.group_id, spec.block_granularity
        end = (prompt + rows - 1) // rows
        begin = 0 if gid == "full" else max(0, prompt - 512) // rows
        packing = dict(layout.group_packing)[gid]
        # Leave a sibling unused and cross packed-parent boundaries.
        first = (parent - 1) * packing + 1 + (packing > 1)
        ids = np.arange(first, first + end - begin, dtype=np.int32)
        parents = set(((ids - 1) // packing + 1).tolist())
        assert not occupied & parents  # Parents belong to only one group.
        occupied.update(parents)
        table = np.zeros((1, end), dtype=np.int32)
        table[0, begin:] = ids[::-1] if reverse else ids
        tables[gid] = table
        parent = max(parents) + 2  # An unallocated parent between groups.
    contract = CacheTransferContract(layout.bind(parent), specs, schema)
    return contract, SimpleNamespace(block_tables_arrays=lambda: tables)


def _manifest(contract, op, prefix, prompt):
    return build_cache_block_manifest(
        op, layout=contract, request_row=0, prefix_len=prefix, prompt_len=prompt
    )


def _chunks(contract, op, prefix, prompt):
    edges = sorted(
        {prefix, prompt}
        | {n for n in (32, 64, 128, 512, 1536, 2048) if prefix < n < prompt}
    )
    for start, end in zip(edges, edges[1:]):
        yield start, end, build_cache_layerwise_block_selection(
            op,
            layout=contract,
            request_row=0,
            prefix_len=prefix,
            prompt_len=prompt,
            chunk_start=start,
            chunk_end=end,
        )


def _planner(p, d, decode_tp):
    return CacheTransferPlanner(
        prefill_tp_size=4,
        decode_tp_size=decode_tp,
        prefill_layout=p,
        decode_layout=d,
        prefill_field_ids=None,
    )


def test_wire_geometry_and_producer_ownership(geometry):
    p, _ = _request(geometry, 513, first_parent=1, reverse=False)
    peer = CacheTransferContract.from_wire_bytes(p.to_wire_bytes())
    assert peer == p
    has_draft = "draft.swa" in dict(geometry[0].group_packing)
    parent_bytes = MTP_PARENT_BYTES if has_draft else PARENT_BYTES
    packing = PACKING | ({"draft.swa": 15} if has_draft else {})
    layers = LAYERS | ({"draft.swa": (46,)} if has_draft else {})
    assert peer.plan.lcm_block_bytes == parent_bytes
    assert len(peer.plan.fields) == 59 + has_draft
    assert {
        g.group_id: g.cache_blocks_per_lcm_block for g in peer.plan.groups
    } == packing
    schedule = geometry[3]
    assert schedule.step_count == 46 + has_draft
    if has_draft:
        assert schedule.fields_by_step[-1] == ("layer.46.latent_kv",)
        assert all(
            "layer.46.latent_kv" not in step for step in schedule.fields_by_step[:-1]
        )

    assert peer.transfer_schema.fields == ()  # All fields are replicated.
    for spec in peer.group_specs:
        gid = spec.group_id
        rows, width = (64, 576) if gid == "full" else (32, 1088)
        assert spec.block_granularity == rows
        assert spec.transfer_policy == "full_suffix" and not spec.replayable
        assert spec.sliding_window_tokens == (None if gid == "full" else 513)
        names = ("latent_kv", "index_k") if gid == "full" else ("latent_kv",)
        assert {f.field_id for f in peer.fields_for_group(gid)} == {
            f"layer.{layer}.{name}" for layer in layers[gid] for name in names
        }
        for layer in layers[gid]:
            assert set(schedule.fields_by_step[layer]) == {
                f"layer.{layer}.{name}" for name in names
            }
        for field in peer.fields_for_group(gid):
            index = field.field_id.endswith("index_k")
            assert field.shape == ((64, 132) if index else (rows, 1, width))
            assert field.dtype == ("uint8" if index else "bfloat16")
            assert field.payload_bytes == (8192 + 256 if index else rows * width * 2)
            assert field.page_stride_bytes == parent_bytes // packing[gid]


@pytest.mark.parametrize("decode_tp", [4, 2])
def test_replicated_tp_transport_plan_only(geometry, decode_tp):
    p, _ = _request(geometry, 513, first_parent=1, reverse=False)
    d, _ = _request(geometry, 513, first_parent=4, reverse=True)
    d = CacheTransferContract.from_wire_bytes(d.to_wire_bytes())
    assert p.plan.num_lcm_blocks != d.plan.num_lcm_blocks
    planner = _planner(p, d, decode_tp)
    for rank in range(decode_tp):
        route = planner.plan_for_decode_rank(rank)
        source_rank = rank * 4 // decode_tp
        assert route.target_prefill_ranks == (source_rank,)
        fragments = route.fragments_by_prefill_rank[source_rank]
        if decode_tp == 4:
            assert fragments == ()  # Equal-TP whole-field fast path, not whole parents.
        else:
            assert len(fragments) == len(p.plan.fields)
            assert {f.field_id for f in fragments} == {
                f.field_id for f in p.plan.fields
            }
            for f in fragments:
                field = p.plan.field(f.field_id)
                assert f.group_id == field.group_id
                assert f.src_byte_offset == f.dst_byte_offset == 0
                assert f.rows_per_page == 1
                assert f.bytes_per_row == field.payload_bytes
                assert (
                    f.src_row_stride_bytes
                    == f.dst_row_stride_bytes
                    == field.payload_bytes
                )


@pytest.mark.parametrize(
    "prompt", [31, 32, 33, 63, 64, 65, 512, 513, 514, 2047, 2048, 2049]
)
def test_cold_and_prefix_aligned_warm_selection(geometry, prompt):
    p, op = _request(geometry, prompt, first_parent=1, reverse=False)
    prefixes = {0, max(0, (prompt - 1) // 64 * 64)}
    if prompt > 512:
        prefixes.add(64)  # Warm prefix before the retained SWA tail as well.
    for prefix in sorted(prefixes):
        manifest = _manifest(p, op, prefix, prompt)
        chunks = tuple(_chunks(p, op, prefix, prompt))
        for i, spec in enumerate(p.group_specs):
            rows = spec.block_granularity
            begin = prefix if spec.group_id == "full" else max(prefix, prompt - 512)
            slots = tuple(range(begin // rows, (prompt + rows - 1) // rows))
            ids = tuple(op.block_tables_arrays()[spec.group_id][0, slots])
            assert manifest.groups[i].block_ids == ids
            actual = []
            for start, end, selection in chunks:
                selected = selection.groups[i]
                positions = tuple(
                    j
                    for j, slot in enumerate(slots)
                    if start < min((slot + 1) * rows, prompt) <= end
                )
                assert selected.destination_positions == positions
                assert selected.source_block_ids == tuple(ids[j] for j in positions)
                actual.extend(zip(positions, selected.source_block_ids))
            assert actual == list(enumerate(ids))


@pytest.mark.parametrize("width", [2, 4])
def test_mtp_candidate_wire_and_final_producer_barrier(inputs, width):
    from tokenspeed.runtime.execution.drafter.eagle import Eagle
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.decode import parse_prefill_status_message
    from tokenspeed.runtime.pd.utils import StepCounter

    configure_mtp(inputs, width=width)
    recipe = Dots3NoteRecipe(**inputs)
    plan = _layout(recipe).bind(1)
    placement, schedules = cache_field_placement(
        plan, (CacheLayerOwnership(46, 1, (0, 46)),)
    )
    assert len(placement[0]) == 60 and len(schedules[0]) == 47
    assert schedules[0][-1] == ("layer.46.latent_kv",)
    counter = SimpleNamespace(record_cache=Mock())
    executor = SimpleNamespace(
        draft_field_writer=Eagle,
        attn_backend=object(),
        draft_attn_backend=object(),
        _draft_final_step_counter=None,
    )
    ModelExecutor.register_draft_final_step_counter(executor, counter)
    ModelExecutor._record_draft_final_cache_step(executor, num_extends=0)
    counter.record_cache.assert_not_called()
    assert not StepCounter.is_step_ready(46, 46)
    ModelExecutor._record_draft_final_cache_step(executor, num_extends=1)
    counter.record_cache.assert_called_once_with()
    assert StepCounter.is_step_ready(47, 46)

    candidates = list(
        range(42, 42 + width)
    )  # Anchor plus proposals, not proposals only.
    messages = []
    manager = SimpleNamespace(
        cached_tokens={9: 512},
        bootstrap_logprobs={},
        bootstrap_token_cond=nullcontext(),
        _connect=lambda endpoint: (
            SimpleNamespace(send_multipart=messages.append),
            nullcontext(),
        ),
    )
    MooncakeKVManagerPrefill.sync_status_to_decode_endpoint(
        manager,
        "127.0.0.1",
        1234,
        9,
        TransferPoll.Success,
        0,
        bootstrap_token=candidates[0],
        spec_candidate_ids=candidates,
    )
    parsed = parse_prefill_status_message(messages[0])
    assert parsed == (9, TransferPoll.Success, 0, candidates[0], candidates, 512, None)
    states = RuntimeStates(
        req_pool_size=2, vocab_size=128, output_length=width, device="cpu"
    )
    states.future_input_map.fill_(-1)
    states.write_remote_spec_candidate_ids(1, parsed[4])
    assert states.future_input_map[1].tolist() == candidates
    assert states.future_input_map[2].tolist() == [-1] * width
    assert states.remote_spec_candidate_ready.tolist() == [False, True, False]
    with pytest.raises(RuntimeError, match="width mismatch"):
        states.write_remote_spec_candidate_ids(1, candidates[1:])


def _cpu_buffer(contract):
    buffer = torch.from_numpy(np.full(contract.plan.arena_bytes, 0xA5, dtype=np.uint8))
    bound, pointer = build_cache_transfer_contract(
        plan=contract.plan,
        buffer=buffer,
        group_specs=contract.group_specs,
        transfer_schema=contract.transfer_schema,
    )
    assert bound == contract and pointer == buffer.data_ptr()
    return buffer.numpy()


def _span(field, page):
    # Independent physical oracle: parent zero is reserved; page IDs start at 1.
    packing = 15 if field.group_id == "draft.swa" else PACKING[field.group_id]
    stride = field.page_stride_bytes
    parent_bytes = (
        MTP_PARENT_BYTES if stride * packing == MTP_PARENT_BYTES else PARENT_BYTES
    )
    assert stride * packing == parent_bytes
    start = parent_bytes + (page - 1) * stride + field.field_offset_bytes
    return slice(start, start + field.payload_bytes)


def test_full_and_layerwise_descriptor_cpu_copies(geometry):
    prompt = 513
    p, pop = _request(geometry, prompt, first_parent=1, reverse=False)
    d, dop = _request(geometry, prompt, first_parent=4, reverse=True)
    source, full, layerwise = _cpu_buffer(p), _cpu_buffer(d), _cpu_buffer(d)
    source.fill(0xE7)
    for field in p.plan.fields:
        for page in pop.block_tables_arrays()[field.group_id][0]:
            if page:
                seed = 13 * int(field.field_id.split(".")[1]) + 7 * int(page)
                pattern = np.arange(field.payload_bytes, dtype=np.uint32) + seed
                pattern = (pattern % 127).astype(np.uint8)
                if field.field_id.endswith("index_k"):
                    # Separate planar FP8 values / FP32 scale bytes.
                    pattern[8192:] |= 0x80
                source[_span(field, int(page))] = pattern
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(cache_layout=p, kv_data_ptr=source.ctypes.data)
    )
    transfer_plan = _planner(p, d, 4).plan_for_decode_rank(0)
    # Reuse slabs, but start each destination from its sentinel.
    for prefix in (0, 448):
        src, dst = _manifest(p, pop, prefix, prompt), _manifest(d, dop, prefix, prompt)
        expected = np.full_like(full, 0xA5)
        payload_bytes = 0
        for sg, dg in zip(src.groups, dst.groups, strict=True):
            assert sg.block_ids != dg.block_ids
            for field in p.fields_for_group(sg.group_id):
                for sp, dp in zip(sg.block_ids, dg.block_ids, strict=True):
                    expected[_span(field, dp)] = source[_span(field, sp)]
                    payload_bytes += field.payload_bytes

        def copy_to(target, *, selection, fields):
            copied = 0
            descriptors = MooncakeKVManagerPrefill._cache_transfer_blocks(
                manager,
                dst_ptr=target.ctypes.data,
                src_block_manifest=src,
                dst_block_manifest=dst,
                transfer_fragments=transfer_plan.fragments_by_prefill_rank[0],
                owner_filters=transfer_plan.owner_filters_by_prefill_rank[0],
                dst_cache_layout=d,
                block_selection=selection,
                field_ids=fields,
            )
            for item in descriptors:
                # Replicated fields use the page-gathered transfer API. Expand
                # its grid on CPU while keeping the independent whole-slab oracle.
                assert isinstance(item, PageFieldCopies)
                for sbase, sstride, dbase, dstride, size in item.fields.tolist():
                    assert size in (64 * 576 * 2, 32 * 1088 * 2, 8448)
                    for spage, dpage in zip(
                        item.src_pages.tolist(), item.dst_pages.tolist(), strict=True
                    ):
                        so = sbase + spage * sstride - source.ctypes.data
                        do = dbase + dpage * dstride - target.ctypes.data
                        assert 0 <= so <= source.size - size
                        assert 0 <= do <= target.size - size
                        target[do : do + size] = source[so : so + size]
                        copied += size
            return copied

        full.fill(0xA5)
        layerwise.fill(0xA5)
        assert copy_to(full, selection=None, fields=None) == payload_bytes
        copied = 0
        for _, _, selection in _chunks(p, pop, prefix, prompt):
            for step in range(geometry[3].step_count):
                fields = geometry[3].fields_in_range(step, step + 1)
                copied += copy_to(layerwise, selection=selection, fields=fields)
        assert copied == payload_bytes
        # Whole-slab equality checks every latent/index byte AND unchanged padding,
        # null parent, warm prefix, unallocated parents and unused packed siblings.
        np.testing.assert_array_equal(full, expected)
        np.testing.assert_array_equal(layerwise, expected)
