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

"""Mixed target/block-draft metadata and CPU adapters; no GPU is required.

Attention tests replace only the unavailable GPU operation. Lifecycle tests
stop at learned embedding after real runner/router/drafter metadata preparation.
Registered kernel numerics and graph execution require separate GPU tests.
"""

import os
import sys

import pytest
import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.layers.attention.backends.paged import mha
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.paged_attention import PagedAttention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# This suite is CPU-only, but the runtime suite is consumed by AMD CI too.
register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _backend(
    *, width=1, is_draft=False, block_decode=False, page_size=16, heads=(12, 2, 1)
):
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim=128,
        attn_tp_size=heads[2],
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=16,
        context_len=1024,
        max_bs=16,
        speculative_num_draft_tokens=width,
        is_draft=is_draft,
        draft_block_decode=block_decode,
        components=(spec,),
    )
    return mha.MHAAttnBackend(config, spec, kernel_page_size=page_size)


def _init(backend, *, lengths, queries, prefixes, mode, tables=None):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    queries = torch.tensor(queries, dtype=torch.int32)
    prefixes = torch.tensor(prefixes, dtype=torch.int32)
    bs = len(lengths)
    if tables is None:
        tables = torch.arange(bs * 64, dtype=torch.int32).view(bs, 64)
    backend.init_forward_metadata(
        bs,
        len(queries),
        lengths,
        tables,
        mode,
        extend_seq_lens=queries,
        extend_seq_lens_cpu=queries,
        extend_prefix_lens=prefixes,
        extend_prefix_lens_cpu=prefixes,
        extend_with_prefix=bool(prefixes.any()),
        query_shard=None,
        page_table_cpu=None,
    )
    return backend.forward_extend_metadata


@pytest.mark.parametrize("width,block_draft", [(1, False), (16, False), (16, True)])
@pytest.mark.parametrize("queries,prefixes", [([1], [0]), ([63, 65], [0, 512])])
@pytest.mark.parametrize("decodes", [1, 2, 4, 8])
def test_mixed_metadata_covers_every_query_and_preserves_request_order(
    width, block_draft, queries, prefixes, decodes
):
    backend = _backend(width=width, is_draft=block_draft, block_decode=block_draft)
    lengths = [q + p for q, p in zip(queries, prefixes)] + [513 + width] * decodes
    metadata = _init(
        backend,
        lengths=lengths,
        queries=queries,
        prefixes=prefixes,
        mode=ForwardMode.MIXED,
    )
    expected_queries = queries + [width] * decodes
    expected_cu = [0]
    expected_kv_cu = [0]
    for q, length in zip(expected_queries, lengths):
        expected_cu.append(expected_cu[-1] + q)
        expected_kv_cu.append(expected_kv_cu[-1] + length)
    assert metadata.extend_seq_lens.tolist() == expected_queries
    assert metadata.extend_seq_lens_cpu == expected_queries
    assert metadata.cu_extend_seq_lens.tolist() == expected_cu
    assert metadata.cu_extend_seq_lens_cpu == expected_cu
    assert metadata.cu_seqlens_kv.tolist() == expected_kv_cu
    assert metadata.extend_prefix_lens.tolist() == prefixes + [513] * decodes
    assert metadata.seq_lens.tolist() == lengths
    assert metadata.max_extend_seq_len == max(expected_queries)
    assert metadata.is_mixed
    assert metadata.page_table.shape[0] == len(lengths)


def test_round_transitions_keep_decode_storage_and_old_extend_metadata_independent():
    backend = _backend(width=16)
    backend.init_cuda_graph_state(16)
    decode_pointer = backend.seq_lens_buf.data_ptr()
    first = _init(
        backend,
        lengths=[65, 529],
        queries=[65],
        prefixes=[0],
        mode=ForwardMode.MIXED,
    )
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([530, 66], dtype=torch.int32),
        torch.ones((2, 64), dtype=torch.int32),
    )
    assert backend.seq_lens_buf.data_ptr() == decode_pointer
    assert backend.forward_decode_metadata.seq_lens.tolist() == [530, 66]
    second = _init(
        backend,
        lengths=[66],
        queries=[1],
        prefixes=[65],
        mode=ForwardMode.EXTEND,
    )
    assert second.cu_extend_seq_lens_cpu == [0, 1]
    assert second.extend_prefix_lens.tolist() == [65]
    assert not second.is_mixed
    assert first.cu_extend_seq_lens_cpu == [0, 65, 81]
    assert first.extend_prefix_lens.tolist() == [0, 513]
    assert backend.forward_decode_metadata.seq_lens.tolist() == [530, 66]


@pytest.mark.parametrize("width", [1, 16])
def test_router_refreshes_both_real_leaves_and_padded_writes_across_mixed_rounds(width):
    groups = {"full_attention": 16, "sliding_attention": 8}
    leaves = {
        gid: _backend(width=width, page_size=size) for gid, size in groups.items()
    }
    router = CacheGroupRouter(
        None,
        is_draft=False,
        spec_num_tokens=width,
        device="cpu",
        consumed_group_ids=None,
    )
    router.bind(
        CacheGroupGeometry(
            granularities={gid: 16 for gid in groups},
            attn_kinds={
                "full_attention": "full_history",
                "sliding_attention": "sliding",
            },
            full_history_group_id="full_attention",
            row_geometry={gid: (16, 1) for gid in groups},
            windows={"full_attention": None, "sliding_attention": 512},
        ),
        leaves,
    )
    bucket = 64
    router.init_cuda_graph_state(4, max_extend_tokens=bucket)
    layers = {}
    for index, gid in enumerate(groups):
        layers[gid] = PagedAttention(
            12,
            128,
            128**-0.5,
            2,
            index,
            sliding_window_size=-1 if index == 0 else 511,
            rotary_emb=None,
            qk_norm=None,
        )
        layers[gid].bind_cache_group(gid)

    # The second round changes request order/count, prefill count, query
    # lengths, prefixes and physical pages while retaining the same token
    # bucket. CPU tensors test live metadata, not actual graph replay.
    rounds = [
        ([7, 2, 9], [3, 5], [16, 0], [19, 5, 48 + width]),
        ([9, 7], [17], [32], [49, 64 + width]),
    ]
    write_pointers = {}
    for round_index, (requests, queries, prefixes, lengths) in enumerate(rounds):
        bs, num_extends = len(requests), len(queries)
        tables = {
            gid: torch.tensor(
                [
                    [
                        1000 * (round_index + 1)
                        + 100 * group_index
                        + 10 * request
                        + column
                        for column in range(6)
                    ]
                    for request in requests
                ],
                dtype=torch.int32,
            )
            for group_index, gid in enumerate(groups)
        }
        query_tensor = torch.tensor(queries, dtype=torch.int32)
        prefix_tensor = torch.tensor(prefixes, dtype=torch.int32)
        router.init_forward_metadata(
            bs,
            num_extends,
            torch.tensor(requests, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            ForwardMode.MIXED,
            block_tables=tables,
            extend_seq_lens=query_tensor,
            extend_seq_lens_cpu=query_tensor.clone(),
            extend_prefix_lens=prefix_tensor,
            extend_prefix_lens_cpu=prefix_tensor.clone(),
            extend_replay_lens_cpu=torch.zeros_like(prefix_tensor),
            extend_prompt_lens_cpu=prefix_tensor + query_tensor,
            extend_with_prefix=bool(prefix_tensor.any()),
            query_shard=None,
            block_tables_cpu={},
        )
        all_queries = queries + [width] * (bs - num_extends)
        cumulative_queries = [0]
        for count in all_queries:
            cumulative_queries.append(cumulative_queries[-1] + count)
        for gid, page_size in groups.items():
            metadata = leaves[gid].forward_extend_metadata
            assert metadata.is_mixed
            assert metadata.seq_lens.tolist() == lengths
            assert metadata.extend_seq_lens.tolist() == all_queries
            assert metadata.extend_seq_lens_cpu == all_queries
            assert metadata.cu_extend_seq_lens.tolist() == cumulative_queries
            assert metadata.cu_extend_seq_lens_cpu == cumulative_queries
            assert metadata.extend_prefix_lens.tolist() == prefixes + [
                length - width for length in lengths[num_extends:]
            ]
            ratio = 16 // page_size
            expected_tables = torch.zeros_like(metadata.page_table)
            for request, blocks in enumerate(tables[gid].tolist()):
                expanded = [
                    block * ratio + offset
                    for block in blocks
                    for offset in range(ratio)
                ]
                expected_tables[request, : len(expanded)] = torch.tensor(
                    expanded, dtype=torch.int32
                )
            assert torch.equal(metadata.page_table, expected_tables)
            # Resolve directly from scheduler block IDs, independently of
            # kernel-page expansion. Both groups must preserve slot identity.
            expected_prefill, expected_decode = [], []
            for request, count in enumerate(all_queries):
                positions = range(lengths[request] - count, lengths[request])
                slots = [
                    int(tables[gid][request, position // 16]) * 16 + position % 16
                    for position in positions
                ]
                (expected_prefill if request < num_extends else expected_decode).extend(
                    slots
                )
            layer = layers[gid]
            assert (
                router.write_locations(layer, ForwardMode.EXTEND).tolist()
                == expected_prefill
            )
            assert (
                router.write_locations(layer, ForwardMode.DECODE).tolist()
                == expected_decode
            )
            expected_live = expected_prefill + expected_decode
            padded = router.padded_write_locations(layer, ForwardMode.MIXED, bucket)
            assert padded.tolist() == expected_live + [0] * (
                bucket - len(expected_live)
            )
            if gid in write_pointers:
                assert padded.data_ptr() == write_pointers[gid]
            write_pointers[gid] = padded.data_ptr()


def test_nonblock_draft_mixed_layout_remains_explicitly_unsupported():
    backend = _backend(width=16, is_draft=True, block_decode=False)
    with pytest.raises(RuntimeError, match="requires block decoding"):
        _init(
            backend,
            lengths=[3, 32],
            queries=[3],
            prefixes=[0],
            mode=ForwardMode.MIXED,
        )


class _BeforeModelWeights(Exception):
    """End the CPU metadata test at the first learned-model operation."""


def _stop_before_weights(*args, **kwargs):
    raise _BeforeModelWeights


def _block_runtime():
    """Real metadata owners; omit only GPU execution and learned weights."""
    groups = ("full_attention", "sliding_attention", "draft_attention")
    geometry = CacheGroupGeometry(
        granularities=dict.fromkeys(groups, 16),
        attn_kinds=dict(zip(groups, ("full_history", "sliding", "sliding"))),
        full_history_group_id=groups[0],
        row_geometry=dict.fromkeys(groups, (16, 1)),
        windows=dict(zip(groups, (None, 512, 4096))),
    )
    leaves = {
        gid: _backend(
            width=16,
            is_draft=index == 2,
            block_decode=index == 2,
            heads=(40 if index == 2 else 48, 8, 8),
        )
        for index, gid in enumerate(groups)
    }
    routers = []
    for is_draft, selected in ((False, groups[:2]), (True, groups[2:])):
        router = CacheGroupRouter(
            None,
            is_draft=is_draft,
            spec_num_tokens=16,
            device="cpu",
            consumed_group_ids=selected,
        )
        router.bind(geometry, {gid: leaves[gid] for gid in selected})
        router.init_cuda_graph_state(4, max_extend_tokens=64)
        routers.append(router)
    layers = {}
    for index, gid in enumerate(groups):
        layers[gid] = PagedAttention(
            leaves[gid].tp_q_head_num,
            128,
            128**-0.5,
            1,
            index,
            sliding_window_size=(-1, 511, 4095)[index],
            rotary_emb=None,
            qk_norm=None,
        )
        layers[gid].bind_cache_group(gid)

    # Constructors also require complete model executors and device streams.
    # Populate their declared metadata state, then invoke the real shared
    # runner and DFlash2 methods. No method under test is replaced.
    drafter = DFlash2.__new__(DFlash2)
    drafter.spec_num_tokens = drafter.draft_query_width = 16
    drafter.input_buffers = InputBuffers(4, 64, 16, device="cpu")
    drafter.runtime_states = RuntimeStates(16, 32, 16, device="cpu")
    drafter.draft_seq_lens_buf = torch.zeros(4, dtype=torch.int32)
    drafter.block_ids_buf = torch.ones((4, 16), dtype=torch.int32)
    drafter.block_positions_buf = torch.zeros((4, 16), dtype=torch.int64)
    drafter.block_offsets = torch.arange(16, dtype=torch.int64).unsqueeze(0)
    drafter.attention_kind = "gqa"
    drafter.attn_backend = routers[1]
    drafter.token_to_kv_pool = None  # No model/cache-storage operation is reached.
    drafter.round_block_tables = None
    drafter.embed_tokens = _stop_before_weights
    runner = ForwardStepRunner.__new__(ForwardStepRunner)
    runner.attn_backend, runner.draft_attn_backend = routers
    runner.drafter = drafter
    runner.max_tokens_per_req = 16
    return runner, drafter, leaves, layers


def _round_tables(groups, slots, generation):
    return {
        gid: torch.tensor(
            [
                [1000 * generation + 100 * index + 8 * slot + col for col in range(8)]
                for slot in slots
            ],
            dtype=torch.int32,
        )
        for index, gid in enumerate(groups)
    }


def _expected_locations(table, starts, width):
    return [
        int(table[row, position // 16]) * 16 + position % 16
        for row, start in enumerate(starts)
        for position in range(start, start + width)
    ]


@pytest.mark.parametrize("accepted", [(0, 7), (1, 16), (16, 0)])
@pytest.mark.parametrize("for_graph_replay", [False, True])
def test_block_draft_mixed_to_decode_refreshes_all_owners_and_reordered_slots(
    monkeypatch, accepted, for_graph_replay
):
    runner, drafter, leaves, layers = _block_runtime()
    buffers = drafter.input_buffers
    slots = torch.tensor([7, 2, 9], dtype=torch.int64)
    lengths = torch.tensor([19, 48, 80], dtype=torch.int32)
    tables = _round_tables(leaves, slots.tolist(), 1)
    buffers.req_pool_indices_buf[:3].copy_(slots)
    buffers.input_lengths_buf[:3] = torch.tensor([3, 1, 1])
    drafter.runtime_states.valid_cache_lengths[slots] = torch.tensor(
        [16, 32, 64], dtype=torch.int32
    )
    positions = torch.cat(
        (torch.arange(16, 19), torch.arange(32, 48), torch.arange(64, 80))
    )
    buffers.positions_buf[:35].copy_(positions)
    query, prefix = torch.tensor([3]), torch.tensor([16])
    runner._init_forward_metadata(
        3,
        1,
        slots,
        lengths,
        ForwardMode.MIXED,
        block_tables=tables,
        extend_seq_lens=query,
        extend_seq_lens_cpu=query,
        extend_prefix_lens=prefix,
        extend_prefix_lens_cpu=prefix,
        extend_replay_lens_cpu=torch.zeros_like(prefix),
        extend_prompt_lens_cpu=query + prefix,
        extend_with_prefix=True,
        query_shard=None,
        block_tables_cpu={},
    )
    assert drafter.round_block_tables is tables
    assert drafter.draft_seq_lens_buf[:3].tolist() == lengths.tolist()
    for gid, leaf in leaves.items():
        assert (leaf.tp_q_head_num, leaf.tp_kv_head_num) == (
            5 if gid == "draft_attention" else 6,
            1,
        )
        assert leaf.forward_extend_metadata.extend_seq_lens.tolist() == [3, 16, 16]
        assert leaf.forward_extend_metadata.extend_prefix_lens.tolist() == [16, 32, 64]
    target = runner.attn_backend
    for gid in target.group_ids:
        padded = target.padded_write_locations(layers[gid], ForwardMode.MIXED, 64)
        expected = _expected_locations(tables[gid][:1], [16], 3)
        expected += _expected_locations(tables[gid][1:], [32, 64], 16)
        assert padded.tolist() == expected + [0] * 29

    ctx = ForwardContext(
        attn_backend=target,
        token_to_kv_pool=None,
        bs=3,
        num_extends=1,
        input_num_tokens=35,
        forward_mode=ForwardMode.MIXED,
        output_layout=ForwardOutputLayout(1, 1, 2, 16),
    )
    accepts = torch.tensor([1, *accepted], dtype=torch.int32)
    drafter._update_draft_prefix_lengths(ctx, accepts)
    expected_prefixes = [19, 32 + accepted[0], 64 + accepted[1]]
    assert drafter.draft_seq_lens_buf[:3].tolist() == expected_prefixes
    # Rejected verification rows can contain arbitrary values. Neither a
    # rejected tail nor a zero-acceptance row may advance the draft history.
    for row, count in enumerate(accepted):
        start = 3 + row * 16
        buffers.positions_buf[start + count : start + 16].fill_(999999)
    drafter._update_draft_prefix_lengths(ctx, accepts)
    assert drafter.draft_seq_lens_buf[:3].tolist() == expected_prefixes
    with pytest.raises(_BeforeModelWeights):
        drafter._draft_native(torch.ones(3, dtype=torch.int32))
    draft = runner.draft_attn_backend
    draft_layer = layers["draft_attention"]
    draft_leaf = leaves["draft_attention"]
    assert draft.forward_write_locations(draft_layer, ForwardMode.DECODE).tolist() == (
        _expected_locations(tables["draft_attention"], expected_prefixes, 16)
    )
    assert draft_leaf.forward_decode_metadata.seq_lens.view(3, 16).tolist() == [
        [prefix + 16] * 16 for prefix in expected_prefixes
    ]

    # Next round: reorder all live slots, replace every physical page, and
    # pad to a four-request decode bucket. The sentinel slot must not inherit
    # any real request's table or history. These are CPU replay *inputs*, not
    # a claim that a device graph was captured or replayed.
    next_slots = torch.tensor([9, 7, 2, 16], dtype=torch.int64)
    next_prefixes = [expected_prefixes[i] for i in (2, 0, 1)]
    next_lengths = torch.tensor(
        [p + 16 for p in next_prefixes] + [1], dtype=torch.int32
    )
    next_tables = _round_tables(leaves, next_slots[:3].tolist(), 2)
    buffers.req_pool_indices_buf.copy_(next_slots)
    runner._prepare_decode_metadata(
        4,
        3,
        next_slots,
        next_lengths,
        ForwardMode.DECODE,
        use_graph=for_graph_replay,
        block_tables=next_tables,
    )
    assert drafter.round_block_tables is next_tables
    assert drafter.draft_seq_lens_buf.tolist() == next_lengths.tolist()
    for gid, leaf in leaves.items():
        metadata = leaf.forward_decode_metadata
        expansion = 16 if gid == "draft_attention" else 1
        expanded = metadata.page_table.view(4, expansion, -1)
        assert not expanded[3].any()
        expected = next_tables[gid][:, None, :].expand(3, expansion, 8)
        assert torch.equal(expanded[:3, :, :8], expected)
        if gid in target.group_ids:
            assert target.write_locations(layers[gid], ForwardMode.DECODE).tolist() == (
                _expected_locations(next_tables[gid], next_prefixes, 16) + [0] * 16
            )
    # The native block step republishes all requests (not just the old mixed
    # decode suffix) after the runner's shared refresh.
    drafter.runtime_states.valid_cache_lengths[next_slots] = torch.tensor(
        next_prefixes + [0], dtype=torch.int32
    )
    buffers.positions_buf.copy_(
        torch.tensor([p + offset for p in next_prefixes + [0] for offset in range(16)])
    )
    next_accepts = torch.tensor([0, 16, 7, 0], dtype=torch.int32)
    block_prefixes = [p + a for p, a in zip(next_prefixes + [0], next_accepts.tolist())]
    pointer = draft_leaf.forward_decode_metadata.seq_lens.data_ptr()
    block_bs = 4 if for_graph_replay else 3
    decode_ctx = ForwardContext(
        attn_backend=target,
        token_to_kv_pool=None,
        bs=block_bs,
        num_extends=0,
        input_num_tokens=block_bs * 16,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, block_bs, 16),
    )
    with monkeypatch.context() as capture, pytest.raises(_BeforeModelWeights):
        if for_graph_replay:
            # Select the in-graph metadata branch without constructing a GPU
            # stream or graph. The outer CPU test runner denies initialization.
            capture.setattr(torch.cuda, "is_available", lambda: True)
            capture.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        drafter._update_draft_prefix_lengths(decode_ctx, next_accepts)
        assert (
            drafter.draft_seq_lens_buf[:block_bs].tolist() == block_prefixes[:block_bs]
        )
        drafter._draft_native(torch.ones(block_bs, dtype=torch.int32))
    assert draft_leaf.forward_decode_metadata.seq_lens.data_ptr() == pointer
    assert draft_leaf.forward_decode_metadata.seq_lens.view(block_bs, 16).tolist() == [
        [p + 16] * 16 for p in block_prefixes[:block_bs]
    ]
    assert draft.forward_write_locations(draft_layer, ForwardMode.DECODE).tolist() == (
        _expected_locations(next_tables["draft_attention"], block_prefixes[:3], 16)
        + [0] * (16 * (block_bs - 3))
    )
    assert drafter.block_positions_buf[:block_bs].tolist() == [
        list(range(p, p + 16)) for p in block_prefixes[:block_bs]
    ]


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.IDLE])
def test_pure_decode_metadata_still_requires_the_unified_refresh(mode):
    with pytest.raises(RuntimeError, match="refresh_decode_metadata"):
        _init(_backend(), lengths=[3], queries=[], prefixes=[], mode=mode)


@pytest.mark.parametrize("queries", [[], [3, 3]])
def test_malformed_mixed_request_partition_is_rejected(queries):
    with pytest.raises(ValueError, match="both prefill and decode"):
        _init(
            _backend(),
            lengths=[3, 3],
            queries=queries,
            prefixes=[0] * len(queries),
            mode=ForwardMode.MIXED,
        )


class _CPUCache:
    """Tiny cache replacing GPU-backed pool storage, not runtime dispatch."""

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def get_key_buffer(self, layer_id):
        assert layer_id == 0
        return self.key

    def get_value_buffer(self, layer_id):
        assert layer_id == 0
        return self.value


def _cached_attention_cpu(
    *,
    q,
    cu_seqlens_q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    window_left,
    **kwargs
):
    """CPU stand-in for the unavailable GPU operation, using its public ABI."""
    outputs = []
    for request, length in enumerate(cache_seqlens.tolist()):
        start, end = cu_seqlens_q[request : request + 2].tolist()
        slots = page_table[request, : (length + 15) // 16].long()
        repeats = q.shape[1] // k_cache.shape[2]
        key = k_cache[slots].flatten(0, 1)[:length].repeat_interleave(repeats, dim=1)
        value = v_cache[slots].flatten(0, 1)[:length].repeat_interleave(repeats, dim=1)
        for offset, query in enumerate(q[start:end]):
            position = length - (end - start) + offset
            left = max(0, position - window_left) if window_left >= 0 else 0
            scores = (
                torch.einsum(
                    "hd,khd->hk", query.double(), key[left : position + 1].double()
                )
                / 128**0.5
            )
            outputs.append(
                torch.einsum(
                    "hk,khd->hd",
                    scores.softmax(-1),
                    value[left : position + 1].double(),
                )
            )
    return torch.stack(outputs).to(q.dtype)


@pytest.mark.parametrize("window", [-1, 2])
@pytest.mark.parametrize("width", [1, 16])
@pytest.mark.parametrize("heads", [(12, 2, 1), (48, 8, 8)])
def test_mixed_forward_keeps_decode_history_and_drops_only_padding(
    monkeypatch, window, width, heads
):
    backend = _backend(width=width, heads=heads)
    # Some registry solutions can prefer fresh-prefill without cache. Mixed
    # decode rows must still consume their history even when prefills are fresh.
    monkeypatch.setattr(backend, "plan", lambda **kwargs: {"extend_mode": "postwrite"})
    monkeypatch.setattr(mha, "mha_extend_with_kvcache", _cached_attention_cpu)
    layer = PagedAttention(
        backend.tp_q_head_num,
        128,
        128**-0.5,
        backend.tp_kv_head_num,
        0,
        sliding_window_size=window,
        rotary_emb=None,
        qk_norm=None,
    )
    generator = torch.Generator().manual_seed(123)
    shape = (6, 16, backend.tp_kv_head_num, 128)
    key = torch.randn(shape, generator=generator).to(torch.bfloat16)
    value = torch.randn(shape, generator=generator).to(torch.bfloat16)
    cache = _CPUCache(key, value)
    tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    query_lengths = [3, width]
    lengths = [3, 16 + width]
    _init(
        backend,
        lengths=lengths,
        queries=[3],
        prefixes=[0],
        mode=ForwardMode.MIXED,
        tables=tables,
    )
    query = torch.randn(
        (sum(query_lengths) + 7, backend.tp_q_head_num, 128), generator=generator
    ).to(torch.bfloat16)
    query[-7:] = float("nan")
    before_key, before_value = key.clone(), value.clone()
    output = backend.forward_extend(
        query,
        query[:, : backend.tp_kv_head_num],
        query[:, : backend.tp_kv_head_num],
        layer,
        torch.empty(0, dtype=torch.int32),
        cache,
        2,
    )
    # Independent per-request evaluation has no mixed metadata or padded rows.
    expected = []
    cursor = 0
    for request, count in enumerate(query_lengths):
        expected.append(
            _cached_attention_cpu(
                q=query[cursor : cursor + count],
                cu_seqlens_q=torch.tensor([0, count]),
                k_cache=key,
                v_cache=value,
                page_table=tables[request : request + 1],
                cache_seqlens=torch.tensor([lengths[request]]),
                window_left=window,
            )
        )
        cursor += count
    torch.testing.assert_close(output, torch.cat(expected).flatten(1), rtol=0, atol=0)
    assert torch.isfinite(output).all()
    assert torch.equal(key, before_key) and torch.equal(value, before_value)


def _cached_decode_cpu(*, q, **kwargs):
    # Block drafting expands each request to sixteen one-query metadata rows,
    # all seeing the same complete block (non-causal within the block).
    return _cached_attention_cpu(
        q=q, cu_seqlens_q=torch.arange(q.shape[0] + 1), **kwargs
    )


@pytest.mark.parametrize("accepted", [0, 7, 16])
@pytest.mark.parametrize("window", [-1, 7])
def test_block_decode_after_mixed_excludes_poison_beyond_accepted_history_and_block(
    monkeypatch, accepted, window
):
    backend = _backend(width=16, is_draft=True, block_decode=True, heads=(40, 8, 8))
    backend.init_cuda_graph_state(2)
    tables = torch.zeros((2, backend.max_num_pages), dtype=torch.int32)
    tables[:, :5] = torch.arange(1, 11).view(2, 5)
    _init(
        backend,
        lengths=[3, 48],
        queries=[3],
        prefixes=[0],
        mode=ForwardMode.MIXED,
        tables=tables,
    )
    # The block starts at the accepted endpoint, not the end of verification.
    # Its sixteen new KV rows replace the rejected tail. Everything outside
    # history + this replacement block is poisoned, including unused pages.
    ends = [3 + 16, 32 + accepted + 16]
    backend.refresh_decode_metadata(2, 2, torch.tensor(ends), tables)
    key = torch.full((11, 16, 1, 128), float("nan"), dtype=torch.bfloat16)
    value = torch.full_like(key, float("nan"))
    for row, end in enumerate(ends):
        for position in range(end):
            page = tables[row, position // 16]
            key[page, position % 16] = 0
            value[page, position % 16] = position
    monkeypatch.setattr(mha, "mha_decode_with_kvcache", _cached_decode_cpu)
    layer = PagedAttention(
        5,
        128,
        128**-0.5,
        1,
        0,
        sliding_window_size=window,
        rotary_emb=None,
        qk_norm=None,
    )
    query = torch.zeros((32, 5, 128), dtype=torch.bfloat16)
    output = backend.forward_decode(
        query,
        None,
        None,
        layer,
        torch.empty(0, dtype=torch.int32),
        _CPUCache(key, value),
        2,
    ).view(2, 16, 5, 128)
    expected = torch.tensor(
        [
            (max(0, end - window - 1) + end - 1) / 2 if window >= 0 else (end - 1) / 2
            for end in ends
        ],
        dtype=torch.bfloat16,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, expected[:, None, None, None].expand_as(output), rtol=0, atol=0
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
