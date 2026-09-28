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

"""Normal-import recipe -> arena -> pool -> router GPU integration.

Tables and model-projected inputs are synthetic; allocation/prefix lifecycle and
model loading are not tested here. The diagnostic probe reuses these checks.
"""

import math
from test.runtime.cache.test_dots3_note_compatibility import build_mtp_pools
from test.runtime.cache.test_dots3_note_recipe import LAYERS, PARENT_BYTES
from test.runtime.cache.test_dots3_note_recipe import inputs as inputs
from test.runtime.cache.test_dots3_note_recipe import mtp_inputs as mtp_inputs
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode_topk
from tokenspeed_kernel.ops.attention.dsa.triton import _flatten_dense_kv_cache

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.configs.dots3_note import Dots3NoteAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import Dots3NoteCachePool
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention, bind_cache_groups

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def build_runtime(inputs):
    """Construct production objects; only the supplied model/serving inputs are fake."""
    config = inputs["attn_config"]
    setup = prepare_cache_setup(family="dots3_note", **inputs)
    arena = create_cache_arena(
        setup.spec, device=config.device, enable_memory_saver=False
    )
    pool = create_cache_pool(
        setup.spec,
        config,
        arena,
        num_layers=setup.num_target_layers,
        rank=0,
        field_layer_offset=0,
    )
    composite = config.component(Dots3NoteAttnConfig)
    layers = torch.nn.ModuleList()
    for layer_id, kind in enumerate(composite.cache_layer_types):
        full = kind == "full_attention"
        spec = composite.full if full else composite.swa
        layers.append(
            PagedAttention(
                num_heads=spec.num_attention_heads // spec.attn_tp_size,
                num_kv_heads=spec.num_attention_heads // spec.attn_tp_size,
                head_dim=spec.kv_cache_dim,
                v_head_dim=spec.kv_lora_rank,
                scaling=spec.scaling,
                layer_id=layer_id,
                logit_cap=0.0,
                sliding_window_size=-1 if full else 512,
                rotary_emb=None,
                qk_norm=None,
            )
        )
    bind_cache_groups(layers, pool)
    return SimpleNamespace(
        config=config,
        setup=setup,
        arena=arena,
        pool=pool,
        router=None,
        layers=layers,
    )


def bind_router(runtime):
    # Require successful package initialization even if an earlier test left
    # imported leaf modules behind after a failed normal backend import.
    import tokenspeed.runtime.layers.attention.backends  # noqa: F401
    from tokenspeed.runtime.layers.attention.registry import _create_attn_backend

    if runtime.router is None:
        router = _create_attn_backend(AttentionArch.DSA, runtime.config)
        router.set_cache_pool(runtime.pool)
        router.init_cuda_graph_state(runtime.config.max_bs)
        runtime.router = router
    return runtime.router


@pytest.fixture
def runtime(inputs):
    args, model = inputs["server_args"], inputs["model_config"]
    args.device = "cuda"
    args.max_num_seqs = 4
    args.max_total_tokens = 8192
    model.context_len = 4098
    inputs["cache_budget_bytes"] = 512 << 20
    inputs["attn_config"] = Dots3NoteAttnConfig.generate(args, model, is_draft=False)
    torch.manual_seed(1234)
    return build_runtime(inputs)


def check_binding(runtime):
    arena, pool = runtime.arena, runtime.pool
    plan = arena.plan
    assert isinstance(pool, Dots3NoteCachePool)
    assert plan.prefix_granularity == 64
    assert plan.lcm_block_bytes == PARENT_BYTES
    assert arena.buffer.numel() == plan.arena_bytes
    assert arena.buffer.count_nonzero() == 0
    assert pool.history_group_by_layer() == {
        layer: gid for gid, layers in LAYERS.items() for layer in layers
    }
    for layer in runtime.layers:
        gid = layer.group_id
        field_id = f"layer.{layer.layer_id}.latent_kv"
        field = plan.field(field_id)
        kv = pool.get_key_buffer(layer.layer_id)
        pages = arena.field_pages(field_id)
        assert kv is pages
        assert pages.shape == (plan.group(gid).page_count, *field.shape)
        assert kv.ndim == 4 and kv.dtype == torch.bfloat16
        assert kv.untyped_storage().data_ptr() == arena.buffer.data_ptr()
        assert kv.data_ptr() - arena.buffer.data_ptr() == plan.field_page_byte_offset(
            field_id, 0
        )
        stride = 213_760 if gid == "swa.2" else PARENT_BYTES
        assert kv.stride(0) * kv.element_size() == stride
        assert tuple(kv.stride()[1:]) == (field.shape[-1], field.shape[-1], 1)
        key, value = pool.get_kv_buffer(layer.layer_id)
        assert key is kv
        assert value.shape == (*kv.shape[:-1], kv.shape[-1] - 64)
        assert value.untyped_storage().data_ptr() == arena.buffer.data_ptr()
        assert pool.get_value_buffer(layer.layer_id).data_ptr() == value.data_ptr()
        if gid == "full":
            index = pool.get_index_k_buffer(layer.layer_id)
            assert index.shape == (kv.shape[0], 8448)
            assert index.stride() == (PARENT_BYTES, 1)
            assert index.untyped_storage().data_ptr() == arena.buffer.data_ptr()
            assert (
                index.data_ptr()
                == arena.field_pages(f"layer.{layer.layer_id}.index_k").data_ptr()
            )
            dense = _flatten_dense_kv_cache(kv)
            assert dense.untyped_storage().data_ptr() == arena.buffer.data_ptr()
            assert dense.stride(0) == kv.stride(0)
        else:
            with pytest.raises(ValueError, match="has no index cache"):
                pool.get_index_k_buffer(layer.layer_id)
    with pytest.raises(ValueError, match="not planned"):
        arena.field_pages("missing")


def check_router_binding(runtime):
    router = bind_router(runtime)
    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
    from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend

    for layer in runtime.layers:
        leaf = router.leaf_for(layer)
        assert leaf.cache_pool is runtime.pool
        assert (
            leaf.kernel_page_size
            == runtime.pool.get_key_buffer(layer.layer_id).shape[1]
        )
        assert leaf.kv_lora_rank == layer.v_head_dim
        assert isinstance(
            leaf, DSABackend if layer.group_id == "full" else MLAAttnBackend
        )


def _expect_latent(buffer, plan, layer_id, locations, rows):
    """Independent byte addressing, never a test-only strided cache view."""
    field_id = f"layer.{layer_id}.latent_kv"
    field = plan.field(field_id)
    page_size, _, width = field.shape
    offsets = (
        plan.field_page_byte_offset(field_id, 0)
        + locations // page_size * field.page_stride_bytes
        + locations % page_size * width * 2
    )
    columns = torch.arange(width * 2, device=buffer.device)
    buffer[offsets[:, None] + columns] = (
        rows.contiguous().view(torch.uint8).reshape(-1, width * 2)
    )


def check_latent_scatter(runtime, count):
    arena, pool = runtime.arena, runtime.pool
    arena.buffer.fill_(0xA5)
    expected = arena.buffer.clone()
    for layer in runtime.layers:
        group = arena.plan.group(layer.group_id)
        page_size = pool.get_key_buffer(layer.layer_id).shape[1]
        # Distinct parent ranges: groups are alternative uses of the same plane.
        parent = 1 + list(LAYERS).index(layer.group_id) * 20
        first_page = 1 + (parent - 1) * group.cache_blocks_per_lcm_block
        if count == 6:
            slots = [
                0,
                first_page * page_size,
                (first_page + 1) * page_size - 1,
                (first_page + 1) * page_size,
                (first_page + 5) * page_size - 1,
                (first_page + 5) * page_size,
            ]
            loc = torch.tensor(slots, device="cuda", dtype=torch.int64)
        else:
            loc = torch.cat(
                (
                    torch.zeros(1, device="cuda", dtype=torch.int64),
                    torch.randperm(count - 1, device="cuda") + first_page * page_size,
                )
            )
        rows = torch.randn(
            (count, 1, layer.head_dim), device="cuda", dtype=torch.bfloat16
        )
        pool.set_mla_kv_buffer(layer, loc, rows[..., :-64], rows[..., -64:])
        _expect_latent(expected, arena.plan, layer.layer_id, loc, rows)
        assert torch.equal(
            arena.buffer, expected
        ), f"layer {layer.layer_id}: unexpected arena bytes"
        for dtype in (torch.bfloat16, torch.float32):
            latent, rope = pool.get_mla_kv_buffer(layer, loc.flip(0), dst_dtype=dtype)
            torch.testing.assert_close(
                latent, rows.flip(0)[..., :-64].to(dtype), atol=0, rtol=0
            )
            torch.testing.assert_close(
                rope, rows.flip(0)[..., -64:].to(dtype), atol=0, rtol=0
            )
    # Slot 0 is writable graph padding. All other null-parent bytes and live
    # fields not named by these writes were covered by the full-buffer oracle.


def check_index_scatter(runtime):
    arena, pool = runtime.arena, runtime.pool
    arena.buffer.fill_(0xA5)
    expected = arena.buffer.clone()
    loc = torch.tensor(
        [0, 64, 95, 127, 128, 383, 384], device="cuda", dtype=torch.int64
    )
    keys = (
        torch.arange(loc.numel() * 128, device="cuda")
        .remainder(17)
        .reshape(-1, 128)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.tensor(
        [0.125, 0.25, 0.5, 1, 2, 4, 8], device="cuda", dtype=torch.float32
    )[:, None]
    for layer_id in LAYERS["full"]:
        pool.set_index_k_buffer(layer_id, loc, keys, scales)
        field_id = f"layer.{layer_id}.index_k"
        for i, slot in enumerate(loc.tolist()):
            page, row = divmod(slot, 64)
            base = arena.plan.field_page_byte_offset(field_id, page)
            expected[base + row * 128 : base + (row + 1) * 128] = keys[i].view(
                torch.uint8
            )
            expected[base + 8192 + row * 4 : base + 8192 + (row + 1) * 4] = scales[
                i
            ].view(torch.uint8)
        assert torch.equal(
            arena.buffer, expected
        ), f"layer {layer_id}: index planar write or sentinel mismatch"
    with pytest.raises(ValueError, match="E4M3FN keys and FP32 scales"):
        pool.set_index_k_buffer(0, loc, keys.float(), scales)


def _tables(runtime, bs):
    bind_router(runtime)
    return {
        gid: torch.zeros((bs, leaf.max_num_pages), device="cuda", dtype=torch.int32)
        for gid, leaf in runtime.router.leaves.items()
    }


def _refresh(runtime, tables, lengths, *, actual_bs, replay):
    runtime.router.refresh_decode_metadata(
        len(lengths),
        actual_bs,
        torch.arange(len(lengths), device="cuda"),
        lengths,
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=replay,
    )


def _forward(runtime, layer, query, **kwargs):
    return runtime.router.forward(
        query,
        None,
        None,
        layer,
        runtime.pool,
        ForwardMode.DECODE,
        query.shape[0],
        save_kv_cache=False,
        **kwargs,
    ).view(query.shape[0], layer.tp_q_head_num, layer.v_head_dim)


def check_extend_writes(runtime):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    tables = _tables(runtime, 2)
    for i, (gid, leaf) in enumerate(runtime.router.leaves.items()):
        packing = runtime.arena.plan.group(gid).cache_blocks_per_lcm_block
        first = 1 + i * 10 * packing
        tables[gid][:, :3] = torch.arange(
            first, first + 6, device="cuda", dtype=torch.int32
        ).view(2, 3)
    prefixes = torch.tensor([31, 63], dtype=torch.int32)
    extends = torch.tensor([3, 3], dtype=torch.int32)
    # The executor normally publishes these serving limits before prefill.
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(global_server_args_dict, "chunked_prefill_size", 512)
        patch.setitem(global_server_args_dict, "mla_chunk_multiplier", 4)
        runtime.router.init_forward_metadata(
            2,
            2,
            torch.arange(2, device="cuda"),
            (prefixes + extends).cuda(),
            ForwardMode.EXTEND,
            block_tables=tables,
            extend_seq_lens=extends.cuda(),
            extend_seq_lens_cpu=extends,
            extend_prefix_lens=prefixes.cuda(),
            extend_prefix_lens_cpu=prefixes,
            extend_replay_lens_cpu=torch.zeros_like(extends),
            extend_prompt_lens_cpu=prefixes + extends,
            extend_with_prefix=True,
            query_shard=None,
            block_tables_cpu=None,
        )
    runtime.arena.buffer.fill_(0xA5)
    expected = runtime.arena.buffer.clone()
    positions = torch.tensor([31, 32, 33, 63, 64, 65], device="cuda")
    requests = torch.tensor([0, 0, 0, 1, 1, 1], device="cuda")
    for layer in runtime.layers:
        p = runtime.router.leaf_for(layer).kernel_page_size
        loc = runtime.router.write_locations(layer, ForwardMode.EXTEND)
        oracle = (
            tables[layer.group_id][requests, positions // p].long() * p + positions % p
        )
        torch.testing.assert_close(loc.long(), oracle, atol=0, rtol=0)
        rows = torch.randn((6, 1, layer.head_dim), device="cuda", dtype=torch.bfloat16)
        runtime.pool.set_mla_kv_buffer(layer, loc, rows[..., :-64], rows[..., -64:])
        _expect_latent(expected, runtime.arena.plan, layer.layer_id, oracle, rows)
    assert torch.equal(runtime.arena.buffer, expected)


def check_swa_decode(runtime, gid):
    runtime.arena.clear()
    layer = runtime.layers[LAYERS[gid][0]]
    cache = runtime.pool.get_key_buffer(layer.layer_id)
    cache[0].fill_(float("nan"))
    n = 2049
    pages = torch.randperm(math.ceil(n / 32), device="cuda", dtype=torch.int32) + 1
    pos = torch.arange(n, device="cuda")
    loc = pages[pos // 32].long() * 32 + pos % 32
    rows = torch.randn((n, 1, 1088), device="cuda", dtype=torch.bfloat16)
    runtime.pool.set_mla_kv_buffer(layer, loc, rows[..., :-64], rows[..., -64:])
    query = torch.randn(
        (4, layer.tp_q_head_num, 1088), device="cuda", dtype=torch.bfloat16
    )
    tables = _tables(runtime, 4)
    for lengths, live in (
        ((1, 31, 32, 33), 4),
        ((512, 513, 514, 545), 4),
        ((2049, 1, 1, 1), 1),
    ):
        tables[gid].zero_()
        for b, length in enumerate(lengths[:live]):
            count = math.ceil(length / 32)
            tables[gid][b, :count] = pages[:count]
            tables[gid][b, : max(0, length - 513) // 32] = 0
        seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        _refresh(runtime, tables, seq, actual_bs=live, replay=False)
        actual = _forward(runtime, layer, query)
        # Padded outputs are not consumed; the leaf seeds a dummy length of 1.
        for b, length in enumerate(lengths[:live]):
            visible = rows[max(0, length - 513) : length, 0].float()
            expected = (query[b].float() @ visible.T * layer.scaling).softmax(
                -1
            ) @ visible[:, :1024]
            torch.testing.assert_close(
                actual[b].float(), expected, atol=5e-4, rtol=5e-3
            )


def check_index_to_dsa(runtime):
    runtime.arena.clear()
    layer = runtime.layers[LAYERS["full"][0]]
    n = 4097
    pages = torch.randperm(math.ceil(n / 64), device="cuda", dtype=torch.int32) + 1
    pos = torch.arange(n, device="cuda")
    loc = pages[pos // 64].long() * 64 + pos % 64
    rows = torch.randn((n, 1, 576), device="cuda", dtype=torch.bfloat16)
    runtime.pool.set_mla_kv_buffer(layer, loc, rows[..., :-64], rows[..., -64:])
    keys = torch.randn((n, 128), device="cuda")
    scales = torch.exp2(
        torch.ceil(torch.log2(keys.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 448))
    )
    fp8 = (keys / scales).to(torch.float8_e4m3fn)
    runtime.pool.set_index_k_buffer(layer.layer_id, loc, fp8, scales)
    dequant = fp8.float() * scales
    index_query = torch.randn((4, 64, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((4, 64), device="cuda")
    query = torch.randn(
        (4, layer.tp_q_head_num, 576), device="cuda", dtype=torch.bfloat16
    )
    tables = _tables(runtime, 4)
    tables["full"][:, : len(pages)] = pages
    leaf = runtime.router.leaf_for(layer)

    def run():
        metadata = leaf.forward_decode_metadata
        selected, lens = dsa_decode_topk(
            q=index_query,
            index_k_cache=runtime.pool.get_index_k_buffer(layer.layer_id),
            weights=weights,
            seq_lens=metadata.seq_lens,
            block_table=metadata.page_table,
            page_size=64,
            topk=2048,
            softmax_scale=128**-0.5,
            q_len_per_req=1,
            batch_invariant=leaf.batch_invariant,
            slot_order=leaf.slot_order,
            topk_layout="global_slots",
            block_table_base_offsets=None,
            out=None,
            lens_out=None,
            solution="triton",
        )
        return (
            selected,
            lens,
            _forward(runtime, layer, query, topk_indices=selected, topk_lens=lens),
        )

    def verify(result, lengths, live):
        selected, lens, actual = result
        # Padding may select dummy rows, but must never select a live cache page.
        assert torch.all((selected[live:] >= -1) & (selected[live:] < 64))
        for b, length in enumerate(lengths[:live]):
            count = min(length, 2048)
            assert lens[b].item() == count
            scores = (
                (index_query[b].float() @ dequant[:length].T).relu()
                * weights[b, :, None]
            ).sum(0) * 128**-0.5
            chosen = scores.topk(count).indices
            torch.testing.assert_close(
                selected[b, :count].long().sort().values,
                loc[chosen].sort().values,
                atol=0,
                rtol=0,
            )
            assert torch.all(selected[b, count:] == -1)
            visible = rows[chosen, 0].float()
            expected = (query[b].float() @ visible.T * layer.scaling).softmax(
                -1
            ) @ visible[:, :512]
            torch.testing.assert_close(
                actual[b].float(), expected, atol=1e-3, rtol=5e-3
            )

    seq = torch.tensor([1, 65, 2047, 2048], device="cuda", dtype=torch.int32)
    _refresh(runtime, tables, seq, actual_bs=4, replay=False)
    verify(run(), seq.tolist(), 4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for lengths, live in (
        ((2049, 4097, 1, 1), 2),
        ((1, 1, 1, 1), 0),
        ((65, 2048, 2049, 4097), 4),
    ):
        seq.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        index_query.neg_()
        query.mul_(0.75)
        _refresh(runtime, tables, seq, actual_bs=live, replay=True)
        graph.replay()
        verify(captured, lengths, live)


def check_decode_graph(runtime):
    """Capture pool writes and all four attention leaves, then refresh outside."""
    runtime.arena.clear()
    layers = [runtime.layers[ids[0]] for ids in LAYERS.values()]
    tables = _tables(runtime, 4)
    lengths = torch.ones(4, device="cuda", dtype=torch.int32)
    rows, queries = {}, {}
    for layer in layers:
        rows[layer.group_id] = torch.zeros(
            (4, 1, layer.head_dim), device="cuda", dtype=torch.bfloat16
        )
        queries[layer.group_id] = torch.zeros(
            (4, layer.tp_q_head_num, layer.head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
    topk = torch.full((4, 2048), -1, device="cuda", dtype=torch.int32)
    topk_lens = torch.ones(4, device="cuda", dtype=torch.int32)

    def forward():
        outputs = {}
        for layer in layers:
            gid = layer.group_id
            loc = runtime.router.write_locations(layer, ForwardMode.DECODE)
            value = rows[gid]
            runtime.pool.set_mla_kv_buffer(
                layer, loc, value[..., :-64], value[..., -64:]
            )
            kwargs = {}
            if gid == "full":
                # This write/refresh test selects the current row, not the indexer.
                # The separate scatter->top-k->DSA check covers real selection.
                topk[:, 0].copy_(loc)
                topk_lens.copy_(
                    runtime.router.leaf_for(
                        layer
                    ).forward_decode_metadata.seq_lens.clamp(0, 1)
                )
                kwargs = dict(topk_indices=topk, topk_lens=topk_lens)
            outputs[gid] = _forward(runtime, layer, queries[gid], **kwargs)
        return outputs

    _refresh(runtime, tables, lengths, actual_bs=0, replay=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    pointers = {}
    # Cross page32, page64 and retention boundaries; change physical parents,
    # shrink the live batch, idle, then grow it without recapturing.
    for step, (length, live) in enumerate(
        ((32, 4), (33, 2), (64, 3), (65, 4), (513, 2), (514, 0), (545, 3))
    ):
        lengths.fill_(length)
        expected = runtime.arena.buffer.clone()
        for i, layer in enumerate(layers):
            gid = layer.group_id
            p = runtime.pool.get_key_buffer(layer.layer_id).shape[1]
            packing = runtime.arena.plan.group(gid).cache_blocks_per_lcm_block
            # One group owns each parent; each live request has distinct pages.
            first = 1 + ((0, 40, 120, 200)[i] + step % 2 * 4) * packing
            tables[gid].zero_()
            for b in range(live):
                first_visible = 0 if gid == "full" else max(0, length - 513)
                begin, end = first_visible // p, math.ceil(length / p)
                physical = torch.arange(end - begin, device="cuda", dtype=torch.int32)
                tables[gid][b, begin:end] = (
                    physical.flip(0) + first + b * math.ceil(545 / p)
                )
            rows[gid].fill_((i + 1) * (step + 1) / 8)
        runtime.arena.clear()
        expected.zero_()
        _refresh(runtime, tables, lengths, actual_bs=live, replay=True)
        for layer in layers:
            gid = layer.group_id
            leaf = runtime.router.leaf_for(layer)
            meta = leaf.forward_decode_metadata
            loc = runtime.router.write_locations(layer, ForwardMode.DECODE)
            addresses = (
                loc.data_ptr(),
                meta.page_table.data_ptr(),
                meta.seq_lens.data_ptr(),
            )
            assert pointers.setdefault(gid, addresses) == addresses
            assert meta.page_table[live:].count_nonzero() == 0
            assert loc[live:].count_nonzero() == 0
            p = leaf.kernel_page_size
            expected_locs = (
                tables[gid][:live, (length - 1) // p].long() * p + (length - 1) % p
            )
            torch.testing.assert_close(loc[:live].long(), expected_locs, atol=0, rtol=0)
            # Each padded lane writes the identical value to slot 0.
            unique = min(live + 1, 4)
            _expect_latent(
                expected,
                runtime.arena.plan,
                layer.layer_id,
                loc[:unique].long(),
                rows[gid][:unique],
            )
        graph.replay()
        assert torch.equal(runtime.arena.buffer, expected)
        for layer in layers:
            gid = layer.group_id
            loc = runtime.router.write_locations(layer, ForwardMode.DECODE)
            latent, rope = runtime.pool.get_mla_kv_buffer(layer, loc[:live])
            torch.testing.assert_close(
                latent, rows[gid][:live, :, :-64], atol=0, rtol=0
            )
            torch.testing.assert_close(rope, rows[gid][:live, :, -64:], atol=0, rtol=0)
            # Only the current row is nonzero; zero query gives a uniform mean.
            if gid == "full":
                expected_out = (
                    rows[gid][:live, 0, :512]
                    .unsqueeze(1)
                    .expand(-1, layer.tp_q_head_num, -1)
                )
            else:
                visible = min(length, 513)
                expected_out = (
                    (rows[gid][:live, 0, :1024].float() / visible)
                    .unsqueeze(1)
                    .expand(-1, layer.tp_q_head_num, -1)
                )
            torch.testing.assert_close(
                captured[gid][:live].float(), expected_out.float(), atol=5e-4, rtol=5e-3
            )
        # The padding contract protects live storage, not padded output values.
        eager = forward()
        for gid in captured:
            torch.testing.assert_close(
                captured[gid][:live], eager[gid][:live], atol=0, rtol=0
            )


def test_mtp_pool_masked_writes_preserve_excluded_rows(mtp_inputs):
    _, draft, _ = build_mtp_pools(mtp_inputs, device="cuda")
    draft.arena.buffer.fill_(0xA5)
    expected = draft.arena.buffer.clone()
    locations = torch.tensor([63, 64, 95, 96], dtype=torch.int64, device="cuda")
    rows = torch.randn((4, 1, 1088), dtype=torch.bfloat16, device="cuda")
    mask = torch.tensor([True, False, False, True], device="cuda")
    draft.set_mla_kv_buffer(
        SimpleNamespace(layer_id=0),
        locations,
        rows[..., :1024],
        rows[..., 1024:],
        write_mask=mask,
    )
    _expect_latent(expected, draft.arena.plan, 46, locations[mask], rows[mask])
    assert torch.equal(draft.arena.buffer, expected)


def test_mtp_draft_verify_writes_then_accepted_row_attention(mtp_inputs):
    """The Eagle step-0 window is written before metadata narrows to live rows."""
    torch.manual_seed(140)
    target, draft, router = build_mtp_pools(mtp_inputs, device="cuda")
    width = mtp_inputs["decode_input_tokens"]
    layer = PagedAttention(
        num_heads=8,
        num_kv_heads=8,
        head_dim=1088,
        v_head_dim=1024,
        scaling=256**-0.5,
        layer_id=0,
        logit_cap=0.0,
        sliding_window_size=512,
        rotary_emb=None,
        qk_norm=None,
    )
    bind_cache_groups(torch.nn.ModuleList([layer]), draft)
    leaf = router.leaf_for(layer)
    page_ids = torch.arange(
        14, 14 + 2 * leaf.max_num_pages, dtype=torch.int32, device="cuda"
    ).reshape(2, -1)
    tables = {"draft.swa": page_ids}
    reqs = torch.arange(2, device="cuda")
    pointers = None
    for start in (31, 32, 63, 64, 512, 513):
        draft.arena.buffer.zero_()
        target.get_key_buffer(0)[32].fill_(7)  # A different, target-owned parent.
        expected_bytes = draft.arena.buffer.clone()
        base = torch.tensor([start, start + 2], dtype=torch.int32, device="cuda")
        lengths = base + width
        router.refresh_decode_metadata(
            2,
            2,
            reqs,
            lengths,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=False,
        )
        locations = router.write_locations(layer, ForwardMode.DECODE)
        positions = base[:, None] + torch.arange(width, device="cuda")
        expected_locations = (
            page_ids.gather(1, positions.long() // 32) * 32 + positions % 32
        )
        torch.testing.assert_close(
            locations, expected_locations.flatten().to(locations.dtype)
        )
        assert leaf.forward_decode_metadata.q_len_per_req == 1
        current_pointers = (
            locations.data_ptr(),
            leaf.forward_decode_metadata.seq_lens.data_ptr(),
            leaf.forward_decode_metadata.page_table.data_ptr(),
        )
        pointers = current_pointers if pointers is None else pointers
        assert pointers == current_pointers
        rows = torch.randn((2 * width, 1, 1088), device="cuda", dtype=torch.bfloat16)
        draft.set_mla_kv_buffer(
            layer, locations, rows[..., :1024], rows[..., 1024:], write_mask=None
        )
        _expect_latent(expected_bytes, draft.arena.plan, 46, locations, rows)
        assert torch.equal(draft.arena.buffer, expected_bytes)
        # One rejection-heavy request and one accepting the entire window.
        accepted = base + torch.tensor([1, width], dtype=torch.int32, device="cuda")
        router.advance_draft_forward_metadata(accepted)
        assert (
            router.write_locations(layer, ForwardMode.DECODE).data_ptr()
            == locations.data_ptr()
        )
        torch.testing.assert_close(
            locations, expected_locations.flatten().to(locations.dtype)
        )
        lengths.add_(7)
        torch.testing.assert_close(leaf.forward_decode_metadata.seq_lens, accepted)
        q = torch.randn((2, 8, 1088), device="cuda", dtype=torch.bfloat16) * 0.1
        output = router.forward(
            q, None, None, layer, draft, ForwardMode.DECODE, 2, save_kv_cache=False
        )
        expected = []
        for request, end in enumerate(accepted.tolist()):
            positions = torch.arange(max(0, end - 513), end, device="cuda")
            kv = draft.get_key_buffer(0)[
                page_ids[request, positions // 32].long(), positions % 32, 0
            ].float()
            scores = q[request].float() @ kv.T * layer.scaling
            expected.append(scores.softmax(-1) @ kv[:, :1024])
        torch.testing.assert_close(
            output.reshape(2, 8, 1024).float(),
            torch.stack(expected),
            atol=1e-3,
            rtol=1e-2,
        )
        for step in range(1, width):
            frontier = accepted + step - 1
            published = router.publish_draft_step_locations(frontier, 1)
            router.advance_draft_forward_metadata(frontier + 1)
            expected_step = (
                page_ids.gather(1, frontier.long()[:, None] // 32).flatten() * 32
                + frontier % 32
            )
            torch.testing.assert_close(published, expected_step.to(published.dtype))
            assert (
                router.write_locations(layer, ForwardMode.DECODE).data_ptr()
                == published.data_ptr()
            )
    # Padding / idle use the same refreshed tables; no stale draft pages survive.
    for live in (1, 0, 2):
        router.refresh_decode_metadata(
            2,
            live,
            reqs,
            accepted,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=True,
        )
        assert leaf.forward_decode_metadata.page_table[live:].count_nonzero() == 0
        assert (
            router.write_locations(layer, ForwardMode.DECODE)[
                live * width :
            ].count_nonzero()
            == 0
        )


def test_mtp_graph_draft_step_reuses_refreshed_tables(mtp_inputs):
    _, draft, router = build_mtp_pools(mtp_inputs, device="cuda")
    layer = PagedAttention(
        num_heads=8,
        num_kv_heads=8,
        head_dim=1088,
        v_head_dim=1024,
        scaling=256**-0.5,
        layer_id=0,
        logit_cap=0.0,
        sliding_window_size=512,
        rotary_emb=None,
        qk_norm=None,
    )
    bind_cache_groups(torch.nn.ModuleList([layer]), draft)
    leaf = router.leaf_for(layer)
    reqs = torch.arange(2, device="cuda")
    frontier = torch.tensor([31, 63], dtype=torch.int32, device="cuda")
    table = torch.full((2, leaf.max_num_pages), 15, dtype=torch.int32, device="cuda")
    table[1].fill_(45)
    q = torch.randn((2, 8, 1088), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((2, 1, 1088), dtype=torch.bfloat16, device="cuda")

    def refresh(live):
        router.refresh_decode_metadata(
            2,
            live,
            reqs,
            frontier + mtp_inputs["decode_input_tokens"],
            forward_mode=ForwardMode.DECODE,
            block_tables={"draft.swa": table},
            num_extends=0,
            for_graph_replay=True,
        )

    def step():
        router.advance_draft_forward_metadata(frontier + 1)
        locs = router.publish_draft_step_locations(frontier, 1)
        draft.set_mla_kv_buffer(
            layer, locs, kv[..., :1024], kv[..., 1024:], write_mask=None
        )
        return router.forward(
            q, None, None, layer, draft, ForwardMode.DECODE, 2, save_kv_cache=False
        )

    refresh(2)
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = step()
    step_locations = router.write_locations(layer, ForwardMode.DECODE)
    addresses = (
        output.data_ptr(),
        step_locations.data_ptr(),
        leaf.forward_decode_metadata.seq_lens.data_ptr(),
    )
    for start, page, live in ((32, 16, 2), (512, 30, 1), (513, 31, 0), (63, 15, 2)):
        frontier.fill_(start)
        table[0].fill_(page)
        table[1].fill_(page + 30)
        refresh(live)
        graph.replay()
        actual = output.clone()
        locations = step_locations  # The one-token view recorded by the graph.
        assert addresses == (
            output.data_ptr(),
            locations.data_ptr(),
            leaf.forward_decode_metadata.seq_lens.data_ptr(),
        )
        assert locations[:live].tolist() == [
            (page + 30 * i) * 32 + start % 32 for i in range(live)
        ]
        assert locations[live:].count_nonzero() == 0
        torch.testing.assert_close(leaf.forward_decode_metadata.seq_lens, frontier + 1)
        # Padded outputs are ignored; their writes may race on the dummy slot.
        torch.testing.assert_close(actual[:live], step()[:live], atol=0, rtol=0)


def test_mtp_mixed_metadata_keeps_prefill_and_draft_slots_separate(
    mtp_inputs, monkeypatch
):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 512)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 4)
    _, draft, router = build_mtp_pools(mtp_inputs, device="cuda")
    leaf = router.leaves["draft.swa"]
    layer = SimpleNamespace(group_id="draft.swa", layer_id=0)
    tables = {
        "draft.swa": torch.arange(
            1, 1 + 2 * leaf.max_num_pages, dtype=torch.int32, device="cuda"
        ).reshape(2, -1)
    }
    reqs = torch.arange(2, device="cuda")
    extend = torch.tensor([2], dtype=torch.int32)
    prefix = torch.tensor([31], dtype=torch.int32)
    lengths = torch.tensor([33, 65], dtype=torch.int32, device="cuda")
    router.init_forward_metadata(
        2,
        1,
        reqs,
        lengths,
        ForwardMode.MIXED,
        block_tables=tables,
        extend_seq_lens=extend.cuda(),
        extend_seq_lens_cpu=extend,
        extend_prefix_lens=prefix.cuda(),
        extend_prefix_lens_cpu=prefix,
        extend_replay_lens_cpu=torch.zeros_like(prefix),
        extend_prompt_lens_cpu=prefix + extend,
        extend_with_prefix=True,
        query_shard=None,
        block_tables_cpu=None,
    )
    prefill = leaf.forward_prefill_metadata
    assert leaf.forward_decode_metadata is None
    extend_locations = router.write_locations(layer, ForwardMode.EXTEND).clone()
    router.refresh_decode_metadata(
        2,
        2,
        reqs,
        lengths,
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=1,
        for_graph_replay=False,
    )
    assert leaf.forward_prefill_metadata is prefill
    assert leaf.forward_decode_metadata.num_extends == 1
    decode_locations = router.write_locations(layer, ForwardMode.DECODE)
    assert decode_locations.numel() == mtp_inputs["decode_input_tokens"]
    torch.testing.assert_close(
        router.forward_write_locations(layer, ForwardMode.DECODE),
        torch.cat((extend_locations, decode_locations)),
    )
    router.advance_draft_forward_metadata(
        torch.tensor([33, 63], dtype=torch.int32, device="cuda")
    )
    assert leaf.forward_prefill_metadata is prefill
    torch.testing.assert_close(
        router.write_locations(layer, ForwardMode.EXTEND), extend_locations
    )
    torch.testing.assert_close(prefill.seq_lens, lengths[:1])


def test_production_binding_all_layers(runtime):
    check_binding(runtime)


def test_production_router_binding_all_layers(runtime):
    check_router_binding(runtime)


@pytest.mark.parametrize("count", [6, 513])
def test_pool_latent_scatter_gather_and_sentinels(runtime, count):
    check_latent_scatter(runtime, count)


def test_pool_index_planar_scatter_and_sentinels(runtime):
    check_index_scatter(runtime)


def test_pool_router_extend_boundary_writes(runtime):
    check_extend_writes(runtime)


@pytest.mark.parametrize("gid", ["swa.0", "swa.1", "swa.2"])
def test_pool_swa_decode(runtime, gid):
    check_swa_decode(runtime, gid)


def test_pool_scatter_topk_dsa_and_graph(runtime):
    check_index_to_dsa(runtime)


def test_pool_router_graph_refresh_writes_and_padding(runtime):
    check_decode_graph(runtime)
