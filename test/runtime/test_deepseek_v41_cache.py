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

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    DeepseekV41AttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import DeepseekV41Config
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_COMPRESSOR_TAIL_GROUP_ID as TAIL,
)
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_GLOBAL_R1_GROUP_ID as R1,
)
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_GLOBAL_R2_GROUP_ID as R2,
)
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_GROUP_GEOMETRY,
    V41_GROUP_PACKING,
)
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_SWA_GROUP_ID as SWA,
)
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    v41_layer_mapping,
    v41_table_widths,
)
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v41 import (
    DeepseekV41CachePool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v41 import (
    DeepseekV41Recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack

RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
OWNERS = (2, 8, 14, 20)
SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)


def _config(device):
    owners, sources = v41_layer_mapping(RATIOS, OWNERS, SOURCES, 20)
    spec = DeepseekV41Config(
        backend_name="deepseek_v41",
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=512,
        attn_tp_size=1,
        cache_layer_types=("sliding_attention",) * 40,
        sliding_window_tokens=128,
        compress_ratios=RATIOS,
        kv_owners=owners,
        index_sources=sources,
        candidate_source=20,
        index_topk=4,
        candidate_topk=2,
        candidate_block_size=8,
        max_query_tokens=514,
    )
    return AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.uint8,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        prefix_granularity=256,
        kernel_page_size=64,
        context_len=512,
        max_bs=2,
        pd_disaggregation_enabled=False,
        speculative_num_steps=0,
        speculative_num_draft_tokens=1,
        is_draft=False,
        draft_block_decode=False,
        components=(spec,),
    )


def _recipe(device):
    return DeepseekV41Recipe(
        server_args=SimpleNamespace(
            pipeline_parallel_size=1,
            chunked_prefill_size=512,
            max_num_seqs=2,
            max_total_tokens=1024,
        ),
        model_config=SimpleNamespace(
            num_attention_layers=40,
            hf_config=SimpleNamespace(
                compress_ratios=RATIOS + (0, 0, 0),
                kv_source_layers=OWNERS,
                index_source_layers=SOURCES,
                candidate_source_layer=20,
                head_dim=512,
                index_head_dim=128,
                sliding_window=128,
            ),
        ),
        attn_config=_config(device),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=256 << 20,
        decode_input_tokens=1,
        overlap_schedule_depth=1,
    )


def _layout(recipe):
    groups = recipe.groups()
    return pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )


def _backend(device, max_bs):
    recipe = _recipe(device)
    arena = CacheArena(
        _layout(recipe).bind(16),
        device,
        cache_group_specs=tuple(s for s, _ in recipe.groups()),
        token_capacity=1024,
        enable_memory_saver=False,
    )
    pool = DeepseekV41CachePool(arena, layer_num=40, rank=0, field_layer_offset=0)
    config = recipe.attn_config
    backend = DeepseekV41AttentionBackend(config, config.component(DeepseekV41Config))
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(
        max_bs, max_tokens_per_req=1, overlap_schedule_depth=1
    )
    return backend


def _tables(device):
    tables = {
        gid: torch.zeros((2, width), dtype=torch.int32, device=device)
        for gid, width in v41_table_widths(512, 0).items()
    }
    # Different parent assignments: SWA 1..8, r2 parent9, r1 parent10,
    # tails parents11..16. Groups are mutually exclusive tenants, not slices
    # that can all bind the same parent at the same time.
    for req in range(2):
        tables[SWA][req, :4] = torch.arange(1 + req * 4, 5 + req * 4, device=device)
        tables[R2][req] = torch.arange(161 + req * 4, 165 + req * 4, device=device)
        tables[R1][req] = torch.arange(541 + req * 8, 549 + req * 8, device=device)
        tables[TAIL][req, :100] = torch.arange(
            541 + req * 100, 641 + req * 100, device=device
        )
    return tables


def _extend(backend, tables, lengths, prefixes):
    device = backend.device
    counts = torch.tensor(lengths, dtype=torch.int32)
    prefix = torch.tensor(prefixes, dtype=torch.int32)
    backend.init_forward_metadata(
        len(lengths),
        len(lengths),
        torch.arange(len(lengths), device=device),
        (counts + prefix).to(device),
        ForwardMode.EXTEND,
        block_tables=tables,
        extend_seq_lens=counts.to(device),
        extend_seq_lens_cpu=counts,
        extend_prefix_lens=prefix.to(device),
        extend_prefix_lens_cpu=prefix,
        extend_with_prefix=any(prefixes),
    )
    return backend.query_metadata(ForwardMode.EXTEND)


@pytest.mark.parametrize("for_graph_replay", [False, True])
@pytest.mark.parametrize(
    "group,message", [(SWA, "SWA prefix"), (TAIL, "compressor tail")]
)
def test_refresh_rejects_missing_history_before_execution(
    for_graph_replay, group, message
):
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    tables[group][0].zero_()
    with pytest.raises(RuntimeError, match=message):
        backend.refresh_decode_metadata(
            2,
            1,
            torch.tensor([19]),
            torch.tensor([4]),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=for_graph_replay,
        )


def test_full_scan_is_bounded_by_table_and_physical_capacity():
    pool = _backend("cpu", 2).cache_pool
    config = replace(_config("cpu"), context_len=1 << 20)
    backend = DeepseekV41AttentionBackend(config, config.component(DeepseekV41Config))
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(2, max_tokens_per_req=1, overlap_schedule_depth=1)
    backend.refresh_decode_metadata(
        2,
        1,
        torch.tensor([0]),
        torch.tensor([1]),
        forward_mode=ForwardMode.DECODE,
        block_tables=_tables("cpu"),
        num_extends=0,
        for_graph_replay=False,
    )
    meta = backend.query_metadata(ForwardMode.DECODE)
    with patch("tokenspeed_kernel.ops.attention.dsv41.index_topk") as topk:
        for layer in (2, 20):
            backend.select_global(
                layer,
                torch.zeros(2, 2, 128, dtype=torch.bfloat16),
                torch.zeros(2, 2, dtype=torch.bfloat16),
                meta.positions,
                meta.request_indices,
                ForwardMode.DECODE,
                None,
            )
            assert topk.call_args.args[3].shape[1] == pool.index_k(layer).shape[0] - 1


def test_mixed_decode_rejects_missing_history():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    tables[SWA][1].zero_()
    with pytest.raises(RuntimeError, match="SWA prefix"):
        backend.init_forward_metadata(
            2,
            1,
            torch.tensor([23, 17]),
            torch.tensor([2, 9]),
            ForwardMode.MIXED,
            block_tables=tables,
            extend_seq_lens=torch.tensor([2]),
            extend_seq_lens_cpu=torch.tensor([2]),
            extend_prefix_lens=torch.tensor([0]),
            extend_prefix_lens_cpu=torch.tensor([0]),
            extend_with_prefix=False,
        )


def test_compressor_prefill_rejects_missing_history():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    tables[TAIL].zero_()
    meta = _extend(backend, tables, [1], [3])
    with pytest.raises(RuntimeError, match="compressor tail"):
        backend.compress(
            2,
            torch.zeros(1, 512),
            torch.zeros(1, 512),
            meta.positions,
            meta.request_indices,
            ForwardMode.EXTEND,
            norm_weight=None,
            norm_eps=0.0,
        )


def _decode_compute(backend, inputs, bs):
    content, scores, q, swa, iq, iw, sink = inputs
    mode = ForwardMode.DECODE
    meta = backend.query_metadata(mode)
    pos, req = meta.positions, meta.request_indices
    pooled, pair_pos, pair_req = backend.compress(
        2, content[:bs], scores[:bs], pos, req, mode, norm_weight=None, norm_eps=0.0
    )
    latent = torch.nn.functional.rms_norm(
        pooled.to(torch.bfloat16), (512,), weight=None, eps=1e-6
    )
    backend.write_global(2, latent, latent[:, :128], pair_pos, pair_req, mode)
    backend.write_global(20, swa[:bs], iq[:bs, 0], pos, req, mode)
    outputs = [pooled, pair_pos, pair_req]
    query_q, query_swa, query_iq, query_iw = q[:bs], swa[:bs], iq[:bs], iw[:bs]
    for layer in (2, 3, 20, 24, 25):
        if layer == 24:
            order = torch.arange(bs - 1, -1, -2, device=backend.device)
            pos, req = pos[order], req[order]
            query_q, query_swa = query_q[order], query_swa[order]
            query_iq, query_iw = query_iq[order], query_iw[order]
        source = layer in (2, 20, 24)
        outputs.append(
            backend.forward_v41(
                query_q,
                query_swa,
                layer_id=layer,
                positions=pos,
                request_indices=req,
                forward_mode=mode,
                index_q=query_iq if source else None,
                index_weights=query_iw if source else None,
                attn_sink=sink,
                softmax_scale=512**-0.5,
                index_process_group=None,
                swa_rope_cache=None,
            )
        )
        if layer == 20 and bs:
            candidates = backend.sparse_topk.decode.candidates
            outputs.extend((candidates.block_ids, candidates.lengths))
            # Reordered/subset consumers use absolute request/position lookup,
            # even when a formerly live row becomes graph padding on replay.
            order = torch.arange(bs - 1, -1, -2, device=backend.device)
            outputs.extend(
                backend.select_global(
                    21, None, None, pos[order], req[order], mode, None
                )
            )
    return outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shared_pool", [False, True])
def test_gpu_backend_decode_capture_replay_and_above_ladder(shared_pool):
    from tokenspeed_kernel.ops.attention import dsv41

    assert DeepseekV41AttentionBackend.cuda_graph_support.decode_graph
    assert not DeepseekV41AttentionBackend.cuda_graph_support.prefill_graph
    torch.manual_seed(42)
    backend = _backend("cuda", 5)
    backend.cache_pool.arena.buffer.zero_()
    tables = _tables("cuda")
    inputs = (
        torch.randn(5, 512, device="cuda"),
        torch.randn(5, 512, device="cuda"),
        torch.randn(5, 2, 512, device="cuda", dtype=torch.bfloat16),
        torch.randn(5, 512, device="cuda", dtype=torch.bfloat16),
        torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16),
        torch.rand(5, 2, device="cuda", dtype=torch.bfloat16),
        torch.zeros(2, device="cuda"),
    )
    meta = _extend(backend, tables, [12, 12], [0, 0])
    history = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16)
    for layer in (2, 3, 20, 24, 25):
        dsv41.cache_scatter(
            history, backend.cache_pool.swa(layer), meta.swa_write_slots, "swa"
        )
    backend.write_global(
        20,
        history,
        history[:, :128],
        meta.positions,
        meta.request_indices,
        ForwardMode.EXTEND,
    )
    pooled, pos, req = backend.compress(
        2,
        history.float(),
        torch.randn(24, 512, device="cuda"),
        meta.positions,
        meta.request_indices,
        ForwardMode.EXTEND,
        norm_weight=None,
        norm_eps=0.0,
    )
    backend.write_global(2, pooled, pooled[:, :128], pos, req, ForwardMode.EXTEND)

    captures = {}
    pool = torch.cuda.graph_pool_handle() if shared_pool else None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for bs in (2, 1):
            for _ in range(2):
                backend.init_forward_metadata_capture_cuda_graph(
                    bs,
                    torch.zeros(bs, device="cuda", dtype=torch.int64),
                    torch.ones(bs, device="cuda", dtype=torch.int32),
                    ForwardMode.DECODE,
                    block_tables=tables,
                )
                _decode_compute(backend, inputs, bs)
            # Warmup has already prepared native scheduler metadata. Actual
            # capture must obtain fresh producer state even without another
            # refresh of these identical input buffers.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = _decode_compute(backend, inputs, bs)
            captures[bs] = graph, output
    torch.cuda.current_stream().wait_stream(stream)
    arena = backend.cache_pool.arena.buffer
    pointers = {bs: backend._decode_view(bs).positions.data_ptr() for bs in (1, 2)}
    # Alternate captures with shared or private pools and intervening eager work.
    for step, (bs, actual, lengths) in enumerate(
        (
            (2, 2, [8, 5]),
            (1, 1, [9]),
            (2, 1, [10]),
            (2, 0, []),
            (2, 2, [11, 6]),
            (1, 1, [12]),
        )
    ):
        for tensor in inputs[:-1]:
            tensor.normal_()
        live_tables = {
            gid: table.flip(0) if step % 2 else table for gid, table in tables.items()
        }
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        requests = torch.arange(actual, device="cuda", dtype=torch.int64) + step * 7
        before = arena.clone()
        backend.refresh_decode_metadata(
            bs,
            actual,
            requests,
            lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=live_tables,
            num_extends=0,
            for_graph_replay=False,
        )
        expected = [tensor.clone() for tensor in _decode_compute(backend, inputs, bs)]
        expected_cache = arena.clone()
        arena.copy_(before)
        backend.refresh_decode_metadata(
            bs,
            actual,
            requests,
            lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=live_tables,
            num_extends=0,
            for_graph_replay=True,
        )
        graph, output = captures[bs]
        graph.replay()
        for got, want in zip(output, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert torch.equal(arena, expected_cache)
        assert (
            backend.query_metadata(ForwardMode.DECODE).positions.data_ptr()
            == pointers[bs]
        )
        if actual == 0:
            assert torch.equal(arena, before)
        assert not bool(backend.cache_pool.swa(2)[0].any())
        assert not bool(backend.cache_pool.global_kv(2)[0].any())
        assert not bool(backend.cache_pool.compressor_tail(2)[0].any())

    # Capacity is five, capture ladder ends at two; run the identical forward.
    wide_tables = {
        gid: torch.zeros(5, t.shape[1], device="cuda", dtype=torch.int32)
        for gid, t in tables.items()
    }
    for r in range(5):
        wide_tables[SWA][r, 0] = r + 1
        wide_tables[R1][r, 0] = 541 + r
        wide_tables[R2][r, 0] = 161 + r
        wide_tables[TAIL][r, 0] = 541 + r
    for bs in (5, 0):
        backend.refresh_decode_metadata(
            bs,
            bs,
            torch.arange(bs, device="cuda"),
            torch.ones(bs, device="cuda", dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables=wide_tables,
            num_extends=0,
            for_graph_replay=False,
        )
        output = _decode_compute(backend, inputs, bs)
        assert output[0].shape == (bs, 512)
        assert output[3].shape == (bs, 2, 512)
    meta = backend.query_metadata(ForwardMode.DECODE)
    for layer in (20, 24, 25):
        source = layer != 25
        slots, lens = backend.select_global(
            layer,
            inputs[4][:0] if source else None,
            inputs[5][:0] if source else None,
            meta.positions,
            meta.request_indices,
            ForwardMode.DECODE,
            None,
        )
        assert slots.shape == (0, backend.spec.index_topk) and lens.shape == (0,)
    torch.cuda.synchronize()


def test_recipe_exact_geometry_capacity_and_dispatch():
    recipe = _recipe("cpu")
    layout = _layout(recipe)
    recipe.check_layout(layout)
    assert layout.lcm_block_bytes == 1_382_400
    assert layout.plane_bytes == (("flatkv", 1_382_400),)
    assert dict(layout.group_packing) == V41_GROUP_PACKING
    specs = {s.group_id: s for s, _ in recipe.groups()}
    assert [specs[g].block_granularity for g in V41_GROUP_GEOMETRY] == [64, 128, 64, 2]
    assert all(s.family == "history" for s in specs.values())
    assert specs[SWA].sliding_window_tokens == 130
    assert specs[TAIL].sliding_window_tokens == 4
    payload = {
        gid: sum(f.payload_bytes for f in fields)
        for (spec, fields) in recipe.groups()
        for gid in (spec.group_id,)
    }
    assert payload == {SWA: 1_351_680, R2: 68_352, R1: 22_784, TAIL: 24_576}
    assert len(layout.fields) == 51
    assert all(f.page_stride_bytes % 256 == 0 for f in layout.fields)
    setup = recipe.setup()
    assert setup.spec.token_capacity == 1024
    assert (
        setup.spec.memory_plan.arena_bytes + setup.fixed_workspace_bytes
        <= recipe.cache_budget_bytes
    )
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import _RECIPES
    from tokenspeed.runtime.layers.attention.registry import (
        _create_attn_backend,
        _resolve_cache_family,
    )

    assert _RECIPES["deepseek_v41"] is DeepseekV41Recipe
    assert isinstance(
        _create_attn_backend(AttentionArch.MLA, recipe.attn_config),
        DeepseekV41AttentionBackend,
    )
    model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41_text"))
    assert _resolve_cache_family(None, model, recipe.attn_config) == "deepseek_v41"


def test_owner_topology_and_reject_invalid_recipes():
    owners, sources = v41_layer_mapping(RATIOS, OWNERS, SOURCES, 20)
    assert owners == (-1, -1) + (2,) * 6 + (8,) * 6 + (14,) * 6 + (20,) * 20
    assert sources[20:] == (20,) * 4 + (24,) * 4 + (28,) * 4 + (32,) * 4 + (36,) * 4
    with pytest.raises(ValueError):
        v41_layer_mapping(RATIOS, OWNERS, SOURCES, 14)
    with pytest.raises(ValueError):
        v41_layer_mapping((0, 1), (1,), (0, 1), 1)
    config = _config("cpu")
    with pytest.raises(ValueError, match="capacities"):
        DeepseekV41AttentionBackend(
            config, replace(config.component(DeepseekV41Config), index_topk=513)
        )
    recipe = _recipe("cpu")
    recipe.decode_input_tokens = 5
    with pytest.raises(NotImplementedError, match="speculation"):
        recipe.groups()
    recipe = _recipe("cpu")
    recipe.attn_config = replace(recipe.attn_config, prefix_granularity=64)
    with pytest.raises(ValueError, match="128"):
        recipe.groups()


def test_pool_strides_owner_fences_and_no_redundant_fields():
    backend = _backend("cpu", 2)
    pool = backend.cache_pool
    tracker = Mock()
    pool.layerwise_load_tracker = tracker
    for view, shape, stride, owner in (
        (pool.swa(7), (64, 528), 1_382_400, 7),
        (pool.global_kv(8), (64, 288), 69_120, 8),
        (pool.index_k(20), (64, 68), 23_040, 20),
        (pool.compressor_tail(14), (2, 2, 512), 25_600, 14),
    ):
        assert tuple(view.shape[1:]) == shape
        assert view.stride(0) * view.element_size() == stride
        assert (
            view.untyped_storage().data_ptr()
            == pool.arena.buffer.untyped_storage().data_ptr()
        )
        tracker.wait_for_layer.assert_any_call(owner)
    with pytest.raises(ValueError, match="not planned"):
        pool.global_kv(24)
    assert pool.history_group_by_layer() == {layer: SWA for layer in range(40)}


def test_refresh_pointer_stability_padding_and_mapping():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    for actual in (2, 1, 0, 2):
        backend.refresh_decode_metadata(
            2,
            actual,
            torch.tensor([19, 7]),
            torch.tensor([129, 4]),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=False,
        )
        meta = backend.query_metadata(ForwardMode.DECODE)
        pointers = [t.data_ptr() for t in meta.block_tables.values()] + [
            meta.positions.data_ptr(),
            backend.write_locations(None, ForwardMode.DECODE).data_ptr(),
        ]
        if actual == 2 and not hasattr(backend, "test_pointers"):
            backend.test_pointers = pointers
        assert pointers == backend.test_pointers
        assert meta.positions.tolist() == [128, 3][:actual] + [-1] * (2 - actual)
        assert all(not bool(t[actual:].any()) for t in meta.block_tables.values())
        slots = backend.cache_slots(
            SWA, meta.positions, meta.request_indices, ForwardMode.DECODE
        )
        assert (slots[actual:] == -1).all()
    rows = torch.tensor([[0, 63, 64, 65, -1]])
    slots = backend.global_read_slots(
        2, rows, torch.tensor([129]), torch.tensor([0]), ForwardMode.DECODE
    )
    assert slots.tolist() == [[161 * 64, 161 * 64 + 63, 162 * 64, -1, -1]]
    tables[SWA][0, 2] = 0
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([1]),
        torch.tensor([129]),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=False,
    )
    assert backend.write_locations(None, ForwardMode.DECODE).tolist() == [-1]
    assert (
        DeepseekV41AttentionBackend.init_forward_metadata_capture_cuda_graph
        is AttentionBackend.init_forward_metadata_capture_cuda_graph
    )
    backend.init_forward_metadata_capture_cuda_graph(
        1, torch.tensor([0]), torch.tensor([1]), ForwardMode.DECODE, block_tables=tables
    )
    assert backend.query_metadata(ForwardMode.DECODE).positions.tolist() == [-1]
    with pytest.raises(ValueError, match="missing cache block table"):
        backend.refresh_decode_metadata(
            1,
            1,
            torch.tensor([0]),
            torch.tensor([1]),
            forward_mode=ForwardMode.DECODE,
            block_tables={},
            num_extends=0,
            for_graph_replay=False,
        )


def test_target_runner_refresh_omits_num_extends_after_mixed_metadata():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    req_pool_indices = torch.tensor([19, 7], dtype=torch.int64, device="cpu")
    seq_lens = torch.tensor([129, 4], dtype=torch.int32, device="cpu")
    backend.refresh_decode_metadata(
        2,
        2,
        req_pool_indices,
        seq_lens,
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=1,
        for_graph_replay=False,
    )
    assert backend.query_metadata(ForwardMode.DECODE).positions.tolist() == [-1, 3]
    # Exercise the real runner call site without constructing a model or graphs.
    runner = SimpleNamespace(
        attn_backend=backend, draft_attn_backend=None, max_tokens_per_req=1
    )
    for actual_bs in (2, 1, 0):
        ForwardStepRunner._prepare_decode_metadata(
            runner,
            2,
            actual_bs,
            req_pool_indices,
            seq_lens,
            ForwardMode.DECODE,
            use_graph=False,
            block_tables=tables,
        )
        meta = backend.query_metadata(ForwardMode.DECODE)
        assert meta.num_extends == 0
        assert meta.positions.tolist() == [128, 3][:actual_bs] + [-1] * (2 - actual_bs)
        assert backend.write_locations(None, ForwardMode.DECODE).tolist() == (
            [192, 323][:actual_bs] + [-1] * (2 - actual_bs)
        )


def test_compressor_odd_chunks_arbitrary_requests_and_rejected_suffix():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    torch.manual_seed(5)
    content, scores = torch.randn(7, 512), torch.randn(7, 512)
    expected = (
        content[:6].view(3, 2, 512) * scores[:6].view(3, 2, 512).softmax(1)
    ).sum(1)
    parts = []
    for prefix, count in ((0, 3), (3, 2), (5, 2)):
        meta = _extend(backend, tables, [count], [prefix])
        pooled, pos, req = backend.compress(
            2,
            content[prefix : prefix + count],
            scores[prefix : prefix + count],
            meta.positions,
            meta.request_indices,
            ForwardMode.EXTEND,
            norm_weight=None,
            norm_eps=0.0,
        )
        assert pooled.shape == (count, 512)
        active = pos >= 0
        parts.append(pooled[active])
        assert (pos[active] % 2 == 0).all() and (req[active] == 0).all()
        assert not bool(pooled[~active].any())
        assert (req[~active] == -1).all()
    torch.testing.assert_close(torch.cat(parts), expected)
    # Position-addressed history survives an uncommitted suffix; this is not
    # a claim that the gated speculative scheduler/commit path is supported.
    meta = _extend(backend, tables, [2], [7])
    backend.compress(
        2,
        torch.randn(2, 512),
        torch.randn(2, 512),
        meta.positions,
        meta.request_indices,
        ForwardMode.EXTEND,
        norm_weight=None,
        norm_eps=0.0,
    )
    # Reject that suffix logically, then complete a different token7.
    meta = _extend(backend, tables, [1], [7])
    new_content, new_scores = torch.randn(1, 512), torch.randn(1, 512)
    pooled, _, _ = backend.compress(
        2,
        new_content,
        new_scores,
        meta.positions,
        meta.request_indices,
        ForwardMode.EXTEND,
        norm_weight=None,
        norm_eps=0.0,
    )
    want = (
        torch.stack((content[6], new_content[0]))
        * torch.stack((scores[6], new_scores[0])).softmax(0)
    ).sum(0)
    torch.testing.assert_close(pooled[0], want)
    lookup = backend._lookup_rows(
        torch.tensor([4, 2, 4]),
        torch.tensor([1, 0, 0]),
        torch.tensor([4, 4, 3]),
        torch.tensor([0, 1, 0]),
    )
    assert lookup.tolist() == [2, 0, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_quantized_joint_attention_and_reindex_reuse():
    from tokenspeed_kernel.ops.attention import dsv41

    backend = _backend("cuda", 2)
    tables = _tables("cuda")
    meta = _extend(backend, tables, [17, 3], [0, 0])
    torch.manual_seed(7)
    n = meta.positions.numel()
    main = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
    index = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
    backend.write_global(
        20, main, index, meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    q = torch.randn(n, 2, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
    iq = torch.randn(n, 2, 128, device="cuda", dtype=torch.bfloat16)
    iw = torch.rand(n, 2, device="cuda", dtype=torch.bfloat16)
    sink = torch.tensor([0.3, -0.7], device="cuda", dtype=torch.float32)
    out = backend.forward_v41(
        q,
        swa,
        layer_id=20,
        positions=meta.positions,
        request_indices=meta.request_indices,
        forward_mode=ForwardMode.EXTEND,
        index_q=iq,
        index_weights=iw,
        attn_sink=sink,
        softmax_scale=512**-0.5,
        index_process_group=None,
        swa_rope_cache=None,
    )
    record = backend.sparse_topk.prefill
    dq_swa = dsv41.cache_unpack(dsv41.cache_pack(swa, "swa", None), "swa", None)
    dq_main = dsv41.cache_unpack(dsv41.cache_pack(main, "global", None), "global", None)
    expected = []
    for i in range(n):
        p, r = int(meta.positions[i]), int(meta.request_indices[i])
        req_rows = torch.where(meta.request_indices == r)[0]
        sw = dq_swa[req_rows[max(0, p - 127) : p + 1]]
        ids = record.logical_rows[i, : int(record.lengths[i])].long()
        kv = torch.cat((sw, dq_main[req_rows[ids]]))
        logits = q[i].float() @ kv.float().T * 512**-0.5
        probs = torch.cat((logits, sink[:, None]), 1).softmax(1)[:, :-1]
        expected.append((probs @ kv.float()).to(torch.bfloat16))
    torch.testing.assert_close(out, torch.stack(expected), rtol=0.02, atol=0.02)
    candidates = record.candidates
    assert candidates is not None
    # Different query order/subset, then another reindex: candidates must not
    # be replaced by layer24's TopK or interpreted as tensor row numbers.
    pick = torch.tensor([18, 16, 4], device="cuda")
    positions, requests = meta.positions[pick], meta.request_indices[pick]
    reuse, _ = backend.select_global(
        21, None, None, positions, requests, ForwardMode.EXTEND, None
    )
    want = backend.global_read_slots(
        20, record.logical_rows[pick], positions, requests, ForwardMode.EXTEND
    )
    torch.testing.assert_close(reuse, want)
    for layer in (24, 28):
        slots, lengths = backend.select_global(
            layer, iq[pick], iw[pick], positions, requests, ForwardMode.EXTEND, None
        )
        assert backend.sparse_topk.prefill.candidates is candidates
        assert (lengths > 0).all() and (slots[:, 0] >= 0).all()
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], device="cuda"),
        torch.tensor([18], device="cuda"),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=False,
    )
    assert backend.sparse_topk.prefill is None and backend.sparse_topk.decode is None
    torch.cuda.synchronize()


def test_mixed_metadata_query_windows_and_capacity():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    backend.init_forward_metadata(
        2,
        1,
        torch.tensor([23, 17]),
        torch.tensor([5, 9]),
        ForwardMode.MIXED,
        block_tables=tables,
        extend_seq_lens=torch.tensor([3]),
        extend_seq_lens_cpu=torch.tensor([3]),
        extend_prefix_lens=torch.tensor([2]),
        extend_prefix_lens_cpu=torch.tensor([2]),
        extend_with_prefix=True,
    )
    assert backend.query_metadata(ForwardMode.MIXED).positions.tolist() == [2, 3, 4, 8]
    assert backend.query_metadata(ForwardMode.EXTEND).request_indices.tolist() == [
        0,
        0,
        0,
    ]
    assert backend.query_metadata(ForwardMode.DECODE).request_indices.tolist() == [1]
    assert backend.query_metadata(ForwardMode.DECODE).request_pool_indices.tolist() == [
        23,
        17,
    ]
    assert backend.write_locations(None, ForwardMode.DECODE).tolist() == [5 * 64 + 8]
    with pytest.raises(ValueError, match="capacity"):
        backend.refresh_decode_metadata(
            3,
            0,
            torch.empty(0),
            torch.empty(0),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            for_graph_replay=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_ratio2_prefill_to_decode_including_empty_compressor_step():
    backend = _backend("cuda", 2)
    tables = _tables("cuda")
    torch.manual_seed(11)
    content = torch.randn(10, 512, device="cuda", dtype=torch.float32)
    scores = torch.randn_like(content)
    q = torch.randn(10, 2, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(10, 512, device="cuda", dtype=torch.bfloat16)
    iq = torch.randn(10, 2, 128, device="cuda", dtype=torch.bfloat16)
    iw = torch.rand(10, 2, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)

    def run(prefix, count, mode):
        if mode.is_decode():
            backend.refresh_decode_metadata(
                1,
                1,
                torch.tensor([0], device="cuda"),
                torch.tensor([prefix + 1], device="cuda"),
                forward_mode=mode,
                block_tables=tables,
                num_extends=0,
                for_graph_replay=False,
            )
            meta = backend.query_metadata(mode)
        else:
            meta = _extend(backend, tables, [count], [prefix])
        stop = prefix + count
        pooled, pos, req = backend.compress(
            2,
            content[prefix:stop],
            scores[prefix:stop],
            meta.positions,
            meta.request_indices,
            mode,
            norm_weight=None,
            norm_eps=0.0,
        )
        latent = torch.nn.functional.rms_norm(
            pooled.to(torch.bfloat16), (512,), weight=None, eps=1e-6
        )
        backend.write_global(2, latent, latent[:, :128].contiguous(), pos, req, mode)
        return backend.forward_v41(
            q[prefix:stop],
            swa[prefix:stop],
            layer_id=2,
            positions=meta.positions,
            request_indices=meta.request_indices,
            forward_mode=mode,
            index_q=iq[prefix:stop],
            index_weights=iw[prefix:stop],
            attn_sink=sink,
            softmax_scale=512**-0.5,
            index_process_group=None,
            swa_rope_cache=None,
        )

    full = run(0, 10, ForwardMode.EXTEND)
    backend.cache_pool.arena.buffer.zero_()
    parts = [run(0, 3, ForwardMode.EXTEND), run(3, 4, ForwardMode.EXTEND)]
    parts.extend(run(p, 1, ForwardMode.DECODE) for p in (7, 8, 9))
    torch.testing.assert_close(torch.cat(parts), full, rtol=0, atol=0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_swa_prefill_cross_chunk_matches_full_and_null_is_untouched():
    backend = _backend("cuda", 2)
    tables = _tables("cuda")
    torch.manual_seed(8)
    q = torch.randn(140, 2, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(140, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)
    meta = _extend(backend, tables, [140], [0])
    full = backend.forward_v41(
        q,
        swa,
        layer_id=0,
        positions=meta.positions,
        request_indices=meta.request_indices,
        forward_mode=ForwardMode.EXTEND,
        index_q=None,
        index_weights=None,
        attn_sink=sink,
        softmax_scale=512**-0.5,
        index_process_group=None,
        swa_rope_cache=None,
    )
    backend.cache_pool.arena.buffer.zero_()
    chunks = []
    for prefix, count in ((0, 3), (3, 64), (67, 73)):
        meta = _extend(backend, tables, [count], [prefix])
        chunks.append(
            backend.forward_v41(
                q[prefix : prefix + count],
                swa[prefix : prefix + count],
                layer_id=0,
                positions=meta.positions,
                request_indices=meta.request_indices,
                forward_mode=ForwardMode.EXTEND,
                index_q=None,
                index_weights=None,
                attn_sink=sink,
                softmax_scale=512**-0.5,
                index_process_group=None,
                swa_rope_cache=None,
            )
        )
    torch.testing.assert_close(torch.cat(chunks), full, rtol=0, atol=0)
    assert not bool(backend.cache_pool.swa(0)[0].any())
    torch.cuda.synchronize()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefill_plan_reuses_addresses_and_refresh_rechecks_pages(device):
    backend = _backend(device, 2)
    tables = _tables(device)
    meta = _extend(backend, tables, [3], [128])
    first = backend._swa_query_plan(
        meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    assert (
        backend._swa_query_plan(
            meta.positions, meta.request_indices, ForwardMode.EXTEND
        )
        is first
    )
    assert first.requests[0].prefix_slots.numel() == 127
    old_slots = first.requests[0].prefix_slots.clone()
    tables[SWA][0, :2] = torch.tensor([3, 4], device=device)
    meta = _extend(backend, tables, [3], [128])
    second = backend._swa_query_plan(
        meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    assert second is not first
    assert not torch.equal(old_slots, second.requests[0].prefix_slots)
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], device=device),
        torch.tensor([131], device=device),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        for_graph_replay=False,
    )
    assert not backend._swa_plans


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefill_plan_distinguishes_reordered_subset_and_checks_dependencies(device):
    backend = _backend(device, 2)
    tables = _tables(device)
    meta = _extend(backend, tables, [4], [128])
    full = backend._swa_query_plan(
        meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    pick = torch.tensor([3, 1], device=device)
    positions, requests = meta.positions[pick], meta.request_indices[pick]
    subset = backend._swa_query_plan(positions, requests, ForwardMode.EXTEND)
    assert subset is not full
    assert backend._swa_query_plan(positions, requests, ForwardMode.EXTEND) is subset
    request = subset.requests[0]
    # Tables initially map logical rows P to physical slots P+64.
    workspace_positions = torch.cat(
        (request.prefix_slots - 64, positions[request.rows])
    )
    wanted = positions[:, None] - torch.arange(127, -1, -1, device=device)
    actual = workspace_positions[request.swa_indices.long()]
    torch.testing.assert_close(actual, wanted)
    # A missing required historical row must fail, even if a same-sized window
    # was previously cached. Failed validation must not publish a plan.
    backend._swa_plans.clear()
    tables[SWA][0, 0] = 0
    with pytest.raises(RuntimeError, match="SWA prefix is missing"):
        backend._swa_query_plan(positions, requests, ForwardMode.EXTEND)
    assert not backend._swa_plans
    tables[SWA][0, 0] = 1
    assert backend._swa_query_plan(positions, requests, ForwardMode.EXTEND).requests


@pytest.mark.parametrize("position", [0, 126, 127, 128, None])
def test_native_decode_receives_compact_window_and_real_lengths(monkeypatch, position):
    from tokenspeed_kernel.ops.attention import dsv41

    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    backend.refresh_decode_metadata(
        1,
        0 if position is None else 1,
        torch.tensor([0]),
        torch.tensor([0 if position is None else position + 1]),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        for_graph_replay=False,
    )
    monkeypatch.setattr(dsv41, "cache_scatter", lambda *args: None)
    monkeypatch.setattr(dsv41, "new_attention_schedule", object)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    captured = []

    def selected(*args):
        captured.append(args)
        return torch.zeros_like(args[0])

    monkeypatch.setattr(dsv41, "selected_attention", selected)
    meta = backend.forward_decode_metadata
    backend.forward_v41(
        torch.zeros(1, 2, 512, dtype=torch.bfloat16),
        torch.zeros(1, 512, dtype=torch.bfloat16),
        layer_id=0,
        positions=meta.positions,
        request_indices=meta.request_indices,
        forward_mode=ForwardMode.DECODE,
        index_q=None,
        index_weights=None,
        attn_sink=torch.zeros(2),
        softmax_scale=512**-0.5,
        index_process_group=None,
        swa_rope_cache=None,
    )
    slots, lengths = captured[0][2:4]
    count = 0 if position is None else min(position + 1, 128)
    assert lengths.tolist() == [count]
    assert (slots[:, count:] == -1).all()
    if count:
        expected = torch.arange(position - count + 1, position + 1) + 64
        torch.testing.assert_close(slots[0, :count], expected.int())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("count", [0, 17, 385])
def test_gpu_compressor_pool_preserves_fp32_math_and_dynamic_graph_inputs(count):
    from tokenspeed_kernel.ops.attention import dsv41

    torch.manual_seed(112)
    fused = torch.randn(count, 1024, device="cuda")
    content, scores = fused.split(512, dim=-1)
    storage = torch.randn(8, 3, 2, 512, device="cuda")
    tail = storage[:, :2]
    previous = torch.arange(count, device="cuda") - 1
    previous[::3] = -1
    slots = torch.randint(2, 16, (count,), device="cuda")
    active = torch.rand(count, device="cuda") > 0.2
    previous[~active], slots[~active] = -1, -1

    def reference():
        if not count:
            return torch.empty_like(content)
        missing = previous < 0
        history = tail[slots.clamp_min(1) // 2, slots.clamp_min(1) % 2]
        old = content[previous.clamp_min(0)]
        gates = scores[previous.clamp_min(0)]
        old = torch.where(missing[:, None], history[:, 0], old)
        gates = torch.where(missing[:, None], history[:, 1], gates)
        weights = torch.stack((gates, scores), dim=1).softmax(1)
        return (weights[:, 0] * old + weights[:, 1] * content).masked_fill(
            ~active[:, None], 0
        )

    before = storage.clone()
    out = dsv41.compressor_pool(
        content,
        scores,
        previous,
        tail,
        slots,
        active,
        None,
        norm_weight=None,
        norm_eps=0.0,
    )
    torch.testing.assert_close(out, reference(), rtol=0, atol=0)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)
    if count == 17:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dsv41.compressor_pool(
                content,
                scores,
                previous,
                tail,
                slots,
                active,
                out,
                norm_weight=None,
                norm_eps=0.0,
            )
        for _ in range(3):
            fused.mul_(0.7)
            previous.copy_(previous.roll(1))
            active.logical_not_()
            slots.fill_(3)
            graph.replay()
            torch.testing.assert_close(out, reference(), rtol=0, atol=0)
