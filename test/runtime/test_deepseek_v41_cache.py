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
from unittest.mock import Mock

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


def _backend(device):
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
    backend.init_cuda_graph_state(2, max_tokens_per_req=1, overlap_schedule_depth=1)
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
    backend = _backend("cpu")
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
    backend = _backend("cpu")
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
    backend = _backend("cpu")
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
    backend = _backend("cpu")
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
        )
        parts.append(pooled)
        assert (pos % 2 == 0).all() and (req == 0).all()
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

    backend = _backend("cuda")
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
    backend = _backend("cpu")
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
    backend = _backend("cuda")
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
        )

    full = run(0, 10, ForwardMode.EXTEND)
    backend.cache_pool.arena.buffer.zero_()
    parts = [run(0, 3, ForwardMode.EXTEND), run(3, 4, ForwardMode.EXTEND)]
    parts.extend(run(p, 1, ForwardMode.DECODE) for p in (7, 8, 9))
    torch.testing.assert_close(torch.cat(parts), full, rtol=0, atol=0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_swa_prefill_cross_chunk_matches_full_and_null_is_untouched():
    backend = _backend("cuda")
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
            )
        )
    torch.testing.assert_close(torch.cat(chunks), full, rtol=0, atol=0)
    assert not bool(backend.cache_pool.swa(0)[0].any())
    torch.cuda.synchronize()
