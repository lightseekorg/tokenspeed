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
    V41CompressorPlan,
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
    return _verify_backend(device, max_bs, 1)


def _pool(recipe, device):
    arena = CacheArena(
        _layout(recipe).bind(16),
        device,
        cache_group_specs=tuple(s for s, _ in recipe.groups()),
        token_capacity=1024,
        enable_memory_saver=False,
    )
    return DeepseekV41CachePool(arena, layer_num=40, rank=0, field_layer_offset=0)


def _verify_backend(device, max_bs, verify_width):
    recipe = _recipe(device)
    recipe.decode_input_tokens = verify_width
    recipe.attn_config = replace(
        recipe.attn_config,
        max_bs=max_bs,
        speculative_num_draft_tokens=verify_width,
        components=(
            replace(
                recipe.attn_config.component(DeepseekV41Config),
                max_query_tokens=max(
                    recipe.server_args.chunked_prefill_size, max_bs * verify_width
                ),
            ),
        ),
    )
    pool = _pool(recipe, device)
    config = recipe.attn_config
    backend = DeepseekV41AttentionBackend(config, config.component(DeepseekV41Config))
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(
        max_bs, max_tokens_per_req=verify_width, overlap_schedule_depth=1
    )
    return backend


def test_rebinding_cache_pool_drops_pool_derived_state():
    backend = _backend("cpu", 2)
    old_buffers = backend._decode_buffers
    backend.forward_metadata = backend._decode_view(1)
    backend.forward_prefill_metadata = backend.forward_metadata
    backend.forward_decode_metadata = backend.forward_metadata
    backend._swa_plans[ForwardMode.DECODE] = object()
    backend._prefill_spans = ((0, 0, 0, 1),)
    backend._decode_schedule_keepalive.append(torch.empty(1))
    backend._prepared_selections[()] = ()

    new_pool = _pool(_recipe("cpu"), "cpu")
    backend.set_cache_pool(new_pool)

    assert backend.cache_pool is new_pool
    assert backend.forward_metadata is None
    assert backend.forward_prefill_metadata is None
    assert backend.forward_decode_metadata is None
    assert not backend._decode_views_by_bs
    assert backend._decode_buffers is None
    assert backend._decode_history_status is None
    assert backend._max_decode_bs == 0
    assert not backend._swa_plans
    assert not backend._prefill_spans
    assert not backend._decode_schedule_keepalive
    assert not backend._prepared_selections
    backend.init_cuda_graph_state(2, max_tokens_per_req=1, overlap_schedule_depth=1)
    assert backend._decode_buffers is not old_buffers


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


@pytest.mark.parametrize("verify_width", [1, 3, 5, 6])
@pytest.mark.parametrize("for_graph_replay", [False, True])
def test_packed_refresh_positions_padding_and_token_count(
    verify_width, for_graph_replay
):
    backend = _verify_backend("cpu", 3, verify_width)
    tables = _tables("cpu")
    backend.init_forward_metadata_capture_cuda_graph(
        3,
        torch.zeros(3, dtype=torch.int64),
        torch.ones(3, dtype=torch.int32),
        ForwardMode.DECODE,
        block_tables=tables,
        num_tokens=3 * verify_width,
    )
    meta = backend.query_metadata(ForwardMode.DECODE)
    tensors = list(meta.block_tables.values()) + [
        meta.positions,
        meta.request_indices,
        meta.request_pool_indices,
        meta.seq_lens,
        meta.swa_write_slots,
        meta.swa_read_slots,
        meta.swa_read_lens,
        *meta.compressor,
    ]
    assert backend._decode_history_status.numel() == 3 * verify_width
    assert all(tensor.shape == (3 * verify_width,) for tensor in meta.compressor)
    pointers = [tensor.data_ptr() for tensor in tensors]
    runner = SimpleNamespace(
        attn_backend=backend, draft_attn_backend=None, max_tokens_per_req=verify_width
    )
    for actual in (2, 1, 0, 2):
        ForwardStepRunner._prepare_decode_metadata(
            runner,
            3,
            actual,
            torch.tensor([19, 7]),
            torch.tensor([129 + verify_width, 3 + verify_width]),
            ForwardMode.DECODE,
            use_graph=for_graph_replay,
            block_tables=tables,
        )
        assert backend.query_metadata(ForwardMode.DECODE) is meta
        assert [tensor.data_ptr() for tensor in tensors] == pointers
        expected = list(range(129, 129 + verify_width)) + list(
            range(3, 3 + verify_width)
        )
        assert meta.positions.tolist() == expected[: actual * verify_width] + [-1] * (
            (3 - actual) * verify_width
        )
        assert meta.request_indices.tolist() == [
            r for r in range(actual) for _ in range(verify_width)
        ] + [-1] * ((3 - actual) * verify_width)
        assert meta.seq_lens.shape == meta.request_pool_indices.shape == (3,)
        assert all(
            table.shape[0] == 3 and not table[actual:].any()
            for table in meta.block_tables.values()
        )
        assert (meta.swa_write_slots[actual * verify_width :] == -1).all()
        assert not meta.compressor.active[actual * verify_width :].any()
        for tensor in meta.compressor[1:]:
            assert (tensor[actual * verify_width :] == -1).all()
        assert meta.swa_read_slots.shape == (3 * verify_width, 128)
        assert meta.swa_read_lens.shape == (3 * verify_width,)
        assert meta.swa_read_slots.dtype == meta.swa_read_lens.dtype == torch.int32
        assert (meta.swa_read_slots[actual * verify_width :] == -1).all()
        torch.testing.assert_close(
            meta.swa_read_lens, (meta.positions + 1).clamp(0, 128).int()
        )
        torch.testing.assert_close(
            meta.swa_write_slots,
            backend.cache_slots(
                SWA, meta.positions, meta.request_indices, ForwardMode.DECODE
            ),
        )
    with pytest.raises(ValueError, match="num_tokens"):
        backend.refresh_decode_metadata(
            3,
            0,
            torch.empty(0),
            torch.empty(0),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            num_tokens=3 * verify_width - 1,
            for_graph_replay=for_graph_replay,
        )
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([7]),
        torch.tensor([0]),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        num_tokens=verify_width,
        for_graph_replay=for_graph_replay,
    )
    meta = backend.query_metadata(ForwardMode.DECODE)
    assert meta.positions.tolist() == list(range(verify_width))
    assert meta.seq_lens.tolist() == [verify_width]


@pytest.mark.parametrize("verify_width", [1, 3, 5])
def test_packed_mixed_metadata_and_count_validation(verify_width):
    backend = _verify_backend("cpu", 2, verify_width)
    tables = _tables("cpu")
    kwargs = dict(
        block_tables=tables,
        extend_seq_lens=torch.tensor([3]),
        extend_seq_lens_cpu=torch.tensor([3]),
        extend_prefix_lens=torch.tensor([2]),
        extend_prefix_lens_cpu=torch.tensor([2]),
        extend_with_prefix=True,
        num_tokens=3 + verify_width,
    )
    backend.init_forward_metadata(
        2,
        1,
        torch.tensor([23, 17]),
        torch.tensor([5, 129 + verify_width]),
        ForwardMode.MIXED,
        **kwargs,
    )
    prefill = backend.query_metadata(ForwardMode.EXTEND)
    decode = backend.query_metadata(ForwardMode.DECODE)
    assert prefill.positions.tolist() == [2, 3, 4]
    assert decode.positions.tolist() == list(range(129, 129 + verify_width))
    assert decode.request_indices.tolist() == [1] * verify_width
    assert (
        backend.query_metadata(ForwardMode.MIXED).positions.numel() == 3 + verify_width
    )
    assert backend.write_locations(None, ForwardMode.DECODE).shape == (verify_width,)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([23, 17]),
        torch.tensor([5, 129 + verify_width]),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=1,
        num_tokens=2 * verify_width,
        for_graph_replay=False,
    )
    assert backend.query_metadata(ForwardMode.EXTEND) is prefill
    assert backend.query_metadata(ForwardMode.DECODE).positions.tolist() == [
        -1
    ] * verify_width + list(range(129, 129 + verify_width))
    kwargs["num_tokens"] += 1
    with pytest.raises(ValueError, match="num_tokens"):
        backend.init_forward_metadata(
            2,
            1,
            torch.tensor([23, 17]),
            torch.tensor([5, 129 + verify_width]),
            ForwardMode.MIXED,
            **kwargs,
        )


@pytest.mark.parametrize("for_graph_replay", [False, True])
@pytest.mark.parametrize(
    "group,start,column,message",
    [(SWA, 190, 0, "SWA prefix"), (TAIL, 3, 1, "compressor tail")],
)
def test_packed_history_checks_each_window_start(
    for_graph_replay, group, start, column, message
):
    width = 5
    backend = _verify_backend("cpu", 2, width)
    tables = _tables("cpu")
    # Request 0 has no external tail; missing internal pair rows must not be
    # treated as old history. Request 1 exercises the first query, not its last.
    tables[TAIL][0].zero_()
    tables[group][1, column] = 0
    with pytest.raises(RuntimeError, match=message):
        backend.refresh_decode_metadata(
            2,
            2,
            torch.tensor([19, 7]),
            torch.tensor([width, start + width]),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            num_tokens=2 * width,
            for_graph_replay=for_graph_replay,
        )
    tables[group][1, column] = _tables("cpu")[group][1, column]
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([19, 7]),
        torch.tensor([width, start + width]),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        num_tokens=2 * width,
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
    with pytest.raises(RuntimeError, match="compressor tail"):
        _extend(backend, tables, [1], [3])


def _decode_compute(backend, inputs, bs):
    content, scores, q, swa, iq, iw, sink = inputs
    mode = ForwardMode.DECODE
    meta = backend.query_metadata(mode)
    pos, req = meta.positions, meta.request_indices
    pooled, pair_pos, pair_req = backend.compress(
        2, content[:bs], scores[:bs], mode, norm_weight=None, norm_eps=0.0
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
@pytest.mark.parametrize("verify_width", [1, 3, 5])
def test_gpu_backend_decode_capture_replay_and_above_ladder(shared_pool, verify_width):
    from tokenspeed_kernel.ops.attention import dsv41

    assert DeepseekV41AttentionBackend.cuda_graph_support.decode_graph
    assert not DeepseekV41AttentionBackend.cuda_graph_support.prefill_graph
    torch.manual_seed(42)
    backend = _verify_backend("cuda", 5, verify_width)
    backend.cache_pool.arena.buffer.zero_()
    tables = _tables("cuda")
    capacity = 5 * verify_width
    inputs = (
        torch.randn(capacity, 512, device="cuda"),
        torch.randn(capacity, 512, device="cuda"),
        torch.randn(capacity, 2, 512, device="cuda", dtype=torch.bfloat16),
        torch.randn(capacity, 512, device="cuda", dtype=torch.bfloat16),
        torch.randn(capacity, 2, 128, device="cuda", dtype=torch.bfloat16),
        torch.rand(capacity, 2, device="cuda", dtype=torch.bfloat16),
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
                    num_tokens=bs * verify_width,
                )
                _decode_compute(backend, inputs, bs * verify_width)
            # Warmup has already prepared native scheduler metadata. Actual
            # capture must obtain fresh producer state even without another
            # refresh of these identical input buffers.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = _decode_compute(backend, inputs, bs * verify_width)
            captures[bs] = graph, output
    torch.cuda.current_stream().wait_stream(stream)
    arena = backend.cache_pool.arena.buffer
    pointers = {
        bs: tuple(
            t.data_ptr()
            for t in (
                backend._decode_view(bs).positions,
                *backend._decode_view(bs).compressor,
            )
        )
        for bs in (1, 2)
    }
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
            num_tokens=bs * verify_width,
            for_graph_replay=False,
        )
        expected = [
            tensor.clone()
            for tensor in _decode_compute(backend, inputs, bs * verify_width)
        ]
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
            num_tokens=bs * verify_width,
            for_graph_replay=True,
        )
        graph, output = captures[bs]
        graph.replay()
        for got, want in zip(output, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert torch.equal(arena, expected_cache)
        assert (
            tuple(
                t.data_ptr()
                for t in (
                    backend.query_metadata(ForwardMode.DECODE).positions,
                    *backend.query_metadata(ForwardMode.DECODE).compressor,
                )
            )
            == pointers[bs]
        )
        meta = backend.query_metadata(ForwardMode.DECODE)
        for tensor in (meta.positions, meta.request_indices, meta.swa_write_slots):
            assert (tensor[actual * verify_width :] == -1).all()
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
        tail_pages = (verify_width + 1) // 2
        wide_tables[TAIL][r, :tail_pages] = torch.arange(
            541 + r * tail_pages, 541 + (r + 1) * tail_pages, device="cuda"
        )
    for bs in (5, 0):
        backend.refresh_decode_metadata(
            bs,
            bs,
            torch.arange(bs, device="cuda"),
            torch.ones(bs, device="cuda", dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables=wide_tables,
            num_extends=0,
            num_tokens=bs * verify_width,
            for_graph_replay=False,
        )
        output = _decode_compute(backend, inputs, bs * verify_width)
        assert output[0].shape == (bs * verify_width, 512)
        assert output[3].shape == (bs * verify_width, 2, 512)
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


@pytest.mark.parametrize("verify_width", [1, 3, 6, 8])
@pytest.mark.parametrize("overlap_depth", [0, 1])
def test_packed_config_and_recipe_capacity(verify_width, overlap_depth):
    recipe = _recipe("cpu")
    hf = recipe.model_config.hf_config
    hf.candidate_topk_blocks, hf.candidate_block_size, hf.index_topk = 2, 8, 4
    args = SimpleNamespace(
        device="cpu",
        attn_tp_size=1,
        data_parallel_size=2,
        mapping=SimpleNamespace(
            attn=SimpleNamespace(
                tp_size=1, dp_size=2, dcp_size=1, dcp_rank=0, dcp_group=(0,)
            )
        ),
        prefix_granularity=128,
        spec_context_pad=2 * verify_width,
        max_num_seqs=4,
        chunked_prefill_size=3,
        max_total_tokens=1024,
        kv_cache_quant_method="none",
        disaggregation_mode="null",
        pipeline_parallel_size=1,
        speculative_algorithm="DSPARK" if verify_width > 1 else None,
        speculative_num_draft_tokens=verify_width,
        speculative_num_steps=verify_width - 1,
    )
    model = SimpleNamespace(
        hf_config=hf,
        num_attention_layers=40,
        num_attention_heads=2,
        dtype=torch.bfloat16,
        context_len=512,
    )
    config = DeepseekV41Config.generate(args, model, False)
    assert config.speculative_num_draft_tokens == verify_width
    assert config.max_bs == 2
    assert config.context_len == 512 + args.spec_context_pad
    queries = max(3, 2 * verify_width)
    assert config.component(DeepseekV41Config).max_query_tokens == queries
    recipe.server_args, recipe.attn_config = args, config
    recipe.decode_input_tokens, recipe.overlap_schedule_depth = (
        verify_width,
        overlap_depth,
    )
    layout = _layout(recipe)
    assert layout.lcm_block_bytes == 1_382_400 and len(layout.fields) == 51
    specs = {spec.group_id: spec for spec, _ in recipe.groups()}
    horizon = (1 + overlap_depth) * verify_width
    assert specs[SWA].sliding_window_tokens == 128 + horizon
    assert specs[TAIL].sliding_window_tokens == 2 + horizon
    tables = (
        4 * config.max_bs * sum(v41_table_widths(config.context_len, horizon).values())
    )
    # These small shapes fit one query tile. Portable TopK scratch dominates
    # with 576 score columns and block_k=128; no fixed 64-MiB fallback remains.
    assert recipe.workspace_bytes() == (
        2 * (tables + config.max_bs * 32 + queries * (64 + 128 * 4 + 4) + 4)
        + queries * ((2048 + 2 * 512 + 128) * 8 + 528)
        + queries * 512 * 4
        + 2 * queries * 64 * 512 * 2
        + queries * 2 * 512 * 2
        + (config.context_len + queries + config.max_bs * 128) * 512 * 2
        + queries * (512 * 2 + (128 + 512) * 4)
        + queries * 32 * (128 * 2 + 68)
        + queries * 24 * (512 + 128) * 12
        + 576 * 68
    )
    with pytest.raises(NotImplementedError, match="target attention"):
        DeepseekV41Config.generate(args, model, True)
    args.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError, match="PP=1"):
        recipe.groups()
    args.pipeline_parallel_size = 1
    recipe.attn_config = replace(config, pd_disaggregation_enabled=True)
    with pytest.raises(NotImplementedError, match="PD"):
        recipe.groups()


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
        _resolve_attn_side,
        _resolve_cache_family,
    )

    assert _RECIPES["deepseek_v41"] is DeepseekV41Recipe
    assert isinstance(
        _create_attn_backend(AttentionArch.MLA, recipe.attn_config),
        DeepseekV41AttentionBackend,
    )
    model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41_text"))
    profile = _resolve_attn_side(model, requested_backend=None)
    assert _resolve_cache_family(profile, recipe.attn_config) == "deepseek_v41"


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
    recipe.draft_attn_config = recipe.attn_config
    recipe.draft_model_config = SimpleNamespace(num_attention_layers=3)
    with pytest.raises(NotImplementedError, match="target layers only"):
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


@pytest.mark.parametrize("prefix", [127, 128])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4, 5])
def test_packed_compressor_rejected_suffix_every_acceptance(prefix, accepted):
    width = 5
    backend = _verify_backend("cpu", 1, width)
    tables = _tables("cpu")
    torch.manual_seed(41)
    content, scores = torch.randn(2, prefix + width, 512)
    new_content, new_scores = torch.randn(2, width, 512)
    meta = _extend(backend, tables, [prefix], [0])
    backend.compress(
        2,
        content[:prefix],
        scores[:prefix],
        ForwardMode.EXTEND,
        norm_weight=None,
        norm_eps=0.0,
    )
    for start, values, gates in (
        (prefix, content[prefix:], scores[prefix:]),
        (prefix + accepted, new_content, new_scores),
    ):
        backend.refresh_decode_metadata(
            1,
            1,
            torch.tensor([19]),
            torch.tensor([start + width]),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=0,
            num_tokens=width,
            for_graph_replay=False,
        )
        meta = backend.query_metadata(ForwardMode.DECODE)
        pooled, positions, _ = backend.compress(
            2,
            values,
            gates,
            ForwardMode.DECODE,
            norm_weight=None,
            norm_eps=0.0,
        )
    # The rejected branch is never restored or zeroed. New odd positions pair
    # with either retained accepted inputs or an input in the replacement window.
    full_content = torch.cat((content[: prefix + accepted], new_content))
    full_scores = torch.cat((scores[: prefix + accepted], new_scores))
    live = positions >= 0
    pair = positions[live, None] + torch.arange(2)
    expected = (full_content[pair] * full_scores[pair].softmax(1)).sum(1)
    torch.testing.assert_close(pooled[live], expected, rtol=0, atol=0)
    assert not pooled[~live].any()
    assert not backend.cache_pool.compressor_tail(2)[0].any()


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compressor_plan_shared_owners_and_reused_tail_page(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed_kernel.ops.attention import dsv41

    backend = _backend(device, 2)
    tables = _tables(device)
    # The first completed pair needs token2 before token4 reuses its tail slot.
    tables[TAIL][0, 2] = tables[TAIL][0, 1]
    calls = []
    prepare = dsv41.compressor_metadata

    def counted(*args):
        calls.append(args)
        return prepare(*args)

    monkeypatch.setattr(dsv41, "compressor_metadata", counted)
    meta = _extend(backend, tables, [4, 3], [3, 0])
    assert len(calls) == 1
    assert meta.compressor.previous.tolist() == [-1, -1, 1, -1, -1, 4, -1]
    before = tuple(t.clone() for t in meta.compressor)
    torch.manual_seed(61)
    for owner in (2, 8, 14):
        content = torch.randn(7, 512, device=device)
        scores = torch.randn_like(content)
        tail = backend.cache_pool.compressor_tail(owner)
        tail.normal_()
        old_tail = tail.clone()
        expected = torch.zeros_like(content)
        # Independent pair lookup, including the cross-chunk tail read.
        coords = list(
            zip(meta.positions.tolist(), meta.request_indices.tolist(), strict=True)
        )
        for i, (position, request) in enumerate(coords):
            if position % 2 != 1:
                continue
            if (position - 1, request) in coords:
                prior = coords.index((position - 1, request))
                c, s = content[prior], scores[prior]
            else:
                page = tables[TAIL][request, (position - 1) // 2]
                c, s = old_tail[page, (position - 1) % 2]
            weights = torch.stack((s, scores[i])).softmax(0)
            expected[i] = weights[0] * c + weights[1] * content[i]
        got, positions, requests = backend.compress(
            owner, content, scores, ForwardMode.EXTEND, None, 0.0
        )
        torch.testing.assert_close(got, expected)
        assert positions is meta.compressor.pair_positions
        assert requests is meta.compressor.pair_requests
        for original, value in zip(before, meta.compressor, strict=True):
            torch.testing.assert_close(original, value, rtol=0, atol=0)
    assert len(calls) == 1
    # The canonical compressor entry point rejects a partial projection window.
    with pytest.raises(ValueError, match="content/scores"):
        backend.compress(2, content[:2], scores[:2], ForwardMode.EXTEND, None, 0.0)
    # Preparing a new forward must resolve the changed LCM assignment again.
    tables[TAIL][0, 1] = 0
    with pytest.raises(RuntimeError, match="compressor tail"):
        _extend(backend, tables, [4, 3], [3, 0])
    assert len(calls) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("verify_width", [1, 3, 5, 6])
@pytest.mark.parametrize("decode_prefix", [8, 9])
def test_mixed_compressor_plan_windows_match_combined_pooling(
    device, verify_width, decode_prefix
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    backend = _verify_backend(device, 2, verify_width)
    backend.init_forward_metadata(
        2,
        1,
        torch.tensor([23, 17], device=device),
        torch.tensor([5, decode_prefix + verify_width], device=device),
        ForwardMode.MIXED,
        block_tables=_tables(device),
        extend_seq_lens=torch.tensor([3], device=device),
        extend_seq_lens_cpu=torch.tensor([3]),
        extend_prefix_lens=torch.tensor([2], device=device),
        extend_prefix_lens_cpu=torch.tensor([2]),
        extend_with_prefix=True,
    )
    full = backend.query_metadata(ForwardMode.MIXED).compressor
    extend = backend.query_metadata(ForwardMode.EXTEND).compressor
    decode = backend.query_metadata(ForwardMode.DECODE).compressor
    expected_previous = [
        i - 1 if i > 0 and (decode_prefix + i) % 2 else -1 for i in range(verify_width)
    ]
    assert decode.previous.tolist() == expected_previous
    assert full.previous.tolist() == [-1, 0, -1] + [
        previous + 3 if previous >= 0 else -1 for previous in expected_previous
    ]
    for name, all_rows, leading, trailing in zip(
        full._fields, full, extend, decode, strict=True
    ):
        assert leading.data_ptr() == all_rows.data_ptr()
        if name != "previous":
            assert trailing.data_ptr() == all_rows[3:].data_ptr()
    before = tuple(tensor.clone() for tensor in full)
    torch.manual_seed(62)
    content = torch.randn(3 + verify_width, 512, device=device)
    # Isolate address rebasing from CPU softmax's batch-size-dependent rounding.
    # Equal scores give exact 0.5 weights while random content exposes wrong rows.
    scores = torch.zeros_like(content)
    backend.cache_pool.arena.buffer.zero_()
    expected = backend.compress(2, content, scores, ForwardMode.MIXED, None, 0.0)
    backend.cache_pool.arena.buffer.zero_()
    leading = backend.compress(
        2, content[:3], scores[:3], ForwardMode.EXTEND, None, 0.0
    )
    trailing = backend.compress(
        2, content[3:], scores[3:], ForwardMode.DECODE, None, 0.0
    )
    for got, left, right in zip(expected, leading, trailing, strict=True):
        torch.testing.assert_close(got, torch.cat((left, right)), rtol=0, atol=0)
    for original, value in zip(before, full, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compressor_plan_ragged_windows_rebase_only_predecessors(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tokenspeed_kernel.ops.attention import dsv41

    positions = torch.tensor([2, 3, 4, 10, 11, 12, 7, 8, 9, 10], device=device)
    requests = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], device=device)
    table = torch.ones((3, 8), dtype=torch.int32, device=device)
    full = V41CompressorPlan.allocate(positions.numel(), torch.device(device))
    dsv41.compressor_metadata(positions, requests, table, 2, *full)
    before = tuple(tensor.clone() for tensor in full)
    for start, stop in ((0, 3), (3, 10), (6, 10), (4, 8), (6, 6), (0, 0)):
        window = full.window(start, stop)
        expected = V41CompressorPlan.allocate(stop - start, torch.device(device))
        dsv41.compressor_metadata(
            positions[start:stop], requests[start:stop], table, 2, *expected
        )
        for name, parent, got, want in zip(
            full._fields, full, window, expected, strict=True
        ):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            if start == 0 or name != "previous":
                assert got.data_ptr() == parent[start:stop].data_ptr()
    for original, value in zip(before, full, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
