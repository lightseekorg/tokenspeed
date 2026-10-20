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

import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    DeepseekV41AttentionBackend,
    V41CompressorPlan,
    V41PrefillSpan,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import DeepseekV41Config
from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_CACHE_FORMATS,
)
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
from tokenspeed.runtime.utils.server_args import ServerArgs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

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
        attention_query_swa_min_tokens=4096,
        attention_query_min_tokens=1024,
        attention_query_rank=0,
        attention_query_group=(0,),
        indexer_query_rank=0,
        indexer_query_group=(0,),
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
        # The geometry assertions in this file pin the checkpoint's own rows,
        # so CPU fixtures stay on them; a test that runs real kernels has to
        # store the rows this GPU's FlashMLA can actually read.
        cache_format=(
            "v4" if device != "cpu" and current_platform().is_hopper else "v41"
        ),
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
            kv_offload_config=None,
            pipeline_parallel_size=1,
            chunked_prefill_size=512,
            max_num_seqs=2,
            max_total_tokens=1024,
            disaggregation_mode="null",
            enable_prefix_caching=True,
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
        probe_batch_rows=None,
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
    backend._decode_schedules[(0, 0)] = (torch.empty(1), None, object())
    backend._prepared_selections[()] = ()
    backend.begin_indexer_forward()
    backend._index_geometry[()] = object()

    new_pool = _pool(_recipe("cpu"), "cpu")
    backend.set_cache_pool(new_pool)

    assert backend.cache_pool is new_pool
    assert backend.forward_metadata is None
    assert backend.forward_prefill_metadata is None
    assert backend.forward_decode_metadata is None
    assert not backend._decode_views_by_bs
    assert backend._decode_buffers is None
    assert backend._max_decode_bs == 0
    assert not backend._swa_plans
    assert not backend._prefill_spans
    assert not backend._decode_schedule_keepalive
    assert not backend._decode_schedules
    assert not backend._prepared_selections
    assert not backend._index_geometry
    backend.end_indexer_forward()
    backend.init_cuda_graph_state(2, max_tokens_per_req=1, overlap_schedule_depth=1)
    assert backend._decode_buffers is not old_buffers


def _tables(device):
    packing = _config(device).component(DeepseekV41Config).row_layout().group_packing
    return _tables_with_packing(device, packing)


def _extend(backend, tables, lengths, prefixes, replays, prompt_lens):
    """One EXTEND batch; ``prefixes`` are the extend starts (replay rows
    included), ``replays`` the leading rows re-fed inside a prefix hit and
    ``prompt_lens`` the full prompt lengths deciding which spans complete."""
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
        extend_replay_lens_cpu=torch.tensor(replays, dtype=torch.int32),
        extend_prompt_lens_cpu=torch.tensor(prompt_lens, dtype=torch.int32),
        extend_with_prefix=any(prefixes),
        query_shard=None,
    )
    return backend.query_metadata(ForwardMode.EXTEND)


def _final(lengths, prefixes):
    """Prompt lengths for spans whose chunk completes the prompt."""
    return [p + n for p, n in zip(prefixes, lengths)]


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
        extend_replay_lens_cpu=torch.tensor([0]),
        extend_prompt_lens_cpu=torch.tensor([5]),
        extend_with_prefix=True,
        query_shard=None,
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


def test_replay_span_truncates_swa_prefix_and_validates_lengths():
    """Rows re-fed inside a prefix hit start their SWA window at the replay
    start: the plan carries no retained-prefix slots for them, while an
    ordinary chunk at the same offset looks back into its own rows."""
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    meta = _extend(backend, tables, [131], [64], [0], [195])
    plain = backend._swa_query_plan(
        meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    assert plain.requests[0].prefix_slots.numel() == 64
    meta = _extend(backend, tables, [131], [64], [128], [195])
    replay = backend._swa_query_plan(
        meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    assert replay.requests[0].prefix_slots.numel() == 0
    assert meta.global_write_floor.tolist() == [192]
    assert backend.query_metadata(ForwardMode.DECODE).global_write_floor is None
    for lengths, prefixes, replays, prompts in (
        ([3], [4], [5], [7]),  # replay longer than the chunk
        ([3], [4], [-1], [7]),
        ([3], [4], [0], [6]),  # chunk runs past its prompt
    ):
        with pytest.raises(ValueError, match="replay/prompt"):
            _extend(backend, tables, lengths, prefixes, replays, prompts)


def test_write_global_masks_rows_below_the_replay_floor(monkeypatch):
    """A replayed row's global/index rows already sit in the pages the hit
    claimed: the owner's scatter gets slot -1 for every compression group
    whose last position lies below the request's floor, for both ratios."""
    from tokenspeed_kernel.ops.attention import dsv41

    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    # Request 0 replays [4, 8) of a hit at 8 and adds [8, 12); request 1 is a
    # plain chunk with no floor above its own prefix.
    meta = _extend(backend, tables, [8, 3], [4, 2], [4, 0], [12, 5])
    assert meta.global_write_floor.tolist() == [8, 2]
    seen = {}
    monkeypatch.setattr(
        dsv41,
        "cache_scatter",
        lambda rows, cache, slots, fmt: seen.__setitem__(fmt, slots.clone()),
    )
    rows = meta.positions.numel()
    backend.write_global(
        20,
        torch.zeros(rows, 512),
        torch.zeros(rows, 128),
        meta.positions,
        meta.request_indices,
        ForwardMode.EXTEND,
        main_rope_cache=None,
    )
    unmasked = backend.cache_slots(
        R1, meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    masked = (meta.positions < meta.global_write_floor[meta.request_indices]).tolist()
    assert masked == [True] * 4 + [False] * 4 + [False] * 3
    for fmt in ("global", "index"):
        assert torch.equal(seen[fmt], unmasked.masked_fill(torch.tensor(masked), -1))
    # Ratio 2: pair starts at 4 and 6 end below the floor of 8; the pair at 8
    # is the first the request may write. Request 1's pair at 2 is its own.
    pairs = torch.tensor([4, 6, 8, 10, 2])
    requests = torch.tensor([0, 0, 0, 0, 1])
    backend.write_global(
        2,
        torch.zeros(5, 512),
        torch.zeros(5, 128),
        pairs,
        requests,
        ForwardMode.EXTEND,
        main_rope_cache=None,
    )
    unmasked = backend.cache_slots(R2, pairs, requests, ForwardMode.EXTEND)
    expected = unmasked.masked_fill(torch.tensor([True, True, False, False, False]), -1)
    assert torch.equal(seen["global"], expected)
    # Padding rows keep their negative slot and never index the floor.
    backend.write_global(
        20,
        torch.zeros(1, 512),
        torch.zeros(1, 128),
        torch.tensor([-1]),
        torch.tensor([-1]),
        ForwardMode.EXTEND,
        main_rope_cache=None,
    )
    assert seen["global"].tolist() == [-1]


def test_decoder_view_is_the_identity_for_decode_and_complete_short_chunks():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    meta = _extend(backend, tables, [5, 3], [0, 0], [0, 0], [5, 3])
    view = backend.decoder_view()
    assert view.keep_rows is None and view.logits_rows is None
    assert view.metadata is meta and view.prefill is meta
    assert [(s.request, s.offset, s.prefix, s.count) for s in view.spans] == [
        (0, 0, 0, 5),
        (1, 5, 0, 3),
    ]
    # Keeping every query row does not make an older SWA prefix visible.
    meta = _extend(backend, tables, [5, 3], [0, 2], [0, 0], [5, 5])
    view = backend.decoder_view()
    assert view.keep_rows.tolist() == list(range(8))
    assert view.logits_rows.tolist() == [4, 7]
    assert view.metadata is not meta
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([0, 1]),
        torch.tensor([23, 17]),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=False,
    )
    view = backend.decoder_view()
    assert view.keep_rows is None and view.logits_rows is None
    assert view.metadata is backend.query_metadata(ForwardMode.DECODE)
    assert view.prefill is None and view.spans == ()
    # Graph replay at the same batch size refreshes the captured decode
    # window in place: the view names that same metadata and no tensors.
    backend.refresh_decode_metadata(
        2,
        1,
        torch.tensor([0]),
        torch.tensor([24]),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=True,
    )
    assert backend.decoder_view().metadata is view.metadata
    assert backend.decoder_view()[1:] == (None, (), None, None)


@pytest.mark.parametrize("open_count", [1, 17])
def test_decoder_view_preserves_compute_rows_without_open_chunk_logits(open_count):
    backend = _verify_backend("cpu", 2, 1)
    _extend(
        backend,
        _tables("cpu"),
        [128, open_count],
        [0, 0],
        [0, 0],
        [128, 256],
    )
    view = backend.decoder_view()
    # Dropping the last row changes the GEMM shape seen by the completing
    # request. Keep that computation, but never sample the open request.
    assert view.keep_rows.tolist() == list(range(128)) + [127 + open_count]
    assert view.metadata.request_indices.tolist() == [0] * 128 + [1]
    assert view.logits_rows.tolist() == [127]


def test_decoder_view_keeps_compute_rows_and_selects_only_valid_logits():
    """The CED decoder runs on each prompt-completing chunk's last window
    (the whole chunk when shorter), on one numerical-shape row of an open
    chunk, and on every decode row; open chunks have no sampled row."""
    backend = _verify_backend("cpu", 3, 2)
    tables = _tables("cpu")
    tables = {gid: torch.cat((t, t[:1])) for gid, t in tables.items()}
    backend.init_forward_metadata(
        3,
        2,
        torch.tensor([0, 1, 2]),
        torch.tensor([9, 5, 9]),
        ForwardMode.MIXED,
        block_tables=tables,
        extend_seq_lens=torch.tensor([5, 3]),
        extend_seq_lens_cpu=torch.tensor([5, 3]),
        extend_prefix_lens=torch.tensor([4, 2]),
        extend_prefix_lens_cpu=torch.tensor([4, 2]),
        extend_replay_lens_cpu=torch.tensor([4, 0]),
        extend_prompt_lens_cpu=torch.tensor([9, 12]),
        extend_with_prefix=True,
        query_shard=None,
    )
    full = backend.query_metadata(ForwardMode.MIXED)
    view = backend.decoder_view()
    # Keep the open request's last compute row, but omit it from logits.
    assert view.keep_rows.tolist() == [0, 1, 2, 3, 4, 7, 8, 9]
    assert view.metadata.positions.tolist() == [4, 5, 6, 7, 8, 4, 7, 8]
    assert view.metadata.request_indices.tolist() == [0, 0, 0, 0, 0, 1, 2, 2]
    assert view.logits_rows.tolist() == [4, 6, 7]
    assert view.spans == (
        V41PrefillSpan(0, 0, 4, 5, 4),
        V41PrefillSpan(1, 5, 4, 1, 4),
    )
    assert view.prefill.positions.tolist() == [4, 5, 6, 7, 8, 4]
    assert torch.equal(
        view.metadata.swa_write_slots, full.swa_write_slots[view.keep_rows]
    )
    assert view.metadata.global_write_floor is full.global_write_floor
    # The view's prefill rows are a canonical window: planning them uses the
    # host spans (no device snapshot) and each row's SWA starts at its span.
    assert backend._window(
        view.prefill.positions, view.prefill.request_indices, ForwardMode.EXTEND
    ) == (view.prefill, view.spans)
    plan = backend._swa_query_plan(
        view.prefill.positions, view.prefill.request_indices, ForwardMode.EXTEND
    )
    assert [r.prefix_slots.numel() for r in plan.requests] == [0, 0]
    # A completing chunk longer than the window keeps exactly the window.
    # Without decode rows the view is its own prefill window, so the rows
    # the decoder layers hand back are recognized as canonical.
    meta = _extend(backend, tables, [140], [0], [0], [140])
    view = backend.decoder_view()
    assert view.keep_rows.tolist() == list(range(12, 140))
    assert view.spans == (V41PrefillSpan(0, 0, 12, 128, 12),)
    assert view.logits_rows.tolist() == [127]
    assert view.prefill is view.metadata
    assert view.metadata.positions.tolist() == meta.positions[12:].tolist()
    assert backend._window(
        view.metadata.positions, view.metadata.request_indices, ForwardMode.EXTEND
    ) == (view.metadata, view.spans)


def _decode_compute(backend, inputs, bs):
    backend.begin_indexer_stage()
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
    backend.write_global(
        2, latent, latent[:, :128], pair_pos, pair_req, mode, main_rope_cache=None
    )
    backend.write_global(20, swa[:bs], iq[:bs, 0], pos, req, mode, main_rope_cache=None)
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
                swa_norm_weight=None,
                swa_norm_eps=0.0,
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
                    21,
                    None,
                    None,
                    pos[order],
                    req[order],
                    mode,
                    None,
                )
            )
    return outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shared_pool", [False, True])
@pytest.mark.parametrize("verify_width", [1, 3, 5])
def test_gpu_backend_decode_capture_replay_and_above_ladder(shared_pool, verify_width):
    from tokenspeed_kernel.ops.attention import dsv41

    assert DeepseekV41AttentionBackend.cuda_graph_support.decode_graph
    # Decoder narrowing makes the prefill row count depend on prompt
    # completion, not on the token bucket; the prefill graph captures the
    # encoder and decoder stages separately around it (NarrowingPrefillModel).
    assert DeepseekV41AttentionBackend.cuda_graph_support.prefill_graph
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
    meta = _extend(backend, tables, [12, 12], [0, 0], [0, 0], _final([12, 12], [0, 0]))
    history = torch.randn(24, 512, device="cuda", dtype=torch.bfloat16)
    for layer in (2, 3, 20, 24, 25):
        dsv41.cache_scatter(
            history,
            backend.cache_pool.swa(layer),
            meta.swa_write_slots,
            backend.swa_format,
        )
    backend.write_global(
        20,
        history,
        history[:, :128],
        meta.positions,
        meta.request_indices,
        ForwardMode.EXTEND,
        main_rope_cache=None,
    )
    pooled, pos, req = backend.compress(
        2,
        history.float(),
        torch.randn(24, 512, device="cuda"),
        ForwardMode.EXTEND,
        norm_weight=None,
        norm_eps=0.0,
    )
    backend.write_global(
        2, pooled, pooled[:, :128], pos, req, ForwardMode.EXTEND, main_rope_cache=None
    )

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
    args = ServerArgs(
        model="x",
        device="cpu",
        attn_tp_size=1,
        deepseek_v41_attention_query_swa_min_tokens=4096,
        deepseek_v41_attention_query_min_tokens=1024,
        deepseek_v41_attention_query_parallel_size=1,
        deepseek_v41_indexer_query_parallel_size=1,
        data_parallel_size=2,
        prefix_granularity=128,
        max_num_seqs=4,
        chunked_prefill_size=3,
        max_total_tokens=1024,
        kv_cache_quant_method="none",
        disaggregation_mode="null",
        disaggregation_layerwise_interval=1,
        pipeline_parallel_size=1,
        # spec_context_pad is derived (3 * draft tokens); a width-1 DSPARK
        # keeps context_len in the 576-score-column bucket the workspace
        # arithmetic below pins, which no real no-spec config can reach.
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=verify_width,
        speculative_num_steps=verify_width - 1,
    )
    args.mapping.rank = 0
    model = SimpleNamespace(
        hf_config=hf,
        num_attention_layers=40,
        num_attention_heads=2,
        dtype=torch.bfloat16,
        context_len=512,
    )
    config = DeepseekV41Config.generate(args, model, False)
    spec = config.component(DeepseekV41Config)
    assert spec.cache_format in V41_CACHE_FORMATS
    config = replace(config, components=(replace(spec, cache_format="v41"),))
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
    # Retention is the attention window (or the pair) whatever the verify
    # width or schedule depth: the scheduled rows are reserved, not retained.
    assert specs[SWA].sliding_window_tokens == 128
    assert specs[TAIL].sliding_window_tokens == 2
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
    # A PD role must transfer the cache once per prompt, not per layer.
    args.disaggregation_mode = "prefill"
    with pytest.raises(NotImplementedError, match="layerwise-interval 0"):
        DeepseekV41Config.generate(args, model, False)
    args.disaggregation_layerwise_interval = 0
    assert DeepseekV41Config.generate(args, model, False).pd_disaggregation_enabled
    args.disaggregation_mode = "null"
    args.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError, match="PP=1"):
        recipe.groups()
    args.pipeline_parallel_size = 1
    recipe.attn_config = replace(config, pd_disaggregation_enabled=True)
    pd_specs = {spec.group_id: spec for spec, _ in recipe.groups()}
    assert set(pd_specs) == set(specs)
    assert all(spec.transfer_policy == "full_suffix" for spec in pd_specs.values())
    assert all(spec.transfer_policy is None for spec in specs.values())


def test_pool_reports_the_whole_arena_as_kv_bytes():
    # The startup memory summary reads this; without it the arena is
    # misattributed to activations and the KV row prints 0.
    pool = _pool(_recipe("cpu"), "cpu")
    assert pool.get_kv_size_bytes() == pool.arena.buffer.nbytes > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dspark", [False, True])
def test_pool_zeroes_fresh_pages_per_group(dspark):
    recipe = _dspark_recipe("cuda") if dspark else _recipe("cuda")
    pool = _pool(recipe, "cuda")
    assert pool.requires_page_zeroing
    pool.arena.buffer.fill_(1)
    r2_packing = pool.arena.plan.group(R2).cache_blocks_per_lcm_block
    r2_page = 2 * r2_packing + 1
    # SWA uses parent 1 and R2 uses parent 3. Clearing a fresh page must
    # preserve its neighbor, including when two SWA pages share a parent.
    pool.zero_new_blocks({SWA: np.asarray([1]), R2: np.asarray([r2_page])})
    assert pool.swa(3)[1].count_nonzero() == 0 and pool.swa(3)[2].count_nonzero() > 0
    assert pool.global_kv(8)[r2_page].count_nonzero() == 0
    assert pool.global_kv(8)[r2_page + 1].count_nonzero() > 0
    if dspark:
        for stage in range(3):
            assert pool.dspark_kv(stage)[1].count_nonzero() == 0
            assert pool.dspark_kv(stage)[2].count_nonzero() > 0


def test_dspark_windows_join_the_swa_group():
    recipe = _dspark_recipe("cpu")
    assert recipe.dspark_stages() == 3
    groups = dict(recipe.groups())
    assert {s.group_id for s in groups} == set(V41_GROUP_GEOMETRY)
    spec = next(s for s in groups if s.group_id == SWA)
    fields = {f.field_id: f for f in groups[spec]}
    assert len(fields) == 43
    for stage in range(3):
        field = fields[f"layer.39.dspark_kv{stage}"]
        assert field.shape == (64, 512) and field.dtype == "bfloat16"
    # The three stage rows enlarge each SWA page. Two fit in a parent while
    # all groups meet the index alignment and stay inside the padding budget.
    layout = _layout(recipe)
    assert layout.lcm_block_bytes == 3_225_600
    assert dict(layout.group_packing) == {SWA: 2, R2: 45, R1: 140, TAIL: 126}
    recipe.check_layout(layout)
    setup = recipe.setup()
    assert setup.spec.token_capacity == 1024
    assert (
        setup.spec.memory_plan.arena_bytes + setup.fixed_workspace_bytes
        <= recipe.cache_budget_bytes
    )
    pool = _pool(recipe, "cpu")
    assert pool.dspark_kv(2).shape[1:] == (64, 512)
    assert pool.dspark_kv(2).dtype == torch.bfloat16
    recipe.attn_config = replace(recipe.attn_config, pd_disaggregation_enabled=True)
    from tokenspeed.runtime.pd.cache_protocol import (
        build_arena_cache_transfer_contract,
    )

    contract, _ = build_arena_cache_transfer_contract(_pool(recipe, "cpu").arena)
    assert {f.field_id for f in contract.fields_for_group(SWA)} >= {
        f"layer.39.dspark_kv{stage}" for stage in range(3)
    }
    recipe.server_args.speculative_algorithm = None
    assert recipe.dspark_stages() == 0
    assert _layout(recipe).lcm_block_bytes == 1_382_400
    recipe.server_args.speculative_algorithm = "DSPARK"
    recipe.draft_model_config.hf_config.dspark_num_stages = 0
    with pytest.raises(ValueError, match="positive stage count"):
        recipe.groups()


@pytest.mark.parametrize("dspark", [False, True])
def test_pd_contract_plan_and_manifest(dspark):
    import numpy as np

    from tokenspeed.runtime.pd.cache_protocol import (
        build_arena_cache_transfer_contract,
        build_cache_block_manifest,
        validate_cache_peer_layout,
    )
    from tokenspeed.runtime.pd.transfer_plan import CacheTransferPlanner

    recipe = _dspark_recipe("cpu") if dspark else _recipe("cpu")
    recipe.attn_config = replace(recipe.attn_config, pd_disaggregation_enabled=True)
    pool = _pool(recipe, "cpu")
    assert pool.arena.supports_disaggregation is True
    contract, base_addr = build_arena_cache_transfer_contract(pool.arena)
    assert base_addr == pool.arena.buffer.data_ptr()
    validate_cache_peer_layout(contract, contract)
    assert [spec.group_id for spec in contract.group_specs] == [SWA, R2, R1, TAIL]
    # The replayable groups keep their declaration under PD (a peer that
    # cached them would disagree on what a hit means) and travel as ordinary
    # sliding windows: the whole retained tail ships, the replay regenerates
    # exactly that window, and nothing is re-fed on the decode side.
    specs_by_id = {spec.group_id: spec for spec in contract.group_specs}
    for gid in (SWA, TAIL):
        assert specs_by_id[gid].replayable
    assert all(spec.transfer_policy == "full_suffix" for spec in contract.group_specs)
    cached_peer = replace(
        contract,
        group_specs=tuple(
            replace(spec, replayable=False) if spec.group_id == SWA else spec
            for spec in contract.group_specs
        ),
    )
    with pytest.raises(Exception, match="semantics"):
        validate_cache_peer_layout(contract, cached_peer)
    swa_fields = {f"layer.{i}.swa" for i in range(40)}
    if dspark:
        swa_fields.update(f"layer.39.dspark_kv{stage}" for stage in range(3))
    assert {f.field_id for f in contract.fields_for_group(SWA)} == swa_fields
    assert {f.field_id for f in contract.fields_for_group(R2)} == {
        f"layer.{o}.{name}" for o in (2, 8, 14) for name in ("global_kv", "index_k")
    }
    assert {f.field_id for f in contract.fields_for_group(TAIL)} == {
        f"layer.{o}.compressor_tail" for o in (2, 8, 14)
    }
    # No V4.1 field is head-sharded, so unequal TP copies whole fields from
    # one replicated source rank per decode rank.
    planner = CacheTransferPlanner(
        prefill_tp_size=4,
        decode_tp_size=2,
        prefill_layout=contract,
        decode_layout=contract,
        # A single-stage prefill owns every field.
        prefill_field_ids=None,
    )
    for decode_rank, source in ((0, 0), (1, 2)):
        plan = planner.plan_for_decode_rank(decode_rank)
        assert plan.target_prefill_ranks == (source,)
        fragments = plan.fragments_by_prefill_rank[source]
        assert len(fragments) == len(contract.plan.fields)
        by_field = {f.field_id: f for f in contract.plan.fields}
        assert all(
            fragment.rows_per_page == 1
            and fragment.bytes_per_row == by_field[fragment.field_id].payload_bytes
            for fragment in fragments
        )

    # An odd prompt leaves an unfinished ratio-2 pair: its input lives in the
    # compressor tail, so the tail and the last global_r2 page must both ship.
    prompt_len, prefix_len = 4097, 3840
    tables = {}
    for spec in contract.group_specs:
        columns = prompt_len // spec.block_granularity + 2
        capacity = contract.plan.group(spec.group_id).page_count
        # Any non-null page ID inside the group's capacity is a valid block.
        tables[spec.group_id] = (
            1 + np.arange(2 * columns).reshape(2, columns) % (capacity - 1)
        ).astype(np.int32)
    manifest = build_cache_block_manifest(
        SimpleNamespace(block_tables_arrays=lambda: tables),
        layout=contract,
        request_row=1,
        prefix_len=prefix_len,
        prompt_len=prompt_len,
        state_prefix_cached=False,
    )
    blocks = {group.group_id: group.block_ids for group in manifest.groups}
    specs = {spec.group_id: spec for spec in contract.group_specs}

    def logical(gid, begin, end):
        return tuple(int(tables[gid][1, slot]) for slot in range(begin, end))

    assert blocks[R2] == logical(R2, prefix_len // 128, (prompt_len + 127) // 128)
    assert blocks[R1] == logical(R1, prefix_len // 64, (prompt_len + 63) // 64)
    swa_begin = (prompt_len - specs[SWA].sliding_window_tokens + 1) // 64
    assert blocks[SWA] == logical(SWA, swa_begin, (prompt_len + 63) // 64)
    tail_begin = (prompt_len - specs[TAIL].sliding_window_tokens + 1) // 2
    assert blocks[TAIL] == logical(TAIL, tail_begin, (prompt_len + 1) // 2)
    assert (prompt_len - 1) // 2 in range(tail_begin, (prompt_len + 1) // 2)


@pytest.mark.parametrize("cache_format", ["v41", "v4"])
@pytest.mark.parametrize("dspark", [False, True])
def test_every_cache_format_packs_to_the_plane_it_declares(cache_format, dspark):
    """Each format's frozen packing must be what the planner derives from it."""
    rows = V41_CACHE_FORMATS[cache_format]
    recipe = _recipe("cpu")
    spec = recipe.attn_config.component(DeepseekV41Config)
    recipe.attn_config = replace(
        recipe.attn_config, components=(replace(spec, cache_format=cache_format),)
    )
    if dspark:
        recipe.server_args.speculative_algorithm = "DSPARK"
        recipe.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dspark_num_stages=3)
        )
    expected_packing = rows.dspark_group_packing if dspark else rows.group_packing
    expected_plane = rows.dspark_lcm_block_bytes if dspark else rows.lcm_block_bytes
    layout = _layout(recipe)
    recipe.check_layout(layout)
    assert layout.lcm_block_bytes == expected_plane
    assert layout.plane_bytes == (("flatkv", expected_plane),)
    assert dict(layout.group_packing) == dict(expected_packing)
    widths = {f.field_id.split(".")[-1]: f.shape[-1] for f in layout.fields}
    assert widths["swa"] == rows.swa_row_bytes
    assert widths["global_kv"] == rows.global_row_bytes
    assert widths["index_k"] == rows.index_row_bytes
    if cache_format == "v41":
        # Native sparse index scoring reads both ordinary and DSpark arenas.
        assert all(
            f.page_stride_bytes % 512 == 0
            for f in layout.fields
            if f.field_id.endswith(".index_k")
        )


def test_recipe_exact_geometry_capacity_and_dispatch():
    recipe = _recipe("cpu")
    layout = _layout(recipe)
    recipe.check_layout(layout)
    assert layout.lcm_block_bytes == 1_382_400
    assert layout.plane_bytes == (("flatkv", 1_382_400),)
    assert dict(layout.group_packing) == dict(V41_CACHE_FORMATS["v41"].group_packing)
    specs = {s.group_id: s for s, _ in recipe.groups()}
    assert [specs[g].block_granularity for g in V41_GROUP_GEOMETRY] == [64, 128, 64, 2]
    assert all(s.attn_kind != "state" for s in specs.values())
    assert specs[SWA].sliding_window_tokens == 128
    assert specs[TAIL].sliding_window_tokens == 2
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
    model = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41_text"),
        model_profile=None,
    )
    profile = _resolve_attn_side(model, requested_backend=None)
    assert _resolve_cache_family(profile, recipe.attn_config) == "deepseek_v41"


def test_recipe_declares_the_private_groups_replayable():
    """The SWA and compressor-tail groups leave prefix caching: the recipe
    marks them replayable, the scheduler bridge forwards the flag, and the
    backend refuses a pool that would share them through a hit."""
    from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups

    recipe = _recipe("cpu")
    specs = {s.group_id: s for s, _ in recipe.groups()}
    expected = {SWA: True, R2: False, R1: False, TAIL: True}
    assert {gid: s.replayable for gid, s in specs.items()} == expected
    # What a hit re-feeds is the whole retention window: the attention window
    # or the pair, with no second number to declare.
    assert {gid: s.sliding_window_tokens for gid, s in specs.items()} == {
        SWA: 128,
        R2: None,
        R1: None,
        TAIL: 2,
    }
    backend = _backend("cpu", 2)
    groups = {g.group_id: g for g in pool_to_cache_groups(backend.cache_pool)}
    assert {gid: g.replayable for gid, g in groups.items()} == expected
    with pytest.raises(ValueError, match="sliding-window"):
        replace(specs[R1], replayable=True)
    cached_swa = tuple(
        replace(spec, replayable=False) if spec.group_id == SWA else spec
        for spec, _ in recipe.groups()
    )
    arena = CacheArena(
        _layout(recipe).bind(16),
        "cpu",
        cache_group_specs=cached_swa,
        token_capacity=1024,
        enable_memory_saver=False,
    )
    pool = DeepseekV41CachePool(arena, layer_num=40, rank=0, field_layer_offset=0)
    with pytest.raises(ValueError, match="must be a replayable group"):
        backend.set_cache_pool(pool)


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
    meta = _extend(backend, tables, [prefix], [0], [0], _final([prefix], [0]))
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
    # CPU float32 kernels round differently across vector widths (one ulp on
    # some hosts), so compare at float32 precision rather than bit for bit.
    torch.testing.assert_close(pooled[live], expected, rtol=1.3e-6, atol=1e-6)
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
        meta = _extend(
            backend, tables, [count], [prefix], [0], _final([count], [prefix])
        )
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
    meta = _extend(backend, tables, [2], [7], [0], _final([2], [7]))
    backend.compress(
        2,
        torch.randn(2, 512),
        torch.randn(2, 512),
        ForwardMode.EXTEND,
        norm_weight=None,
        norm_eps=0.0,
    )
    # Reject that suffix logically, then complete a different token7.
    meta = _extend(backend, tables, [1], [7], [0], _final([1], [7]))
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
    meta = _extend(backend, tables, [17, 3], [0, 0], [0, 0], _final([17, 3], [0, 0]))
    torch.manual_seed(7)
    n = meta.positions.numel()
    main = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
    index = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
    backend.write_global(
        20,
        main,
        index,
        meta.positions,
        meta.request_indices,
        ForwardMode.EXTEND,
        main_rope_cache=None,
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
        swa_norm_weight=None,
        swa_norm_eps=0.0,
    )
    record = backend.sparse_topk.prefill
    swa_fmt, global_fmt = backend.swa_format, backend.global_format
    dq_swa = dsv41.cache_unpack(dsv41.cache_pack(swa, swa_fmt, None), swa_fmt, None)
    dq_main = dsv41.cache_unpack(
        dsv41.cache_pack(main, global_fmt, None), global_fmt, None
    )
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
        21,
        None,
        None,
        positions,
        requests,
        ForwardMode.EXTEND,
        None,
    )
    want = backend.global_read_slots(
        20, record.logical_rows[pick], positions, requests, ForwardMode.EXTEND
    )
    torch.testing.assert_close(reuse, want)
    for layer in (24, 28):
        slots, lengths = backend.select_global(
            layer,
            iq[pick],
            iw[pick],
            positions,
            requests,
            ForwardMode.EXTEND,
            None,
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
        extend_replay_lens_cpu=torch.tensor([0]),
        extend_prompt_lens_cpu=torch.tensor([5]),
        extend_with_prefix=True,
        query_shard=None,
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
            meta = _extend(
                backend, tables, [count], [prefix], [0], _final([count], [prefix])
            )
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
        backend.write_global(
            2,
            latent,
            latent[:, :128].contiguous(),
            pos,
            req,
            mode,
            main_rope_cache=None,
        )
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
            swa_norm_weight=None,
            swa_norm_eps=0.0,
        )

    full = run(0, 10, ForwardMode.EXTEND)
    backend.cache_pool.arena.buffer.zero_()
    parts = [run(0, 3, ForwardMode.EXTEND), run(3, 4, ForwardMode.EXTEND)]
    parts.extend(run(p, 1, ForwardMode.DECODE) for p in (7, 8, 9))
    # Prefill and decode may run different attention kernels over the same
    # dequantized rows, so their reductions agree only to one BF16 ulp of the
    # row's largest entries, which near-zero entries see as an absolute error.
    torch.testing.assert_close(torch.cat(parts), full, rtol=2**-7, atol=2**-8)
    torch.cuda.synchronize()


def test_deferred_swa_norm_without_rope_cache_is_refused():
    # Only the fused RoPE writers apply a deferred SWA norm; without them the
    # norm would be silently dropped.
    backend = _backend("cpu", 2)
    meta = _extend(backend, _tables("cpu"), [4], [0], [0], _final([4], [0]))
    with pytest.raises(ValueError, match="requires the SWA RoPE cache"):
        backend.forward_v41(
            torch.randn(4, 2, 512, dtype=torch.bfloat16),
            torch.randn(4, 512, dtype=torch.bfloat16),
            layer_id=0,
            positions=meta.positions,
            request_indices=meta.request_indices,
            forward_mode=ForwardMode.EXTEND,
            index_q=None,
            index_weights=None,
            attn_sink=torch.zeros(2, dtype=torch.float32),
            softmax_scale=512**-0.5,
            index_process_group=None,
            swa_rope_cache=None,
            swa_norm_weight=torch.ones(512, dtype=torch.bfloat16),
            swa_norm_eps=1e-6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_swa_prefill_cross_chunk_matches_full_and_null_is_untouched():
    backend = _backend("cuda", 2)
    tables = _tables("cuda")
    torch.manual_seed(8)
    q = torch.randn(140, 2, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(140, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)
    meta = _extend(backend, tables, [140], [0], [0], _final([140], [0]))
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
        swa_norm_weight=None,
        swa_norm_eps=0.0,
    )
    backend.cache_pool.arena.buffer.zero_()
    chunks = []
    for prefix, count in ((0, 3), (3, 64), (67, 73)):
        meta = _extend(
            backend, tables, [count], [prefix], [0], _final([count], [prefix])
        )
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
                swa_norm_weight=None,
                swa_norm_eps=0.0,
            )
        )
    torch.testing.assert_close(torch.cat(chunks), full, rtol=0, atol=0)
    assert not bool(backend.cache_pool.swa(0)[0].any())
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_replayed_rows_attend_from_the_window_start_only():
    """Bounded replay truncates SWA at the replay start: a chunk re-feeding
    [64, 128) before 12 new rows attends exactly like a fresh 76-row prompt
    (RoPE is applied by the model, so only the window shape matters here),
    not like the same rows inside the full prompt."""
    backend = _backend("cuda", 2)
    tables = _tables("cuda")
    torch.manual_seed(9)
    q = torch.randn(140, 2, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(140, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)

    def attend(meta, rows):
        return backend.forward_v41(
            q[rows],
            swa[rows],
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
            swa_norm_weight=None,
            swa_norm_eps=0.0,
        )

    full = attend(_extend(backend, tables, [140], [0], [0], [140]), slice(0, 140))
    backend.cache_pool.arena.buffer.zero_()
    replayed = attend(_extend(backend, tables, [76], [64], [64], [140]), slice(64, 140))
    backend.cache_pool.arena.buffer.zero_()
    fresh = attend(_extend(backend, tables, [76], [0], [0], [76]), slice(64, 140))
    torch.testing.assert_close(replayed, fresh, rtol=0, atol=0)
    assert not torch.equal(replayed, full[64:])
    torch.cuda.synchronize()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prefill_plan_reuses_addresses_and_refresh_rechecks_pages(device):
    backend = _backend(device, 2)
    tables = _tables(device)
    meta = _extend(backend, tables, [3], [128], [0], _final([3], [128]))
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
    meta = _extend(backend, tables, [3], [128], [0], _final([3], [128]))
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
    meta = _extend(backend, tables, [4], [128], [0], _final([4], [128]))
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
    # A changed table must resolve to a new plan, not the cached one.
    backend._swa_plans.clear()
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

    def selected(*args, lse):
        assert lse is None
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
        swa_norm_weight=None,
        swa_norm_eps=0.0,
    )
    slots, lengths = captured[0][2:4]
    count = 0 if position is None else min(position + 1, 128)
    assert lengths.tolist() == [count]
    assert (slots[:, count:] == -1).all()
    if count:
        expected = torch.arange(position - count + 1, position + 1) + 64
        torch.testing.assert_close(slots[0, :count], expected.int())


def test_native_decode_schedule_shared_per_length_pair_within_a_forward(
    monkeypatch,
):
    from tokenspeed_kernel.ops.attention import dsv41

    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    produced = []
    monkeypatch.setattr(
        dsv41,
        "new_attention_schedule",
        lambda: produced.append(object()) or produced[-1],
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def refresh():
        backend.refresh_decode_metadata(
            1,
            1,
            torch.tensor([0]),
            torch.tensor([6]),
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=False,
        )

    refresh()
    swa_lens = backend.forward_decode_metadata.swa_read_lens
    global_lens = torch.tensor([512])
    # Every layer that reads the same length tensors gets the same producer;
    # SWA-only layers (no global lengths) form their own group.
    shared = backend._decode_schedule(swa_lens, global_lens)
    assert backend._decode_schedule(swa_lens, global_lens) is shared
    swa_only = backend._decode_schedule(swa_lens, None)
    assert swa_only is not shared
    assert backend._decode_schedule(swa_lens, None) is swa_only
    assert backend._decode_schedule(swa_lens, torch.tensor([512])) is not shared
    assert len(produced) == 3
    # The next forward's refreshed lengths must not reuse an initialized producer.
    refresh()
    assert backend._decode_schedule(swa_lens, global_lens) is not shared
    assert len(produced) == 4


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
    meta = _extend(backend, tables, [4, 3], [3, 0], [0, 0], _final([4, 3], [3, 0]))
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
    _extend(backend, tables, [4, 3], [3, 0], [0, 0], _final([4, 3], [3, 0]))
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
        extend_replay_lens_cpu=torch.tensor([0]),
        extend_prompt_lens_cpu=torch.tensor([5]),
        extend_with_prefix=True,
        query_shard=None,
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


def test_open_prefill_chunk_has_no_decoder_or_logits_rows():
    backend = _verify_backend("cpu", 1, 1)
    _extend(backend, _tables("cpu"), [8], [0], [0], [64])
    view = backend.decoder_view()
    assert view.keep_rows.numel() == 0
    assert view.logits_rows.numel() == 0
    assert view.metadata.positions.numel() == 0
    assert view.prefill.positions.numel() == 0
    assert view.spans == ()


def _global_rope_table():
    angles = (
        torch.arange(512, device="cuda")[:, None].float()
        * torch.exp(-torch.arange(32, device="cuda").float() / 4)[None, :]
    )
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _global_write_reference(
    main, index, positions, rope_cache, main_cache, index_cache, slots
):
    expected_main = main_cache.clone()
    expected_index = index_cache.clone()
    rotated = dsv41.rope_inplace(main.clone(), positions, rope_cache, None)
    dsv41.cache_scatter(rotated, expected_main, slots, "global")
    if isinstance(index, dsv41.IndexKeyInput):
        index = dsv41.index_key_norm_rope(
            index.values,
            index.norm_weight,
            positions,
            index.cos_sin_cache,
            index.norm_eps,
        )
    dsv41.cache_scatter(index, expected_index, slots, "index")
    return expected_main, expected_index


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_blackwell,
    reason="requires Blackwell global FP4 cache",
)
@pytest.mark.parametrize("owner,ratio,group", [(20, 1, R1), (2, 2, R2)])
@pytest.mark.parametrize("deferred", [False, True])
@torch.inference_mode()
def test_global_rope_write_preserves_replayed_shared_pages_and_null_page(
    owner, ratio, group, deferred
):
    backend = _backend("cuda", 2)
    tables = _tables("cuda")
    # Another request aliases the immutable first page. No global row below
    # the prefix-hit floor may overwrite it while the SWA history is replayed.
    tables[group][1, 0] = tables[group][0, 0]
    tables[group][0, 256 // (ratio * 64)] = 0
    meta = _extend(backend, tables, [130], [0], [128], [130])
    assert meta.global_write_floor.tolist() == [128]
    positions = torch.cat(
        (
            torch.arange(0, 130, ratio, device="cuda"),
            torch.tensor([-1, 256], device="cuda"),
        )
    )
    requests = torch.zeros_like(positions)
    requests[-2] = -1
    slots = backend.cache_slots(group, positions, requests, ForwardMode.EXTEND)
    expected_slots = slots.masked_fill(
        (positions >= 0) & (positions + ratio - 1 < 128), -1
    )
    assert slots[-2:].tolist() == [-1, -1]
    assert (expected_slots >= 0).sum().item() == (2 if ratio == 1 else 1)
    torch.manual_seed(101 + owner)
    main = torch.randn(positions.numel(), 512, device="cuda", dtype=torch.bfloat16)
    index = torch.randn(positions.numel(), 128, device="cuda", dtype=torch.bfloat16)
    main_before, index_before = main.clone(), index.clone()
    rope_cache = _global_rope_table()
    norm_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    index_input = (
        dsv41.IndexKeyInput(index, norm_weight, rope_cache, 1e-6) if deferred else index
    )
    main_cache = backend.cache_pool.global_kv(owner)
    index_cache = backend.cache_pool.index_k(owner)
    main_cache.fill_(165)
    index_cache.fill_(165)
    shared_page = int(tables[group][1, 0])
    main_alias = main_cache[shared_page]
    index_alias = index_cache[shared_page]
    expected_main, expected_index = _global_write_reference(
        main,
        index_input,
        positions,
        rope_cache,
        main_cache,
        index_cache,
        expected_slots,
    )
    backend.write_global(
        owner,
        main,
        index_input,
        positions,
        requests,
        ForwardMode.EXTEND,
        main_rope_cache=rope_cache,
    )
    torch.testing.assert_close(main_cache, expected_main, rtol=0, atol=0)
    torch.testing.assert_close(index_cache, expected_index, rtol=0, atol=0)
    torch.testing.assert_close(main, main_before, rtol=0, atol=0)
    torch.testing.assert_close(index, index_before, rtol=0, atol=0)
    assert torch.all(main_alias == 165) and torch.all(index_alias == 165)
    assert torch.all(main_cache[0] == 165) and torch.all(index_cache[0] == 165)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_blackwell,
    reason="requires Blackwell global FP4 cache",
)
@pytest.mark.parametrize("owner,group", [(20, R1), (2, R2)])
@pytest.mark.parametrize("deferred", [False, True])
@torch.inference_mode()
def test_global_rope_write_graph_refreshes_positions_slots_and_padding(
    owner, group, deferred
):
    backend = _backend("cuda", 2)
    tables = _tables("cuda")
    rope_cache = _global_rope_table()
    main = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
    index = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    index_input = (
        dsv41.IndexKeyInput(index, norm_weight, rope_cache, 1e-6) if deferred else index
    )

    def refresh(lengths, live_tables):
        backend.refresh_decode_metadata(
            2,
            len(lengths),
            torch.arange(len(lengths), device="cuda"),
            torch.tensor(lengths, device="cuda", dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables=live_tables,
            num_extends=0,
            for_graph_replay=True,
        )

    refresh([64, 128], tables)
    meta = backend.query_metadata(ForwardMode.DECODE)
    positions = meta.positions if owner == 20 else meta.compressor.pair_positions
    requests = meta.request_indices if owner == 20 else meta.compressor.pair_requests
    main_cache = backend.cache_pool.global_kv(owner)
    index_cache = backend.cache_pool.index_k(owner)

    def run():
        backend.write_global(
            owner,
            main,
            index_input,
            positions,
            requests,
            ForwardMode.DECODE,
            main_rope_cache=rope_cache,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    addresses = [positions.data_ptr(), requests.data_ptr(), main_cache.data_ptr()]
    for step, lengths in enumerate(([64, 128], [65], [130, 66], [], [192, 193])):
        main.normal_()
        index.normal_()
        norm_weight.normal_()
        main_before, index_before = main.clone(), index.clone()
        live_tables = {
            gid: table.flip(0) if step % 2 else table for gid, table in tables.items()
        }
        refresh(lengths, live_tables)
        slots = backend.cache_slots(group, positions, requests, ForwardMode.DECODE)
        main_cache.fill_(165)
        index_cache.fill_(165)
        expected_main, expected_index = _global_write_reference(
            main, index_input, positions, rope_cache, main_cache, index_cache, slots
        )
        graph.replay()
        torch.testing.assert_close(main_cache, expected_main, rtol=0, atol=0)
        torch.testing.assert_close(index_cache, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(main, main_before, rtol=0, atol=0)
        torch.testing.assert_close(index, index_before, rtol=0, atol=0)
        assert [
            positions.data_ptr(),
            requests.data_ptr(),
            main_cache.data_ptr(),
        ] == addresses
        assert torch.all(main_cache[0] == 165) and torch.all(index_cache[0] == 165)


def _dspark_recipe(device):
    recipe = _recipe(device)
    recipe.server_args.speculative_algorithm = "DSPARK"
    recipe.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dspark_num_stages=3)
    )
    return recipe


def _tables_with_packing(device, packing):
    widths = v41_table_widths(512, 0)
    tables = {
        gid: torch.zeros((2, width), dtype=torch.int32, device=device)
        for gid, width in widths.items()
    }
    # Disjoint parent assignments: SWA fits in parents 1..8, R2 starts at 9,
    # R1 at 10 and tails at 11. Groups are mutually exclusive tenants. Each
    # group numbers its own pages with page 0 reserved, so parent n begins at
    # packing * (n - 1) + 1. Derive the IDs from the pool's packing so DSpark
    # can put multiple SWA pages in one parent.
    used = {SWA: 4, R2: widths[R2], R1: widths[R1], TAIL: 100}
    for gid, parent in ((R2, 8), (R1, 9), (TAIL, 10)):
        first = packing[gid] * parent + 1
        assert (
            first + 2 * used[gid] <= packing[gid] * 16
        ), f"{gid} needs {2 * used[gid]} pages from parent {parent}"
    for req in range(2):
        tables[SWA][req, :4] = torch.arange(1 + req * 4, 5 + req * 4, device=device)
        for gid, parent in ((R2, 8), (R1, 9), (TAIL, 10)):
            start = packing[gid] * parent + 1 + req * used[gid]
            tables[gid][req, : used[gid]] = torch.arange(
                start, start + used[gid], device=device
            )
    return tables


@pytest.mark.parametrize(
    "bound, kinds",
    [
        (512, ("all", "all", "all")),
        (513, ("all", "full", "full")),
        (1024, ("all", "full", "full")),
        (1026, ("full", "full", "full")),
        (16384, ("full", "full", "full")),
        (16385, ("full", "full", "candidates")),
        (None, ("full", "full", "candidates")),
    ],
)
def test_index_selection_follows_model_coverage(bound, kinds):
    config = _config("cpu")
    spec = replace(
        config.component(DeepseekV41Config), index_topk=512, candidate_topk=2048
    )
    backend = DeepseekV41AttentionBackend(config, spec)
    assert (
        tuple(backend._index_selection_kind(layer, bound) for layer in (2, 20, 24))
        == kinds
    )


@pytest.mark.parametrize("length", [0, 1, 4, 5, 16, 17])
def test_prefill_selection_skips_only_redundant_work(length):
    backend = _backend("cpu", 2)
    meta = _extend(
        backend, _tables("cpu"), [max(length, 1)], [0], [0], [max(length, 1)]
    )
    if length == 0:
        meta.positions.fill_(-1)
    q = torch.zeros(max(length, 1), 2, 128, dtype=torch.bfloat16)
    weights = torch.ones(max(length, 1), 2, dtype=torch.bfloat16)

    def scored(
        q,
        weights,
        cache,
        table,
        visible,
        candidates,
        topk,
        candidate_topk,
        block_size,
        query_chunk,
        score_chunk,
        group,
        out,
        solution,
        index_metadata,
    ):
        for ids, counts, stride in ((out[0], out[1], 1), (out[2], out[3], 8)):
            values = torch.arange(ids.shape[1]).expand_as(ids)
            ids.copy_(values.masked_fill(values * stride >= visible[:, None], -1))
            counts.copy_((ids >= 0).sum(dim=1))
        return out

    with patch.object(dsv41, "index_topk", side_effect=scored) as topk:
        for layer in (20, 24):
            needs_scores = backend.indexer_needs_scores(
                layer, meta.positions, meta.request_indices, ForwardMode.EXTEND
            )
            assert needs_scores == (length > 4)
            backend.select_global(
                layer,
                q if needs_scores else None,
                weights if needs_scores else None,
                meta.positions,
                meta.request_indices,
                ForwardMode.EXTEND,
                None,
            )
        if length <= 4:
            topk.assert_not_called()
        else:
            assert topk.call_count == 2
            assert topk.call_args_list[0].args[5] is None
            assert (topk.call_args_list[1].args[5] is None) == (length <= 16)
    record = backend.sparse_topk.prefill
    assert record.candidates.lengths[-1].item() == min((length + 7) // 8, 2)
    assert record.lengths[-1].item() == min(length, 4)
    slots, lengths = backend.select_global(
        25, None, None, meta.positions, meta.request_indices, ForwardMode.EXTEND, None
    )
    assert slots is record.physical_slots and lengths is record.lengths


def test_free_query_window_does_not_inherit_canonical_prefill_length():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    meta = _extend(backend, tables, [1], [0], [0], [1])
    assert not backend.indexer_needs_scores(
        20, meta.positions, meta.request_indices, ForwardMode.EXTEND
    )
    assert backend.indexer_needs_scores(
        20, torch.tensor([100]), torch.tensor([0]), ForwardMode.EXTEND
    )
    assert (
        backend._index_history_bound(
            torch.tensor([100]), torch.tensor([0]), ForwardMode.EXTEND
        )
        == tables[R1].shape[1] * 64
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "verify_width,dspark,deferred",
    [(1, False, True), (5, True, True), (5, False, False)],
)
def test_one_index_graph_crosses_coverage_limits_with_padding(
    verify_width, dspark, deferred, monkeypatch
):
    backend = _verify_backend("cuda", 2, verify_width)
    if dspark:
        backend.set_cache_pool(_pool(_dspark_recipe("cuda"), "cuda"))
        backend.init_cuda_graph_state(
            2, max_tokens_per_req=verify_width, overlap_schedule_depth=1
        )
    backend.spec = replace(backend.spec, index_topk=64, candidate_topk=16)
    packing = {
        group.group_id: group.cache_blocks_per_lcm_block
        for group in backend.cache_pool.arena.plan.groups
    }
    tables = _tables_with_packing("cuda", packing)
    sparse_scores = None
    if current_platform().is_blackwell:
        from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm

        sparse_scores = Mock(wraps=deep_gemm._sparse_candidate_scores)
        monkeypatch.setattr(deep_gemm, "_sparse_candidate_scores", sparse_scores)
    cache = backend.cache_pool.index_k(20)
    logical = torch.arange(256, device="cuda")
    slots = tables[R1][0, logical // 64].long() * 64 + logical % 64
    torch.manual_seed(304)
    dsv41.cache_scatter(
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16),
        cache,
        slots,
        backend.index_format,
    )
    index_heads = 32 if current_platform().is_blackwell else 2
    q = torch.randn(
        2 * verify_width, index_heads, 128, device="cuda", dtype=torch.bfloat16
    )
    weights = torch.rand(
        2 * verify_width, index_heads, device="cuda", dtype=torch.bfloat16
    )
    req = torch.zeros(2, dtype=torch.int64, device="cuda")
    seq = torch.zeros(2, dtype=torch.int32, device="cuda")
    rope_cache = _global_rope_table()

    def refresh(length, use_graph):
        seq.fill_(length)
        incoming = dict(tables)
        incoming[R1] = tables[R1][:1, : max(1, (length + 63) // 64)]
        backend.refresh_decode_metadata(
            2,
            int(length > 0),
            req,
            seq,
            forward_mode=ForwardMode.DECODE,
            block_tables=incoming,
            for_graph_replay=use_graph,
        )

    def run():
        backend.begin_indexer_stage()
        meta = backend.query_metadata(ForwardMode.DECODE)
        query = (
            dsv41.IndexQueryInput(q, meta.positions, rope_cache, 0.023456789)
            if deferred
            else q
        )
        for layer in (20, 24, 28):
            assert backend.indexer_needs_scores(
                layer, meta.positions, meta.request_indices, ForwardMode.DECODE
            )
            backend.select_global(
                layer,
                query,
                weights,
                meta.positions,
                meta.request_indices,
                ForwardMode.DECODE,
                None,
            )
        record = backend.sparse_topk.decode
        return (
            record.logical_rows,
            record.lengths,
            record.candidates.block_ids,
            record.candidates.lengths,
        )

    refresh(0, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = run()
    torch.cuda.current_stream().wait_stream(stream)
    for length in (1, 64, 65, 128, 129, 191, 0, 1):
        refresh(length, True)
        graph.replay()
        captured = tuple(t.clone() for t in outputs)
        refresh(length, False)
        for actual, wanted in zip(captured, run(), strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        end = max(length, verify_width)
        live_lengths = (
            [min(i, 64) for i in range(end - verify_width + 1, end + 1)]
            if length
            else [0] * verify_width
        )
        assert captured[1].tolist() == live_lengths + [0] * verify_width
        padding = verify_width if length else 0
        for ids, counts in ((captured[0], captured[1]), (captured[2], captured[3])):
            assert (ids[padding:] == -1).all() and (counts[padding:] == 0).all()
    if sparse_scores is not None:
        assert sparse_scores.call_count > 0


def test_index_geometry_reuse_and_forward_lifetime():
    backend = _backend("cpu", 2)
    tables = _tables("cpu")
    positions = torch.tensor([127, 255])
    requests = torch.tensor([0, 1])

    def prepare(pos, req, group, ratio):
        table = tables[group]
        return backend._selection_geometry(
            pos, req, table, table, ratio, 0, 2, None, ForwardMode.DECODE
        )

    previous = prepare(positions, requests, R2, 2)
    backend.begin_indexer_forward()
    first = prepare(positions, requests, R2, 2)
    assert first is not previous
    assert prepare(positions.clone(), requests, R2, 2) is not first
    assert prepare(positions, requests.flip(0), R2, 2) is not first
    assert prepare(positions, requests, R1, 1) is not first
    for _ in range(3):
        backend.begin_indexer_stage()
        assert prepare(positions, requests, R2, 2) is first
    # Refresh invalidates scratch even inside a full-forward scope.
    backend.refresh_decode_metadata(
        2,
        2,
        requests,
        (positions + 1).to(torch.int32),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        for_graph_replay=False,
    )
    assert not backend._index_geometry
    prepare(positions, requests, R2, 2)
    backend.end_indexer_forward()
    assert not backend._index_geometry
    assert not backend._in_model_forward
    for _ in range(2):
        previous = prepare(positions, requests, R2, 2)
        positions.sub_(32)
        backend.begin_indexer_stage()
        refreshed = prepare(positions, requests, R2, 2)
        assert refreshed is not previous
        torch.testing.assert_close(refreshed.lengths, previous.lengths - 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_index_geometry_scope_preserves_all_indexer_selections_in_graph(monkeypatch):
    backend = _backend("cuda", 2)
    backend.cache_pool.arena.buffer.zero_()
    tables = _tables("cuda")
    requests = torch.tensor([0, 1], device="cuda")
    lengths = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
    backend.refresh_decode_metadata(
        2,
        2,
        requests,
        lengths,
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        for_graph_replay=False,
    )
    meta = backend.query_metadata(ForwardMode.DECODE)
    index_q = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16)
    index_weights = torch.rand(2, 2, device="cuda", dtype=torch.bfloat16)
    selection_table = Mock(wraps=dsv41.selection_table)
    monkeypatch.setattr(dsv41, "selection_table", selection_table)

    def forward(shared):
        outputs = []
        if shared:
            backend.begin_indexer_forward()
        for stage in ((2, 8, 14), (20,), (24, 28, 32, 36)):
            backend.begin_indexer_stage()
            for layer in stage:
                if not shared:
                    backend.begin_indexer_stage()
                outputs.extend(
                    backend.select_global(
                        layer,
                        index_q,
                        index_weights,
                        meta.positions,
                        meta.request_indices,
                        ForwardMode.DECODE,
                        None,
                    )
                )
        if shared:
            backend.end_indexer_forward()
            assert not backend._index_geometry
        return outputs

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(True)
        selection_table.reset_mock()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward(True)
        assert selection_table.call_count == 2
    torch.cuda.current_stream().wait_stream(stream)
    for live, seq_lens in ((2, [129, 257]), (1, [320]), (0, []), (2, [32, 512])):
        index_q.normal_()
        index_weights.uniform_()
        for group in (R2, R1):
            tables[group].copy_(tables[group].roll(1, dims=1))
            tables[group][:, 0] = 0
        backend.refresh_decode_metadata(
            2,
            live,
            requests[:live],
            torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            block_tables={group: table.flip(0) for group, table in tables.items()},
            for_graph_replay=True,
        )
        selection_table.reset_mock()
        expected = forward(False)
        assert selection_table.call_count == 8
        graph.replay()
        for got, want in zip(captured, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,layer,total,history,head_group,expected",
    [
        (ForwardMode.EXTEND, 2, 8192, 65536, None, True),
        (ForwardMode.EXTEND, 8, 8192, 65536, None, True),
        (ForwardMode.EXTEND, 14, 8192, 65536, None, True),
        (ForwardMode.EXTEND, 2, 4095, 65536, None, False),
        (ForwardMode.EXTEND, 2, 8192, 4096, None, False),
        (ForwardMode.EXTEND, 3, 8192, 65536, None, False),
        (ForwardMode.EXTEND, 20, 8192, 65536, None, False),
        (ForwardMode.EXTEND, 24, 8192, 65536, None, False),
        (ForwardMode.DECODE, 2, 8192, 65536, None, False),
        (ForwardMode.MIXED, 2, 8192, 65536, None, False),
        (ForwardMode.EXTEND, 2, 8192, 65536, object(), False),
    ],
)
def test_indexer_projection_plan_preserves_execution_boundaries(
    mode, layer, total, history, head_group, expected
):
    backend = _backend("cpu", 2)
    backend.spec = replace(
        backend.spec, indexer_query_rank=3, indexer_query_group=tuple(range(8))
    )
    # The planner uses host metadata only; no CUDA runtime is needed here.
    positions = SimpleNamespace(is_cuda=True, numel=lambda: total)
    spans = (SimpleNamespace(prefix=history * 2 - total, count=total),)
    with patch.object(backend, "_window", return_value=(None, spans)):
        plan = backend.plan_indexer_queries(layer, positions, None, mode, head_group)
    assert (plan is not None) == expected
    if expected:
        assert plan.local_rows == 1024
        assert plan.local_slice == slice(3, total, 8)


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_projected_indexer_queries_are_not_sliced_twice(projected, deferred):
    from tokenspeed.runtime.layers.attention.query_partition import InterleavedQueryPlan

    backend = _backend("cpu", 2)
    _extend(backend, _tables("cpu"), [5, 4], [20, 20], [0, 0], [25, 24])
    meta = backend.query_metadata(ForwardMode.EXTEND)
    plan = InterleavedQueryPlan(total_rows=9, rank=1, size=2)
    backend.spec = replace(
        backend.spec, indexer_query_rank=1, indexer_query_group=(0, 1)
    )
    q = torch.arange(9, dtype=torch.bfloat16).reshape(9, 1, 1).expand(9, 2, 128)
    weights = torch.ones(9, 2, dtype=torch.bfloat16)
    seen = []

    if deferred:
        q = dsv41.IndexQueryInput(
            q, meta.positions, torch.zeros(25, 64), weight_scale=0.25
        )

    def score(*args):
        query = args[0]
        values = query.values if deferred else query
        seen.extend(values[:, 0, 0].tolist())
        if deferred:
            expected_positions = meta.positions[values[:, 0, 0].long()]
            torch.testing.assert_close(query.positions, expected_positions)
            assert query.weight_scale == 0.25
        rows, lengths, _, _ = args[12]
        rows.fill_(0)
        lengths.fill_(1)

    def gather(self, rows, lengths, group):
        assert self == plan and group == (0, 1)
        assert rows.shape[0] == 4 and lengths.tolist() == [1] * 4
        return torch.zeros(9, rows.shape[1], dtype=torch.int32), torch.ones(
            9, dtype=torch.int32
        )

    with (
        patch.object(backend, "plan_indexer_queries", return_value=plan),
        patch("tokenspeed_kernel.ops.attention.dsv41.index_topk", side_effect=score),
        patch.object(InterleavedQueryPlan, "gather_selection", new=gather),
    ):
        slots, lengths = backend.prepare_global_selection(
            2,
            q[plan.local_slice] if projected else q,
            weights[plan.local_slice] if projected else weights,
            meta.positions,
            meta.request_indices,
            ForwardMode.EXTEND,
            None,
            projected_partition=plan if projected else None,
        )
    assert seen == [1, 3, 5, 7]
    assert slots.shape[0] == 9 and lengths.shape == (9,)
    record = backend.sparse_topk.prefill
    assert record.positions is meta.positions
    assert record.request_indices is meta.request_indices
