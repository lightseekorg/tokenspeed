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

"""Synthetic decode allocation uses real scheduler/cache geometry, without a GPU."""

import pytest
from tokenspeed_scheduler import AttnKind, CacheGroupConfig, SchedulerConfig

from tokenspeed.runtime.execution.simulation import allocate_simulated_decode


def configuration(lengths, width, pages):
    config = SchedulerConfig()
    config.prefix_granularity = 128
    config.num_device_pages = pages
    config.num_host_pages = 1
    config.num_snapshot_pages = 1
    config.max_scheduled_tokens = 8192
    config.max_batch_size = len(lengths)
    config.decode_input_tokens = width
    config.overlap_schedule_depth = 0
    config.disable_prefix_cache = True
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full",
            block_granularity=64,
            total_pages=2 * pages,
            attn_kind=AttnKind.Full,
            cache_blocks_per_lcm_block=2,
        ),
        CacheGroupConfig(
            group_id="compressed",
            block_granularity=128,
            total_pages=pages,
            attn_kind=AttnKind.Full,
        ),
        CacheGroupConfig(
            group_id="window",
            block_granularity=64,
            total_pages=2 * pages,
            attn_kind=AttnKind.SlidingWindow,
            sliding_window_tokens=128,
            cache_blocks_per_lcm_block=2,
        ),
        CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=pages,
            attn_kind=AttnKind.State,
        ),
    ]
    return config


@pytest.mark.parametrize("lengths", [(129,), (127, 128, 129, 4097), (250000,) * 4])
@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("host_pages", [0, 1])
def test_fixture_retains_request_lengths_and_cache_groups(lengths, width, host_pages):
    pages = 2 * sum((n + width + 127) // 128 for n in lengths) + 512
    cfg = configuration(lengths, width, pages)
    cfg.num_host_pages = host_pages
    prompts = [[index + 1] * length for index, length in enumerate(lengths)]
    tokens = list(range(31, 31 + len(lengths)))
    fixture = allocate_simulated_decode(cfg, prompts, tokens, max_new_tokens=100)
    batch = fixture.decode_plan.forward[0]
    assert batch.num_extends() == 0
    assert batch.input_lengths == [width] * len(lengths)
    assert fixture.context_lengths == tuple(lengths)
    assert fixture.next_token_ids == tuple(tokens)
    assert len(set(batch.request_pool_indices)) == len(lengths)
    assert set(batch.block_tables) == {"full", "compressed", "window", "state"}
    assert fixture.scheduler.decoding_size() == len(lengths)


def test_fixture_reports_insufficient_capacity_instead_of_partial_batch():
    lengths = [4096] * 4
    cfg = configuration(lengths, 1, 180)
    with pytest.raises(RuntimeError, match="capacity"):
        allocate_simulated_decode(
            cfg,
            [[i + 1] * n for i, n in enumerate(lengths)],
            [1] * 4,
            max_new_tokens=100,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dense_and_sliding_history_need_no_index_keys(device):
    from test.runtime.cache_pool_test_utils import (
        make_layer_group_ids,
        make_mha_memory_plan,
        make_pool,
        specs_for_layers,
    )

    import torch

    from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups
    from tokenspeed.runtime.execution.simulation import seed_simulated_index_history
    from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an admitted GPU")
    layers = ("full_attention", "sliding_attention")
    plan = make_mha_memory_plan(
        size=2048,
        prefix_granularity=64,
        layer_num=2,
        kv_heads=2,
        head_dim=32,
        dtype=torch.bfloat16,
        layer_types=layers,
        sliding_window_tokens=128,
    )
    groups = make_layer_group_ids(
        layer_num=2, layer_types=layers, sliding_window_tokens=128
    )
    arena, pool = make_pool(
        MHATokenToKVPool,
        plan,
        device=device,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=32,
        layer_num=2,
        rank=0,
        cache_group_specs=specs_for_layers(
            layer_types=layers,
            group_ids=groups,
            prefix_granularity=64,
            sliding_window_tokens=128,
        ),
    )
    cfg = configuration([129, 257], 1, arena.plan.num_lcm_blocks + 1)
    cfg.prefix_granularity = 64
    cfg.cache_groups = pool_to_cache_groups(pool)
    fixture = allocate_simulated_decode(
        cfg, [[1] * 129, [2] * 257], [3, 4], max_new_tokens=100
    )
    arena.clear()
    assert (
        seed_simulated_index_history(
            pool, fixture, index_format=None, seed=42, rows_per_chunk=128
        )
        == 0
    )
    assert arena.buffer.eq(0).all()
    batch = fixture.decode_plan.forward[0]
    assert set(batch.block_tables) == {
        spec.group_id for spec in arena.cache_group_specs
    }
    for layer in range(2):
        assert torch.isfinite(pool.get_key_buffer(layer)).all()
        assert torch.isfinite(pool.get_value_buffer(layer)).all()


def test_sparse_mha_history_is_not_treated_as_dense():
    from test.runtime.cache_pool_test_utils import make_pool, plan_fields

    import torch

    from tokenspeed.runtime.execution.simulation import seed_simulated_index_history
    from tokenspeed.runtime.layers.attention.kv_cache.msa import MSATokenToKVPool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheFieldSpec

    plan = plan_fields(
        {
            "full": tuple(
                CacheFieldSpec(
                    f"layer.0.{plane}", f"unit.0.{plane}", (64, 1, 32), "bfloat16"
                )
                for plane in ("k", "v", "index_k")
            )
        },
        prefix_granularity=64,
        num_lcm_blocks=4,
    )
    _, pool = make_pool(
        MSATokenToKVPool,
        plan,
        device="cpu",
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=32,
        layer_num=1,
        rank=0,
    )
    assert (0, "index_k") in pool.layer_fields
    with pytest.raises(NotImplementedError, match="MHA sparse index"):
        seed_simulated_index_history(
            pool, None, index_format=None, seed=42, rows_per_chunk=128
        )


@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("token_limit", [32, 64])
def test_fixture_seeds_real_runtime_state_and_engram_history(
    width, device, token_limit
):
    import torch

    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.execution.types import RequestHistorySeeds, StopCriteria

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an admitted GPU")
    prompts = [[1], list(range(1, 18))]
    cfg = configuration([len(p) for p in prompts], width, 512)
    fixture = allocate_simulated_decode(cfg, prompts, [19, 23], max_new_tokens=100)
    batch = fixture.decode_plan.forward[0]
    states = RuntimeStates(3, 64, 64, width, device=device)
    states.future_input_map.fill_(63)
    states.init_ngram_state(3)
    states.init_request_token_history(32)
    states.init_draft_request_token_history(32)
    states.init_draft_trees(width)
    candidates = tuple(
        tuple(token + i for i in range(width)) for token in fixture.next_token_ids
    )
    seeds = RequestHistorySeeds(
        slots=tuple(batch.request_pool_indices),
        prefix_lengths=fixture.context_lengths,
        tokens=fixture.prompt_tokens,
    )
    states.bind_stop_criteria(
        batch.request_ids,
        batch.request_pool_indices,
        [StopCriteria((), 32)] * len(prompts),
    )
    states.seed_simulated_decode(
        seeds, candidates, request_ids=tuple(batch.request_ids)
    )
    states.bind_stop_criteria(
        batch.request_ids,
        batch.request_pool_indices,
        [StopCriteria((), token_limit)] * len(prompts),
    )
    assert states.token_limits[list(seeds.slots)].eq(token_limit).all()
    for i, slot in enumerate(seeds.slots):
        n = fixture.context_lengths[i]
        assert int(states.valid_cache_lengths[slot]) == n
        assert states.future_input_map[slot].tolist() == list(candidates[i])
        assert bool(states.spec_candidates_ready[slot]) == (width > 1)
        assert states.request_token_history_ids[slot, :n].tolist() == prompts[i]
        assert (
            states.draft_request_token_history_ids[slot, : n - 1].tolist()
            == prompts[i][1:]
        )
        assert states.ngram_request_ids[slot] == batch.request_ids[i]
        assert not states.ngram_needs_seed[slot]
        assert torch.equal(states.future_parent_map[slot], states.chain_parents)
    assert states.ngram_accepted_tokens[seeds.slots[0]].tolist() == [1, -1, -1]
    assert states.ngram_accepted_tokens[seeds.slots[1]].tolist() == [17, 16, 15]
    assert states.future_input_map[[0, -1]].eq(63).all()
    assert states.valid_cache_lengths[[0, -1]].eq(0).all()
    snapshot = fixture.ngram_inputs(3)
    assert snapshot.tokens == ((19, 1, -1, -1), (23, 17, 16, 15))
    assert snapshot.positions == (1, 17)


def test_synthetic_state_rejects_invalid_candidates_before_mutating_rows():
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.execution.types import RequestHistorySeeds

    states = RuntimeStates(2, 32, 32, 6, device="cpu")
    states.valid_cache_lengths.fill_(9)
    seeds = RequestHistorySeeds(slots=(1,), prefix_lengths=(1,), tokens=((7,),))
    with pytest.raises(ValueError, match="decode width"):
        states.seed_simulated_decode(seeds, ((1,),), request_ids=("request",))
    with pytest.raises(ValueError, match="vocabulary"):
        states.seed_simulated_decode(
            seeds, ((1, 2, 3, 4, 5, 32),), request_ids=("request",)
        )
    assert states.valid_cache_lengths.eq(9).all()


@pytest.mark.parametrize("family", ["kimi_fp8", "kimi_bf16", "deepseek"])
def test_simulated_index_history_uses_real_cache_codecs(family):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires an admitted GPU")
    from test.runtime.test_deepseek_v41_cache import _pool, _recipe
    from test.runtime.test_kimi_k31 import _k31_pool

    from tokenspeed_kernel.ops.attention.dsv41 import cache_gather

    from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups
    from tokenspeed.runtime.execution.simulation import seed_simulated_index_history
    from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
        DeepseekV41Config,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.dsa import split_index_k_rows

    if family == "deepseek":
        recipe = _recipe("cuda")
        pool = _pool(recipe, "cuda")
        encoding = (
            recipe.attn_config.component(DeepseekV41Config)
            .row_layout()
            .index_kernel_format
        )
    else:
        pool, _ = _k31_pool(
            "cuda",
            num_lcm_blocks=16,
            index_k_format="bf16" if family == "kimi_bf16" else "fp8_scaled",
        )
        encoding = None
    arena = pool.arena
    cfg = configuration([129, 257], 1, arena.plan.num_lcm_blocks + 1)
    cfg.prefix_granularity = arena.runtime_contract.prefix_granularity
    cfg.cache_groups = pool_to_cache_groups(pool)
    fixture = allocate_simulated_decode(
        cfg, [[1] * 129, [2] * 257], [3, 4], max_new_tokens=100
    )
    arena.clear()
    count = seed_simulated_index_history(
        pool, fixture, index_format=encoding, seed=42, rows_per_chunk=128
    )
    assert count > 0
    tables = fixture.decode_plan.forward[0].block_tables
    untouched = arena.buffer.clone()
    for (layer, plane), field_id in pool.layer_fields.items():
        field = arena.plan.field(field_id)
        view = arena.field_pages(field_id)
        pages = sorted({p for row in tables[field.group_id] for p in row if p > 0})
        if plane != "index_k":
            # Unallocated views can alias another group's live LCM storage.
            # Only the scheduler's live pages belong to this field.
            assert not view[pages].any(), field_id
            continue
        assert not view[0].any()
        for page in pages:
            start = arena.plan.field_page_byte_offset(field_id, page)
            untouched[start : start + field.payload_bytes].zero_()
        slots = (
            torch.tensor(pages, device="cuda")[:, None] * field.shape[0]
            + torch.arange(field.shape[0], device="cuda")
        ).flatten()
        if family == "deepseek":
            values = cache_gather(view, slots, encoding, None).float()
        else:
            fmt = "bf16" if family == "kimi_bf16" else "fp8_scaled"
            packed = pool.gather_index_k_rows(layer, slots, index_k_format=fmt)
            values, scales = split_index_k_rows(
                packed, index_head_dim=128, index_k_format=fmt
            )
            if scales is not None:
                values = values.view(torch.float8_e4m3fn).float()
                values *= scales.repeat_interleave(128, dim=-1)
            else:
                values = values.float()
        assert torch.isfinite(values).all()
        assert values.std().item() > 0.2
        assert not torch.equal(values[0], values[1])
    # Check every byte outside the intended index writes, including padding
    # and unallocated storage, without interpreting overlapping field views.
    assert not untouched.any()


@pytest.mark.parametrize("prefix_granularity", [64, 128])
@pytest.mark.parametrize("width", [1, 6])
def test_fixed_decode_snapshot_restores_cache_and_slot_state_without_copying_prefix(
    prefix_granularity,
    width,
):
    import torch

    from tokenspeed.runtime.engine.scheduler_utils import cache_group_config
    from tokenspeed.runtime.execution.forward_benchmark import FixedDecodeState
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.execution.types import RequestHistorySeeds
    from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldSpec,
        pack,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    groups = (
        CacheGroupSpec(
            "full",
            "full_history",
            rows_per_page=64,
            entry_stride_tokens=1,
            replayable=False,
        ),
        CacheGroupSpec(
            "window",
            "sliding_window",
            rows_per_page=64,
            entry_stride_tokens=1,
            sliding_window_tokens=128,
            replayable=False,
        ),
        CacheGroupSpec("state", "state", replayable=False),
    )
    layout = pack(
        tuple(
            (
                spec,
                (CacheFieldSpec(f"layer.{i}.kv", f"layer.{i}.kv", (64, 8), "float32"),),
            )
            for i, spec in enumerate(groups)
        ),
        prefix_granularity=prefix_granularity,
        cache_blocks_per_lcm_block={
            spec.group_id: 1 if spec.attn_kind == "state" else prefix_granularity // 64
            for spec in groups
        },
        alignment=256,
        max_padding_fraction=4.0,
    )
    arena = CacheArena(
        layout.bind(128),
        "cpu",
        cache_group_specs=groups,
        token_capacity=128 * prefix_granularity,
    )
    config = configuration([129, 1024], width, 129)
    config.max_scheduled_tokens = 256
    config.prefix_granularity = prefix_granularity
    config.cache_groups = [
        cache_group_config(
            spec,
            total_pages=arena.cache_group_page_counts[spec.group_id],
            cache_blocks_per_lcm_block=(
                1 if spec.attn_kind == "state" else prefix_granularity // 64
            ),
        )
        for spec in groups
    ]
    fixture = allocate_simulated_decode(
        config, [[1] * 129, [2] * 1024], [3, 4], max_new_tokens=1024
    )
    batch = fixture.decode_plan.forward[0]
    states = RuntimeStates(3, 32, 32, width, device="cpu")
    states.init_request_token_history(2048)
    states.init_ngram_state(3)
    states.seed_simulated_decode(
        RequestHistorySeeds(
            slots=tuple(batch.request_pool_indices),
            prefix_lengths=fixture.context_lengths,
            tokens=fixture.prompt_tokens,
        ),
        [[3] * width, [4] * width],
        request_ids=tuple(batch.request_ids),
    )
    arena.buffer.fill_(1)
    snapshot = FixedDecodeState(arena, states, (states,), batch, stream=None)
    assert snapshot.nbytes < arena.buffer.nbytes // 2
    # A decode can change the tail/state pages and token-derived rows.
    arena.buffer.fill_(2)
    states.valid_cache_lengths.fill_(17)
    states.future_input_map.fill_(19)
    states.ngram_accepted_tokens.fill_(21)
    states.request_token_history_ids.fill_(22)
    states.ngram_request_ids[:] = [None] * len(states.ngram_request_ids)
    snapshot.restore()
    for field in arena.plan.fields:
        view = arena.field_pages(field.field_id).view(torch.uint8)
        for row, prefix in zip(
            batch.block_tables[field.group_id], fixture.context_lengths, strict=True
        ):
            if field.group_id == "full":
                # Derive the actual target/draft write pages from positions,
                # independently of the number of pre-reserved output pages.
                written = {
                    row[position // 64]
                    for position in range(prefix - width, prefix + 2 * width)
                }
                assert row[-1] not in written
            else:
                written = {page for page in row if page > 0}
            for page in written:
                assert view[page].eq(1).all()
        assert view[0].eq(2).all()
    first_full = next(p for p in batch.block_tables["full"][0] if p > 0)
    assert arena.field_pages("layer.0.kv")[first_full].view(torch.uint8).eq(2).all()
    for i, slot in enumerate(batch.request_pool_indices):
        assert states.valid_cache_lengths[slot].item() == fixture.context_lengths[i]
        assert (
            states.future_input_map[slot].tolist()
            == [fixture.next_token_ids[i]] * width
        )
        assert states.ngram_accepted_tokens[slot].tolist() == [i + 1] * 3
        assert (
            states.request_token_history_ids[slot, : fixture.context_lengths[i]]
            .eq(i + 1)
            .all()
        )
        assert states.ngram_request_ids[slot] == batch.request_ids[i]


@pytest.mark.parametrize("request_token_history", [False, True])
def test_direct_driver_builds_chunk_boundary_and_decode_history(request_token_history):
    from test.forward_benchmark.bench_decode import _plan

    from tokenspeed_scheduler import RequestSpec, Scheduler

    prompt = list(range(1, 18))
    config = configuration([len(prompt)], 6, 512)
    scheduler = Scheduler(config)
    request = RequestSpec()
    request.request_id, request.tokens, request.max_new_tokens = "input", prompt, 100
    scheduler.submit_requests([request])
    prefill = scheduler.next_execution_plan().forward[0]
    planned = _plan(
        prefill,
        {"input": prompt},
        {},
        request_token_history=request_token_history,
        ngram_width=3,
        max_new_tokens=100,
    )
    assert planned.ngram_inputs.tokens == ((1, -1, -1, -1),)
    assert planned.ngram_inputs.positions == (0,)
    if request_token_history:
        assert planned.request_history_seeds.tokens == ((),)
    else:
        assert planned.request_history_seeds is None
    assert planned.stop_criteria[0].token_limit == 117
    fixture = allocate_simulated_decode(config, [prompt], [19], max_new_tokens=100)
    decode = fixture.decode_plan.forward[0]
    rid = decode.request_ids[0]
    planned = _plan(
        decode,
        {rid: prompt},
        {rid: 19},
        request_token_history=request_token_history,
        ngram_width=3,
        max_new_tokens=100,
    )
    assert planned.ngram_inputs.tokens == ((19, 17, 16, 15),)
    assert planned.ngram_inputs.positions == (17,)
    assert planned.request_history_seeds is None


def test_direct_driver_validates_token_corpus_before_device_initialization(tmp_path):
    import json
    from test.forward_benchmark.bench_decode import _read_tokens

    path = tmp_path / "tokens.json"
    prompts = [list(range(10)), list(reversed(range(10)))]
    path.write_text(json.dumps(prompts))
    limits = dict(
        batches=[1, 2],
        contexts=[3, 8],
        vocab_size=10,
        max_batch_size=2,
        max_context=16,
        width=6,
    )
    assert _read_tokens(path, **limits) == prompts
    with pytest.raises(ValueError, match="capacity"):
        _read_tokens(path, **{**limits, "max_context": 13})
    for invalid in ([[0] * 8] * 2, [[0] * 8 + [10]] * 2, [[False] * 9] * 2):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="invalid token IDs"):
            _read_tokens(path, **limits)
