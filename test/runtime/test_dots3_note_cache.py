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

"""Dots3-note cache geometry, retention and arena ownership."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack

_DSA_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45)
_LAYER_TYPES = tuple(
    "full_attention" if i in _DSA_LAYERS else "sliding_attention" for i in range(46)
)


def _recipe(*, draft_layers: int, pd: bool):
    from tokenspeed.runtime.layers.attention.configs.dots3_note import (
        Dots3NoteAttentionConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.dots3_note import (
        Dots3NoteRecipe,
    )

    swa = MLAConfig(
        backend_name="mla",
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=256,
        attn_tp_size=8,
        kv_lora_rank=1024,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=256**-0.5,
        kv_cache_dim=1088,
        sliding_window_tokens=513,
    )
    dsa = Dots3NoteAttentionConfig(
        backend_name="dsa",
        num_attention_heads=128,
        num_kv_heads=128,
        head_dim=192,
        attn_tp_size=8,
        cache_layer_types=_LAYER_TYPES,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=64,
        index_k_format="fp8_scaled",
        swa=swa,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_quant_method="none",
        context_len=8192,
        max_bs=2,
        pd_disaggregation_enabled=pd,
        components=(dsa,),
    )
    draft_config = (
        replace(
            config,
            is_draft=True,
            kernel_page_size=32,
            components=(replace(swa, cache_layer_types=("sliding_attention",)),),
        )
        if draft_layers
        else None
    )
    return Dots3NoteRecipe(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            max_total_tokens=None,
            chunked_prefill_size=512,
            disaggregation_mode="prefill" if pd else "null",
            enable_prefix_caching=True,
            speculative_algorithm="MTP" if draft_layers else None,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(num_attention_layers=46),
        attn_config=config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=draft_layers) if draft_layers else None
        ),
        draft_attn_config=draft_config,
        cache_budget_bytes=256 << 20,
        probe_batch_rows=None,
        decode_input_tokens=4 if draft_layers else 1,
        overlap_schedule_depth=0,
    )


def _layout(recipe):
    return pack(
        recipe.groups(),
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(recipe.groups()),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )


@pytest.mark.parametrize("draft_layers,parent_bytes", [(0, 1_178_112), (1, 1_183_744)])
@pytest.mark.parametrize("pd", [False, True])
def test_history_and_draft_fields_share_the_planned_arena(
    draft_layers, parent_bytes, pd
):
    recipe = _recipe(draft_layers=draft_layers, pd=pd)
    groups = {spec.group_id: spec for spec, _ in recipe.groups()}
    assert len(groups) == 2
    assert groups["dsa"].block_granularity == 64
    assert groups["dsa"].attn_kind == "full_history"
    assert groups["swa"].block_granularity == 32
    assert groups["swa"].attn_kind == "sliding_window"
    assert groups["swa"].sliding_window_tokens == 513
    assert not any(spec.replayable for spec in groups.values())
    if pd:
        assert all(spec.transfer_policy == "full_suffix" for spec in groups.values())

    layout = _layout(recipe)
    assert layout.lcm_block_bytes == parent_bytes
    assert dict(layout.group_packing) == {"dsa": 2, "swa": 1}
    plan = layout.bind(4)
    arena = CacheArena(plan, "cpu", cache_group_specs=tuple(groups.values()))
    assert arena.buffer.numel() == 5 * parent_bytes
    for field in plan.fields:
        view = arena.field(field.field_id)
        assert view.untyped_storage().data_ptr() == arena.buffer.data_ptr()
        assert view.stride(0) == field.page_stride_bytes
        assert field.page_stride_bytes % 256 == 0

    # DSA pages 1/2 occupy parent 1; SWA page 2 occupies parent 2.
    # Distinct fields and separate parents must survive each other's writes.
    dsa_kv = arena.field("layer.0.latent_kv")
    index_k = arena.field("layer.0.index_k")
    swa_kv = arena.field("layer.2.latent_kv")
    dsa_kv[1].fill_(11)
    dsa_kv[2].fill_(12)
    index_k[1].fill_(13)
    swa_kv[2].fill_(14)
    assert torch.all(dsa_kv[1] == 11)
    assert torch.all(dsa_kv[2] == 12)
    assert torch.all(index_k[1] == 13)
    assert torch.all(swa_kv[2] == 14)
    assert not dsa_kv.is_contiguous()

    if pd:
        from tokenspeed.runtime.pd.cache_protocol import (
            build_arena_cache_transfer_contract,
        )

        contract, base_addr = build_arena_cache_transfer_contract(arena)
        for field in contract.plan.fields:
            # Each field lands where its page 0 sits in the one arena.
            assert contract.field_addresses[field.field_id] == base_addr + (
                contract.plan.field_page_byte_offset(field.field_id, 0)
            )
            for page_id in (0, 1, 2):
                offset = contract.plan.field_page_byte_offset(field.field_id, page_id)
                assert (
                    base_addr + offset
                    == arena.field(field.field_id)[page_id].data_ptr()
                )
    if draft_layers:
        draft = arena.field("layer.46.latent_kv")
        draft[2].fill_(15)
        assert plan.field("layer.46.latent_kv").group_id == "swa"
        assert torch.all(swa_kv[2] == 14)


def test_capacity_includes_window_and_inflight_pages():
    recipe = _recipe(draft_layers=1, pd=False)
    layout = _layout(recipe)
    setup = recipe.setup()
    parents = setup.spec.memory_plan.num_lcm_blocks
    capacity = setup.spec.token_capacity
    assert setup.spec.memory_plan.arena_bytes <= recipe.cache_budget_bytes
    assert recipe.parents_needed(layout, capacity) <= parents
    assert recipe.parents_needed(layout, capacity + 1) > parents
    wider = _recipe(draft_layers=1, pd=False)
    wider.decode_input_tokens = 65
    wider.overlap_schedule_depth = 1
    assert wider.parents_needed(layout, 8192) > recipe.parents_needed(layout, 8192)


def test_unsupported_cache_encoding_is_rejected_before_allocation():
    recipe = _recipe(draft_layers=0, pd=False)
    recipe.attn_config = replace(
        recipe.attn_config, kv_cache_quant_method="per_token_head"
    )
    with pytest.raises(ValueError, match="unit-scale"):
        recipe.groups()


def test_prefix_grain_holds_whole_history_pages_without_flattening_them():
    recipe = _recipe(draft_layers=0, pd=False)
    spans = [spec.block_granularity for spec, _ in recipe.groups()]
    assert spans == [64, 32]
    # A grain equal to a field's page size would request a flat view.
    assert recipe.prefix_granularity == 128
    assert all(recipe.prefix_granularity % span == 0 for span in spans)


def test_pool_views_keep_layer_geometry_and_draft_ownership():
    from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import (
        Dots3NoteTokenToKVPool,
    )

    recipe = _recipe(draft_layers=1, pd=True)
    setup = recipe.setup()
    arena = CacheArena(
        _layout(recipe).bind(2), "cpu", cache_group_specs=setup.spec.cache_group_specs
    )
    target = Dots3NoteTokenToKVPool(
        arena,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method="none",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=46,
        rank=0,
        index_head_dim=128,
    )
    draft = Dots3NoteTokenToKVPool(
        arena,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method="none",
        kv_lora_rank=1024,
        qk_rope_head_dim=64,
        layer_num=1,
        rank=0,
        index_head_dim=128,
        field_layer_offset=46,
    )
    assert target.get_key_buffer(0).shape[1:] == (64, 1, 576)
    assert target.get_value_buffer(0).shape[-1] == 512
    assert target.get_key_buffer(2).shape[1:] == (32, 1, 1088)
    assert target.get_value_buffer(2).shape[-1] == 1024
    assert draft.get_value_buffer(0).shape[-1] == 1024
    assert draft.history_group_by_layer() == {0: "swa"}
    assert target.get_kv_size_bytes() == arena.buffer.nbytes > 0
    assert draft.get_kv_size_bytes() == target.get_kv_size_bytes()
    slots = torch.tensor([32, 63])
    assert (
        target.kv_write_target(2, slots, None).kv_cache.data_ptr()
        == arena.field("layer.2.latent_kv").data_ptr()
    )
    assert (
        draft.kv_write_target(0, slots, None).kv_cache.data_ptr()
        == arena.field("layer.46.latent_kv").data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_registered_pool_and_router_bind_distinct_leaf_geometries(monkeypatch):
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
    from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
    from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import cache_recipe
    from tokenspeed.runtime.layers.attention.registry import create_paged_router
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "dsa_slot_order", "selection")
    recipe = _recipe(draft_layers=1, pd=False)
    assert type(cache_recipe("dots3_note", **vars(recipe))) is type(recipe)
    recipe.attn_config = replace(recipe.attn_config, device="cuda")
    setup = recipe.setup()
    arena = CacheArena(
        _layout(recipe).bind(2), "cuda", cache_group_specs=setup.spec.cache_group_specs
    )
    pool = create_cache_pool(
        setup.spec, recipe.attn_config, arena, num_layers=46, rank=0
    )
    router = create_paged_router(recipe.attn_config, AttentionArch.DSA)
    router.set_cache_pool(pool)
    assert isinstance(router.leaves["dsa"], DSABackend)
    assert isinstance(router.leaves["swa"], MLAAttnBackend)
    assert router.leaves["dsa"].kernel_solution == "triton"
    # The dots override must not change the default used by other DSA models.
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
        TRTLLMMLABackend,
    )

    standard = DSABackend(
        recipe.attn_config, recipe.attn_config.components[0], kernel_page_size=64
    )
    assert isinstance(standard._dense_backend, TRTLLMMLABackend)
    assert router.leaves["dsa"].kernel_page_size == 64
    assert router.leaves["swa"].kernel_page_size == 32
    assert router.leaves["dsa"].kv_lora_rank == 512
    assert router.leaves["swa"].kv_lora_rank == 1024


def test_pd_prefill_records_once_after_latent_and_index_writes():
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.dots3_note import (
        create_dots3_note_router,
    )
    from tokenspeed.runtime.models.dots3_note import Dots3NoteAttention

    router = create_dots3_note_router(_recipe(draft_layers=0, pd=True).attn_config)
    router.leaves = {
        group: SimpleNamespace(step_counter=None) for group in ("dsa", "swa")
    }
    events = []
    router.register_step_counter(
        SimpleNamespace(record_cache=lambda: events.append("ready"))
    )
    attention = Dots3NoteAttention.__new__(Dots3NoteAttention)
    torch.nn.Module.__init__(attention)
    rows = torch.ones(1, 4)

    for leaf in router.leaves.values():

        def prefill(*args):
            events.append("cache_write")
            if leaf.step_counter is not None:
                leaf.step_counter.record_cache()
            events.append("attention_done")
            return rows

        attention._forward = prefill
        ctx = SimpleNamespace(attn_backend=router, forward_mode=ForwardMode.EXTEND)
        attention(torch.arange(1), rows, ctx, None)
        assert events == ["cache_write", "attention_done", "ready"]
        events.clear()


def test_router_rejects_a_global_page_override():
    from tokenspeed.runtime.layers.attention.backends.dots3_note import (
        create_dots3_note_router,
    )

    config = _recipe(draft_layers=0, pd=False).attn_config
    with pytest.raises(ValueError, match="kernel_page_size"):
        create_dots3_note_router(replace(config, kernel_page_size=128))


def test_scheduler_reuses_both_history_groups():
    import tokenspeed_scheduler as ts

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
        cache_group_config,
    )

    recipe = _recipe(draft_layers=0, pd=False)
    arena = CacheArena(
        _layout(recipe).bind(128),
        "cpu",
        cache_group_specs=tuple(spec for spec, _ in recipe.groups()),
    )
    contract = arena.runtime_contract
    config = ts.SchedulerConfig()
    config.prefix_granularity = contract.prefix_granularity
    config.num_device_pages = contract.num_lcm_blocks + 1
    config.num_host_pages = 0
    config.num_snapshot_pages = 1  # The null page alone; this test does not retract.
    config.max_scheduled_tokens = 1024
    config.max_batch_size = 2
    config.enable_l3_storage = False
    config.disable_l2_cache = True
    config.disable_prefix_cache = False
    config.overlap_schedule_depth = 0
    config.cache_groups = [
        cache_group_config(
            spec,
            total_pages=contract.virtual_block_counts[spec.group_id],
            cache_blocks_per_lcm_block=contract.virtual_packing[spec.group_id],
        )
        for spec in contract.group_specs
    ]
    scheduler = ts.Scheduler(config)

    def submit(request_id, tokens):
        request = ts.RequestSpec()
        request.request_id = request_id
        request.tokens = tokens
        scheduler.submit_requests([request])
        return next(
            batch
            for batch in scheduler.next_execution_plan().forward
            if dict(batch.block_tables)
        )

    def advance(event, request_id, tokens=None):
        event.request_id = request_id
        if tokens is not None:
            event.tokens = tokens
        execution = ts.ExecutionEvent()
        execution.add_event(event)
        scheduler.advance(execution)

    tokens = list(range(1, 770))
    first = submit("cold", tokens)
    assert list(first.extend_prefix_lens) == [0]
    cold_pages = {
        group: list(rows[0]) for group, rows in dict(first.block_tables).items()
    }
    advance(ts.ForwardEvent.ExtendResult(), "cold", [9001])
    scheduler.next_execution_plan()
    advance(ts.ForwardEvent.ExtendResult(), "cold", [9002])
    advance(ts.ForwardEvent.Finish(), "cold")
    scheduler.next_execution_plan()

    hit = submit("warm", tokens[:768] + [10001])
    assert list(hit.extend_prefix_lens) == [768]
    assert list(hit.extend_replay_lens) == [0]
    assert list(hit.input_ids) == [10001]
    warm_pages = dict(hit.block_tables)
    assert list(warm_pages["dsa"][0][:12]) == cold_pages["dsa"][:12]
    # Query 768 needs the preceding 512 tokens: SWA pages [8, 24).
    assert all(page > 0 for page in cold_pages["swa"][8:24])
    assert list(warm_pages["swa"][0][8:24]) == cold_pages["swa"][8:24]


def test_draft_must_use_the_swa_page_size():
    recipe = _recipe(draft_layers=1, pd=False)
    recipe.draft_attn_config = replace(recipe.draft_attn_config, kernel_page_size=64)
    with pytest.raises(ValueError, match="MTP.*SWA"):
        recipe.groups()


@pytest.mark.parametrize("component", ["dsa", "swa", "draft"])
def test_equal_cache_width_cannot_hide_a_different_rope_split(component):
    recipe = _recipe(draft_layers=1, pd=False)
    dsa = recipe.attn_config.components[0]
    spec = {
        "dsa": dsa,
        "swa": dsa.swa,
        "draft": recipe.draft_attn_config.components[0],
    }[component]
    wrong = replace(spec, kv_lora_rank=spec.kv_lora_rank - 64, qk_rope_head_dim=128)
    if component == "draft":
        recipe.draft_attn_config = replace(
            recipe.draft_attn_config, components=(wrong,)
        )
    else:
        recipe.attn_config = replace(
            recipe.attn_config,
            components=(wrong if component == "dsa" else replace(dsa, swa=wrong),),
        )
    with pytest.raises(ValueError, match="geometry"):
        recipe.groups()


@pytest.fixture
def dots_runtime(monkeypatch):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "dsa_slot_order", "selection")
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 128)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
    from tokenspeed.runtime.layers.attention.registry import create_paged_router

    recipe = _recipe(draft_layers=0, pd=False)
    recipe.attn_config = replace(recipe.attn_config, device="cuda", max_bs=3)
    setup = recipe.setup()
    arena = CacheArena(
        _layout(recipe).bind(40),
        "cuda",
        cache_group_specs=setup.spec.cache_group_specs,
    )
    pool = create_cache_pool(
        setup.spec, recipe.attn_config, arena, num_layers=46, rank=0
    )
    router = create_paged_router(recipe.attn_config, AttentionArch.DSA)
    router.set_cache_pool(pool)
    return router, pool


@pytest.fixture
def swa_leaf(dots_runtime):
    return dots_runtime[0].leaves["swa"]


def test_swa_prefill_gathers_only_live_window_pages(swa_leaf):
    leaf = swa_leaf
    prefix_cpu = torch.tensor([4097, 45, 0], dtype=torch.int32)
    extend_cpu = torch.tensor([33, 1, 514], dtype=torch.int32)
    prefix, extend = prefix_cpu.cuda(), extend_cpu.cuda()
    table = torch.zeros((3, 256), device="cuda", dtype=torch.int32)
    table[0, 112:129] = torch.arange(1, 18, device="cuda", dtype=torch.int32)
    table[1, :2] = torch.tensor([18, 19], device="cuda", dtype=torch.int32)
    leaf._init_prefill_metadata(
        prefix + extend, prefix, prefix_cpu, extend, extend_cpu, table
    )
    meta = leaf.chunked_prefill_metadata
    # The partial first page is retained; only its pre-window token is masked.
    assert meta.extend_prefix_lens_cpu.tolist() == [513, 45, 0]
    assert meta.prefix_query_start_offsets[0].cpu().tolist() == [513, 45, 0]
    actual = [[], [], []]
    for i, slots in enumerate(meta.chunk_kv_indices_list):
        starts = meta.cu_chunked_seq_len[i].cpu().tolist()
        for request in range(3):
            actual[request].extend(
                slots[starts[request] : starts[request + 1]].cpu().tolist()
            )
    assert actual[0] == list(range(32, 32 + 513))
    assert actual[1] == list(range(18 * 32, 18 * 32 + 45))
    assert actual[2] == []


def test_swa_leaf_cold_prefill_masks_the_left_boundary(swa_leaf):
    leaf = swa_leaf
    length, heads = 514, 8
    lens_cpu = torch.tensor([length], dtype=torch.int32)
    zeros_cpu = torch.tensor([0], dtype=torch.int32)
    leaf._init_prefill_metadata(
        lens_cpu.cuda(),
        zeros_cpu.cuda(),
        zeros_cpu,
        lens_cpu.cuda(),
        lens_cpu,
        torch.zeros((1, 256), device="cuda", dtype=torch.int32),
    )
    q = torch.zeros(length, heads, 256, device="cuda", dtype=torch.bfloat16)
    k, v = torch.zeros_like(q), torch.zeros(
        length, heads, 128, device="cuda", dtype=torch.bfloat16
    )
    q[..., 0] = 8
    k[0, :, 0] = 8
    v[0] = 2
    layer = SimpleNamespace(
        tp_q_head_num=heads,
        tp_k_head_num=heads,
        tp_v_head_num=heads,
        qk_head_dim=256,
        v_head_dim=128,
        scaling=1 / 16,
        logit_cap=0.0,
        sliding_window_size=512,
        layer_id=2,
    )
    out = leaf.forward_extend(q, k, v, layer, None, None, 1).view(length, heads, 128)
    assert torch.count_nonzero(out[-1]).item() == 0
    assert torch.all(out[-2] > 0.1)


def test_swa_leaf_prefix_chunks_require_and_use_query_offsets(swa_leaf):
    leaf = swa_leaf
    q = torch.ones(3, 8, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.ones(4, 8, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.ones(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    args = dict(
        cum_seq_lens_q=cu_q,
        cum_seq_lens_kv=cu_kv,
        max_q_len=3,
        max_kv_len=4,
        seq_lens=cu_kv[1:],
        batch_size=1,
        causal=False,
    )
    with pytest.raises(ValueError, match="query_start_offsets"):
        leaf.forward_extend_chunked(q, k, v, 1 / 16, **args)
    out, lse = leaf.forward_extend_chunked(
        q,
        k,
        v,
        1 / 16,
        **args,
        query_start_offsets=torch.tensor([516], device="cuda", dtype=torch.int32),
    )
    assert torch.count_nonzero(out).item() == 0
    assert torch.isneginf(lse).all()


def test_swa_prefix_lengths_do_not_recompile_metadata(swa_leaf):
    from unittest.mock import patch

    from tokenspeed.runtime.layers.attention.chunk import (
        create_chunked_cache_kv_indices_paged,
    )

    table = torch.ones((3, 256), device="cuda", dtype=torch.int32)
    extend_cpu = torch.tensor([3, 1, 4], dtype=torch.int32)
    extend = extend_cpu.cuda()

    def initialize(prefix_lengths):
        prefix_cpu = torch.tensor(prefix_lengths, dtype=torch.int32)
        prefix = prefix_cpu.cuda()
        swa_leaf._init_prefill_metadata(
            prefix + extend, prefix, prefix_cpu, extend, extend_cpu, table
        )

    initialize([4097, 45, 0])
    kernel = create_chunked_cache_kv_indices_paged
    with patch.object(kernel, "_do_compile", wraps=kernel._do_compile) as compiles:
        initialize([32, 63, 0])
    assert compiles.call_count == 0, "SWA page-table width must be fixed for the model"


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_dsa_dense_paths_on_compact_fp8_cache(dots_runtime, mode):
    router, pool = dots_runtime
    leaf = router.leaves["dsa"]
    torch.manual_seed(17)
    lengths = [1, 65, 513]
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    scale = 192**-0.5
    if mode == "decode":
        cache = pool.get_key_buffer(0)
        cache.copy_((torch.randn(cache.shape, device="cuda") * 0.2).to(cache.dtype))
        table = torch.zeros((3, 128), device="cuda", dtype=torch.int32)
        table[:, :9] = torch.arange(2, 20, 2, device="cuda", dtype=torch.int32)
        q = (torch.randn(3, 16, 576, device="cuda") * 0.2).to(cache.dtype)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=16,
            head_dim=576,
            v_head_dim=512,
            scaling=scale,
            logit_cap=0.0,
            sliding_window_size=-1,
        )
        leaf.init_cuda_graph_state(3)
        leaf.refresh_decode_metadata(3, 3, lens, table)
        out = leaf.forward_decode(q, None, None, layer, None, pool, 3)
        expected = []
        for batch, length in enumerate(lengths):
            rows = cache[table[batch, : (length + 63) // 64].long()]
            rows = rows.float().reshape(-1, 576)[:length]
            probs = torch.softmax(q[batch].float() @ rows.T * scale, dim=-1)
            expected.append(probs @ rows[:, :512])
        expected = torch.stack(expected)
        assert not cache.is_contiguous()
        assert cache.untyped_storage().data_ptr() == pool.arena.buffer.data_ptr()
    else:
        q_lens = [1, 3, 33]
        cu_q = torch.tensor([0, 1, 4, 37], device="cuda", dtype=torch.int32)
        cu_kv = torch.tensor([0, 1, 66, 579], device="cuda", dtype=torch.int32)
        q = torch.randn(37, 16, 192, device="cuda", dtype=torch.bfloat16) * 0.2
        k = torch.randn(579, 16, 192, device="cuda", dtype=torch.bfloat16) * 0.2
        v = torch.randn(579, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        out, lse = leaf.forward_extend_chunked(
            q,
            k,
            v,
            scale,
            None,
            cum_seq_lens_q=cu_q,
            cum_seq_lens_kv=cu_kv,
            max_q_len=33,
            max_kv_len=513,
            seq_lens=lens,
            batch_size=3,
            causal=True,
        )
        expected = []
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lens, lengths):
            qs = q[q_start : q_start + q_len].float().transpose(0, 1)
            ks = k[kv_start : kv_start + kv_len].float().transpose(0, 1)
            vs = v[kv_start : kv_start + kv_len].float().transpose(0, 1)
            scores = qs @ ks.transpose(1, 2) * scale
            positions = torch.arange(q_len, device="cuda") + kv_len - q_len
            mask = torch.arange(kv_len, device="cuda")[None, :] <= positions[:, None]
            probs = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
            expected.append((probs @ vs).transpose(0, 1))
            q_start += q_len
            kv_start += kv_len
        expected = torch.cat(expected)
        assert torch.isfinite(lse).all()
    torch.testing.assert_close(
        out.reshape_as(expected).float(), expected, atol=0.008, rtol=0.008
    )
