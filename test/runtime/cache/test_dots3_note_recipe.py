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

"""CPU recipe integration, using normal imports and the real C++ CapacityModel.

Run with PYTHONPATH=python:tokenspeed-kernel/python. Arena/pool binding and
GPU scatter/attention integration are deliberately outside this recipe suite.
"""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_scheduler import CapacityModel, SchedulerConfig

import tokenspeed.runtime.layers.attention.backends  # noqa: F401
from tokenspeed.runtime.configs.model_config import configure_dots3_note_attention
from tokenspeed.runtime.layers.attention.configs.dots3_note import Dots3NoteAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.dots3_note import (
    Dots3NoteRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheLayout, pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    cache_group_config,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    prepare_cache_setup,
)

PARENT_BYTES = 1_068_800
MTP_PARENT_BYTES = 1_071_360
LAYERS = {
    "full": (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45),
    "swa.0": (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20),
    "swa.1": (22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40),
    "swa.2": (42, 43, 44),
}
PACKING = {"full": 1, "swa.0": 1, "swa.1": 1, "swa.2": 5}


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_CI_SMALL_KV_SIZE", raising=False)
    hf = SimpleNamespace(
        model_type="dots3_note",
        architectures=["Dots3NoteForCausalLM"],
        hidden_size=7168,
        intermediate_size=13824,
        swa_rope_theta=50000,
        num_hidden_layers=46,
        layer_types=tuple(
            "full_attention" if i in LAYERS["full"] else "sliding_attention"
            for i in range(46)
        ),
        num_attention_heads=128,
        num_key_value_heads=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=64,
        swa_kv_lora_rank=1024,
        swa_qk_nope_head_dim=192,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        sliding_window_size=513,
    )
    model = SimpleNamespace(
        hf_config=hf,
        hf_text_config=hf,
        num_attention_layers=46,
        context_len=4096,
        dtype=torch.bfloat16,
    )
    args = SimpleNamespace(
        device="cpu",
        model="dots3-test-checkpoint",
        drafter_attention_backend=None,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        speculative_eagle_topk=1,
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        pipeline_parallel_size=1,
        attention_backend="dots3_note",
        kv_cache_dtype="bfloat16",
        kv_cache_quant_method="none",
        attn_tp_size=8,
        data_parallel_size=1,
        mapping=SimpleNamespace(
            attn=SimpleNamespace(
                tp_size=8,
                dp_size=1,
                dcp_size=1,
                dcp_rank=0,
                dcp_group=(0,),
                qcp_size=1,
                qcp_rank=0,
                qcp_group=(0,),
            )
        ),
        prefix_granularity=64,
        spec_context_pad=0,
        max_num_seqs=2,
        max_total_tokens=2048,
        chunked_prefill_size=512,
        mla_chunk_multiplier=4,
        disaggregation_mode="null",
        enable_prefix_caching=True,
    )
    return dict(
        server_args=args,
        model_config=model,
        attn_config=Dots3NoteAttnConfig.generate(args, model, is_draft=False),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 30,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def configure_mtp(inputs, *, width):
    """Use the production generators; the HF checkpoint stays 46 layers."""
    args, model = inputs["server_args"], inputs["model_config"]
    args.speculative_algorithm = "MTP"
    args.speculative_num_steps = width - 1
    args.speculative_num_draft_tokens = width
    hf = copy(model.hf_config)
    hf.architectures = ["Dots3NoteForCausalLMNextN"]
    draft = SimpleNamespace(
        **(
            vars(model)
            | {"hf_config": hf, "hf_text_config": hf, "num_attention_layers": 1}
        )
    )
    configure_dots3_note_attention(draft, args)
    inputs.update(
        attn_config=Dots3NoteAttnConfig.generate(args, model, is_draft=False),
        draft_model_config=draft,
        draft_attn_config=Dots3NoteAttnConfig.generate(args, draft, is_draft=True),
        decode_input_tokens=width,
    )
    return inputs


@pytest.fixture(params=[2, 4])
def mtp_inputs(inputs, request):
    return configure_mtp(inputs, width=request.param)


def _layout(recipe):
    groups = recipe.groups()
    return pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )


def _capacity_model(inputs, setup):
    """Construct the scheduler independently of the recipe's sizing helper."""
    args, attn = inputs["server_args"], inputs["attn_config"]
    config = SchedulerConfig()
    config.prefix_hash_lookahead_tokens = int(inputs["decode_input_tokens"] > 1)
    config.role = {
        "null": SchedulerConfig.Role.Fused,
        "prefill": SchedulerConfig.Role.P,
        "decode": SchedulerConfig.Role.D,
    }[args.disaggregation_mode]
    config.max_batch_size = attn.max_bs
    config.max_scheduled_tokens = args.chunked_prefill_size
    config.prefix_granularity = attn.prefix_granularity
    config.disable_prefix_cache = not args.enable_prefix_caching
    config.decode_input_tokens = inputs["decode_input_tokens"]
    config.overlap_schedule_depth = inputs["overlap_schedule_depth"]
    plan = setup.spec.memory_plan
    config.cache_groups = [
        cache_group_config(
            spec,
            total_pages=plan.group(spec.group_id).page_count,
            cache_blocks_per_lcm_block=plan.group(
                spec.group_id
            ).cache_blocks_per_lcm_block,
        )
        for spec in setup.spec.cache_group_specs
    ]
    return CapacityModel(config)


@pytest.mark.parametrize("grain", [64, 128, 192])
def test_real_setup_exact_layout_and_null_parent(inputs, grain):
    inputs["attn_config"] = replace(inputs["attn_config"], prefix_granularity=grain)
    setup = prepare_cache_setup(family="dots3_note", **inputs)
    plan = setup.spec.memory_plan
    recipe = Dots3NoteRecipe(**inputs)
    assert setup.spec.family == "dots3_note"
    assert setup.num_target_layers == 46 and setup.num_draft_layers == 0
    assert setup.spec.layer_types == inputs["model_config"].hf_text_config.layer_types
    assert setup.spec.token_capacity == 2048
    assert plan.prefix_granularity == grain
    assert plan.lcm_block_bytes == PARENT_BYTES
    assert [
        (p.plane_id, p.bytes_per_lcm_block, p.arena_offset_bytes) for p in plan.planes
    ] == [("flatkv", PARENT_BYTES, 0)]
    assert {g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups} == PACKING
    assert len(plan.fields) == 59
    assert recipe.max_padding_fraction == 0.025
    assert plan.arena_bytes == (plan.num_lcm_blocks + 1) * PARENT_BYTES
    assert plan.arena_bytes + setup.fixed_workspace_bytes <= setup.cache_budget_bytes
    assert setup.fixed_workspace_bytes == recipe.workspace_bytes()

    for spec, declared_fields in recipe.groups():
        gid = spec.group_id
        rows, width = (64, 576) if gid == "full" else (32, 1088)
        stride = 213_760 if gid == "swa.2" else PARENT_BYTES
        payload = 1_068_288 if gid == "full" else len(LAYERS[gid]) * 32 * 1088 * 2
        assert spec.rows_per_page == spec.block_granularity == rows
        assert spec.entry_stride_tokens == 1 and spec.family == "history"
        assert spec.retention == ("full_history" if gid == "full" else "sliding_window")
        assert spec.sliding_window_tokens == (None if gid == "full" else 513)
        assert not spec.replayable and spec.transfer_policy is None
        assert plan.group(gid).page_count == 1 + plan.num_lcm_blocks * PACKING[gid]
        assert 0 <= (stride - payload) / payload < 0.025
        expected_fields = []
        for layer in LAYERS[gid]:
            names = ("latent_kv", "index_k") if gid == "full" else ("latent_kv",)
            for name in names:
                field_id = f"layer.{layer}.{name}"
                expected_fields.append(field_id)
                # pack orders field IDs lexicographically, not by layer number.
                offset = sum(
                    f.payload_bytes for f in declared_fields if f.field_id < field_id
                )
                field = plan.field(field_id)
                assert field.group_id == gid and field.plane_id == "flatkv"
                assert field.shape == (
                    (rows, 1, width) if name == "latent_kv" else (64, 132)
                )
                assert field.dtype == ("bfloat16" if name == "latent_kv" else "uint8")
                assert field.page_stride_bytes == stride
                assert field.field_offset_bytes == offset
                null = plan.field_page_byte_offset(field_id, 0)
                assert null == PARENT_BYTES - stride + offset
                assert null + field.payload_bytes <= PARENT_BYTES
                assert plan.field_page_byte_offset(field_id, 1) == PARENT_BYTES + offset
                last = plan.group(gid).page_count - 1
                assert (
                    plan.field_page_byte_offset(field_id, last) + field.payload_bytes
                    <= plan.arena_bytes
                )
                if gid == "swa.2":
                    assert (
                        plan.field_page_byte_offset(field_id, 5)
                        == PARENT_BYTES + 4 * stride + offset
                    )
                    assert (
                        plan.field_page_byte_offset(field_id, 6)
                        == 2 * PARENT_BYTES + offset
                    )
        assert [f.field_id for f in declared_fields] == expected_fields
        assert all(
            not f.exact_page_stride and f.page_stride_alignment_bytes == 256
            for f in declared_fields
        )
        assert sum(f.payload_bytes for f in declared_fields) == payload


def test_group_assignment_follows_config_not_a_second_layer_map(inputs):
    attn = inputs["attn_config"]
    spec = attn.component(Dots3NoteAttnConfig)
    labels = tuple(reversed(spec.cache_layer_types))
    inputs["attn_config"] = replace(
        attn, components=(replace(spec, cache_layer_types=labels),)
    )
    setup = prepare_cache_setup(family="dots3_note", **inputs)
    sliding = 0
    for layer, kind in enumerate(labels):
        gid = "full" if kind == "full_attention" else f"swa.{sliding // 15}"
        assert setup.spec.memory_plan.field(f"layer.{layer}.latent_kv").group_id == gid
        sliding += kind == "sliding_attention"


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
@pytest.mark.parametrize(
    "batch,chunk,overlap,prefix_cache,grain",
    [
        (1, 64, 0, True, 64),
        (8, 512, 1, True, 64),
        (4, 128, 1, False, 128),
    ],
)
def test_capacity_is_scheduler_inverse(
    inputs, role, batch, chunk, overlap, prefix_cache, grain
):
    args = inputs["server_args"]
    args.max_total_tokens = None
    args.disaggregation_mode = role
    args.chunked_prefill_size = chunk
    args.enable_prefix_caching = prefix_cache
    # max_num_seqs is global; the recipe must use rank-local attn.max_bs.
    args.max_num_seqs = batch * 8
    inputs["attn_config"] = replace(
        inputs["attn_config"],
        max_bs=batch,
        prefix_granularity=grain,
        pd_disaggregation_enabled=role != "null",
    )
    inputs["overlap_schedule_depth"] = overlap
    budgeted = 2048
    workspace = Dots3NoteRecipe(**inputs).workspace_bytes()
    inputs["cache_budget_bytes"] = (
        workspace + (budgeted + 1) * PARENT_BYTES + PARENT_BYTES - 1
    )
    setup = prepare_cache_setup(family="dots3_note", **inputs)
    model = _capacity_model(inputs, setup)

    def needed(tokens):
        return model.lcm_blocks_needed_for(
            model.concurrent_group_pages(
                max_total_tokens=tokens,
                max_context_len=inputs["attn_config"].context_len,
            )
        )

    capacity = setup.spec.token_capacity
    assert needed(capacity) == setup.spec.memory_plan.num_lcm_blocks <= budgeted
    assert needed(capacity + 1) > budgeted
    assert (
        setup.spec.memory_plan.arena_bytes + workspace <= inputs["cache_budget_bytes"]
    )
    assert all(
        spec.transfer_policy == (None if role == "null" else "full_suffix")
        for spec in setup.spec.cache_group_specs
    )
    assert all(not spec.replayable for spec in setup.spec.cache_group_specs)


def test_concurrency_cost_is_not_just_window_pages(inputs):
    capacities = []
    for batch in (1, 8, 32):
        args = inputs["server_args"]
        args.max_total_tokens = None
        inputs["attn_config"] = replace(inputs["attn_config"], max_bs=batch)
        recipe = Dots3NoteRecipe(**inputs)
        inputs["cache_budget_bytes"] = recipe.workspace_bytes() + 2049 * PARENT_BYTES
        setup = prepare_cache_setup(family="dots3_note", **inputs)
        capacities.append(setup.spec.token_capacity)
    assert capacities[0] > capacities[1] > capacities[2]


@pytest.mark.parametrize("limit", [1, 63, 64, 65, 1025])
def test_token_limit_is_not_rounded_to_a_page(inputs, limit):
    inputs["server_args"].max_total_tokens = limit
    setup = prepare_cache_setup(family="dots3_note", **inputs)
    assert setup.spec.token_capacity == limit
    recipe = Dots3NoteRecipe(**inputs)
    assert setup.spec.memory_plan.num_lcm_blocks == recipe.parents_needed(
        _layout(recipe), limit
    )


def test_ci_token_limit_and_null_parent_budget(inputs, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_CI_SMALL_KV_SIZE", "65")
    recipe = Dots3NoteRecipe(**inputs)
    layout = _layout(recipe)
    parents = recipe.parents_needed(layout, 65)
    workspace = recipe.workspace_bytes()
    inputs["cache_budget_bytes"] = workspace + (parents + 1) * PARENT_BYTES
    setup = prepare_cache_setup(family="dots3_note", **inputs)
    assert setup.spec.token_capacity == 65
    assert (
        setup.spec.memory_plan.arena_bytes + workspace == inputs["cache_budget_bytes"]
    )
    inputs["cache_budget_bytes"] -= 1
    smaller = prepare_cache_setup(family="dots3_note", **inputs)
    assert smaller.spec.token_capacity < 65


@pytest.mark.parametrize("usable_parents", [0, 1])
def test_insufficient_budget_fails_before_allocation(inputs, usable_parents):
    recipe = Dots3NoteRecipe(**inputs)
    inputs["cache_budget_bytes"] = (
        recipe.workspace_bytes() + (usable_parents + 1) * PARENT_BYTES
    )
    with pytest.raises(ValueError, match="null parent|cannot admit one token"):
        prepare_cache_setup(family="dots3_note", **inputs)


@pytest.mark.parametrize("grain", [0, -64, 32, 96])
def test_rejects_invalid_prefix_grain(inputs, grain):
    inputs["attn_config"] = replace(inputs["attn_config"], prefix_granularity=grain)
    with pytest.raises(ValueError, match="positive multiple of 64"):
        prepare_cache_setup(family="dots3_note", **inputs)


@pytest.mark.parametrize(
    "leaf,changes",
    [
        ("full", {"num_attention_heads": 64}),
        ("full", {"num_kv_heads": 64}),
        ("full", {"kv_lora_rank": 256}),
        ("full", {"qk_nope_head_dim": 64}),
        ("full", {"qk_rope_head_dim": 32}),
        ("full", {"v_head_dim": 64}),
        ("full", {"head_dim": 256}),
        ("full", {"kv_cache_dim": 512}),
        ("full", {"index_head_dim": 256}),
        ("full", {"index_n_heads": 32}),
        ("full", {"index_topk": 1024}),
        ("full", {"index_kpool": 2}),
        ("full", {"sliding_window_tokens": 513}),
        ("swa", {"num_attention_heads": 128}),
        ("swa", {"num_kv_heads": 128}),
        ("swa", {"kv_lora_rank": 512}),
        ("swa", {"qk_nope_head_dim": 128}),
        ("swa", {"qk_rope_head_dim": 32}),
        ("swa", {"v_head_dim": 64}),
        ("swa", {"head_dim": 192}),
        ("swa", {"kv_cache_dim": 576}),
        ("swa", {"sliding_window_tokens": 512}),
        ("swa", {"attn_tp_size": 4}),
        ("swa", {"backend_name": "flashmla"}),
    ],
)
def test_rejects_unsupported_leaf_geometry(inputs, leaf, changes):
    attn = inputs["attn_config"]
    spec = attn.component(Dots3NoteAttnConfig)
    changed = replace(spec.full if leaf == "full" else spec.swa, **changes)
    inputs["attn_config"] = replace(
        attn, components=(replace(spec, **{leaf: changed}),)
    )
    with pytest.raises(ValueError, match="dots3"):
        prepare_cache_setup(family="dots3_note", **inputs)


@pytest.mark.parametrize(
    "changes",
    [
        {"cache_layer_types": ("full_attention",) * 46},
        {"cache_layer_types": ("sliding_attention",) * 45},
        {"sliding_window_tokens": 512},
        {"attn_tp_size": 3},
        {"attn_tp_size": 0},
    ],
)
def test_rejects_unsupported_component_geometry(inputs, changes):
    attn = inputs["attn_config"]
    spec = replace(attn.component(Dots3NoteAttnConfig), **changes)
    inputs["attn_config"] = replace(attn, components=(spec,))
    with pytest.raises(ValueError, match="dots3"):
        prepare_cache_setup(family="dots3_note", **inputs)


@pytest.mark.parametrize(
    "changes",
    [
        {"kv_cache_dtype": torch.float8_e4m3fn},
        {"dtype": torch.float16},
        {"kv_cache_mxfp8": True},
        {"kv_cache_quant_method": "per_token_head"},
        {"kernel_page_size": 64},
    ],
)
def test_rejects_incompatible_storage(inputs, changes):
    inputs["attn_config"] = replace(inputs["attn_config"], **changes)
    with pytest.raises(ValueError, match="dots3"):
        prepare_cache_setup(family="dots3_note", **inputs)


@pytest.mark.parametrize(
    "mode",
    [
        "draft_model",
        "draft_attn",
        "speculation",
        "draft_path",
        "pp",
        "decode_width",
        "draft_config",
    ],
)
def test_rejects_unsupported_serving_modes(inputs, mode):
    if mode == "draft_model":
        inputs["draft_model_config"] = SimpleNamespace(num_attention_layers=0)
    elif mode == "draft_attn":
        inputs["draft_attn_config"] = inputs["attn_config"]
    elif mode == "speculation":
        inputs["server_args"].speculative_algorithm = "DSPARK"
    elif mode == "draft_path":
        inputs["server_args"].speculative_draft_model_path = "unused"
    elif mode == "pp":
        inputs["server_args"].pipeline_parallel_size = 2
    elif mode == "decode_width":
        inputs["decode_input_tokens"] = 2
    else:
        inputs["attn_config"] = replace(inputs["attn_config"], is_draft=True)
    with pytest.raises(NotImplementedError, match="dots3"):
        prepare_cache_setup(family="dots3_note", **inputs)


def test_check_layout_rejects_extra_plane(inputs):
    recipe = Dots3NoteRecipe(**inputs)
    layout = _layout(recipe)
    assert isinstance(layout, CacheLayout)
    with pytest.raises(ValueError, match="one 1,068,800-byte"):
        recipe.check_layout(
            replace(layout, plane_bytes=(("flatkv", PARENT_BYTES), ("extra", 256)))
        )


def test_workspace_tracks_actual_bounded_buffers(inputs):
    # Q=512, B=2, C=4096, TP8. SWA gathers at most 512+256 rows,
    # not all prefill queries times their 513-key windows.
    recipe = Dots3NoteRecipe(**inputs)
    q, b, c = 512, 2, 4096
    tables = 4 * b * (4 * 128 + 64 + 3 * 128)
    metadata = 2 * (tables + 4 * (b * 32 + q * 8 + 4))
    metadata += b * c * 64 + 4 * 4 * (6 * b + 1) * 4
    index_queries = q * (64 * (128 * 13 + 24) + 128 * 16)
    logits = q * b * c * 4
    scores = 2 * logits + logits // 256 + q * 72
    selections = q * (2048 * 27 + 64)
    full_attention = q * 2 * (576 + 16 * (192 + 2 * 512 + 576 + 3 * 128))
    swa_attention = q * 2 * (1088 + 8 * (256 + 3 * 128)) + b * 8 * (2 * 1024 + 1088) * 2
    swa_tile = 2 * 768 * ((1088 + 1024) * 2 + 8 * (192 + 2 * 128 + 256) * 2 + 32)
    swa_tile += 256 * 8 * (256 + 128) * 2 + 16
    assert recipe.workspace_bytes() == (
        metadata
        + index_queries
        + scores
        + selections
        + full_attention
        + swa_attention
        + swa_tile
    )
    inputs["cache_budget_bytes"] *= 8
    inputs["server_args"].max_total_tokens = None
    assert Dots3NoteRecipe(**inputs).workspace_bytes() == recipe.workspace_bytes()


def test_workspace_covers_uncapped_decode_scores_and_one_prefill_row(inputs):
    inputs["attn_config"] = replace(
        inputs["attn_config"], max_bs=64, context_len=1_048_577
    )
    recipe = Dots3NoteRecipe(**inputs)
    rounded_context = (1_048_577 + 63) // 64 * 64
    # Decode is not query-tiled. A single prefill score row over all request
    # histories can itself exceed the nominal 64-MiB score tile limit.
    assert recipe.workspace_bytes() > 2 * 64 * rounded_context * 4 > 64 << 20


def test_mtp_fifth_group_exact_packing_and_target_fields(mtp_inputs):
    setup = prepare_cache_setup(family="dots3_note", **mtp_inputs)
    recipe = Dots3NoteRecipe(**mtp_inputs)
    plan = setup.spec.memory_plan
    assert (setup.num_target_layers, setup.num_draft_layers) == (46, 1)
    assert len(setup.spec.layer_types) == 47
    assert setup.spec.layer_types[-1] == "sliding_attention"
    assert len(plan.fields) == 60
    assert plan.lcm_block_bytes == MTP_PARENT_BYTES == PARENT_BYTES + 2560
    assert {
        g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups
    } == PACKING | {"draft.swa": 15}
    assert plan.arena_bytes == (plan.num_lcm_blocks + 1) * MTP_PARENT_BYTES
    assert plan.arena_bytes + setup.fixed_workspace_bytes <= setup.cache_budget_bytes
    assert recipe.max_padding_fraction == 0.026
    draft = mtp_inputs["draft_attn_config"]
    assert type(draft.components[0]) is Dots3NoteAttnConfig
    assert draft.components[0].num_attention_heads == 64
    assert draft.components[0].head_dim == 256
    assert draft.components[0].swa == mtp_inputs["attn_config"].components[0].swa
    assert draft.kernel_page_size == 32 and draft.is_draft
    assert draft.components[0].cache_layer_types == ("sliding_attention",)
    assert mtp_inputs["draft_model_config"].hf_text_config.num_hidden_layers == 46
    assert len(mtp_inputs["draft_model_config"].hf_text_config.layer_types) == 46
    for group, fields in recipe.groups():
        packing = plan.group(group.group_id).cache_blocks_per_lcm_block
        stride = MTP_PARENT_BYTES // packing
        payload = sum(f.payload_bytes for f in fields)
        assert stride % 256 == 0
        assert (stride - payload) / payload < 0.026
        assert group.sliding_window_tokens == (
            None if group.group_id == "full" else 513
        )
        assert not group.replayable
        layers = (46,) if group.group_id == "draft.swa" else LAYERS[group.group_id]
        names = ("latent_kv", "index_k") if group.group_id == "full" else ("latent_kv",)
        assert {f.field_id for f in fields} == {
            f"layer.{i}.{name}" for i in layers for name in names
        }
        for field in fields:
            planned = plan.field(field.field_id)
            assert planned.page_stride_bytes == stride
            assert planned.field_offset_bytes == sum(
                f.payload_bytes for f in fields if f.field_id < field.field_id
            )
    field = plan.field("layer.46.latent_kv")
    assert (field.group_id, field.shape, field.dtype) == (
        "draft.swa",
        (32, 1, 1088),
        "bfloat16",
    )
    assert field.page_stride_bytes == 71_424
    assert plan.field_page_byte_offset(field.field_id, 0) == MTP_PARENT_BYTES - 71_424
    assert (
        plan.field_page_byte_offset(field.field_id, 15)
        == MTP_PARENT_BYTES + 14 * 71_424
    )
    assert plan.field_page_byte_offset(field.field_id, 16) == 2 * MTP_PARENT_BYTES


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
@pytest.mark.parametrize("overlap", [0, 1])
def test_mtp_capacity_reserves_use_the_scheduler(mtp_inputs, role, overlap):
    args = mtp_inputs["server_args"]
    args.disaggregation_mode = role
    args.max_total_tokens = None
    args.max_num_seqs = 8
    args.chunked_prefill_size = 64
    configure_mtp(mtp_inputs, width=args.speculative_num_draft_tokens)
    mtp_inputs["overlap_schedule_depth"] = overlap
    recipe = Dots3NoteRecipe(**mtp_inputs)
    mtp_inputs["cache_budget_bytes"] = (
        recipe.workspace_bytes() + 1025 * MTP_PARENT_BYTES
    )
    setup = prepare_cache_setup(family="dots3_note", **mtp_inputs)
    model = _capacity_model(mtp_inputs, setup)
    capacity = setup.spec.token_capacity
    for tokens in (capacity, capacity + 1):
        pages = model.concurrent_group_pages(
            max_total_tokens=tokens,
            max_context_len=mtp_inputs["attn_config"].context_len,
        )
        required = model.lcm_blocks_needed_for(pages)
        assert (required <= 1024) == (tokens == capacity)
    assert all(
        g.transfer_policy == (None if role == "null" else "full_suffix")
        for g in setup.spec.cache_group_specs
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("kernel_page_size", 64),
        ("is_draft", False),
        ("kv_cache_dtype", torch.float8_e4m3fn),
        ("speculative_num_draft_tokens", 8),
        ("context_len", 128),
    ],
)
def test_mtp_rejects_mismatched_draft_attention(mtp_inputs, field, value):
    mtp_inputs["draft_attn_config"] = replace(
        mtp_inputs["draft_attn_config"], **{field: value}
    )
    with pytest.raises(ValueError, match="matching BF16 page32 SWA MLA"):
        prepare_cache_setup(family="dots3_note", **mtp_inputs)


@pytest.mark.parametrize("missing", ["draft_model_config", "draft_attn_config"])
def test_mtp_requires_actual_draft_config(mtp_inputs, missing):
    mtp_inputs[missing] = None
    with pytest.raises(ValueError, match="actual one-layer"):
        prepare_cache_setup(family="dots3_note", **mtp_inputs)


def test_mtp_rejects_wrong_draft_layer_count(mtp_inputs):
    mtp_inputs["draft_model_config"].num_attention_layers = 46
    with pytest.raises(ValueError, match="actual one-layer"):
        prepare_cache_setup(family="dots3_note", **mtp_inputs)


def test_mtp_workspace_sizes_verify_queries_and_draft_temporaries(mtp_inputs):
    args = mtp_inputs["server_args"]
    args.max_num_seqs = 64
    args.chunked_prefill_size = 1
    mtp_inputs["model_config"].context_len = 131073
    configure_mtp(mtp_inputs, width=args.speculative_num_draft_tokens)
    workspace = Dots3NoteRecipe(**mtp_inputs).workspace_bytes()
    queries = 64 * args.speculative_num_draft_tokens
    rounded_context = (131073 + 63) // 64 * 64
    # Both uncapped decode score tiles, index query/selection rows and the
    # independent draft's fusion/dense temporaries must fit, even above chunk size.
    minimum = 2 * queries * rounded_context * 4
    minimum += queries * (64 * (128 * 13 + 24) + 128 * 16 + 2048 * 27 + 64)
    minimum += queries * 2 * (8 * 7168 + 3 * 13824)
    assert workspace > minimum
    mtp_inputs["cache_budget_bytes"] *= 4
    assert Dots3NoteRecipe(**mtp_inputs).workspace_bytes() == workspace
