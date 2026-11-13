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

"""Qwen side-cache planning does not require recurrent-attention layers."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    qsa_compressed_field,
    qsa_raw_key_field,
    qsa_rope_position_field,
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen4_exp import (
    Qwen4ExpRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    CacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.registry import (
    _resolve_attn_side,
    _resolve_cache_family,
)


def _recipe(
    *, layer_types: tuple[str, ...], speculative: bool, width: int
) -> Qwen4ExpRecipe:
    text_config = SimpleNamespace(
        ple_layer_ids=(1, 2),
        short_conv_layer_ids=(0, 1),
        ngram_context_len=2,
        short_conv_state_shape=(4, 3),
        indexer_n_heads=1,
        indexer_head_dim=8,
        indexer_compress_ratio=4,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config),
        hf_text_config=text_config,
        num_attention_layers=len(layer_types),
    )
    spec = MHAConfig(
        backend_name="qsa",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        cache_layer_types=layer_types,
        sliding_window_tokens=None,
    )
    config = AttnConfig(
        # Planning allocates no device tensors. CUDA plus enabled replay makes
        # this exercise the missing-GDN gate before any kernel capability probe.
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=64,
        context_len=1024,
        max_bs=2,
        pd_disaggregation_enabled=False,
        speculative_num_steps=2 if speculative else 0,
        speculative_num_draft_tokens=width,
        is_draft=False,
        draft_block_decode=False,
        components=(spec,),
    )
    draft_config = (
        replace(
            config,
            is_draft=True,
            components=(replace(spec, cache_layer_types=(FULL_ATTENTION,)),),
        )
        if speculative
        else None
    )
    return Qwen4ExpRecipe(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            block_size=64,
            max_total_tokens=None,
            speculative_num_draft_tokens=width,
            speculative_eagle_topk=1,
            speculative_draft_tree=False,
            enable_replay_ssm=True,
            mamba_ssm_dtype="float32",
            enable_qsa_prefill_bf16_scratch=False,
            qwen4_exp_flat_kv=False,
        ),
        model_config=model_config,
        attn_config=config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=1, hf_text_config=text_config)
            if speculative
            else None
        ),
        draft_attn_config=draft_config,
        cache_budget_bytes=1 << 20,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForCausalLMNextN",
        "Qwen3_5ForConditionalGeneration",
    ],
)
@pytest.mark.parametrize("indexer_n_heads", [None, 1])
@pytest.mark.parametrize("has_gdn", [False, True])
def test_cache_family_preserves_qwen4_without_indexer_or_gdn(
    architecture, indexer_n_heads, has_gdn
) -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[architecture],
            text_config=SimpleNamespace(indexer_n_heads=indexer_n_heads),
        ),
        model_profile=None,
    )
    profile = _resolve_attn_side(model_config, None)
    config = _recipe(
        layer_types=(FULL_ATTENTION,), speculative=False, width=1
    ).attn_config
    if has_gdn:
        config = replace(
            config,
            components=(
                replace(
                    config.component(MHAConfig),
                    cache_layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
                ),
                LinearAttnConfig(
                    num_k_heads=1,
                    num_v_heads=1,
                    head_k_dim=2,
                    head_v_dim=2,
                    conv_kernel_size=4,
                    layer_ids=(0,),
                    tp_size=1,
                ),
            ),
        )
    is_qwen4 = architecture.startswith("Qwen4Exp")
    assert profile.is_qwen4_exp is is_qwen4
    assert profile.is_qsa is (is_qwen4 and indexer_n_heads is not None)
    expected_family = "qwen4_exp" if is_qwen4 else "qwen_gdn" if has_gdn else "mha"
    assert _resolve_cache_family(profile, config) == expected_family


@pytest.mark.parametrize("speculative,width", [(False, 1), (True, 1), (True, 3)])
def test_qwen4_full_attention_keeps_ple_and_qsa_without_gdn(speculative, width) -> None:
    recipe = _recipe(
        layer_types=(FULL_ATTENTION, FULL_ATTENTION),
        speculative=speculative,
        width=width,
    )
    assert not recipe.replay_ssm
    setup = recipe.setup()
    assert setup.spec.family == "qwen4_exp"
    assert {spec.group_id for spec in setup.spec.cache_group_specs} == {
        FULL_ATTENTION,
        QWEN4_EXP_PLE_CACHE_GROUP,
        QWEN4_EXP_QSA_CACHE_GROUP,
        QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    }
    fields = {field.field_id for field in setup.spec.memory_plan.fields}
    for layer_id in (0, 1):
        assert f"layer.{layer_id}.k" in fields
        assert f"layer.{layer_id}.v" in fields
        assert qwen4_exp_ple_conv_field(layer_id) in fields
        assert qsa_raw_key_field(layer_id) in fields
        assert qsa_rope_position_field(layer_id) in fields
        assert qsa_compressed_field(layer_id) in fields
    assert qwen4_exp_ple_context_field(0) in fields
    assert not any(field.endswith(".ssm") for field in fields)
    if speculative:
        assert qsa_raw_key_field(2) in fields
    if speculative and width > 1:
        # PLE: eight 64-byte rows plus 64 bytes of commit indices. QSA:
        # two layers' 192-byte keys, 216 shared bytes and 32 address bytes.
        assert setup.fixed_workspace_bytes == 576 + 440
    else:
        assert setup.fixed_workspace_bytes == 0


@pytest.mark.parametrize(
    "group_rows,entry_stride,field_rows",
    [(128, 4, 128), (64, 8, 64), (64, 4, 128)],
)
def test_qwen4_qsa_rejects_geometry_outside_model_page(
    monkeypatch, group_rows, entry_stride, field_rows
) -> None:
    recipe = _recipe(layer_types=(FULL_ATTENTION,), speculative=False, width=1)
    compressed, recent = recipe._qsa_fields()
    monkeypatch.setattr(
        recipe,
        "_qsa_fields",
        lambda: (
            tuple(
                replace(field, shape=(field_rows, *field.shape[1:]))
                for field in compressed
            ),
            recent,
        ),
    )

    def group_spec(**kwargs):
        spec = CacheGroupSpec(**kwargs)
        if spec.group_id == QWEN4_EXP_QSA_CACHE_GROUP:
            spec = replace(
                spec, rows_per_page=group_rows, entry_stride_tokens=entry_stride
            )
        return spec

    monkeypatch.setattr(
        "tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen4_exp.CacheGroupSpec",
        group_spec,
    )
    with pytest.raises(ValueError, match="must match one model page"):
        recipe.setup()


def test_qwen4_state_layer_still_requires_linear_geometry() -> None:
    with pytest.raises(ValueError, match="linear-attention component"):
        _recipe(
            layer_types=(LINEAR_ATTENTION, FULL_ATTENTION), speculative=False, width=1
        )


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("has_gdn", [False, True])
@pytest.mark.parametrize("ssm_dtype", ["float32", "bfloat16"])
def test_qwen4_side_caches_preserve_kv_capacity(
    speculative: bool, has_gdn: bool, ssm_dtype: str
) -> None:
    recipe = _recipe(
        layer_types=(FULL_ATTENTION, FULL_ATTENTION),
        speculative=speculative,
        width=1,
    )
    recipe.server_args.mamba_ssm_dtype = ssm_dtype
    if has_gdn:
        layer_types = (LINEAR_ATTENTION, FULL_ATTENTION) * 2
        recipe.model_config.num_attention_layers = len(layer_types)
        components = (
            replace(
                recipe.attn_config.component(MHAConfig), cache_layer_types=layer_types
            ),
            LinearAttnConfig(
                num_k_heads=1,
                num_v_heads=1,
                head_k_dim=32,
                head_v_dim=32,
                conv_kernel_size=4,
                layer_ids=(0, 2),
                tp_size=1,
            ),
        )
        recipe = Qwen4ExpRecipe(
            server_args=recipe.server_args,
            model_config=recipe.model_config,
            attn_config=replace(recipe.attn_config, components=components),
            draft_model_config=recipe.draft_model_config,
            draft_attn_config=recipe.draft_attn_config,
            cache_budget_bytes=recipe.cache_budget_bytes,
            probe_batch_rows=None,
            decode_input_tokens=1,
            overlap_schedule_depth=0,
        )
    # Keep fixed workspace and the reserved null parent small beside the budget.
    recipe.cache_budget_bytes = 16 << 20
    plan = recipe.setup().spec.memory_plan
    full = next(group for group in plan.groups if group.group_id == FULL_ATTENTION)
    groups = recipe.groups()
    decoder_layout = pack(
        tuple(group for group in groups if group[0].group_id in recipe.group_ids),
        prefix_granularity=recipe.prefix_granularity,
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    decoder_pages = (
        recipe.num_lcm_blocks(decoder_layout)
        * dict(decoder_layout.group_packing)[FULL_ATTENTION]
    )
    # Side groups own separate parents, so their bytes need not be reserved
    # again in every parent occupied by attention KV.
    assert full.page_count - 1 >= 0.95 * decoder_pages

    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    plan = layout.bind(2 * len(groups))
    arena = CacheArena(plan, "cpu", cache_group_specs=tuple(spec for spec, _ in groups))
    written = []
    for parent, group in enumerate(plan.groups):
        count = group.cache_blocks_per_lcm_block
        first = 1 + 2 * parent * count
        for field in plan.fields:
            if field.group_id != group.group_id:
                continue
            pages = arena.field(field.field_id).view(group.page_count, *field.shape)
            for page in {first, first + count - 1}:
                value = len(written) + 1
                pages[page].fill_(value)
                written.append((pages[page], value))
    # Fields within one group and occupied parents from other groups remain
    # independent, including the first and last child page of a parent.
    for value, expected in written:
        assert torch.all(value == expected)


@pytest.mark.parametrize("geometry", ["wide_ple", "different_draft_index_width"])
def test_qwen4_side_cache_geometry_can_require_separate_planes(geometry: str) -> None:
    recipe = _recipe(
        layer_types=(FULL_ATTENTION, FULL_ATTENTION), speculative=True, width=1
    )
    recipe.cache_budget_bytes = 16 << 20
    if geometry == "wide_ple":
        recipe.model_config.hf_text_config.short_conv_state_shape = (2048, 3)
    else:
        draft_text = copy(recipe.draft_model_config.hf_text_config)
        draft_text.indexer_head_dim = 16
        recipe.draft_model_config.hf_text_config = draft_text
    setup = recipe.setup()
    arena = CacheArena(
        setup.spec.memory_plan,
        "cpu",
        cache_group_specs=setup.spec.cache_group_specs,
    )
    assert arena.field(qwen4_exp_ple_conv_field(0)).shape[-2:] == (
        recipe.model_config.hf_text_config.short_conv_state_shape
    )
    assert arena.field(qsa_compressed_field(2)).shape[-1] == (
        recipe.draft_model_config.hf_text_config.indexer_head_dim
    )


def test_qwen4_zero_index_width_keeps_geometry_validation() -> None:
    recipe = _recipe(layer_types=(FULL_ATTENTION,), speculative=False, width=1)
    recipe.model_config.hf_text_config.indexer_head_dim = 0
    with pytest.raises(ValueError, match="invalid geometry"):
        recipe.setup()


def _flat_kv_recipe() -> Qwen4ExpRecipe:
    recipe = _recipe(
        layer_types=(FULL_ATTENTION, FULL_ATTENTION), speculative=False, width=1
    )
    recipe.server_args.qwen4_exp_flat_kv = True
    recipe.server_args.mapping = SimpleNamespace(pp_size=1)
    recipe.cache_budget_bytes = 16 << 20
    return recipe


def _flat_kv_pool(recipe):
    setup = recipe.setup()
    config = replace(recipe.attn_config, device="cpu")
    arena = create_cache_arena(setup.spec, device="cpu", enable_memory_saver=False)
    pool = create_cache_pool(
        setup.spec,
        config,
        arena,
        num_layers=setup.num_target_layers,
        rank=0,
    )
    return arena, pool


@pytest.mark.parametrize("kernel_page_size", [64, 256])
def test_flat_kv_atom_pages_resolve_to_the_original_field_bytes(
    kernel_page_size: int,
) -> None:
    recipe = _flat_kv_recipe()
    arena, pool = _flat_kv_pool(recipe)
    plan = arena.plan
    kv_ids = {f"layer.{layer}.{name}" for layer in range(2) for name in ("k", "v")}
    assert {field.plane_id for field in plan.fields if field.field_id in kv_ids} == {
        "flatkv"
    }

    granularity = plan.prefix_granularity
    ratio = granularity // kernel_page_size
    atom_pages = pool.atom_pages_per_block(FULL_ATTENTION, kernel_page_size)
    group = plan.group(FULL_ATTENTION)
    last_block = group.page_count - 1
    raw = torch.tensor([[0, 1, last_block]], dtype=torch.int32)
    stacks = GroupTableStacks(
        [
            GroupTableSpec(
                FULL_ATTENTION,
                granularity,
                kernel_page_size,
                3 * ratio,
                atom_pages,
            )
        ],
        max_bs=1,
        max_tokens_per_req=1,
        max_extend_tokens=0,
        device="cpu",
    )
    stacks.fill(1, 1, {FULL_ATTENTION: raw})
    table = stacks.table(FULL_ATTENTION, 1)
    expected = [0] * ratio + [
        block * atom_pages + subpage
        for block in (1, last_block)
        for subpage in range(ratio)
    ]
    assert table.tolist() == [expected]
    assert stacks.host_table(FULL_ATTENTION, raw, 1).tolist() == [expected]

    # The two-layer layout is [k0, v0, k1, v1] in every scheduler block.
    # At Q=G, logic block 1 therefore maps to atom page 4 for every field.
    if kernel_page_size == granularity:
        assert atom_pages == 4
        assert table[0, ratio].item() == 4

    for layer in range(2):
        for name in ("k", "v"):
            field_id = f"layer.{layer}.{name}"
            field = plan.field(field_id)
            row_bytes = field.payload_bytes // granularity
            cache = (
                pool.get_key_buffer(layer)
                if name == "k"
                else pool.get_value_buffer(layer)
            )
            for block in (1, last_block):
                for subpage in range(ratio):
                    column = block * ratio + subpage
                    atom_page = table[0, column].item()
                    row = atom_page * kernel_page_size + 3
                    value = 10 + layer * 20 + (name == "v") * 40 + subpage
                    cache[row].fill_(value)
                    expected_offset = (
                        plan.field_page_byte_offset(field_id, block)
                        + (subpage * kernel_page_size + 3) * row_bytes
                    )
                    actual = arena.buffer.view(torch.bfloat16)[
                        expected_offset // 2 : expected_offset // 2 + cache.shape[-1]
                    ]
                    assert torch.all(actual == value), (
                        field_id,
                        block,
                        subpage,
                        expected_offset,
                    )


def test_flat_kv_rejects_a_stride_that_is_not_whole_atom_pages() -> None:
    arena, pool = _flat_kv_pool(_flat_kv_recipe())
    field_id = "layer.0.k"
    arena.plan = replace(
        arena.plan,
        fields=tuple(
            (
                replace(field, page_stride_bytes=field.page_stride_bytes + 1)
                if field.field_id == field_id
                else field
            )
            for field in arena.plan.fields
        ),
    )
    with pytest.raises(ValueError, match="is not divisible by atom-page payload"):
        pool.atom_pages_per_block(FULL_ATTENTION, 64)


def test_flat_kv_rejects_pipeline_parallelism() -> None:
    recipe = _flat_kv_recipe()
    recipe.server_args.mapping.pp_size = 2
    with pytest.raises(NotImplementedError, match="PP=1"):
        recipe.fields_for_layer(0, FULL_ATTENTION, 0)
