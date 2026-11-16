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

"""CPU regressions for heterogeneous Inkling cache capacity."""

from dataclasses import replace
from pathlib import Path
from test.runtime.models.test_inkling2_config import _checkpoint

import pytest
from tokenspeed_scheduler import CapacityModel

from tokenspeed.runtime.configs.inkling2 import Inkling2MMConfig, Inkling2MTPConfig
from tokenspeed.runtime.configs.inkling_config import InklingMMConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
    InklingAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import InklingRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    capacity_model,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CacheSetup
from tokenspeed.runtime.layers.attention.registry import (
    _prepare_fixed_workspaces,
    _resolve_attn_side,
    _resolve_cache_family,
    _resolve_heterogeneous_draft_family,
    create_paged_router,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

_CONTEXT = 1048576


def _scheduler_model(recipe: InklingRecipe, setup: CacheSetup) -> CapacityModel:
    plan = setup.spec.memory_plan
    return capacity_model(
        setup.spec.cache_group_specs,
        prefix_granularity=plan.prefix_granularity,
        virtual_packing={
            group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
        },
        limits=recipe.scheduler_limits,
    )


def _parents_needed(model: CapacityModel, tokens: int) -> int:
    return model.lcm_blocks_needed_for(
        model.concurrent_group_pages(max_total_tokens=tokens, max_context_len=_CONTEXT)
    )


def _recipe(
    path: Path,
    *,
    inkling2: bool,
    token_limit: int | None,
    budget_bytes: int,
    overlap_depth: int,
) -> InklingRecipe:
    config_cls = Inkling2MMConfig if inkling2 else InklingMMConfig
    architecture = (
        "Inkling2ForConditionalGeneration"
        if inkling2
        else "InklingForConditionalGeneration"
    )
    config_cls(
        architectures=[architecture],
        text_config=dict(
            model_max_length=_CONTEXT,
            torch_dtype="bfloat16",
            hidden_size=4608 if inkling2 else 1536,
            num_hidden_layers=48,
            vocab_size=201024,
            unpadded_vocab_size=200058,
            num_attention_heads=48,
            num_key_value_heads=8,
            head_dim=128,
            d_rel=16,
            rel_extent=1024,
            local_layer_ids=[i for i in range(48) if i % 6 != 5],
            swa_num_attention_heads=48,
            swa_num_key_value_heads=8,
            swa_head_dim=128,
            sliding_window_size=1024,
            sconv_kernel_size=4,
            rms_norm_eps=1e-6,
            q_bias=False,
            o_bias=False,
            n_routed_experts=384,
            n_shared_experts=0,
            num_experts_per_tok=8,
            intermediate_size=2304,
            route_scale=8.0,
            use_gate_bias=True,
            log_scaling_n_floor=128000,
            log_scaling_alpha=0.1,
            final_logit_softcapping=30.0,
            use_value_norm_no_gamma=True,
            qk_l2_clip_threshold=22.627416997969522,
            rel_l2_clip_threshold=8.0,
            n_layers_per_attn_res_block=6,
            attn_res_normalize_values=True,
            mlp_gate_softcap=4.0,
            mlp_up_softcap=25.0,
        ),
        vision_config=None,
        mtp_config=None,
        eos_token_id=200006,
    ).save_pretrained(path)
    args = ServerArgs(
        model=str(path),
        max_model_len=_CONTEXT,
        max_total_tokens=token_limit,
        max_num_seqs=16,
        chunked_prefill_size=2048,
        attn_tp_size=4,
        attention_backend="triton",
        kv_cache_dtype="bfloat16",
        enable_prefix_caching=False,
        disable_kvstore=True,
        device="cuda",
    )
    args.mapping.rank = 0
    model = ModelConfig(
        str(path),
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=args,
    )
    return InklingRecipe(
        server_args=args,
        model_config=model,
        attn_config=MHAConfig.generate(args, model, is_draft=False),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=budget_bytes,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=overlap_depth,
    )


@pytest.mark.parametrize(
    "inkling2,fp8_sconv", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("overlap_depth", [0, 1])
def test_token_limit_sizes_all_cache_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inkling2: bool,
    overlap_depth: int,
    fp8_sconv: bool,
) -> None:
    monkeypatch.setenv("INKLING_FP8_SCONV", "1" if fp8_sconv else "0")
    recipe = _recipe(
        tmp_path,
        inkling2=inkling2,
        token_limit=_CONTEXT,
        budget_bytes=64 << 30,
        overlap_depth=overlap_depth,
    )
    setup = recipe.setup()
    plan = setup.spec.memory_plan
    model = _scheduler_model(recipe, setup)
    assert setup.spec.token_capacity == _CONTEXT
    assert plan.num_lcm_blocks == _parents_needed(model, _CONTEXT)
    assert plan.arena_bytes + setup.fixed_workspace_bytes <= setup.cache_budget_bytes
    assert model.max_single_request_tokens(plan.num_lcm_blocks) >= _CONTEXT


def test_native_draft_workspace_matches_target_ring(tmp_path: Path) -> None:
    recipe = _draft_recipe(tmp_path, native=True)
    recipe.attn_config = replace(recipe.attn_config, device="cpu", max_bs=1)
    recipe.draft_attn_config = replace(recipe.draft_attn_config, device="cpu", max_bs=1)
    target_router = create_paged_router(
        recipe.attn_config,
        recipe.model_config.attention_arch,
        backend_name="triton",
    )
    target = InklingAttnBackend.from_config(
        target_router,
        recipe.model_config.hf_text_config,
        recipe.attn_config,
        num_layers=recipe.num_target_layers,
        is_draft=False,
        enable_layerwise_cache_ready=False,
    )
    draft = create_paged_router(
        recipe.draft_attn_config,
        recipe.draft_model_config.attention_arch,
        backend_name="triton",
    )
    assert draft.fixed_workspace_bytes() == 0
    assert target.fixed_workspace_bytes() == (
        target.conv_pool.conv_state.nbytes
        + target.conv_pool.remote_restore_pending.nbytes
    )
    _prepare_fixed_workspaces(
        server_args=recipe.server_args,
        config=recipe.attn_config,
        backend=target,
        draft_backend=draft,
        uses_paged_state_verify=False,
        is_inkling=True,
        qsa_bf16_scratch=None,
        expected_bytes=recipe.workspace_bytes(),
    )
    with pytest.raises(RuntimeError, match="does not match allocated tensors"):
        _prepare_fixed_workspaces(
            server_args=recipe.server_args,
            config=recipe.attn_config,
            backend=target,
            draft_backend=draft,
            uses_paged_state_verify=False,
            is_inkling=True,
            qsa_bf16_scratch=None,
            expected_bytes=recipe.workspace_bytes() + 1,
        )


@pytest.mark.parametrize("token_limit", [None, _CONTEXT])
def test_budget_limited_capacity_is_maximal(
    tmp_path: Path, token_limit: int | None
) -> None:
    recipe = _recipe(
        tmp_path,
        inkling2=True,
        token_limit=token_limit,
        budget_bytes=8 << 30,
        overlap_depth=1,
    )
    setup = recipe.setup()
    plan = setup.spec.memory_plan
    capacity = setup.spec.token_capacity
    model = _scheduler_model(recipe, setup)
    assert 0 < capacity < _CONTEXT
    assert _parents_needed(model, capacity) <= plan.num_lcm_blocks
    assert _parents_needed(model, capacity + 1) > plan.num_lcm_blocks
    assert plan.arena_bytes + setup.fixed_workspace_bytes <= setup.cache_budget_bytes


def _draft_recipe(path: Path, *, native: bool) -> InklingRecipe:
    target = _recipe(
        path / "target",
        inkling2=native,
        token_limit=_CONTEXT,
        budget_bytes=64 << 30,
        overlap_depth=1,
    )
    if native:
        draft_config = Inkling2MTPConfig.from_target(Inkling2MMConfig(**_checkpoint()))
    else:
        values = target.model_config.hf_config.to_dict()
        values.update(
            architectures=["InklingForConditionalGenerationNextN"],
            mtp_config=dict(num_nextn_predict_layers=3, local_layer_ids=[0, 2]),
        )
        draft_config = InklingMMConfig(**values)
    draft_path = path / "draft"
    draft_config.save_pretrained(draft_path)
    args = target.server_args
    args.speculative_algorithm = "DSPARK" if native else "MTP"
    args.speculative_num_steps = 7 if native else 3
    args.speculative_num_draft_tokens = 8 if native else 4
    draft = ModelConfig(
        str(draft_path),
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=args,
        is_draft_worker=True,
    )
    return InklingRecipe(
        server_args=args,
        model_config=target.model_config,
        attn_config=MHAConfig.generate(args, target.model_config, is_draft=False),
        draft_model_config=draft,
        draft_attn_config=MHAConfig.generate(args, draft, is_draft=True),
        cache_budget_bytes=target.cache_budget_bytes,
        probe_batch_rows=None,
        decode_input_tokens=args.speculative_num_draft_tokens,
        overlap_schedule_depth=1,
    )


@pytest.mark.parametrize("native", [False, True])
def test_draft_cache_uses_its_own_architecture(tmp_path: Path, native: bool) -> None:
    recipe = _draft_recipe(tmp_path, native=native)
    if native:
        assert recipe.attn_config.speculative_num_draft_tokens == 8
        assert recipe.draft_attn_config.speculative_num_draft_tokens == 8
        assert not recipe.draft_model_config.draft_sample_from_anchor
    draft_family = _resolve_cache_family(
        _resolve_attn_side(recipe.draft_model_config, "triton"),
        recipe.draft_attn_config,
    )
    assert _resolve_heterogeneous_draft_family(
        "inkling", draft_family, draft_family_declared=False
    ) == ("mha" if native else None)
    setup = recipe.setup()
    plan = setup.spec.memory_plan
    target_layers = recipe.num_target_layers
    draft_layers = 5 if native else 3
    assert setup.num_draft_layers == draft_layers
    draft_fields = tuple(
        field
        for field in plan.fields
        if int(field.field_id.split(".")[1]) >= target_layers
    )
    expected_names = (
        ["k", "v"]
        if native
        else ["k", "v", "kvconv_k", "kvconv_v", "attnconv", "mlpconv"]
    )
    assert {field.field_id for field in draft_fields} == {
        f"layer.{target_layers + i}.{name}"
        for i in range(draft_layers)
        for name in expected_names
    }
    if native:
        assert {field.group_id for field in draft_fields} == {"full_attention"}
        assert all(field.shape == (128, 2, 128) for field in draft_fields)
        # The target still verifies eight rows. Only its conv ring is reserved.
        text = recipe.model_config.hf_text_config
        conv_dim = 2 * text.hidden_size + 2 * (8 // 4) * 128
        rows = recipe.attn_config.max_bs + 2
        expected_ring = target_layers * rows * (4 - 1 + 8) * conv_dim * 2 + rows
        assert setup.fixed_workspace_bytes == expected_ring
        model = _scheduler_model(recipe, setup)
        assert plan.num_lcm_blocks == _parents_needed(model, setup.spec.token_capacity)
    else:
        # Only the convolutional depth draft must not wrap its trained layers.
        recipe.server_args.speculative_num_steps = draft_layers + 1
        with pytest.raises(ValueError, match="would wrap depths"):
            recipe.num_draft_layers
    assert plan.arena_bytes + setup.fixed_workspace_bytes <= setup.cache_budget_bytes
