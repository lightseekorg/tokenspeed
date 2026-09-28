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

"""CPU startup through the real registry, recipe, arena, pools and backend tree."""

import importlib
import sys
from dataclasses import replace
from test.runtime.cache.test_dots3_note_recipe import configure_mtp
from test.runtime.cache.test_dots3_note_recipe import inputs as inputs
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tokenspeed.runtime.layers.attention.backends  # noqa: F401
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    configure_mla_attention,
)
from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.layers.attention import registry
from tokenspeed.runtime.layers.attention.backends import specific
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.specific.dots3_note import (
    Dots3NoteAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    DOTS3_NOTE_ARCHITECTURES,
    Dots3NoteAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache import factory as pool_factory
from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import Dots3NoteCachePool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes import setup as recipe_setup
from tokenspeed.runtime.plugins import registry as plugin_registry


@pytest.fixture
def runtime_inputs(inputs):
    model = inputs["model_config"]
    inputs["model_config"] = SimpleNamespace(
        **vars(model),
        attention_arch=AttentionArch.DSA,
        model_profile=None,
    )
    args = inputs["server_args"]
    args.mapping = SimpleNamespace(**vars(args.mapping), has_pp=False, pp_rank=0)
    return inputs


def _build(inputs, *, reuse):
    return registry.create_attn_components(
        inputs["server_args"],
        inputs["model_config"],
        gpu_id=0,
        rank=0,
        gpu_memory=0,
        draft_model_config=inputs["draft_model_config"],
        decode_input_tokens=inputs["decode_input_tokens"],
        graph_reserve_bytes=0,
        post_profile_bytes=0,
        probe_batch_rows=None,
        profiled_cache_bytes=inputs["cache_budget_bytes"],
        reuse_target_backend=reuse.attn_backend if reuse is not None else None,
        reuse_draft_backend=reuse.draft_attn_backend if reuse is not None else None,
    )


@pytest.mark.parametrize("architecture", DOTS3_NOTE_ARCHITECTURES)
def test_architecture_registration_owns_config_and_cache_family(
    runtime_inputs, architecture
):
    is_draft = architecture.endswith("NextN")
    if is_draft:
        configure_mtp(runtime_inputs, width=4)
    model = runtime_inputs["draft_model_config" if is_draft else "model_config"]
    # Registration uses the HF architecture, not a shared model_type branch.
    model.hf_config.architectures = [architecture]
    model.attention_arch = AttentionArch.MLA if is_draft else AttentionArch.DSA
    config = registry._create_attn_config(
        runtime_inputs["server_args"], model, is_draft
    )
    spec = config.component(Dots3NoteAttnConfig)
    assert spec is not None and spec.backend_name == "dots3_note"
    assert (spec.num_attention_heads, spec.head_dim) == (
        (64, 256) if is_draft else (128, 192)
    )
    assert (
        registry._resolve_cache_family(registry._resolve_attn_side(model, None), config)
        == "dots3_note"
    )
    assert isinstance(
        registry._create_attn_backend(model.attention_arch, config),
        Dots3NoteAttnBackend,
    )


@pytest.mark.parametrize(
    "width,target_backend,draft_backend",
    [
        (1, None, None),
        (2, None, None),
        (4, "dots3_note", "dots3_note"),
        (4, None, "triton"),
    ],
)
def test_create_components_and_rebind(
    runtime_inputs, width, target_backend, draft_backend
):
    args = runtime_inputs["server_args"]
    args.attention_backend = target_backend
    args.drafter_attention_backend = draft_backend
    if width > 1:
        configure_mtp(runtime_inputs, width=width)
    build = _build(runtime_inputs, reuse=None)
    router, pool = build.attn_backend, build.token_to_kv_pool
    assert isinstance(router, Dots3NoteAttnBackend)
    assert isinstance(pool, Dots3NoteCachePool)
    assert build.attention_backend_name == "dots3_note"
    assert pool.field_layer_range == range(46)
    assert router.group_ids == ("full", "swa.0", "swa.1", "swa.2")
    for gid, leaf in router.leaves.items():
        full = gid == "full"
        assert isinstance(leaf, DSABackend if full else MLAAttnBackend)
        assert leaf.kernel_page_size == (64 if full else 32)
        assert leaf.kv_lora_rank == (512 if full else 1024)
        assert leaf.kernel_solution == "triton"
        assert leaf.cache_pool is pool
    router.init_cuda_graph_state(args.max_num_seqs)
    if width == 1:
        assert build.draft_attn_backend is None
        assert build.draft_token_to_kv_pool is None
    else:
        draft_router, draft_pool = (
            build.draft_attn_backend,
            build.draft_token_to_kv_pool,
        )
        assert isinstance(draft_router, Dots3NoteAttnBackend)
        assert isinstance(draft_pool, Dots3NoteCachePool)
        assert build.draft_attention_backend_name == "dots3_note"
        assert draft_pool.arena is pool.arena
        assert draft_pool.field_layer_range == range(46, 47)
        assert draft_router.group_ids == ("draft.swa",)
        leaf = draft_router.leaves["draft.swa"]
        assert isinstance(leaf, MLAAttnBackend) and leaf.is_draft
        assert (leaf.kernel_page_size, leaf.kv_lora_rank, leaf.num_local_heads) == (
            32,
            1024,
            8,
        )
        assert leaf.kernel_solution == "triton" and leaf.supports_layer_sliding_window
        assert draft_pool.get_key_buffer(0).stride(0) * 2 == 71_424
        draft_router.init_cuda_graph_state(args.max_num_seqs)
        assert "layer.46.latent_kv" in build.producer_fields_by_step[-1]
    assert len(build.cache_fields_by_stage[0]) == (60 if width > 1 else 59)
    # Probe/serving rebind retains the same tree but drops pool-derived buffers.
    rebound = _build(runtime_inputs, reuse=build)
    assert rebound.attn_backend is router
    assert rebound.token_to_kv_pool.arena is not pool.arena
    assert all(
        leaf.cache_pool is rebound.token_to_kv_pool for leaf in router.leaves.values()
    )
    assert all(leaf.page_table_buf is None for leaf in router.leaves.values())
    router.init_cuda_graph_state(args.max_num_seqs)
    if width > 1:
        assert rebound.draft_attn_backend is draft_router
        assert (
            draft_router.leaves["draft.swa"].cache_pool
            is rebound.draft_token_to_kv_pool
        )
        draft_router.init_cuda_graph_state(args.max_num_seqs)
    assert (args.attention_backend, args.drafter_attention_backend) == (
        target_backend,
        draft_backend,
    )


@pytest.mark.parametrize(
    "backend", ["hybrid_linear_attn", "mla", "dsa", "triton", "fa3"]
)
def test_explicit_unsupported_target_backend_is_not_discarded(runtime_inputs, backend):
    args = runtime_inputs["server_args"]
    args.attention_backend = backend
    with pytest.raises(
        ValueError, match="dots3 Plan A requires the dots3_note attention backend"
    ):
        _build(runtime_inputs, reuse=None)
    assert args.attention_backend == backend


@pytest.mark.parametrize("backend", ["hybrid_linear_attn", "mla", "dsa", "fa3"])
def test_explicit_unsupported_draft_backend_is_not_discarded(runtime_inputs, backend):
    configure_mtp(runtime_inputs, width=4)
    args = runtime_inputs["server_args"]
    args.drafter_attention_backend = backend
    with pytest.raises(
        ValueError, match="dots3 MTP requires the Triton draft attention backend"
    ):
        _build(runtime_inputs, reuse=None)
    assert args.drafter_attention_backend == backend


@pytest.mark.parametrize(
    "mode", ["pp", "qcp", "dcp", "fp8", "fp16", "draft_checkpoint", "draft_layers"]
)
def test_model_owned_config_rejects_unsupported_modes(runtime_inputs, mode):
    args, model = runtime_inputs["server_args"], runtime_inputs["model_config"]
    if mode == "pp":
        args.pipeline_parallel_size = 2
    elif mode == "qcp":
        args.mapping.attn.qcp_size = 2
    elif mode == "dcp":
        args.mapping.attn.dcp_size = 2
    elif mode == "fp8":
        args.kv_cache_dtype = "fp8"
    elif mode == "fp16":
        model.dtype = torch.float16
    else:
        configure_mtp(runtime_inputs, width=4)
        if mode == "draft_checkpoint":
            args.speculative_draft_model_path = "other-checkpoint"
        else:
            runtime_inputs["draft_model_config"].num_attention_layers = 2
    with pytest.raises((ValueError, NotImplementedError), match="dots3"):
        _build(runtime_inputs, reuse=None)


def test_ordinary_backend_dispatch_has_no_triton_mla_alias(runtime_inputs):
    registry.validate_attention_backend_name("dots3_note", flag="--attention-backend")
    assert registry._get_backend_cls("triton", AttentionArch.MHA) is MHAAttnBackend
    with pytest.raises(ValueError, match="does not support arch"):
        registry._get_backend_cls("triton", AttentionArch.MLA)
    config = runtime_inputs["attn_config"]
    spec = replace(config.component(Dots3NoteAttnConfig).swa, backend_name="mla")
    router = registry._create_attn_backend(
        AttentionArch.MLA, replace(config, components=(spec,))
    )
    assert type(router) is CacheGroupRouter


@pytest.fixture
def fresh_dots3_registration(monkeypatch):
    # Reimport only this built-in, restoring both module identity and registries.
    monkeypatch.setattr(
        registry, "_BACKEND_REGISTRY", registry._BACKEND_REGISTRY.copy()
    )
    monkeypatch.setattr(registry, "_MODEL_ATTENTION", registry._MODEL_ATTENTION.copy())
    monkeypatch.setattr(recipe_setup, "_RECIPES", recipe_setup._RECIPES.copy())
    monkeypatch.setattr(
        pool_factory, "_POOL_FACTORIES", pool_factory._POOL_FACTORIES.copy()
    )
    del recipe_setup._RECIPES["dots3_note"]
    del pool_factory._POOL_FACTORIES["dots3_note"]
    module_name = specific.dots3_note.__name__
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.delattr(specific, "dots3_note")
    return module_name


@pytest.mark.parametrize("preinstalled_override", [False, True])
def test_builtin_cache_registration_survives_plugin_rollback(
    fresh_dots3_registration, preinstalled_override
):
    recipe_override, pool_override = Mock(), Mock()
    if preinstalled_override:
        plugin_registry.register_cache_recipe(
            "dots3_note", recipe_override, override=True
        )
        plugin_registry.register_cache_pool("dots3_note", pool_override, override=True)
    with pytest.raises(RuntimeError, match="failed plugin"):
        with plugin_registry.recording():
            module = importlib.import_module(fresh_dots3_registration)
            plugin_registry.register_cache_pool("failed_plugin_pool", Mock())
            raise RuntimeError("failed plugin")
    assert importlib.import_module(fresh_dots3_registration) is module
    assert "failed_plugin_pool" not in pool_factory._POOL_FACTORIES
    assert recipe_setup._RECIPES["dots3_note"] is (
        recipe_override if preinstalled_override else module.Dots3NoteRecipe
    )
    assert pool_factory._POOL_FACTORIES["dots3_note"] is (
        pool_override if preinstalled_override else module.create_dots3_note_pool
    )


def test_model_profile_overrides_native_target_and_draft_registration(runtime_inputs):
    architecture = "Dots3NoteForCausalLM"
    profile = ModelProfile(
        configure_attention=configure_mla_attention,
        cache_family="mla",
        linear_attention=None,
        default_attention_backend="mla",
        default_prefix_granularity=None,
        request_token_history=False,
        tokenizer_kwargs={},
        attention_instances_per_layer=1,
        numerics_envelopes=frozenset({"auto"}),
    )
    model = SimpleNamespace(
        **(
            vars(runtime_inputs["model_config"])
            | {
                "hf_config": SimpleNamespace(architectures=[architecture]),
                "model_profile": profile,
                "num_attention_layers": 2,
                "num_attention_heads": 128,
                "num_key_value_heads": 128,
                "head_dim": 0,
                "kv_lora_rank": 0,
                "qk_nope_head_dim": 0,
                "qk_rope_head_dim": 0,
                "v_head_dim": 0,
                "scaling": 0.0,
            }
        )
    )
    args = runtime_inputs["server_args"]
    profile.configure_attention(model, args)
    draft = SimpleNamespace(
        **(
            vars(model)
            | {
                "hf_config": SimpleNamespace(architectures=[f"{architecture}NextN"]),
                "num_attention_layers": 1,
            }
        )
    )
    args.attention_backend = args.drafter_attention_backend = "hybrid_linear_attn"
    args.speculative_algorithm = "MTP"
    runtime_inputs.update(
        model_config=model, draft_model_config=draft, decode_input_tokens=4
    )
    target_side = registry._resolve_attn_side(model, args.attention_backend)
    draft_side = registry._resolve_attn_side(draft, args.drafter_attention_backend)
    registry._apply_backend_overrides(args, target_side, draft_side)
    # The profile has no linear attention: retain the ordinary override behavior.
    assert args.attention_backend is None and args.drafter_attention_backend is None
    for side_model, side, is_draft in (
        (model, target_side, False),
        (draft, draft_side, True),
    ):
        config = registry._create_attn_config(args, side_model, is_draft=is_draft)
        assert type(config.components[0]) is MLAConfig
        assert config.components[0].kv_lora_rank == 512
        assert registry._resolve_cache_family(side, config) == "mla"
    build = _build(runtime_inputs, reuse=None)
    assert build.attention_backend_name == build.draft_attention_backend_name == "mla"
    assert (
        type(build.attn_backend) is type(build.draft_attn_backend) is CacheGroupRouter
    )
    assert (
        type(build.token_to_kv_pool)
        is type(build.draft_token_to_kv_pool)
        is MLATokenToKVPool
    )
    assert build.draft_token_to_kv_pool.arena is build.token_to_kv_pool.arena
