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

"""Runtime plugin discovery and the registries it opens."""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime import plugins
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    configure_mla_attention,
)
from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.layers.attention import registry as attention_registry
from tokenspeed.runtime.layers.attention.kv_cache import factory as pool_factory
from tokenspeed.runtime.layers.attention.kv_cache.recipes import setup as recipe_setup
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.plugins import registry
from tokenspeed.runtime.utils import hf_transformers_utils


def _profile(**overrides) -> ModelProfile:
    fields = dict(
        configure_attention=configure_mla_attention,
        cache_family="fixture_family",
        linear_attention="fixture_linear",
        default_attention_backend="mla",
        default_prefix_granularity=64,
        request_token_history=True,
        tokenizer_kwargs={"fix_mistral_regex": True},
    )
    fields.update(overrides)
    return ModelProfile(**fields)


class FixtureForCausalLM(torch.nn.Module):
    @classmethod
    def model_profile(cls, hf_config) -> ModelProfile:
        return _profile(
            linear_attention=(
                "fixture_linear" if hf_config.fixture_linear_layers else None
            )
        )


class FixtureConfig(PretrainedConfig):
    model_type = "fixture_type"

    def __init__(self, fixture_linear_layers: bool = True, **kwargs) -> None:
        self.fixture_linear_layers = fixture_linear_layers
        super().__init__(**kwargs)


class _Dist:
    metadata = {"Name": "fixture-plugin"}
    version = "1.2.3"


class _EntryPoint:
    def __init__(self, name: str, register) -> None:
        self.name = name
        self.value = f"fixture_plugin:{name}"
        self.dist = _Dist()
        self._register = register

    def load(self):
        return self._register


@pytest.fixture
def isolated(monkeypatch):
    """Fresh loader state and registry tables, restored after the test."""
    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "_loaded_plugins", {})
    monkeypatch.setattr(registry, "_MODELS", {})
    monkeypatch.setattr(
        hf_transformers_utils,
        "_CONFIG_REGISTRY",
        dict(hf_transformers_utils._CONFIG_REGISTRY),
    )
    monkeypatch.setattr(hf_transformers_utils, "_ARCHITECTURE_CONFIG_REGISTRY", {})
    monkeypatch.setattr(
        attention_registry,
        "_BACKEND_REGISTRY",
        dict(attention_registry._BACKEND_REGISTRY),
    )
    monkeypatch.setattr(
        attention_registry,
        "_LINEAR_ATTN_BACKENDS",
        dict(attention_registry._LINEAR_ATTN_BACKENDS),
    )
    monkeypatch.setattr(recipe_setup, "_RECIPES", dict(recipe_setup._RECIPES))
    monkeypatch.setattr(
        pool_factory, "_POOL_FACTORIES", dict(pool_factory._POOL_FACTORIES)
    )
    kernel_discovery = []
    import tokenspeed_kernel.plugins as kernel_plugins

    monkeypatch.setattr(
        kernel_plugins, "discover_plugins", lambda: kernel_discovery.append(True)
    )
    monkeypatch.delenv(plugins.DISABLE_ENV_VAR, raising=False)
    return kernel_discovery


def _install(monkeypatch, *entry_points: _EntryPoint) -> None:
    def fake_entry_points(*, group: str):
        assert group == plugins.ENTRY_POINT_GROUP
        return list(entry_points)

    monkeypatch.setattr(plugins.importlib_metadata, "entry_points", fake_entry_points)


def _register_fixture() -> None:
    registry.register_config(
        FixtureConfig,
        model_type=None,
        architectures=("FixtureForCausalLM",),
    )
    registry.register_model(FixtureForCausalLM)


def test_loader_registers_once_and_reports(monkeypatch, isolated) -> None:
    calls = []

    def register() -> None:
        calls.append(True)
        _register_fixture()

    _install(monkeypatch, _EntryPoint("fixture", register))
    (info,) = plugins.ensure_loaded()
    assert plugins.ensure_loaded() == [info]
    assert calls == [True]
    assert isolated == [True]
    assert info.package == "fixture-plugin" and info.version == "1.2.3"
    assert info.registrations == (
        ("config architecture", ("FixtureForCausalLM",)),
        ("model", ("FixtureForCausalLM",)),
    )
    assert registry.registered_model("FixtureForCausalLM").cls is FixtureForCausalLM


def test_failed_plugin_is_undone_and_skipped(monkeypatch, isolated) -> None:
    def register() -> None:
        _register_fixture()
        raise RuntimeError("vendor library missing")

    _install(monkeypatch, _EntryPoint("broken", register))
    with pytest.warns(UserWarning, match="Failed to load plugin 'broken'"):
        assert plugins.ensure_loaded() == []
    assert registry.registered_model("FixtureForCausalLM") is None
    assert "FixtureForCausalLM" not in (
        hf_transformers_utils._ARCHITECTURE_CONFIG_REGISTRY
    )


def test_disabled_plugin_is_skipped(monkeypatch, isolated) -> None:
    _install(monkeypatch, _EntryPoint("fixture", _register_fixture))
    monkeypatch.setenv(plugins.DISABLE_ENV_VAR, "other, fixture")
    assert plugins.ensure_loaded() == []
    assert registry.registered_model("FixtureForCausalLM") is None


def test_model_registration_requires_a_profile(isolated) -> None:
    with pytest.raises(TypeError, match="model_profile"):
        registry.register_model(torch.nn.Linear)


def test_collisions_raise_unless_overridden(isolated) -> None:
    registry.register_model(FixtureForCausalLM)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_model(FixtureForCausalLM)
    registry.register_model(FixtureForCausalLM, override=True)
    assert registry.registered_model("FixtureForCausalLM").override

    registry.register_attention_backend("fixture", {AttentionArch.MLA}, object)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_attention_backend("fixture", {AttentionArch.MLA}, object)


def test_in_tree_architecture_needs_override(isolated) -> None:
    from tokenspeed.runtime.models.registry import _ModelRegistry

    in_tree = type("FixtureForCausalLM", (torch.nn.Module,), {})
    model_registry = _ModelRegistry(models={"FixtureForCausalLM": in_tree})
    registry.register_model(FixtureForCausalLM)
    with pytest.raises(ValueError, match="override=True"):
        model_registry.resolve_model_cls(["FixtureForCausalLM"])

    registry.register_model(FixtureForCausalLM, override=True)
    assert model_registry.resolve_model_cls(["FixtureForCausalLM"]) == (
        FixtureForCausalLM,
        "FixtureForCausalLM",
    )
    assert "FixtureForCausalLM" in model_registry.get_supported_archs()


def test_config_resolves_by_architecture_without_model_type(tmp_path, isolated) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"architectures": ["FixtureForCausalLM"], "fixture_linear_layers": False}
        )
    )
    registry.register_config(
        FixtureConfig, model_type=None, architectures=("FixtureForCausalLM",)
    )
    config = hf_transformers_utils.get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, FixtureConfig)
    assert config.fixture_linear_layers is False


def test_profile_resolution_follows_the_checkpoint(isolated) -> None:
    registry.register_model(FixtureForCausalLM)
    hybrid = registry.resolve_model_profile(
        ["Unknown", "FixtureForCausalLM"], FixtureConfig()
    )
    assert hybrid.linear_attention == "fixture_linear"
    assert hybrid.tokenizer_kwargs == {"fix_mistral_regex": True}
    with pytest.raises(TypeError):
        hybrid.tokenizer_kwargs["fix_mistral_regex"] = False
    dense = registry.resolve_model_profile(
        ["FixtureForCausalLM"], FixtureConfig(fixture_linear_layers=False)
    )
    assert dense.linear_attention is None
    assert registry.resolve_model_profile(["Unknown"], FixtureConfig()) is None


def test_profile_rejects_missing_facts() -> None:
    with pytest.raises(ValueError, match="cache_family"):
        _profile(cache_family="")
    with pytest.raises(ValueError, match="default_prefix_granularity"):
        _profile(default_prefix_granularity=0)
    with pytest.raises(TypeError):
        ModelProfile(configure_attention=configure_mla_attention)


def _model_config(profile: ModelProfile | None) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["FixtureForCausalLM"]),
        model_profile=profile,
    )


def test_attention_side_reads_the_profile() -> None:
    profile = _profile()
    side = attention_registry._resolve_attn_side(_model_config(profile), "trtllm_mla")
    assert side.linear_attention == "fixture_linear"
    assert side.is_hybrid_linear
    assert not side.is_kda and not side.is_hybrid_gdn
    assert (
        attention_registry._resolve_cache_family(side, config=None) == "fixture_family"
    )
    assert (
        attention_registry._resolve_full_attn_backend_name(
            side, softmax_attn=None, hybrid_request="trtllm_mla"
        )
        == "trtllm_mla"
    )
    # The user left the choice to the architecture default leaf.
    assert (
        attention_registry._resolve_full_attn_backend_name(
            side,
            softmax_attn=None,
            hybrid_request=attention_registry.HYBRID_LINEAR_ATTN_BACKEND,
        )
        is None
    )

    dense = attention_registry._resolve_attn_side(
        _model_config(_profile(linear_attention=None)), None
    )
    assert not dense.is_hybrid_linear


def test_in_tree_hybrids_resolve_through_the_linear_registry() -> None:
    side = attention_registry._resolve_attn_side(
        SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["KimiK3ForConditionalGeneration"],
                model_type="kimi_k3",
            ),
            model_profile=None,
        ),
        None,
    )
    assert side.linear_attention == "kda"
    assert set(attention_registry._LINEAR_ATTN_BACKENDS) >= {"kda", "gdn"}


def test_linear_backend_registration(isolated) -> None:
    def factory(server_args, config):
        return (server_args, config)

    registry.register_linear_attention_backend("fixture_linear", factory)
    assert attention_registry._LINEAR_ATTN_BACKENDS["fixture_linear"] is factory
    with pytest.raises(ValueError, match="already registered"):
        registry.register_linear_attention_backend("kda", factory)


class _FixtureRecipe(CacheRecipe):
    family = "fixture_family"

    @property
    def layer_types(self) -> tuple[str, ...]:
        return ()


def test_cache_recipe_and_pool_registration(isolated) -> None:
    registry.register_cache_recipe("fixture_family", _FixtureRecipe)
    assert recipe_setup._RECIPES["fixture_family"] is _FixtureRecipe
    with pytest.raises(ValueError, match="declares family"):
        registry.register_cache_recipe("other_family", _FixtureRecipe)

    calls = []

    def pool(spec, config, arena, *, num_layers, rank, field_layer_offset):
        calls.append((spec.family, num_layers, rank, field_layer_offset))
        return "pool"

    registry.register_cache_pool("fixture_family", pool)
    spec = SimpleNamespace(family="fixture_family")
    assert (
        pool_factory.create_cache_pool(spec, None, None, num_layers=3, rank=1) == "pool"
    )
    assert calls == [("fixture_family", 3, 1, 0)]
    with pytest.raises(TypeError, match="no cache pool"):
        pool_factory.create_cache_pool(
            SimpleNamespace(family="unregistered"), None, None, num_layers=1, rank=0
        )


def test_in_tree_paged_state_verify_is_a_recipe_fact() -> None:
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        KimiK3Recipe,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        OrdinaryRecipe,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen4_exp import (
        Qwen4ExpRecipe,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
        QwenGDNRecipe,
    )

    assert KimiK3Recipe.uses_paged_state_verify
    assert QwenGDNRecipe.uses_paged_state_verify
    assert Qwen4ExpRecipe.uses_paged_state_verify
    assert not OrdinaryRecipe.uses_paged_state_verify


class _FixtureDrafter:
    """Stands in for a BaseDrafter subclass; resolution never instantiates."""


class _FixtureNextN(torch.nn.Module):
    pass


def test_register_drafter_adds_an_algorithm() -> None:
    from tokenspeed.runtime.execution import drafter

    registry.register_drafter("FIXTURE_SPEC", _FixtureDrafter)
    try:
        assert "FIXTURE_SPEC" in drafter.registered_drafter_algorithms()
        drafter.validate_drafter_algorithm("FIXTURE_SPEC")
        assert (
            drafter.get_drafter_impl("FIXTURE_SPEC", torch.nn.Module())
            is _FixtureDrafter
        )
    finally:
        drafter._PLUGIN_DRAFTERS.pop("FIXTURE_SPEC", None)


def test_register_drafter_scopes_an_in_tree_algorithm_by_model_class() -> None:
    from tokenspeed.runtime.execution import drafter
    from tokenspeed.runtime.execution.drafter.eagle import Eagle

    registry.register_drafter("MTP", _FixtureDrafter, model_cls=_FixtureNextN)
    try:
        assert drafter.get_drafter_impl("MTP", _FixtureNextN()) is _FixtureDrafter
        # Other draft models keep the in-tree resolution.
        assert drafter.get_drafter_impl("MTP", torch.nn.Module()) is Eagle
    finally:
        drafter._PLUGIN_DRAFTERS.pop("MTP", None)


def test_register_drafter_default_collides_with_in_tree() -> None:
    from tokenspeed.runtime.execution import drafter

    with pytest.raises(ValueError, match="already registered"):
        registry.register_drafter("MTP", _FixtureDrafter)
    assert "MTP" not in drafter._PLUGIN_DRAFTERS


def test_unknown_drafter_algorithm_is_rejected() -> None:
    from tokenspeed.runtime.execution import drafter

    with pytest.raises(ValueError, match="available"):
        drafter.validate_drafter_algorithm("NEXTN")


def test_drafter_registration_unwinds_with_the_recording() -> None:
    from tokenspeed.runtime.execution import drafter

    with pytest.raises(RuntimeError, match="boom"):
        with registry.recording():
            registry.register_drafter("FIXTURE_SPEC", _FixtureDrafter)
            raise RuntimeError("boom")
    assert "FIXTURE_SPEC" not in drafter.registered_drafter_algorithms()


def test_profile_declares_attention_instances_per_layer() -> None:
    from tokenspeed.runtime.configs.model_config import _derive_num_attention_layers

    paired = _profile(attention_instances_per_layer=2)
    config = SimpleNamespace(architectures=["FixtureForCausalLM"])
    assert _derive_num_attention_layers(config, 14, paired) == 28
    assert _derive_num_attention_layers(config, 14, _profile()) == 14
    assert _derive_num_attention_layers(config, 14, None) == 14
    with pytest.raises(ValueError, match="attention_instances_per_layer"):
        _profile(attention_instances_per_layer=0)


def test_drafter_override_rollback_restores_the_prior_entry() -> None:
    from tokenspeed.runtime.execution import drafter

    class _SecondDrafter(_FixtureDrafter):
        pass

    registry.register_drafter(
        "FIXTURE_SPEC", _FixtureDrafter, defaults_to_base_checkpoint=False
    )
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with registry.recording():
                registry.register_drafter(
                    "FIXTURE_SPEC",
                    _SecondDrafter,
                    defaults_to_base_checkpoint=True,
                    override=True,
                )
                raise RuntimeError("boom")
        entries = drafter._PLUGIN_DRAFTERS["FIXTURE_SPEC"]
        assert entries.default is _FixtureDrafter
        assert entries.defaults_to_base_checkpoint is False
    finally:
        drafter._PLUGIN_DRAFTERS.pop("FIXTURE_SPEC", None)


def test_cache_recipe_missing_family_declaration_is_rejected() -> None:
    class _UndeclaredRecipe(CacheRecipe):
        @property
        def layer_types(self) -> tuple[str, ...]:
            return ()

    with pytest.raises(ValueError, match="must declare family"):
        registry.register_cache_recipe("fixture_family", _UndeclaredRecipe)


def test_register_config_rolls_back_the_autoconfig_mirror(isolated) -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    class _RollbackConfig(PretrainedConfig):
        model_type = "fixture_rollback_type"

    try:
        with pytest.raises(RuntimeError, match="boom"):
            with registry.recording():
                registry.register_config(
                    _RollbackConfig, model_type="fixture_rollback_type"
                )
                assert "fixture_rollback_type" in CONFIG_MAPPING._extra_content
                raise RuntimeError("boom")
        assert "fixture_rollback_type" not in CONFIG_MAPPING._extra_content
    finally:
        CONFIG_MAPPING._extra_content.pop("fixture_rollback_type", None)


def test_declared_draft_cache_family_must_match_a_custom_target() -> None:
    resolve = attention_registry._resolve_heterogeneous_draft_family

    # A draft profile declaring the target's own family shares its view.
    assert (
        resolve("fixture_family", "fixture_family", draft_family_declared=True) is None
    )
    # An in-tree draft without a profile rides the target recipe's draft
    # view (the deepseek_v4 NextN pattern).
    assert resolve("fixture_family", "mla", draft_family_declared=False) is None
    # A declared, different family is a layout contradiction.
    with pytest.raises(RuntimeError, match="declares cache family"):
        resolve("fixture_family", "mla", draft_family_declared=True)


def test_draft_profile_backend_default_lands_on_the_drafter_field() -> None:
    from tokenspeed.runtime.configs.model_config import _apply_attention_defaults
    from tokenspeed.runtime.utils.server_args import ServerArgs

    args = ServerArgs(model="x")
    _apply_attention_defaults(
        args,
        name="Fixture",
        default_backend="fixture_backend",
        default_prefix_granularity=None,
        is_draft_worker=True,
    )
    assert args.drafter_attention_backend == "fixture_backend"
    assert args.attention_backend is None
    _apply_attention_defaults(
        args,
        name="Fixture",
        default_backend="target_backend",
        default_prefix_granularity=None,
        is_draft_worker=False,
    )
    assert args.attention_backend == "target_backend"
    assert args.drafter_attention_backend == "fixture_backend"
