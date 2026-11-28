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


"""Plugin launch options retain types and ownership across entry points."""

import argparse
import json
import pickle
from dataclasses import FrozenInstanceError, dataclass, field, replace
from typing import Literal

import pytest

from tokenspeed.runtime.plugins.registry import recording, register_extra_server_args
from tokenspeed.runtime.plugins.server_args import (
    add_extra_server_args,
    collect_extra_server_args,
    get_extra_server_args,
    normalize_extra_server_args,
)


@dataclass(frozen=True, kw_only=True)
class Options:
    implementation: Literal["native", "fused"]
    enabled: bool = False
    capacity: int | None = None
    numerics: str = field(default="auto", metadata={"native_server_arg": True})


@pytest.fixture
def parser():
    with recording() as registrations:
        register_extra_server_args("fixture", Options, validate=None)
    try:
        result = argparse.ArgumentParser()
        add_extra_server_args(result)
        yield result
    finally:
        for undo in reversed(registrations.undo):
            undo()


def test_json_and_programmatic_values_agree(parser):
    values = {"implementation": "fused", "enabled": False, "capacity": 16}
    assert (
        collect_extra_server_args(
            parser.parse_args(["--extra-server-args", json.dumps(values)])
        )
        == values
    )
    assert normalize_extra_server_args(values) == values
    assert normalize_extra_server_args(pickle.loads(pickle.dumps(values))) == values
    assert normalize_extra_server_args(json.loads(json.dumps(values))) == values


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--fixture-implementation", "native"),
        ("--fixture-enabled", "false"),
        ("--fixture-capacity", "16"),
    ],
)
def test_individual_plugin_flags_are_not_a_transport(parser, flag, value):
    with pytest.raises(SystemExit) as error:
        parser.parse_args(
            ["--extra-server-args", '{"implementation":"native"}', flag, value]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("values", [[], None, "{}"])
def test_cli_json_requires_an_object(parser, values):
    args = parser.parse_args(["--extra-server-args", json.dumps(values)])
    with pytest.raises(TypeError, match="JSON object"):
        collect_extra_server_args(args)


@pytest.mark.parametrize(
    "values",
    [
        {"implementation": "unknown"},
        {"implementation": "native", "enabled": "false"},
        {"implementation": "native", "enabled": 1},
        {"implementation": "native", "capacity": True},
        {"implementation": "native", "unknown": 2},
        {"missing": {"enabled": True}},
    ],
)
def test_invalid_programmatic_values_fail(parser, values):
    with pytest.raises(ValueError):
        normalize_extra_server_args(values)


@pytest.mark.parametrize("values", [[], None, "{}"])
def test_nonobject_values_fail(parser, values):
    with pytest.raises(TypeError):
        normalize_extra_server_args(values)


def test_normalization_does_not_share_launch_state(parser):
    original = {"implementation": "native"}
    copied = normalize_extra_server_args(original)
    copied["implementation"] = "fused"
    assert original == {"implementation": "native"}


def test_failed_plugin_registration_removes_its_options():
    with pytest.raises(RuntimeError, match="activation failed"):
        with recording():
            register_extra_server_args("failed_fixture", Options, validate=None)
            raise RuntimeError("activation failed")
    with pytest.raises(ValueError, match="Unknown extra server arguments"):
        normalize_extra_server_args({"implementation": "native"})


def test_two_plugins_cannot_register_the_same_field(parser):
    with pytest.raises(ValueError, match="fields are already registered"):
        register_extra_server_args("another_fixture", Options, validate=None)


def test_field_ownership_does_not_require_json_namespaces(parser):
    @dataclass(frozen=True)
    class SecondOptions:
        label: str = ""

    with recording() as registrations:
        register_extra_server_args("second_fixture", SecondOptions, validate=None)
    try:
        values = {"implementation": "native", "label": "worker"}
        assert normalize_extra_server_args(values) == values
    finally:
        for undo in reversed(registrations.undo):
            undo()


def test_cli_json_does_not_coerce_integer_to_boolean(parser):
    args = parser.parse_args(
        ["--extra-server-args", '{"implementation":"native","enabled":1}']
    )
    with pytest.raises(ValueError, match="enabled"):
        collect_extra_server_args(args)


def test_omitted_options_do_not_share_parser_state(parser):
    first = collect_extra_server_args(parser.parse_args([]))
    second = collect_extra_server_args(parser.parse_args([]))
    first["implementation"] = "native"
    assert second == {}


def test_plugin_sampling_backend_registration_rolls_back():
    from tokenspeed.runtime.plugins.registry import register_sampling_backend
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
    from tokenspeed.runtime.sampling.registry import get_sampling_backend_class
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    args = prepare_server_args(
        ["--model", "unused", "--sampling-backend", "fixture_sampler"]
    )
    with pytest.raises(RuntimeError, match="activation failed"):
        with recording():
            register_sampling_backend("fixture_sampler", GreedySamplingBackend)
            assert get_sampling_backend_class(args) is GreedySamplingBackend
            raise RuntimeError("activation failed")
    with pytest.raises(ValueError, match="Unknown sampling backend"):
        get_sampling_backend_class(args)


def test_models_share_immutable_arguments_within_one_launch(parser):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    launch = ServerArgs(model="unused", extra_server_args={"implementation": "native"})
    target = get_extra_server_args(launch, "fixture", Options)
    draft = get_extra_server_args(launch, "fixture", Options)
    assert target is draft
    with pytest.raises(FrozenInstanceError):
        target.enabled = True
    independent = replace(launch)
    assert get_extra_server_args(independent, "fixture", Options) is not target
    worker = pickle.loads(pickle.dumps(launch))
    worker_target = get_extra_server_args(worker, "fixture", Options)
    assert worker_target is get_extra_server_args(worker, "fixture", Options)
    assert worker_target is not target and worker_target == target


def test_published_arguments_cannot_be_changed(parser):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    launch = ServerArgs(model="unused", extra_server_args={"implementation": "native"})
    get_extra_server_args(launch, "fixture", Options)
    launch.extra_server_args["enabled"] = True
    with pytest.raises(ValueError, match="changed after publication"):
        get_extra_server_args(launch, "fixture", Options)


def test_native_fields_cannot_be_supplied_through_extra_json_or_flags(parser):
    with pytest.raises(ValueError, match="Unknown extra server arguments"):
        collect_extra_server_args(
            parser.parse_args(
                [
                    "--extra-server-args",
                    '{"implementation":"native","numerics":"rl-bitwise"}',
                ]
            )
        )
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--fixture-numerics", "rl-bitwise"])
    assert error.value.code == 2


def test_native_binding_does_not_claim_another_plugins_public_field(parser):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    @dataclass(frozen=True)
    class SecondOptions:
        numerics: str = ""

    with recording() as registrations:
        register_extra_server_args("second_fixture", SecondOptions, validate=None)
    try:
        launch = ServerArgs(
            model="unused",
            numerics="rl-bitwise",
            extra_server_args={"implementation": "native", "numerics": "public-value"},
        )
        assert (
            get_extra_server_args(launch, "fixture", Options).numerics == "rl-bitwise"
        )
        assert (
            get_extra_server_args(launch, "second_fixture", SecondOptions).numerics
            == "public-value"
        )
    finally:
        for undo in reversed(registrations.undo):
            undo()


def test_derived_state_is_not_an_argument_schema():
    @dataclass(frozen=True)
    class Derived:
        state: dict = field(init=False, default_factory=dict)

    with pytest.raises(TypeError, match="derived state"):
        register_extra_server_args("derived_fixture", Derived, validate=None)


@pytest.mark.parametrize("conflict", [False, True])
def test_model_constructor_owns_loading_inputs(parser, tmp_path, conflict):
    import torch

    from tokenspeed.runtime.configs.load_config import LoadConfig
    from tokenspeed.runtime.configs.model_config import (
        ModelConfig,
        configure_mha_attention,
    )
    from tokenspeed.runtime.configs.model_profile import ModelProfile
    from tokenspeed.runtime.model_loader.loader import _initialize_model
    from tokenspeed.runtime.plugins.registry import register_model
    from tokenspeed.runtime.utils.server_args import ServerArgs

    def constructor_inputs(model, loading):
        if conflict:
            return {"runtime_config": "conflicting input"}
        return {"context_len": model.context_len, "download_dir": loading.download_dir}

    class FixtureConstructorModel(torch.nn.Module):
        def __init__(
            self,
            config,
            mapping,
            quant_config,
            *,
            runtime_config,
            context_len,
            download_dir
        ):
            super().__init__()
            self.runtime_config = runtime_config
            self.context_len = context_len
            self.download_dir = download_dir

        @classmethod
        def model_profile(cls, hf_config):
            return ModelProfile(
                configure_attention=configure_mha_attention,
                cache_family=None,
                linear_attention=None,
                default_attention_backend="triton",
                request_token_history=False,
                is_multimodal=False,
                tokenizer_kwargs={},
                attention_instances_per_layer=1,
                numerics_envelopes=frozenset({"auto"}),
                create_attention_config=None,
                create_runtime_config=lambda model, launch: get_extra_server_args(
                    launch, "fixture", Options
                ),
                create_model_kwargs=constructor_inputs,
            )

    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="llama",
                architectures=["FixtureConstructorModel"],
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=64,
                max_position_embeddings=128,
            )
        )
    )
    with recording() as registrations:
        register_model(FixtureConstructorModel)
    try:
        launch = ServerArgs(
            model=str(tmp_path), extra_server_args={"implementation": "native"}
        )
        model_config = ModelConfig(
            str(tmp_path), model_override_args="{}", server_args=launch
        )
        if conflict:
            with pytest.raises(ValueError, match="already supplied"):
                _initialize_model(model_config, LoadConfig())
        else:
            first = _initialize_model(model_config, LoadConfig(download_dir="first"))
            second = _initialize_model(model_config, LoadConfig(download_dir="second"))
            assert (
                first.runtime_config
                is second.runtime_config
                is get_extra_server_args(launch, "fixture", Options)
            )
            assert first.context_len == second.context_len == 128
            assert (first.download_dir, second.download_dir) == ("first", "second")
            assert "download_dir" not in model_config.hf_config.to_dict()
    finally:
        for undo in reversed(registrations.undo):
            undo()
