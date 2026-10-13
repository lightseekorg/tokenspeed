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

"""``/update_weights_from_mooncake`` from the HTTP route down to the SDK call.

The Model Updater SDK is not a dependency: the adapter is exercised against a
fake module installed in ``sys.modules`` under the configured import path.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch
from fastapi.testclient import TestClient

from tokenspeed.runtime.engine.io_struct import (
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromMooncakeReqInput,
    mooncake_load_weight_version,
)
from tokenspeed.runtime.entrypoints import control_server
from tokenspeed.runtime.entrypoints.sglang_compat_http import (
    build_sglang_compat_app,
)
from tokenspeed.runtime.execution.device import DeviceHandle
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.model_update import (
    ModelUpdateAdapter,
    model_update_adapter_for,
)
from tokenspeed.runtime.utils.server_args import ServerArgs, prepare_server_args

# --------------------------------------------------------------------------
# HTTP route
# --------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, *, storage_backend=None) -> None:
        self.server_args = SimpleNamespace(
            weight_version="default",
            model="model-x",
            kvstore_storage_backend=storage_backend,
        )
        self.updates: list = []
        self.succeed = True

    async def update_weights_from_mooncake(self, obj):
        self.updates.append(obj)
        return self.succeed, "applied"


def _post(llm, body):
    return TestClient(build_sglang_compat_app(llm)).post(
        "/update_weights_from_mooncake", json=body
    )


def test_route_requires_an_integer_version():
    llm = _FakeLLM()
    for body in ({}, {"version": "3"}, {"version": 3.5}, {"version": True}):
        response = _post(llm, body)
        assert response.status_code == 400, body
        assert not response.json()["success"]
    assert llm.updates == []
    assert llm.server_args.weight_version == "default"


def test_flushed_load_defaults_the_namespace_to_the_version_and_stamps():
    llm = _FakeLLM()
    response = _post(llm, {"version": 7})
    assert response.status_code == 200
    (req,) = llm.updates
    assert isinstance(req, UpdateWeightsFromMooncakeReqInput)
    assert (req.version, req.flush_cache, req.weight_version) == (7, True, "7")
    assert llm.server_args.weight_version == "7"
    assert "Weight version updated to 7" in response.json()["message"]


def test_explicit_weight_version_and_unflushed_load():
    llm = _FakeLLM()
    response = _post(llm, {"version": 7, "weight_version": "ckpt-7"})
    assert response.status_code == 200
    assert llm.updates[-1].weight_version == "ckpt-7"
    assert llm.server_args.weight_version == "ckpt-7"

    response = _post(llm, {"version": 8, "flush_cache": False})
    assert response.status_code == 200
    assert llm.updates[-1].weight_version is None
    assert llm.server_args.weight_version == "ckpt-7"


def test_failure_leaves_the_version_alone():
    llm = _FakeLLM()
    llm.succeed = False
    response = _post(llm, {"version": 7})
    assert response.status_code == 400
    assert llm.server_args.weight_version == "default"


def test_request_fields_are_explicit_and_the_default_lives_in_one_helper():
    # The wire default is the HTTP route's; the request object and the
    # Engine API take every field explicitly.
    import inspect

    from tokenspeed.runtime.entrypoints.engine import Engine

    with pytest.raises(TypeError):
        UpdateWeightsFromMooncakeReqInput(version=3)
    flush_cache = inspect.signature(Engine.update_weights_from_mooncake).parameters[
        "flush_cache"
    ]
    assert flush_cache.kind is inspect.Parameter.KEYWORD_ONLY
    assert flush_cache.default is inspect.Parameter.empty
    assert (
        mooncake_load_weight_version(version=3, flush_cache=True, weight_version=None)
        == "3"
    )
    assert (
        mooncake_load_weight_version(version=3, flush_cache=False, weight_version=None)
        is None
    )
    assert (
        mooncake_load_weight_version(
            version=3, flush_cache=False, weight_version="ckpt"
        )
        == "ckpt"
    )


def test_control_server_proxies_the_route_with_a_long_read_timeout():
    routes = {
        (route.path, frozenset(route.methods or []))
        for route in control_server.app.routes
    }
    assert ("/update_weights_from_mooncake", frozenset({"POST"})) in routes
    assert control_server._WEIGHT_LOAD_PROXY_TIMEOUT.sock_read > (
        control_server._PROXY_TIMEOUT.sock_read
    )


# --------------------------------------------------------------------------
# Device dispatch: which models the SDK updates
# --------------------------------------------------------------------------


def _mooncake_req(version: int) -> UpdateWeightsFromMooncakeReqInput:
    return UpdateWeightsFromMooncakeReqInput(
        version=version, flush_cache=True, weight_version=None
    )


def _executor(*, draft_policy, with_draft: bool):
    target = SimpleNamespace(name="target")
    draft = SimpleNamespace(name="draft")
    runner = mock.MagicMock()
    runner.model = target
    runner.server_args = SimpleNamespace(model_update_draft_weights=draft_policy)
    runner.update_weights_from_mooncake.return_value = (True, "applied")
    forward_thread = mock.MagicMock()
    forward_thread.run.side_effect = lambda callback: callback()
    return SimpleNamespace(
        model_runner=runner,
        draft_model_runner=SimpleNamespace(model=draft) if with_draft else None,
        drafter=mock.MagicMock(spec=BaseDrafter) if with_draft else None,
        forward_thread=forward_thread,
    )


@pytest.mark.parametrize(
    ("draft_policy", "with_draft", "expected"),
    [
        ("retain", True, ["target"]),
        ("refresh", True, ["target", "draft"]),
        ("refresh", False, ["target"]),
    ],
)
def test_device_dispatches_the_model_list_by_draft_policy(
    draft_policy, with_draft, expected
):
    executor = _executor(draft_policy=draft_policy, with_draft=with_draft)
    handle = DeviceHandle(executor)
    req = _mooncake_req(3)

    assert handle.update_weights(req) == (True, "applied")

    (call,) = executor.model_runner.update_weights_from_mooncake.call_args_list
    version, models = call.args
    assert version == 3
    assert [model.name for model in models] == expected
    if with_draft:
        executor.drafter.on_target_weights_updated.assert_called_once_with()


def test_device_does_not_notify_the_drafter_on_failure():
    executor = _executor(draft_policy="retain", with_draft=True)
    executor.model_runner.update_weights_from_mooncake.return_value = (False, "no")
    handle = DeviceHandle(executor)

    assert handle.update_weights(_mooncake_req(3)) == (False, "no")
    executor.drafter.on_target_weights_updated.assert_not_called()


def test_device_still_notifies_after_a_distributed_update():
    executor = _executor(draft_policy="retain", with_draft=True)
    executor.model_runner.update_weights_from_distributed.return_value = (True, "ok")
    handle = DeviceHandle(executor)
    req = UpdateWeightsFromDistributedReqInput(
        names=[], dtype_names=[], shapes=[], flush_cache=True, weight_version=None
    )
    assert handle.update_weights(req) == (True, "ok")
    executor.drafter.on_target_weights_updated.assert_called_once_with()


# --------------------------------------------------------------------------
# Adapter and runner against a fake SDK module
# --------------------------------------------------------------------------

_CONFIG = {
    "weight_store_config": {"metadata_server": "etcd://x"},
    "local_host": "10.0.0.1",
    "hf_type": "deepseek_v3",
    "hf_safetensors_path": "/ckpt",
}


class _FakeSdk(ModuleType):
    """The surface the adapter uses, recording every construction."""

    def __init__(self) -> None:
        super().__init__("fake_model_updater")
        sdk = self
        self.stores: list = []
        self.updater_configs: list = []
        self.updates: list = []
        self.fail_with: Exception | None = None

        class EngineType:
            members = {"FLUENT_LLM": "fluent-llm"}

            def __class_getitem__(cls, name):
                return cls.members[name]

        class FluentLlmModelUpdateInitConfig(SimpleNamespace):
            @classmethod
            def from_dict(cls, data):
                return cls(**data)

        class FluentLlmEngineConfig(SimpleNamespace):
            pass

        class ModelUpdaterConfig(SimpleNamespace):
            pass

        class MooncakeWeightStore:
            def __init__(self, store_config, *, local_host):
                self.store_config = store_config
                self.local_host = local_host
                sdk.stores.append(self)

        class _Updater:
            def update_weights(self, models, *, version):
                if sdk.fail_with is not None:
                    raise sdk.fail_with
                for model in models:
                    model.load_weights([("a", torch.ones(1))])
                    model.load_weights([("b", torch.ones(1))])
                sdk.updates.append((models, version))
                return {"version": version}

        def make_model_updater(config, *, weight_store):
            sdk.updater_configs.append((config, weight_store))
            return _Updater()

        self.EngineType = EngineType
        self.FluentLlmModelUpdateInitConfig = FluentLlmModelUpdateInitConfig
        self.FluentLlmEngineConfig = FluentLlmEngineConfig
        self.ModelUpdaterConfig = ModelUpdaterConfig
        self.MooncakeWeightStore = MooncakeWeightStore
        self.make_model_updater = make_model_updater


@pytest.fixture
def sdk(monkeypatch):
    module = _FakeSdk()
    monkeypatch.setitem(sys.modules, "fake_model_updater", module)
    return module


def _adapter(reader_rank=5, engine_type="fluent_llm", module="fake_model_updater"):
    return ModelUpdateAdapter(
        sdk_module=module,
        config_json=json.dumps(_CONFIG),
        engine_type=engine_type,
        reader_rank=reader_rank,
    )


def test_adapter_builds_the_sdk_client_once_on_first_update(sdk):
    adapter = _adapter(reader_rank=5)
    assert sdk.stores == []
    model = mock.MagicMock()

    assert adapter.update([model], 9) == "{'version': 9}"
    assert adapter.update([model], 10) == "{'version': 10}"

    (store,) = sdk.stores
    assert (store.store_config, store.local_host) == (
        _CONFIG["weight_store_config"],
        "10.0.0.1",
    )
    ((config, weight_store),) = sdk.updater_configs
    assert weight_store is store
    assert config.role == "target"
    assert config.reader_rank == 5
    assert config.engine_type == "fluent-llm"
    assert (config.engine_config.hf_type, config.engine_config.hf_safetensors_path) == (
        "deepseek_v3",
        "/ckpt",
    )
    assert [version for _, version in sdk.updates] == [9, 10]


def test_adapter_reports_a_missing_sdk_module_clearly():
    adapter = _adapter(module="definitely_not_installed_sdk")
    with pytest.raises(ImportError, match="--model-update-sdk-module"):
        adapter.update([mock.MagicMock()], 1)


def test_adapter_rejects_an_unknown_engine_type(sdk):
    adapter = _adapter(engine_type="other_engine")
    with pytest.raises(ValueError, match="--model-update-engine-type"):
        adapter.update([mock.MagicMock()], 1)


def _runner(server_args, global_rank=3, is_draft_worker=False):
    runner = object.__new__(ModelRunner)
    runner.server_args = server_args
    runner.global_rank = global_rank
    runner.gpu_id = 0
    runner.model_update = model_update_adapter_for(
        server_args, global_rank=global_rank, is_draft_worker=is_draft_worker
    )
    return runner


def _model_update_args(**overrides):
    args = dict(
        model_update_config=json.dumps(_CONFIG),
        model_update_sdk_module="fake_model_updater",
        model_update_engine_type="fluent_llm",
        model_update_draft_weights="retain",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_runner_uses_the_global_rank_as_reader_rank_and_wraps_a_session(
    sdk, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "empty_cache", mock.Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", mock.Mock())
    runner = _runner(_model_update_args(), global_rank=3)

    class _Model:
        def __init__(self):
            self.calls: list = []

        def load_weights(self, weights):
            self.calls.append([name for name, _ in weights])

    model = _Model()
    ok, message = runner.update_weights_from_mooncake(4, [model])

    assert ok, message
    assert "4" in message
    assert sdk.updater_configs[0][0].reader_rank == 3
    assert model.calls == [["a"], ["b"]]
    torch.cuda.empty_cache.assert_called_once_with()
    torch.cuda.synchronize.assert_called_once()


def test_runner_turns_an_sdk_error_into_a_failed_result(sdk, monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", mock.Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", mock.Mock())
    sdk.fail_with = RuntimeError("store unreachable")
    runner = _runner(_model_update_args())

    assert runner.update_weights_from_mooncake(4, [mock.MagicMock()]) == (
        False,
        "store unreachable",
    )
    # The SDK's staging buffers are released on the failure path too.
    torch.cuda.empty_cache.assert_called_once_with()


def test_runner_without_model_update_config_fails_the_request():
    runner = _runner(SimpleNamespace(model_update_config=None))
    ok, message = runner.update_weights_from_mooncake(4, [mock.MagicMock()])
    assert not ok
    assert "--model-update-config" in message


def test_only_the_target_runner_owns_an_adapter():
    args = _model_update_args()
    assert isinstance(
        model_update_adapter_for(args, global_rank=0, is_draft_worker=False),
        ModelUpdateAdapter,
    )
    assert model_update_adapter_for(args, global_rank=0, is_draft_worker=True) is None
    draft = _runner(args, is_draft_worker=True)
    ok, message = draft.update_weights_from_mooncake(4, [mock.MagicMock()])
    assert not ok
    assert "target runner" in message


# --------------------------------------------------------------------------
# Server args
# --------------------------------------------------------------------------

_FLAGS = [
    "--model-update-sdk-module",
    "fake_model_updater",
    "--model-update-engine-type",
    "fluent_llm",
    "--model-update-draft-weights",
    "retain",
]


def test_server_args_parse_the_model_update_flags_together():
    args = prepare_server_args(
        ["model-x", "--model-update-config", json.dumps(_CONFIG), *_FLAGS]
    )
    assert json.loads(args.model_update_config) == _CONFIG
    assert args.model_update_sdk_module == "fake_model_updater"
    assert args.model_update_engine_type == "fluent_llm"
    assert args.model_update_draft_weights == "retain"


def test_server_args_default_to_no_model_update():
    args = ServerArgs(model="model-x")
    assert args.model_update_config is None
    assert args.model_update_sdk_module is None
    assert args.model_update_engine_type is None
    assert args.model_update_draft_weights is None


def test_server_args_require_every_companion_flag_with_the_config():
    with pytest.raises(ValueError, match="--model-update-draft-weights"):
        prepare_server_args(
            ["model-x", "--model-update-config", json.dumps(_CONFIG), *_FLAGS[:4]]
        )


def test_server_args_reject_companion_flags_without_the_config():
    with pytest.raises(ValueError, match="require --model-update-config"):
        prepare_server_args(["model-x", *_FLAGS])


def test_server_args_reject_a_non_object_config():
    with pytest.raises(ValueError, match="JSON object"):
        prepare_server_args(["model-x", "--model-update-config", "[1]", *_FLAGS])
