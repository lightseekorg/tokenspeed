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

"""The ``UpdateWeightsFromMooncake`` control operation from the device down
to the SDK call.

The Model Updater SDK is not a dependency: the adapter is exercised against a
fake module installed in ``sys.modules`` under the configured import path.
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pipe
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.distributed.process_group_manager import (  # noqa: E402
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromMooncakeReqInput,
)
from tokenspeed.runtime.execution.device import DeviceHandle  # noqa: E402
from tokenspeed.runtime.execution.drafter.base import BaseDrafter  # noqa: E402
from tokenspeed.runtime.execution.model_runner import ModelRunner  # noqa: E402
from tokenspeed.runtime.execution.model_update import (  # noqa: E402
    ModelUpdateAdapter,
    model_update_adapter_for,
)
from tokenspeed.runtime.execution.weight_memory import (  # noqa: E402
    WeightMemoryState,
)
from tokenspeed.runtime.models.base import BaseCausalLM  # noqa: E402
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    ServerArgs,
    prepare_server_args,
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
    return SimpleNamespace(
        model_runner=runner,
        draft_model_runner=SimpleNamespace(model=draft) if with_draft else None,
        drafter=mock.MagicMock(spec=BaseDrafter) if with_draft else None,
        device="cpu",
        # The weight update is a data-plane task: it finishes a pending draft
        # round first; none is pending here.
        complete_pending_drafting=lambda: None,
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
    assert call.kwargs == {"require_complete": False}
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
    runner.mapping = Mapping(rank=global_rank, world_size=8)
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
    ok, message = runner.update_weights_from_mooncake(
        4, [model], require_complete=False
    )

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

    assert runner.update_weights_from_mooncake(
        4, [mock.MagicMock()], require_complete=False
    ) == (
        False,
        "store unreachable",
    )
    # The SDK's staging buffers are released on the failure path too.
    torch.cuda.empty_cache.assert_called_once_with()


def test_runner_without_model_update_config_fails_the_request():
    runner = _runner(SimpleNamespace(model_update_config=None))
    ok, message = runner.update_weights_from_mooncake(
        4, [mock.MagicMock()], require_complete=False
    )
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
    ok, message = draft.update_weights_from_mooncake(
        4, [mock.MagicMock()], require_complete=False
    )
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


class _TinyDiscardModel(BaseCausalLM):
    """A complete streamed checkpoint and derived buffer, all on the CPU."""

    supports_weight_discard = True

    def __init__(self):
        super().__init__(
            config=SimpleNamespace(), mapping=SimpleNamespace(), encoder_only=True
        )
        self.a = torch.nn.Parameter(torch.tensor([4.0]))
        self.b = torch.nn.Parameter(torch.tensor([5.0]))
        self.register_buffer("constant", torch.tensor([7]))
        self.register_buffer("total", torch.tensor([9.0]))

    def required_weight_update_names(self):
        return {"a", "b"}

    def weight_discard_preserved_tensors(self):
        return (self.constant,)

    def load_weights(self, weights):
        loaded = set()
        with torch.no_grad():
            for name, value in weights:
                getattr(self, name).copy_(value)
                loaded.add(name)
        self.post_load_weights()
        return loaded

    def post_load_weights(self):
        with torch.no_grad():
            self.total.copy_(self.a + self.b)


def _discard_device(monkeypatch):
    # Only transport/device operations are replaced. The handle, runner,
    # updater adapter, session, model and validity state are the real classes.
    monkeypatch.setattr(torch.cuda, "empty_cache", mock.Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", mock.Mock())
    model = _TinyDiscardModel()
    state = WeightMemoryState(policy="discard", models=[model])
    runner = _runner(_model_update_args())
    runner.model = model
    executor = _executor(draft_policy="retain", with_draft=False)
    executor.model_runner = runner
    return DeviceHandle(executor, weight_memory=state), state, model


def _lose_weight_pages(state, model):
    state.mark_kv_released()

    def lose_pages():
        with torch.no_grad():
            model.a.fill_(float("nan"))
            model.b.fill_(float("nan"))
            model.total.fill_(float("nan"))
            model.constant.zero_()

    state.release(lose_pages)


def test_discarded_device_waits_for_global_commit_before_kv_wake(sdk, monkeypatch):
    handle, state, model = _discard_device(monkeypatch)
    address = model.a.data_ptr()
    _lose_weight_pages(state, model)
    state.resume(lambda: None)
    assert model.constant.item() == 7
    with pytest.raises(RuntimeError, match="unavailable"):
        handle.before_kv_resume()

    success, message = handle.update_weights(_mooncake_req(11))
    assert success, message
    assert model.a.item() == model.b.item() == 1
    assert model.total.item() == 2
    assert model.a.data_ptr() == address
    # Even a fully loaded local checkpoint stays unavailable until Rust
    # reports success across the whole replica and all DP ranks.
    with pytest.raises(RuntimeError, match="unavailable"):
        handle.before_kv_resume()
    handle.commit_weight_update(True)
    handle.before_kv_resume()
    state.require_usable()
    assert sdk.updates[0][1] == 11


@pytest.mark.parametrize("remote_failure", [False, True])
def test_discarded_device_poisoned_by_local_or_remote_update_failure(
    sdk, monkeypatch, remote_failure
):
    handle, state, model = _discard_device(monkeypatch)
    _lose_weight_pages(state, model)
    state.resume(lambda: None)
    if not remote_failure:
        sdk.fail_with = RuntimeError("weight store read failed")
    success, message = handle.update_weights(_mooncake_req(11))
    assert success is remote_failure
    if not remote_failure:
        assert "weight store read failed" in message
    handle.commit_weight_update(False)
    assert state.failed
    with pytest.raises(RuntimeError, match="restart"):
        handle.before_kv_resume()
    with pytest.raises(RuntimeError, match="restart"):
        state.require_usable()
    sdk.fail_with = None
    success, message = handle.update_weights(_mooncake_req(12))
    assert not success and "restart" in message


@pytest.mark.parametrize("unmapped", [False, True])
def test_discarded_device_rejects_update_before_kv_release_or_weight_remap(
    sdk, monkeypatch, unmapped
):
    handle, state, model = _discard_device(monkeypatch)
    if unmapped:
        _lose_weight_pages(state, model)
    # The control plane's admission check refuses without touching anything.
    ready, reason = handle.check_weight_update(_mooncake_req(11))
    assert not ready
    assert ("Resume weight storage" if unmapped else "Release KV") in reason
    assert not state.failed and not state.updating
    # A direct call refuses the same way.
    success, message = handle.update_weights(_mooncake_req(11))
    assert (success, message) == (False, reason)
    assert sdk.stores == []
    assert sdk.updates == []
    assert not state.failed and not state.updating
    if unmapped:
        state.resume(lambda: None)
    else:
        # Never released: the old weights stay usable.
        state.require_usable()
        assert (model.a.item(), model.b.item()) == (4, 5)
        state.mark_kv_released()
    # Once every rank passes the check, the same request loads and commits.
    assert handle.check_weight_update(_mooncake_req(11))[0]
    assert not state.updating
    success, message = handle.update_weights(_mooncake_req(11))
    assert success, message
    handle.commit_weight_update(True)
    handle.before_kv_resume()
    assert (model.a.item(), model.b.item(), model.total.item()) == (1, 1, 2)


def test_discarded_device_refuses_partial_non_sdk_updates(sdk, monkeypatch):
    handle, state, model = _discard_device(monkeypatch)
    _lose_weight_pages(state, model)
    state.resume(lambda: None)
    success, message = handle.update_weights(
        UpdateWeightsFromDistributedReqInput(
            names=[], dtype_names=[], shapes=[], flush_cache=True, weight_version=None
        )
    )
    assert not success
    assert "complete Model Updater SDK update" in message
    assert sdk.updates == []
    assert not state.failed and not state.updating
    # The refusal touched nothing: a complete SDK update still loads.
    assert handle.check_weight_update(_mooncake_req(11))[0]
    success, message = handle.update_weights(_mooncake_req(11))
    assert success, message
    handle.commit_weight_update(True)
    handle.before_kv_resume()
    assert model.total.item() == 2


def test_device_without_memory_saver_admits_every_weight_load():
    handle = DeviceHandle(_executor(draft_policy="retain", with_draft=False))
    assert handle.check_weight_update(_mooncake_req(3)) == (
        True,
        "ready for weight update",
    )
    with pytest.raises(TypeError, match="weight-load request"):
        handle.check_weight_update(object())


@pytest.mark.parametrize(
    "memory_saver,protocol,rejected",
    [(True, None, True), (False, None, False), (True, 1, True), (True, 2, False)],
)
def test_host_bootstrap_checks_the_actual_binary_protocol_before_device_build(
    monkeypatch, memory_saver, protocol, rejected
):
    from tokenspeed.runtime.engine import rust_host

    built = object()
    build = mock.Mock(return_value=built)
    monkeypatch.setattr(rust_host, "HostBootstrap", build)
    receiver, sender = Pipe(duplex=False)
    try:
        args = SimpleNamespace(
            enable_memory_saver=memory_saver, disaggregation_mode=None
        )
        sender.send((args, None, None))
        call_args = (os.dup(receiver.fileno()), None)
        if protocol is not None:
            call_args += (protocol,)
        if rejected:
            with pytest.raises(RuntimeError, match="rebuild tokenspeed-engine"):
                rust_host.bootstrap(*call_args)
            build.assert_not_called()
        else:
            assert rust_host.bootstrap(*call_args) is built
            build.assert_called_once()
    finally:
        receiver.close()
        sender.close()


@pytest.fixture(autouse=True)
def _local_weight_update_group(monkeypatch):
    # These CPU adapter tests lack a process group; the multiprocess session
    # regression exercises real rank agreement and collective post-load hooks.
    monkeypatch.setattr(pg_manager, "get_process_group", lambda *a, **kw: None)
