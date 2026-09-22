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

from __future__ import annotations

import contextlib
import sys
import threading
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel.ops import tuning
from tokenspeed_kernel.ops.tuning import (
    autotune,
    autotune_cache_path,
    load_autotune_cache,
    save_autotune_cache,
)


class _FakeTuner:
    def __init__(self) -> None:
        self.active = False
        self.is_tuning_mode = False
        self._blocklist = types.SimpleNamespace(_invalid={})

    def load_configs(self, path: str) -> bool:
        assert Path(path).read_bytes() == b"tactics"
        self.active = True
        return True

    def save_configs(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"tactics")

    def clear_cache(self) -> None:
        self.active = False


def _install_fake_flashinfer(monkeypatch, *, metadata):
    tuner = _FakeTuner()
    calls = []

    autotuner_module = types.ModuleType("flashinfer.autotuner")

    class AutoTuner:
        choose_one = Mock()

        @staticmethod
        def get():
            return tuner

    @contextlib.contextmanager
    def fake_autotune(tune_mode, **kwargs):
        calls.append((tune_mode, kwargs))
        previous, tuner.is_tuning_mode = tuner.is_tuning_mode, tune_mode
        try:
            yield
        finally:
            tuner.is_tuning_mode = previous

    autotuner_module.AutoTuner = AutoTuner
    autotuner_module.autotune = fake_autotune
    autotuner_module._collect_metadata = lambda: metadata

    flashinfer_module = types.ModuleType("flashinfer")
    flashinfer_module.__path__ = []
    flashinfer_module.autotuner = autotuner_module
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner_module)
    monkeypatch.setattr(tuning, "_autotuner", autotuner_module)
    return tuner, calls


def test_cache_path_is_environment_and_config_scoped(monkeypatch, tmp_path) -> None:
    metadata = {"gpu": "GB300"}
    _install_fake_flashinfer(monkeypatch, metadata=metadata)
    monkeypatch.setenv("TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))

    config = {"model": "model-a", "tp": 8, "ep": 1}
    first = autotune_cache_path(config)
    same = autotune_cache_path({"ep": 1, "tp": 8, "model": "model-a"})
    other = autotune_cache_path({**config, "tp": 1, "ep": 8})

    assert first == same
    assert first != other
    assert first is not None
    assert Path(first).parent.parent == tmp_path
    assert Path(first).name == "autotune_configs.json"
    metadata["gpu"] = "B300"
    assert autotune_cache_path(config) != first


def test_autotune_forwards_decode_bucket_override(monkeypatch) -> None:
    tuner, calls = _install_fake_flashinfer(monkeypatch, metadata={})

    with autotune(
        tune_mode=True,
        tuning_buckets=(64, 1, 2, 2),
        round_up=False,
    ):
        pass
    with autotune(tune_mode=False, tuning_buckets=(1, 2, 4), round_up=None):
        pass

    assert calls == [
        (
            True,
            {
                "tuning_buckets": (64, 1, 2, 2),
                "round_up": False,
            },
        ),
        (False, {"tuning_buckets": (1, 2, 4), "round_up": None}),
    ]
    assert tuner._blocklist._invalid["bf16_gemm::TGVRunner"] == set(range(16, 29))


def test_cache_roundtrip_and_failures(monkeypatch, tmp_path) -> None:
    tuner, _ = _install_fake_flashinfer(monkeypatch, metadata={})
    monkeypatch.setenv("TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    path = autotune_cache_path({"model": "model-a"})
    assert save_autotune_cache(path, None, 0)
    assert Path(path).read_bytes() == b"tactics"
    assert load_autotune_cache(path, None, 0)
    assert tuner.active

    assert not load_autotune_cache(str(tmp_path / "missing.json"), None, 0)
    assert not tuner.active

    def failed_load(path):
        tuner.active = True
        raise KeyError("malformed tactic")

    monkeypatch.setattr(tuner, "load_configs", failed_load)
    assert not load_autotune_cache(path, None, 0)
    assert not tuner.active

    monkeypatch.setattr(
        tuner, "save_configs", Mock(side_effect=TypeError("invalid tactic"))
    )
    assert not save_autotune_cache(path, None, 0)


@pytest.mark.parametrize(
    "tokens,local,queries",
    [(512, 112, [512, 64]), (513, 112, [513, 65]), (512, 896, [512])],
)
def test_ep_candidates_keep_full_profile_inputs(monkeypatch, tokens, local, queries):
    seen = []
    inputs = [torch.empty(tokens), torch.empty(tokens, 4)]

    class MoERunner:
        num_experts, num_local_experts = 896, local
        num_fused_shared_experts = 0

        def get_valid_tactics(self, tensors, profile):
            seen.append(tensors[1].shape[0])
            assert tensors[0] is inputs[0]
            return [(128, 17)] if tensors[1].shape[0] > 100 else [(16, 377), (128, 17)]

    class AutoTuner:
        _lock = threading.RLock()

        def choose_one(self, op, runners, config, tensors, **kwargs):
            candidates = runners[0].get_valid_tactics(tensors, None)
            assert tensors is inputs and tensors[1].shape[0] == tokens
            if kwargs.get("fail"):
                raise RuntimeError("profile failed")
            return candidates

    monkeypatch.setitem(
        sys.modules, "flashinfer.autotuner", types.SimpleNamespace(AutoTuner=AutoTuner)
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.fused_moe.core",
        types.SimpleNamespace(
            MoeRunnerInputs=types.SimpleNamespace(idx=lambda name: 1)
        ),
    )
    monkeypatch.setattr(
        tuning, "_autotuner", types.SimpleNamespace(AutoTuner=AutoTuner)
    )
    original_choose, original_valid = AutoTuner.choose_one, MoERunner.get_valid_tactics
    with tuning._ep_moe_candidates():
        args = ("flashinfer::trtllm_fp4_block_scale_moe", [MoERunner()], None, inputs)
        result = AutoTuner().choose_one(*args)
        assert seen == queries
        assert result == ([(128, 17), (16, 377)] if local < 896 else [(128, 17)])
        with pytest.raises(RuntimeError, match="profile failed"):
            AutoTuner().choose_one(*args, fail=True)
        assert MoERunner.get_valid_tactics is original_valid
    assert AutoTuner.choose_one is original_choose


def test_tuning_mode_is_owned_by_flashinfer(monkeypatch):
    _install_fake_flashinfer(monkeypatch, metadata={})
    assert not tuning.is_autotuning()
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        assert tuning.is_autotuning()
        with pytest.raises(RuntimeError), autotune(
            tune_mode=False, tuning_buckets=None, round_up=None
        ):
            assert not tuning.is_autotuning()
            raise RuntimeError("body failed")
        assert tuning.is_autotuning()
    assert not tuning.is_autotuning()


def test_missing_flashinfer_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(tuning, "_autotuner", None)
    assert tuning.autotune_cache_path({}) is None
    assert not tuning.load_autotune_cache(str(tmp_path / "cache.json"), None, 0)
    assert not tuning.save_autotune_cache(str(tmp_path / "cache.json"), None, 0)
    tuning.set_autotune_process_group(None)
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        assert not tuning.is_autotuning()


def test_peer_cache_failure_discards_local_results(monkeypatch, tmp_path):
    tuner, _ = _install_fake_flashinfer(monkeypatch, metadata={})
    path = str(tmp_path / "configs.json")
    assert save_autotune_cache(path, None, 0)
    monkeypatch.setattr(tuning.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(tuning.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        tuning.dist, "broadcast_object_list", lambda *args, **kwargs: None
    )

    def disagree(states, loaded, *, group):
        states[:] = [loaded, False]

    monkeypatch.setattr(tuning.dist, "all_gather_object", disagree)
    assert not load_autotune_cache(path, object(), 0)
    assert not tuner.active


def test_default_cache_directory_is_flashinfer_autotune(monkeypatch, tmp_path):
    _install_fake_flashinfer(monkeypatch, metadata={"gpu": "B300"})
    monkeypatch.delenv("TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = autotune_cache_path({"model": "model-a"})
    assert Path(path).parent.parent == tmp_path / "tokenspeed" / "flashinfer-autotune"
    assert Path(path).name == "autotune_configs.json"


@pytest.mark.parametrize(
    "rank,local_payload",
    [(None, b"tactics"), (0, b"tactics"), (1, b"tactics"), (1, b"stale"), (1, None)],
)
def test_read_only_cache_load_does_not_rewrite_files(
    monkeypatch, tmp_path, rank, local_payload
):
    tuner, _ = _install_fake_flashinfer(monkeypatch, metadata={})
    directory = tmp_path / "read-only"
    directory.mkdir()
    path = directory / "configs.json"
    if local_payload is not None:
        path.write_bytes(local_payload)
        path.chmod(0o444)
    directory.chmod(0o555)
    installer = Mock(side_effect=AssertionError("loading must not rewrite the cache"))
    monkeypatch.setattr(tuning, "_install_autotune_cache_bytes", installer)
    loader = Mock(wraps=tuner.load_configs)
    monkeypatch.setattr(tuner, "load_configs", loader)
    group = None if rank is None else object()
    if group is not None:
        monkeypatch.setattr(tuning.dist, "get_rank", lambda: rank)
        monkeypatch.setattr(tuning.dist, "get_world_size", lambda group: 2)

        def broadcast(payload_box, *, src, group):
            assert src == 0
            assert payload_box == ([b"tactics"] if rank == 0 else [None])
            payload_box[0] = b"tactics"

        def gather(states, loaded, *, group):
            states[:] = [loaded, loaded]

        monkeypatch.setattr(tuning.dist, "broadcast_object_list", broadcast)
        monkeypatch.setattr(tuning.dist, "all_gather_object", gather)
    try:
        assert load_autotune_cache(str(path), group, 0)
        assert tuner.active
        installer.assert_not_called()
        loaded_path = Path(loader.call_args.args[0])
        if rank in (None, 0):
            assert loaded_path == path
        else:
            assert loaded_path != path
            assert not loaded_path.exists()
        if local_payload is None:
            assert not path.exists()
        else:
            assert path.read_bytes() == local_payload
    finally:
        directory.chmod(0o755)
        if path.exists():
            path.chmod(0o644)
