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
    flashinfer_autotune_cache_path,
    load_flashinfer_autotune_cache,
    save_flashinfer_autotune_cache,
)


class _FakeTuner:
    def __init__(self) -> None:
        self.active = False
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
        yield

    autotuner_module.AutoTuner = AutoTuner
    autotuner_module.autotune = fake_autotune
    autotuner_module._collect_metadata = lambda: metadata

    flashinfer_module = types.ModuleType("flashinfer")
    flashinfer_module.__path__ = []
    flashinfer_module.autotuner = autotuner_module
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", autotuner_module)
    return tuner, calls


def test_cache_path_is_environment_and_config_scoped(monkeypatch, tmp_path) -> None:
    metadata = {"gpu": "GB300"}
    _install_fake_flashinfer(monkeypatch, metadata=metadata)
    monkeypatch.setenv("TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))

    config = {"model": "model-a", "tp": 8, "ep": 1}
    first = flashinfer_autotune_cache_path(config)
    same = flashinfer_autotune_cache_path({"ep": 1, "tp": 8, "model": "model-a"})
    other = flashinfer_autotune_cache_path({**config, "tp": 1, "ep": 8})

    assert first == same
    assert first != other
    assert first is not None
    assert Path(first).parent.parent == tmp_path
    assert Path(first).name == "autotune_configs.json"
    metadata["gpu"] = "B300"
    assert flashinfer_autotune_cache_path(config) != first


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
    path = flashinfer_autotune_cache_path({"model": "model-a"})
    assert save_flashinfer_autotune_cache(path, None, 0)
    assert Path(path).read_bytes() == b"tactics"
    assert load_flashinfer_autotune_cache(path, None, 0)
    assert tuner.active

    assert not load_flashinfer_autotune_cache(str(tmp_path / "missing.json"), None, 0)
    assert not tuner.active

    def failed_load(path):
        tuner.active = True
        raise KeyError("malformed tactic")

    monkeypatch.setattr(tuner, "load_configs", failed_load)
    assert not load_flashinfer_autotune_cache(path, None, 0)
    assert not tuner.active

    monkeypatch.setattr(
        tuner, "save_configs", Mock(side_effect=TypeError("invalid tactic"))
    )
    assert not save_flashinfer_autotune_cache(path, None, 0)


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
