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

import json

import pytest
import tokenspeed_kernel.profiling as profiling
import torch


class _FakeScope:
    def __init__(self, log: list[str]):
        self._log = log

    def __enter__(self):
        self._log.append("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type, exc, tb
        self._log.append("exit")


class _FakeProton:
    def __init__(self):
        self.start_calls: list[tuple] = []
        self.finalize_calls: list[tuple] = []
        self.scope_calls: list[tuple[str, dict[str, object]]] = []
        self.scope_log: list[str] = []

    def start(self, *args, **kwargs):
        self.start_calls.append((args, kwargs))
        return 123

    def finalize(self, *args, **kwargs):
        self.finalize_calls.append((args, kwargs))

    def scope(self, name: str, metrics: dict[str, object] | None = None):
        self.scope_calls.append((name, {} if metrics is None else dict(metrics)))
        return _FakeScope(self.scope_log)


@pytest.fixture(autouse=True)
def _reset_state():
    profiling.stop_profiling()
    profiling.ProfilingState.reset()
    capture = profiling.ShapeCapture.get()
    capture.enabled = False
    capture.clear()
    profiling.ShapeCapture.reset()
    profiling._BOOTSTRAPPED = False
    yield
    profiling.stop_profiling()
    profiling.ProfilingState.reset()
    capture = profiling.ShapeCapture.get()
    capture.enabled = False
    capture.clear()
    profiling.ShapeCapture.reset()
    profiling._BOOTSTRAPPED = False


def test_start_and_stop_calls_proton(monkeypatch):
    fake = _FakeProton()
    monkeypatch.setattr(profiling, "_HAS_PROTON", True)
    monkeypatch.setattr(profiling, "proton", fake)

    cfg = profiling.ProfilingConfig(
        output="run_profile",
        data="trace",
        backend="cupti",
        mode="pcsampling",
        hook="triton",
        output_format="chrome_trace",
    )
    session = profiling.start_profiling(cfg)
    assert session == 123
    assert profiling.ProfilingState.get().active

    profiling.stop_profiling()
    assert not profiling.ProfilingState.get().active

    assert fake.start_calls == [
        (
            ("run_profile",),
            {
                "data": "trace",
                "backend": "cupti",
                "mode": "pcsampling",
                "hook": "triton",
            },
        )
    ]
    assert fake.finalize_calls == [((123, "chrome_trace"), {})]


def test_start_warns_when_proton_missing(monkeypatch):
    monkeypatch.setattr(profiling, "_HAS_PROTON", False)
    monkeypatch.setattr(profiling, "proton", None)

    with pytest.warns(UserWarning, match="Proton not installed"):
        session = profiling.start_profiling()

    assert session is None
    assert not profiling.ProfilingState.get().active


def test_kernel_scope_noop_when_inactive():
    scope = profiling.kernel_scope(
        "gemm",
        "mm",
        torch.float16,
        kernel_name="triton_mm_fp8_scaled",
        M=16,
        N=32,
        K=64,
    )
    assert scope is profiling._NOOP_SCOPE
    with scope:
        pass


def test_kernel_scope_uses_proton_scope_when_active(monkeypatch):
    fake = _FakeProton()
    monkeypatch.setattr(profiling, "_HAS_PROTON", True)
    monkeypatch.setattr(profiling, "proton", fake)
    profiling.start_profiling()

    with profiling.kernel_scope(
        "gemm",
        "mm",
        torch.float16,
        kernel_name="triton_mm_fp8_scaled",
        M=32,
        N=64,
        K=128,
    ):
        pass

    assert fake.scope_calls == [
        (
            "gemm.mm[triton_mm_fp8_scaled]",
            {
                "dtype": "torch.float16",
                "M": 32,
                "N": 64,
                "K": 128,
            },
        )
    ]
    assert fake.scope_log == ["enter", "exit"]


def test_bootstrap_reads_env_and_only_runs_once(monkeypatch):
    fake = _FakeProton()
    registrations: list[object] = []
    monkeypatch.setattr(profiling, "_HAS_PROTON", True)
    monkeypatch.setattr(profiling, "proton", fake)
    monkeypatch.setattr(
        profiling.atexit,
        "register",
        lambda fn: registrations.append(fn),
    )
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE", "1")
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE_OUTPUT", "env_profile")
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE_DATA", "trace")

    profiling.bootstrap_profiling_from_env()
    profiling.bootstrap_profiling_from_env()

    assert len(fake.start_calls) == 1
    assert fake.start_calls[0] == (
        ("env_profile",),
        {
            "data": "trace",
            "backend": None,
            "mode": None,
            "hook": "triton",
        },
    )
    assert len(registrations) == 2


def test_proton_available_reflects_import(monkeypatch):
    monkeypatch.setattr(profiling, "_HAS_PROTON", True)
    assert profiling.proton_available()

    monkeypatch.setattr(profiling, "_HAS_PROTON", False)
    assert not profiling.proton_available()


def test_profile_config_from_env_defaults(monkeypatch):
    for env in (
        "TOKENSPEED_KERNEL_PROFILE_OUTPUT",
        "TOKENSPEED_KERNEL_PROFILE_DATA",
        "TOKENSPEED_KERNEL_PROFILE_BACKEND",
        "TOKENSPEED_KERNEL_PROFILE_MODE",
        "TOKENSPEED_KERNEL_PROFILE_HOOK",
        "TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT",
    ):
        monkeypatch.delenv(env, raising=False)

    cfg = profiling.profile_config_from_env()
    assert cfg == profiling.ProfilingConfig()


def test_profile_config_from_env_output_override_wins(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE_OUTPUT", "env_profile")
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE_DATA", "trace")
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE_BACKEND", "roctracer")
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE_MODE", "periodic_flushing")
    monkeypatch.setenv("TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT", "chrome_trace")

    cfg = profiling.profile_config_from_env(output="rank0/step5.proton")

    assert cfg.output == "rank0/step5.proton"
    assert cfg.data == "trace"
    assert cfg.backend == "roctracer"
    assert cfg.mode == "periodic_flushing"
    assert cfg.hook == "triton"
    assert cfg.output_format == "chrome_trace"


def test_shape_capture_records_dump_and_clear(tmp_path):
    output = tmp_path / "shapes.json"

    profiling.start_shape_capture()
    cap = profiling.ShapeCapture.get()
    cap.record("gemm", "mm", "k1", torch.float16, {"M": 16, "N": 32, "K": 64})
    cap.record(
        "attention",
        "decode",
        "k2",
        torch.bfloat16,
        {"batch": 2, "seq_len": 128, "num_q_heads": 8, "head_dim": 64},
    )

    profiling.stop_shape_capture(output)

    payload = json.loads(output.read_text())
    assert len(payload) == 2
    assert payload[0]["family"] == "gemm"
    assert payload[0]["shape_params"] == {"K": 64, "M": 16, "N": 32}
    assert payload[1]["family"] == "attention"
    assert payload[1]["shape_params"]["batch"] == 2
    assert not profiling.ShapeCapture.get().enabled


def test_shape_capture_context_manager_writes_output(tmp_path):
    output = tmp_path / "ctx_shapes.json"

    with profiling.shape_capture(output):
        profiling.ShapeCapture.get().record(
            "gemm",
            "mm",
            "k1",
            torch.float32,
            {"M": 8, "N": 8, "K": 8},
        )

    payload = json.loads(output.read_text())
    assert len(payload) == 1
    assert payload[0]["kernel_name"] == "k1"


def test_bootstrap_shape_capture_from_env_and_atexit_dump(tmp_path, monkeypatch):
    registrations: list[object] = []
    output = tmp_path / "env_shapes.json"

    monkeypatch.setattr(
        profiling.atexit,
        "register",
        lambda fn: registrations.append(fn),
    )
    monkeypatch.setenv("TOKENSPEED_KERNEL_CAPTURE_SHAPES", "1")
    monkeypatch.setenv("TOKENSPEED_KERNEL_CAPTURE_SHAPES_OUTPUT", str(output))

    profiling.bootstrap_profiling_from_env()

    cap = profiling.ShapeCapture.get()
    assert cap.enabled
    cap.record("gemm", "mm", "k1", torch.float16, {"M": 4, "N": 4, "K": 4})

    for fn in registrations:
        fn()

    payload = json.loads(output.read_text())
    assert len(payload) == 1
    assert payload[0]["shape_params"] == {"K": 4, "M": 4, "N": 4}
    assert not profiling.ShapeCapture.get().enabled
