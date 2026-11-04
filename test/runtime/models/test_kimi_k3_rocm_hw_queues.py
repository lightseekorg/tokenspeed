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

"""Kimi-K3 launches cap ROCm at two hardware queues per stream priority.

See ``launcher.set_rocm_hw_queue_default``. CPU-only: engine setup reads the
checkpoint's architectures, the torch build and the environment.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

from tokenspeed.runtime.entrypoints import engine as engine_module

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.models.conftest import server_args as _server_args  # noqa: E402

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

KEY = "GPU_MAX_HW_QUEUES"


@pytest.fixture
def environ(monkeypatch):
    """Engine setup on a private copy of the environment without the queue
    cap; its process-wide signal handler, start method, file limit and socket
    interface are left alone."""
    env = {key: value for key, value in os.environ.items() if key != KEY}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(engine_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(engine_module.mp, "set_start_method", lambda *a, **k: None)
    monkeypatch.setattr(engine_module, "set_ulimit", lambda: None)
    monkeypatch.setattr(engine_module, "_set_socket_interface", lambda args: None)
    return env


def _checkpoint(root, architecture):
    path = root / architecture
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    return path


@pytest.mark.parametrize(
    "architecture",
    ["KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration"],
)
@pytest.mark.parametrize(
    "hip, caller, expected",
    [("7.2", {}, "2"), ("7.2", {KEY: "4"}, "4"), (None, {}, None)],
    ids=["rocm", "explicit-keep", "cuda"],
)
def test_kimi_k3_caps_rocm_hw_queues(
    tmp_path, monkeypatch, environ, architecture, hip, caller, expected
):
    monkeypatch.setattr(torch.version, "hip", hip)
    environ.update(caller)

    engine_module._set_envs_and_config(
        _server_args(_checkpoint(tmp_path, architecture))
    )

    assert environ.get(KEY) == expected


def test_other_models_keep_the_hip_default(tmp_path, monkeypatch, environ):
    monkeypatch.setattr(torch.version, "hip", "7.2")

    engine_module._set_envs_and_config(
        _server_args(_checkpoint(tmp_path, "Qwen3ForCausalLM"))
    )

    assert KEY not in environ
