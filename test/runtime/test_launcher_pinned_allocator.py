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

"""Optional pinned-allocation tuning must remain valid for the native allocator."""

import json
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.utils import launcher  # noqa: E402

register_cuda_ci(est_time=240, suite="runtime-1gpu")

KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
SETTING = "pinned_max_round_threshold_mb:1024"
BACKGROUND = "pinned_use_background_threads:True"
pytestmark = pytest.mark.skipif(
    torch.version.cuda is None and torch.version.hip is None,
    reason="Requires a CUDA or ROCm allocator",
)


@pytest.fixture(autouse=True)
def allocator_environment(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def _native_supports(setting):
    # Read actual parser support independently of the launcher's defaults.
    # Isolate its mutable settings and keep the probe independent of GPU access.
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
    )
    for key in KEYS:
        env.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import torch; torch.cuda.memory._set_allocator_settings({setting!r})",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    name = setting.split(":", 1)[0]
    if result.returncode:
        assert (
            f"Unrecognized CachingAllocator option: {name}" in result.stderr
            or f"Unrecognized key '{name}'" in result.stderr
        ), (result.stdout + result.stderr)
    return result.returncode == 0


@pytest.fixture(scope="module")
def native_supports_threshold():
    return _native_supports(SETTING)


@pytest.fixture(scope="module")
def background_suffix():
    """What the launcher appends for the helper-thread setting, if supported."""
    return f",{BACKGROUND}" if _native_supports(BACKGROUND) else ""


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("config", ["", "max_split_size_mb:64,"])
def test_preserves_existing_options(
    monkeypatch, native_supports_threshold, background_suffix, key, config
):
    monkeypatch.setenv(key, config)
    launcher.set_pinned_allocator_default()
    added = (f",{SETTING}" if native_supports_threshold else "") + background_suffix
    existing = config.rstrip(",")
    expected = f"{existing}{added}" if existing else added.lstrip(",")
    assert os.environ[key] == (expected if added else config)
    assert all(other not in os.environ for other in KEYS if other != key)


def test_no_existing_variable(native_supports_threshold, background_suffix):
    launcher.set_pinned_allocator_default()
    added = (f",{SETTING}" if native_supports_threshold else "") + background_suffix
    assert os.environ.get("PYTORCH_ALLOC_CONF") == (added.lstrip(",") or None)
    assert all(key not in os.environ for key in KEYS[:2])


def test_explicit_settings_and_variable_precedence(monkeypatch, background_suffix):
    monkeypatch.setenv(KEYS[0], "pinned_max_round_threshold_mb:256")
    monkeypatch.setenv(KEYS[1], "max_split_size_mb:64")
    monkeypatch.setenv(KEYS[2], "max_split_size_mb:128")
    before = {key: os.environ[key] for key in KEYS}
    launcher.set_pinned_allocator_default()
    before[KEYS[0]] += background_suffix
    assert {key: os.environ[key] for key in KEYS} == before

    monkeypatch.setenv(
        KEYS[0],
        "pinned_max_round_threshold_mb:256,pinned_use_background_threads:False",
    )
    launcher.set_pinned_allocator_default()
    assert os.environ[KEYS[0]] == (
        "pinned_max_round_threshold_mb:256,pinned_use_background_threads:False"
    )


def test_highest_precedence_variable_gets_default(
    monkeypatch, native_supports_threshold, background_suffix
):
    monkeypatch.setenv(KEYS[0], "max_split_size_mb:64")
    monkeypatch.setenv(KEYS[1], "max_split_size_mb:128")
    monkeypatch.setenv(KEYS[2], "max_split_size_mb:256")
    launcher.set_pinned_allocator_default()
    suffix = (f",{SETTING}" if native_supports_threshold else "") + background_suffix
    assert os.environ[KEYS[0]] == f"max_split_size_mb:64{suffix}"
    assert os.environ[KEYS[1]] == "max_split_size_mb:128"
    assert os.environ[KEYS[2]] == "max_split_size_mb:256"


def test_probe_propagates_other_parser_errors():
    with pytest.raises(RuntimeError, match="Cannot probe pinned allocator setting"):
        launcher._supports_allocator_setting("max_split_size_mb:0")


def test_probe_timeout_is_propagated(tmp_path, monkeypatch):
    # A stalled interpreter exercises the real subprocess deadline and cleanup.
    blocked_python = tmp_path / "blocked-python"
    blocked_python.write_text("#!/bin/sh\nexec sleep 65\n")
    blocked_python.chmod(0o755)
    monkeypatch.setattr(launcher.sys, "executable", str(blocked_python))
    with pytest.raises(subprocess.TimeoutExpired):
        launcher._supports_allocator_setting(SETTING)


def test_fresh_workers_can_allocate_after_launcher_default(native_supports_threshold):
    # Load the standalone launcher directly to measure its CUDA initialization.
    # The legacy variable also reproduces the failure on older native parsers.
    code = '''
import importlib.util
import json
import os
import sys
import torch
spec = importlib.util.spec_from_file_location("_isolated_launcher", sys.argv[1])
isolated_launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = isolated_launcher
spec.loader.exec_module(isolated_launcher)
set_pinned_allocator_default = isolated_launcher.set_pinned_allocator_default
assert not torch.cuda.is_initialized(), "CUDA initialized before launcher call"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
set_pinned_allocator_default()
assert not torch.cuda.is_initialized(), "Launcher initialized CUDA"
pid = os.fork()
if pid == 0:
    tensor = torch.ones(1, device="cuda")
    assert tensor.item() == 1
    settings = torch.cuda.memory._snapshot()["allocator_settings"]
    assert settings["max_split_size"] == 64 * 1024 * 1024
    if os.environ["EXPECT_PINNED_THRESHOLD"] == "1":
        assert settings["max_round_threshold"] == 1024 * 1024 * 1024
    staged = tensor.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    assert staged.item() == 1
    del staged
    assert torch.empty(1, pin_memory=True).is_pinned()
    os._exit(0)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
worker = """
import os
import torch
tensor = torch.ones(1, device="cuda")
assert tensor.item() == 1
settings = torch.cuda.memory._snapshot()["allocator_settings"]
assert settings["max_split_size"] == 64 * 1024 * 1024
if os.environ["EXPECT_PINNED_THRESHOLD"] == "1":
    assert settings["max_round_threshold"] == 1024 * 1024 * 1024
staged = tensor.to("cpu", non_blocking=True)
torch.cuda.synchronize()
assert staged.item() == 1
del staged
assert torch.empty(1, pin_memory=True).is_pinned()
"""
result = os.spawnve(os.P_WAIT, sys.executable,
    [sys.executable, "-c", worker],
    os.environ)
assert result == 0
print(json.dumps({"fork": "passed", "fresh_process": "passed"}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code, launcher.__file__],
        env=dict(
            os.environ, EXPECT_PINNED_THRESHOLD=str(int(native_supports_threshold))
        ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "fork": "passed",
        "fresh_process": "passed",
    }
