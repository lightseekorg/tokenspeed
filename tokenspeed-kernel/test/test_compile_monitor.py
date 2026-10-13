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

import importlib.machinery
import importlib.util
import logging
import os

import pytest
import tokenspeed_kernel.compile_monitor as compile_monitor
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.compile_monitor import (
    CompileMonitor,
    UnboundedSpecializationError,
)

KERNEL = "ops.example._kernel"
SITE = "runtime/models/example.py:12 (forward)"


@pytest.fixture(autouse=True)
def startup(monkeypatch):
    # mark_serving() closes the process-wide compile switch; reopen it afterwards.
    monkeypatch.setattr(compile_monitor, "_serving", False)


def _record(monitor, n, block=64, site=SITE):
    monitor.record(
        KERNEL,
        {"X": "*fp32 divisibility=16", "N": repr(n), "BLOCK": repr(block)},
        {"N": n, "BLOCK": block},
        0.1,
        site,
    )


def _warnings(caplog):
    return [r.message for r in caplog.records if r.levelno == logging.WARNING]


def test_startup_compiles_are_counted_but_not_reported(caplog):
    monitor = CompileMonitor("warn", 2)
    with caplog.at_level(logging.INFO, logger=compile_monitor.__name__):
        for n in (3, 5, 7, 9):
            _record(monitor, n)
    assert not caplog.records
    stats = monitor.stats()
    assert (stats.startup_compiles, stats.serving_compiles) == (4, 0)
    assert stats.startup_seconds == pytest.approx(0.4)


def test_serving_compile_names_what_changed(caplog):
    monitor = CompileMonitor("warn", 8)
    _record(monitor, 33)
    monitor.serving = True
    with caplog.at_level(logging.INFO, logger=compile_monitor.__name__):
        _record(monitor, 1483)
    (message,) = [r.message for r in caplog.records]
    assert f"compiled {KERNEL} while serving in 100 ms" in message
    assert f"(N: 33 -> 1483) from {SITE}" in message
    assert "BLOCK" not in message
    assert monitor.stats().serving_compiles == 1


def test_parameter_with_unbounded_values_is_named(caplog):
    monitor = CompileMonitor("warn", 3)
    _record(monitor, 33)
    monitor.serving = True
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        for n in (48, 97, 130):
            _record(monitor, n)
        assert not _warnings(caplog)
        _record(monitor, 1483)
    (message,) = _warnings(caplog)
    assert (
        f"{KERNEL}: compile-time parameter N has compiled 4 new values while "
        f"serving from {SITE} (48, 97, 130, 1483)"
    ) in message


def test_values_are_counted_per_call_site(caplog):
    monitor = CompileMonitor("warn", 2)
    monitor.serving = True
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        # Three layers, each launching with its own fixed dimension.
        for n, line in ((3, 10), (5, 20), (7, 30)):
            _record(monitor, n, site=f"runtime/models/example.py:{line} (f)")
        assert not _warnings(caplog)
        # One site whose value follows the batch.
        for n in (9, 11, 13):
            _record(monitor, n)
    (message,) = _warnings(caplog)
    assert f"compiled 3 new values while serving from {SITE}" in message


def test_report_repeats_only_when_the_count_doubles(caplog):
    monitor = CompileMonitor("warn", 2)
    monitor.serving = True
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        for n in range(3, 3 + 2 * 12, 2):
            _record(monitor, n)
    counts = [int(m.split(" has compiled ")[1].split()[0]) for m in _warnings(caplog)]
    assert counts == [3, 7]


def test_powers_of_two_and_startup_values_are_not_counted(caplog):
    monitor = CompileMonitor("warn", 2)
    for n in (48, 97, 130):
        _record(monitor, n)
    monitor.serving = True
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        # Already compiled at startup, or a power-of-two bucket.
        for n in (48, 97, 130, 256, 512, 1024, 2048, 4096):
            _record(monitor, n)
        # Only a bool changes; flags are bounded.
        monitor.record(KERNEL, {"FLAG": "True"}, {"FLAG": True}, 0.0, SITE)
        monitor.record(KERNEL, {"FLAG": "False"}, {"FLAG": False}, 0.0, SITE)
    assert not _warnings(caplog)


def test_float_parameters_are_counted(caplog):
    monitor = CompileMonitor("warn", 2)
    monitor.serving = True
    with caplog.at_level(logging.WARNING, logger=compile_monitor.__name__):
        for temperature in (0.6, 0.7, 0.8):
            monitor.record(
                KERNEL,
                {"TEMP": repr(temperature)},
                {"TEMP": temperature},
                0.0,
                SITE,
            )
    (message,) = _warnings(caplog)
    assert "parameter TEMP has compiled 3 new values while serving" in message
    assert "(0.6, 0.7, 0.8)" in message


def test_error_mode_raises():
    monitor = CompileMonitor("error", 2)
    monitor.serving = True
    _record(monitor, 3)
    _record(monitor, 5)
    with pytest.raises(UnboundedSpecializationError, match="parameter N"):
        _record(monitor, 7)


def test_serving_mark_closes_the_compile_switch_without_a_monitor(monkeypatch):
    monkeypatch.setattr(compile_monitor, "_hooks", None)
    assert not compile_monitor.is_serving()
    compile_monitor.mark_serving()
    assert compile_monitor.is_serving()


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="on_unbounded"):
        CompileMonitor("loud", 8)


def test_call_sites_skip_the_amd_kernel_package(monkeypatch, tmp_path):
    amd_dir = str(tmp_path / "tokenspeed_kernel_amd")
    spec = importlib.machinery.ModuleSpec("tokenspeed_kernel_amd", None)
    spec.submodule_search_locations = [amd_dir]
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: spec if name == "tokenspeed_kernel_amd" else None,
    )
    assert amd_dir + os.sep in compile_monitor._internal_dirs()


@triton.jit
def _fill_kernel(X, n, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = (offsets < n) & (offsets < N)
    tl.store(X + offsets, offsets.to(tl.float32) * SCALE, mask=mask)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return torch.device("cuda:0")


@pytest.fixture
def installed():
    compile_monitor.install_compile_monitor("warn")
    yield
    compile_monitor.uninstall_compile_monitor()


def test_hooks_report_triton_compilations(device, installed, caplog):
    x = torch.empty(64, device=device)
    # A fresh SCALE keeps this test's specializations out of other tests'
    # in-memory cache.
    scale = float(torch.randint(1, 1 << 20, ()).item())
    _fill_kernel[(1,)](x, 64, 33, scale, 64)
    compile_monitor.mark_serving()
    with caplog.at_level(logging.INFO, logger=compile_monitor.__name__):
        # Runtime-argument specializations (n == 1, n % 16 != 0) are bounded
        # and never name a parameter; n == 32 reuses the startup binary.
        for n in (1, 17, 32):
            _fill_kernel[(1,)](x, n, 33, scale, 64)
        assert not _warnings(caplog)
        for count in (41, 42, 43, 44, 45, 46, 47, 49, 50):
            _fill_kernel[(1,)](x, 64, count, scale, 64)
    torch.testing.assert_close(x[:50], torch.arange(50, device=device) * scale)
    infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 11
    assert "(n: i32 divisibility=16 -> 1) from " in infos[0]
    assert "(n: i32 divisibility=16 -> i32)" in infos[1]
    assert "(N: 33 -> 50)" in infos[-1]
    # The call site is this test, not the Triton or torch frames above it.
    assert "test/test_compile_monitor.py:" in infos[-1]
    (warning,) = _warnings(caplog)
    assert "compile-time parameter N has compiled 5 new values" in warning
    assert "_fill_kernel" in warning
    stats = compile_monitor.compile_stats()
    assert (stats.startup_compiles, stats.serving_compiles) == (1, 11)
    assert stats.serving_seconds > 0


def test_error_mode_raises_from_the_launch(device):
    compile_monitor.install_compile_monitor("error")
    try:
        x = torch.empty(64, device=device)
        scale = float(torch.randint(1, 1 << 20, ()).item())
        compile_monitor.mark_serving()
        with pytest.raises(UnboundedSpecializationError, match="_fill_kernel"):
            for count in range(3, 64, 2):
                _fill_kernel[(1,)](x, 64, count, scale, 64)
    finally:
        compile_monitor.uninstall_compile_monitor()


def test_install_chains_and_uninstall_restores_hooks(device):
    calls = []

    def cache_hook(**kwargs):
        calls.append(("cache", kwargs["fn"].name))

    def post_compile_hook(**kwargs):
        calls.append(("post", kwargs["fn"].name))

    runtime = triton.knobs.runtime
    runtime.jit_cache_hook = cache_hook
    runtime.jit_post_compile_hook = post_compile_hook
    try:
        compile_monitor.install_compile_monitor("warn")
        x = torch.empty(64, device=device)
        scale = float(torch.randint(1, 1 << 20, ()).item())
        _fill_kernel[(1,)](x, 64, 33, scale, 64)
        assert calls == [("cache", "_fill_kernel"), ("post", "_fill_kernel")]
        assert compile_monitor.compile_stats().startup_compiles == 1
        compile_monitor.uninstall_compile_monitor()
        assert runtime.jit_cache_hook is cache_hook
        assert runtime.jit_post_compile_hook is post_compile_hook
        assert compile_monitor.compile_stats() is None
    finally:
        compile_monitor.uninstall_compile_monitor()
        runtime.jit_cache_hook = None
        runtime.jit_post_compile_hook = None


def test_launch_options_are_plain_values():
    # With a compile hook installed, Triton serializes the launch options to
    # JSON, so a constexpr object passed as e.g. num_warps fails every compile.
    import ast
    from pathlib import Path

    options = {"num_warps", "num_stages", "num_ctas", "waves_per_eu", "maxnreg"}
    roots = [Path(compile_monitor.__file__).parent]
    amd = importlib.util.find_spec("tokenspeed_kernel_amd")
    if amd is not None and amd.submodule_search_locations:
        roots += [Path(path) for path in amd.submodule_search_locations]
    offenders = []
    for path in (p for root in roots for p in root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        constexprs = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func).endswith("constexpr")
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                offenders += [
                    f"{path}:{node.lineno} {kw.arg}={kw.value.id}"
                    for kw in node.keywords
                    if kw.arg in options
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id in constexprs
                ]
    assert not offenders, offenders
