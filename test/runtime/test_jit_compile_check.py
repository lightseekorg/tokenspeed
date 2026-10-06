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

"""The serving-time JIT compile check: arming from the env and exporting."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry
from tokenspeed_kernel import compile_monitor
from tokenspeed_kernel.compile_monitor import CompileStats

from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.utils import jit_compile_check


@pytest.fixture
def uninstall(monkeypatch):
    # Marking serving closes the process-wide compile switch; reopen it afterwards.
    monkeypatch.setattr(compile_monitor, "_serving", False)
    yield
    compile_monitor.uninstall_compile_monitor()


@pytest.mark.parametrize("mode", ["warn", "error"])
def test_install_follows_the_env(monkeypatch, uninstall, mode):
    monkeypatch.setenv("TOKENSPEED_JIT_COMPILE_CHECK", mode)
    jit_compile_check.install_jit_compile_check()
    assert compile_monitor._hooks.monitor.on_unbounded == mode
    assert not compile_monitor._hooks.monitor.serving
    jit_compile_check.mark_jit_compile_serving()
    assert compile_monitor._hooks.monitor.serving
    assert compile_monitor.is_serving()


def test_off_leaves_the_monitor_uninstalled(monkeypatch, uninstall):
    monkeypatch.setenv("TOKENSPEED_JIT_COMPILE_CHECK", "off")
    jit_compile_check.install_jit_compile_check()
    assert compile_monitor.compile_stats() is None
    # Without a monitor, marking serving still closes the compile switch.
    jit_compile_check.mark_jit_compile_serving()
    assert compile_monitor.is_serving()


def test_install_rejects_unknown_modes(monkeypatch, uninstall):
    monkeypatch.setenv("TOKENSPEED_JIT_COMPILE_CHECK", "loud")
    with pytest.raises(ValueError, match="warn, error or off"):
        jit_compile_check.install_jit_compile_check()


def test_metrics_export_serving_compiles_as_deltas():
    registry = CollectorRegistry()
    metrics = EngineMetrics({"model_name": "m"}, enabled=True, registry=registry)

    def exported():
        return (
            registry.get_sample_value(
                "tokenspeed:jit_serving_compiles_total", {"model_name": "m"}
            ),
            registry.get_sample_value(
                "tokenspeed:jit_serving_compile_seconds_total", {"model_name": "m"}
            ),
        )

    metrics.record_jit_compiles(None)
    metrics.record_jit_compiles(CompileStats(40, 12.0, 0, 0.0))
    # Startup compilations are not exported; the series appears with the
    # first serving compilation.
    assert exported() == (None, None)
    metrics.record_jit_compiles(CompileStats(40, 12.0, 2, 0.25))
    metrics.record_jit_compiles(CompileStats(40, 12.0, 2, 0.25))
    assert exported() == (2.0, 0.25)
    metrics.record_jit_compiles(CompileStats(40, 12.0, 5, 1.0))
    assert exported() == (5.0, 1.0)
