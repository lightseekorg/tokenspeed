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

"""Metric plugin registration, discovery and process-safe configuration."""

import importlib
import os
import sys
from importlib.metadata import EntryPoint
from types import ModuleType

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _unexpected_creation(config):
    raise AssertionError("Registration must not create resources")


def _register(registry, name, *, override, validate):
    registry.register_metric_reporter(
        name,
        validate=validate,
        create_sink=_unexpected_creation,
        start_node=None,
        override=override,
    )


@pytest.fixture
def registry(monkeypatch):
    module = importlib.import_module("tokenspeed.runtime.metrics.registry")
    monkeypatch.setattr(module, "_REPORTERS", {})
    return module


def test_registration_stores_callbacks_without_creating_resources(monkeypatch):
    registry = importlib.import_module("tokenspeed.runtime.metrics.registry")
    monkeypatch.setattr(registry, "_REPORTERS", {})
    registry.register_metric_reporter(
        "fixture",
        validate=_unexpected_creation,
        create_sink=_unexpected_creation,
        start_node=_unexpected_creation,
        override=False,
    )
    entry = registry.get_metric_reporter("fixture")
    for callback in (entry.validate, entry.create_sink, entry.start_node):
        with pytest.raises(AssertionError, match="Registration must not create"):
            callback({})


def test_failed_plugin_removes_registration_and_restores_override(registry):
    from tokenspeed.runtime.plugins.registry import recording

    original_validate = lambda config: config
    _register(registry, "existing", override=False, validate=original_validate)
    original = registry.get_metric_reporter("existing")
    with pytest.raises(RuntimeError, match="plugin failed"):
        with recording() as record:
            _register(registry, "existing", override=True, validate=dict)
            _register(registry, "new", override=False, validate=dict)
            assert record.names == {"metric reporter": ["existing", "new"]}
            raise RuntimeError("plugin failed")
    assert registry.get_metric_reporter("existing") is original
    with pytest.raises(ValueError, match="Unknown metric reporter"):
        registry.get_metric_reporter("new")


def test_duplicate_requires_override_and_prometheus_is_reserved(registry):
    _register(registry, "fixture", override=False, validate=dict)
    original = registry.get_metric_reporter("fixture")
    with pytest.raises(ValueError, match="override=True"):
        _register(registry, "fixture", override=False, validate=_unexpected_creation)
    assert registry.get_metric_reporter("fixture") is original
    _register(registry, "fixture", override=True, validate=_unexpected_creation)
    assert registry.get_metric_reporter("fixture").validate is _unexpected_creation
    for override in (False, True):
        with pytest.raises(ValueError, match="reserved"):
            _register(registry, "prometheus", override=override, validate=dict)


def test_config_copy_is_independent_and_accepts_shared_noncyclic_values(registry):
    child = {"labels": ["model", None, True, 3, 0.25]}
    original = {"one": child, "two": child}
    copied = registry.copy_metric_config(original)
    assert copied == original
    copied["one"]["labels"].append("extra")
    assert original["one"]["labels"] == ["model", None, True, 3, 0.25]
    assert copied["two"]["labels"] == original["two"]["labels"]


def test_config_rejects_cycles(registry):
    config = {"children": []}
    config["children"].append(config)
    with pytest.raises(ValueError, match="cyclic"):
        registry.copy_metric_config(config)


def _install_plugin(monkeypatch, register):
    import tokenspeed_kernel.plugins as kernel_plugins

    from tokenspeed.runtime import plugins

    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "_loaded_plugins", {})
    monkeypatch.delenv(plugins.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(kernel_plugins, "discover_plugins", lambda: None)
    module = ModuleType("fixture")
    module.register = register
    monkeypatch.setitem(sys.modules, "fixture", module)
    ep = EntryPoint(
        name="fixture", value="fixture:register", group="tokenspeed.plugins"
    )
    monkeypatch.setattr(plugins.importlib_metadata, "entry_points", lambda **_: [ep])
    return plugins


def test_discovery_loads_entry_points_once(registry, monkeypatch):
    _register(registry, "existing", override=False, validate=dict)
    external_calls = []

    def register_plugin():
        existing = registry.get_metric_reporter("existing")
        external_calls.append(existing.validate({"endpoint": "test"}))
        _register(registry, "fixture", override=False, validate=dict)

    plugins = _install_plugin(monkeypatch, register_plugin)
    registry.ensure_metric_reporters_loaded()
    registry.ensure_metric_reporters_loaded()
    assert external_calls == [{"endpoint": "test"}]
    assert registry.get_metric_reporter("fixture").validate({"queue_size": 8}) == {
        "queue_size": 8
    }
    assert plugins.list_plugins()[0].registrations == (
        ("metric reporter", ("fixture",)),
    )


def test_discovery_preserves_existing_registration(registry, monkeypatch):
    def validate(config):
        return {**config, "queue_size": 8}

    _register(registry, "existing", override=False, validate=validate)
    _install_plugin(
        monkeypatch,
        lambda: _register(registry, "fixture", override=False, validate=dict),
    )
    registry.ensure_metric_reporters_loaded()
    assert registry.get_metric_reporter("existing").validate({"endpoint": "test"}) == {
        "endpoint": "test",
        "queue_size": 8,
    }


@pytest.mark.parametrize(
    "override, fail_after_registration", [(False, False), (True, False), (True, True)]
)
def test_generic_loader_first_enforces_override_and_rollback(
    registry, monkeypatch, override, fail_after_registration
):
    def original_validate(config):
        return {**config, "queue_size": 8}

    def replacement_validate(config):
        return {**config, "queue_size": 16}

    _register(registry, "existing", override=False, validate=original_validate)

    def register_plugin():
        _register(
            registry, "existing", override=override, validate=replacement_validate
        )
        if fail_after_registration:
            raise RuntimeError("plugin failed after overriding existing reporter")

    plugins = _install_plugin(monkeypatch, register_plugin)
    successful_override = override and not fail_after_registration
    if successful_override:
        plugins.ensure_loaded()
    else:
        with pytest.warns(UserWarning, match="Failed to load plugin"):
            plugins.ensure_loaded()

    # The generic loader must restore the previous reporter on failure,
    # before metrics discovery is called.
    expected = {"endpoint": "test", "queue_size": 16 if successful_override else 8}
    assert (
        registry.get_metric_reporter("existing").validate({"endpoint": "test"})
        == expected
    )
    registry.ensure_metric_reporters_loaded()
    assert (
        registry.get_metric_reporter("existing").validate({"endpoint": "test"})
        == expected
    )
    if successful_override:
        assert plugins.list_plugins()[0].registrations == (
            ("metric reporter", ("existing",)),
        )
    else:
        assert plugins.list_plugins() == []
