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

"""Reporter selection and plugin configuration without loading a model."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest

from tokenspeed.runtime.metrics.config import metrics_enabled, resolve_metrics_options
from tokenspeed.runtime.utils.server_args import ServerArgs


def options(**changes):
    values = dict(
        enable_metrics=True,
        metrics_reporters=None,
        metrics_reporter_config=None,
    )
    values.update(changes)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.set_defaults(**values)
    return parser.parse_args(["--model", "test-model"])


def test_legacy_default_and_explicit_plugin_selection():
    legacy = options()
    resolve_metrics_options(legacy)
    assert legacy.metrics_reporters == ["prometheus"]
    assert metrics_enabled(legacy, "prometheus")
    assert not metrics_enabled(legacy, "external")
    raw = options(metrics_reporters=["external"])
    resolve_metrics_options(raw)
    assert metrics_enabled(raw, "external")
    assert not metrics_enabled(raw, "prometheus")


@pytest.mark.parametrize("enabled", [False, True])
def test_unset_reporters_keep_legacy_metrics_without_normalization(enabled):
    args = options(enable_metrics=enabled)
    assert metrics_enabled(args, "prometheus") is enabled
    assert not metrics_enabled(args, "external")


def test_dual_reporter_and_disabled_metrics():
    args = options(metrics_reporters=["external", "prometheus", "external"])
    resolve_metrics_options(args)
    assert args.metrics_reporters == ["external", "prometheus"]
    assert metrics_enabled(args, "external") and metrics_enabled(args, "prometheus")
    args.enable_metrics = False
    assert not metrics_enabled(args, "external")
    assert not metrics_enabled(args, "prometheus")


@pytest.mark.parametrize(
    "reporters",
    [[], [""], ["   "], [1]],
)
def test_reject_invalid_reporter_selection(reporters):
    args = options(metrics_reporters=reporters)
    with pytest.raises(ValueError, match="reporter"):
        resolve_metrics_options(args)


def test_external_name_is_resolved_after_argument_parsing():
    args = options(metrics_reporters=["external", "external"])
    resolve_metrics_options(args)
    assert args.metrics_reporters == ["external"]


@pytest.fixture
def plugin_registry(monkeypatch):
    from tokenspeed.runtime.metrics import registry

    monkeypatch.setattr(registry, "_REPORTERS", {})
    discoveries = []
    monkeypatch.setattr(
        registry, "ensure_metric_reporters_loaded", lambda: discoveries.append(True)
    )
    return registry, discoveries


def test_external_configuration_is_validated_and_copied(tmp_path, plugin_registry):
    from tokenspeed.runtime.metrics import config

    registry, discoveries = plugin_registry
    validated = []

    def validate(raw):
        normalized = {"queue_size": int(raw["queue_size"]), "labels": raw["labels"]}
        validated.append(normalized)
        return normalized

    registry.register_metric_reporter(
        "external",
        validate=validate,
        create_sink=lambda cfg: None,
        start_node=None,
        override=False,
    )
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"external": {"queue_size": "10", "labels": ["a"]}}))
    args = options(
        metrics_reporters=["prometheus", "external"], metrics_reporter_config=str(path)
    )
    result = config.resolve_reporter_configs(args)
    assert result == {"external": {"queue_size": 10, "labels": ["a"]}}
    validated[0]["labels"].append("b")
    assert result["external"]["labels"] == ["a"]
    assert discoveries == [True]


@pytest.mark.parametrize(
    "enabled, names", [(False, ["external"]), (True, ["prometheus"])]
)
def test_no_raw_outputs_do_not_discover_plugins(plugin_registry, enabled, names):
    from tokenspeed.runtime.metrics import config

    _, discoveries = plugin_registry
    assert (
        config.resolve_reporter_configs(
            options(enable_metrics=enabled, metrics_reporters=names)
        )
        == {}
    )
    assert discoveries == []


def test_unknown_selection_fails_at_runtime_preparation(plugin_registry):
    from tokenspeed.runtime.metrics import config

    with pytest.raises(ValueError, match="Unknown metric reporter"):
        config.resolve_reporter_configs(options(metrics_reporters=["missing"]))


@pytest.mark.parametrize(
    "content, message",
    [
        ("[]", "object"),
        ('{"prometheus": {}}', "CLI options"),
        ('{"unused": {}}', "not selected"),
        ('{"external": {}, "external": {}}', "Duplicate"),
        ('{"external": []}', "object"),
        ('{"external": {"x": NaN}}', "finite"),
    ],
)
def test_bad_plugin_file_is_rejected_before_creation(
    tmp_path, plugin_registry, content, message
):
    from tokenspeed.runtime.metrics import config

    path = tmp_path / "metrics.json"
    path.write_text(content)
    args = options(metrics_reporters=["external"], metrics_reporter_config=str(path))
    with pytest.raises(ValueError, match=message):
        config.resolve_reporter_configs(args)


def test_raw_output_selection_includes_external_plugins():
    from tokenspeed.runtime.metrics import config

    assert config.raw_metrics_enabled(options(metrics_reporters=["external"]))
    assert not config.raw_metrics_enabled(options(metrics_reporters=["prometheus"]))
    assert not config.raw_metrics_enabled(
        options(enable_metrics=False, metrics_reporters=["external"])
    )
