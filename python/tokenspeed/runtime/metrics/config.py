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

"""Selection and validation shared by metrics producers and launchers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tokenspeed.runtime.metrics.events import MetricConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs


def metrics_enabled(args: ServerArgs, reporter: str) -> bool:
    """Whether an output is enabled; an unset selection keeps Prometheus."""
    if not args.enable_metrics:
        return False
    if args.metrics_reporters is None:
        return reporter == "prometheus"
    return reporter in args.metrics_reporters


def raw_metrics_enabled(args: ServerArgs) -> bool:
    """Whether raw metric outputs are selected and enabled."""
    return args.enable_metrics and any(
        name != "prometheus" for name in (args.metrics_reporters or [])
    )


def resolve_metrics_options(args: ServerArgs) -> None:
    """Normalize CLI options without discovering or constructing plugins."""
    reporters = args.metrics_reporters
    if reporters is None:
        reporters = ["prometheus"]
    if not reporters or any(
        not isinstance(name, str) or not name.strip() for name in reporters
    ):
        raise ValueError("metrics reporters must contain nonempty names")
    args.metrics_reporters = list(dict.fromkeys(reporters))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in metrics reporter configuration")
        result[key] = value
    return result


def resolve_reporter_configs(args: ServerArgs) -> dict[str, MetricConfig]:
    """Discover selected event outputs and validate configuration before spawn.

    Call after global environment setup, especially the Prometheus multiprocess
    directory. Configuration values stay out of server arguments and logs.
    """
    if not args.enable_metrics:
        return {}

    from tokenspeed.runtime.metrics.registry import (
        copy_metric_config,
        ensure_metric_reporters_loaded,
        get_metric_reporter,
    )

    selected = args.metrics_reporters or ["prometheus"]
    external: dict[str, MetricConfig] = {}
    if args.metrics_reporter_config is not None:
        with open(args.metrics_reporter_config, encoding="utf-8") as source:
            data = copy_metric_config(
                json.load(source, object_pairs_hook=_unique_object)
            )
        for name, value in data.items():
            if name == "prometheus":
                raise ValueError(
                    "Use existing CLI options to configure Prometheus metrics"
                )
            if name not in selected:
                raise ValueError(
                    f"Metric reporter configuration {name!r} is not selected"
                )
            external[name] = copy_metric_config(value)

    if not raw_metrics_enabled(args):
        return {}
    ensure_metric_reporters_loaded()
    resolved = {}
    for name in selected:
        if name == "prometheus":
            continue
        config = external.get(name, {})
        reporter = get_metric_reporter(name)
        resolved[name] = copy_metric_config(reporter.validate(config))
    return resolved
