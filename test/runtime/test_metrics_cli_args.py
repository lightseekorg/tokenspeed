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

"""Plugin selection and configuration through the engine argument parser."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest

from tokenspeed.runtime.metrics.config import resolve_metrics_options
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.mark.parametrize(
    "flags, expected",
    [
        ([], ["prometheus"]),
        (["--metrics-reporters", "external"], ["external"]),
        (
            ["--metrics-reporters=external", "--metrics-reporters=prometheus"],
            ["external", "prometheus"],
        ),
        (
            ["--metrics-reporters=first", "--metrics-reporters=second"],
            ["first", "second"],
        ),
    ],
)
def test_reporter_flags_do_not_append_prometheus_implicitly(flags, expected):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "test-model",
            "--enable-metrics",
            *flags,
        ]
    )
    resolve_metrics_options(args)
    assert args.metrics_reporters == expected


def test_plugin_config_path_is_preserved_without_loading_file():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "test-model",
            "--metrics-reporters",
            "external",
            "--metrics-reporter-config",
            "/not-opened-during-parsing.json",
        ]
    )
    resolve_metrics_options(args)
    assert args.metrics_reporter_config == "/not-opened-during-parsing.json"
