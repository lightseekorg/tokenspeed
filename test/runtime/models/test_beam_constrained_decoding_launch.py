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

"""Beam constrained decoding launch defaults, on CPU."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from tokenspeed.runtime.utils import server_args as server_args_module

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.models.conftest import model_config as _model_config
from test.runtime.models.conftest import server_args as _server_args

GRAMMAR_NONE_WARNING = (
    "Beam was explicitly launched with --grammar-backend none; constrained "
    "decoding requests will be rejected. This override is intended only for "
    "benchmarking."
)


def test_beam_defaults_to_xgrammar_and_declares_policy(checkpoints, caplog):
    with caplog.at_level(logging.INFO, logger=server_args_module.__name__):
        args = _server_args(checkpoints["beam"])
    assert args.grammar_backend == "xgrammar"
    assert [
        record.getMessage()
        for record in caplog.records
        if "grammar_backend=xgrammar" in record.getMessage()
    ] == ["Beam default grammar_backend=xgrammar."]

    _model_config(args)
    assert args.constrained_decoding == "always"


def test_other_models_keep_grammar_disabled(checkpoints):
    args = _server_args(checkpoints["other"])
    assert args.grammar_backend == "none"
    _model_config(args)
    assert args.constrained_decoding is None


def test_explicit_grammar_backend_wins_and_none_warns_once(checkpoints, caplog):
    explicit = _server_args(checkpoints["beam"], grammar_backend="xgrammar")
    assert explicit.grammar_backend == "xgrammar"

    with caplog.at_level(logging.WARNING, logger=server_args_module.__name__):
        disabled = _server_args(checkpoints["beam"], grammar_backend="none")
        disabled.resolve_model_defaults()
    assert disabled.grammar_backend == "none"
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage() == GRAMMAR_NONE_WARNING
    ] == [GRAMMAR_NONE_WARNING]


def test_draft_config_does_not_replace_the_policy(checkpoints):
    args = _server_args(checkpoints["beam"])
    _model_config(args)
    _model_config(args, is_draft_worker=True)
    assert args.constrained_decoding == "always"


def test_cli_default_is_unset():
    parser = argparse.ArgumentParser()
    server_args_module.ServerArgs.add_cli_args(parser)
    assert parser.parse_args([]).grammar_backend is None
    assert parser.parse_args(["--grammar-backend", "none"]).grammar_backend == "none"
