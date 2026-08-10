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

"""Regression tests for GrammarManager construction under skip_tokenizer_init.

The headless msgpack launch (``ts serve --headless``, driven by SMG over ZMQ)
forces ``skip_tokenizer_init=True`` because the frontend passes token ids.
GrammarManager used to treat that flag as "no tokenizer → no grammar backend"
and silently discarded ``--grammar-backend xgrammar``, aborting every
constrained request (json_schema / regex / ebnf / structural_tag) at admission
with zero tokens. The premise was false: RequestHandler loads the
scheduler-side tokenizer unconditionally and passes it in, so the backend can
always be built. These tests pin that contract:

1. ``skip_tokenizer_init=True`` does NOT suppress the grammar backend; the
   configured backend is built with the tokenizer GrammarManager receives.
2. ``--grammar-backend none`` remains the one way to get no backend, so the
   admission-time abort message naming it stays truthful.

Pure CPU; no model, tokenizer, or GPU required.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.grammar import (  # noqa: E402
    grammar_manager as grammar_manager_mod,
)
from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    create_grammar_backend,
)
from tokenspeed.runtime.grammar.grammar_manager import GrammarManager  # noqa: E402


def _server_args(**overrides) -> SimpleNamespace:
    """The minimal attribute surface GrammarManager.__init__ reads."""
    args = SimpleNamespace(
        grammar_backend="xgrammar",
        grammar_compile_timeout_secs=1.0,
        grammar_compile_max_retries=1,
        skip_tokenizer_init=False,
        disable_any_whitespace=False,
        # Single-rank attn TP group: the gloo sync-group branch short-circuits.
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=[0])),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestGrammarManagerHeadless(unittest.TestCase):
    def test_skip_tokenizer_init_does_not_suppress_backend(self):
        """The headless flag must not discard a configured grammar backend."""
        args = _server_args(skip_tokenizer_init=True)
        tokenizer = object()
        backend = object()

        with mock.patch.object(
            grammar_manager_mod, "create_grammar_backend", return_value=backend
        ) as factory:
            manager = GrammarManager(args, tokenizer, vocab_size=32)

        factory.assert_called_once_with(args, tokenizer, 32)
        self.assertIs(manager.grammar_backend, backend)

    def test_backend_none_only_from_explicit_none(self):
        """--grammar-backend none is the sole path to a missing backend."""
        args = _server_args(grammar_backend="none", skip_tokenizer_init=True)
        self.assertIsNone(create_grammar_backend(args, tokenizer=None, vocab_size=32))

        manager = GrammarManager(args, tokenizer=None, vocab_size=32)
        self.assertIsNone(manager.grammar_backend)


if __name__ == "__main__":
    unittest.main()
