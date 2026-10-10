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

"""A grammar that does not compile is the request's error: a 400.

A request whose json_schema, regex, ebnf or structural_tag fails to compile
is aborted with HTTP 400, whether the failure is cached or arrives with the
compile future, and so is a key whose compile kept timing out until it was
given up. A timeout that a later request may still compile keeps no status,
as before.

Pure CPU; no model, tokenizer, or GPU required.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    RequestState,
)
from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from tokenspeed.runtime.grammar.grammar_manager import GrammarManager  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import (  # noqa: E402
    SamplingParams,
)

SCHEMA = '{"type": "object"}'
KEY = ("json", SCHEMA)


class _Grammar(BaseGrammarObject):
    def copy(self):
        return self


class _Backend(BaseGrammarBackend):
    """Compiles to ``result``: a grammar, an invalid one, or an exception to
    raise; ``gate`` holds the compile until it is set."""

    def __init__(self, result, gate: threading.Event | None = None):
        super().__init__()
        self.result = result
        self.gate = gate

    def init_value_impl(self, key, require_reasoning: bool = False):
        if self.gate is not None:
            self.gate.wait()
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _manager(backend: _Backend, timeout_secs: float = 5.0) -> GrammarManager:
    args = SimpleNamespace(
        grammar_backend="none",
        grammar_compile_timeout_secs=timeout_secs,
        grammar_compile_max_retries=1,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=[0])),
    )
    manager = GrammarManager(args, tokenizer=None, vocab_size=32)
    manager.grammar_backend = backend
    return manager


def _state() -> RequestState:
    return RequestState(
        prompt_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=8, json_schema=SCHEMA),
        stream=False,
        tokenizer=None,
        computes_prompt_logprobs=False,
    )


def _admit(manager: GrammarManager, deadline: float = 5.0) -> RequestState:
    """Admits one request, through the queue when its compile is in flight."""
    state = _state()
    if not manager.process_req_with_grammar(state):
        manager.add_to_queue(SimpleNamespace(request_id="r"), state, None)
        end = time.monotonic() + deadline
        while not manager.get_ready_grammar_requests():
            if time.monotonic() > end:
                raise AssertionError("the queued request was never promoted")
            time.sleep(0.01)
    return state


class TestCompileFailureStatus(unittest.TestCase):
    def _assert_abort(self, state: RequestState, status: int | None, text: str):
        reason = state.finished_reason.to_json()
        self.assertEqual(reason["type"], "abort")
        self.assertEqual(reason.get("status_code"), status)
        self.assertIn(text, reason["message"])

    def test_compile_failure_from_the_future_is_bad_request(self):
        manager = _manager(_Backend(InvalidGrammarObject("bad schema")))
        self._assert_abort(_admit(manager), 400, "bad schema")

    def test_leaked_compile_exception_is_bad_request(self):
        manager = _manager(_Backend(KeyError("format")))
        self._assert_abort(_admit(manager), 400, "KeyError")

    def test_cached_compile_failure_is_bad_request(self):
        manager = _manager(_Backend(InvalidGrammarObject("bad schema")))
        _admit(manager)
        state = _state()
        self.assertTrue(manager.process_req_with_grammar(state))
        self._assert_abort(state, 400, "bad schema")

    def test_cached_transient_timeout_keeps_no_status(self):
        backend = _Backend(_Grammar())
        backend.record_compile_timeout(KEY, "timed out", ttl_secs=60, max_retries=1)
        state = _state()
        self.assertTrue(_manager(backend).process_req_with_grammar(state))
        self._assert_abort(state, None, "timed out")

    def test_given_up_timeout_is_bad_request(self):
        backend = _Backend(_Grammar())
        for _ in range(2):
            backend.record_compile_timeout(KEY, "timed out", ttl_secs=60, max_retries=1)
        state = _state()
        self.assertTrue(_manager(backend).process_req_with_grammar(state))
        self._assert_abort(state, 400, "gave up after")

    def test_compile_timeout_keeps_no_status(self):
        gate = threading.Event()
        manager = _manager(_Backend(_Grammar(), gate), timeout_secs=0.05)
        try:
            self._assert_abort(_admit(manager), None, "timed out")
        finally:
            gate.set()

    def test_valid_grammar_is_admitted(self):
        state = _admit(_manager(_Backend(_Grammar())))
        self.assertFalse(state.finished)
        self.assertIsInstance(state.grammar, _Grammar)


if __name__ == "__main__":
    unittest.main()
