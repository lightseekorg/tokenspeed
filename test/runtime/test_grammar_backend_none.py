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

"""A constrained request on a server without a grammar backend is a 400.

With ``--grammar-backend none`` a request carrying json_schema, regex, ebnf
or structural_tag is rejected at admission. The rejection must reach the
frontend as a client error: AsyncLLM raises ``EngineGenerateError`` (a
``ValueError``), which the gRPC servicer reports as INVALID_ARGUMENT and the
gateway as HTTP 400. Reported as an internal error, clients retry it.

Pure CPU; no model, tokenizer, or GPU required.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import asyncio  # noqa: E402

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.exceptions import EngineGenerateError  # noqa: E402
from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    RequestState,
)
from tokenspeed.runtime.engine.output_processor import ReqState  # noqa: E402
from tokenspeed.runtime.engine.request_types import FINISH_ABORT  # noqa: E402
from tokenspeed.runtime.grammar.grammar_manager import GrammarManager  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import (  # noqa: E402
    SamplingParams,
)

CONSTRAINTS = {
    "json_schema": '{"type": "object"}',
    "regex": "[a-z]+",
    "ebnf": 'root ::= "a"',
    "structural_tag": '{"type": "structural_tag", "format": {"type": "any_text"}}',
}


def _manager_without_backend() -> GrammarManager:
    args = SimpleNamespace(
        grammar_backend="none",
        grammar_compile_timeout_secs=1.0,
        grammar_compile_max_retries=1,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=[0])),
    )
    return GrammarManager(args, tokenizer=None, vocab_size=32)


def _state(**constraint) -> RequestState:
    return RequestState(
        prompt_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=8, **constraint),
        stream=False,
        tokenizer=None,
    )


class TestAdmissionWithoutBackend(unittest.TestCase):
    def test_constraint_is_rejected_as_bad_request(self):
        manager = _manager_without_backend()
        for field, value in CONSTRAINTS.items():
            with self.subTest(field=field):
                state = _state(**{field: value})
                self.assertTrue(manager.process_req_with_grammar(state))
                reason = state.finished_reason.to_json()
                self.assertEqual(reason["type"], "abort")
                self.assertEqual(reason["status_code"], 400)
                self.assertIn("--grammar-backend none", reason["message"])

    def test_unconstrained_request_is_admitted(self):
        state = _state()
        self.assertTrue(_manager_without_backend().process_req_with_grammar(state))
        self.assertFalse(state.finished)


class _StubAsyncLLM(AsyncLLM):
    """Only the attributes ``_wait_one_response`` reads on a finish."""

    def __init__(self) -> None:
        self.rid_to_state: dict[str, ReqState] = {}
        self.log_requests = False
        self.enable_metrics = False
        self.tokenizer = None
        self.model_config = SimpleNamespace(is_multimodal_gen=False)
        self.engine_core_client = SimpleNamespace(
            send_to_scheduler=SimpleNamespace(send_pyobj=lambda obj: None)
        )


class TestFrontendSeesClientError(unittest.IsolatedAsyncioTestCase):
    async def _finish(self, reason: dict, *, stream: bool) -> list:
        llm = _StubAsyncLLM()
        obj = SimpleNamespace(rid="r1", stream=stream, input_ids=None, text=None)
        state = ReqState(
            RequestOutputCollector(), False, asyncio.Event(), obj, created_time=0.0
        )
        llm.rid_to_state[obj.rid] = state
        state.finished = True
        state.collector.put(
            {"text": "", "output_ids": [], "meta_info": {"finish_reason": reason}},
            stream=stream,
        )
        state.event.set()
        return [out async for out in llm._wait_one_response(obj)]

    async def test_bad_request_abort_raises_value_error(self):
        state = _state(json_schema=CONSTRAINTS["json_schema"])
        _manager_without_backend().process_req_with_grammar(state)
        reason = state.finished_reason.to_json()
        for stream in (False, True):
            with self.subTest(stream=stream):
                with self.assertRaises(EngineGenerateError) as ctx:
                    await self._finish(reason, stream=stream)
                self.assertIsInstance(ctx.exception, ValueError)
                self.assertIn("--grammar-backend none", str(ctx.exception))

    async def test_abort_without_status_is_still_a_finish(self):
        reason = FINISH_ABORT("engine failure").to_json()
        self.assertNotIn("status_code", reason)
        outputs = await self._finish(reason, stream=False)
        self.assertEqual(outputs[0]["meta_info"]["finish_reason"], reason)


if __name__ == "__main__":
    unittest.main()
