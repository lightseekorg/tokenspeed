# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
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

import asyncio
import os
import sys
import unittest
from dataclasses import dataclass

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.collector import RequestOutputCollector
from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.engine.output_processor import ReqState


@dataclass
class ModelForLogging:
    is_multimodal_gen: bool


class TokenizerForLogging:
    """Replace model tokenizer weights; expose an accidental full-input decode."""

    def __init__(self, allow_decode: bool):
        self.allow_decode = allow_decode

    def decode(self, token_ids, skip_special_tokens):
        if not self.allow_decode:
            raise AssertionError("Metadata logging must not decode the prompt")
        return "decoded input"


class LoggingLLM(AsyncLLM):
    """Use real response state without starting model, network or GPU workers."""

    def __init__(self, level: int):
        self.rid_to_state = {}
        self.log_requests = True
        self.log_requests_level = level
        self.model_config = ModelForLogging(is_multimodal_gen=False)
        self.tokenizer = TokenizerForLogging(allow_decode=level != 0)
        self.log_request_metadata = self.get_log_request_metadata()


class TestFinishLogging(unittest.IsolatedAsyncioTestCase):
    async def test_finished_response_preserves_output_and_logging_levels(self):
        for level in (0, 1, 2):
            for stream in (False, True):
                with self.subTest(level=level, stream=stream):
                    llm = LoggingLLM(level=level)
                    request = GenerateReqInput(input_ids=[1, 2, 3], stream=stream)
                    request.normalize_batch_and_arguments()
                    state = ReqState(
                        RequestOutputCollector(),
                        False,
                        asyncio.Event(),
                        request,
                        created_time=0.0,
                    )
                    expected = {
                        "text": "answer",
                        "output_ids": [4],
                        "meta_info": {"finish_reason": {"type": "length"}},
                    }
                    state.collector.put(expected, stream=stream, cumulative=False)
                    state.finished = True
                    state.event.set()
                    llm.rid_to_state[request.rid] = state
                    with self.assertLogs(
                        "tokenspeed.runtime.engine.async_llm", level="INFO"
                    ) as logs:
                        output = [
                            item async for item in llm._wait_one_response(request)
                        ]
                    self.assertEqual(output, [expected])
                    self.assertEqual(llm.rid_to_state, {})
                    self.assertTrue(any("Finish:" in line for line in logs.output))
                    if level == 0:
                        self.assertIsNone(request.text)
                        self.assertFalse(
                            any("decoded input" in line for line in logs.output)
                        )
                    else:
                        self.assertEqual(request.text, "decoded input")
                        self.assertTrue(
                            any("decoded input" in line for line in logs.output)
                        )


if __name__ == "__main__":
    unittest.main()
