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

"""How ``AsyncLLM`` hands a request's finish to its caller: an abort the
engine refused as a client error (``status_code`` 400) raises, any other
finish is returned."""

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.exceptions import EngineGenerateError  # noqa: E402
from tokenspeed.runtime.engine.output_processor import ReqState  # noqa: E402
from tokenspeed.runtime.engine.request_types import FINISH_ABORT  # noqa: E402


class _StubAsyncLLM(AsyncLLM):
    """Only the attributes ``_wait_one_response`` reads on a finish."""

    def __init__(self) -> None:
        self.rid_to_state: dict[str, ReqState] = {}
        self.log_requests = False
        self.enable_metrics = False
        self.tokenizer = None
        self.model_config = SimpleNamespace(is_multimodal_gen=False)
        self._abort_tasks = set()


async def _finish(reason: dict, *, stream: bool) -> list:
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
        cumulative=False,
    )
    state.event.set()
    return [out async for out in llm._wait_one_response(obj)]


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("stream", [False, True])
async def test_a_client_error_abort_raises(stream: bool) -> None:
    # The engine refuses a constraint it has no grammar backend for.
    reason = {
        "type": "abort",
        "message": "Invalid request: json_schema is not supported when the "
        "server is launched with --grammar-backend none",
        "status_code": 400,
    }
    with pytest.raises(EngineGenerateError) as ctx:
        await _finish(reason, stream=stream)
    assert isinstance(ctx.value, ValueError)
    assert "--grammar-backend none" in str(ctx.value)


@pytest.mark.asyncio(loop_scope="function")
async def test_an_abort_without_a_status_is_still_a_finish() -> None:
    reason = FINISH_ABORT("engine failure").to_json()
    assert "status_code" not in reason
    outputs = await _finish(reason, stream=False)
    assert outputs[0]["meta_info"]["finish_reason"] == reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
