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

"""In-engine HTTP generation used for request fields not carried by the gateway."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tokenspeed.runtime.engine.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

router = APIRouter()


def _llm(request: Request) -> "AsyncLLM":
    async_llm = getattr(request.app.state, "async_llm", None)
    if async_llm is None:
        raise RuntimeError("AsyncLLM is not configured on this server.")
    return async_llm


def _generate_input(data: dict[str, Any]) -> GenerateReqInput:
    return GenerateReqInput(
        text=data.get("text"),
        input_ids=data.get("input_ids"),
        input_multi_ids=data.get("input_multi_ids"),
        input_embeds=data.get("input_embeds"),
        input_extra_infos=data.get("input_extra_infos"),
        sampling_params=data.get("sampling_params"),
        user_rid=data.get("user_rid"),
        return_logprob=data.get("return_logprob"),
        logprob_start_len=data.get("logprob_start_len"),
        top_logprobs_num=data.get("top_logprobs_num"),
        token_ids_logprob=data.get("token_ids_logprob"),
        return_text_in_logprobs=bool(data.get("return_text_in_logprobs", False)),
        logprob_format=data.get("logprob_format"),
        stream=bool(data.get("stream", False)),
        log_metrics=bool(data.get("log_metrics", True)),
        session_params=data.get("session_params"),
        custom_logit_processor=data.get("custom_logit_processor"),
        return_hidden_states=bool(data.get("return_hidden_states", False)),
        bootstrap_host=data.get("bootstrap_host"),
        bootstrap_port=data.get("bootstrap_port"),
        bootstrap_room=data.get("bootstrap_room"),
    )


def _response_output_ids(output: dict[str, Any]) -> dict[str, Any]:
    """Expose generated IDs without the prompt prefix kept by engine state."""
    output_ids = output.get("output_ids")
    meta = output.get("meta_info")
    if not isinstance(output_ids, list) or not isinstance(meta, dict):
        return output
    completion_tokens = meta.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 0:
        return output
    output["output_ids"] = output_ids[-completion_tokens:] if completion_tokens else []
    return output


def _render_output(output: Any) -> Any:
    if isinstance(output, dict):
        return _response_output_ids(output)
    if isinstance(output, list):
        return [
            _response_output_ids(item) if isinstance(item, dict) else item
            for item in output
        ]
    return output


@router.post("/_engine/generate")
async def engine_generate(request: Request):
    """Run a native generation request on the engine's asyncio loop."""
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        obj = _generate_input(data)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if obj.stream:

        async def _stream():
            async for output in _llm(request).generate_request(obj):
                yield f"data: {json.dumps(_render_output(output))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    try:
        final = None
        async for output in _llm(request).generate_request(obj):
            final = output
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(_render_output(final) if final is not None else {})


def build_native_generate_app(async_llm: "AsyncLLM") -> FastAPI:
    """Return a standalone app for isolated tests."""
    app = FastAPI(title="tokenspeed native generation")
    app.state.async_llm = async_llm
    app.include_router(router)
    return app
