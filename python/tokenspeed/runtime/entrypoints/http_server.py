# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.
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

"""
The entry point of inference server.

This file implements HTTP APIs for the inference engine via fastapi.
"""

# ruff: noqa: E402
import asyncio
import dataclasses
import json
import multiprocessing as multiprocessing
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from http import HTTPStatus
from typing import Any


def _ignore_threading_atexit(*args, **kwargs) -> None:
    return None


# Fix a bug of Python threading
setattr(threading, "_register_atexit", _ignore_threading_atexit)

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import orjson
import requests
import uvicorn
import uvloop
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    ParseFunctionCallReq,
    ProfileReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SeparateReasoningReqInput,
    SetInternalStateReq,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
    VertexGenerateReqInput,
)
from tokenspeed.runtime.entrypoints.engine import _launch_subprocesses
from tokenspeed.runtime.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ResponsesRequest,
)
from tokenspeed.runtime.entrypoints.openai.serving_chat import OpenAIServingChat
from tokenspeed.runtime.entrypoints.openai.serving_completions import (
    OpenAIServingCompletion,
)
from tokenspeed.runtime.entrypoints.openai.usage_processor import UsageProcessor
from tokenspeed.runtime.grammar.function_call_parser import FunctionCallParser
from tokenspeed.runtime.inputs.reasoning_parser import ReasoningParser
from tokenspeed.runtime.inputs.template_manager import TemplateManager
from tokenspeed.runtime.metrics.func_timer import enable_func_timer
from tokenspeed.runtime.pd.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    register_disaggregation_server,
)
from tokenspeed.runtime.utils import (
    add_api_key_middleware,
    add_prometheus_middleware,
    delete_directory,
    get_colorful_logger,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.network import set_uvicorn_logging_configs
from tokenspeed.runtime.utils.process import kill_process_tree
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.warmup import execute_warmups
from tokenspeed.version import __version__

logger = get_colorful_logger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

HEALTH_CHECK_TIMEOUT = envs.TOKENSPEED_HEALTH_CHECK_TIMEOUT.get()


# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: AsyncLLM
    template_manager: TemplateManager
    scheduler_info: dict


_global_state: _GlobalState | None = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = OpenAIServingChat(
        _global_state.tokenizer_manager, _global_state.template_manager
    )

    server_args: ServerArgs = fast_api_app.server_args

    tool_server = None
    if server_args.tool_server == "demo":
        from tokenspeed.runtime.entrypoints.openai.tool_server import DemoToolServer

        tool_server = DemoToolServer()
    elif server_args.tool_server:
        raise ValueError("Only the built-in demo tool server is supported")

    try:
        from tokenspeed.runtime.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        fast_api_app.state.openai_serving_responses = OpenAIServingResponses(
            _global_state.tokenizer_manager,
            _global_state.template_manager,
            enable_prompt_tokens_details=True,
            enable_force_include_usage=True,
            tool_server=tool_server,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.warning("Cannot initialize OpenAIServingResponses: %s", e)

    if server_args.warmups is not None:
        await execute_warmups(
            server_args.disaggregation_mode,
            server_args.warmups.split(","),
            _global_state.tokenizer_manager,
        )
        logger.info("Warmup ended")

    warmup_thread = getattr(fast_api_app, "warmup_thread", None)
    if warmup_thread is not None:
        warmup_thread.start()
    yield


# Fast API
app = FastAPI(
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enrich HTTP exception with status code and other details"""
    error = ErrorResponse(
        object="error",
        message=exc.detail,
        type=str(exc.status_code),
        code=exc.status_code,
    )
    return ORJSONResponse(content=error.model_dump(), status_code=exc.status_code)


# Custom exception handlers to change validation error status codes
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Override FastAPI's default 422 validation error with 400"""
    exc_str = str(exc)
    errors_str = str(exc.errors())

    if errors_str and errors_str != exc_str:
        message = f"{exc_str} {errors_str}"
    else:
        message = exc_str

    err = ErrorResponse(
        message=message,
        type=HTTPStatus.BAD_REQUEST.phrase,
        code=HTTPStatus.BAD_REQUEST.value,
    )

    return ORJSONResponse(
        status_code=400,
        content=err.model_dump(),
    )


async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


##### Native API endpoints #####


@app.get("/health")
@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """

    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    if _global_state.tokenizer_manager.is_server_starting():
        return Response(status_code=503)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    label = f"HEALTH_CHECK_{time.time()}"

    if _global_state.tokenizer_manager.is_image_gen:
        # Keep this branch for some internal use cases.
        raise NotImplementedError("Image generation is not supported yet.")
    elif _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            user_rid=label,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST
            gri.bootstrap_room = 0
    else:
        raise NotImplementedError("Only generation models are supported.")

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri):
            break

    task = asyncio.create_task(gen())

    # As long as we receive any response from the detokenizer/scheduler, we consider the server is healthy.
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.drop_request_state(gri.rid)
            _global_state.tokenizer_manager.mark_server_up()
            return Response(status_code=200)

    task.cancel()
    tic_time = time.strftime("%H:%M:%S", time.localtime(tic))
    last_receive_time = time.strftime(
        "%H:%M:%S", time.localtime(_global_state.tokenizer_manager.last_receive_tstamp)
    )
    logger.error(
        "Health check failed. Server couldn't get a response from detokenizer for last %s seconds. tic start time: %s. last_heartbeat time: %s",
        HEALTH_CHECK_TIMEOUT,
        tic_time,
        last_receive_time,
    )
    _global_state.tokenizer_manager.drop_request_state(gri.rid)
    _global_state.tokenizer_manager.mark_server_unhealthy()
    return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": _global_state.tokenizer_manager.model_path,
        "tokenizer_path": _global_state.tokenizer_manager.server_args.tokenizer,
        "is_generation": _global_state.tokenizer_manager.is_generation,
        "preferred_sampling_params": _global_state.tokenizer_manager.server_args.preferred_sampling_params,
    }
    return result


@app.get("/get_server_info")
async def get_server_info():
    # Returns interna states per DP.
    internal_states: list[dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        **_global_state.scheduler_info,
        "internal_states": internal_states,
        "version": __version__,
    }


@app.get("/get_load")
async def get_load():
    return await _global_state.tokenizer_manager.get_load()


# example usage:
# curl -s -X POST http://localhost:30000/set_internal_state -H "Content-Type: application/json" -d '{"server_args": {"max_micro_batch_size": 8}}'
@app.api_route("/set_internal_state", methods=["POST", "PUT"])
async def set_internal_state(obj: SetInternalStateReq, request: Request):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# fastapi implicitly converts json in the request to obj (dataclass)
@app.api_route("/generate", methods=["POST", "PUT"])
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                enable_cache_report = (
                    _global_state.tokenizer_manager.server_args.enable_cache_report
                )

                # State tracking for streaming
                prompt_tokens = {}
                completion_tokens = {}
                cached_tokens = {}
                spec_verify_tokens = {}

                async for out in _global_state.tokenizer_manager.generate_request(obj):
                    index = out.get("index", 0)
                    meta_info = out.get("meta_info", {})

                    # Track token counts
                    prompt_tokens[index] = meta_info.get("prompt_tokens", 0)
                    completion_tokens[index] = meta_info.get("completion_tokens", 0)
                    cached_tokens[index] = meta_info.get("cached_tokens", 0)
                    spec_verify_tokens[index] = meta_info.get("spec_verify_ct", 0)

                    # For streaming, check if this is the last chunk (has finish_reason)
                    finish_reason = meta_info.get("finish_reason")
                    if finish_reason is not None:
                        # This is the final chunk, add usage information
                        usage = UsageProcessor.calculate_streaming_usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cached_tokens=cached_tokens,
                            spec_verify_tokens=spec_verify_tokens,
                            n_choices=1,
                            enable_cache_report=enable_cache_report,
                        )
                        out["usage"] = usage.model_dump(
                            exclude=(
                                {"prompt_tokens_details"}
                                if not enable_cache_report
                                else None
                            )
                        )

                    yield (
                        b"data: "
                        + orjson.dumps(out, option=orjson.OPT_NON_STR_KEYS)
                        + b"\n\n"
                    )
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                logger.error("[http_server] Request failed: %s", e)
                yield (
                    b"data: "
                    + orjson.dumps(out, option=orjson.OPT_NON_STR_KEYS)
                    + b"\n\n"
                )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj
            ).__anext__()

            # Add usage information for non-streaming request
            enable_cache_report = (
                _global_state.tokenizer_manager.server_args.enable_cache_report
            )

            # Always add usage information (prompt_tokens, completion_tokens, total_tokens)
            responses = [ret] if not isinstance(ret, list) else ret
            usage = UsageProcessor.calculate_response_usage(
                responses=responses,
                n_choices=1,
                enable_cache_report=enable_cache_report,
            )

            # Add usage to response(s)
            if isinstance(ret, list):
                # For batch responses, add usage to each response
                for item in ret:
                    item["usage"] = usage.model_dump(
                        exclude=(
                            {"prompt_tokens_details"}
                            if not enable_cache_report
                            else None
                        )
                    )
            else:
                # For single response
                ret["usage"] = usage.model_dump(
                    exclude=(
                        {"prompt_tokens_details"} if not enable_cache_report else None
                    )
                )

            return ret
        except ValueError as e:
            logger.error("[http_server] Request failed: %s", e)
            return _create_error_response(e)


@app.api_route("/generate_from_file", methods=["POST"])
async def generate_from_file_request(file: UploadFile, request: Request):
    """Handle a generate request, this is purely to work with input_embeds."""
    content = await file.read()
    input_embeds = json.loads(content.decode("utf-8"))

    obj = GenerateReqInput(
        input_embeds=input_embeds,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
    )

    try:
        ret = await _global_state.tokenizer_manager.generate_request(obj).__anext__()
        return ret
    except ValueError as e:
        logger.error("File generation request failed: %s", e)
        return _create_error_response(e)


@app.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache():
    """Flush the prefix cache."""
    ret = await _global_state.tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200 if ret.success else HTTPStatus.BAD_REQUEST,
    )


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: ProfileReqInput | None = None):
    """Start profiling."""
    if obj is None:
        obj = ProfileReqInput()

    await _global_state.tokenizer_manager.start_profile(
        output_dir=obj.output_dir,
        start_step=obj.start_step,
        num_steps=obj.num_steps,
        activities=obj.activities,
        with_stack=obj.with_stack,
        record_shapes=obj.record_shapes,
        profile_by_stage=obj.profile_by_stage,
        profile_id=obj.profile_id,
    )
    return Response(
        content="Start profiling.\n",
        status_code=200,
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    """Stop profiling."""
    await _global_state.tokenizer_manager.stop_profile()
    return Response(
        content="Stop profiling. This will take some time.\n",
        status_code=200,
    )


@app.api_route("/start_expert_distribution_record", methods=["GET", "POST"])
async def start_expert_distribution_record_async():
    """Start recording the expert distribution. Clear the previous record if any."""
    await _global_state.tokenizer_manager.start_expert_distribution_record()
    return Response(
        content="Start recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/stop_expert_distribution_record", methods=["GET", "POST"])
async def stop_expert_distribution_record_async():
    """Stop recording the expert distribution."""
    await _global_state.tokenizer_manager.stop_expert_distribution_record()
    return Response(
        content="Stop recording the expert distribution.\n",
        status_code=200,
    )


@app.api_route("/dump_expert_distribution_record", methods=["GET", "POST"])
async def dump_expert_distribution_record_async():
    """Dump expert distribution record."""
    await _global_state.tokenizer_manager.dump_expert_distribution_record()
    return Response(
        content="Dump expert distribution record.\n",
        status_code=200,
    )


@app.post("/update_weights_from_disk")
async def update_weights_from_disk(obj: UpdateWeightFromDiskReqInput):
    """Update the weights from disk inplace without re-launching the server."""
    (
        success,
        message,
        num_paused_requests,
    ) = await _global_state.tokenizer_manager.update_weights_from_disk(obj)
    content = {
        "success": success,
        "message": message,
        "num_paused_requests": num_paused_requests,
    }
    if success:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK,
        )
    else:
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.BAD_REQUEST,
        )


@app.post("/init_weights_update_group")
async def init_weights_update_group(
    obj: InitWeightsUpdateGroupReqInput, request: Request
):
    """Initialize the parameter update group."""
    success, message = await _global_state.tokenizer_manager.init_weights_update_group(
        obj
    )
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.post("/update_weights_from_tensor")
async def update_weights_from_tensor(
    obj: UpdateWeightsFromTensorReqInput, request: Request
):
    """Update the weights from tensor inplace without re-launching the server.
    Notes:
    1. Ensure that the model is on the correct device (e.g., GPU) before calling this endpoint. If the model is moved to the CPU unexpectedly, it may cause performance issues or runtime errors.
    2. HTTP will transmit only the metadata of the tensor, while the tensor itself will be directly copied to the model.
    3. Any binary data in the named tensors should be base64 encoded.
    """

    success, message = await _global_state.tokenizer_manager.update_weights_from_tensor(
        obj
    )
    content = {"success": success, "message": message}
    return ORJSONResponse(
        content, status_code=200 if success else HTTPStatus.BAD_REQUEST
    )


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(
    obj: UpdateWeightsFromDistributedReqInput, request: Request
):
    """Update model parameter from distributed online."""
    (
        success,
        message,
    ) = await _global_state.tokenizer_manager.update_weights_from_distributed(obj)
    content = {"success": success, "message": message}
    if success:
        return ORJSONResponse(content, status_code=200)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)


@app.api_route("/get_weights_by_name", methods=["GET", "POST"])
async def get_weights_by_name(obj: GetWeightsByNameReqInput):
    """Get model parameter by name."""
    try:
        ret = await _global_state.tokenizer_manager.get_weights_by_name(obj)
        if ret is None:
            return _create_error_response("Get parameter by name failed")
        else:
            return ORJSONResponse(ret, status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/release_memory_occupation", methods=["GET", "POST"])
async def release_memory_occupation(
    obj: ReleaseMemoryOccupationReqInput, request: Request
):
    """Release GPU memory occupation temporarily."""
    try:
        await _global_state.tokenizer_manager.release_memory_occupation(obj)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/resume_memory_occupation", methods=["GET", "POST"])
async def resume_memory_occupation(
    obj: ResumeMemoryOccupationReqInput, request: Request
):
    """Resume GPU memory occupation."""
    try:
        await _global_state.tokenizer_manager.resume_memory_occupation(obj)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/open_session", methods=["GET", "POST"])
async def open_session(obj: OpenSessionReqInput):
    """Open a session, and return its unique session id."""
    try:
        session_id = await _global_state.tokenizer_manager.open_session(obj)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
        return session_id
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/close_session", methods=["GET", "POST"])
async def close_session(obj: CloseSessionReqInput, request: Request):
    """Close the session."""
    try:
        await _global_state.tokenizer_manager.close_session(obj)
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.api_route("/configure_logging", methods=["GET", "POST"])
async def configure_logging(obj: ConfigureLoggingReq, request: Request):
    """Configure the request logging options."""
    _global_state.tokenizer_manager.configure_logging(obj)
    return Response(status_code=200)


@app.post("/abort_request")
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)


@app.post("/parse_function_call")
async def parse_function_call_request(obj: ParseFunctionCallReq, request: Request):
    """
    A native API endpoint to parse function calls from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = FunctionCallParser(tools=obj.tools, tool_call_parser=obj.tool_call_parser)

    # 2) Call the non-stream parsing method (non-stream)
    normal_text, calls = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "normal_text": normal_text,
        "calls": [
            call.model_dump() for call in calls
        ],  # Convert pydantic objects to dictionaries
    }

    return ORJSONResponse(content=response_data, status_code=200)


@app.post("/separate_reasoning")
async def separate_reasoning_request(obj: SeparateReasoningReqInput, request: Request):
    """
    A native API endpoint to separate reasoning from a text.
    """
    # 1) Initialize the parser based on the request body
    parser = ReasoningParser(model_type=obj.reasoning_parser)

    # 2) Call the non-stream parsing method (non-stream)
    reasoning_text, normal_text = parser.parse_non_stream(obj.text)

    # 3) Organize the response content
    response_data = {
        "reasoning_text": reasoning_text,
        "text": normal_text,
    }

    return ORJSONResponse(content=response_data, status_code=200)


##### OpenAI-compatible API endpoints #####


@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


@app.get("/v1/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(
            ModelCard(
                id=served_model_name,
                root=served_model_name,
                max_model_len=_global_state.tokenizer_manager.model_config.context_len,
            )
        )
    return ModelList(data=model_cards)


@app.get("/v1/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieves a model instance, providing basic information about the model."""
    served_model_names = [_global_state.tokenizer_manager.served_model_name]

    if model not in served_model_names:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    return ModelCard(
        id=model,
        root=model,
        max_model_len=_global_state.tokenizer_manager.model_config.context_len,
    )


@app.post("/v1/responses", dependencies=[Depends(validate_json_request)])
async def v1_responses_request(request: dict, raw_request: Request):
    """Endpoint for the responses API with reasoning support."""

    request_obj = ResponsesRequest(**request)
    result = await raw_request.app.state.openai_serving_responses.create_responses(
        request_obj, raw_request
    )

    # Handle streaming responses
    if isinstance(result, AsyncGenerator):
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return result


@app.get("/v1/responses/{response_id}")
async def v1_retrieve_responses(response_id: str, raw_request: Request):
    """Retrieve a response by ID."""
    return await raw_request.app.state.openai_serving_responses.retrieve_responses(
        response_id
    )


@app.post("/v1/responses/{response_id}/cancel")
async def v1_cancel_responses(response_id: str, raw_request: Request):
    """Cancel a background response."""
    return await raw_request.app.state.openai_serving_responses.cancel_responses(
        response_id
    )


## SageMaker API
@app.get("/ping")
async def sagemaker_health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.post("/invocations")
async def sagemaker_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )


## Vertex AI API
@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate"))
async def vertex_generate(vertex_req: VertexGenerateReqInput, raw_request: Request):
    if not vertex_req.instances:
        return []
    inputs = {}
    for input_key in ("text", "input_ids", "input_embeds"):
        if vertex_req.instances[0].get(input_key):
            inputs[input_key] = [
                instance.get(input_key) for instance in vertex_req.instances
            ]
            break
    image_data = [
        instance.get("image_data")
        for instance in vertex_req.instances
        if instance.get("image_data") is not None
    ] or None
    req = GenerateReqInput(
        **inputs,
        image_data=image_data,
        **(vertex_req.parameters or {}),
    )
    ret = await generate_request(req, raw_request)
    if isinstance(ret, Response):
        return ret
    return ORJSONResponse({"predictions": ret})


def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


def api_server(
    server_args: ServerArgs,
    pipe_finish_writer: multiprocessing.connection.Connection | None = None,
    launch_callback: Callable[[], None] | None = None,
):
    """
    Launch TokenSpeed Runtime Server.

    The TokenSpeed Runtime server consists of an HTTP server and an TokenSpeed Runtime engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
        server_args=server_args
    )
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        warmup_thread.join()


def _execute_server_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: multiprocessing.connection.Connection | None,
):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"
    mapping = server_args.mapping

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error("Initialization failed during warmup: %s", last_traceback)
        kill_process_tree(os.getpid())
        return success

    # Send a warmup request
    request_name = "/generate"
    max_new_tokens = 8
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [[10, 11, 12] for _ in range(mapping.attn.dp_size)]
        if not mapping.attn.has_dp:
            json_data["input_ids"] = json_data["input_ids"][0]
    else:
        json_data["text"] = ["The capital city of France is"] * mapping.attn.dp_size
        if not mapping.attn.has_dp:
            json_data["text"] = json_data["text"][0]

    try:
        if server_args.disaggregation_mode == "null":
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res}"
            _global_state.tokenizer_manager.mark_server_up()
    except Exception:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        logger.error("Initialization failed during warmup: %s", last_traceback)
        kill_process_tree(os.getpid())
        return False

    return success


def _wait_and_warmup(
    server_args: ServerArgs,
    pipe_finish_writer: multiprocessing.connection.Connection | None,
    launch_callback: Callable[[], None] | None = None,
):
    if not server_args.skip_server_warmup:
        if not _execute_server_warmup(
            server_args,
            pipe_finish_writer,
        ):
            return
    else:
        _global_state.tokenizer_manager.mark_server_up()

    logger.info(
        "TokenSpeed server started on %s:%s and it's ready to accept requests",
        server_args.host,
        server_args.port,
    )

    if pipe_finish_writer is not None:
        pipe_finish_writer.send("ready")

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model)

    if server_args.pdlb_url is not None:
        register_disaggregation_server(
            server_args.disaggregation_mode,
            server_args.port,
            server_args.disaggregation_bootstrap_port,
            server_args.pdlb_url,
        )

    if launch_callback is not None:
        launch_callback()
