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

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.engine.protocol import EngineClient
from tokenspeed.runtime.entrypoints.openai.protocol import (
    ErrorResponse,
    OpenAIServingRequest,
)
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)

_request_conversion_executor: ThreadPoolExecutor | None = None
_request_conversion_executor_lock = Lock()


def _get_request_conversion_executor() -> ThreadPoolExecutor:
    global _request_conversion_executor

    with _request_conversion_executor_lock:
        if _request_conversion_executor is None:
            _request_conversion_executor = ThreadPoolExecutor(
                max_workers=envs.TOKENSPEED_REQUEST_CONVERSION_WORKERS.get(),
                thread_name_prefix="openai-request-conversion",
            )
        return _request_conversion_executor


# Default poll interval for the disconnect watchdog. 1s is small enough
# that a cancelled request frees its scheduler slot in roughly one tick,
# large enough not to chew CPU in the steady state.
_DISCONNECT_POLL_INTERVAL_S = 1.0


async def await_with_disconnect_watchdog(
    awaitable: Any,
    raw_request: Request | None,
    poll_interval: float = _DISCONNECT_POLL_INTERVAL_S,
) -> Any:
    """Race ``awaitable`` against the request's ``http.disconnect`` signal.

    Starlette/Uvicorn does **not** auto-cancel non-streaming handlers when
    the client goes away — the disconnect message arrives via ``receive()``
    which a non-streaming handler doesn't drive while awaiting inference.
    Without this watchdog, a cancelled request keeps generating up to its
    ``max_tokens`` and latches a ``--max-num-seqs`` slot until then;
    fixing that requires explicit disconnect detection.

    On disconnect we cancel the task; the inflight ``CancelledError``
    propagates into ``async_llm.generate_request``'s ``finally:`` block,
    which calls ``abort_request(rid)`` — i.e. the existing cancellation
    contract works again, the way the docstring says it should.

    ``raw_request=None`` is honored as "no watchdog" (used when the caller
    doesn't have a FastAPI ``Request`` handle, e.g. in-process invocation).
    """
    if raw_request is None:
        return await awaitable

    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_interval)
            if task in done:
                return task.result()
            if await raw_request.is_disconnected():
                task.cancel()
                # Drain the cancellation so the ``finally`` block in
                # ``async_llm.generate_request`` runs and sends AbortReq
                # to the scheduler before we propagate.
                #
                # Catch ``CancelledError`` (the expected outcome) and
                # ``Exception`` (engine raised mid-cancel, e.g. ``ValueError``)
                # explicitly — leave ``KeyboardInterrupt`` /
                # ``SystemExit`` to propagate.
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                raise asyncio.CancelledError("client disconnected")
    finally:
        if not task.done():
            task.cancel()


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, engine_client: EngineClient):
        self.engine_client = engine_client
        self.request_conversion_executor = _get_request_conversion_executor()

    async def handle_request_engine(
        self, request: OpenAIServingRequest
    ) -> AsyncIterator[str]:
        """Handle request and return serialized result"""
        response = await self.handle_request(request, None)

        if isinstance(response, StreamingResponse):
            """Handle streaming response output"""
            try:
                if hasattr(response, "body_iterator"):
                    async for chunk in response.body_iterator:
                        yield (
                            chunk.decode("utf-8")
                            if isinstance(chunk, bytes)
                            else str(chunk)
                        )
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        else:
            """Handle non-streaming response output"""
            try:
                if hasattr(response, "model_dump_json"):
                    yield response.model_dump_json()
                elif isinstance(response, ORJSONResponse):
                    yield response.body.decode("utf-8")
                else:
                    yield "no type"
            except Exception as e:
                logger.exception("Serialization error: %s", e)
                yield json.dumps(
                    {
                        "error": {
                            "message": f"Serialization failed: {str(e)}",
                            "type": "InternalServerError",
                        }
                    },
                    ensure_ascii=False,
                )

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Any | StreamingResponse | ErrorResponse:
        """Handle the specific request type with common pattern"""
        try:
            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Convert to internal format
            (
                adapted_request,
                processed_request,
            ) = await self._convert_to_internal_request_async(request)

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
        except HTTPException as e:
            return self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
        except asyncio.CancelledError:
            # Client disconnected mid-request (raised by await_with_disconnect_watchdog).
            # asyncio.CancelledError is a BaseException subclass, not Exception, so it
            # would otherwise propagate to uvicorn and be logged as "Exception in ASGI
            # application". Return 499 (Client Closed Request) to let uvicorn close the
            # connection quietly. The scheduler slot has already been freed by the
            # abort_request() call in async_llm.generate_request's finally block.
            logger.debug("Request cancelled: client disconnected")
            return self.create_error_response(
                message="Request cancelled: client disconnected",
                err_type="ClientDisconnect",
                status_code=499,
            )
        except Exception as e:
            logger.exception("Error in request: %s", e)
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        """Convert OpenAI request to internal format"""
        pass

    async def _convert_to_internal_request_async(
        self,
        request: OpenAIServingRequest,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.request_conversion_executor,
            self._convert_to_internal_request,
            request,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> StreamingResponse | ErrorResponse | ORJSONResponse:
        """Handle streaming request

        Override this method in child classes that support streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Any | ErrorResponse | ORJSONResponse:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, _: OpenAIServingRequest) -> str | None:
        """Validate request"""
        pass

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: str | None = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps({"error": error.model_dump()})
