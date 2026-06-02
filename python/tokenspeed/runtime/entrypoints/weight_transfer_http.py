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

"""vLLM-compatible weight-transfer HTTP control plane.

Byte-compatible with vLLM's ``vllm/entrypoints/serve/rlhf/api_router.py`` so the
same RL trainer code (verl / slime / AReaL / miles) can drive tokenspeed
unchanged -- same paths, methods, request/response JSON, and call lifecycle.

The handlers are thin: they parse the vLLM-shaped request, call into a
``WeightTransferManager`` (``runtime/engine/weight_transfer/manager.py``), and
return the exact vLLM status payloads. Heavy weight payloads travel out-of-band
(NCCL broadcast / CUDA-IPC); only metadata flows through here.

Deployment note: this app must run on the same asyncio event loop as the
``AsyncLLM`` it controls -- the manager toggles a loop-bound admission event and
awaits loop-bound scheduler communicators. Use :func:`run_weight_transfer_server`
from the engine process (see ``runtime/entrypoints/engine.py``), or mount
:data:`router` onto an app whose ``state.weight_transfer_manager`` is set.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from tokenspeed.runtime.engine.weight_transfer.manager import (
    PAUSE_MODES,
    WeightTransferStateError,
)
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.weight_transfer.manager import WeightTransferManager

logger = get_colorful_logger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _manager(request: Request) -> "WeightTransferManager":
    manager = getattr(request.app.state, "weight_transfer_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="Weight transfer manager is not configured on this server.",
        )
    return manager


async def _read_json(request: Request) -> dict[str, Any]:
    """Parse a JSON object body, 400 on invalid JSON (vLLM parity)."""
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object"
        )
    return body


async def _run(action: Awaitable[Any], success_content: dict[str, Any]) -> JSONResponse:
    """Await a manager action and map exceptions to vLLM-compatible status codes.

    - ``WeightTransferStateError`` (lifecycle misuse) -> 409 Conflict
    - ``NotImplementedError`` (backend path not wired) -> 501 Not Implemented
    - ``ValueError`` (bad request payload) -> 400 Bad Request
    - anything else -> 500 Internal Server Error
    """
    try:
        await action
    except WeightTransferStateError as e:
        return JSONResponse({"error": str(e)}, status_code=HTTPStatus.CONFLICT.value)
    except NotImplementedError as e:
        return JSONResponse(
            {"error": str(e)}, status_code=HTTPStatus.NOT_IMPLEMENTED.value
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=HTTPStatus.BAD_REQUEST.value)
    except Exception as e:  # noqa: BLE001 - defensive, mirrors vLLM
        logger.exception("Weight transfer action failed")
        return JSONResponse(
            {"error": f"Weight transfer action failed: {e}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    return JSONResponse(content=success_content)


# --------------------------------------------------------------------------- #
# Weight-update lifecycle
# --------------------------------------------------------------------------- #


@router.post("/init_weight_transfer_engine")
async def init_weight_transfer_engine(raw_request: Request) -> JSONResponse:
    body = await _read_json(raw_request)
    init_info = body.get("init_info")
    if init_info is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'init_info' in request body",
        )
    return await _run(
        _manager(raw_request).init_engine(init_info),
        {"message": "Weight transfer initialized"},
    )


@router.post("/start_weight_update")
async def start_weight_update(raw_request: Request) -> JSONResponse:
    body = await _read_json(raw_request)
    is_checkpoint_format = body.get("is_checkpoint_format", True)
    return await _run(
        _manager(raw_request).start_update(is_checkpoint_format=is_checkpoint_format),
        {"message": "Weight update started"},
    )


@router.post("/update_weights")
async def update_weights(raw_request: Request) -> JSONResponse:
    body = await _read_json(raw_request)
    update_info = body.get("update_info")
    if update_info is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'update_info' in request body",
        )
    return await _run(
        _manager(raw_request).update(update_info),
        {"message": "Weights updated"},
    )


@router.post("/finish_weight_update")
async def finish_weight_update(raw_request: Request) -> JSONResponse:
    return await _run(
        _manager(raw_request).finish_update(),
        {"message": "Weight update finished"},
    )


# --------------------------------------------------------------------------- #
# Pause / resume
# --------------------------------------------------------------------------- #


@router.post("/pause")
async def pause_generation(
    raw_request: Request,
    mode: str = Query("abort"),
    wait_for_inflight_requests: bool = Query(False),
    clear_cache: bool = Query(True),
) -> JSONResponse:
    """Pause generation so weights can be updated.

    Args:
        mode: ``"abort"`` (default), ``"wait"``, or ``"keep"``.
        wait_for_inflight_requests: DEPRECATED (vLLM parity). When True, treated
            as ``mode="wait"``.
        clear_cache: Flush KV/prefix cache after draining. Ignored for
            ``mode="keep"``.
    """
    # vLLM's deprecated knob: honor it as mode="wait" so older trainers work.
    if wait_for_inflight_requests:
        mode = "wait"
    if mode not in PAUSE_MODES:
        return JSONResponse(
            {
                "error": f"Invalid pause mode: {mode!r}. Must be one of {list(PAUSE_MODES)}."
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    return await _run(
        _manager(raw_request).pause(mode=mode, clear_cache=clear_cache),
        {"status": "paused"},
    )


@router.post("/resume")
async def resume_generation(raw_request: Request) -> JSONResponse:
    return await _run(_manager(raw_request).resume(), {"status": "resumed"})


@router.get("/is_paused")
async def is_paused(raw_request: Request) -> JSONResponse:
    try:
        paused = _manager(raw_request).is_paused()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - defensive, mirrors vLLM
        logger.exception("Failed to fetch pause status")
        return JSONResponse(
            {"error": f"Failed to fetch pause status: {e}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    return JSONResponse(content={"is_paused": paused})


@router.get("/get_world_size")
async def get_world_size(
    raw_request: Request,
    include_dp: bool = Query(True),
) -> JSONResponse:
    """Get the inference world size used to size the NCCL group.

    Args:
        include_dp: If True (default), TP*CP*DP (vLLM ``world_size_across_dp``);
            if False, TP*CP (vLLM ``world_size``).
    """
    world_size = _manager(raw_request).get_world_size(include_dp=include_dp)
    return JSONResponse(content={"world_size": world_size})


# --------------------------------------------------------------------------- #
# App construction / server lifecycle
# --------------------------------------------------------------------------- #


def build_weight_transfer_app(manager: "WeightTransferManager") -> FastAPI:
    """Return a FastAPI app exposing the weight-transfer endpoints.

    The app holds the manager on ``app.state.weight_transfer_manager``; handlers
    fetch it per request (mirrors vLLM's ``request.app.state.engine_client``).
    """
    app = FastAPI(title="tokenspeed weight transfer")
    app.state.weight_transfer_manager = manager
    app.include_router(router)
    return app


def build_weight_transfer_server(
    manager: "WeightTransferManager",
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> uvicorn.Server:
    """Build an unstarted ``uvicorn.Server`` for the weight-transfer app."""
    app = build_weight_transfer_app(manager)
    logger.info("Configuring weight-transfer HTTP control plane on %s:%d", host, port)
    return uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )


async def run_weight_transfer_server(
    manager: "WeightTransferManager",
    *,
    host: str = "127.0.0.1",
    port: int,
) -> None:
    """Serve the weight-transfer app on the current event loop.

    Call this from the engine process on the SAME loop that runs ``AsyncLLM`` so
    the manager's loop-bound primitives (admission event, scheduler
    communicators) are toggled/awaited correctly.
    """
    await build_weight_transfer_server(manager, host=host, port=port).serve()


def attach_to_main_loop(
    manager: "WeightTransferManager",
    *,
    host: str,
    port: int,
    create_task: Callable[[Awaitable[Any]], Any],
) -> Any:
    """Schedule the server as a task via ``create_task`` (e.g. ``loop.create_task``).

    Returns the created task so the caller can track/cancel it on shutdown.
    """
    return create_task(run_weight_transfer_server(manager, host=host, port=port))
