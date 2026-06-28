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

"""SGLang-compatible weight-sync HTTP routes.

A thin adapter exposing the SGLang HTTP weight-sync surface (the endpoint names,
methods, and JSON field names RL trainers such as slime / miles -- and verl's
SGLang rollout -- POST to), forwarding each call to ``AsyncLLM``'s existing
weight-control methods. No new engine logic; only name/field translation.

These routes are mounted on the **same** in-engine RL control-plane app as the
vLLM-compatible endpoints (``vllm_compat_http.py``) -- one app, one port. Mount
``router`` onto an app whose ``state.async_llm`` is set, or use
:func:`build_sglang_compat_app` for a standalone app (tests).

Like the rest of the RL control plane, it must run on ``AsyncLLM``'s event loop:
handlers toggle the loop-bound admission gate and await loop-bound scheduler
communicators. Heavy weight payloads travel out-of-band (NCCL / CUDA-IPC); only
metadata flows here, and the worker-side receive+load is the same deferred piece
as the rest of the RL plumbing.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from tokenspeed.runtime.engine.io_struct import (
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

logger = get_colorful_logger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _llm(request: Request) -> "AsyncLLM":
    async_llm = getattr(request.app.state, "async_llm", None)
    if async_llm is None:
        raise RuntimeError("AsyncLLM is not configured on this server.")
    return async_llm


async def _guarded(
    build_payload: Callable[[], Awaitable[dict[str, Any]]],
) -> JSONResponse:
    """Run a handler body and map exceptions to SGLang-style responses.

    Bad/missing request fields -> 400; anything else -> 500. The body returns the
    success payload dict (``success=False`` is surfaced as 400).
    """
    try:
        payload = await build_payload()
    except (KeyError, TypeError, ValueError) as e:
        return JSONResponse(
            {"success": False, "message": f"Invalid request: {e}"},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as e:  # noqa: BLE001 - surface engine errors as 500
        logger.exception("sglang-compat request failed")
        return JSONResponse(
            {"success": False, "message": str(e)},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    status = (
        HTTPStatus.OK.value
        if payload.get("success", True)
        else HTTPStatus.BAD_REQUEST.value
    )
    return JSONResponse(payload, status_code=status)


# --------------------------------------------------------------------------- #
# Process group setup
# --------------------------------------------------------------------------- #


@router.post("/init_weights_update_group")
async def init_weights_update_group(request: Request) -> JSONResponse:
    body = await request.json()

    async def _do() -> dict[str, Any]:
        obj = InitWeightsUpdateGroupReqInput(
            master_address=str(body["master_address"]),
            master_port=int(body["master_port"]),
            rank_offset=int(body["rank_offset"]),
            world_size=int(body["world_size"]),
            group_name=str(body.get("group_name", "weight_update_group")),
            backend=str(body.get("backend", "nccl")),
        )
        success, message = await _llm(request).init_weights_update_group(obj)
        return {"success": success, "message": message}

    return await _guarded(_do)


@router.post("/destroy_weights_update_group")
async def destroy_weights_update_group(request: Request) -> JSONResponse:
    # tokenspeed tears the group down with the engine; expose a no-op so
    # trainers that always call it (slime swallows failures) succeed.
    return JSONResponse({"success": True, "message": "noop"})


# --------------------------------------------------------------------------- #
# Weight updates
# --------------------------------------------------------------------------- #


@router.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: Request) -> JSONResponse:
    body = await request.json()

    async def _do() -> dict[str, Any]:
        names = list(body["names"])
        dtypes = list(body["dtypes"])  # SGLang field name
        shapes = [list(s) for s in body["shapes"]]
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError("names, dtypes, shapes must have equal length")
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtype_names=dtypes,  # translate dtypes -> dtype_names
            shapes=shapes,
            group_name=str(body.get("group_name", "weight_update_group")),
            flush_cache=bool(body.get("flush_cache", False)),
        )
        success, message = await _llm(request).update_weights_from_distributed(obj)
        return {"success": success, "message": message}

    return await _guarded(_do)


@router.post("/update_weights_from_tensor")
async def update_weights_from_tensor(request: Request) -> JSONResponse:
    body = await request.json()

    async def _do() -> dict[str, Any]:
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=body["serialized_named_tensors"],
            load_format=body.get("load_format"),
            flush_cache=bool(body.get("flush_cache", False)),
        )
        success, message = await _llm(request).update_weights_from_tensor(obj)
        return {"success": success, "message": message}

    return await _guarded(_do)


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request) -> JSONResponse:
    body = await request.json()

    async def _do() -> dict[str, Any]:
        obj = UpdateWeightFromDiskReqInput(
            model_path=str(body["model_path"]),
            load_format=body.get("load_format"),
        )
        success, message, *_ = await _llm(request).update_weights_from_disk(obj)
        return {"success": success, "message": message}

    return await _guarded(_do)


# --------------------------------------------------------------------------- #
# Pause / resume (admission gate)
# --------------------------------------------------------------------------- #


@router.post("/pause_generation")
async def pause_generation(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        # Block new admission. In-flight requests are not aborted (matches
        # SGLang's freeze intent); true KV-preserving freeze needs the C++
        # scheduler (same keep-mode limitation as the rest of the plumbing).
        _llm(request).weight_transfer_block_admission()
        return {"success": True, "message": "Paused generation."}

    return await _guarded(_do)


@router.post("/continue_generation")
async def continue_generation(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        _llm(request).weight_transfer_allow_admission()
        return {"success": True, "message": "Continued generation."}

    return await _guarded(_do)


# --------------------------------------------------------------------------- #
# Cache / memory
# --------------------------------------------------------------------------- #


@router.get("/flush_cache")
async def flush_cache(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        await _llm(request).flush_cache()
        return {"success": True, "message": "Cache flushed."}

    return await _guarded(_do)


@router.post("/release_memory_occupation")
async def release_memory_occupation(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        await _llm(request).release_memory_occupation(ReleaseMemoryOccupationReqInput())
        return {"success": True}

    return await _guarded(_do)


@router.post("/resume_memory_occupation")
async def resume_memory_occupation(request: Request) -> JSONResponse:
    # SGLang's multi-stage `tags` (weights/kv_cache) is accepted and ignored;
    # tokenspeed resumes the full occupation.
    async def _do() -> dict[str, Any]:
        await _llm(request).resume_memory_occupation(ResumeMemoryOccupationReqInput())
        return {"success": True}

    return await _guarded(_do)


# --------------------------------------------------------------------------- #
# Misc / health
# --------------------------------------------------------------------------- #


@router.post("/abort_request")
async def abort_request(request: Request) -> JSONResponse:
    body = await request.json()

    async def _do() -> dict[str, Any]:
        llm = _llm(request)
        if body.get("abort_all"):
            llm.weight_transfer_abort_inflight()
        elif body.get("rid"):
            llm.abort_request(str(body["rid"]))
        return {"success": True}

    return await _guarded(_do)


@router.get("/health_generate")
async def health_generate() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# --------------------------------------------------------------------------- #
# App construction (standalone, for tests)
# --------------------------------------------------------------------------- #


def build_sglang_compat_app(async_llm: "AsyncLLM") -> FastAPI:
    """Return a standalone FastAPI app exposing only the SGLang-compat routes.

    In production these routes are mounted on the shared RL control-plane app
    (see ``AsyncLLM._serve_rl_control_plane``). This helper is for isolated tests.
    """
    app = FastAPI(title="tokenspeed sglang-compat weight sync")
    app.state.async_llm = async_llm
    app.include_router(router)
    return app
