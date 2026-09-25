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

"""SGLang-compatible HTTP control routes for RL rollout engines.

The endpoint names and JSON fields match the surface used by trainers such as
slime. Handlers translate requests into TokenSpeed's native scheduler, memory,
and weight-update operations. Heavy distributed weight payloads travel over the
trainer-created process group; HTTP carries only metadata.

The app must run on ``AsyncLLM``'s event loop because scheduler communicators are
loop-bound. Use :func:`build_sglang_compat_app` to construct it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, get_args

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from tokenspeed.runtime.cache.l3.backend import (
    L3_FLUSH_REQUIRES_WEIGHT_VERSION,
    resolve_l3_weight_version,
)
from tokenspeed.runtime.engine.io_struct import (
    SUPPORTED_WEIGHT_UPDATE_SOURCES,
    DestroyWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqInput,
    PauseMode,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from tokenspeed.runtime.entrypoints.rl_control import install_bearer_auth
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


def _stamp_weight_version(request: Request, version: str | None, message: str) -> str:
    """Apply weight_version to server_args on success, return updated message."""
    if version is not None:
        _llm(request).server_args.weight_version = version
        message += f" Weight version updated to {version}."
    return message


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


async def _json_body(request: Request) -> dict[str, Any]:
    """The JSON object body. Missing, malformed or non-object is a client error."""
    raw = await request.body()
    if not raw:
        raise ValueError("request body must be a JSON object")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"request body is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


async def _optional_json_body(request: Request) -> dict[str, Any]:
    """Like :func:`_json_body`, but an absent or malformed body is ``{}``."""
    try:
        return await _json_body(request)
    except ValueError:
        return {}


PAUSE_MODES: frozenset[str] = frozenset(get_args(PauseMode))


def _unsupported_source(source: str) -> JSONResponse | None:
    """A 501 refusal unless the scheduler implements this weight-update source.

    The scheduler's dispatcher raises ``NotImplementedError`` on a request type
    it has no branch for, which kills the scheduler process and the engine with
    it. Refuse here instead of forwarding. ``None`` means the source is
    supported and the route may run.
    """
    if source in SUPPORTED_WEIGHT_UPDATE_SOURCES:
        return None
    supported = ", ".join(sorted(SUPPORTED_WEIGHT_UPDATE_SOURCES))
    return JSONResponse(
        {
            "success": False,
            "message": (
                f"update_weights_from_{source} is not implemented by this "
                f"build's scheduler; supported sources: {supported}"
            ),
        },
        status_code=HTTPStatus.NOT_IMPLEMENTED.value,
    )


# --------------------------------------------------------------------------- #
# Process group setup
# --------------------------------------------------------------------------- #


@router.post("/init_weights_update_group")
async def init_weights_update_group(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        body = await _json_body(request)
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
    async def _do() -> dict[str, Any]:
        # Body is optional: trainers that always call destroy (e.g. slime) may
        # send only ``{group_name}`` or nothing at all. Tolerate an empty body.
        body = await _optional_json_body(request)
        obj = DestroyWeightsUpdateGroupReqInput(
            group_name=str(body.get("group_name", "weight_update_group")),
        )
        success, message = await _llm(request).destroy_weights_update_group(obj)
        return {"success": success, "message": message}

    return await _guarded(_do)


# --------------------------------------------------------------------------- #
# Weight updates
# --------------------------------------------------------------------------- #


@router.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        body = await _json_body(request)
        names = list(body["names"])
        dtypes = list(body["dtypes"])  # SGLang field name
        shapes = [list(s) for s in body["shapes"]]
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError("names, dtypes, shapes must have equal length")
        flush_cache = bool(body.get("flush_cache", False))
        llm = _llm(request)
        requested_version = body.get("weight_version")
        if requested_version is not None:
            requested_version = str(requested_version)
        storage_backend = getattr(llm.server_args, "kvstore_storage_backend", None)
        if flush_cache and requested_version is None and storage_backend is not None:
            return {
                "success": False,
                "message": L3_FLUSH_REQUIRES_WEIGHT_VERSION,
            }
        weight_version = resolve_l3_weight_version(
            llm.server_args.weight_version,
            requested_version,
            flush_cache=flush_cache,
            storage_backend=storage_backend,
        )
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtype_names=dtypes,  # translate dtypes -> dtype_names
            shapes=shapes,
            group_name=str(body.get("group_name", "weight_update_group")),
            flush_cache=flush_cache,
            weight_version=weight_version,
        )
        success, message = await llm.update_weights_from_distributed(obj)
        if success:
            message = _stamp_weight_version(request, obj.weight_version, message)
        return {"success": success, "message": message}

    return await _guarded(_do)


@router.post("/update_weights_from_tensor")
async def update_weights_from_tensor(request: Request) -> JSONResponse:
    refusal = _unsupported_source("tensor")
    if refusal is not None:
        return refusal

    async def _do() -> dict[str, Any]:
        body = await _json_body(request)
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=body["serialized_named_tensors"],
            load_format=body.get("load_format"),
            flush_cache=bool(body.get("flush_cache", False)),
            weight_version=body.get("weight_version"),
        )
        success, message = await _llm(request).update_weights_from_tensor(obj)
        if success:
            message = _stamp_weight_version(request, obj.weight_version, message)
        return {"success": success, "message": message}

    return await _guarded(_do)


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request) -> JSONResponse:
    refusal = _unsupported_source("disk")
    if refusal is not None:
        return refusal

    async def _do() -> dict[str, Any]:
        body = await _json_body(request)
        obj = UpdateWeightFromDiskReqInput(
            model_path=str(body["model_path"]),
            load_format=body.get("load_format"),
            weight_version=body.get("weight_version"),
        )
        success, message, *_ = await _llm(request).update_weights_from_disk(obj)
        if success:
            message = _stamp_weight_version(request, obj.weight_version, message)
        return {"success": success, "message": message}

    return await _guarded(_do)


# --------------------------------------------------------------------------- #
# Pause / resume (admission gate)
# --------------------------------------------------------------------------- #


@router.post("/pause_generation")
async def pause_generation(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        body = await _optional_json_body(request)
        mode = str(body.get("mode", "wait"))
        if mode not in PAUSE_MODES:
            raise ValueError(
                f"invalid pause mode: {mode!r} (expected one of {sorted(PAUSE_MODES)})"
            )
        # Stop frontend admission before the native scheduler drain. Otherwise
        # a newly buffered request could hold the model-update reader lock.
        llm = _llm(request)
        llm.block_generation_admission()
        try:
            if not await llm.pause_scheduler(mode=mode):
                raise RuntimeError("Failed to pause generation.")
        except BaseException:
            llm.allow_generation_admission()
            raise
        return {"success": True, "message": "Paused generation.", "mode": mode}

    return await _guarded(_do)


@router.post("/continue_generation")
async def continue_generation(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        llm = _llm(request)
        if not await llm.resume_scheduler():
            raise RuntimeError("Failed to continue generation.")
        llm.allow_generation_admission()
        return {"success": True, "message": "Continued generation."}

    return await _guarded(_do)


# --------------------------------------------------------------------------- #
# Cache / memory
# --------------------------------------------------------------------------- #


@router.api_route("/flush_cache", methods=["GET", "POST"])
async def flush_cache(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        await _llm(request).flush_cache()
        return {"success": True, "message": "Cache flushed."}

    return await _guarded(_do)


@router.post("/release_memory_occupation")
async def release_memory_occupation(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        body = await _optional_json_body(request)
        result = await _llm(request).release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=body.get("tags"))
        )
        return {"success": result.success, "message": result.message}

    return await _guarded(_do)


@router.post("/resume_memory_occupation")
async def resume_memory_occupation(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        body = await _optional_json_body(request)
        result = await _llm(request).resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=body.get("tags"))
        )
        return {"success": result.success, "message": result.message}

    return await _guarded(_do)


# --------------------------------------------------------------------------- #
# Misc / health
# --------------------------------------------------------------------------- #


@router.post("/abort_request")
async def abort_request(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        body = await _json_body(request)
        llm = _llm(request)
        if body.get("abort_all"):
            # Native abort mode waits until scheduler state is drained. Resume
            # immediately because SGLang's abort endpoint does not leave the
            # server paused.
            llm.block_generation_admission()
            try:
                if not await llm.pause_scheduler(mode="abort"):
                    raise RuntimeError("Failed to abort all requests.")
            except BaseException:
                llm.allow_generation_admission()
                raise
            if not await llm.resume_scheduler():
                raise RuntimeError("Failed to resume after aborting requests.")
            llm.allow_generation_admission()
        elif body.get("rid"):
            llm.abort_request(str(body["rid"]))
        return {"success": True}

    return await _guarded(_do)


@router.get("/health_generate")
async def health_generate() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@router.get("/v1/loads")
async def get_loads(request: Request) -> JSONResponse:
    loads = await _llm(request).get_load()
    return JSONResponse(
        {
            "loads": [
                {
                    "dp_rank": load.dp_rank,
                    "num_reqs": load.num_reqs,
                    "num_waiting_reqs": load.num_waiting_reqs,
                    "num_pages": load.num_pages,
                }
                for load in loads
            ]
        }
    )


@router.get("/get_weight_version")
async def get_weight_version(request: Request) -> JSONResponse:
    llm = _llm(request)
    return JSONResponse({"weight_version": llm.server_args.weight_version})


@router.post("/update_weight_version")
async def update_weight_version(request: Request) -> JSONResponse:
    async def _do() -> dict[str, Any]:
        body = await _json_body(request)
        new_version = body.get("new_version")
        if new_version is None:
            raise ValueError("Missing 'new_version' in request body")
        llm = _llm(request)
        if getattr(llm.server_args, "kvstore_storage_backend", None) is not None:
            raise ValueError(
                "Direct weight-version changes are not supported with L3 storage; "
                "use /update_weights_from_distributed with an explicit "
                "weight_version and flush_cache=True"
            )
        llm.server_args.weight_version = str(new_version)
        return {
            "success": True,
            "message": f"Weight version updated to {new_version}",
            "new_version": str(new_version),
        }

    return await _guarded(_do)


@router.get("/model_info")
async def model_info(request: Request) -> JSONResponse:
    llm = _llm(request)
    return JSONResponse(
        {
            "model_path": llm.server_args.model,
            "weight_version": llm.server_args.weight_version,
        }
    )


# --------------------------------------------------------------------------- #
# App construction
# --------------------------------------------------------------------------- #


def build_sglang_compat_app(async_llm: "AsyncLLM") -> FastAPI:
    """Return the FastAPI app exposing the RL control routes slime and a fronting gateway drive.

    This is the app ``AsyncLLM._serve_rl_control_plane`` serves in production
    on ``--rl-control-port``; tests build it the same way. When
    ``--rl-control-api-key`` is set every route requires that bearer.
    """
    app = FastAPI(title="tokenspeed SGLang-compatible RL control")
    app.state.async_llm = async_llm
    server_args = getattr(async_llm, "server_args", None)
    api_key = getattr(server_args, "rl_control_api_key", None)
    if api_key:
        install_bearer_auth(app, api_key)
    app.include_router(router)
    return app
