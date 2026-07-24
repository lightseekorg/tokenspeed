from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import signal
import sys
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from smg_grpc_servicer.tokenspeed.scheduler_launcher import launch_engine

from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.utils.server_args import prepare_server_args
from tokenspeed.version import __version__

logger = logging.getLogger("tokenspeed_http_worker")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
)

app = FastAPI()
async_llm = None
server_args = None
scheduler_info: dict[str, Any] = {}
started_at = time.time()


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        return _jsonable(dataclasses.asdict(obj))
    return str(obj)


def _model_id() -> str:
    served = getattr(server_args, "served_model_name", None)
    if served:
        if isinstance(served, (list, tuple)):
            return str(served[0])
        return str(served)
    return str(getattr(server_args, "model", "tokenspeed-model"))


def _messages_to_text(
    messages: list[dict[str, Any]], continue_final_message: bool = False
) -> str:
    tokenizer = getattr(async_llm, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not continue_final_message,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                logger.exception("apply_chat_template failed")
        except Exception:
            logger.exception("apply_chat_template failed")
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_bits = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in (
                    "text",
                    "input_text",
                ):
                    text_bits.append(str(item.get("text", "")))
            content = "\n".join(text_bits)
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def _sampling_params(data: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "repetition_penalty": "repetition_penalty",
        "stop": "stop",
        "stop_token_ids": "stop_token_ids",
        "ignore_eos": "ignore_eos",
        "skip_special_tokens": "skip_special_tokens",
        "spaces_between_special_tokens": "spaces_between_special_tokens",
        "no_stop_trim": "no_stop_trim",
        "n": "n",
        "logit_bias": "logit_bias",
        "regex": "regex",
        "json_schema": "json_schema",
        "structural_tag": "structural_tag",
    }
    out: dict[str, Any] = {}
    for src, dst in mapping.items():
        if src in data and data[src] is not None:
            out[dst] = data[src]
    max_tokens = data.get(
        "max_completion_tokens", data.get("max_tokens", data.get("max_new_tokens"))
    )
    if max_tokens is not None:
        out["max_new_tokens"] = int(max_tokens)
    min_tokens = data.get(
        "min_completion_tokens", data.get("min_tokens", data.get("min_new_tokens"))
    )
    if min_tokens is not None:
        out["min_new_tokens"] = int(min_tokens)
    return out


def _request_to_generate_input(data: dict[str, Any], *, rid: str) -> GenerateReqInput:
    if "input_ids" in data and data["input_ids"] is not None:
        text = None
        input_ids = data["input_ids"]
    elif "prompt" in data and data["prompt"] is not None:
        text = data["prompt"]
        input_ids = None
    elif "text" in data and data["text"] is not None:
        text = data["text"]
        input_ids = None
    elif "messages" in data and data["messages"] is not None:
        text = _messages_to_text(
            data["messages"],
            continue_final_message=bool(data.get("continue_final_message", False)),
        )
        input_ids = None
    else:
        raise ValueError("request must include messages, prompt, text, or input_ids")

    obj = GenerateReqInput(
        text=text,
        input_ids=input_ids,
        sampling_params=_sampling_params(data),
        return_logprob=bool(data.get("return_logprob", data.get("logprobs", False))),
        top_logprobs_num=int(
            data.get("top_logprobs", data.get("top_logprobs_num", 0)) or 0
        ),
        token_ids_logprob=data.get("token_ids_logprob"),
        stream=bool(data.get("stream", False)),
        bootstrap_host=data.get("bootstrap_host"),
        bootstrap_port=data.get("bootstrap_port"),
        bootstrap_room=data.get("bootstrap_room"),
        custom_logit_processor=data.get("custom_logit_processor"),
        return_hidden_states=bool(data.get("return_hidden_states", False)),
    )
    obj.rid = rid
    return obj


def _finish_reason(meta: dict[str, Any]) -> str | None:
    reason = meta.get("finish_reason")
    if reason is None:
        return None
    if isinstance(reason, dict):
        kind = reason.get("type") or reason.get("name")
        if kind == "length":
            return "length"
        if kind == "abort":
            return "abort"
        return "stop"
    text = str(reason).lower()
    if "length" in text:
        return "length"
    if "abort" in text:
        return "abort"
    return "stop"


def _usage(meta: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
    completion_tokens = int(meta.get("completion_tokens", 0) or 0)
    cached_tokens = int(meta.get("cached_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


def _openai_response(
    data: dict[str, Any], output: dict[str, Any], *, chat: bool, rid: str
) -> dict[str, Any]:
    meta = output.get("meta_info", {}) or {}
    text = output.get("text", "") or ""
    choice: dict[str, Any] = {
        "index": int(output.get("index", 0) or 0),
        "finish_reason": _finish_reason(meta) or "stop",
    }
    if chat:
        choice["message"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
    if "output_ids" in output:
        choice["output_ids"] = output.get("output_ids")
    return {
        "id": rid,
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": data.get("model") or _model_id(),
        "choices": [choice],
        "usage": _usage(meta),
        "meta_info": _jsonable(meta),
    }


async def _run_generate(data: dict[str, Any], *, chat: bool) -> dict[str, Any]:
    rid = data.get("request_id") or data.get("id") or f"cmpl-{uuid.uuid4().hex}"
    obj = _request_to_generate_input(data, rid=rid)
    final = None
    async for output in async_llm.generate_request(obj):
        final = output
    if isinstance(final, list):
        final = final[0] if final else {"text": "", "meta_info": {}}
    if final is None:
        final = {"text": "", "meta_info": {"finish_reason": "stop"}}
    return _openai_response(data, final, chat=chat, rid=rid)


async def _stream_generate(data: dict[str, Any], *, chat: bool):
    rid = data.get("request_id") or data.get("id") or f"cmpl-{uuid.uuid4().hex}"
    obj = _request_to_generate_input(data, rid=rid)
    prev = ""
    async for output in async_llm.generate_request(obj):
        if isinstance(output, list):
            output = output[0] if output else {"text": "", "meta_info": {}}
        meta = output.get("meta_info", {}) or {}
        text = output.get("text", "") or ""
        delta = text[len(prev) :] if text.startswith(prev) else text
        prev = text
        finish = _finish_reason(meta)
        choice: dict[str, Any] = {
            "index": int(output.get("index", 0) or 0),
            "finish_reason": finish,
        }
        if chat:
            choice["delta"] = {"content": delta}
        else:
            choice["text"] = delta
        payload = {
            "id": rid,
            "object": "chat.completion.chunk" if chat else "text_completion.chunk",
            "created": int(time.time()),
            "model": data.get("model") or _model_id(),
            "choices": [choice],
        }
        if finish is not None:
            payload["usage"] = _usage(meta)
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
@app.get("/readiness")
async def health():
    return {"status": "ok", "healthy": True}


@app.get("/version")
async def version():
    return {"version": __version__}


@app.get("/server_info")
async def server_info():
    return {
        "server_type": "tokenspeed-http-worker",
        "active_requests": len(getattr(async_llm, "rid_to_state", {}) or {}),
        "uptime_seconds": time.time() - started_at,
        "model_path": getattr(server_args, "model", None),
        "scheduler_info": _jsonable(scheduler_info),
    }


@app.get("/v1/models")
@app.get("/models")
async def models():
    return {"object": "list", "data": [{"id": _model_id(), "object": "model"}]}


@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    resp = await _run_generate(data, chat=False)
    return ORJSONResponse(content=resp)


@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json()
    if data.get("stream", False):
        return StreamingResponse(
            _stream_generate(data, chat=False), media_type="text/event-stream"
        )
    return ORJSONResponse(content=await _run_generate(data, chat=False))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    if data.get("stream", False):
        return StreamingResponse(
            _stream_generate(data, chat=True), media_type="text/event-stream"
        )
    return ORJSONResponse(content=await _run_generate(data, chat=True))


@app.post("/start_profile")
async def start_profile():
    await async_llm.start_profile()
    return {"status": "ok"}


@app.post("/stop_profile")
async def stop_profile():
    await async_llm.stop_profile()
    return {"status": "ok"}


def main(argv: list[str] | None = None) -> None:
    global async_llm, server_args, scheduler_info
    if argv is None:
        argv = sys.argv[1:]
    server_args = prepare_server_args(argv)
    logger.info(
        "Launching TokenSpeed HTTP worker on %s:%s", server_args.host, server_args.port
    )
    async_llm, scheduler_info = launch_engine(server_args)

    def _shutdown(signum, frame):
        logger.info("received signal %s", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)
    uvicorn.run(app, host=server_args.host, port=server_args.port, log_level="info")


if __name__ == "__main__":
    main()
