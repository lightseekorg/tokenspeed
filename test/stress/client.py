"""Thin aiohttp client for /v1/chat/completions with cancel-aware streaming.

Exposes one coroutine: `send_chat(...)`. It drives a single request through
its lifecycle, emits events into the sink, and honours an asyncio.Event
cancel signal observed before queue / prefill / decode.

We treat "before first token" as the prefill stage and "after first token" as
the decode stage. That's a slight abuse (the request may still be queued when
we cancel pre-TTFT), but it matches what's observable from the client side.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

from .audits import AuditConfig, ResponseRecord, run_audits
from .events import (
    AUDIT_FINDING,
    REQUEST_CANCELLED,
    REQUEST_COMPLETED,
    REQUEST_ERROR,
    REQUEST_FIRST_TOKEN,
    REQUEST_STALL,
    REQUEST_SUBMITTED,
    JsonlSink,
)


@dataclass
class ChatRequest:
    messages: List[Dict[str, str]]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    stop: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None  # forwarded into the JSON body as-is
    # Cancellation timing. Exactly one of these should be set (or none).
    # `cancel_after_s` fires N seconds after submit, regardless of stage.
    cancel_after_s: Optional[float] = None
    # `cancel_at_stage` fires the moment the request enters that stage:
    #   "queue"   -> immediately after submit (before any bytes)
    #   "prefill" -> same as above, but wait a bit first to land in prefill
    #   "decode"  -> after first token
    cancel_at_stage: Optional[str] = None
    # Optional correlation id (workload name).
    workload: str = ""
    # When set, the `json_schema` auditor validates the completion content
    # against this JSON Schema. A failure is recorded as an audit finding; the
    # request is still reported completed (the server did respond — it just
    # responded wrongly).
    validate_schema: Optional[Dict[str, Any]] = None


def _audit(record: ResponseRecord, cfg: AuditConfig, sink: JsonlSink) -> None:
    """Run the enabled auditors over a completed response, emit findings."""
    if not cfg.enabled:
        return
    for finding in run_audits(record, cfg.enabled):
        sink.emit(
            AUDIT_FINDING,
            rid=record.rid,
            workload=record.workload,
            check=finding.check,
            severity=finding.severity,
            detail=finding.detail,
            value=finding.value,
        )


def _body(model: str, req: ChatRequest) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }
    if req.stop:
        body["stop"] = req.stop
    if req.extra:
        body.update(req.extra)
    return body


async def _maybe_cancel_timer(
    req: ChatRequest,
    first_token: asyncio.Event,
    cancel: asyncio.Event,
) -> None:
    """Background task: decides when to trip the cancel event."""
    stage = req.cancel_at_stage
    if req.cancel_after_s is not None:
        try:
            await asyncio.sleep(req.cancel_after_s)
        except asyncio.CancelledError:
            return
        cancel.set()
        return
    if stage == "queue":
        # Fire instantly -- most likely the request is still in the server queue.
        cancel.set()
        return
    if stage == "prefill":
        # Give the server a moment to dequeue + start prefill.
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return
        if not first_token.is_set():
            cancel.set()
        return
    if stage == "decode":
        try:
            await first_token.wait()
        except asyncio.CancelledError:
            return
        # Let a few tokens stream so we're solidly in decode.
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return
        cancel.set()
        return
    # No cancellation requested.
    return


async def send_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    req: ChatRequest,
    sink: JsonlSink,
    timeout_s: float = 600.0,
    audit_cfg: Optional[AuditConfig] = None,
) -> str:
    """Drive one request. Returns outcome: 'completed', 'cancelled', or 'error'.

    Errors are also emitted into the sink; the return value is there so the
    runner's circuit breaker can react without re-parsing events. When
    ``audit_cfg`` enables auditors, the completed response is checked for
    output-quality problems (emitting `audit_finding` events); a positive
    ``stall_timeout_s`` also flags streaming wedges via `request_stall`.
    """
    cfg = audit_cfg if audit_cfg is not None else AuditConfig(stall_timeout_s=0.0)
    capture = bool(cfg.enabled)

    rid = f"ss-{uuid.uuid4().hex[:12]}"
    submit_ts = time.time()
    sink.emit(
        REQUEST_SUBMITTED,
        rid=rid,
        workload=req.workload,
        stream=req.stream,
        max_tokens=req.max_tokens,
        cancel_at_stage=req.cancel_at_stage,
    )

    first_token = asyncio.Event()
    cancel = asyncio.Event()
    canceller = asyncio.create_task(_maybe_cancel_timer(req, first_token, cancel))

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = _body(model, req)

    # Server-reported usage vs. what the client actually observed on the wire.
    # They diverge when the engine bills tokens that carry no visible content.
    reported_prompt_tokens = 0
    reported_completion_tokens = 0
    observed_visible_tokens = 0
    finish_reason: Optional[str] = None
    ttft_s: Optional[float] = None
    max_gap = 0.0
    content_parts: List[str] = []
    content_len = 0
    content_truncated = False

    def _add_content(text: str) -> None:
        nonlocal content_len, content_truncated
        if not capture or not text:
            return
        remaining = cfg.content_cap - content_len
        if remaining <= 0:
            content_truncated = True
            return
        if len(text) > remaining:
            content_parts.append(text[:remaining])
            content_len = cfg.content_cap
            content_truncated = True
        else:
            content_parts.append(text)
            content_len += len(text)

    async def _emit_cancelled(stage: str) -> None:
        sink.emit(REQUEST_CANCELLED, rid=rid, stage=stage, workload=req.workload)

    def _audit_completed(stream: bool) -> None:
        _audit(
            ResponseRecord(
                rid=rid,
                workload=req.workload,
                stream=stream,
                content="".join(content_parts),
                content_truncated=content_truncated,
                reported_completion_tokens=reported_completion_tokens,
                reported_prompt_tokens=reported_prompt_tokens,
                observed_visible_tokens=observed_visible_tokens,
                finish_reason=finish_reason,
                ttft_s=ttft_s,
                e2e_s=time.time() - submit_ts,
                max_inter_token_gap_s=max_gap,
                validate_schema=req.validate_schema,
            ),
            cfg,
            sink,
        )

    outcome = "error"
    try:
        # Pre-flight: if cancel-at-queue already tripped, bail before opening.
        if cancel.is_set():
            await _emit_cancelled("queue_preflight")
            outcome = "cancelled"
            return outcome

        timeout = aiohttp.ClientTimeout(total=timeout_s)

        if not req.stream:
            # Non-streaming servers send no response headers until generation
            # is finished, so `session.post` blocks for the whole request and
            # there is nothing to interrupt "mid-stream". Race the entire
            # request against the cancel event instead; if cancel wins we
            # cancel the request task, which aborts the connection so the
            # server observes a client disconnect.
            async def _issue_nonstream():
                async with session.post(url, json=body, timeout=timeout) as resp:
                    if resp.status != 200:
                        return ("http_error", resp.status, await resp.text())
                    return ("ok", resp.status, await resp.json())

            req_task = asyncio.create_task(_issue_nonstream())
            cancel_task = asyncio.create_task(cancel.wait())
            done, pending = await asyncio.wait(
                [req_task, cancel_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel and reap the loser so it can't leak as a pending task.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if req_task not in done:
                await _emit_cancelled("non_stream")
                outcome = "cancelled"
                return outcome
            # .result() re-raises any ClientError/TimeoutError from the task,
            # which the outer handlers below turn into a request_error event.
            status_kind, status, body_payload = req_task.result()
            if status_kind == "http_error":
                sink.emit(
                    REQUEST_ERROR,
                    rid=rid,
                    error_kind=f"http_{status}",
                    detail=str(body_payload)[:500],
                    workload=req.workload,
                )
                return outcome
            usage = body_payload.get("usage") or {}
            reported_prompt_tokens = usage.get("prompt_tokens", 0)
            reported_completion_tokens = usage.get("completion_tokens", 0)
            try:
                choice0 = (body_payload.get("choices") or [{}])[0]
                content = choice0.get("message", {}).get("content") or ""
                finish_reason = choice0.get("finish_reason")
            except Exception:
                content = ""
            observed_visible_tokens = len(content.split())
            _add_content(content)
            _audit_completed(stream=False)
            sink.emit(
                REQUEST_COMPLETED,
                rid=rid,
                workload=req.workload,
                prompt_tokens=reported_prompt_tokens,
                output_tokens=reported_completion_tokens,
                visible_tokens=observed_visible_tokens,
                stream=False,
            )
            outcome = "completed"
            return outcome

        # Streaming path: headers arrive promptly, so cancel is checked
        # per-chunk inside the read loop below.
        async with session.post(url, json=body, timeout=timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                sink.emit(
                    REQUEST_ERROR,
                    rid=rid,
                    error_kind=f"http_{resp.status}",
                    detail=text[:500],
                    workload=req.workload,
                )
                return outcome

            # Streaming: parse SSE line-by-line, mark TTFT on first content
            # chunk. Each chunk read is bounded by the stall timeout so a
            # mid-decode wedge surfaces long before the total request timeout.
            stall_timeout = cfg.stall_timeout_s
            aiter = resp.content.__aiter__()
            last_chunk_ts = time.time()
            while True:
                if cancel.is_set():
                    stage = "decode" if first_token.is_set() else "prefill"
                    await _emit_cancelled(stage)
                    outcome = "cancelled"
                    return outcome
                try:
                    if stall_timeout and stall_timeout > 0:
                        raw = await asyncio.wait_for(
                            aiter.__anext__(), timeout=stall_timeout
                        )
                    else:
                        raw = await aiter.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    stage = "decode" if first_token.is_set() else "prefill"
                    gap = time.time() - last_chunk_ts
                    sink.emit(
                        REQUEST_STALL,
                        rid=rid,
                        stage=stage,
                        gap_s=round(gap, 3),
                        workload=req.workload,
                    )
                    sink.emit(
                        REQUEST_ERROR,
                        rid=rid,
                        error_kind="stall",
                        detail=f"no token for {gap:.1f}s in {stage}",
                        workload=req.workload,
                    )
                    return outcome  # outcome stays "error"
                now = time.time()
                if first_token.is_set():
                    max_gap = max(max_gap, now - last_chunk_ts)
                last_chunk_ts = now
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # OpenAI-style: choices[0].delta.content for streaming.
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                    if content:
                        if not first_token.is_set():
                            first_token.set()
                            ttft_s = time.time() - submit_ts
                            sink.emit(
                                REQUEST_FIRST_TOKEN, rid=rid, workload=req.workload
                            )
                        observed_visible_tokens += 1
                        _add_content(content)
                # Some servers put usage only in the final chunk.
                usage = chunk.get("usage")
                if usage:
                    reported_prompt_tokens = usage.get(
                        "prompt_tokens", reported_prompt_tokens
                    )
                    if usage.get("completion_tokens"):
                        reported_completion_tokens = usage["completion_tokens"]
            # If the server never reported completion tokens, fall back to the
            # visible count so throughput accounting still works.
            if reported_completion_tokens == 0:
                reported_completion_tokens = observed_visible_tokens
            _audit_completed(stream=True)
            sink.emit(
                REQUEST_COMPLETED,
                rid=rid,
                workload=req.workload,
                prompt_tokens=reported_prompt_tokens,
                output_tokens=reported_completion_tokens,
                visible_tokens=observed_visible_tokens,
                stream=True,
            )
            outcome = "completed"
    except asyncio.TimeoutError:
        sink.emit(REQUEST_ERROR, rid=rid, error_kind="timeout", workload=req.workload)
    except aiohttp.ClientError as e:
        sink.emit(
            REQUEST_ERROR,
            rid=rid,
            error_kind=type(e).__name__,
            detail=str(e)[:500],
            workload=req.workload,
        )
    except asyncio.CancelledError:
        # Runner-level cancel (e.g. Ctrl-C): don't swallow.
        raise
    except Exception as e:  # noqa: BLE001 -- we want a single catch-all for the sink
        sink.emit(
            REQUEST_ERROR,
            rid=rid,
            error_kind=type(e).__name__,
            detail=str(e)[:500],
            workload=req.workload,
        )
    finally:
        canceller.cancel()
        try:
            await canceller
        except asyncio.CancelledError:
            pass
    return outcome
