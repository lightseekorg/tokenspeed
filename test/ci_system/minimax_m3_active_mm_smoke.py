#!/usr/bin/env python3
"""Validate MiniMax-M3 active image serving and encoder CUDA Graph reuse."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from minimax_m3_gsm8k import probe_health

ENCODER_GRAPH_BUDGETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 2304)
EXPECTED_TP_RANKS = 4
DOG_EXPECTED_TOKEN = "Dog"
DOG_EXPECTED_PROMPT_TOKENS = 254
DOG_MAX_ABS_LOGPROB_DELTA = 0.02
VIDEO_ERROR_CODE = "invalid_multimodal_request"
MAX_RECORDED_FAILURES = 50

_INITIALIZED_RE = re.compile(
    r"EncoderCudaGraphWrapper initialized: modality=image, budgets=\[([^]]+)\]"
)
_CAPTURED_RE = re.compile(r"Captured encoder cudagraph: modality=image, budget=(\d+)")
_COMPLETE_RE = re.compile(
    r"Encoder CUDA graph capture complete: modality=image, (\d+) budget graphs\."
)
_INSTALLED_RE = re.compile(r"Installed encoder CUDA graphs for .*image_encoder")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle)
    os.replace(temporary, destination)


def _append_failure(failures: list[str], message: str) -> None:
    if len(failures) < MAX_RECORDED_FAILURES:
        failures.append(message)


@dataclass(frozen=True)
class ActiveMMSmokeConfig:
    base_url: str
    control_url: str
    model: str
    dog: Path
    pug: Path
    banner: Path
    reference: Path
    server_log: Path
    output_dir: Path
    request_timeout_seconds: float


@dataclass(frozen=True)
class RequestCase:
    name: str
    payload: dict[str, Any]
    expected_text: str | None = None


def make_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError(f"unsupported image fixture type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_budgets(value: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        return None


def summarize_encoder_graph_log(text: str) -> dict[str, Any]:
    failures: list[str] = []
    initialized_matches = _INITIALIZED_RE.findall(text)
    parsed_initialized = [_parse_budgets(value) for value in initialized_matches]
    initialized = len(initialized_matches)
    if initialized != EXPECTED_TP_RANKS:
        failures.append(
            f"expected {EXPECTED_TP_RANKS} encoder wrapper initializations, found {initialized}"
        )
    for rank_index, budgets in enumerate(parsed_initialized):
        if budgets != ENCODER_GRAPH_BUDGETS:
            failures.append(
                f"encoder wrapper initialization {rank_index} has budgets {budgets!r}"
            )

    complete_values = [int(value) for value in _COMPLETE_RE.findall(text)]
    if len(complete_values) != EXPECTED_TP_RANKS:
        failures.append(
            f"expected {EXPECTED_TP_RANKS} encoder capture completions, "
            f"found {len(complete_values)}"
        )
    if any(value != len(ENCODER_GRAPH_BUDGETS) for value in complete_values):
        failures.append(
            f"encoder capture completion reported wrong graph count: {complete_values}"
        )

    # Per-budget capture messages include tensor-buffer details and are DEBUG
    # logs. The stable INFO contract is the exact initialized budget list plus
    # one completion count per TP rank. Cross-check DEBUG details when present,
    # but do not require a production server to run at DEBUG level.
    captured_budgets = [int(value) for value in _CAPTURED_RE.findall(text)]
    captured_by_budget = Counter(captured_budgets)
    capture_details_available = bool(captured_budgets)
    expected_capture_total = EXPECTED_TP_RANKS * len(ENCODER_GRAPH_BUDGETS)
    if capture_details_available:
        if len(captured_budgets) != expected_capture_total:
            failures.append(
                f"expected {expected_capture_total} detailed budget capture events, "
                f"found {len(captured_budgets)}"
            )
        unexpected_budgets = sorted(set(captured_budgets) - set(ENCODER_GRAPH_BUDGETS))
        if unexpected_budgets:
            failures.append(
                f"captured unexpected encoder budgets: {unexpected_budgets}"
            )
        for budget in ENCODER_GRAPH_BUDGETS:
            count = captured_by_budget[budget]
            if count != EXPECTED_TP_RANKS:
                failures.append(
                    f"encoder budget {budget} had {count} detailed capture events; "
                    f"expected {EXPECTED_TP_RANKS}"
                )

    installed = len(_INSTALLED_RE.findall(text))
    if installed != EXPECTED_TP_RANKS:
        failures.append(
            f"expected {EXPECTED_TP_RANKS} image_encoder installations, found {installed}"
        )
    return {
        "initialized": initialized,
        "capture_complete": len(complete_values),
        "installed": installed,
        "captured_total": sum(complete_values),
        "capture_details_available": capture_details_available,
        "capture_details_total": len(captured_budgets),
        "captures_by_budget": (
            {
                str(budget): captured_by_budget[budget]
                for budget in ENCODER_GRAPH_BUDGETS
            }
            if capture_details_available
            else None
        ),
        "failures": failures,
    }


def _chat_payload(
    model: str,
    content: list[dict[str, Any]],
    *,
    max_tokens: int,
    logprobs: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"thinking_mode": "disabled"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 0
    return payload


def build_requests(
    config: ActiveMMSmokeConfig,
    *,
    dog_prompt: str,
) -> list[RequestCase]:
    dog = make_data_url(config.dog)
    pug = make_data_url(config.pug)
    banner = make_data_url(config.banner)
    return [
        RequestCase(
            name="text",
            expected_text="text path ok",
            payload=_chat_payload(
                config.model,
                [{"type": "text", "text": "Reply with exactly: text path ok"}],
                max_tokens=16,
            ),
        ),
        # This must remain the first image request: it proves cold replay can
        # pack two different dynamic grids without a request-time recapture.
        RequestCase(
            name="two_image",
            expected_text="2",
            payload=_chat_payload(
                config.model,
                [
                    {"type": "image_url", "image_url": {"url": pug}},
                    {"type": "image_url", "image_url": {"url": banner}},
                    {
                        "type": "text",
                        "text": "How many images are attached? Reply with only the numeral.",
                    },
                ],
                max_tokens=4,
            ),
        ),
        RequestCase(
            name="banner",
            expected_text="TokenSpeed",
            payload=_chat_payload(
                config.model,
                [
                    {"type": "image_url", "image_url": {"url": banner}},
                    {
                        "type": "text",
                        "text": (
                            "What single word is displayed prominently? "
                            "Reply with exactly one word."
                        ),
                    },
                ],
                max_tokens=4,
            ),
        ),
        RequestCase(
            name="pug",
            expected_text="Pug",
            payload=_chat_payload(
                config.model,
                [
                    {"type": "image_url", "image_url": {"url": pug}},
                    {
                        "type": "text",
                        "text": (
                            "What breed of dog is shown? Reply with exactly one word."
                        ),
                    },
                ],
                max_tokens=4,
            ),
        ),
        RequestCase(
            name="dog",
            payload=_chat_payload(
                config.model,
                [
                    {"type": "image_url", "image_url": {"url": dog}},
                    {"type": "text", "text": dog_prompt},
                ],
                max_tokens=1,
                logprobs=True,
            ),
        ),
        RequestCase(
            name="video",
            payload=_chat_payload(
                config.model,
                [
                    {
                        "type": "video_url",
                        "video_url": {"url": "data:video/mp4;base64,AAAA"},
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
                max_tokens=4,
            ),
        ),
    ]


def _response_content(body: Any) -> Any:
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def validate_chat_response(
    case: RequestCase,
    *,
    status_code: int | None,
    headers: Mapping[str, str],
    body: Any,
    reference_logprob: float,
) -> dict[str, Any]:
    failures: list[str] = []
    if case.name == "video":
        header_code = headers.get("X-SMG-Error-Code") or headers.get("x-smg-error-code")
        try:
            error = body["error"]
            body_code = error["code"]
            message = error["message"]
        except (KeyError, TypeError):
            body_code = None
            message = None
        if status_code != 400:
            failures.append(f"video request returned HTTP {status_code}, expected 400")
        if header_code != VIDEO_ERROR_CODE:
            failures.append(f"video error header is {header_code!r}")
        if body_code != VIDEO_ERROR_CODE:
            failures.append(f"video error body code is {body_code!r}")
        normalized_message = message.lower() if isinstance(message, str) else ""
        if (
            "modality video" not in normalized_message
            or "not supported" not in normalized_message
        ):
            failures.append(f"video error message is not explicit: {message!r}")
        return {
            "status_code": status_code,
            "header_error_code": header_code,
            "body_error_code": body_code,
            "message": message,
            "failures": failures,
        }

    content = _response_content(body)
    if status_code != 200:
        failures.append(f"{case.name} request returned HTTP {status_code}")
    if case.name != "dog":
        if content != case.expected_text:
            failures.append(
                f"{case.name} expected {case.expected_text!r}, found {content!r}"
            )
        return {
            "status_code": status_code,
            "expected": case.expected_text,
            "actual": content,
            "failures": failures,
        }

    try:
        prompt_tokens = body["usage"]["prompt_tokens"]
        token_entry = body["choices"][0]["logprobs"]["content"][0]
        logprob_token = token_entry["token"]
        actual_logprob = float(token_entry["logprob"])
    except (KeyError, IndexError, TypeError, ValueError):
        prompt_tokens = None
        logprob_token = None
        actual_logprob = math.nan
    absolute_delta = abs(actual_logprob - reference_logprob)
    if content != DOG_EXPECTED_TOKEN:
        failures.append(f"dog content is {content!r}, expected {DOG_EXPECTED_TOKEN!r}")
    if logprob_token != DOG_EXPECTED_TOKEN:
        failures.append(
            f"dog logprob token is {logprob_token!r}, expected {DOG_EXPECTED_TOKEN!r}"
        )
    if prompt_tokens != DOG_EXPECTED_PROMPT_TOKENS:
        failures.append(
            f"dog prompt_tokens is {prompt_tokens!r}, expected {DOG_EXPECTED_PROMPT_TOKENS}"
        )
    if not math.isfinite(actual_logprob):
        failures.append("dog sampled logprob is not finite")
    elif absolute_delta > DOG_MAX_ABS_LOGPROB_DELTA:
        failures.append(
            f"dog logprob delta {absolute_delta:.12g} exceeds {DOG_MAX_ABS_LOGPROB_DELTA}"
        )
    return {
        "status_code": status_code,
        "expected_token": DOG_EXPECTED_TOKEN,
        "actual_content": content,
        "actual_logprob_token": logprob_token,
        "prompt_tokens": prompt_tokens,
        "reference_logprob": reference_logprob,
        "actual_logprob": actual_logprob,
        "absolute_delta": absolute_delta,
        "maximum_absolute_delta": DOG_MAX_ABS_LOGPROB_DELTA,
        "failures": failures,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure_result(config: ActiveMMSmokeConfig, failures: list[str]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "workload": "minimax_m3_active_mm",
        "ok": False,
        "request_order": [],
        "cases": {},
        "encoder_graph": {},
        "health": {},
        "provenance": {},
        "failures": failures[:MAX_RECORDED_FAILURES],
        "failure_count": len(failures),
    }
    _write_json_atomic(config.output_dir / "validation.json", result)
    return result


def run(
    config: ActiveMMSmokeConfig,
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    required_paths = {
        "dog": config.dog,
        "pug": config.pug,
        "banner": config.banner,
        "reference": config.reference,
        "server_log": config.server_log,
    }
    for label, path in required_paths.items():
        if not path.is_file():
            failures.append(f"required {label} file is missing: {path}")
    if failures:
        return _failure_result(config, failures)

    try:
        reference = json.loads(config.reference.read_text())
        dog_prompt = reference["prompt"]
        reference_token = reference["reference_logits"]["top_tokens"][0]
        reference_logprob = float(reference["reference_logits"]["top_logprobs"][0])
        reference_prompt_tokens = reference["input_contract"]["input_ids"][1]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        return _failure_result(
            config, [f"invalid visual reference: {type(exc).__name__}: {exc}"]
        )
    if reference_token != DOG_EXPECTED_TOKEN:
        failures.append(
            f"reference top token is {reference_token!r}, expected {DOG_EXPECTED_TOKEN!r}"
        )
    if not math.isfinite(reference_logprob):
        failures.append("reference logprob is not finite")
    if reference_prompt_tokens != DOG_EXPECTED_PROMPT_TOKENS:
        failures.append(
            f"reference prompt token count is {reference_prompt_tokens!r}, "
            f"expected {DOG_EXPECTED_PROMPT_TOKENS}"
        )

    before = summarize_encoder_graph_log(config.server_log.read_text(errors="replace"))
    failures.extend(f"before requests: {failure}" for failure in before.pop("failures"))
    try:
        request_cases = build_requests(config, dog_prompt=dog_prompt)
    except (OSError, ValueError) as exc:
        failures.append(f"failed to load image fixtures: {type(exc).__name__}: {exc}")
        return _failure_result(config, failures)

    owned_session = session is None
    http = requests.Session() if session is None else session
    case_results: dict[str, Any] = {}
    request_order: list[str] = []
    try:
        for case in request_cases:
            request_order.append(case.name)
            status_code: int | None = None
            response_headers: Mapping[str, str] = {}
            body: Any = None
            try:
                response = http.post(
                    f"{config.base_url.rstrip('/')}/v1/chat/completions",
                    json=case.payload,
                    timeout=config.request_timeout_seconds,
                )
                status_code = response.status_code
                response_headers = response.headers
                try:
                    body = response.json()
                except ValueError:
                    _append_failure(failures, f"{case.name} response is not valid JSON")
            except requests.RequestException as exc:
                _append_failure(
                    failures,
                    f"{case.name} request failed: {type(exc).__name__}",
                )
            response_artifact = {
                "status_code": status_code,
                "headers": {
                    "x-smg-error-code": response_headers.get("X-SMG-Error-Code")
                    or response_headers.get("x-smg-error-code")
                },
                "body": body,
            }
            _write_json_atomic(
                config.output_dir / f"{case.name}_response.json",
                response_artifact,
            )
            case_result = validate_chat_response(
                case,
                status_code=status_code,
                headers=response_headers,
                body=body,
                reference_logprob=reference_logprob,
            )
            failures.extend(
                f"{case.name}: {failure}" for failure in case_result.pop("failures")
            )
            case_results[case.name] = case_result

        health = probe_health(config.base_url, config.control_url, session=http)
        failures.extend(health.pop("failures"))
        _write_json_atomic(config.output_dir / "post_health.json", health)
    finally:
        if owned_session:
            http.close()

    # The live server log continues to grow during shutdown. Preserve the exact
    # bytes used for the request-time graph validation so provenance remains
    # stable after the final lifecycle messages are appended.
    server_log_snapshot = config.output_dir / "server_log_at_validation.log"
    _copy_file_atomic(config.server_log, server_log_snapshot)
    after = summarize_encoder_graph_log(server_log_snapshot.read_text(errors="replace"))
    failures.extend(f"after requests: {failure}" for failure in after.pop("failures"))
    counts_unchanged = before == after
    if not counts_unchanged:
        failures.append("encoder CUDA Graph capture counts changed during requests")

    provenance = {
        label: {"path": str(path), "sha256": _sha256(path)}
        for label, path in required_paths.items()
        if label != "server_log"
    }
    provenance["server_log"] = {
        "path": str(server_log_snapshot),
        "source_path": str(config.server_log),
        "sha256": _sha256(server_log_snapshot),
        "size_bytes": server_log_snapshot.stat().st_size,
        "snapshot_at_validation": True,
    }
    result = {
        "schema_version": 1,
        "workload": "minimax_m3_active_mm",
        "ok": not failures,
        "request_order": request_order,
        "cases": case_results,
        "encoder_graph": {
            "budgets": list(ENCODER_GRAPH_BUDGETS),
            "expected_tp_ranks": EXPECTED_TP_RANKS,
            "before": before,
            "after": after,
            "counts_unchanged": counts_unchanged,
        },
        "health": health,
        "provenance": provenance,
        "failures": failures[:MAX_RECORDED_FAILURES],
        "failure_count": len(failures),
    }
    _write_json_atomic(config.output_dir / "validation.json", result)
    print(
        f"Active-MM validation: {'PASS' if result['ok'] else 'FAIL'} "
        f"cases={len(case_results)}/6 graph_unchanged={counts_unchanged}",
        flush=True,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--pug", type=Path, required=True)
    parser.add_argument("--banner", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        not math.isfinite(args.request_timeout_seconds)
        or args.request_timeout_seconds <= 0
    ):
        raise SystemExit("request timeout must be positive")
    config = ActiveMMSmokeConfig(
        base_url=args.base_url,
        control_url=args.control_url,
        model=args.model,
        dog=args.dog,
        pug=args.pug,
        banner=args.banner,
        reference=args.reference,
        server_log=args.server_log,
        output_dir=args.output_dir,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    return 0 if run(config)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
