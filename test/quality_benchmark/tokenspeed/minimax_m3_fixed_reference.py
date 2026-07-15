#!/usr/bin/env python3
"""Collect and compare MiniMax-M3 BF16/FP8 fixed-reference quality evidence.

The client intentionally does not launch, reconfigure, flush, or stop a server.
In particular, it leaves the release-default overlap scheduler untouched.  Start
each server arm separately with output logprobs enabled, collect one JSON artifact
per arm, and then compare the two artifacts with the provisional Phase 5 gates.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

SCHEMA_VERSION = 1
BENCHMARK_NAME = "minimax_m3_fixed_reference_bf16_fp8"
EXPECTED_PROMPTS = 5
EXPECTED_STEPS_PER_PROMPT = 8
EXPECTED_TEACHER_CONTEXTS = EXPECTED_PROMPTS * EXPECTED_STEPS_PER_PROMPT
EXPECTED_AUTOREGRESSIVE_REPEATS = 3
ALLOWED_SERVER_ARG_DIFFERENCES = frozenset({"kv_cache_dtype"})


class BenchmarkError(RuntimeError):
    """Raised when a reference, response, or artifact violates the contract."""


@dataclass(frozen=True)
class ReferenceStep:
    token_id: int
    top_ids: tuple[int, ...]
    top_logprobs: tuple[float, ...]

    @property
    def top_margin(self) -> float:
        """Return the HF top-1 versus top-2 logprob margin."""
        if len(self.top_logprobs) < 2:
            raise BenchmarkError("Reference step needs at least two top logprobs")
        return self.top_logprobs[0] - self.top_logprobs[1]


@dataclass(frozen=True)
class ReferencePrompt:
    prompt: str
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    steps: tuple[ReferenceStep, ...]


@dataclass(frozen=True)
class ReferenceSuite:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    prompts: tuple[ReferencePrompt, ...]


@dataclass(frozen=True)
class CollectConfig:
    arm: str
    model: str
    reference_path: Path
    output_path: Path
    base_url: str
    generate_path: str
    server_info_path: str
    server_sha: str
    request_timeout_seconds: float
    server_info_timeout_seconds: float
    autoregressive_repeats: int = 3
    seed: int = 20260715

    def validate(self) -> None:
        if not self.arm.strip():
            raise BenchmarkError("--arm must not be empty")
        if not self.model.strip():
            raise BenchmarkError("--model must not be empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise BenchmarkError("--base-url must start with http:// or https://")
        if not self.server_sha.strip():
            raise BenchmarkError("--server-sha must not be empty")
        if self.request_timeout_seconds <= 0:
            raise BenchmarkError("--request-timeout-seconds must be positive")
        if self.server_info_timeout_seconds <= 0:
            raise BenchmarkError("--server-info-timeout-seconds must be positive")
        if self.autoregressive_repeats != EXPECTED_AUTOREGRESSIVE_REPEATS:
            raise BenchmarkError(
                "--autoregressive-repeats must be 3 for the fixed quality gate"
            )


@dataclass(frozen=True)
class QualityGates:
    """Provisional Phase 5 BF16-versus-FP8 acceptance gates."""

    intrarm_logprob_atol: float = 1e-6
    teacher_match_min: int = 39
    hf_top5_min_per_arm: int = 40
    mismatch_margin_max: float = 0.25
    matched_logprob_mean_max: float = 0.10
    matched_logprob_p95_max: float = 0.25
    matched_logprob_max: float = 0.50
    autoregressive_exact_min: int = 4
    common_prefix_tokens_min: int = 33
    require_no_step0_divergence: bool = True


class HttpClient(Protocol):
    def get_json(self, url: str, timeout_seconds: float) -> Any: ...

    def post_json(
        self, url: str, body: dict[str, Any], timeout_seconds: float
    ) -> Any: ...


class UrlLibJsonClient:
    """Small dependency-free JSON HTTP client used by the benchmark CLI."""

    @staticmethod
    def _request(request: urllib.request.Request, timeout_seconds: float) -> Any:
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BenchmarkError(
                f"HTTP {exc.code} from {request.full_url}: {body[:2000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BenchmarkError(
                f"Request failed for {request.full_url}: {exc}"
            ) from exc
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            preview = payload.decode("utf-8", errors="replace")[:2000]
            raise BenchmarkError(
                f"Non-JSON response from {request.full_url}: {preview}"
            ) from exc

    def get_json(self, url: str, timeout_seconds: float) -> Any:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        return self._request(request, timeout_seconds)

    def post_json(self, url: str, body: dict[str, Any], timeout_seconds: float) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(body, separators=(",", ":")).encode(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._request(request, timeout_seconds)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be a JSON object")
    return value


def _require_int_list(
    value: Any, label: str, *, allow_empty: bool = False
) -> list[int]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "" if allow_empty else " non-empty"
        raise BenchmarkError(f"{label} must be a{qualifier} list of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise BenchmarkError(f"{label} must contain only integers")
    return value


def _require_float_list(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise BenchmarkError(f"{label} must be a non-empty list of numbers")
    converted: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BenchmarkError(f"{label} must contain only numbers")
        item_float = float(item)
        if not math.isfinite(item_float):
            raise BenchmarkError(f"{label} contains a non-finite value")
        converted.append(item_float)
    return converted


def load_reference(path: Path) -> ReferenceSuite:
    """Load and strictly validate the fixed 5-by-8 HF reference artifact."""
    resolved = path.expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"Cannot read reference {resolved}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid reference JSON {resolved}: {exc}") from exc
    root = _require_mapping(raw, "reference")
    raw_results = root.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != EXPECTED_PROMPTS:
        raise BenchmarkError(
            f"reference.results must contain exactly {EXPECTED_PROMPTS} prompts"
        )

    prompts: list[ReferencePrompt] = []
    for prompt_index, raw_prompt in enumerate(raw_results):
        prompt_data = _require_mapping(raw_prompt, f"results[{prompt_index}]")
        prompt = prompt_data.get("prompt")
        if not isinstance(prompt, str):
            raise BenchmarkError(f"results[{prompt_index}].prompt must be a string")
        input_ids = _require_int_list(
            prompt_data.get("input_ids"), f"results[{prompt_index}].input_ids"
        )
        generated_ids = _require_int_list(
            prompt_data.get("generated_ids"),
            f"results[{prompt_index}].generated_ids",
        )
        raw_steps = prompt_data.get("steps")
        if not isinstance(raw_steps, list):
            raise BenchmarkError(f"results[{prompt_index}].steps must be a list")
        if len(raw_steps) != EXPECTED_STEPS_PER_PROMPT:
            raise BenchmarkError(
                f"results[{prompt_index}].steps must contain exactly "
                f"{EXPECTED_STEPS_PER_PROMPT} steps"
            )
        if len(generated_ids) != len(raw_steps):
            raise BenchmarkError(
                f"results[{prompt_index}] generated_ids/steps length mismatch"
            )

        steps: list[ReferenceStep] = []
        for step_index, raw_step in enumerate(raw_steps):
            step_data = _require_mapping(
                raw_step, f"results[{prompt_index}].steps[{step_index}]"
            )
            token_id = step_data.get("token_id")
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise BenchmarkError(
                    f"results[{prompt_index}].steps[{step_index}].token_id "
                    "must be an integer"
                )
            if token_id != generated_ids[step_index]:
                raise BenchmarkError(
                    f"results[{prompt_index}].steps[{step_index}] token_id does "
                    "not match generated_ids"
                )
            top_ids = _require_int_list(
                step_data.get("top_ids"),
                f"results[{prompt_index}].steps[{step_index}].top_ids",
            )
            top_logprobs = _require_float_list(
                step_data.get("top_logprobs"),
                f"results[{prompt_index}].steps[{step_index}].top_logprobs",
            )
            if len(top_ids) != len(top_logprobs):
                raise BenchmarkError(
                    f"results[{prompt_index}].steps[{step_index}] top IDs/logprobs "
                    "length mismatch"
                )
            if len(top_ids) < 5:
                raise BenchmarkError(
                    f"results[{prompt_index}].steps[{step_index}] needs five top IDs"
                )
            steps.append(
                ReferenceStep(
                    token_id=token_id,
                    top_ids=tuple(top_ids),
                    top_logprobs=tuple(top_logprobs),
                )
            )
        prompts.append(
            ReferencePrompt(
                prompt=prompt,
                input_ids=tuple(input_ids),
                generated_ids=tuple(generated_ids),
                steps=tuple(steps),
            )
        )

    metadata = {key: value for key, value in root.items() if key != "results"}
    return ReferenceSuite(
        path=resolved,
        sha256=_sha256(resolved),
        metadata=metadata,
        prompts=tuple(prompts),
    )


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _normalize_generation_object(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, list):
        if len(raw_response) != 1:
            raise BenchmarkError(
                "Expected a generation object or a singleton generation list"
            )
        raw_response = raw_response[0]
    return _require_mapping(raw_response, "generation response")


def _parse_sampled_logprobs(
    raw_logprobs: Any, output_ids: Sequence[int]
) -> list[dict[str, float | int]]:
    if not isinstance(raw_logprobs, list):
        raise BenchmarkError(
            "meta_info.output_token_logprobs is missing; start the server with "
            "--enable-output-logprobs"
        )
    if len(raw_logprobs) != len(output_ids):
        raise BenchmarkError(
            "output IDs/logprobs length mismatch: "
            f"{len(output_ids)} IDs versus {len(raw_logprobs)} logprobs"
        )

    parsed: list[dict[str, float | int]] = []
    for position, entry in enumerate(raw_logprobs):
        if isinstance(entry, list) and len(entry) >= 2:
            logprob, token_id = entry[0], entry[1]
        elif isinstance(entry, dict):
            logprob, token_id = entry.get("logprob"), entry.get("token_id")
        else:
            raise BenchmarkError(
                f"output logprob at position {position} has an unsupported shape"
            )
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise BenchmarkError(
                f"output logprob token ID at position {position} is not an integer"
            )
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise BenchmarkError(
                f"output logprob at position {position} is not numeric"
            )
        value = float(logprob)
        if not math.isfinite(value):
            raise BenchmarkError(f"output logprob at position {position} is non-finite")
        if token_id != output_ids[position]:
            raise BenchmarkError(
                f"output ID/logprob token mismatch at position {position}: "
                f"{output_ids[position]} versus {token_id}"
            )
        parsed.append({"token_id": token_id, "logprob": value})
    return parsed


def parse_generation_response(raw_response: Any) -> dict[str, Any]:
    """Normalize one /generate response while retaining the unmodified body."""
    response = _normalize_generation_object(raw_response)
    output_ids = _require_int_list(response.get("output_ids"), "response.output_ids")
    meta_info = _require_mapping(response.get("meta_info"), "response.meta_info")
    raw_logprobs = meta_info.get("output_token_logprobs")
    sampled_logprobs = _parse_sampled_logprobs(raw_logprobs, output_ids)
    return {
        "output_ids": output_ids,
        "sampled_logprobs": sampled_logprobs,
        "output_token_logprobs_raw": raw_logprobs,
        "raw_body": raw_response,
    }


def _extract_server_args(raw_info: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    info = _require_mapping(raw_info, "get_server_info response")
    server_args = _require_mapping(info.get("server_args"), "server_info.server_args")
    if server_args.get("enable_output_logprobs") is not True:
        raise BenchmarkError(
            "Server must report enable_output_logprobs=true before quality collection"
        )
    return info, server_args


def _request_body(
    model: str, input_ids: Sequence[int], max_new_tokens: int, seed: int
) -> dict[str, Any]:
    return {
        "model": model,
        "input_ids": list(input_ids),
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "sampling_params": {
            "temperature": 0,
            "top_k": 1,
            "ignore_eos": True,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
    }


def collect_arm(config: CollectConfig, http_client: HttpClient | None = None) -> dict:
    """Collect one arm and checkpoint a durable JSON artifact after every request."""
    config.validate()
    reference = load_reference(config.reference_path)
    client = http_client or UrlLibJsonClient()
    info_url = _join_url(config.base_url, config.server_info_path)
    generate_url = _join_url(config.base_url, config.generate_path)
    raw_info = client.get_json(info_url, config.server_info_timeout_seconds)
    server_info, server_args = _extract_server_args(raw_info)

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "collecting",
        "arm": config.arm,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "reference": {
            "path": str(reference.path),
            "sha256": reference.sha256,
            "metadata": reference.metadata,
            "prompt_count": len(reference.prompts),
            "teacher_context_count": sum(
                len(prompt.steps) for prompt in reference.prompts
            ),
        },
        "client_config": {
            "base_url": config.base_url,
            "generate_url": generate_url,
            "server_info_url": info_url,
            "request_timeout_seconds": config.request_timeout_seconds,
            "server_info_timeout_seconds": config.server_info_timeout_seconds,
            "autoregressive_repeats": config.autoregressive_repeats,
            "model": config.model,
            "temperature": 0,
            "top_k": 1,
            "ignore_eos": True,
            "seed": config.seed,
        },
        "server": {
            "sha": config.server_sha,
            "args": server_args,
            "raw_info": server_info,
        },
        "autoregressive": [],
        "teacher_forced": [],
    }
    _atomic_write_json(config.output_path, artifact)

    try:
        for prompt_index, prompt in enumerate(reference.prompts):
            for repeat_index in range(config.autoregressive_repeats):
                request_body = _request_body(
                    config.model,
                    prompt.input_ids,
                    max_new_tokens=len(prompt.generated_ids),
                    seed=config.seed,
                )
                raw_response = client.post_json(
                    generate_url, request_body, config.request_timeout_seconds
                )
                artifact["autoregressive"].append(
                    {
                        "key": f"prompt-{prompt_index}/repeat-{repeat_index}",
                        "prompt_index": prompt_index,
                        "repeat_index": repeat_index,
                        "prompt": prompt.prompt,
                        "reference_input_ids": list(prompt.input_ids),
                        "reference_generated_ids": list(prompt.generated_ids),
                        "request_body": request_body,
                        "response": parse_generation_response(raw_response),
                    }
                )
                artifact["updated_at_utc"] = _utc_now()
                _atomic_write_json(config.output_path, artifact)

        for prompt_index, prompt in enumerate(reference.prompts):
            for step_index, step in enumerate(prompt.steps):
                context_ids = (*prompt.input_ids, *prompt.generated_ids[:step_index])
                request_body = _request_body(
                    config.model, context_ids, max_new_tokens=1, seed=config.seed
                )
                raw_response = client.post_json(
                    generate_url, request_body, config.request_timeout_seconds
                )
                artifact["teacher_forced"].append(
                    {
                        "key": f"prompt-{prompt_index}/step-{step_index}",
                        "prompt_index": prompt_index,
                        "step_index": step_index,
                        "prompt": prompt.prompt,
                        "context_input_ids": list(context_ids),
                        "reference": {
                            "token_id": step.token_id,
                            "top5_ids": list(step.top_ids[:5]),
                            "top5_logprobs": list(step.top_logprobs[:5]),
                            "top1_top2_margin": step.top_margin,
                        },
                        "request_body": request_body,
                        "response": parse_generation_response(raw_response),
                    }
                )
                artifact["updated_at_utc"] = _utc_now()
                _atomic_write_json(config.output_path, artifact)
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["updated_at_utc"] = _utc_now()
        _atomic_write_json(config.output_path, artifact)
        raise

    artifact["status"] = "complete"
    artifact["updated_at_utc"] = _utc_now()
    _atomic_write_json(config.output_path, artifact)
    return artifact


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"Cannot read arm artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid arm artifact JSON {path}: {exc}") from exc
    artifact = _require_mapping(raw, f"arm artifact {path}")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError(f"Unsupported schema_version in {path}")
    if artifact.get("benchmark") != BENCHMARK_NAME:
        raise BenchmarkError(f"Unexpected benchmark name in {path}")
    if artifact.get("status") != "complete":
        raise BenchmarkError(f"Arm artifact is not complete: {path}")
    _require_mapping(artifact.get("server"), f"{path}.server")
    return artifact


def _validate_arm_identity(
    artifact: Mapping[str, Any], expected_arm: str, path: Path
) -> tuple[str, str]:
    arm = artifact.get("arm")
    if not isinstance(arm, str) or arm.casefold() != expected_arm:
        raise BenchmarkError(
            f"{path} must identify itself as the {expected_arm.upper()} arm"
        )
    server = _require_mapping(artifact.get("server"), f"{path}.server")
    server_sha = server.get("sha")
    if not isinstance(server_sha, str) or not server_sha.strip():
        raise BenchmarkError(f"{path}.server.sha must be a non-empty string")
    server_args = _require_mapping(server.get("args"), f"{path}.server.args")
    kv_cache_dtype = server_args.get("kv_cache_dtype")
    accepted_dtypes = (
        {"auto", "bf16", "bfloat16"} if expected_arm == "bf16" else {"fp8", "fp8_e4m3"}
    )
    if not isinstance(kv_cache_dtype, str) or kv_cache_dtype.casefold() not in (
        accepted_dtypes
    ):
        raise BenchmarkError(
            f"{path} has kv_cache_dtype={kv_cache_dtype!r}; expected "
            f"the {expected_arm.upper()} cache arm"
        )
    return server_sha, kv_cache_dtype


def _validate_server_args_equivalent(
    bf16: Mapping[str, Any], fp8: Mapping[str, Any]
) -> None:
    bf16_server = _require_mapping(bf16.get("server"), "BF16 server")
    fp8_server = _require_mapping(fp8.get("server"), "FP8 server")
    bf16_args = _require_mapping(bf16_server.get("args"), "BF16 server.args")
    fp8_args = _require_mapping(fp8_server.get("args"), "FP8 server.args")
    compared_keys = (set(bf16_args) | set(fp8_args)) - ALLOWED_SERVER_ARG_DIFFERENCES
    drifted = sorted(
        key for key in compared_keys if bf16_args.get(key) != fp8_args.get(key)
    )
    if drifted:
        raise BenchmarkError(
            "BF16/FP8 server args drift outside the explicit "
            f"{sorted(ALLOWED_SERVER_ARG_DIFFERENCES)} whitelist: {drifted}"
        )


def _sample_response(sample: Mapping[str, Any], label: str) -> dict[str, Any]:
    response = _require_mapping(sample.get("response"), f"{label}.response")
    _require_int_list(response.get("output_ids"), f"{label}.response.output_ids")
    logprobs = response.get("sampled_logprobs")
    if not isinstance(logprobs, list):
        raise BenchmarkError(f"{label}.response.sampled_logprobs must be a list")
    if len(logprobs) != len(response["output_ids"]):
        raise BenchmarkError(f"{label} response IDs/logprobs length mismatch")
    for position, raw_entry in enumerate(logprobs):
        entry = _require_mapping(raw_entry, f"{label}.sampled_logprobs[{position}]")
        token_id = entry.get("token_id")
        logprob = entry.get("logprob")
        if token_id != response["output_ids"][position]:
            raise BenchmarkError(f"{label} sampled logprob token mismatch")
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise BenchmarkError(f"{label} sampled logprob must be numeric")
        if not math.isfinite(float(logprob)):
            raise BenchmarkError(f"{label} sampled logprob must be finite")
    return response


def _group_autoregressive(artifact: Mapping[str, Any]) -> dict[int, list[dict]]:
    raw_samples = artifact.get("autoregressive")
    if not isinstance(raw_samples, list):
        raise BenchmarkError("artifact.autoregressive must be a list")
    grouped: dict[int, list[dict]] = defaultdict(list)
    seen_keys: set[tuple[int, int]] = set()
    for sample_index, raw_sample in enumerate(raw_samples):
        sample = _require_mapping(raw_sample, f"autoregressive[{sample_index}]")
        prompt_index = sample.get("prompt_index")
        repeat_index = sample.get("repeat_index")
        if not isinstance(prompt_index, int) or not isinstance(repeat_index, int):
            raise BenchmarkError("autoregressive indices must be integers")
        key = (prompt_index, repeat_index)
        if key in seen_keys:
            raise BenchmarkError(f"duplicate autoregressive key {key}")
        seen_keys.add(key)
        _sample_response(sample, f"autoregressive[{sample_index}]")
        grouped[prompt_index].append(sample)
    if set(grouped) != set(range(EXPECTED_PROMPTS)):
        raise BenchmarkError("autoregressive samples must cover prompt indices 0..4")
    for prompt_index, samples in grouped.items():
        samples.sort(key=lambda sample: sample["repeat_index"])
        expected_repeats = list(range(len(samples)))
        actual_repeats = [sample["repeat_index"] for sample in samples]
        if (
            actual_repeats != expected_repeats
            or len(samples) != EXPECTED_AUTOREGRESSIVE_REPEATS
        ):
            raise BenchmarkError(
                f"prompt {prompt_index} repeats must be contiguous from zero"
            )
    return grouped


def _teacher_by_key(artifact: Mapping[str, Any]) -> dict[tuple[int, int], dict]:
    raw_samples = artifact.get("teacher_forced")
    if not isinstance(raw_samples, list):
        raise BenchmarkError("artifact.teacher_forced must be a list")
    samples: dict[tuple[int, int], dict] = {}
    for sample_index, raw_sample in enumerate(raw_samples):
        sample = _require_mapping(raw_sample, f"teacher_forced[{sample_index}]")
        prompt_index = sample.get("prompt_index")
        step_index = sample.get("step_index")
        if not isinstance(prompt_index, int) or not isinstance(step_index, int):
            raise BenchmarkError("teacher-forced indices must be integers")
        key = (prompt_index, step_index)
        if key in samples:
            raise BenchmarkError(f"duplicate teacher-forced key {key}")
        response = _sample_response(sample, f"teacher_forced[{sample_index}]")
        if len(response["output_ids"]) != 1:
            raise BenchmarkError(f"teacher-forced sample {key} must emit one token")
        reference = _require_mapping(sample.get("reference"), f"teacher sample {key}")
        _require_int_list(reference.get("top5_ids"), f"teacher sample {key}.top5_ids")
        margin = reference.get("top1_top2_margin")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise BenchmarkError(f"teacher sample {key} margin must be numeric")
        samples[key] = sample
    expected = {
        (prompt_index, step_index)
        for prompt_index in range(EXPECTED_PROMPTS)
        for step_index in range(EXPECTED_STEPS_PER_PROMPT)
    }
    if set(samples) != expected:
        raise BenchmarkError("teacher-forced samples must cover the fixed 5-by-8 grid")
    return samples


def _logprob_values(response: Mapping[str, Any]) -> list[float]:
    return [float(entry["logprob"]) for entry in response["sampled_logprobs"]]


def _intrarm_determinism(
    grouped: Mapping[int, Sequence[Mapping[str, Any]]], atol: float
) -> dict[str, Any]:
    id_mismatches: list[str] = []
    logprob_mismatches: list[str] = []
    for prompt_index, samples in sorted(grouped.items()):
        canonical = _sample_response(samples[0], f"prompt {prompt_index} repeat 0")
        canonical_ids = canonical["output_ids"]
        canonical_logprobs = _logprob_values(canonical)
        for sample in samples[1:]:
            repeat_index = sample["repeat_index"]
            response = _sample_response(
                sample, f"prompt {prompt_index} repeat {repeat_index}"
            )
            if response["output_ids"] != canonical_ids:
                id_mismatches.append(f"prompt-{prompt_index}/repeat-{repeat_index}")
            values = _logprob_values(response)
            if len(values) != len(canonical_logprobs) or any(
                abs(left - right) > atol
                for left, right in zip(values, canonical_logprobs)
            ):
                logprob_mismatches.append(
                    f"prompt-{prompt_index}/repeat-{repeat_index}"
                )
    return {
        "ids_deterministic": not id_mismatches,
        "logprobs_deterministic": not logprob_mismatches,
        "id_mismatches": id_mismatches,
        "logprob_mismatches": logprob_mismatches,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _check(name: str, passed: bool, actual: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "actual": actual,
        "requirement": requirement,
    }


def compare_arms(
    bf16_path: Path,
    fp8_path: Path,
    output_path: Path,
    gates: QualityGates | None = None,
) -> dict[str, Any]:
    """Compare two complete arm artifacts and write the gate report."""
    gates = gates or QualityGates()
    bf16_path = bf16_path.expanduser().resolve()
    fp8_path = fp8_path.expanduser().resolve()
    bf16 = _read_artifact(bf16_path)
    fp8 = _read_artifact(fp8_path)
    bf16_server_sha, bf16_cache_dtype = _validate_arm_identity(bf16, "bf16", bf16_path)
    fp8_server_sha, fp8_cache_dtype = _validate_arm_identity(fp8, "fp8", fp8_path)
    if bf16_server_sha != fp8_server_sha:
        raise BenchmarkError(
            "BF16 and FP8 artifacts must come from the same server SHA: "
            f"{bf16_server_sha} versus {fp8_server_sha}"
        )
    _validate_server_args_equivalent(bf16, fp8)
    bf16_reference = _require_mapping(bf16.get("reference"), "BF16 reference")
    fp8_reference = _require_mapping(fp8.get("reference"), "FP8 reference")
    if bf16_reference.get("sha256") != fp8_reference.get("sha256"):
        raise BenchmarkError("BF16 and FP8 artifacts use different references")
    if bf16_reference.get("sha256") is None:
        raise BenchmarkError("Arm artifacts do not contain a reference SHA256")

    bf16_auto = _group_autoregressive(bf16)
    fp8_auto = _group_autoregressive(fp8)
    if any(len(bf16_auto[index]) != len(fp8_auto[index]) for index in bf16_auto):
        raise BenchmarkError("BF16 and FP8 autoregressive repeat counts differ")
    bf16_determinism = _intrarm_determinism(bf16_auto, gates.intrarm_logprob_atol)
    fp8_determinism = _intrarm_determinism(fp8_auto, gates.intrarm_logprob_atol)

    bf16_teacher = _teacher_by_key(bf16)
    fp8_teacher = _teacher_by_key(fp8)
    teacher_matches = 0
    bf16_hf_top5 = 0
    fp8_hf_top5 = 0
    matched_logprob_deltas: list[float] = []
    teacher_mismatches: list[dict[str, Any]] = []
    for key in sorted(bf16_teacher):
        bf16_sample = bf16_teacher[key]
        fp8_sample = fp8_teacher[key]
        bf16_reference_data = _require_mapping(
            bf16_sample["reference"], f"BF16 teacher {key}.reference"
        )
        fp8_reference_data = _require_mapping(
            fp8_sample["reference"], f"FP8 teacher {key}.reference"
        )
        if bf16_reference_data != fp8_reference_data:
            raise BenchmarkError(f"Reference metadata differs at teacher context {key}")
        top5_ids = bf16_reference_data["top5_ids"]
        margin = float(bf16_reference_data["top1_top2_margin"])
        bf16_response = bf16_sample["response"]
        fp8_response = fp8_sample["response"]
        bf16_id = bf16_response["output_ids"][0]
        fp8_id = fp8_response["output_ids"][0]
        bf16_hf_top5 += int(bf16_id in top5_ids)
        fp8_hf_top5 += int(fp8_id in top5_ids)
        if bf16_id == fp8_id:
            teacher_matches += 1
            matched_logprob_deltas.append(
                abs(
                    float(bf16_response["sampled_logprobs"][0]["logprob"])
                    - float(fp8_response["sampled_logprobs"][0]["logprob"])
                )
            )
        else:
            teacher_mismatches.append(
                {
                    "prompt_index": key[0],
                    "step_index": key[1],
                    "bf16_token_id": bf16_id,
                    "fp8_token_id": fp8_id,
                    "hf_top5_ids": top5_ids,
                    "hf_top1_top2_margin": margin,
                }
            )

    mean_delta = fmean(matched_logprob_deltas) if matched_logprob_deltas else None
    p95_delta = (
        _percentile(matched_logprob_deltas, 0.95) if matched_logprob_deltas else None
    )
    max_delta = max(matched_logprob_deltas) if matched_logprob_deltas else None
    mismatch_margins_ok = all(
        mismatch["hf_top1_top2_margin"] <= gates.mismatch_margin_max
        for mismatch in teacher_mismatches
    )

    autoregressive_exact = 0
    common_prefix_tokens = 0
    step0_divergences: list[int] = []
    autoregressive_details: list[dict[str, Any]] = []
    for prompt_index in range(EXPECTED_PROMPTS):
        bf16_ids = bf16_auto[prompt_index][0]["response"]["output_ids"]
        fp8_ids = fp8_auto[prompt_index][0]["response"]["output_ids"]
        exact = bf16_ids == fp8_ids
        prefix = _common_prefix_length(bf16_ids, fp8_ids)
        step0_same = bool(bf16_ids and fp8_ids and bf16_ids[0] == fp8_ids[0])
        autoregressive_exact += int(exact)
        common_prefix_tokens += prefix
        if not step0_same:
            step0_divergences.append(prompt_index)
        autoregressive_details.append(
            {
                "prompt_index": prompt_index,
                "exact": exact,
                "common_prefix_tokens": prefix,
                "step0_same": step0_same,
                "bf16_output_ids": bf16_ids,
                "fp8_output_ids": fp8_ids,
            }
        )

    checks = [
        _check(
            "bf16_intrarm_ids_deterministic",
            bf16_determinism["ids_deterministic"],
            bf16_determinism["id_mismatches"],
            "no mismatches across repeats",
        ),
        _check(
            "bf16_intrarm_logprobs_deterministic",
            bf16_determinism["logprobs_deterministic"],
            bf16_determinism["logprob_mismatches"],
            f"absolute delta <= {gates.intrarm_logprob_atol}",
        ),
        _check(
            "fp8_intrarm_ids_deterministic",
            fp8_determinism["ids_deterministic"],
            fp8_determinism["id_mismatches"],
            "no mismatches across repeats",
        ),
        _check(
            "fp8_intrarm_logprobs_deterministic",
            fp8_determinism["logprobs_deterministic"],
            fp8_determinism["logprob_mismatches"],
            f"absolute delta <= {gates.intrarm_logprob_atol}",
        ),
        _check(
            "teacher_bf16_fp8_token_matches",
            teacher_matches >= gates.teacher_match_min,
            teacher_matches,
            f">= {gates.teacher_match_min}/{EXPECTED_TEACHER_CONTEXTS}",
        ),
        _check(
            "bf16_teacher_tokens_in_hf_top5",
            bf16_hf_top5 >= gates.hf_top5_min_per_arm,
            bf16_hf_top5,
            f">= {gates.hf_top5_min_per_arm}/{EXPECTED_TEACHER_CONTEXTS}",
        ),
        _check(
            "fp8_teacher_tokens_in_hf_top5",
            fp8_hf_top5 >= gates.hf_top5_min_per_arm,
            fp8_hf_top5,
            f">= {gates.hf_top5_min_per_arm}/{EXPECTED_TEACHER_CONTEXTS}",
        ),
        _check(
            "teacher_mismatches_have_small_hf_margin",
            mismatch_margins_ok,
            [mismatch["hf_top1_top2_margin"] for mismatch in teacher_mismatches],
            f"every margin <= {gates.mismatch_margin_max}",
        ),
        _check(
            "matched_teacher_logprob_mean_delta",
            mean_delta is not None and mean_delta <= gates.matched_logprob_mean_max,
            mean_delta,
            f"<= {gates.matched_logprob_mean_max}",
        ),
        _check(
            "matched_teacher_logprob_p95_delta",
            p95_delta is not None and p95_delta <= gates.matched_logprob_p95_max,
            p95_delta,
            f"<= {gates.matched_logprob_p95_max}",
        ),
        _check(
            "matched_teacher_logprob_max_delta",
            max_delta is not None and max_delta <= gates.matched_logprob_max,
            max_delta,
            f"<= {gates.matched_logprob_max}",
        ),
        _check(
            "autoregressive_exact_prompts",
            autoregressive_exact >= gates.autoregressive_exact_min,
            autoregressive_exact,
            f">= {gates.autoregressive_exact_min}/{EXPECTED_PROMPTS}",
        ),
        _check(
            "autoregressive_common_prefix_tokens",
            common_prefix_tokens >= gates.common_prefix_tokens_min,
            common_prefix_tokens,
            f">= {gates.common_prefix_tokens_min}/{EXPECTED_TEACHER_CONTEXTS}",
        ),
        _check(
            "autoregressive_no_step0_divergence",
            not step0_divergences,
            step0_divergences,
            "no prompt diverges at step 0",
        ),
    ]

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "report_type": "bf16_fp8_comparison",
        "created_at_utc": _utc_now(),
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "sources": {
            "bf16": {
                "path": str(bf16_path),
                "sha256": _sha256(bf16_path),
                "arm": bf16.get("arm"),
                "server_sha": bf16_server_sha,
                "kv_cache_dtype": bf16_cache_dtype,
            },
            "fp8": {
                "path": str(fp8_path),
                "sha256": _sha256(fp8_path),
                "arm": fp8.get("arm"),
                "server_sha": fp8_server_sha,
                "kv_cache_dtype": fp8_cache_dtype,
            },
            "reference_sha256": bf16_reference["sha256"],
        },
        "gates": dataclasses.asdict(gates),
        "metrics": {
            "teacher_matches": teacher_matches,
            "teacher_contexts": EXPECTED_TEACHER_CONTEXTS,
            "bf16_hf_top5": bf16_hf_top5,
            "fp8_hf_top5": fp8_hf_top5,
            "matched_teacher_logprob_delta": {
                "count": len(matched_logprob_deltas),
                "mean": mean_delta,
                "p95_linear_interpolation": p95_delta,
                "max": max_delta,
            },
            "autoregressive_exact_prompts": autoregressive_exact,
            "autoregressive_prompt_count": EXPECTED_PROMPTS,
            "autoregressive_common_prefix_tokens": common_prefix_tokens,
            "autoregressive_reference_token_count": EXPECTED_TEACHER_CONTEXTS,
            "step0_divergences": step0_divergences,
        },
        "checks": checks,
        "details": {
            "bf16_intrarm": bf16_determinism,
            "fp8_intrarm": fp8_determinism,
            "teacher_mismatches": teacher_mismatches,
            "autoregressive": autoregressive_details,
        },
    }
    _atomic_write_json(output_path, report)
    return report


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _exactly_three(value: str) -> int:
    parsed = int(value)
    if parsed != EXPECTED_AUTOREGRESSIVE_REPEATS:
        raise argparse.ArgumentTypeError("must be 3 for the fixed quality gate")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="collect one server arm into a durable JSON artifact"
    )
    collect_parser.add_argument("--arm", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--reference", required=True, type=Path)
    collect_parser.add_argument("--output", required=True, type=Path)
    collect_parser.add_argument("--base-url", required=True)
    collect_parser.add_argument("--generate-path", default="/generate")
    collect_parser.add_argument("--server-info-path", default="/get_server_info")
    collect_parser.add_argument("--server-sha", required=True)
    collect_parser.add_argument("--seed", type=int, default=20260715)
    collect_parser.add_argument(
        "--request-timeout-seconds", type=_positive_float, default=600.0
    )
    collect_parser.add_argument(
        "--server-info-timeout-seconds", type=_positive_float, default=30.0
    )
    collect_parser.add_argument(
        "--autoregressive-repeats", type=_exactly_three, default=3
    )

    compare_parser = subparsers.add_parser(
        "compare", help="apply the fixed provisional gates to BF16 and FP8 arms"
    )
    compare_parser.add_argument("--bf16-arm", required=True, type=Path)
    compare_parser.add_argument("--fp8-arm", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            config = CollectConfig(
                arm=args.arm,
                model=args.model,
                reference_path=args.reference,
                output_path=args.output,
                base_url=args.base_url,
                generate_path=args.generate_path,
                server_info_path=args.server_info_path,
                server_sha=args.server_sha,
                request_timeout_seconds=args.request_timeout_seconds,
                server_info_timeout_seconds=args.server_info_timeout_seconds,
                autoregressive_repeats=args.autoregressive_repeats,
                seed=args.seed,
            )
            artifact = collect_arm(config)
            summary = {
                "status": artifact["status"],
                "arm": artifact["arm"],
                "output": str(config.output_path.resolve()),
                "autoregressive_requests": len(artifact["autoregressive"]),
                "teacher_forced_requests": len(artifact["teacher_forced"]),
                "reference_sha256": artifact["reference"]["sha256"],
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        report = compare_arms(args.bf16_arm, args.fp8_arm, args.output)
        summary = {
            "status": report["status"],
            "output": str(args.output.resolve()),
            "failed_checks": [
                check["name"] for check in report["checks"] if not check["passed"]
            ],
            "metrics": report["metrics"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
