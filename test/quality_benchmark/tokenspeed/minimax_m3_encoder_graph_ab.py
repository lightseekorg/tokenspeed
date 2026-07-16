#!/usr/bin/env python3
"""Collect and compare MiniMax-M3 vision-encoder CUDA Graph performance.

This client never starts, stops, or reconfigures a server.  Collect every arm of
at least two eager/graph TP4 launch pairs whose only material configuration
difference is ``--enable-mm-encoder-cuda-graph``.  Each request is serialized,
and its encoder critical path is the maximum of the four rank-local
``mm_timing encoder_ms ... encode=`` values.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = 1
BENCHMARK_NAME = "minimax_m3_encoder_cuda_graph_ab"
EXPECTED_TP_RANKS = 4
EXPECTED_RANKS = tuple(range(EXPECTED_TP_RANKS))
EXPECTED_WARMUP_REQUESTS = 10
EXPECTED_MEASURE_REQUESTS = 50
EXPECTED_GRAPH_BUDGETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 2304)
EXPECTED_GRAPH_COUNT = EXPECTED_TP_RANKS * len(EXPECTED_GRAPH_BUDGETS)
MIN_LAUNCH_PAIRS = 2
EXPECTED_MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
EXPECTED_REVISION = "c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"
EXPECTED_FIXTURE_SHA256 = (
    "e1cd91db28149f21f3c410ffa074d0fb8bc8950740ba140c7eaae130f0493464"
)
EXPECTED_PROMPT = "What animal is in this image? Reply with one word."
EXPECTED_TOKEN = "Dog"
EXPECTED_PROMPT_TOKENS = 254
EXPECTED_ENCODER_OUTPUT_TOKENS = 77
EXPECTED_INPUT_CONTRACT = {
    "input_ids": [1, EXPECTED_PROMPT_TOKENS],
    "pixel_values": [308, 1176],
    "image_grid_thw": [[1, 14, 22]],
    "merged_image_tokens": EXPECTED_ENCODER_OUTPUT_TOKENS,
}
REFERENCE_LOGPROB_MAX_ABS_DELTA = 0.02
ARM_LOGPROB_MAX_ABS_DELTA = 0.02
MEDIAN_RATIO_MAX = 0.90
BOOTSTRAP_RATIO_CI_UPPER_MAX = 1.0
PER_LAUNCH_RATIO_MAX = 1.05
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260715
EPHEMERAL_SERVER_ARGS = frozenset({"port", "rl_control_port"})
ALLOWED_SERVER_ARG_DIFFERENCES = frozenset({"enable_mm_encoder_cuda_graph"})
REQUIRED_PROVENANCE_KEYS = frozenset(
    {"hardware", "pip_install_report", "runtime_environment", "smg_packages"}
)

_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ENCODER_TIMING_RE = re.compile(
    rf"ATTN TP RANK\s+(?P<rank>\d+).*?"
    rf"mm_timing encoder_ms modality=(?P<modality>\w+) "
    rf"items=(?P<items>\d+) encoder_output_tokens=(?P<tokens>\d+) "
    rf"move_h2d=(?P<move>{_FLOAT_RE}) encode=(?P<encode>{_FLOAT_RE}) "
    r"per_item_tokens=\[(?P<per_item>[^]]*)\]"
)
_RANK_PREFIX_RE = r"ATTN TP RANK\s+(?P<rank>\d+).*?"
_INITIALIZED_RE = re.compile(
    _RANK_PREFIX_RE + r"EncoderCudaGraphWrapper initialized: modality=image, "
    r"budgets=\[(?P<budgets>[^]]+)\]"
)
_CAPTURED_RE = re.compile(
    _RANK_PREFIX_RE
    + r"Captured encoder cudagraph: modality=image, budget=(?P<budget>\d+)"
)
_COMPLETE_RE = re.compile(
    _RANK_PREFIX_RE + r"Encoder CUDA graph capture complete: modality=image, "
    r"(?P<count>\d+) budget graphs\."
)
_INSTALLED_RE = re.compile(
    _RANK_PREFIX_RE + r"Installed encoder CUDA graphs for .*image_encoder"
)
_COMPILE_PATTERNS = (
    (
        "torch_compile",
        re.compile(
            r"torch(?:\.|_)(?:compile|dynamo|inductor)|TorchDynamo|TorchInductor",
            re.IGNORECASE,
        ),
    ),
    (
        "jit_compile",
        re.compile(
            r"(?:\bJIT\b|just-in-time).*(?:compil|kernel)|"
            r"(?:compil|kernel).*(?:\bJIT\b|just-in-time)",
            re.IGNORECASE,
        ),
    ),
    (
        "triton_compile_or_autotune",
        re.compile(
            r"Triton.*(?:compil|autotun|benchmarking|generat(?:e|ing).*kernel)|"
            r"(?:compil|autotun).*(?:Triton|PTX)",
            re.IGNORECASE,
        ),
    ),
    (
        "kernel_compile_or_autotune",
        re.compile(
            r"(?:compil(?:e|ing|ation)|autotun(?:e|ing)).*(?:kernel|config)|"
            r"(?:kernel|config).*(?:compil(?:e|ing|ation)|autotun(?:e|ing))",
            re.IGNORECASE,
        ),
    ),
)


class BenchmarkError(RuntimeError):
    """Raised when an input, server, log, or artifact violates the contract."""


class HttpClient(Protocol):
    def get_json(self, url: str, timeout_seconds: float) -> Any: ...

    def post_json(
        self, url: str, body: Mapping[str, Any], timeout_seconds: float
    ) -> Any: ...


class UrlLibJsonClient:
    """Dependency-free JSON client used by the benchmark CLI."""

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
        return self._request(
            urllib.request.Request(url, headers={"Accept": "application/json"}),
            timeout_seconds,
        )

    def post_json(
        self, url: str, body: Mapping[str, Any], timeout_seconds: float
    ) -> Any:
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


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str
    raw: dict[str, Any]
    reference_logprob: float


@dataclass(frozen=True)
class CollectConfig:
    arm: str
    launch_id: str
    base_url: str
    server_info_base_url: str
    model: str
    dog: Path
    reference: Path
    server_log: Path
    server_sha: str
    output: Path
    request_timeout_seconds: float = 600.0
    server_info_timeout_seconds: float = 30.0
    log_settle_timeout_seconds: float = 30.0
    warmup_requests: int = EXPECTED_WARMUP_REQUESTS
    measure_requests: int = EXPECTED_MEASURE_REQUESTS
    provenance_files: tuple[tuple[str, Path], ...] = ()

    def validate(self) -> None:
        if self.arm not in {"eager", "graph"}:
            raise BenchmarkError("--arm must be eager or graph")
        if not self.launch_id.strip():
            raise BenchmarkError("--launch-id must not be empty")
        for label, value in (
            ("--base-url", self.base_url),
            ("--server-info-base-url", self.server_info_base_url),
        ):
            if not value.startswith(("http://", "https://")):
                raise BenchmarkError(f"{label} must start with http:// or https://")
        if self.model != "minimax-m3":
            raise BenchmarkError("--model must be the fixed served name minimax-m3")
        if not self.server_sha.strip():
            raise BenchmarkError("--server-sha must not be empty")
        for label, value in (
            ("request timeout", self.request_timeout_seconds),
            ("server-info timeout", self.server_info_timeout_seconds),
            ("log-settle timeout", self.log_settle_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise BenchmarkError(f"{label} must be positive and finite")
        if self.warmup_requests != EXPECTED_WARMUP_REQUESTS:
            raise BenchmarkError(
                f"warmup request count must be {EXPECTED_WARMUP_REQUESTS}"
            )
        if self.measure_requests != EXPECTED_MEASURE_REQUESTS:
            raise BenchmarkError(
                f"measurement request count must be {EXPECTED_MEASURE_REQUESTS}"
            )
        provenance_keys = [key for key, _ in self.provenance_files]
        if len(provenance_keys) != len(set(provenance_keys)):
            raise BenchmarkError("--provenance-file keys must be unique")
        for key in provenance_keys:
            if re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
                raise BenchmarkError(
                    f"invalid provenance key {key!r}; use lowercase snake_case"
                )


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _load_provenance(
    provenance_files: Sequence[tuple[str, Path]],
) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for key, path in provenance_files:
        if not path.is_file():
            raise BenchmarkError(f"provenance file {key!r} is missing: {path}")
        payload = path.read_bytes()
        provenance[key] = {
            "path": str(path),
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
    return provenance


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{label} must be an object")
    return value


def load_reference(path: Path, dog: Path) -> Reference:
    """Load and enforce the immutable dog fixture, prompt, and shape contract."""
    if not path.is_file():
        raise BenchmarkError(f"reference file is missing: {path}")
    if not dog.is_file():
        raise BenchmarkError(f"dog fixture is missing: {dog}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot load reference {path}: {exc}") from exc
    reference = _require_mapping(raw, "reference")
    expected_fields = {
        "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "prompt": EXPECTED_PROMPT,
        "input_contract": EXPECTED_INPUT_CONTRACT,
    }
    for field, expected in expected_fields.items():
        if reference.get(field) != expected:
            raise BenchmarkError(
                f"reference.{field} is {reference.get(field)!r}, expected {expected!r}"
            )
    fixture = _require_mapping(reference.get("fixture"), "reference.fixture")
    if fixture.get("name") != "dog.jpg":
        raise BenchmarkError("reference.fixture.name must be dog.jpg")
    if fixture.get("sha256") != EXPECTED_FIXTURE_SHA256:
        raise BenchmarkError("reference fixture SHA256 is not the fixed dog digest")
    dog_sha256 = _sha256_file(dog)
    if dog_sha256 != EXPECTED_FIXTURE_SHA256:
        raise BenchmarkError(
            f"dog fixture SHA256 mismatch: expected {EXPECTED_FIXTURE_SHA256}, "
            f"found {dog_sha256}"
        )
    logits = _require_mapping(reference.get("reference_logits"), "reference logits")
    if logits.get("top_tokens") != [EXPECTED_TOKEN]:
        raise BenchmarkError("reference top token must be Dog")
    logprobs = logits.get("top_logprobs")
    if not isinstance(logprobs, list) or len(logprobs) != 1:
        raise BenchmarkError("reference must contain exactly one top logprob")
    try:
        reference_logprob = float(logprobs[0])
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("reference logprob must be numeric") from exc
    if not math.isfinite(reference_logprob):
        raise BenchmarkError("reference logprob must be finite")
    return Reference(
        path=path,
        sha256=_sha256_file(path),
        raw=dict(reference),
        reference_logprob=reference_logprob,
    )


def _make_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type != "image/jpeg":
        raise BenchmarkError(f"dog fixture must be JPEG, found {mime_type!r}")
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_request(model: str, dog: Path) -> dict[str, Any]:
    """Build the fixed greedy, one-token OpenAI request."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _make_data_url(dog)}},
                    {"type": "text", "text": EXPECTED_PROMPT},
                ],
            }
        ],
        "chat_template_kwargs": {"thinking_mode": "disabled"},
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 0,
    }


def validate_response(raw: Any, reference_logprob: float) -> dict[str, Any]:
    """Normalize one response and return all fixed-reference gate failures."""
    body = _require_mapping(raw, "chat response")
    failures: list[str] = []
    try:
        request_id = body["id"]
        choice = body["choices"][0]
        content = choice["message"]["content"]
        prompt_tokens = body["usage"]["prompt_tokens"]
        token_entry = choice["logprobs"]["content"][0]
        logprob_token = token_entry["token"]
        actual_logprob = float(token_entry["logprob"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"chat response has an unsupported shape: {exc}") from exc
    if not isinstance(request_id, str) or not request_id.strip():
        raise BenchmarkError("chat response id must be a non-empty string")
    if content != EXPECTED_TOKEN:
        failures.append(f"content is {content!r}, expected {EXPECTED_TOKEN!r}")
    if logprob_token != EXPECTED_TOKEN:
        failures.append(
            f"logprob token is {logprob_token!r}, expected {EXPECTED_TOKEN!r}"
        )
    if prompt_tokens != EXPECTED_PROMPT_TOKENS:
        failures.append(
            f"prompt_tokens is {prompt_tokens!r}, expected {EXPECTED_PROMPT_TOKENS}"
        )
    if not math.isfinite(actual_logprob):
        raise BenchmarkError("sampled logprob is not finite")
    reference_delta = abs(actual_logprob - reference_logprob)
    if reference_delta > REFERENCE_LOGPROB_MAX_ABS_DELTA:
        failures.append(
            f"reference logprob delta {reference_delta:.12g} exceeds "
            f"{REFERENCE_LOGPROB_MAX_ABS_DELTA}"
        )
    return {
        "request_id": request_id,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "logprob_token": logprob_token,
        "logprob": actual_logprob,
        "reference_logprob": reference_logprob,
        "reference_logprob_absolute_delta": reference_delta,
        "failures": failures,
    }


def _parse_per_item_tokens(raw: str) -> list[int]:
    if not raw.strip():
        return []
    try:
        return [int(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise BenchmarkError(f"invalid per_item_tokens value: {raw!r}") from exc


def find_encoder_timings(text: str) -> list[dict[str, Any]]:
    """Extract rank-local image-encoder timing rows from a server-log window."""
    clean = _ANSI_RE.sub("", text)
    rows: list[dict[str, Any]] = []
    for match in _ENCODER_TIMING_RE.finditer(clean):
        move_h2d_ms = float(match.group("move"))
        encode_ms = float(match.group("encode"))
        if not math.isfinite(move_h2d_ms) or not math.isfinite(encode_ms):
            raise BenchmarkError("encoder timing values must be finite")
        if move_h2d_ms < 0 or encode_ms <= 0:
            raise BenchmarkError("encoder H2D must be non-negative and encode positive")
        rows.append(
            {
                "rank": int(match.group("rank")),
                "modality": match.group("modality"),
                "items": int(match.group("items")),
                "encoder_output_tokens": int(match.group("tokens")),
                "move_h2d_ms": move_h2d_ms,
                "encode_ms": encode_ms,
                "per_item_tokens": _parse_per_item_tokens(match.group("per_item")),
            }
        )
    return rows


def validate_encoder_timings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate one request's four rank rows and compute critical-path spread."""
    ranks = [int(row["rank"]) for row in rows]
    counts = Counter(ranks)
    if len(rows) != EXPECTED_TP_RANKS or sorted(counts) != list(EXPECTED_RANKS):
        raise BenchmarkError(
            f"expected one encoder timing for ranks {EXPECTED_RANKS}, found {ranks}"
        )
    duplicates = sorted(rank for rank, count in counts.items() if count != 1)
    if duplicates:
        raise BenchmarkError(f"duplicate encoder timing ranks: {duplicates}")
    for row in rows:
        if row["modality"] != "image":
            raise BenchmarkError(f"unexpected encoder modality: {row['modality']!r}")
        if row["items"] != 1:
            raise BenchmarkError(f"encoder items must be 1, found {row['items']!r}")
        if row["encoder_output_tokens"] != EXPECTED_ENCODER_OUTPUT_TOKENS:
            raise BenchmarkError(
                "encoder output token count mismatch: "
                f"{row['encoder_output_tokens']!r}"
            )
        if row["per_item_tokens"] != [EXPECTED_ENCODER_OUTPUT_TOKENS]:
            raise BenchmarkError(
                f"per-item encoder tokens mismatch: {row['per_item_tokens']!r}"
            )
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["rank"])
    encode_values = [float(row["encode_ms"]) for row in ordered]
    maximum = max(encode_values)
    minimum = min(encode_values)
    return {
        "per_rank": {str(row["rank"]): row for row in ordered},
        "critical_path_encode_ms": maximum,
        "rank_spread_ms": maximum - minimum,
        "rank_spread_fraction_of_max": (
            (maximum - minimum) / maximum if maximum > 0 else 0.0
        ),
    }


def find_compile_events(text: str) -> list[dict[str, Any]]:
    """Return possible JIT/Triton compile or autotune events in a log window."""
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _ANSI_RE.sub("", raw_line)
        for name, pattern in _COMPILE_PATTERNS:
            if pattern.search(line):
                events.append(
                    {
                        "kind": name,
                        "line_number_in_window": line_number,
                        "line": line[:1000],
                    }
                )
                break
    return events


def summarize_encoder_graph_log(text: str) -> dict[str, Any]:
    """Summarize stable encoder-graph startup and capture markers."""
    clean = _ANSI_RE.sub("", text)
    per_rank: dict[int, dict[str, Any]] = {}

    def rank_summary(rank: int) -> dict[str, Any]:
        return per_rank.setdefault(
            rank,
            {
                "initialized_count": 0,
                "initialized_budgets": [],
                "capture_complete_count": 0,
                "capture_complete_graph_counts": [],
                "capture_details_total": 0,
                "captures_by_budget": {},
                "installed_count": 0,
                "captured_total": 0,
            },
        )

    initialized_budgets: list[list[int] | None] = []
    for match in _INITIALIZED_RE.finditer(clean):
        rank = int(match.group("rank"))
        raw_budgets = match.group("budgets")
        try:
            budgets: list[int] | None = [
                int(value.strip()) for value in raw_budgets.split(",")
            ]
        except ValueError:
            budgets = None
        initialized_budgets.append(budgets)
        details = rank_summary(rank)
        details["initialized_count"] += 1
        details["initialized_budgets"].append(budgets)

    captured: list[int] = []
    captured_counter: Counter[int] = Counter()
    for match in _CAPTURED_RE.finditer(clean):
        rank = int(match.group("rank"))
        budget = int(match.group("budget"))
        captured.append(budget)
        captured_counter[budget] += 1
        details = rank_summary(rank)
        details["capture_details_total"] += 1
        rank_budget_counts = details["captures_by_budget"]
        rank_budget_counts[str(budget)] = rank_budget_counts.get(str(budget), 0) + 1

    complete_counts: list[int] = []
    for match in _COMPLETE_RE.finditer(clean):
        rank = int(match.group("rank"))
        count = int(match.group("count"))
        complete_counts.append(count)
        details = rank_summary(rank)
        details["capture_complete_count"] += 1
        details["capture_complete_graph_counts"].append(count)
        details["captured_total"] += count

    installed_count = 0
    for match in _INSTALLED_RE.finditer(clean):
        rank = int(match.group("rank"))
        rank_summary(rank)["installed_count"] += 1
        installed_count += 1

    return {
        "initialized": len(initialized_budgets),
        "initialized_budgets": initialized_budgets,
        "capture_complete": len(complete_counts),
        "capture_complete_graph_counts": complete_counts,
        "capture_details_available": bool(captured),
        "capture_details_total": len(captured),
        "captures_by_budget": (
            {str(budget): captured_counter[budget] for budget in EXPECTED_GRAPH_BUDGETS}
            if captured
            else None
        ),
        "installed": installed_count,
        "captured_total": sum(complete_counts),
        "observed_ranks": sorted(per_rank),
        "per_rank": {str(rank): per_rank[rank] for rank in sorted(per_rank)},
    }


def validate_graph_summary(arm: str, summary: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if arm == "eager":
        for field in (
            "initialized",
            "capture_complete",
            "capture_details_total",
            "installed",
            "captured_total",
        ):
            if summary.get(field) != 0:
                failures.append(f"eager server reported encoder graph {field}")
        return failures
    observed_ranks = summary.get("observed_ranks")
    if observed_ranks != list(EXPECTED_RANKS):
        failures.append(
            f"encoder graph markers cover ranks {observed_ranks!r}, "
            f"expected {list(EXPECTED_RANKS)!r}"
        )
    raw_per_rank = summary.get("per_rank")
    per_rank = raw_per_rank if isinstance(raw_per_rank, Mapping) else {}
    expected_budgets = list(EXPECTED_GRAPH_BUDGETS)
    for rank in EXPECTED_RANKS:
        details = per_rank.get(str(rank))
        if not isinstance(details, Mapping):
            failures.append(f"rank {rank} has no encoder graph startup summary")
            continue
        if details.get("initialized_count") != 1:
            failures.append(
                f"rank {rank} initialization count is "
                f"{details.get('initialized_count')!r}, expected 1"
            )
        if details.get("initialized_budgets") != [expected_budgets]:
            failures.append(
                f"rank {rank} initialized budgets are "
                f"{details.get('initialized_budgets')!r}"
            )
        if details.get("capture_complete_count") != 1:
            failures.append(
                f"rank {rank} completion count is "
                f"{details.get('capture_complete_count')!r}, expected 1"
            )
        if details.get("capture_complete_graph_counts") != [
            len(EXPECTED_GRAPH_BUDGETS)
        ]:
            failures.append(
                f"rank {rank} completed graph counts are "
                f"{details.get('capture_complete_graph_counts')!r}, expected [9]"
            )
        if details.get("installed_count") != 1:
            failures.append(
                f"rank {rank} installation count is "
                f"{details.get('installed_count')!r}, expected 1"
            )
        if details.get("captured_total") != len(EXPECTED_GRAPH_BUDGETS):
            failures.append(
                f"rank {rank} captured total is {details.get('captured_total')!r}, "
                "expected 9"
            )
        if summary.get("capture_details_available"):
            if details.get("capture_details_total") != len(EXPECTED_GRAPH_BUDGETS):
                failures.append(
                    f"rank {rank} detailed capture total is "
                    f"{details.get('capture_details_total')!r}, expected 9"
                )
            expected_rank_captures = {
                str(budget): 1 for budget in EXPECTED_GRAPH_BUDGETS
            }
            if details.get("captures_by_budget") != expected_rank_captures:
                failures.append(
                    f"rank {rank} detailed capture budgets are "
                    f"{details.get('captures_by_budget')!r}"
                )
    if summary.get("initialized") != EXPECTED_TP_RANKS:
        failures.append(
            f"expected {EXPECTED_TP_RANKS} graph initializations, "
            f"found {summary.get('initialized')!r}"
        )
    for rank, budgets in enumerate(summary.get("initialized_budgets", [])):
        if budgets != expected_budgets:
            failures.append(f"graph initialization {rank} has budgets {budgets!r}")
    if summary.get("capture_complete") != EXPECTED_TP_RANKS:
        failures.append(
            f"expected {EXPECTED_TP_RANKS} graph completions, "
            f"found {summary.get('capture_complete')!r}"
        )
    completion_counts = summary.get("capture_complete_graph_counts")
    if completion_counts != [len(EXPECTED_GRAPH_BUDGETS)] * EXPECTED_TP_RANKS:
        failures.append(
            "each TP rank must report exactly nine completed graph budgets; "
            f"found {completion_counts!r}"
        )
    if summary.get("captured_total") != EXPECTED_GRAPH_COUNT:
        failures.append(
            f"expected {EXPECTED_GRAPH_COUNT} startup graphs, "
            f"found {summary.get('captured_total')!r}"
        )
    if summary.get("installed") != EXPECTED_TP_RANKS:
        failures.append(
            f"expected {EXPECTED_TP_RANKS} graph installations, "
            f"found {summary.get('installed')!r}"
        )
    if summary.get("capture_details_available"):
        if summary.get("capture_details_total") != EXPECTED_GRAPH_COUNT:
            failures.append("detailed graph capture total is not 36")
        captures_by_budget = summary.get("captures_by_budget") or {}
        for budget in EXPECTED_GRAPH_BUDGETS:
            if captures_by_budget.get(str(budget)) != EXPECTED_TP_RANKS:
                failures.append(
                    f"budget {budget} detailed capture count is not "
                    f"{EXPECTED_TP_RANKS}"
                )
    return failures


def _read_log(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BenchmarkError(f"cannot read server log {path}: {exc}") from exc


def _read_log_window(path: Path, offset: int) -> tuple[bytes, int]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size < offset:
                raise BenchmarkError(
                    f"server log was truncated from offset {offset} to size {size}"
                )
            handle.seek(offset)
            payload = handle.read()
    except OSError as exc:
        raise BenchmarkError(f"cannot read server log window: {exc}") from exc
    return payload, offset + len(payload)


def wait_for_request_log_window(
    path: Path,
    offset: int,
    timeout_seconds: float,
    *,
    request_id: str,
) -> tuple[str, int, list[dict[str, Any]]]:
    """Wait for all rank timings and the matching request-finish marker."""
    deadline = time.monotonic() + timeout_seconds
    latest = b""
    latest_size = offset
    while True:
        latest, latest_size = _read_log_window(path, offset)
        text = latest.decode("utf-8", errors="replace")
        rows = find_encoder_timings(text)
        clean = _ANSI_RE.sub("", text)
        finished = (
            re.search(rf"Req:\s*{re.escape(request_id)}\s+Finish!", clean) is not None
        )
        if len(rows) >= EXPECTED_TP_RANKS and finished:
            return text, latest_size, rows
        if time.monotonic() >= deadline:
            raise BenchmarkError(
                "timed out waiting for request log completion: "
                f"timings={len(rows)}/{EXPECTED_TP_RANKS}, finish={finished}, "
                f"request_id={request_id!r}"
            )
        time.sleep(0.05)


def _extract_server_info(raw: Any, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    info = dict(_require_mapping(raw, "get_server_info response"))
    args = dict(_require_mapping(info.get("server_args"), "server_info.server_args"))
    required = {
        "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "language_model_only": False,
        "enable_log_mm_timing": True,
        "enforce_eager": True,
        "disable_prefill_graph": True,
        "enable_prefix_caching": False,
        "enable_output_logprobs": True,
        "sampling_backend": "greedy",
        "enable_mm_encoder_cuda_graph": arm == "graph",
    }
    failures = [
        f"server_args.{field} is {args.get(field)!r}, expected {expected!r}"
        for field, expected in required.items()
        if args.get(field) != expected
    ]
    try:
        tp_size = int(args.get("attn_tp_size"))
    except (TypeError, ValueError):
        tp_size = None
    if tp_size != EXPECTED_TP_RANKS:
        failures.append(
            f"server_args.attn_tp_size is {args.get('attn_tp_size')!r}, "
            f"expected {EXPECTED_TP_RANKS}"
        )
    if failures:
        raise BenchmarkError("; ".join(failures))
    return info, args


def fetch_server_info_with_retry(
    client: HttpClient, url: str, timeout_seconds: float
) -> tuple[Any, int]:
    """Read the control plane, tolerating its short post-gateway startup lag."""
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_error: BenchmarkError | None = None
    while True:
        attempts += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = f": {last_error}" if last_error is not None else ""
            raise BenchmarkError(
                f"server-info endpoint did not become ready after {attempts - 1} "
                f"attempt(s){detail}"
            )
        try:
            return client.get_json(url, min(5.0, remaining)), attempts
        except BenchmarkError as exc:
            last_error = exc
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


def summarize_samples(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise BenchmarkError("statistics require non-empty finite samples")
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": percentile(ordered, 90.0),
        "p95": percentile(ordered, 95.0),
        "p99": percentile(ordered, 99.0),
        "population_stddev": statistics.pstdev(ordered),
    }


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile for pre-sorted values."""
    if not values:
        raise BenchmarkError("percentile requires at least one value")
    if not 0 <= percent <= 100:
        raise BenchmarkError("percentile must be in [0, 100]")
    position = (len(values) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower]) * (1.0 - weight) + float(values[upper]) * weight


def _arm_statistics(measured: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    critical = [
        float(request["timing"]["critical_path_encode_ms"]) for request in measured
    ]
    e2e = [float(request["e2e_ms"]) for request in measured]
    spread = [float(request["timing"]["rank_spread_ms"]) for request in measured]
    per_rank: dict[str, dict[str, Any]] = {}
    for rank in EXPECTED_RANKS:
        values = [
            float(request["timing"]["per_rank"][str(rank)]["encode_ms"])
            for request in measured
        ]
        per_rank[str(rank)] = summarize_samples(values)
    return {
        "critical_path_encode_ms": summarize_samples(critical),
        "e2e_ms": summarize_samples(e2e),
        "rank_spread_ms": summarize_samples(spread),
        "per_rank_encode_ms": per_rank,
    }


def collect_arm(
    config: CollectConfig, http_client: HttpClient | None = None
) -> dict[str, Any]:
    """Collect one fixed 10-warmup/50-measure arm without managing the server."""
    config.validate()
    reference = load_reference(config.reference, config.dog)
    provenance = _load_provenance(config.provenance_files)
    request_body = build_request(config.model, config.dog)
    client = http_client or UrlLibJsonClient()
    info_url = _join_url(config.server_info_base_url, "/get_server_info")
    chat_url = _join_url(config.base_url, "/v1/chat/completions")
    raw_info, server_info_attempts = fetch_server_info_with_retry(
        client, info_url, config.server_info_timeout_seconds
    )
    server_info, server_args = _extract_server_info(raw_info, config.arm)
    initial_log = _read_log(config.server_log)
    initial_text = initial_log.decode("utf-8", errors="replace")
    initial_graph = summarize_encoder_graph_log(initial_text)
    graph_failures = validate_graph_summary(config.arm, initial_graph)

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "collecting",
        "ok": False,
        "arm": config.arm,
        "launch_id": config.launch_id,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "contract": {
            "concurrency": 1,
            "warmup_requests": config.warmup_requests,
            "measure_requests": config.measure_requests,
            "temperature": 0,
            "max_tokens": 1,
            "expected_tp_ranks": EXPECTED_TP_RANKS,
            "expected_encoder_output_tokens": EXPECTED_ENCODER_OUTPUT_TOKENS,
            "reference_logprob_max_abs_delta": REFERENCE_LOGPROB_MAX_ABS_DELTA,
        },
        "fixture": {
            "path": str(config.dog),
            "sha256": EXPECTED_FIXTURE_SHA256,
        },
        "reference": {
            "path": str(reference.path),
            "sha256": reference.sha256,
            "raw": reference.raw,
        },
        "provenance": provenance,
        "client": {
            "base_url": config.base_url,
            "chat_url": chat_url,
            "server_info_url": info_url,
            "model": config.model,
            "request_timeout_seconds": config.request_timeout_seconds,
            "server_info_timeout_seconds": config.server_info_timeout_seconds,
            "server_info_attempts": server_info_attempts,
            "log_settle_timeout_seconds": config.log_settle_timeout_seconds,
        },
        "server": {
            "sha": config.server_sha,
            "args": server_args,
            "raw_info": server_info,
        },
        "server_log": {
            "path": str(config.server_log),
            "before_requests": {
                "size_bytes": len(initial_log),
                "sha256": _sha256_bytes(initial_log),
                "encoder_graph": initial_graph,
            },
        },
        "warmup": [],
        "measured": [],
    }
    _atomic_write_json(config.output, artifact)

    quality_failures: list[str] = []
    measured_compile_events: list[dict[str, Any]] = []
    warmup_compile_events: list[dict[str, Any]] = []
    measurement_log_start_offset: int | None = None
    try:
        for phase, count in (
            ("warmup", config.warmup_requests),
            ("measured", config.measure_requests),
        ):
            destination = artifact[phase]
            if phase == "measured":
                try:
                    measurement_log_start_offset = config.server_log.stat().st_size
                except OSError as exc:
                    raise BenchmarkError(
                        f"cannot snapshot measurement log start: {exc}"
                    ) from exc
            for index in range(count):
                try:
                    request_log_offset = config.server_log.stat().st_size
                except OSError as exc:
                    raise BenchmarkError(
                        f"cannot snapshot server log before {phase}[{index}]: {exc}"
                    ) from exc
                request_start_ns = time.perf_counter_ns()
                raw_response = client.post_json(
                    chat_url, request_body, config.request_timeout_seconds
                )
                e2e_ms = (time.perf_counter_ns() - request_start_ns) / 1_000_000.0
                if not math.isfinite(e2e_ms) or e2e_ms <= 0:
                    raise BenchmarkError(f"{phase}[{index}] E2E time is not positive")
                response = validate_response(raw_response, reference.reference_logprob)
                log_window, log_end_offset, timing_rows = wait_for_request_log_window(
                    config.server_log,
                    request_log_offset,
                    config.log_settle_timeout_seconds,
                    request_id=response["request_id"],
                )
                timing = validate_encoder_timings(timing_rows)
                label = f"{phase}[{index}]"
                quality_failures.extend(
                    f"{label}: {failure}" for failure in response["failures"]
                )
                compile_events = find_compile_events(log_window)
                annotated_compile_events = [
                    {"phase": phase, "request_index": index, **event}
                    for event in compile_events
                ]
                if phase == "measured":
                    measured_compile_events.extend(annotated_compile_events)
                else:
                    warmup_compile_events.extend(annotated_compile_events)
                destination.append(
                    {
                        "index": index,
                        "e2e_ms": e2e_ms,
                        "timing": timing,
                        "response": response,
                        "compile_events": compile_events,
                        "raw": {
                            "response": raw_response,
                            "server_log_window": log_window,
                            "server_log_window_sha256": _sha256_bytes(
                                log_window.encode("utf-8")
                            ),
                            "server_log_start_offset": request_log_offset,
                            "server_log_end_offset": log_end_offset,
                        },
                    }
                )
                artifact["updated_at_utc"] = _utc_now()
                _atomic_write_json(config.output, artifact)
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["updated_at_utc"] = _utc_now()
        _atomic_write_json(config.output, artifact)
        raise

    final_log = _read_log(config.server_log)
    assert measurement_log_start_offset is not None
    measurement_log_bytes = final_log[measurement_log_start_offset:]
    measurement_log_text = measurement_log_bytes.decode("utf-8", errors="replace")
    measurement_span_compile_events = find_compile_events(measurement_log_text)
    final_graph = summarize_encoder_graph_log(
        final_log.decode("utf-8", errors="replace")
    )
    graph_failures.extend(validate_graph_summary(config.arm, final_graph))
    capture_counts_unchanged = initial_graph == final_graph
    if not capture_counts_unchanged:
        graph_failures.append(
            "encoder CUDA Graph capture counts changed during requests"
        )
    if measurement_span_compile_events:
        compile_failure = (
            f"measurement span contains {len(measurement_span_compile_events)} "
            "JIT/Triton compile or autotune event(s)"
        )
    else:
        compile_failure = None
    failures = [*quality_failures, *graph_failures]
    if compile_failure:
        failures.append(compile_failure)
    artifact["server_log"]["after_requests"] = {
        "size_bytes": len(final_log),
        "sha256": _sha256_bytes(final_log),
        "encoder_graph": final_graph,
        "capture_counts_unchanged": capture_counts_unchanged,
    }
    artifact["compile_events"] = {
        "warmup": warmup_compile_events,
        "measured_by_request": measured_compile_events,
        "measurement_span": {
            "server_log_start_offset": measurement_log_start_offset,
            "server_log_end_offset": len(final_log),
            "server_log_sha256": _sha256_bytes(measurement_log_bytes),
            "events": measurement_span_compile_events,
        },
        "measurement_count": len(measurement_span_compile_events),
    }
    artifact["statistics"] = _arm_statistics(artifact["measured"])
    artifact["validation"] = {
        "ok": not failures,
        "quality_failures": quality_failures,
        "graph_failures": graph_failures,
        "measurement_compile_events_absent": not measurement_span_compile_events,
        "failures": failures,
    }
    artifact["status"] = "complete"
    artifact["ok"] = not failures
    artifact["updated_at_utc"] = _utc_now()
    _atomic_write_json(config.output, artifact)
    return artifact


def _read_arm(path: Path, expected_arm: str) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot load arm artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise BenchmarkError(f"arm artifact {path} must be an object")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": "complete",
        "arm": expected_arm,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise BenchmarkError(
                f"{path}.{field} is {artifact.get(field)!r}, expected {value!r}"
            )
    launch_id = artifact.get("launch_id")
    if not isinstance(launch_id, str) or not launch_id.strip():
        raise BenchmarkError(f"{path}.launch_id must be a non-empty string")
    contract = _require_mapping(artifact.get("contract"), f"{path}.contract")
    if contract.get("concurrency") != 1:
        raise BenchmarkError(f"{path} was not collected at concurrency 1")
    if contract.get("warmup_requests") != EXPECTED_WARMUP_REQUESTS:
        raise BenchmarkError(f"{path} does not contain 10 warmup requests")
    if contract.get("measure_requests") != EXPECTED_MEASURE_REQUESTS:
        raise BenchmarkError(f"{path} does not contain 50 measured requests")
    if len(artifact.get("warmup", [])) != EXPECTED_WARMUP_REQUESTS:
        raise BenchmarkError(f"{path} warmup row count is not 10")
    if len(artifact.get("measured", [])) != EXPECTED_MEASURE_REQUESTS:
        raise BenchmarkError(f"{path} measured row count is not 50")
    server = _require_mapping(artifact.get("server"), f"{path}.server")
    server_sha = server.get("sha")
    if not isinstance(server_sha, str) or not server_sha.strip():
        raise BenchmarkError(f"{path}.server.sha must be non-empty")
    server_args = _require_mapping(server.get("args"), f"{path}.server.args")
    expected_graph_flag = expected_arm == "graph"
    if server_args.get("enable_mm_encoder_cuda_graph") is not expected_graph_flag:
        raise BenchmarkError(
            f"{path} does not report enable_mm_encoder_cuda_graph="
            f"{expected_graph_flag}"
        )
    return artifact


def _normalized_server_args(artifact: Mapping[str, Any]) -> dict[str, Any]:
    server = _require_mapping(artifact.get("server"), "artifact.server")
    args = _require_mapping(server.get("args"), "artifact.server.args")
    return {
        key: value for key, value in args.items() if key not in EPHEMERAL_SERVER_ARGS
    }


def _server_arg_differences(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(left) | set(right))
    return {
        key: {"eager": left.get(key), "graph": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    }


def _provenance_hashes(
    artifact: Mapping[str, Any], label: str
) -> tuple[dict[str, str], list[str]]:
    failures: list[str] = []
    raw = artifact.get("provenance")
    if not isinstance(raw, Mapping):
        return {}, [f"{label} lacks provenance"]
    missing = sorted(REQUIRED_PROVENANCE_KEYS - set(raw))
    if missing:
        failures.append(f"{label} missing provenance keys {missing}")
    hashes: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            failures.append(f"{label} provenance {key!r} must be an object")
            continue
        sha256 = value.get("sha256")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            failures.append(f"{label} provenance {key!r} has invalid SHA256")
            continue
        hashes[str(key)] = sha256
    return hashes, failures


def _measurement_values(artifact: Mapping[str, Any], field: str) -> list[float]:
    measured = artifact["measured"]
    if field == "critical_path_encode_ms":
        return [float(row["timing"][field]) for row in measured]
    if field == "e2e_ms":
        return [float(row[field]) for row in measured]
    raise BenchmarkError(f"unsupported measurement field: {field}")


def _bootstrap_median_ratio(
    eager_launches: Sequence[Sequence[float]],
    graph_launches: Sequence[Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise BenchmarkError("bootstrap sample count must be positive")
    if not eager_launches or len(eager_launches) != len(graph_launches):
        raise BenchmarkError("bootstrap requires matching non-empty launch groups")
    if any(not launch for launch in (*eager_launches, *graph_launches)):
        raise BenchmarkError("bootstrap launch groups must not be empty")
    if any(min(launch) <= 0 for launch in (*eager_launches, *graph_launches)):
        raise BenchmarkError("bootstrap inputs must be positive and non-empty")
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        eager_sample: list[float] = []
        graph_sample: list[float] = []
        # Resample launch pairs first, then requests within each selected arm.
        # This retains between-launch variation instead of pretending that all
        # requests from multiple server starts are independent launches.
        for _ in eager_launches:
            launch_index = rng.randrange(len(eager_launches))
            eager_launch = eager_launches[launch_index]
            graph_launch = graph_launches[launch_index]
            eager_sample.extend(
                eager_launch[rng.randrange(len(eager_launch))] for _ in eager_launch
            )
            graph_sample.extend(
                graph_launch[rng.randrange(len(graph_launch))] for _ in graph_launch
            )
        ratios.append(statistics.median(graph_sample) / statistics.median(eager_sample))
    ratios.sort()
    return {
        "samples": samples,
        "seed": seed,
        "confidence_level": 0.95,
        "method": "launch-pair cluster bootstrap with within-arm request resampling",
        "launch_pairs": len(eager_launches),
        "ratio_ci": {
            "lower": percentile(ratios, 2.5),
            "upper": percentile(ratios, 97.5),
        },
    }


def compare_arms(
    eager_paths: Sequence[Path],
    graph_paths: Sequence[Path],
    *,
    output: Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare ordered arm pairs; fewer than two pairs fail the release gate."""
    if not eager_paths or len(eager_paths) != len(graph_paths):
        raise BenchmarkError("provide the same non-zero number of eager and graph arms")
    for label, paths in (("eager", eager_paths), ("graph", graph_paths)):
        normalized_paths = [str(path.resolve()) for path in paths]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise BenchmarkError(f"duplicate {label} artifact path is not allowed")
    eager_arms = [_read_arm(path, "eager") for path in eager_paths]
    graph_arms = [_read_arm(path, "graph") for path in graph_paths]
    eager_launch_ids = [artifact["launch_id"] for artifact in eager_arms]
    graph_launch_ids = [artifact["launch_id"] for artifact in graph_arms]
    if len(eager_launch_ids) != len(set(eager_launch_ids)):
        raise BenchmarkError("duplicate eager launch_id is not allowed")
    if len(graph_launch_ids) != len(set(graph_launch_ids)):
        raise BenchmarkError("duplicate graph launch_id is not allowed")
    if eager_launch_ids != graph_launch_ids:
        raise BenchmarkError(
            "ordered eager and graph artifacts must use matching launch_id values"
        )
    minimum_launch_pairs_gate = len(eager_arms) >= MIN_LAUNCH_PAIRS
    failures: list[str] = []
    if not minimum_launch_pairs_gate:
        failures.append(
            f"release comparison requires at least {MIN_LAUNCH_PAIRS} independent "
            f"launch pairs; found {len(eager_arms)}"
        )
    server_differences: list[dict[str, Any]] = []
    provenance_failures: list[str] = []
    launch_rows: list[dict[str, Any]] = []
    graph_eager_logprob_deltas: list[float] = []

    baseline_sha = eager_arms[0]["server"]["sha"]
    baseline_reference_sha = eager_arms[0]["reference"]["sha256"]
    baseline_fixture_sha = eager_arms[0]["fixture"]["sha256"]
    baseline_provenance, baseline_provenance_failures = _provenance_hashes(
        eager_arms[0], "eager launch " + str(eager_arms[0]["launch_id"])
    )
    provenance_failures.extend(baseline_provenance_failures)
    all_arms = [*eager_arms, *graph_arms]
    for artifact in all_arms:
        if artifact["server"]["sha"] != baseline_sha:
            failures.append("all arms must use the same TokenSpeed SHA")
        if artifact["reference"]["sha256"] != baseline_reference_sha:
            failures.append("all arms must use the same visual reference")
        if artifact["fixture"]["sha256"] != baseline_fixture_sha:
            failures.append("all arms must use the same dog fixture")
        if artifact.get("ok") is not True:
            failures.append(
                f"{artifact['arm']} launch {artifact['launch_id']} did not pass collection"
            )
        provenance_label = f"{artifact['arm']} launch {artifact['launch_id']}"
        current_provenance, current_failures = _provenance_hashes(
            artifact, provenance_label
        )
        if artifact is not eager_arms[0]:
            provenance_failures.extend(current_failures)
        if current_provenance != baseline_provenance:
            provenance_failures.append(f"{provenance_label} provenance differs")

    failures.extend(provenance_failures)

    baseline_args = _normalized_server_args(eager_arms[0])
    for pair_index, (eager, graph) in enumerate(zip(eager_arms, graph_arms)):
        eager_args = _normalized_server_args(eager)
        graph_args = _normalized_server_args(graph)
        for label, args in (("eager", eager_args), ("graph", graph_args)):
            intra_differences = _server_arg_differences(baseline_args, args)
            unexpected = {
                key: value
                for key, value in intra_differences.items()
                if key not in ALLOWED_SERVER_ARG_DIFFERENCES
            }
            if unexpected:
                failures.append(
                    f"{label} launch {pair_index} differs from baseline server args"
                )
                server_differences.append(
                    {
                        "pair_index": pair_index,
                        "scope": label,
                        "differences": unexpected,
                    }
                )
        pair_differences = _server_arg_differences(eager_args, graph_args)
        unexpected_pair = {
            key: value
            for key, value in pair_differences.items()
            if key not in ALLOWED_SERVER_ARG_DIFFERENCES
        }
        if unexpected_pair:
            failures.append(
                f"launch pair {pair_index} differs by more than the encoder graph flag"
            )
            server_differences.append(
                {
                    "pair_index": pair_index,
                    "scope": "pair",
                    "differences": unexpected_pair,
                }
            )

        eager_encode = _measurement_values(eager, "critical_path_encode_ms")
        graph_encode = _measurement_values(graph, "critical_path_encode_ms")
        eager_e2e = _measurement_values(eager, "e2e_ms")
        graph_e2e = _measurement_values(graph, "e2e_ms")
        encode_ratio = statistics.median(graph_encode) / statistics.median(eager_encode)
        e2e_ratio = statistics.median(graph_e2e) / statistics.median(eager_e2e)
        launch_regression_ok = encode_ratio <= PER_LAUNCH_RATIO_MAX
        if not launch_regression_ok:
            failures.append(
                f"launch pair {pair_index} encoder median ratio {encode_ratio:.6f} "
                f"exceeds {PER_LAUNCH_RATIO_MAX}"
            )
        launch_rows.append(
            {
                "pair_index": pair_index,
                "eager_launch_id": eager["launch_id"],
                "graph_launch_id": graph["launch_id"],
                "critical_path_encode_median_ratio": encode_ratio,
                "e2e_median_ratio": e2e_ratio,
                "no_encoder_regression_over_five_percent": launch_regression_ok,
            }
        )
        for request_index, (eager_row, graph_row) in enumerate(
            zip(eager["measured"], graph["measured"])
        ):
            delta = abs(
                float(graph_row["response"]["logprob"])
                - float(eager_row["response"]["logprob"])
            )
            graph_eager_logprob_deltas.append(delta)
            if delta > ARM_LOGPROB_MAX_ABS_DELTA:
                failures.append(
                    f"pair {pair_index} request {request_index} graph/eager "
                    f"logprob delta {delta:.12g} exceeds {ARM_LOGPROB_MAX_ABS_DELTA}"
                )

    eager_encode_launches = [
        _measurement_values(artifact, "critical_path_encode_ms")
        for artifact in eager_arms
    ]
    graph_encode_launches = [
        _measurement_values(artifact, "critical_path_encode_ms")
        for artifact in graph_arms
    ]
    eager_e2e_launches = [
        _measurement_values(artifact, "e2e_ms") for artifact in eager_arms
    ]
    graph_e2e_launches = [
        _measurement_values(artifact, "e2e_ms") for artifact in graph_arms
    ]
    pooled_eager_encode = [
        value for launch in eager_encode_launches for value in launch
    ]
    pooled_graph_encode = [
        value for launch in graph_encode_launches for value in launch
    ]
    pooled_eager_e2e = [value for launch in eager_e2e_launches for value in launch]
    pooled_graph_e2e = [value for launch in graph_e2e_launches for value in launch]
    encode_ratio = statistics.median(pooled_graph_encode) / statistics.median(
        pooled_eager_encode
    )
    e2e_ratio = statistics.median(pooled_graph_e2e) / statistics.median(
        pooled_eager_e2e
    )
    encode_bootstrap = _bootstrap_median_ratio(
        eager_encode_launches,
        graph_encode_launches,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    e2e_bootstrap = _bootstrap_median_ratio(
        eager_e2e_launches,
        graph_e2e_launches,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    median_speed_gate = encode_ratio <= MEDIAN_RATIO_MAX
    ci_gate = encode_bootstrap["ratio_ci"]["upper"] < BOOTSTRAP_RATIO_CI_UPPER_MAX
    per_launch_gate = all(
        row["no_encoder_regression_over_five_percent"] for row in launch_rows
    )
    logprob_gate = (
        not graph_eager_logprob_deltas
        or max(graph_eager_logprob_deltas) <= ARM_LOGPROB_MAX_ABS_DELTA
    )
    if not median_speed_gate:
        failures.append(
            f"pooled encoder median ratio {encode_ratio:.6f} exceeds {MEDIAN_RATIO_MAX}"
        )
    if not ci_gate:
        failures.append(
            "encoder median-ratio bootstrap CI crosses or touches 1.0: "
            f"{encode_bootstrap['ratio_ci']}"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "created_at_utc": _utc_now(),
        "ok": not failures,
        "inputs": {
            "eager": [str(path) for path in eager_paths],
            "graph": [str(path) for path in graph_paths],
            "launch_pairs": len(eager_paths),
            "minimum_launch_pairs": MIN_LAUNCH_PAIRS,
            "server_sha": baseline_sha,
            "reference_sha256": baseline_reference_sha,
            "fixture_sha256": baseline_fixture_sha,
        },
        "statistics": {
            "eager": {
                "critical_path_encode_ms": summarize_samples(pooled_eager_encode),
                "e2e_ms": summarize_samples(pooled_eager_e2e),
            },
            "graph": {
                "critical_path_encode_ms": summarize_samples(pooled_graph_encode),
                "e2e_ms": summarize_samples(pooled_graph_e2e),
            },
            "critical_path_encode": {
                "graph_over_eager_median_ratio": encode_ratio,
                "speedup": 1.0 / encode_ratio,
                "bootstrap": encode_bootstrap,
            },
            "e2e": {
                "graph_over_eager_median_ratio": e2e_ratio,
                "speedup": 1.0 / e2e_ratio,
                "bootstrap": e2e_bootstrap,
                "gate": "informational_only",
            },
            "graph_eager_logprob_absolute_delta": summarize_samples(
                graph_eager_logprob_deltas
            ),
        },
        "launch_pairs": launch_rows,
        "server_arg_differences": server_differences,
        "provenance": {
            "required_keys": sorted(REQUIRED_PROVENANCE_KEYS),
            "sha256_by_key": baseline_provenance,
            "failures": provenance_failures,
        },
        "gates": {
            "at_least_two_independent_launch_pairs": minimum_launch_pairs_gate,
            "median_encoder_at_least_ten_percent_faster": median_speed_gate,
            "encoder_ratio_bootstrap_ci_upper_below_one": ci_gate,
            "no_launch_encoder_regression_over_five_percent": per_launch_gate,
            "graph_eager_logprob_delta_at_most_0_02": logprob_gate,
            "all_arm_collectors_passed": all(
                artifact.get("ok") is True for artifact in all_arms
            ),
            "server_args_differ_only_by_graph_flag": not server_differences,
            "hardware_environment_and_packages_match": not provenance_failures,
        },
        "failures": failures,
    }
    _atomic_write_json(output, report)
    return report


def parse_provenance_files(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key or not raw_path:
            raise BenchmarkError(
                "--provenance-file must use lowercase_key=/path/to/artifact"
            )
        parsed.append((key, Path(raw_path)))
    return tuple(parsed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect one eager or graph arm")
    collect.add_argument("--arm", choices=("eager", "graph"), required=True)
    collect.add_argument("--launch-id", default="launch-1")
    collect.add_argument("--base-url", required=True)
    collect.add_argument("--server-info-base-url", required=True)
    collect.add_argument("--model", default="minimax-m3")
    collect.add_argument("--dog", type=Path, required=True)
    collect.add_argument("--reference", type=Path, required=True)
    collect.add_argument("--server-log", type=Path, required=True)
    collect.add_argument("--server-sha", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--request-timeout-seconds", type=float, default=600.0)
    collect.add_argument("--server-info-timeout-seconds", type=float, default=30.0)
    collect.add_argument("--log-settle-timeout-seconds", type=float, default=30.0)
    collect.add_argument(
        "--provenance-file",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Bind a hardware/environment/package artifact into this arm",
    )

    compare = subparsers.add_parser("compare", help="compare ordered arm pairs")
    compare.add_argument("--eager", type=Path, action="append", required=True)
    compare.add_argument("--graph", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    compare.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "collect":
            config = CollectConfig(
                arm=args.arm,
                launch_id=args.launch_id,
                base_url=args.base_url,
                server_info_base_url=args.server_info_base_url,
                model=args.model,
                dog=args.dog,
                reference=args.reference,
                server_log=args.server_log,
                server_sha=args.server_sha,
                output=args.output,
                request_timeout_seconds=args.request_timeout_seconds,
                server_info_timeout_seconds=args.server_info_timeout_seconds,
                log_settle_timeout_seconds=args.log_settle_timeout_seconds,
                provenance_files=parse_provenance_files(args.provenance_file),
            )
            result = collect_arm(config)
        else:
            result = compare_arms(
                args.eager,
                args.graph,
                output=args.output,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"{BENCHMARK_NAME} {args.command}: {'PASS' if result['ok'] else 'FAIL'}",
        flush=True,
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
