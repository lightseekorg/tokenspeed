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

"""Shared logic of the native Mooncake cross-instance serving gate.

The gate (``test/runtime/test_l3_serving_e2e.py``) checks that a prefix one
engine persists to a native Mooncake store is restored by a fresh engine's
FIRST request and serves the same output as a recomputation oracle. This
module carries every decision the gate makes that does not need a GPU: the
operator configuration schema, engine command lines, Prometheus evidence
extraction, the evidence/output comparators, the prompts and their
alignment, and the bounded waits and child-process cleanup.

Only the Python standard library is used here, so the CPU contract tests
(``test/test_l3_serving_gate.py``) exercise these functions in a minimal
environment: importing this module must never need torch, CUDA, the
``tokenspeed`` runtime package or a store library.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import IO

# The gate runs only when this variable names the operator's JSON config.
CONFIG_ENV_VAR = "L3_SERVING_E2E_CONFIG"

SCHEMA_VERSION = 1

# Printed on its own line by the provider process once its segment is
# mounted.
PROVIDER_READY_TOKEN = "L3_PROVIDER_READY"

# Prometheus exposition names the gate reads as evidence, as the engine host
# exports them (ts-metrics; counters gain ``_total``).
L3_IO_KEYS = "tokenspeed:l3_io_keys_total"
L3_IO_ADAPTER_BYTES = "tokenspeed:l3_io_adapter_bytes_total"
L3_IO_OPERATIONS = "tokenspeed:l3_io_operations"
L3_SCHEDULER_RESOURCES = "tokenspeed:l3_scheduler_resources"
L3_IO_QUARANTINED = "tokenspeed:l3_io_quarantined"
L3_DROPPED_BACKUP_KEYS = "tokenspeed:l3_dropped_backup_keys_total"
# Token spans of the keys a storage QUERY found in the store, counted per
# cache group when the query completes -- before, and independently of,
# any GET or admission. It shows the store held the reader's keys; it is
# not reuse. Reuse is proven by the fresh reader's exact ``cached_tokens``
# together with positive GET deltas.
PREFIX_L3_QUERY_HIT_TOKENS = "tokenspeed:prefix_l3_hit_tokens_total"

# The L3 gauges are set together on every storage-metrics export (at most
# once a second, and whenever storage turns pending or idle); counters
# appear only after their first increment. A gauge missing from a scrape
# therefore means "not exported yet" and can prove nothing, while a missing
# counter legitimately reads as zero.
_GAUGES = {
    "put_jobs": (L3_IO_OPERATIONS, (("kind", "put"),)),
    "get_jobs": (L3_IO_OPERATIONS, (("kind", "get"),)),
    "storage_operations": (L3_SCHEDULER_RESOURCES, (("resource", "operations"),)),
    "backup_candidates": (
        L3_SCHEDULER_RESOURCES,
        (("resource", "backup_candidates"),),
    ),
    "quarantined_blocks": (L3_SCHEDULER_RESOURCES, (("resource", "quarantined"),)),
    "io_quarantined": (L3_IO_QUARANTINED, ()),
}
_COUNTERS = {
    "put_success_keys": (L3_IO_KEYS, (("kind", "put"), ("result", "success"))),
    "put_failed_keys": (L3_IO_KEYS, (("kind", "put"), ("result", "failure"))),
    "put_success_bytes": (L3_IO_ADAPTER_BYTES, (("kind", "put"),)),
    "get_success_keys": (L3_IO_KEYS, (("kind", "get"), ("result", "success"))),
    "get_failed_keys": (L3_IO_KEYS, (("kind", "get"), ("result", "failure"))),
    "get_success_bytes": (L3_IO_ADAPTER_BYTES, (("kind", "get"),)),
    "dropped_backup_keys": (L3_DROPPED_BACKUP_KEYS, ()),
    "l3_query_hit_tokens": (PREFIX_L3_QUERY_HIT_TOKENS, ()),
}

# Engine arguments the harness sets itself. A model case's extra arguments
# must not repeat them: duplicates would silently override (or be overridden
# by) the gate's own choices and could desynchronize the writer's and the
# reader's storage namespace or the oracle's numerics. Aliases of managed
# flags (--model-path, --disable-flashinfer-autotune) are managed too:
# argparse resolves them to the same destination, so allowing them would
# bypass the guard.
MANAGED_FLAGS = frozenset(
    {
        "--model",
        "--model-path",
        "--dtype",
        "--seed",
        "--enable-prefix-caching",
        "--disable-prefix-caching",
        "--disable-kvstore",
        "--kvstore-ratio",
        "--kvstore-size",
        "--kvstore-io-backend",
        "--kvstore-write-policy",
        "--kvstore-storage-backend",
        "--kvstore-storage-backend-extra-config",
        "--enable-metrics",
        "--metrics-reporters",
        "--metrics-port",
        "--metrics-host",
        "--enable-output-logprobs",
        "--max-model-len",
        "--max-num-seqs",
        "--max-prefill-tokens",
        "--chunked-prefill-size",
        "--gpu-memory-utilization",
        "--disable-autotune",
        "--disable-flashinfer-autotune",
        "--data-parallel-address",
        "--data-parallel-rpc-port",
        "--zmq-engine-index",
        "--attn-tp-size",
        "--disaggregation-mode",
    }
)

# Multi-rank metrics and speculative state need their own acceptance matrix.
# This gate deliberately launches one fused rank with ordinary greedy decode.
_OUT_OF_SCOPE_FLAGS = frozenset(
    {
        "--tp",
        "--tensor-parallel-size",
        "--data-parallel-size",
        "--pipeline-parallel-size",
        "--expert-parallel-size",
        "--ep-size",
        "--enable-expert-parallel",
        "--kv-parallel-size",
        "--decode-context-parallel-size",
        "--query-context-parallel-size",
        "--attn-head-tp-size",
        "--lm-head-tp-size",
        "--dense-tp-size",
        "--moe-tp-size",
        "--world-size",
        "--nprocs-per-node",
        "--deepseek-v41-attention-query-parallel-size",
        "--deepseek-v41-indexer-query-parallel-size",
    }
)

# L3 async options the gate's checks depend on, so the config must state
# them instead of inheriting engine defaults.
_REQUIRED_ASYNC_OPTIONS = ("wait_timeout_ms", "max_prefetch_tokens")

_DTYPES = ("bfloat16", "float16", "float32")
_FINISH_OK = ("stop", "length")


class GateConfigError(ValueError):
    """An operator configuration the gate refuses: the run stops here."""


@dataclass(frozen=True)
class MooncakeConnection:
    """Store connection facts shared by the engines and the provider."""

    protocol: str
    device_name: str
    metadata_server: str
    local_hostname: str


@dataclass(frozen=True)
class ProviderConfig:
    """The independent segment provider: bytes it mounts into the store."""

    segment_bytes: int


@dataclass(frozen=True)
class TimeoutConfig:
    """Every deadline the gate waits against, in seconds."""

    engine_startup_seconds: float
    request_seconds: float
    put_evidence_seconds: float
    metrics_poll_interval_seconds: float


@dataclass(frozen=True)
class ModelCase:
    """One model the gate runs, with the oracle's numerical envelope."""

    name: str
    model_path: str
    dtype: str
    extra_engine_args: tuple[str, ...]
    page_granularity_tokens: int
    chunked_prefill_tokens: int
    min_prefix_tokens: int
    max_model_len: int
    max_num_seqs: int
    max_prefill_tokens: int
    gpu_memory_utilization: float
    kvstore_ratio: float
    seed: int
    writer_max_new_tokens: int
    suffix_max_new_tokens: int
    min_output_tokens: int
    max_token_mismatches: int
    logprob_atol: float
    min_put_success_keys: int
    min_put_success_bytes: int
    min_get_success_keys: int
    min_get_success_bytes: int
    min_l3_query_hit_tokens: int


@dataclass(frozen=True)
class GateConfig:
    """The whole operator configuration (``schema_version`` 1)."""

    schema_version: int
    tenant_prefix: str
    mooncake: MooncakeConnection
    provider: ProviderConfig
    timeouts: TimeoutConfig
    async_overrides: tuple[tuple[str, int], ...]
    cases: tuple[ModelCase, ...]


def _fail(path: str, message: str) -> None:
    raise GateConfigError(f"{path}: {message}")


def _mapping(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        _fail(path, f"must be a JSON object, got {type(value).__name__}")
    return value


def _keys(mapping: dict, allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        _fail(path, f"unknown keys {unknown}; allowed keys are {sorted(allowed)}")


def _string(value: object, path: str, *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        _fail(path, f"must be a string, got {type(value).__name__}")
    if not allow_empty and not value:
        _fail(path, "must not be empty")
    return value


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"must be an integer, got {type(value).__name__}")
    if value < minimum:
        _fail(path, f"must be >= {minimum}, got {value}")
    return value


def _real(value: object, path: str, *, minimum: float, below_one: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, f"must be finite, got {result}")
    if below_one and not result < 1.0:
        _fail(path, f"must be < 1, got {result}")
    if not result > minimum:
        _fail(path, f"must be > {minimum}, got {result}")
    return result


def _string_list(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _fail(path, "must be a list of strings")
    return tuple(value)


def check_extra_engine_args(args: tuple[str, ...], path: str) -> None:
    """Reject case arguments that repeat a harness-managed engine flag."""
    for argument in args:
        flag = argument.split("=", 1)[0]
        if flag in _OUT_OF_SCOPE_FLAGS or flag.startswith("--speculative"):
            _fail(
                path,
                f"{flag!r} is outside the gate's single-rank, non-speculative scope",
            )
        if flag in MANAGED_FLAGS:
            _fail(
                path,
                f"{argument!r} repeats the harness-managed {flag}; the gate sets "
                "it from the case/config fields so writer, reader and oracle "
                "cannot drift apart",
            )


def _parse_connection(raw: object, path: str) -> MooncakeConnection:
    mapping = _mapping(raw, path)
    allowed = {"protocol", "device_name", "metadata_server", "local_hostname"}
    _keys(mapping, allowed, path)
    missing = sorted(allowed - set(mapping))
    if missing:
        _fail(path, f"missing required keys {missing}")
    if mapping["protocol"] != "tcp":
        _fail(
            path,
            "this gate currently requires protocol tcp; RDMA validation is separate",
        )
    return MooncakeConnection(
        protocol=_string(mapping["protocol"], f"{path}.protocol", allow_empty=False),
        device_name=_string(
            mapping["device_name"], f"{path}.device_name", allow_empty=True
        ),
        metadata_server=_string(
            mapping["metadata_server"], f"{path}.metadata_server", allow_empty=False
        ),
        local_hostname=_string(
            mapping["local_hostname"], f"{path}.local_hostname", allow_empty=False
        ),
    )


def _parse_provider(raw: object, path: str) -> ProviderConfig:
    mapping = _mapping(raw, path)
    _keys(mapping, {"segment_bytes"}, path)
    return ProviderConfig(
        segment_bytes=_integer(
            mapping.get("segment_bytes"),
            f"{path}.segment_bytes",
            minimum=16 * 1024 * 1024,
        )
    )


def _parse_timeouts(raw: object, path: str) -> TimeoutConfig:
    names = (
        "engine_startup_seconds",
        "request_seconds",
        "put_evidence_seconds",
        "metrics_poll_interval_seconds",
    )
    mapping = _mapping(raw, path)
    _keys(mapping, set(names), path)
    fields = {
        name: _real(mapping.get(name), f"{path}.{name}", minimum=0.0, below_one=False)
        for name in names
    }
    return TimeoutConfig(**fields)


def _parse_async_overrides(raw: object, path: str) -> tuple[tuple[str, int], ...]:
    """The engines' L3 ``async`` object. Option names and ranges are the
    engine's to validate (a refused option fails engine startup); the gate
    only insists on the options its own checks depend on."""
    mapping = _mapping(raw, path)
    missing = [name for name in _REQUIRED_ASYNC_OPTIONS if name not in mapping]
    if missing:
        _fail(
            path,
            f"must set {missing} explicitly: wait_timeout_ms is the soft wait "
            "budget of the reader's first request, and the persisted prefix "
            "must fit max_prefetch_tokens to be restored whole",
        )
    items = []
    for key in sorted(mapping):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", key):
            _fail(path, f"invalid option name {key!r}")
        items.append((key, _integer(mapping[key], f"{path}.{key}", minimum=0)))
    return tuple(items)


def _parse_case(raw: object, path: str) -> ModelCase:
    mapping = _mapping(raw, path)
    allowed = {
        "name",
        "model_path",
        "dtype",
        "extra_engine_args",
        "page_granularity_tokens",
        "chunked_prefill_tokens",
        "min_prefix_tokens",
        "max_model_len",
        "max_num_seqs",
        "max_prefill_tokens",
        "gpu_memory_utilization",
        "kvstore_ratio",
        "seed",
        "writer_max_new_tokens",
        "suffix_max_new_tokens",
        "min_output_tokens",
        "max_token_mismatches",
        "logprob_atol",
        "min_put_success_keys",
        "min_put_success_bytes",
        "min_get_success_keys",
        "min_get_success_bytes",
        "min_l3_query_hit_tokens",
    }
    _keys(mapping, allowed, path)
    missing = sorted(allowed - set(mapping))
    if missing:
        _fail(path, f"missing required keys {missing}")
    extra_args = _string_list(mapping["extra_engine_args"], f"{path}.extra_engine_args")
    check_extra_engine_args(extra_args, f"{path}.extra_engine_args")
    page = _integer(
        mapping["page_granularity_tokens"], f"{path}.page_granularity_tokens", minimum=1
    )
    chunk = _integer(
        mapping["chunked_prefill_tokens"], f"{path}.chunked_prefill_tokens", minimum=1
    )
    if chunk % page != 0:
        _fail(
            path,
            f"chunked_prefill_tokens ({chunk}) must be a multiple of "
            f"page_granularity_tokens ({page}) so a chunk-aligned prefix is "
            "also page-aligned and fully publishable",
        )
    min_prefix = _integer(
        mapping["min_prefix_tokens"], f"{path}.min_prefix_tokens", minimum=chunk
    )
    max_prefill = _integer(
        mapping["max_prefill_tokens"], f"{path}.max_prefill_tokens", minimum=chunk
    )
    max_model_len = _integer(
        mapping["max_model_len"], f"{path}.max_model_len", minimum=min_prefix + 1
    )
    dtype = _string(mapping["dtype"], f"{path}.dtype", allow_empty=False)
    if dtype not in _DTYPES:
        _fail(path, f"dtype must be one of {_DTYPES}, got {dtype!r}")
    min_query_hit = _integer(
        mapping["min_l3_query_hit_tokens"],
        f"{path}.min_l3_query_hit_tokens",
        minimum=1,
    )
    suffix_max_new = _integer(
        mapping["suffix_max_new_tokens"], f"{path}.suffix_max_new_tokens", minimum=1
    )
    min_output = _integer(
        mapping["min_output_tokens"], f"{path}.min_output_tokens", minimum=1
    )
    if min_output > suffix_max_new:
        _fail(
            path,
            f"min_output_tokens ({min_output}) cannot exceed "
            f"suffix_max_new_tokens ({suffix_max_new}): the oracle can never "
            "produce enough tokens for a non-vacuous comparison",
        )
    return ModelCase(
        name=_string(mapping["name"], f"{path}.name", allow_empty=False),
        model_path=_string(
            mapping["model_path"], f"{path}.model_path", allow_empty=False
        ),
        dtype=dtype,
        extra_engine_args=extra_args,
        page_granularity_tokens=page,
        chunked_prefill_tokens=chunk,
        min_prefix_tokens=min_prefix,
        max_model_len=max_model_len,
        max_num_seqs=_integer(
            mapping["max_num_seqs"], f"{path}.max_num_seqs", minimum=1
        ),
        max_prefill_tokens=max_prefill,
        gpu_memory_utilization=_real(
            mapping["gpu_memory_utilization"],
            f"{path}.gpu_memory_utilization",
            minimum=0.0,
            below_one=True,
        ),
        kvstore_ratio=_real(
            mapping["kvstore_ratio"],
            f"{path}.kvstore_ratio",
            minimum=0.0,
            below_one=False,
        ),
        seed=_integer(mapping["seed"], f"{path}.seed", minimum=0),
        writer_max_new_tokens=_integer(
            mapping["writer_max_new_tokens"], f"{path}.writer_max_new_tokens", minimum=1
        ),
        suffix_max_new_tokens=suffix_max_new,
        min_output_tokens=min_output,
        max_token_mismatches=_integer(
            mapping["max_token_mismatches"], f"{path}.max_token_mismatches", minimum=0
        ),
        logprob_atol=_real(
            mapping["logprob_atol"],
            f"{path}.logprob_atol",
            minimum=0.0,
            below_one=False,
        ),
        min_put_success_keys=_integer(
            mapping["min_put_success_keys"], f"{path}.min_put_success_keys", minimum=1
        ),
        min_put_success_bytes=_integer(
            mapping["min_put_success_bytes"], f"{path}.min_put_success_bytes", minimum=1
        ),
        min_get_success_keys=_integer(
            mapping["min_get_success_keys"], f"{path}.min_get_success_keys", minimum=1
        ),
        min_get_success_bytes=_integer(
            mapping["min_get_success_bytes"], f"{path}.min_get_success_bytes", minimum=1
        ),
        min_l3_query_hit_tokens=min_query_hit,
    )


def parse_gate_config(raw: object, source: str) -> GateConfig:
    """Validate the parsed JSON object, or raise ``GateConfigError``."""
    path = f"{source}"
    mapping = _mapping(raw, path)
    allowed = {
        "schema_version",
        "tenant_prefix",
        "mooncake",
        "provider",
        "timeouts",
        "async_overrides",
        "cases",
    }
    _keys(mapping, allowed, path)
    missing = sorted(allowed - set(mapping))
    if missing:
        _fail(path, f"missing required keys {missing}")
    schema_version = _integer(
        mapping["schema_version"], f"{path}.schema_version", minimum=1
    )
    if schema_version != SCHEMA_VERSION:
        _fail(path, f"schema_version must be {SCHEMA_VERSION}, got {schema_version}")
    tenant_prefix = _string(
        mapping["tenant_prefix"], f"{path}.tenant_prefix", allow_empty=False
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", tenant_prefix):
        _fail(
            path,
            f"tenant_prefix must match [a-z0-9][a-z0-9-]*, got {tenant_prefix!r}",
        )
    cases = mapping["cases"]
    if not isinstance(cases, list) or not cases:
        _fail(f"{path}.cases", "must be a non-empty list")
    parsed_cases = tuple(
        _parse_case(case, f"{path}.cases[{index}]") for index, case in enumerate(cases)
    )
    names = [case.name for case in parsed_cases]
    if len(set(names)) != len(names):
        _fail(f"{path}.cases", f"case names must be unique, got {names}")
    return GateConfig(
        schema_version=schema_version,
        tenant_prefix=tenant_prefix,
        mooncake=_parse_connection(mapping["mooncake"], f"{path}.mooncake"),
        provider=_parse_provider(mapping["provider"], f"{path}.provider"),
        timeouts=_parse_timeouts(mapping["timeouts"], f"{path}.timeouts"),
        async_overrides=_parse_async_overrides(
            mapping["async_overrides"], f"{path}.async_overrides"
        ),
        cases=parsed_cases,
    )


def load_gate_config(path: str) -> GateConfig:
    """Read and validate the operator's JSON gate configuration."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as error:
        raise GateConfigError(f"cannot read gate config {path!r}: {error}") from error
    except json.JSONDecodeError as error:
        raise GateConfigError(
            f"gate config {path!r} is not valid JSON: {error}"
        ) from error
    return parse_gate_config(raw, path)


def requested_config_path(environ: Mapping[str, str]) -> str | None:
    """The gate config path the environment requests, or None to skip.

    Only an unset ``CONFIG_ENV_VAR`` skips the gate. Once set, the operator
    asked for the gate: a blank value is a configuration error, never a
    silent skip.
    """
    if CONFIG_ENV_VAR not in environ:
        return None
    path = environ[CONFIG_ENV_VAR]
    if not path.strip():
        raise GateConfigError(
            f"{CONFIG_ENV_VAR} is set but blank; name a gate config JSON or "
            "unset it to skip the gate"
        )
    return path


def max_prefetch_tokens(config: GateConfig) -> int:
    """The L3 prefetch cap the config gives the engines."""
    return dict(config.async_overrides)["max_prefetch_tokens"]


def run_tenant_id(tenant_prefix: str, run_token: str) -> str:
    """The store tenant for one gate run: isolates the run's objects from
    every other run sharing the master, so PUT success means new objects."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", tenant_prefix):
        raise ValueError(f"invalid tenant prefix {tenant_prefix!r}")
    if not run_token:
        raise ValueError("run token must not be empty")
    return f"{tenant_prefix}-{run_token}"


def mooncake_extra_config_json(
    connection: MooncakeConnection,
    async_overrides: tuple[tuple[str, int], ...],
    tenant_id: str,
    master_address: str,
) -> str:
    """The ``--kvstore-storage-backend-extra-config`` payload for the engines.

    ``global_segment_size`` is 0: inference workers mount no storage of their
    own, so the objects outlive them on the independent provider.
    """
    return json.dumps(
        {
            "master_server_address": master_address,
            "local_hostname": connection.local_hostname,
            "metadata_server": connection.metadata_server,
            "protocol": connection.protocol,
            "device_name": connection.device_name,
            "global_segment_size": 0,
            "tenant_id": tenant_id,
            "async": dict(async_overrides),
        }
    )


def serving_engine_argv(
    case: ModelCase,
    connection: MooncakeConnection,
    async_overrides: tuple[tuple[str, int], ...],
    tenant_id: str,
    master_address: str,
    metrics_port: int,
) -> list[str]:
    """The writer's and the fresh reader's identical engine command line.

    The two engines must resolve the same storage namespace, so every
    identity-relevant argument comes from the same case fields; only the
    metrics port differs between the two processes. Output logprobs are a
    static server flag: without ``--enable-output-logprobs`` a request's
    ``return_logprob`` yields empty logprobs.
    """
    return [
        "--model",
        case.model_path,
        "--attn-tp-size",
        "1",
        "--disaggregation-mode",
        "null",
        "--dtype",
        case.dtype,
        "--seed",
        str(case.seed),
        "--enable-prefix-caching",
        "--max-model-len",
        str(case.max_model_len),
        "--max-num-seqs",
        str(case.max_num_seqs),
        "--max-prefill-tokens",
        str(case.max_prefill_tokens),
        "--chunked-prefill-size",
        str(case.chunked_prefill_tokens),
        "--gpu-memory-utilization",
        str(case.gpu_memory_utilization),
        "--kvstore-ratio",
        str(case.kvstore_ratio),
        "--kvstore-storage-backend",
        "mooncake",
        "--kvstore-storage-backend-extra-config",
        mooncake_extra_config_json(
            connection, async_overrides, tenant_id, master_address
        ),
        "--enable-metrics",
        "--metrics-port",
        str(metrics_port),
        "--enable-output-logprobs",
        "--disable-autotune",
        *case.extra_engine_args,
    ]


def oracle_engine_argv(case: ModelCase) -> list[str]:
    """The recomputation oracle's engine command line.

    Same model, dtype, seed, chunking and numerics-relevant extra arguments
    as the serving engines, but with every cache tier off: the oracle
    recomputes the whole prompt on every request and never touches the store.
    """
    return [
        "--model",
        case.model_path,
        "--attn-tp-size",
        "1",
        "--disaggregation-mode",
        "null",
        "--dtype",
        case.dtype,
        "--seed",
        str(case.seed),
        "--disable-prefix-caching",
        "--disable-kvstore",
        "--max-model-len",
        str(case.max_model_len),
        "--max-num-seqs",
        str(case.max_num_seqs),
        "--max-prefill-tokens",
        str(case.max_prefill_tokens),
        "--chunked-prefill-size",
        str(case.chunked_prefill_tokens),
        "--gpu-memory-utilization",
        str(case.gpu_memory_utilization),
        "--enable-output-logprobs",
        "--disable-autotune",
        *case.extra_engine_args,
    ]


def provider_command(
    python_executable: str,
    script_path: str,
    connection: MooncakeConnection,
    tenant_id: str,
    master_address: str,
    segment_bytes: int,
    max_seconds: float,
) -> list[str]:
    """The command line of the independent segment-provider subprocess."""
    return [
        python_executable,
        script_path,
        "--l3-provider",
        "--master",
        master_address,
        "--tenant-id",
        tenant_id,
        "--segment-bytes",
        str(segment_bytes),
        "--protocol",
        connection.protocol,
        "--device-name",
        connection.device_name,
        "--metadata-server",
        connection.metadata_server,
        "--local-hostname",
        connection.local_hostname,
        "--max-seconds",
        str(max_seconds),
    ]


def provider_lifetime_seconds(timeouts: TimeoutConfig, num_cases: int) -> float:
    """A backstop self-exit deadline for the provider process.

    The parent always stops the provider itself; this only bounds how long a
    leaked provider can outlive a dead test process.
    """
    if num_cases < 1:
        raise ValueError(f"num_cases must be >= 1, got {num_cases}")
    per_case = (
        3 * timeouts.engine_startup_seconds
        + 3 * timeouts.request_seconds
        + 3 * timeouts.put_evidence_seconds
        + 300.0
    )
    return per_case * num_cases + 600.0


def metrics_url(metrics_port: int) -> str:
    """The engine's Prometheus endpoint, bound on the loopback host."""
    return f"http://127.0.0.1:{metrics_port}/metrics"


# ---------------------------------------------------------------------------
# Prometheus exposition parsing and evidence snapshots
# ---------------------------------------------------------------------------

# (metric name, sorted (label, value) pairs) -> sample value.
MetricsMap = dict[tuple[str, tuple[tuple[str, str], ...]], float]

_SERIES_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{((?:[^"}]|"(?:[^"\\]|\\.)*")*)\}'
    r"\s+(\S+)(?:\s+\S+)?$"
)
_PLAIN_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(\S+)(?:\s+\S+)?$")
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_LABELS_FULL_RE = re.compile(
    r'[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*"'
    r'(,[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*")*'
)


def parse_prometheus_text(text: str) -> MetricsMap:
    """Parse a Prometheus text exposition into a sample map.

    Comment and empty lines are skipped; any other malformed line is a
    parse error, because unreadable evidence must fail the gate rather than
    silently read as zero. ``NaN``/``Inf`` sample values parse as floats:
    an unrelated series may legitimately carry them, and only the series
    the gate reads are checked for finiteness.
    """
    metrics: MetricsMap = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        series = _SERIES_RE.match(line)
        if series is not None:
            name, label_text, value_text = series.groups()
            if label_text and _LABELS_FULL_RE.fullmatch(label_text) is None:
                raise ValueError(
                    f"metrics line {number} has malformed labels: {label_text!r}"
                )
            labels = tuple(sorted(_LABEL_RE.findall(label_text)))
        else:
            plain = _PLAIN_RE.match(line)
            if plain is None:
                raise ValueError(
                    f"metrics line {number} is not a valid sample: {line!r}"
                )
            name, value_text = plain.groups()
            labels = ()
        try:
            value = float(value_text)
        except ValueError as error:
            raise ValueError(
                f"metrics line {number} has a non-numeric value: {value_text!r}"
            ) from error
        metrics[(name, labels)] = value
    return metrics


def _select(
    metrics: MetricsMap, name: str, labels: tuple[tuple[str, str], ...]
) -> list[tuple[tuple[tuple[str, str], ...], float]]:
    """``name``'s samples carrying all of ``labels``."""
    required = dict(labels)
    selected = []
    for (sample_name, sample_labels), value in metrics.items():
        if sample_name != name:
            continue
        present = dict(sample_labels)
        if all(present.get(key) == wanted for key, wanted in required.items()):
            selected.append((sample_labels, value))
    return selected


@dataclass(frozen=True)
class EvidenceSnapshot:
    """The gate's evidence series from one scrape, summed over the rank and
    model label values (one attention-TP rank in this gate's scope, so the
    scheduler-mirrored gauges are not summed across ranks).

    Counters are cumulative and read 0 before their first increment.
    Gauges are None when absent from the scrape, which proves nothing.
    ``nonfinite_series`` lists every read sample whose value is NaN or
    infinite; such a snapshot is not evidence.
    """

    put_success_keys: float
    put_failed_keys: float
    put_success_bytes: float
    get_success_keys: float
    get_failed_keys: float
    get_success_bytes: float
    dropped_backup_keys: float
    l3_query_hit_tokens: float
    # Executor jobs by kind, every state: queued, running, and completed
    # but not yet polled by the scheduler.
    put_jobs: float | None
    get_jobs: float | None
    storage_operations: float | None
    backup_candidates: float | None
    quarantined_blocks: float | None
    io_quarantined: float | None
    nonfinite_series: tuple[str, ...]


def snapshot_from_prometheus(metrics: MetricsMap) -> EvidenceSnapshot:
    """Extract the evidence series from a parsed exposition."""
    values: dict[str, float | None] = {}
    nonfinite: list[str] = []
    for field, (name, labels) in {**_COUNTERS, **_GAUGES}.items():
        samples = _select(metrics, name, labels)
        for sample_labels, value in samples:
            if not math.isfinite(value):
                rendered = ",".join(f'{key}="{val}"' for key, val in sample_labels)
                nonfinite.append(f"{name}{{{rendered}}}={value}")
        if not samples and field in _GAUGES:
            values[field] = None
        else:
            values[field] = sum(value for _, value in samples)
    return EvidenceSnapshot(nonfinite_series=tuple(nonfinite), **values)


def snapshot_from_exposition(text: str) -> EvidenceSnapshot:
    """Parse and extract in one step (what the gate does per scrape)."""
    return snapshot_from_prometheus(parse_prometheus_text(text))


def missing_gauges(snapshot: EvidenceSnapshot) -> list[str]:
    """The mandatory gauge series a scrape did not carry."""
    missing = []
    for field, (name, labels) in _GAUGES.items():
        if getattr(snapshot, field) is None:
            rendered = ",".join(f'{key}="{value}"' for key, value in labels)
            missing.append(f"{name}{{{rendered}}}" if rendered else name)
    return missing


@dataclass(frozen=True)
class EvidenceThresholds:
    """The minimum evidence a phase must produce, from the model case."""

    min_put_success_keys: float
    min_put_success_bytes: float
    min_get_success_keys: float
    min_get_success_bytes: float
    min_l3_query_hit_tokens: float


def thresholds_from_case(case: ModelCase) -> EvidenceThresholds:
    """The case's evidence bounds as the comparator takes them."""
    return EvidenceThresholds(
        min_put_success_keys=float(case.min_put_success_keys),
        min_put_success_bytes=float(case.min_put_success_bytes),
        min_get_success_keys=float(case.min_get_success_keys),
        min_get_success_bytes=float(case.min_get_success_bytes),
        min_l3_query_hit_tokens=float(case.min_l3_query_hit_tokens),
    )


def readability_violations(snapshot: EvidenceSnapshot, label: str) -> list[str]:
    """A scrape the evidence checks cannot trust: absent gauges or
    non-finite values."""
    violations = []
    missing = missing_gauges(snapshot)
    if missing:
        violations.append(
            f"{label}: mandatory L3 gauges absent from the scrape {missing}; "
            "storage metrics have not been exported yet, so drain and "
            "quarantine health are unproven"
        )
    if snapshot.nonfinite_series:
        violations.append(
            f"{label}: non-finite evidence values {list(snapshot.nonfinite_series)}"
        )
    return violations


def _quarantine_violations(snapshot: EvidenceSnapshot, label: str) -> list[str]:
    violations = []
    if snapshot.io_quarantined not in (None, 0.0):
        violations.append(
            f"{label}: L3 I/O quarantined ({L3_IO_QUARANTINED} is "
            f"{snapshot.io_quarantined}): native retirement unproven, new I/O "
            "disabled until restart"
        )
    if snapshot.quarantined_blocks not in (None, 0.0):
        violations.append(
            f"{label}: {snapshot.quarantined_blocks} Host blocks quarantined "
            "after unproven retirement (scheduler resource quarantined)"
        )
    return violations


def evaluate_writer_evidence(
    snapshot: EvidenceSnapshot, thresholds: EvidenceThresholds
) -> list[str]:
    """Violations in the writer's PUT evidence; empty means the scrape shows
    the prefix persisted and nothing left to publish or back up.

    One scrape cannot prove a later publication will not appear; the caller
    requires this to hold unchanged across several storage-metrics exports,
    and the fresh reader's exact admitted prefix is the final proof.
    """
    violations = readability_violations(snapshot, "writer")
    if violations:
        return violations
    violations += _quarantine_violations(snapshot, "writer")
    if snapshot.put_success_keys < thresholds.min_put_success_keys:
        violations.append(
            "writer PUT evidence missing: "
            f"{L3_IO_KEYS}{{kind=put, result=success}} is "
            f"{snapshot.put_success_keys} < {thresholds.min_put_success_keys}; "
            "the prefix was not persisted to the store"
        )
    if snapshot.put_success_bytes < thresholds.min_put_success_bytes:
        violations.append(
            "writer PUT byte evidence missing: "
            f"{L3_IO_ADAPTER_BYTES}{{kind=put}} is {snapshot.put_success_bytes} "
            f"< {thresholds.min_put_success_bytes}"
        )
    if snapshot.put_failed_keys != 0:
        violations.append(
            f"writer PUT reported {snapshot.put_failed_keys} failed keys: part "
            "of the prefix may be missing from the store"
        )
    if snapshot.dropped_backup_keys != 0:
        violations.append(
            f"writer dropped {snapshot.dropped_backup_keys} backup candidates "
            f"({L3_DROPPED_BACKUP_KEYS}): published blocks were never written"
        )
    if snapshot.put_jobs != 0:
        violations.append(
            f"writeback not drained: {snapshot.put_jobs} PUT jobs queued, "
            "running or not yet polled"
        )
    if snapshot.storage_operations != 0:
        violations.append(
            f"writeback not drained: the scheduler has "
            f"{snapshot.storage_operations} storage operations in flight"
        )
    if snapshot.backup_candidates != 0:
        violations.append(
            f"writeback not drained: {snapshot.backup_candidates} published "
            "blocks still wait for backup (scheduler backup_candidates)"
        )
    return violations


def evaluate_reader_restore(
    before: EvidenceSnapshot,
    after: EvidenceSnapshot,
    thresholds: EvidenceThresholds,
    expected_cached_tokens: int,
    cached_tokens: int,
) -> list[str]:
    """Violations in the fresh reader's first-request restore evidence,
    from the scrapes before and after that request.

    Reuse itself is proven by ``cached_tokens`` equal to the whole persisted
    prefix on an engine that served nothing before, together with positive
    GET deltas (an exact hit without store reads would be a warm local
    cache). The query-hit delta is supporting evidence that the store held
    the reader's keys.
    """
    violations = readability_violations(before, "reader before the probe")
    violations += readability_violations(after, "reader after the probe")
    if violations:
        return violations
    violations += _quarantine_violations(after, "reader")
    if cached_tokens != expected_cached_tokens:
        violations.append(
            f"the probe reused {cached_tokens} cached tokens, expected exactly "
            f"{expected_cached_tokens} (the whole persisted prefix): fewer "
            "means a partial restore or an L3 miss, more means objects "
            "outside this run's prefix"
        )
    get_keys = after.get_success_keys - before.get_success_keys
    if get_keys < thresholds.min_get_success_keys:
        violations.append(
            "reader GET evidence missing: "
            f"{L3_IO_KEYS}{{kind=get, result=success}} delta is {get_keys} < "
            f"{thresholds.min_get_success_keys}; the fresh reader did not read "
            "objects from the store, so no cross-instance restore happened"
        )
    get_bytes = after.get_success_bytes - before.get_success_bytes
    if get_bytes < thresholds.min_get_success_bytes:
        violations.append(
            "reader GET byte evidence missing: "
            f"{L3_IO_ADAPTER_BYTES}{{kind=get}} delta is {get_bytes} < "
            f"{thresholds.min_get_success_bytes}"
        )
    get_failed = after.get_failed_keys - before.get_failed_keys
    if get_failed != 0:
        violations.append(
            f"reader GET reported {get_failed} failed keys: a partial restore "
            "cannot be the whole persisted prefix"
        )
    query_hits = after.l3_query_hit_tokens - before.l3_query_hit_tokens
    if query_hits < thresholds.min_l3_query_hit_tokens:
        violations.append(
            f"reader storage QUERY found too few tokens: {PREFIX_L3_QUERY_HIT_TOKENS} "
            f"delta is {query_hits} < {thresholds.min_l3_query_hit_tokens}; the "
            "store did not hold the persisted prefix's keys under the reader's "
            "namespace (tenant, model, layout or page size differ, or the "
            "objects were lost)"
        )
    return violations


def evaluate_negative_control(
    before: EvidenceSnapshot, after: EvidenceSnapshot, cached_tokens: int
) -> list[str]:
    """Violations in the disjoint negative-control request: it must recompute
    fully, proving the positive restore above is selective."""
    violations = readability_violations(before, "reader before the control")
    violations += readability_violations(after, "reader after the control")
    if violations:
        return violations
    violations += _quarantine_violations(after, "negative control")
    if cached_tokens != 0:
        violations.append(
            f"negative control reused {cached_tokens} cached tokens: a prompt "
            "disjoint from the persisted prefix must recompute"
        )
    get_keys = after.get_success_keys - before.get_success_keys
    if get_keys != 0:
        violations.append(
            f"negative control read {get_keys} store objects: L3 restores are "
            "not selective for the persisted prefix"
        )
    query_hits = after.l3_query_hit_tokens - before.l3_query_hit_tokens
    if query_hits != 0:
        violations.append(
            f"negative control's storage QUERY found {query_hits} tokens: keys "
            "of a disjoint prompt must not be in the store"
        )
    return violations


def evaluate_completion_finish(finish_reason: str | None, label: str) -> list[str]:
    """A request that aborted or was refused invalidates its phase."""
    if finish_reason in _FINISH_OK:
        return []
    return [
        f"{label}: finish_reason is {finish_reason!r}, expected one of "
        f"{_FINISH_OK}; the engine aborted or refused the request"
    ]


def _nonfinite_positions(values: list[float]) -> list[int]:
    return [
        index
        for index, value in enumerate(values)
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ]


def compare_outputs(
    reference_ids: list[int],
    reference_logprobs: list[float],
    candidate_ids: list[int],
    candidate_logprobs: list[float],
    max_token_mismatches: int,
    logprob_atol: float,
    min_output_tokens: int,
) -> list[str]:
    """Violations between the recompute oracle and the restored reader.

    Greedy continuations must agree token for token within
    ``max_token_mismatches`` and per-token logprobs within ``logprob_atol``.
    Every logprob must be finite: a sampled token's logprob is never NaN or
    infinite, and NaN would otherwise slip through the tolerance test. A
    reference shorter than ``min_output_tokens`` makes the comparison vacuous
    and is a violation.
    """
    violations = []
    if len(reference_ids) < min_output_tokens:
        violations.append(
            f"the oracle produced only {len(reference_ids)} tokens "
            f"(< min_output_tokens {min_output_tokens}): the comparison is "
            "vacuous for this case; adjust the suffix or its budget"
        )
    if len(candidate_ids) != len(reference_ids):
        violations.append(
            f"output length differs: reader {len(candidate_ids)} vs oracle "
            f"{len(reference_ids)}"
        )
    mismatches = [
        index
        for index, (expected, actual) in enumerate(zip(reference_ids, candidate_ids))
        if expected != actual
    ]
    if len(mismatches) > max_token_mismatches:
        violations.append(
            f"{len(mismatches)} output tokens differ (allowed "
            f"{max_token_mismatches}); first mismatches at positions "
            f"{mismatches[:5]}: "
            + ", ".join(
                f"[{index}] oracle={reference_ids[index]} reader={candidate_ids[index]}"
                for index in mismatches[:5]
            )
        )
    for side, ids, logprobs in (
        ("oracle", reference_ids, reference_logprobs),
        ("reader", candidate_ids, candidate_logprobs),
    ):
        if len(logprobs) != len(ids):
            violations.append(
                f"the {side} returned {len(logprobs)} logprobs for {len(ids)} "
                "tokens; output logprobs must cover every sampled token (the "
                "request needs return_logprob and the engine "
                "--enable-output-logprobs)"
            )
        bad = _nonfinite_positions(logprobs)
        if bad:
            violations.append(
                f"the {side} returned non-finite logprobs at output positions "
                f"{bad[:5]}: " + ", ".join(f"[{i}]={logprobs[i]!r}" for i in bad[:5])
            )
    reference_bad = set(_nonfinite_positions(reference_logprobs))
    candidate_bad = set(_nonfinite_positions(candidate_logprobs))
    for index in range(min(len(reference_logprobs), len(candidate_logprobs))):
        if index in reference_bad or index in candidate_bad:
            continue
        difference = abs(reference_logprobs[index] - candidate_logprobs[index])
        if difference > logprob_atol:
            violations.append(
                f"logprob at output position {index} differs by {difference} "
                f"(> atol {logprob_atol}): oracle "
                f"{reference_logprobs[index]} vs reader {candidate_logprobs[index]}"
            )
            break
    return violations


def evaluate_codeword_answers(
    codeword: str, oracle_text: str, reader_text: str, control_text: str
) -> list[str]:
    """Violations of the prefix-dependence oracle.

    The codeword is random per run and appears only in the persisted
    prefix, so an answer containing it can only come from attending to that
    prefix. Without this, an oracle and a reader that both ignore the prefix
    would agree and pass the output comparison.
    """
    if not codeword:
        raise ValueError("codeword must not be empty")
    violations = []
    if codeword not in oracle_text:
        violations.append(
            f"the oracle's answer {oracle_text!r} does not contain the run's "
            f"codeword {codeword!r}: the output does not depend on the prefix, "
            "so agreeing with it proves nothing; pick a model/suffix that "
            "answers the question"
        )
    if codeword not in reader_text:
        violations.append(
            f"the restored reader's answer {reader_text!r} does not contain the "
            f"run's codeword {codeword!r}: the restored prefix was not used"
        )
    if codeword in control_text:
        violations.append(
            f"the negative control's answer {control_text!r} contains the "
            "codeword: the disjoint prompt saw the persisted prefix"
        )
    return violations


# ---------------------------------------------------------------------------
# Prompt construction (junction-safe: the engines get token-id concatenations)
# ---------------------------------------------------------------------------

_FILLER_SENTENCES = (
    "The northern archive holds ledgers from the coastal trade years.",
    "Every carton is stamped with its arrival week and the clerk's initials.",
    "The reading room opens after the morning inventory is signed.",
    "Older maps are stored flat between sheets of acid-free paper.",
    "The bindery repairs spines in the order the requests arrived.",
    "Visitors sign the register before handling any unbound material.",
    "The annex keeps the duplicate reels behind the second door.",
    "Humidity is checked twice a day and written into the logbook.",
    "Requests for the restricted shelf need two staff signatures.",
    "The cataloguer marks every gap in the sequence with a blue tag.",
    "Microfilm readers are serviced on the first Monday of the month.",
    "The south stairwell stays closed until the inspection ends.",
    "Each returned box is weighed before it goes back to the stacks.",
    "The conservation bench keeps its tools in numbered trays.",
    "Evening staff walk the aisles once before locking the doors.",
    "New acquisitions wait a full week before they enter the catalog.",
)


def fact_prefix_text(codeword: str, filler_repeats: int) -> str:
    """The persisted prefix: an archival memorandum carrying the codeword in
    its header, followed by ``filler_repeats`` numbered filler sections.

    The suffix question asks for the codeword, so a correct answer needs the
    prefix's facts; the codeword is unique per gate run, so no prior run or
    training data can supply it.
    """
    if not codeword:
        raise ValueError("codeword must not be empty")
    if filler_repeats < 1:
        raise ValueError(f"filler_repeats must be >= 1, got {filler_repeats}")
    header = (
        "Operations memorandum of the regional archive, distribution copy.\n"
        f"The project codeword is {codeword}.\n"
        "Every section below belongs to the same archival record.\n"
    )
    sections = []
    for index in range(filler_repeats):
        sentences = (
            _FILLER_SENTENCES[(3 * index) % len(_FILLER_SENTENCES)],
            _FILLER_SENTENCES[(3 * index + 1) % len(_FILLER_SENTENCES)],
            _FILLER_SENTENCES[(3 * index + 2) % len(_FILLER_SENTENCES)],
        )
        sections.append(f"Section {index + 1}. " + " ".join(sentences))
    return header + "\n".join(sections) + "\n"


def suffix_question_text() -> str:
    """The suffix: answerable only with the prefix's codeword. It opens with
    a newline because the prefix may end mid-sentence (see
    ``aligned_prefix_ids``)."""
    return (
        "\nQuestion: what is the project codeword? "
        "Answer with the codeword only, then stop.\nAnswer:"
    )


def negative_control_text() -> str:
    """A prompt disjoint from the prefix from its first token on."""
    return (
        "Kitchen notebook for a weekday lentil soup.\n"
        "Rinse two cups of green lentils and dice one onion, two carrots and "
        "two celery stalks. Soften the vegetables in olive oil, add the "
        "lentils, a bay leaf and six cups of water, then simmer gently.\n"
        "Question: how long should the soup simmer?\nAnswer:"
    )


def aligned_prefix_length(min_prefix_tokens: int, chunk_tokens: int) -> int:
    """The smallest whole number of prefill chunks (hence of cache pages)
    holding at least ``min_prefix_tokens``."""
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be >= 1, got {chunk_tokens}")
    if min_prefix_tokens < 1:
        raise ValueError(f"min_prefix_tokens must be >= 1, got {min_prefix_tokens}")
    return -(-min_prefix_tokens // chunk_tokens) * chunk_tokens


def aligned_prefix_ids(
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    codeword: str,
    target_tokens: int,
) -> list[int]:
    """The persisted prefix's token ids, exactly ``target_tokens`` long.

    Whole filler sections are added (doubling, so O(log target) encodes)
    until the document reaches the target, and the ids are then cut to it.
    Waiting for a section count whose length happens to be a whole number of
    chunks may never terminate; the cut always does, and is deterministic for
    a given tokenizer and codeword. It may end the prefix mid-sentence or
    mid-word, which is why the suffix opens on a new line; the codeword sits
    in the header, so it survives the cut (checked by decoding). No pad
    tokens are added, so the prefix stays natural text.
    """
    if target_tokens < 1:
        raise ValueError(f"target_tokens must be >= 1, got {target_tokens}")
    repeats = 1
    while True:
        ids = encode(fact_prefix_text(codeword, repeats))
        if len(ids) >= target_tokens:
            break
        if repeats >= target_tokens:
            # Every section adds at least one token for any real tokenizer,
            # so target_tokens sections always reach the target.
            raise ValueError(
                f"{repeats} filler sections encode to only {len(ids)} tokens "
                f"(< {target_tokens}); the tokenizer does not grow with the text"
            )
        repeats = min(target_tokens, 2 * repeats)
    prefix = list(ids[:target_tokens])
    if codeword not in decode(prefix):
        raise ValueError(
            f"the codeword does not survive the cut to {target_tokens} tokens; "
            "raise min_prefix_tokens"
        )
    return prefix


def check_token_budgets(
    case: ModelCase,
    prefetch_cap_tokens: int,
    prefix_tokens: int,
    suffix_tokens: int,
    control_tokens: int,
) -> None:
    """Raise ``GateConfigError`` when the prompts cannot run as designed.

    A truncated prompt invalidates the output comparison; a prefix beyond
    the prefetch cap can only be restored partially; a suffix longer than
    one chunk makes the reader's extend longer than the oracle's last
    prefill chunk.
    """
    if prefix_tokens % case.chunked_prefill_tokens != 0:
        raise GateConfigError(
            f"[{case.name}] the prefix is {prefix_tokens} tokens, not a whole "
            f"number of chunked_prefill_tokens ({case.chunked_prefill_tokens})"
        )
    if prefix_tokens > prefetch_cap_tokens:
        raise GateConfigError(
            f"[{case.name}] the prefix ({prefix_tokens} tokens) exceeds "
            f"async_overrides.max_prefetch_tokens ({prefetch_cap_tokens}): "
            "the reader could restore it only partially"
        )
    if suffix_tokens > case.chunked_prefill_tokens:
        raise GateConfigError(
            f"[{case.name}] the suffix is {suffix_tokens} tokens, longer than "
            f"chunked_prefill_tokens ({case.chunked_prefill_tokens}): the "
            "reader's extend must be a single chunk to match the oracle's "
            "last prefill chunk"
        )
    probe_budget = prefix_tokens + suffix_tokens + case.suffix_max_new_tokens
    if probe_budget > case.max_model_len:
        raise GateConfigError(
            f"[{case.name}] prefix ({prefix_tokens}) + suffix ({suffix_tokens}) "
            f"+ {case.suffix_max_new_tokens} new tokens exceeds max_model_len "
            f"({case.max_model_len}): raise max_model_len or shorten the prefix"
        )
    writer_budget = prefix_tokens + case.writer_max_new_tokens
    if writer_budget > case.max_model_len:
        raise GateConfigError(
            f"[{case.name}] prefix ({prefix_tokens}) + "
            f"{case.writer_max_new_tokens} writer tokens exceeds max_model_len "
            f"({case.max_model_len})"
        )
    control_budget = control_tokens + case.suffix_max_new_tokens
    if control_budget > case.max_model_len:
        raise GateConfigError(
            f"[{case.name}] negative control ({control_tokens}) + "
            f"{case.suffix_max_new_tokens} new tokens exceeds max_model_len "
            f"({case.max_model_len})"
        )


# ---------------------------------------------------------------------------
# Bounded calls and owned child processes
# ---------------------------------------------------------------------------


def run_bounded(call: Callable[[], object], timeout_s: float, label: str) -> object:
    """``call()``'s result, or ``TimeoutError`` after ``timeout_s``.

    The call runs on a daemon thread: a call that never returns (an engine
    that wedges without dying keeps its output wait open) cannot keep the
    interpreter alive at exit, unlike a ``ThreadPoolExecutor`` worker,
    which is joined at exit. The caller then stops whatever the call waits
    on. An exception from the call is re-raised here.
    """
    outcome: queue.Queue = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            outcome.put((True, call()))
        except BaseException as error:  # noqa: BLE001 - re-raised by the caller
            outcome.put((False, error))

    threading.Thread(target=target, name=f"bounded {label}", daemon=True).start()
    try:
        succeeded, value = outcome.get(timeout=timeout_s)
    except queue.Empty:
        raise TimeoutError(f"{label}: no result within {timeout_s}s") from None
    if not succeeded:
        raise value
    return value


@contextlib.contextmanager
def stop_on_exit(stop: Callable[[], str], label: str) -> Iterator[None]:
    """Run ``stop`` when the block exits, without masking the block's error.

    On success a failing ``stop`` raises. When the block raised, its error
    stays the one that propagates: a failing ``stop`` and any diagnostics
    ``stop`` returns (such as a child's log) are attached to it as notes.
    """
    try:
        yield
    except BaseException as primary:
        try:
            diagnostics = stop()
        except BaseException as cleanup_error:  # noqa: BLE001 - kept as a note
            primary.add_note(f"stopping {label} also failed: {cleanup_error!r}")
        else:
            if diagnostics:
                primary.add_note(f"{label} on cleanup: {diagnostics}")
        raise
    stop()


@dataclass
class ProviderProcess:
    """The segment-provider child and the file holding its stdout/stderr."""

    process: subprocess.Popen
    log: IO[bytes]
    # Set by the first stop; later stops return it unchanged.
    stopped: StoppedProcess | None


@dataclass(frozen=True)
class StoppedProcess:
    """A reaped child: its exit status and everything it logged."""

    returncode: int
    log: str


def start_provider_process(command: list[str]) -> ProviderProcess:
    """Start the provider with stdout and stderr in an anonymous temporary
    file, never a pipe: nothing reads a pipe while the provider runs, and a
    full pipe would block the store client's logging threads."""
    log = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log
        )
    except BaseException:
        log.close()
        raise
    return ProviderProcess(process=process, log=log, stopped=None)


def provider_log_text(provider: ProviderProcess) -> str:
    """Everything the provider has logged so far. ``pread`` leaves the file
    offset the child writes at untouched."""
    if provider.stopped is not None:
        return provider.stopped.log
    descriptor = provider.log.fileno()
    size = os.fstat(descriptor).st_size
    return os.pread(descriptor, size, 0).decode(errors="replace")


def await_provider_ready(
    provider: ProviderProcess, ready_token: str, timeout_s: float, poll_s: float
) -> None:
    """Wait until the provider logs ``ready_token`` on a line of its own.

    Raises when it exits first or the deadline passes. It does not stop the
    provider: the caller's ``stop_on_exit`` owns that and attaches the log.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if ready_token in (
            line.strip() for line in provider_log_text(provider).splitlines()
        ):
            return
        returncode = provider.process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"the L3 segment provider exited with {returncode} before "
                "mounting its segment"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"the L3 segment provider did not mount its segment within "
                f"{timeout_s}s"
            )
        time.sleep(poll_s)


def stop_provider_process(provider: ProviderProcess, grace_s: float) -> StoppedProcess:
    """SIGTERM, up to ``grace_s`` to exit, then SIGKILL; reaps the child.

    Signals reach this child's pid only, never a process group or a name
    match. Idempotent: later calls return the first stop's result.
    """
    if provider.stopped is not None:
        return provider.stopped
    process = provider.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()
    log = provider_log_text(provider)
    provider.log.close()
    provider.stopped = StoppedProcess(returncode=process.returncode, log=log)
    return provider.stopped
