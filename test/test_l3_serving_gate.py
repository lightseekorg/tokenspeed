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

"""CPU contract tests of the native Mooncake serving gate's shared logic.

These exercise the real config loader, engine command-line builders,
Prometheus evidence extraction and the evidence/output comparators of
``test/l3_serving_test_utils.py`` -- including the false-positive modes the
gate exists to reject (GET traffic without an admitted hit, an admitted hit
without GET evidence, an unselective hit, a vacuous output comparison). No
GPU, torch, runtime package or store library is needed.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from test.l3_serving_test_utils import (
    EvidenceSnapshot,
    GateConfigError,
    aligned_prefix_ids,
    aligned_prefix_length,
    await_provider_ready,
    check_extra_engine_args,
    check_token_budgets,
    compare_outputs,
    evaluate_codeword_answers,
    evaluate_completion_finish,
    evaluate_negative_control,
    evaluate_reader_restore,
    evaluate_writer_evidence,
    fact_prefix_text,
    load_gate_config,
    metrics_url,
    missing_gauges,
    mooncake_extra_config_json,
    negative_control_text,
    oracle_engine_argv,
    parse_gate_config,
    parse_prometheus_text,
    provider_command,
    provider_lifetime_seconds,
    requested_config_path,
    run_bounded,
    run_tenant_id,
    serving_engine_argv,
    snapshot_from_exposition,
    snapshot_from_prometheus,
    start_provider_process,
    stop_on_exit,
    stop_provider_process,
    suffix_question_text,
    thresholds_from_case,
)

import pytest

_EXAMPLE_CONFIG = (
    Path(__file__).resolve().parent / "runtime" / "l3_serving_e2e_config.example.json"
)


def _valid_config_dict() -> dict:
    return {
        "schema_version": 1,
        "tenant_prefix": "l3serving",
        "mooncake": {
            "protocol": "tcp",
            "device_name": "",
            "metadata_server": "P2PHANDSHAKE",
            "local_hostname": "localhost",
        },
        "provider": {"segment_bytes": 268435456},
        "timeouts": {
            "engine_startup_seconds": 3600,
            "request_seconds": 600,
            "put_evidence_seconds": 300,
            "metrics_poll_interval_seconds": 2,
        },
        "async_overrides": {
            "wait_timeout_ms": 30000,
            "ready_timeout_ms": 5000,
            "max_prefetch_tokens": 2048,
        },
        "cases": [
            {
                "name": "dense",
                "model_path": "example/dense-model",
                "dtype": "bfloat16",
                "extra_engine_args": ["--attention-backend", "example"],
                "page_granularity_tokens": 64,
                "chunked_prefill_tokens": 512,
                "min_prefix_tokens": 1024,
                "max_model_len": 4096,
                "max_num_seqs": 4,
                "max_prefill_tokens": 512,
                "gpu_memory_utilization": 0.7,
                "kvstore_ratio": 2.0,
                "seed": 42,
                "writer_max_new_tokens": 4,
                "suffix_max_new_tokens": 24,
                "min_output_tokens": 8,
                "max_token_mismatches": 0,
                "logprob_atol": 0.0001,
                "min_put_success_keys": 4,
                "min_put_success_bytes": 4096,
                "min_get_success_keys": 4,
                "min_get_success_bytes": 4096,
                "min_l3_query_hit_tokens": 1024,
            }
        ],
    }


def _config_with(mutate) -> dict:
    raw = _valid_config_dict()
    mutate(raw)
    return raw


def _expect_config_error(raw: dict, fragment: str) -> None:
    with pytest.raises(GateConfigError, match=fragment):
        parse_gate_config(raw, "test-config")


# ---------------------------------------------------------------------------
# Configuration loading and validation
# ---------------------------------------------------------------------------


def test_example_config_loads_through_the_real_loader():
    config = load_gate_config(str(_EXAMPLE_CONFIG))
    assert config.schema_version == 1
    assert config.tenant_prefix
    assert len(config.cases) >= 1
    case = config.cases[0]
    assert case.chunked_prefill_tokens % case.page_granularity_tokens == 0
    assert dict(config.async_overrides)["wait_timeout_ms"] >= 1000


def test_missing_config_file_fails():
    with pytest.raises(GateConfigError, match="cannot read gate config"):
        load_gate_config("/nonexistent/l3-serving-gate.json")


def test_invalid_json_fails(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(GateConfigError, match="not valid JSON"):
        load_gate_config(str(path))


def test_unknown_top_level_key_fails():
    _expect_config_error(
        _config_with(lambda raw: raw.__setitem__("mystery", 1)), "unknown keys"
    )


def test_missing_top_level_key_fails():
    _expect_config_error(
        _config_with(lambda raw: raw.pop("timeouts")), "missing required keys"
    )


def test_wrong_schema_version_fails():
    _expect_config_error(
        _config_with(lambda raw: raw.__setitem__("schema_version", 2)),
        "schema_version must be 1",
    )


def test_tenant_prefix_must_be_tenant_safe():
    _expect_config_error(
        _config_with(lambda raw: raw.__setitem__("tenant_prefix", "Bad Prefix")),
        "tenant_prefix",
    )


def test_cases_must_be_non_empty():
    _expect_config_error(
        _config_with(lambda raw: raw.__setitem__("cases", [])), "non-empty list"
    )


def test_duplicate_case_names_fail():
    def mutate(raw):
        raw["cases"].append(copy.deepcopy(raw["cases"][0]))

    _expect_config_error(_config_with(mutate), "unique")


def test_chunk_must_be_multiple_of_page():
    def mutate(raw):
        raw["cases"][0]["page_granularity_tokens"] = 48

    _expect_config_error(_config_with(mutate), "must be a multiple")


def test_query_hit_floor_can_account_for_multiple_cache_groups():
    raw = _valid_config_dict()
    raw["cases"][0]["min_l3_query_hit_tokens"] = 2048
    assert parse_gate_config(raw, "test").cases[0].min_l3_query_hit_tokens == 2048


def test_min_output_cannot_exceed_suffix_budget():
    def mutate(raw):
        raw["cases"][0]["min_output_tokens"] = 64

    _expect_config_error(_config_with(mutate), "cannot exceed suffix_max_new_tokens")


def test_dtype_must_be_supported():
    def mutate(raw):
        raw["cases"][0]["dtype"] = "int8"

    _expect_config_error(_config_with(mutate), "dtype")


def test_gpu_memory_utilization_bounds():
    def mutate(raw):
        raw["cases"][0]["gpu_memory_utilization"] = 1.0

    _expect_config_error(_config_with(mutate), "gpu_memory_utilization")


def test_provider_segment_floor():
    _expect_config_error(
        _config_with(lambda raw: raw.__setitem__("provider", {"segment_bytes": 1024})),
        "segment_bytes",
    )


def test_async_overrides_require_wait_timeout():
    _expect_config_error(
        _config_with(lambda raw: raw.__setitem__("async_overrides", {})),
        "wait_timeout_ms",
    )


def test_max_prefill_must_cover_chunk():
    def mutate(raw):
        raw["cases"][0]["max_prefill_tokens"] = 256

    _expect_config_error(_config_with(mutate), "max_prefill_tokens")


def test_case_missing_key_fails():
    def mutate(raw):
        del raw["cases"][0]["seed"]

    _expect_config_error(_config_with(mutate), "missing required keys")


@pytest.mark.parametrize(
    "flag",
    [
        "--kvstore-storage-backend",
        "--kvstore-storage-backend-extra-config",
        "--disable-prefix-caching",
        "--disable-kvstore",
        "--enable-metrics",
        "--metrics-port",
        "--chunked-prefill-size",
        "--model",
        "--model-path",
        "--disable-flashinfer-autotune",
        "--enable-output-logprobs",
        "--seed",
        "--disable-autotune",
    ],
)
def test_managed_flags_are_rejected(flag):
    _expect_config_error(
        _config_with(
            lambda raw: raw["cases"][0].__setitem__("extra_engine_args", [flag])
        ),
        "harness-managed",
    )


def test_managed_flag_with_equals_form_is_rejected():
    with pytest.raises(GateConfigError, match="harness-managed"):
        check_extra_engine_args(("--metrics-port=9999",), "test")


# ---------------------------------------------------------------------------
# Engine command lines
# ---------------------------------------------------------------------------


def _loaded():
    return parse_gate_config(_valid_config_dict(), "test-config")


def test_serving_argv_writer_and_reader_differ_only_in_metrics_port():
    config = _loaded()
    case = config.cases[0]
    tenant = run_tenant_id(config.tenant_prefix, "abc123")
    writer = serving_engine_argv(
        case, config.mooncake, config.async_overrides, tenant, "127.0.0.1:50051", 9101
    )
    reader = serving_engine_argv(
        case, config.mooncake, config.async_overrides, tenant, "127.0.0.1:50051", 9102
    )
    assert writer[writer.index("--metrics-port") + 1] == "9101"
    assert reader[reader.index("--metrics-port") + 1] == "9102"
    without_ports = [
        [arg for arg in argv if arg not in ("9101", "9102")]
        for argv in (writer, reader)
    ]
    assert without_ports[0] == without_ports[1]
    assert writer[writer.index("--kvstore-storage-backend") + 1] == "mooncake"
    extra = json.loads(
        writer[writer.index("--kvstore-storage-backend-extra-config") + 1]
    )
    assert extra["master_server_address"] == "127.0.0.1:50051"
    assert extra["global_segment_size"] == 0
    assert extra["tenant_id"] == "l3serving-abc123"
    assert extra["async"] == {
        "ready_timeout_ms": 5000,
        "wait_timeout_ms": 30000,
        "max_prefetch_tokens": 2048,
    }
    # The case's own arguments ride along, after every managed flag.
    assert writer[-2:] == ["--attention-backend", "example"]
    # Determinism and identity flags the comparators rely on.
    assert "--enable-output-logprobs" in writer
    assert "--disable-autotune" in writer
    assert writer[writer.index("--seed") + 1] == "42"
    assert writer[writer.index("--chunked-prefill-size") + 1] == "512"


def test_oracle_argv_recomputes_everything_and_never_touches_the_store():
    config = _loaded()
    case = config.cases[0]
    argv = oracle_engine_argv(case)
    assert "--disable-prefix-caching" in argv
    assert "--disable-kvstore" in argv
    assert "--kvstore-storage-backend" not in argv
    assert "--enable-metrics" not in argv
    # The numerics-relevant choices stay identical to the serving engines.
    serving = serving_engine_argv(
        case, config.mooncake, config.async_overrides, "t", "127.0.0.1:1", 9000
    )
    for flag in ("--model", "--dtype", "--seed", "--chunked-prefill-size"):
        assert argv[argv.index(flag) + 1] == serving[serving.index(flag) + 1]
    assert argv[-2:] == ["--attention-backend", "example"]
    assert "--enable-output-logprobs" in argv


def test_provider_command_and_lifetime():
    config = _loaded()
    command = provider_command(
        "/usr/bin/python3",
        "/repo/test/runtime/test_l3_serving_e2e.py",
        config.mooncake,
        "l3serving-abc123",
        "127.0.0.1:50051",
        config.provider.segment_bytes,
        7200.0,
    )
    assert command[1].endswith("test_l3_serving_e2e.py")
    assert command[command.index("--tenant-id") + 1] == "l3serving-abc123"
    assert command[command.index("--segment-bytes") + 1] == "268435456"
    assert command[command.index("--max-seconds") + 1] == "7200.0"
    one_case = provider_lifetime_seconds(config.timeouts, 1)
    two_cases = provider_lifetime_seconds(config.timeouts, 2)
    assert two_cases > one_case > 0
    with pytest.raises(ValueError, match="num_cases"):
        provider_lifetime_seconds(config.timeouts, 0)


def test_run_tenant_id_validation():
    assert run_tenant_id("l3serving", "deadbeef") == "l3serving-deadbeef"
    with pytest.raises(ValueError):
        run_tenant_id("bad prefix", "deadbeef")
    with pytest.raises(ValueError):
        run_tenant_id("l3serving", "")


def test_metrics_url_is_loopback():
    assert metrics_url(9123) == "http://127.0.0.1:9123/metrics"


# ---------------------------------------------------------------------------
# Prometheus parsing and evidence snapshots
# ---------------------------------------------------------------------------

_EXPOSITION = """# HELP tokenspeed:l3_io_keys_total Reported per-key outcomes.
# TYPE tokenspeed:l3_io_keys_total counter
tokenspeed:l3_io_keys_total{app_key="",dp_rank="0",engine_role="fused",kind="put",model_name="m",result="success",tp_rank="0"} 5.0
tokenspeed:l3_io_keys_total{app_key="",dp_rank="0",engine_role="fused",kind="get",model_name="m",result="success",tp_rank="0"} 3.0
tokenspeed:l3_io_keys_total{app_key="",dp_rank="0",engine_role="fused",kind="get",model_name="m",result="failure",tp_rank="0"} 1.0
tokenspeed:l3_io_adapter_bytes_total{app_key="",dp_rank="0",engine_role="fused",kind="get",model_name="m",tp_rank="0"} 24576.0
tokenspeed:l3_io_adapter_bytes_total{app_key="",dp_rank="0",engine_role="fused",kind="put",model_name="m",tp_rank="0"} 40960.0
tokenspeed:l3_io_operations{app_key="",dp_rank="0",engine_role="fused",kind="put",model_name="m",state="queued",tp_rank="0"} 2.0
tokenspeed:l3_io_operations{app_key="",dp_rank="0",engine_role="fused",kind="put",model_name="m",state="running",tp_rank="0"} 1.0
tokenspeed:l3_scheduler_resources{app_key="",dp_rank="0",engine_role="fused",model_name="m",resource="backup_candidates",tp_rank="0"} 7.0
tokenspeed:l3_io_quarantined{app_key="",dp_rank="0",engine_role="fused",model_name="m",tp_rank="0"} 0.0
tokenspeed:prefix_l3_hit_tokens_total{app_key="",dp_rank="0",model_name="m"} 1024.0
tokenspeed:unrelated_counter_total{job="x"} 42.0
"""


def test_parse_exposition_and_snapshot_sums_label_values():
    metrics = parse_prometheus_text(_EXPOSITION)
    assert (
        metrics[
            (
                "tokenspeed:l3_io_keys_total",
                tuple(
                    sorted(
                        [
                            ("app_key", ""),
                            ("dp_rank", "0"),
                            ("engine_role", "fused"),
                            ("kind", "get"),
                            ("model_name", "m"),
                            ("result", "success"),
                            ("tp_rank", "0"),
                        ]
                    )
                ),
            )
        ]
        == 3.0
    )
    snapshot = snapshot_from_prometheus(metrics)
    assert snapshot.put_success_keys == 5.0
    assert snapshot.put_success_bytes == 40960.0
    # Outstanding sums every state of the kind.
    assert snapshot.put_jobs == 3.0
    assert snapshot.get_success_keys == 3.0
    assert snapshot.get_failed_keys == 1.0
    assert snapshot.get_success_bytes == 24576.0
    assert snapshot.backup_candidates == 7.0
    assert snapshot.l3_query_hit_tokens == 1024.0
    assert snapshot.io_quarantined == 0.0


def test_snapshot_from_exposition_roundtrip():
    snapshot = snapshot_from_exposition(_EXPOSITION)
    assert snapshot.get_success_keys == 3.0


def test_parse_rejects_malformed_lines():
    with pytest.raises(ValueError, match="not a valid sample"):
        parse_prometheus_text("this is not a sample line")
    with pytest.raises(ValueError, match="malformed labels"):
        parse_prometheus_text("tokenspeed:x_total{bad labels} 1.0")
    with pytest.raises(ValueError, match="non-numeric value"):
        parse_prometheus_text("tokenspeed:x_total soon")


def test_missing_counters_are_zero_but_missing_gauges_are_unknown():
    snapshot = snapshot_from_exposition('tokenspeed:other_total{job="x"} 9.0\n')
    assert snapshot.get_success_keys == 0.0
    assert snapshot.l3_query_hit_tokens == 0.0
    assert snapshot.put_jobs is None
    assert len(missing_gauges(snapshot)) == 6
    assert evaluate_writer_evidence(snapshot, _thresholds())


def _thresholds():
    return thresholds_from_case(_loaded().cases[0])


def _empty_snapshot():
    return EvidenceSnapshot(
        put_success_keys=0.0,
        put_failed_keys=0.0,
        put_success_bytes=0.0,
        get_success_keys=0.0,
        get_failed_keys=0.0,
        get_success_bytes=0.0,
        dropped_backup_keys=0.0,
        l3_query_hit_tokens=0.0,
        put_jobs=0.0,
        get_jobs=0.0,
        storage_operations=0.0,
        backup_candidates=0.0,
        quarantined_blocks=0.0,
        io_quarantined=0.0,
        nonfinite_series=(),
    )


def _clean_writer_snapshot():
    return replace(_empty_snapshot(), put_success_keys=16.0, put_success_bytes=131072.0)


def _clean_reader_snapshot():
    return replace(
        _empty_snapshot(),
        get_success_keys=16.0,
        get_success_bytes=131072.0,
        l3_query_hit_tokens=1024.0,
    )


def test_writer_evidence_passes_when_persisted_and_drained():
    assert evaluate_writer_evidence(_clean_writer_snapshot(), _thresholds()) == []


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("put_success_keys", 0.0, "PUT evidence missing"),
        ("put_success_bytes", 0.0, "PUT byte evidence"),
        ("put_failed_keys", 1.0, "failed keys"),
        ("dropped_backup_keys", 1.0, "dropped"),
        ("put_jobs", 2.0, "not drained"),
        ("storage_operations", 1.0, "not drained"),
        ("backup_candidates", 3.0, "not drained"),
        ("io_quarantined", 1.0, "quarantine"),
        ("quarantined_blocks", 1.0, "quarantine"),
    ],
)
def test_writer_rejects_incomplete_or_unsafe_evidence(field, value, fragment):
    violations = evaluate_writer_evidence(
        replace(_clean_writer_snapshot(), **{field: value}), _thresholds()
    )
    assert any(fragment in message for message in violations)


def test_reader_restore_passes_with_full_evidence():
    assert (
        evaluate_reader_restore(
            _empty_snapshot(), _clean_reader_snapshot(), _thresholds(), 1024, 1024
        )
        == []
    )


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("get_success_keys", 0.0, "GET evidence missing"),
        ("get_success_bytes", 0.0, "GET byte evidence"),
        ("get_failed_keys", 1.0, "failed keys"),
        ("l3_query_hit_tokens", 0.0, "QUERY found too few"),
        ("io_quarantined", 1.0, "quarantine"),
        ("quarantined_blocks", 1.0, "quarantine"),
    ],
)
def test_reader_rejects_missing_or_unsafe_evidence(field, value, fragment):
    violations = evaluate_reader_restore(
        _empty_snapshot(),
        replace(_clean_reader_snapshot(), **{field: value}),
        _thresholds(),
        1024,
        1024,
    )
    assert any(fragment in message for message in violations)


@pytest.mark.parametrize("cached_tokens", [0, 512, 1536])
def test_query_and_get_success_do_not_prove_admission(cached_tokens):
    violations = evaluate_reader_restore(
        _empty_snapshot(), _clean_reader_snapshot(), _thresholds(), 1024, cached_tokens
    )
    assert any(f"reused {cached_tokens} cached tokens" in item for item in violations)


def test_reader_checks_deltas_not_lifetime_totals():
    snapshot = _clean_reader_snapshot()
    assert evaluate_reader_restore(snapshot, snapshot, _thresholds(), 1024, 1024)
    baseline = replace(
        snapshot,
        get_success_keys=32.0,
        get_success_bytes=262144.0,
        l3_query_hit_tokens=2048.0,
    )
    after = replace(
        baseline,
        get_success_keys=48.0,
        get_success_bytes=393216.0,
        l3_query_hit_tokens=3072.0,
    )
    assert evaluate_reader_restore(baseline, after, _thresholds(), 1024, 1024) == []


def test_negative_control_passes_on_full_recompute():
    before = _clean_reader_snapshot()
    after = replace(before, get_failed_keys=2.0)
    assert evaluate_negative_control(before, after, 0) == []


@pytest.mark.parametrize(
    "field,value,cached_tokens",
    [
        ("get_success_keys", 1.0, 0),
        ("l3_query_hit_tokens", 64.0, 0),
        ("get_success_keys", 0.0, 64),
    ],
)
def test_negative_control_rejects_any_reuse(field, value, cached_tokens):
    before = _empty_snapshot()
    after = replace(before, **{field: value})
    assert evaluate_negative_control(before, after, cached_tokens)


@pytest.mark.parametrize(
    "field",
    [
        "put_jobs",
        "get_jobs",
        "storage_operations",
        "backup_candidates",
        "quarantined_blocks",
        "io_quarantined",
    ],
)
def test_each_missing_gauge_invalidates_every_phase(field):
    incomplete = replace(_clean_writer_snapshot(), **{field: None})
    assert evaluate_writer_evidence(incomplete, _thresholds())
    assert evaluate_reader_restore(
        incomplete, _clean_reader_snapshot(), _thresholds(), 1024, 1024
    )
    assert evaluate_reader_restore(
        _empty_snapshot(), incomplete, _thresholds(), 1024, 1024
    )
    assert evaluate_negative_control(_empty_snapshot(), incomplete, 0)
    assert evaluate_negative_control(incomplete, _empty_snapshot(), 0)


def test_finish_reason_gate():
    assert evaluate_completion_finish("stop", "probe") == []
    assert evaluate_completion_finish("length", "probe") == []
    violations = evaluate_completion_finish("abort", "probe")
    assert any("abort" in message for message in violations)
    assert evaluate_completion_finish(None, "probe") != []


# ---------------------------------------------------------------------------
# The output comparator
# ---------------------------------------------------------------------------


def test_compare_outputs_passes_on_identical_continuations():
    violations = compare_outputs(
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        max_token_mismatches=0,
        logprob_atol=1e-4,
        min_output_tokens=8,
    )
    assert violations == []


def test_compare_outputs_flags_token_mismatches():
    violations = compare_outputs(
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1] * 8,
        [101, 202, 999, 404, 505, 606, 707, 808],
        [-0.1] * 8,
        max_token_mismatches=0,
        logprob_atol=1e-4,
        min_output_tokens=8,
    )
    assert any("1 output tokens differ" in message for message in violations)
    assert any("position" in message for message in violations)


def test_compare_outputs_honours_the_token_envelope():
    violations = compare_outputs(
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1] * 8,
        [101, 202, 999, 404, 505, 606, 707, 808],
        [-0.1] * 8,
        max_token_mismatches=1,
        logprob_atol=1e-4,
        min_output_tokens=8,
    )
    assert violations == []


def test_compare_outputs_flags_logprob_drift_beyond_atol():
    violations = compare_outputs(
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1, -0.2, -0.3, -0.405, -0.5, -0.6, -0.7, -0.8],
        max_token_mismatches=0,
        logprob_atol=1e-4,
        min_output_tokens=8,
    )
    assert any("position 3" in message for message in violations)


def test_compare_outputs_rejects_a_vacuous_reference():
    violations = compare_outputs(
        [101, 202],
        [-0.1, -0.2],
        [101, 202],
        [-0.1, -0.2],
        max_token_mismatches=0,
        logprob_atol=1e-4,
        min_output_tokens=8,
    )
    assert any("vacuous" in message for message in violations)


def test_compare_outputs_flags_length_mismatch():
    violations = compare_outputs(
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1] * 8,
        [101, 202, 303],
        [-0.1] * 3,
        max_token_mismatches=0,
        logprob_atol=1e-4,
        min_output_tokens=8,
    )
    assert any("output length differs" in message for message in violations)


def test_compare_outputs_requires_logprob_coverage():
    violations = compare_outputs(
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1] * 4,
        [101, 202, 303, 404, 505, 606, 707, 808],
        [-0.1] * 8,
        max_token_mismatches=0,
        logprob_atol=1e-4,
        min_output_tokens=8,
    )
    assert any("logprobs" in message for message in violations)


# ---------------------------------------------------------------------------
# Prompt construction and alignment
# ---------------------------------------------------------------------------


def test_fact_prefix_text_is_deterministic_and_carries_the_codeword():
    first = fact_prefix_text("codeword-7f3a", 4)
    second = fact_prefix_text("codeword-7f3a", 4)
    assert first == second
    assert "codeword-7f3a" in first
    assert len(fact_prefix_text("codeword-7f3a", 8)) > len(first)


def test_fact_prefix_text_validates_inputs():
    with pytest.raises(ValueError):
        fact_prefix_text("", 1)
    with pytest.raises(ValueError):
        fact_prefix_text("codeword", 0)


def test_suffix_question_needs_the_prefix():
    question = suffix_question_text()
    assert "codeword" in question
    # The question never reveals the run's codeword itself.
    assert "Answer:" in question


def test_negative_control_is_disjoint_from_the_prefix():
    prefix = fact_prefix_text("codeword-7f3a", 8)
    negative = negative_control_text()
    assert "codeword-7f3a" not in negative
    # No shared leading span at all: the openings differ from token one.
    assert prefix[:40] != negative[:40]
    common = 0
    for left, right in zip(prefix, negative):
        if left != right:
            break
        common += 1
    assert common < 8


def test_aligned_length():
    assert aligned_prefix_length(100, 512) == 512
    assert aligned_prefix_length(1024, 512) == 1024
    assert aligned_prefix_length(1025, 512) == 1536
    with pytest.raises(ValueError):
        aligned_prefix_length(1, 0)
    with pytest.raises(ValueError):
        aligned_prefix_length(0, 64)


def test_mooncake_extra_config_json_payload():
    config = _loaded()
    payload = json.loads(
        mooncake_extra_config_json(
            config.mooncake, config.async_overrides, "l3serving-x", "127.0.0.1:50051"
        )
    )
    assert payload == {
        "master_server_address": "127.0.0.1:50051",
        "local_hostname": "localhost",
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "tcp",
        "device_name": "",
        "global_segment_size": 0,
        "tenant_id": "l3serving-x",
        "async": {
            "ready_timeout_ms": 5000,
            "wait_timeout_ms": 30000,
            "max_prefetch_tokens": 2048,
        },
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["oracle", "reader"])
def test_output_comparison_rejects_nonfinite_logprobs(value, side):
    reference = [value] if side == "oracle" else [-0.1]
    candidate = [value] if side == "reader" else [-0.1]
    violations = compare_outputs([1], reference, [1], candidate, 0, 1e-4, 1)
    assert any("non-finite logprobs" in item for item in violations)


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf"])
@pytest.mark.parametrize(
    "series",
    [
        'tokenspeed:l3_io_keys_total{kind="put",result="success"}',
        'tokenspeed:l3_io_adapter_bytes_total{kind="get"}',
        'tokenspeed:l3_scheduler_resources{resource="backup_candidates"}',
        "tokenspeed:prefix_l3_hit_tokens_total",
    ],
)
def test_nonfinite_metric_samples_cannot_prove_success(series, value):
    text = "\n".join(
        [
            'tokenspeed:l3_io_operations{kind="put",state="running"} 0',
            'tokenspeed:l3_io_operations{kind="get",state="running"} 0',
            'tokenspeed:l3_scheduler_resources{resource="operations"} 0',
            'tokenspeed:l3_scheduler_resources{resource="backup_candidates"} 0',
            'tokenspeed:l3_scheduler_resources{resource="quarantined"} 0',
            "tokenspeed:l3_io_quarantined 0",
            f"{series} {value}",
        ]
    )
    snapshot = snapshot_from_exposition(text)
    assert not missing_gauges(snapshot)
    assert snapshot.nonfinite_series
    assert evaluate_writer_evidence(snapshot, _thresholds())
    assert evaluate_reader_restore(
        _empty_snapshot(), snapshot, _thresholds(), 1024, 1024
    )
    assert evaluate_negative_control(_empty_snapshot(), snapshot, 0)


def test_identical_answers_that_ignore_the_prefix_fail():
    assert compare_outputs([1], [-0.1], [1], [-0.1], 0, 1e-4, 1) == []
    assert evaluate_codeword_answers("a9f317", "I do not know", "I do not know", "Soup")
    assert evaluate_codeword_answers("a9f317", "a9f317", "a9f317", "Soup") == []
    assert evaluate_codeword_answers("a9f317", "a9f317", "a9f317", "a9f317")
    assert evaluate_codeword_answers("a9f317", "a9f317", "different", "Soup")


def _byte_encode(text):
    return list(text.encode("utf-8"))


def _byte_decode(ids):
    return bytes(ids).decode("utf-8")


@pytest.mark.parametrize("codeword", ["f09a", "b7ac512fe902", "0123456789abcdef"])
@pytest.mark.parametrize("target", [512, 1024, 1536])
def test_prefix_cut_reaches_exact_chunk_boundary_and_preserves_fact(codeword, target):
    prefix = aligned_prefix_ids(_byte_encode, _byte_decode, codeword, target)
    assert len(prefix) == target
    assert codeword in _byte_decode(prefix)
    assert prefix == aligned_prefix_ids(_byte_encode, _byte_decode, codeword, target)


def test_prefix_cut_that_loses_fact_fails():
    with pytest.raises(ValueError, match="codeword does not survive"):
        aligned_prefix_ids(_byte_encode, _byte_decode, "codeword", 10)


def test_non_growing_tokenizer_does_not_loop_forever():
    with pytest.raises(ValueError, match="does not grow"):
        aligned_prefix_ids(lambda text: [1], lambda ids: "codeword", "codeword", 16)


@pytest.mark.parametrize(
    "prefetch,prefix,suffix,control,fragment",
    [
        (2048, 1025, 30, 30, "whole number"),
        (512, 1024, 30, 30, "max_prefetch_tokens"),
        (2048, 1024, 513, 30, "single chunk"),
        (4096, 4096, 30, 30, "max_model_len"),
        (2048, 1024, 30, 4096, "negative control"),
    ],
)
def test_prompt_budgets_reject_invalid_runs(
    prefetch, prefix, suffix, control, fragment
):
    with pytest.raises(GateConfigError, match=fragment):
        check_token_budgets(_loaded().cases[0], prefetch, prefix, suffix, control)


def test_valid_prompt_budgets():
    check_token_budgets(_loaded().cases[0], 2048, 1024, 30, 100)


def test_requested_gate_only_skips_when_environment_is_unset():
    assert requested_config_path({}) is None
    assert (
        requested_config_path({"L3_SERVING_E2E_CONFIG": "/tmp/case.json"})
        == "/tmp/case.json"
    )
    for value in ("", " ", "\n\t"):
        with pytest.raises(GateConfigError, match="set but blank"):
            requested_config_path({"L3_SERVING_E2E_CONFIG": value})


@pytest.mark.parametrize(
    "flag",
    [
        "--tp=2",
        "--world-size=2",
        "--data-parallel-size",
        "--pipeline-parallel-size",
        "--speculative-config",
        "--speculative_config",
        "--ep-size=2",
    ],
)
def test_multi_rank_and_speculation_are_out_of_scope(flag):
    with pytest.raises(GateConfigError, match="outside.*scope"):
        check_extra_engine_args((flag,), "case")


def test_tcp_scope_is_explicit():
    raw = _valid_config_dict()
    raw["mooncake"]["protocol"] = "rdma"
    _expect_config_error(raw, "requires protocol tcp")


def test_async_prefetch_cap_is_required():
    raw = _valid_config_dict()
    del raw["async_overrides"]["max_prefetch_tokens"]
    _expect_config_error(raw, "max_prefetch_tokens")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_configuration_rejects_nonfinite_tolerances(value):
    raw = _valid_config_dict()
    raw["cases"][0]["logprob_atol"] = value
    _expect_config_error(raw, "finite")


def test_control_quarantine_fails_even_without_cache_reuse():
    assert evaluate_negative_control(
        _empty_snapshot(), replace(_empty_snapshot(), io_quarantined=1.0), 0
    )


def test_bounded_call_returns_values_and_propagates_failure():
    assert run_bounded(lambda: 42, 1.0, "result") == 42
    error = RuntimeError("request failed")

    def fail():
        raise error

    with pytest.raises(RuntimeError) as caught:
        run_bounded(fail, 1.0, "error")
    assert caught.value is error


def test_hung_request_thread_cannot_prevent_interpreter_exit():
    code = """
import threading
from test.l3_serving_test_utils import run_bounded
try:
    run_bounded(threading.Event().wait, 0.05, "wedged request")
except TimeoutError:
    print("timed out", flush=True)
else:
    raise AssertionError("unexpected result")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "timed out" in result.stdout


def test_provider_noisy_stderr_does_not_block_readiness_or_cleanup():
    code = """
import signal, sys, time
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
sys.stderr.write("x" * (2 * 1024 * 1024) + "\\n")
sys.stderr.flush()
print("READY", flush=True)
while True:
    time.sleep(1)
"""
    provider = start_provider_process([sys.executable, "-u", "-c", code])
    try:
        await_provider_ready(provider, "READY", 5.0, 0.01)
    finally:
        stopped = stop_provider_process(provider, 2.0)
    assert stopped.returncode == 0
    assert len(stopped.log) > 2 * 1024 * 1024
    assert provider.process.poll() == 0
    assert provider.log.closed
    assert stop_provider_process(provider, 2.0) is stopped


def test_provider_early_failure_keeps_original_error_and_log():
    provider = start_provider_process(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('mount failed', file=sys.stderr); sys.exit(7)",
        ]
    )

    def stop():
        return stop_provider_process(provider, 1.0).log

    with pytest.raises(RuntimeError, match="before mounting") as caught:
        with stop_on_exit(stop, "provider"):
            await_provider_ready(provider, "READY", 5.0, 0.01)
    assert "mount failed" in "\n".join(caught.value.__notes__)
    assert provider.process.returncode == 7
    assert provider.log.closed


def test_provider_ignoring_sigterm_is_killed_and_reaped():
    code = """
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("READY", flush=True)
while True:
    time.sleep(1)
"""
    provider = start_provider_process([sys.executable, "-u", "-c", code])
    try:
        await_provider_ready(provider, "READY", 5.0, 0.01)
    finally:
        stopped = stop_provider_process(provider, 0.05)
    assert stopped.returncode == -signal.SIGKILL
    assert provider.log.closed


def test_provider_readiness_timeout_still_cleans_up():
    provider = start_provider_process(
        [sys.executable, "-u", "-c", "import time; time.sleep(30)"]
    )
    with pytest.raises(TimeoutError):
        with stop_on_exit(lambda: stop_provider_process(provider, 0.1).log, "provider"):
            await_provider_ready(provider, "READY", 0.05, 0.01)
    assert provider.process.poll() is not None
    assert provider.log.closed


def test_cleanup_failure_preserves_primary_error_but_fails_success():
    primary = ValueError("bad output")

    def fail_stop():
        raise RuntimeError("close failed")

    with pytest.raises(ValueError) as caught:
        with stop_on_exit(fail_stop, "engine"):
            raise primary
    assert caught.value is primary
    assert "close failed" in "\n".join(caught.value.__notes__)
    with pytest.raises(RuntimeError, match="close failed"):
        with stop_on_exit(fail_stop, "engine"):
            pass


@pytest.mark.parametrize(
    "config_value,exit_code,fragment",
    [
        (None, 0, "1 skipped"),
        (" ", 1, "set but blank"),
        ("/nonexistent/l3-serving-gate.json", 1, "cannot read gate config"),
    ],
)
def test_real_gate_entrypoint_skip_and_configuration_failures(
    config_value, exit_code, fragment
):
    environ = os.environ.copy()
    environ.pop("L3_SERVING_E2E_CONFIG", None)
    if config_value is not None:
        environ["L3_SERVING_E2E_CONFIG"] = config_value
    # Existing runtime conftest fixtures require torch. These CPU checks
    # exercise only the gate entrypoint before any runtime is imported.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-q",
            "test/runtime/test_l3_serving_e2e.py",
        ],
        env=environ,
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert fragment in result.stdout + result.stderr


def test_gate_rejects_native_library_override_before_runtime_import():
    environ = os.environ.copy()
    environ["L3_SERVING_E2E_CONFIG"] = str(_EXAMPLE_CONFIG)
    environ["TOKENSPEED_MOONCAKE_STORE_LIBRARY"] = "/tmp/different-native-library.so"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-q",
            "test/runtime/test_l3_serving_e2e.py",
        ],
        env=environ,
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 1
    assert "unset TOKENSPEED_MOONCAKE_STORE_LIBRARY" in result.stdout
