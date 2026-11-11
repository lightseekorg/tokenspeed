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

"""CPU observations use actual classes and exact token-boundary identities."""

import importlib.util
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from tokenspeed.runtime.engine.service_trace import ServiceTrace

_spec = importlib.util.spec_from_file_location(
    "service_tpot_example",
    Path(__file__).resolve().parents[2] / "examples/profiling/service_tpot.py",
)
_baseline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _baseline
_spec.loader.exec_module(_baseline)


def test_commit_and_emit_preserve_generation_and_actual_chunks():
    output = io.StringIO()
    trace = ServiceTrace(output, run_id="run", nvtx_enabled=False)
    trace.dispatch(8, None)
    trace.commit(
        8,
        [("r", 4, 3, [10, 11], "stop")],
        [("r", 3, [10, 11], "stop")],
        owns_frontend=True,
    )
    trace.emit([("r", 3, [10, 11], "stop")])
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    commit, ready, emitted = records[-3:]
    assert [event["boundary"] for event in records] == [
        "start",
        "dispatch",
        "commit",
        "output_ready",
        "frontend_send_returned",
    ]
    for record in (commit, ready, emitted):
        assert record["forward_id"] == 8
        assert record["generation"] == 4
        assert record["completion_tokens"] == 3
        assert record["output_ids"] == [10, 11]
        assert record["retraction_epoch"] is None
        assert record["finish_reason"] == "stop"
    # Finished RID reuse gets the control plane's new admission, not a
    # retained observer guess from the previous generation.
    trace.dispatch(9, None)
    trace.commit(
        9, [("r", 5, 1, [12], None)], [("r", 1, [12], None)], owns_frontend=True
    )
    trace.emit([("r", 1, [12], None)])
    last = json.loads(output.getvalue().splitlines()[-1])
    assert last["generation"] == 5
    assert last["forward_id"] == 9


def test_unresolved_non_forward_output_remains_unknown():
    output = io.StringIO()
    trace = ServiceTrace(output, run_id="run", nvtx_enabled=False)
    trace.emit([("r", 0, [], "abort")])
    event = json.loads(output.getvalue().splitlines()[-1])
    assert event["forward_id"] is None
    assert event["generation"] is None


def test_client_window_uses_arrival_chunks_without_token_interpolation():
    sample = _baseline.StreamSample()
    for count, timestamp in [(3, 100), (3, 150), (7, 300), (8, 500)]:
        sample.add(
            {
                "meta_info": {
                    "id": "r",
                    "completion_tokens": count,
                    "finish_reason": None,
                }
            },
            timestamp,
        )
    window = sample.window(None, None)
    assert (window["first_count"], window["last_count"]) == (3, 8)
    assert window["tpot_ns"] == 80
    assert window["token_intervals"] == 5
    assert len(sample.chunks) == 4
    assert sample.window(4, 7)["status"] == "insufficient_distinct_arrival_chunks"


def test_independent_nsight_timestamps_match_exact_received_endpoints(tmp_path):
    sample = _baseline.StreamSample()
    for count, timestamp in [(3, 1000), (8, 1500)]:
        sample.add(
            {
                "meta_info": {
                    "id": "r",
                    "completion_tokens": count,
                    "finish_reason": None,
                }
            },
            timestamp,
        )
    database = tmp_path / "trace.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, text TEXT)")
        for count, timestamp in [(1, 100000), (3, 100100), (8, 100500)]:
            event = {
                "boundary": "commit",
                "schema": "tokenspeed.service/v1",
                "run_id": "run",
                "rid": "r",
                "generation": 2,
                "completion_tokens": count,
                "forward_id": count,
                "pid": 99,
            }
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?,?)",
                (timestamp, json.dumps(event, sort_keys=True)),
            )
    events = _baseline.nsight_events(database)
    aligned = _baseline.align_window(
        {"rid": "r"}, sample.window(None, None), events, run_id="run", boundary="commit"
    )
    assert aligned["status"] == "observed"
    assert aligned["nsight_tpot_ns"] == 80
    assert aligned["client_tpot_ns"] == 100
    assert aligned["endpoint_residual_change_ns"] == 100
    assert aligned["first"]["completion_tokens"] == 3
    duplicate = _baseline.align_window(
        {"rid": "r"},
        sample.window(None, None),
        events + [events[-1]],
        run_id="run",
        boundary="commit",
    )
    assert duplicate["status"] == "unknown"


def test_reference_rejects_decreasing_counts():
    sample = _baseline.StreamSample()
    sample.add(
        {"meta_info": {"id": "r", "completion_tokens": 2, "finish_reason": None}}, 1
    )
    with pytest.raises(ValueError, match="decreased"):
        sample.add(
            {"meta_info": {"id": "r", "completion_tokens": 1, "finish_reason": None}}, 2
        )


def test_paired_report_separates_baseline_noise_from_observer_overhead(tmp_path):
    case = {
        "engine_args": {"model": "/readonly/real-weights", "stream_interval": 1},
        "sampling_params": {"temperature": 0, "ignore_eos": True, "max_new_tokens": 8},
        "input_ids": [1, 2],
        "warmup_requests": 1,
        "measured_requests": 1,
        "sequence": ["baseline", "baseline", "profiled"],
        "statistical_window": {"first_count": 2, "last_count": 8},
        "profile_environment": {},
        "cache_policy": "flush_before_each_request",
        "instrumentation_scope": "service_only",
        "coverage_mode": "decode",
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    _, case_hash = _baseline.load_case(path)
    for index, (arm, tpot) in enumerate(zip(case["sequence"], [100, 110, 130])):
        record = {
            "kind": "request",
            "phase": "measured",
            "index": 0,
            "window": {
                "status": "observed",
                "first_count": 1,
                "last_count": 8,
                "tpot_ns": tpot,
            },
            "sample": {"chunks": [{"output_ids_sha256": "same_tokens"}]},
        }
        (tmp_path / f"{index}-{arm}.jsonl").write_text(
            json.dumps({"case_sha256": case_hash, "run_id": f"run-{index}"})
            + "\n"
            + json.dumps(record)
            + "\n"
        )
    report = _baseline.summarize_pair(tmp_path)
    assert report["baseline_noise_pairs"][0]["tpot_difference_ns"] == 10
    assert report["profiled_overhead_pairs"][0]["tpot_difference_ns"] == 20
    assert report["baseline_tpot_ns"]["mean"] == 105
    assert report["profiled_overhead_pairs"][0]["model_layer_op_overhead_ns"] is None
    assert (
        report["profiled_overhead_pairs"][0]["verified_instrumentation_scope"]
        != "model_layer_op"
    )
    assert report["threshold_status"] == "not_set_user_confirms_after_real_data"
    case["engine_args"]["load_format"] = "dummy"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="real weights"):
        _baseline.load_case(path)


def test_non_owner_reuse_does_not_retain_tokens_or_invent_send():
    import gc
    import weakref

    class Tokens(list):
        pass

    stream = io.StringIO()
    trace = ServiceTrace(stream, run_id="run", nvtx_enabled=False)
    refs = []
    for generation in range(100):
        ids = Tokens([generation])
        refs.append(weakref.ref(ids))
        trace.dispatch(generation, None)
        trace.commit(
            generation,
            [("reused", generation, 1, ids, "stop")],
            [("reused", 1, ids, "stop")],
            owns_frontend=False,
        )
    del ids
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert all(
        json.loads(line)["boundary"] != "frontend_send_returned"
        for line in stream.getvalue().splitlines()
    )
    # A cancelled owner's pending output is released after the send phase;
    # a later output cannot inherit that ended generation's identity.
    trace.dispatch(101, None)
    trace.commit(
        101,
        [("cancelled", 101, 1, [7], "abort")],
        [("cancelled", 1, [7], "abort")],
        owns_frontend=True,
    )
    trace.release(["cancelled"])
    trace.emit([("cancelled", 1, [8], None)])
    assert json.loads(stream.getvalue().splitlines()[-1])["generation"] is None


def test_logical_token_parity_ignores_real_chunk_coalescing():
    a, b = _baseline.StreamSample(), _baseline.StreamSample()
    for sample, chunks in (
        (a, [(1, [10]), (2, [11]), (3, [12])]),
        (b, [(2, [10, 11]), (3, [10, 11, 12])]),
    ):
        for count, ids in chunks:
            sample.add(
                {
                    "output_ids": ids,
                    "meta_info": {
                        "id": "r",
                        "completion_tokens": count,
                        "cached_tokens": 0,
                        "finish_reason": None,
                    },
                },
                count,
            )
        sample.finish()
    assert a.logical_output_status == b.logical_output_status == "verified"
    assert a.logical_output_sha256 == b.logical_output_sha256
    assert a.logical_output_ids == [10, 11, 12]


def test_protocol_flushes_every_request_before_timer_and_arms_after_warmup():
    import time
    from dataclasses import dataclass

    @dataclass
    class Reply:
        success: bool
        message: str = "CPU service fixture"

    class CpuService:
        """The absent model/network service, retaining its actual control contract."""

        def __init__(self, *, flush_success):
            self.flush_success = flush_success
            self.last_flush_ns = None
            self.generated_profile_states = []
            self.active = False
            self.stopped = False

        def flush_cache(self):
            self.last_flush_ns = time.perf_counter_ns()
            return Reply(self.flush_success)

        def start_profile(self, *, activities, profile_id):
            assert activities == ["CUDA_PROFILER"]
            assert profile_id == "run"
            self.active = True
            return Reply(True)

        def stop_profile(self):
            self.active = False
            self.stopped = True
            return Reply(True)

        def generate(self, *, input_ids, sampling_params, stream):
            assert self.last_flush_ns is not None and stream
            self.generated_profile_states.append(self.active)
            for count in (1, 2):
                yield {
                    "output_ids": [count],
                    "meta_info": {
                        "id": str(len(self.generated_profile_states)),
                        "completion_tokens": count,
                        "cached_tokens": 0,
                        "finish_reason": None,
                    },
                }

    case = {
        "warmup_requests": 1,
        "measured_requests": 2,
        "input_ids": [1],
        "sampling_params": {},
        "cache_policy": "flush_before_each_request",
        "instrumentation_scope": "service_only",
        "statistical_window": {"first_count": 1, "last_count": 2},
    }
    service, records = CpuService(flush_success=True), []
    _baseline.run_protocol(
        service, case, arm="profiled", run_id="run", write=records.append
    )
    assert service.generated_profile_states == [False, True, True]
    assert service.stopped
    requests = [row for row in records if row["kind"] == "request"]
    assert len([row for row in records if row["kind"] == "cache_control"]) == 3
    assert all(row["cache_policy_status"] == "verified" for row in requests)
    assert all(
        chunk["cached_tokens"] == 0
        for row in requests
        for chunk in row["sample"]["chunks"]
    )
    failed = CpuService(flush_success=False)
    with pytest.raises(RuntimeError, match="cache flush"):
        _baseline.run_protocol(
            failed, case, arm="baseline", run_id="run", write=lambda row: None
        )
    assert failed.generated_profile_states == []


def test_metadata_gate_requires_every_forward_and_native_gate_requires_scope_evidence(
    tmp_path,
):
    import hashlib

    graph = tmp_path / "1-graph"
    graph.mkdir()
    (tmp_path / "model-config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    window = {
        "run_id": "run",
        "rank": 0,
        "writer_flushed": True,
        "writer_error": None,
        "dropped_metadata": 0,
        "unresolved_result_metadata": 0,
    }
    (graph / "rank-0.window.json").write_text(json.dumps(window))
    capture = {
        "run_id": "run",
        "capture_id": "capture",
        "graph_scopes": [{"scope_id": i} for i in [1, 2, 3]],
        "ops": [{"scope_id": 3}],
        "layers": [{"layer_id": 0}, {"layer_id": 1}],
        "unavailable_integer_metadata": [],
    }
    capture_file = graph / "capture.json"
    capture_file.write_text(json.dumps(capture))
    ready, commits = [], []
    for count in (2, 3, 4):
        payload = graph / f"step-{count}.metadata.bin"
        payload.write_bytes(bytes([count]))
        ready.append(
            {
                "event": "device_results_ready",
                "run_id": "run",
                "rank": 0,
                "step_id": count,
                "forward_id": count,
                "request_ids": ["rid"],
                "capture_id": "capture",
                "capture_file": str(capture_file),
                "unavailable_reason": None,
                "binary_payload": {
                    "path": str(payload),
                    "bytes": 1,
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                },
            }
        )
        commits.append(
            {
                "boundary": "commit",
                "run_id": "run",
                "pid": 1,
                "rid": "rid",
                "generation": 9,
                "forward_id": count,
                "completion_tokens": count,
                "output_ids": [count],
                "graph": {"rank": 0, "step_id": count},
            }
        )
    (tmp_path / "1-service-1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in commits)
    )
    index = graph / "rank-0.replays.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in (ready[0], ready[-1])))
    records = [
        {"run_id": "run"},
        {
            "kind": "request",
            "phase": "measured",
            "index": 0,
            "cache_policy_status": "verified",
            "sample": {"rid": "rid"},
            "window": {"status": "observed", "first_count": 2, "last_count": 4},
        },
    ]
    records += [
        {"kind": "profile_control", "action": action, "success": True}
        for action in ("start", "stop")
    ]
    partial = _baseline.verify_instrumentation(
        tmp_path, 1, records, requested_scope="model_layer_op"
    )
    assert partial["requests"]["0"]["status"] == "incomplete"
    assert (
        "missing_or_ambiguous_same_run_forward_ready_record"
        in partial["requests"]["0"]["reasons"]
    )
    index.write_text("".join(json.dumps(row) + "\n" for row in ready))
    metadata = _baseline.verify_instrumentation(
        tmp_path, 1, records, requested_scope="model_layer_op"
    )
    assert metadata["metadata_status"] == "verified"
    assert metadata["status"] == "incomplete"  # Metadata alone is not GPU attribution.
    unavailable_prefill = {
        "event": "device_results_ready",
        "run_id": "run",
        "rank": 0,
        "step_id": 1,
        "forward_id": 1,
        "request_ids": ["rid"],
        "unavailable_reason": "eager_prefill_not_captured",
    }
    window["dropped_metadata"] = 1
    (graph / "rank-0.window.json").write_text(json.dumps(window))
    index.write_text(
        "".join(json.dumps(row) + "\n" for row in [unavailable_prefill, *ready])
    )
    metadata = _baseline.verify_instrumentation(
        tmp_path, 1, records, requested_scope="model_layer_op"
    )
    assert metadata["metadata_status"] == "verified"
    assert metadata["whole_run_scope"] == "partial"
    assert metadata["dropped_metadata_disjoint_from_measured_decode"] is True
    metadata_path = tmp_path / "1-metadata-coverage.json"
    metadata_path.write_text(json.dumps(metadata))
    report = graph / "fixture.nsys-rep"
    report.write_bytes(b"offline analyzer receipt fixture, not a real Nsight report")
    request = metadata["requests"]["0"]
    evidence = {
        key: request[key] for key in ("rid", "generation", "first_count", "last_count")
    }
    evidence["forward_evidence"] = []
    for row in request["forward_evidence"]:
        evidence["forward_evidence"].append(
            {
                **{key: row[key] for key in ("forward_id", "step_id", "capture_id")},
                "graph_node_ids": [100],
                "scope_kinds": ["model", "layer", "op"],
                "layer_coverage": {
                    "expected_ids": [0, 1],
                    "attributed_ids": [0, 1],
                    "missing_ids": [],
                },
                "scope_coverage": {
                    "expected_scope_ids": [1, 2, 3],
                    "attributed_scope_ids": [1, 2, 3],
                    "missing_scope_ids": [],
                },
            }
        )
    native = {
        "schema": "tokenspeed.native_coverage/v1",
        "run_id": "run",
        "rank": 0,
        "scope": "model_layer_op",
        "coverage_mode": "decode",
        "attribution": "same_run_cuda_graph_node_ids",
        "status": "verified",
        "source_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "source_nsight_report": {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
        "requests": {"0": evidence},
    }
    native_path = graph / "rank-0.native-coverage.json"
    native_path.write_text(json.dumps(native))
    rejected = _baseline.verify_native_coverage(tmp_path, 1, metadata, metadata_path)
    assert (
        rejected["status"] == "incomplete"
    )  # Missing critical_unknowns is not defaulted to [].
    for row in evidence["forward_evidence"]:
        row["critical_unknowns"] = []
    native_path.write_text(json.dumps(native))
    verified = _baseline.verify_native_coverage(tmp_path, 1, metadata, metadata_path)
    assert verified["status"] == "verified"
    assert verified["verified_scope"] == "model_layer_op"
    assert verified["native_evidence"]["whole_run_status"] == "not_verified"
    assert verified["native_evidence"]["whole_run_critical_unknowns"] == []

    native.update(
        whole_run_status="partial",
        whole_run_critical_unknowns=["eager_scope_native_activity_unavailable"],
    )
    native_path.write_text(json.dumps(native))
    decode_only = _baseline.verify_native_coverage(tmp_path, 1, metadata, metadata_path)
    assert decode_only["status"] == "verified"
    assert decode_only["native_evidence"]["status"] == "verified"
    assert decode_only["native_evidence"]["whole_run_status"] == "partial"
    assert decode_only["native_evidence"]["whole_run_critical_unknowns"] == [
        "eager_scope_native_activity_unavailable"
    ]

    native.update(
        status="partial", critical_unknowns=["compute_normalization_unavailable"]
    )
    native_path.write_text(json.dumps(native))
    partial = _baseline.verify_native_coverage(tmp_path, 1, metadata, metadata_path)
    assert partial["native_graph_ownership"] == "partial"
    assert partial["native_evidence"]["critical_unknowns"] == [
        "compute_normalization_unavailable"
    ]
    assert partial["status"] == "incomplete"


def test_model_depth_aliases_preserve_declared_depth_and_reject_conflicts():
    for config, depth in [
        ({"num_hidden_layers": 3}, 3),
        ({"num_layers": 5}, 5),
        ({"num_hidden_layers": None, "num_layers": 2}, 2),
        ({"num_hidden_layers": 4, "num_layers": 4}, 4),
        ({"text_config": {"num_layers": 6}}, 6),
    ]:
        raw = json.dumps(config).encode()
        assert _baseline.model_layer_ids(raw) == list(range(depth))
        assert json.loads(raw) == config
    for config in [
        {},
        {"num_layers": 0},
        {"num_layers": -1},
        {"num_layers": True},
        {"num_layers": 2.0},
        {"num_hidden_layers": "3"},
        {"num_hidden_layers": 2, "num_layers": 3},
    ]:
        with pytest.raises(ValueError):
            _baseline.model_layer_ids(json.dumps(config).encode())


def test_run_arm_observes_startup_before_unavailable_engine_import(
    tmp_path, monkeypatch
):
    import builtins
    import hashlib
    import os
    from types import ModuleType

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"num_hidden_layers":1}')
    (tmp_path / "model-config.json").write_bytes((model / "config.json").read_bytes())
    timeline_path = tmp_path / "1-timeline.json"
    case = dict(
        engine_args=dict(model=str(model), stream_interval=1, enforce_eager=False),
        sampling_params=dict(temperature=0, ignore_eos=True, max_new_tokens=3),
        input_ids=[1],
        warmup_requests=0,
        measured_requests=1,
        sequence=["baseline", "profiled", "baseline"],
        statistical_window=dict(first_count=2, last_count=3),
        profile_environment=dict(TOKENSPEED_TIMELINE_CONFIG=str(timeline_path)),
        profile_command=[
            "nsys",
            "profile",
            "--capture-range=none",
            "--cuda-graph-trace=node",
        ],
        cache_policy="flush_before_each_request",
        instrumentation_scope="model_layer_op",
        coverage_mode="decode",
    )
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    case_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()
    run_id = f"{case_sha[:16]}-1"
    timeline_path.write_text(
        json.dumps(dict(run_id=run_id, output_dir=str(tmp_path / "1-graph")))
    )
    monkeypatch.setenv("TOKENSPEED_PROFILE_RUN_ID", run_id)
    monkeypatch.setenv("TOKENSPEED_TIMELINE_CONFIG", str(timeline_path))
    monkeypatch.setenv("TOKENSPEED_SERVICE_NVTX", "0")

    class UnavailableGpuEngine(RuntimeError):
        pass

    class Engine:
        def __init__(self, **kwargs):
            observation = json.loads(timeline_path.read_text())["startup_observation"]
            assert observation["body"]["process_id"] == os.getpid()
            raise UnavailableGpuEngine("GPU runtime boundary")

    module = ModuleType("tokenspeed.runtime.entrypoints.engine")
    module.Engine = Engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original_import = builtins.__import__

    def at_engine_import(name, *args, **kwargs):
        if name == module.__name__:
            observation = json.loads(timeline_path.read_text())["startup_observation"]
            assert observation["body"]["case_sha256"] == case_sha
            assert observation["body"]["timeline_config"]["run_id"] == run_id
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", at_engine_import)
    output = tmp_path / "1-profiled.jsonl"
    with pytest.raises(UnavailableGpuEngine):
        _baseline.run_arm(case_path, output, "profiled", 1)
    observation = json.loads(timeline_path.read_text())["startup_observation"]
    header = json.loads(output.read_text().splitlines()[0])
    assert header["startup_observation_sha256"] == observation["sha256"]
    assert (
        observation["body"]["realtime_before_ns"]
        <= observation["body"]["realtime_after_ns"]
    )
    assert "startup_observation" not in observation["body"]["timeline_config"]


@pytest.mark.parametrize("legacy_config", [False, True])
def test_sealed_manifest_rejects_epoch_later_self_consistent_config_change(
    tmp_path, monkeypatch, legacy_config
):
    import hashlib

    from tokenspeed_kernel.op_profile import _profile_config_for_observation
    from tokenspeed_kernel.profile_nsys import read_run_manifest

    case = dict(instrumentation_scope="model_layer_op")
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case))
    case_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()
    run_id = f"{case_sha[:16]}-1"
    timeline_path = tmp_path / "1-timeline.json"
    timeline_path.write_text(
        json.dumps(
            dict(
                run_id=run_id,
                output_dir=str(tmp_path),
                hardware="original.json",
                layers="layers",
                start_step=1,
                num_steps=1,
                max_scopes=8,
                hbm_policy="unavailable",
            )
        )
    )
    monkeypatch.setenv("TOKENSPEED_TIMELINE_CONFIG", str(timeline_path))
    monkeypatch.setenv("TOKENSPEED_PROFILE_RUN_ID", run_id)
    observation = _baseline.observe_before_engine(case, case_sha, repetition=1)
    config = _profile_config_for_observation(observation, run_id=run_id)
    if legacy_config:
        config["library"] = None
    config_path, window_path, capture_path, replay_path = [
        tmp_path / name
        for name in (
            "rank-0.config.json",
            "rank-0.window.json",
            "capture.json",
            "replays.jsonl",
        )
    ]
    context = dict(
        run_id=run_id,
        rank=0,
        startup_observation=observation,
        profile_config_sha256=_baseline.digest(config),
    )
    window = dict(
        schema="tokenspeed.profile_window/v1",
        run_id=run_id,
        rank=0,
        loaded_library_manifest=dict(
            semantic_provenance=dict(
                source_context=context, baseline=dict(source_context=context)
            )
        ),
    )
    communication = tmp_path / "rank-0.communication.json"
    communication.write_text(
        json.dumps(
            dict(
                schema="tokenspeed.communication_evidence/v1",
                run_id=run_id,
                rank=0,
                config_sha256=context["profile_config_sha256"],
            )
        )
    )
    capture = dict(
        run_id=run_id,
        rank=0,
        capture_id=run_id + ":r0:c1",
        process_id=123,
        static_metadata=dict(
            startup_observation_sha256=observation["sha256"],
            profile_config_sha256=context["profile_config_sha256"],
        ),
    )
    row = {key: capture[key] for key in ("run_id", "rank", "capture_id", "process_id")}
    row["capture_file"] = str(capture_path)
    replay_path.write_text(json.dumps(row) + "\n")
    client, report, model, metadata_path = [
        tmp_path / name
        for name in (
            "client.jsonl",
            "synthetic.nsys-rep",
            "model-config.json",
            "metadata.json",
        )
    ]
    client.write_text(
        json.dumps(
            dict(
                run_id=run_id,
                arm="profiled",
                repetition=1,
                case_sha256=case_sha,
                startup_observation_sha256=observation["sha256"],
            )
        )
        + "\n"
    )
    report.write_bytes(b"CPU manifest fixture; not GPU trace evidence")
    model.write_text('{"num_hidden_layers":1}')

    def seal():
        config_path.write_text(
            json.dumps(
                dict(
                    schema="tokenspeed.profile_config/v1",
                    run_id=run_id,
                    rank=0,
                    config=config,
                    sha256=_baseline.digest(config),
                )
            )
        )
        window_path.write_text(json.dumps(window))
        capture_path.write_text(json.dumps(capture))
        metadata_path.write_text(
            json.dumps(
                dict(
                    schema="tokenspeed.metadata_coverage/v1",
                    run_id=run_id,
                    rank=0,
                    service_sources=[],
                    ready_index_source=_baseline.artifact_source(replay_path),
                    window_manifest_source=_baseline.artifact_source(window_path),
                    model_config_source=_baseline.artifact_source(model),
                )
            )
        )
        paths = [
            case_path,
            timeline_path,
            config_path,
            window_path,
            capture_path,
            replay_path,
            client,
            report,
            model,
            metadata_path,
            Path(observation["body"]["driver_source"]["path"]),
        ]
        if not legacy_config:
            paths.append(communication)
        sources = {
            _baseline.artifact_source(path)["path"]: _baseline.artifact_source(path)
            for path in paths
        }
        document = dict(
            schema="tokenspeed.profile_run/v1",
            run_id=run_id,
            arm_index=1,
            report=str(report),
            client=str(client),
            case=str(case_path),
            metadata=str(metadata_path),
            model_config=str(model),
            timeline=str(timeline_path),
            startup_observation_sha256=observation["sha256"],
            ranks=[
                dict(
                    rank=0,
                    window=str(window_path),
                    replays=str(replay_path),
                    config=str(config_path),
                )
            ],
            artifacts=sources,
        )
        if not legacy_config:
            document["communication_evidence"] = [str(communication)]
        raw = json.dumps(document).encode()
        path = tmp_path / f"run-1.{hashlib.sha256(raw).hexdigest()}.json"
        path.write_bytes(raw)
        return path

    sealed = read_run_manifest(seal(), rank=0)
    assert sealed["graph_window"] == window_path
    if legacy_config:
        assert "sealed_communication_artifacts" not in sealed
    else:
        assert sealed["sealed_communication_artifacts"] == {
            str(item): _baseline.artifact_source(item)
            for item in (communication, config_path)
        }
    # Recompute every downstream hash after changing a real config field while
    # retaining the original pre-import observation. Integrity alone is not enough.
    config["hardware"] = "changed-after-epoch.json"
    context["profile_config_sha256"] = _baseline.digest(config)
    capture["static_metadata"]["profile_config_sha256"] = context[
        "profile_config_sha256"
    ]
    with pytest.raises(ValueError, match="expanded runtime config"):
        read_run_manifest(seal(), rank=0)


def test_python_nvtx_marker_keeps_exact_json_and_callback_thread(monkeypatch):
    import threading

    import torch

    # The CPU build has no native NVTX binding. Represent only that boundary;
    # ServiceTrace, its lock, callback execution, serialization and joins are real.
    marked = []
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "mark",
        lambda text: marked.append((threading.get_native_id(), text)),
    )
    output = io.StringIO()
    callback_threads = []
    initialized = torch.cuda.is_initialized()

    def callback():
        callback_threads.append(threading.get_native_id())
        trace = ServiceTrace(output, run_id="run", nvtx_enabled=True)
        trace.dispatch(1, None)
        trace.commit(
            1,
            [("r", 3, 2, [11, 12], None)],
            [("r", 2, [11, 12], None)],
            owns_frontend=True,
        )
        trace.emit([("r", 2, [11, 12], None)])
        trace.release(["r"])

    worker = threading.Thread(target=callback)
    worker.start()
    worker.join()
    texts = output.getvalue().splitlines()
    assert [text for _, text in marked] == texts
    assert {tid for tid, _ in marked} == set(callback_threads)
    events = [json.loads(text) for text in texts]
    assert [event["boundary"] for event in events] == [
        "start",
        "dispatch",
        "commit",
        "output_ready",
        "frontend_send_returned",
    ]
    assert events[0]["nvtx"] is True
    assert events[-1]["generation"] == 3 and events[-1]["forward_id"] == 1
    assert all(event["schema"] == "tokenspeed.service/v1" for event in events)
    assert torch.cuda.is_initialized() == initialized


def test_unavailable_python_nvtx_fails_before_start_json():
    import torch

    if torch.version.cuda is not None:
        pytest.skip(
            "This negative case uses the CPU build's actual unavailable NVTX binding"
        )
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="NVTX functions not installed"):
        ServiceTrace(output, run_id="run", nvtx_enabled=True)
    assert output.getvalue() == ""
    assert not torch.cuda.is_initialized()


def test_disabled_and_json_only_trace_do_not_import_torch(tmp_path):
    import subprocess

    source = Path(sys.modules[ServiceTrace.__module__].__file__).resolve()
    script = r"""
import builtins,importlib.util,io,json,os,sys
from pathlib import Path
mode,source,output=sys.argv[1:]
if mode == 'package':
    # Existing package logging may load Torch through flash_attn. Bootstrap it
    # before measuring only the service module's disabled-path dependencies.
    import tokenspeed.runtime.engine
else:
    assert 'torch' not in sys.modules

original_import=builtins.__import__
original_import_module=importlib.import_module

def check_import(name):
    if name.split('.')[0] in {'torch','nvtx'}:
        raise AssertionError(f'disabled service marker imported {name}')

def guarded_import(name,*args,**kwargs):
    check_import(name)
    return original_import(name,*args,**kwargs)

def guarded_import_module(name,*args,**kwargs):
    check_import(name)
    return original_import_module(name,*args,**kwargs)

# Guard import requests themselves, including requests satisfied by a previously
# cached Torch module, instead of comparing only sys.modules before and after.
builtins.__import__=guarded_import
importlib.import_module=guarded_import_module
if mode == 'direct':
    spec=importlib.util.spec_from_file_location('isolated_service_trace',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    ServiceTrace=module.ServiceTrace
else:
    from tokenspeed.runtime.engine.service_trace import ServiceTrace

os.environ.pop('TOKENSPEED_SERVICE_TRACE',None)
os.environ.pop('TOKENSPEED_SERVICE_NVTX_LIBRARY',None)
os.environ['TOKENSPEED_SERVICE_NVTX']='1'
assert ServiceTrace.from_environment() is None
os.environ['TOKENSPEED_SERVICE_TRACE']=output
os.environ['TOKENSPEED_SERVICE_NVTX']='0'
os.environ['TOKENSPEED_PROFILE_RUN_ID']='json'
trace=ServiceTrace.from_environment()
trace.emit([('r',0,[],'stop')])
events=[json.loads(line) for line in Path(output).read_text().splitlines()]
assert events[0]['boundary']=='start' and events[0]['nvtx'] is False
assert events[-1]['boundary']=='frontend_send_returned'
if mode == 'direct':
    assert 'torch' not in sys.modules
"""
    for mode in ("direct", "package"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                mode,
                str(source),
                str(tmp_path / f"{mode}.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{mode}: {result.stderr}"


def test_removed_nvtx_library_selection_errors_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_SERVICE_TRACE", str(tmp_path / "service.jsonl"))
    monkeypatch.setenv("TOKENSPEED_SERVICE_NVTX_LIBRARY", "/obsolete/library.so")
    with pytest.raises(ValueError, match="was removed"):
        ServiceTrace.from_environment()


def test_marker_preparation_records_actual_python_sources_before_engine(
    tmp_path, monkeypatch
):
    import os

    import torch

    monkeypatch.setenv("TOKENSPEED_PROFILE_RUN_ID", "provider-fixture")
    monkeypatch.setenv("TOKENSPEED_SERVICE_NVTX", "0")
    initialized = torch.cuda.is_initialized()
    _baseline.prepare_service_nvtx(
        {"profile_command": ["nsys"]}, tmp_path, repetition=1
    )
    receipt = json.loads((tmp_path / "1-service-nvtx.json").read_text())
    assert receipt["selection"] == "torch.cuda.nvtx.mark" and receipt["enabled"] is True
    assert (
        receipt["process_id"] == os.getpid() and receipt["run_id"] == "provider-fixture"
    )
    assert receipt["torch_version"] == str(torch.__version__)
    assert len(receipt["sources"]) == 2
    for source in receipt["sources"]:
        assert _baseline.artifact_source(Path(source["path"])) == source
    assert torch.cuda.is_initialized() == initialized


def test_json_only_marker_preparation_does_not_import_runtime(tmp_path):
    import subprocess

    script = f"""
import importlib.util,json,os,sys
from pathlib import Path
path=Path({str(Path(__file__).resolve().parents[2] / 'examples/profiling/service_tpot.py')!r})
spec=importlib.util.spec_from_file_location('pilot',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
os.environ['TOKENSPEED_PROFILE_RUN_ID']='json-only'
module.prepare_service_nvtx({{'profile_command':[]}},Path({str(tmp_path)!r}),repetition=2)
assert 'torch' not in sys.modules
receipt=json.loads((Path({str(tmp_path)!r})/'2-service-nvtx.json').read_text())
assert receipt['selection']=='json_only' and receipt['enabled'] is False and receipt['sources']==[]
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
