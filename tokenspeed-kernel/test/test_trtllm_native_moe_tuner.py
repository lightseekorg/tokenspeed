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

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import tokenspeed_kernel.benchmark.trtllm_native_moe as tuner
import tokenspeed_kernel.thirdparty.trtllm_native_moe as native_moe
import torch
from tokenspeed_kernel.benchmark.trtllm_native_moe import (
    _DEFAULT_TOPOLOGIES,
    _canonical_sha256,
    _compatibility_identity,
    _merge_reports,
    _validate_plan,
    aggregate_tables,
    build_shared_confirmation_plan,
    finalize_shared_confirmation,
    main,
    make_dummy_topk,
    power_of_two_buckets,
    runtime_token_bucket,
    v4_moe_shape,
)


def _write_routing_trace_manifest(
    tmp_path,
    *,
    m_values: tuple[int, ...] = (4,),
    model_config_sha256: str = "model-sha",
):
    records = {}
    for num_tokens in m_values:
        record_path = tmp_path / f"route-m{num_tokens}.pt"
        topk_ids = torch.arange(num_tokens * 6, dtype=torch.int32).reshape(
            num_tokens, 6
        )
        torch.save(
            {
                "format": "tokenspeed-deepseek-v4-topk-v1",
                "num_tokens": num_tokens,
                "num_experts": 384,
                "top_k": 6,
                "layer_index": 3,
                "topk_weights": torch.full(
                    (num_tokens, 6), 1.0 / 6.0, dtype=torch.float32
                ),
                "topk_ids": topk_ids,
            },
            record_path,
        )
        records[str(num_tokens)] = {
            "path": record_path.name,
            "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "layer_index": 3,
            "role": "target_correction_bias",
        }
    manifest_path = tmp_path / "routing-trace.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "tokenspeed_deepseek_v4_routing_trace_manifest",
                "model_config_sha256": model_config_sha256,
                "num_experts": 384,
                "top_k": 6,
                "records": records,
            }
        )
    )
    return manifest_path


def test_routing_trace_manifest_binds_exact_input_contract(tmp_path) -> None:
    manifest_path = _write_routing_trace_manifest(tmp_path)
    manifest = tuner._load_routing_trace_manifest(manifest_path, required_m_values=(4,))
    config = {
        "routing_distribution": "trace",
        "seed": 0,
        "weight_data": "random",
        **tuner._routing_trace_config(manifest),
    }

    contract = tuner._input_generator_contract(v4_moe_shape("ep8", 7), 4, config)

    assert manifest.model_config_sha256 == "model-sha"
    assert manifest.records[4].topk_ids.shape == (4, 6)
    assert contract["routing"]["trace_manifest_sha256"] == manifest.sha256
    assert contract["routing"]["trace_record"] == {
        "sha256": manifest.records[4].file_sha256,
        "layer_index": 3,
        "role": "target_correction_bias",
    }


def test_routing_trace_manifest_rejects_tampering_and_invalid_values(tmp_path) -> None:
    manifest_path = _write_routing_trace_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    record_path = tmp_path / manifest["records"]["4"]["path"]
    record_path.write_bytes(record_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        tuner._load_routing_trace_manifest(manifest_path, required_m_values=(4,))

    manifest_path = _write_routing_trace_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    record_path = tmp_path / manifest["records"]["4"]["path"]
    payload = torch.load(record_path, map_location="cpu", weights_only=True)
    payload["topk_weights"][0, 0] = -1.0
    torch.save(payload, record_path)
    manifest["records"]["4"]["sha256"] = hashlib.sha256(
        record_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid values"):
        tuner._load_routing_trace_manifest(manifest_path, required_m_values=(4,))


def test_routing_trace_manifest_rejects_record_outside_its_directory(tmp_path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest_path = _write_routing_trace_manifest(manifest_dir)
    manifest = json.loads(manifest_path.read_text())
    record_path = manifest_dir / manifest["records"]["4"]["path"]
    outside_path = tmp_path / record_path.name
    record_path.rename(outside_path)
    manifest["records"]["4"].update(
        {
            "path": f"../{outside_path.name}",
            "sha256": hashlib.sha256(outside_path.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="escapes the manifest directory"):
        tuner._load_routing_trace_manifest(manifest_path, required_m_values=(4,))


def test_native_moe_tuner_uses_upstream_floor_buckets() -> None:
    assert power_of_two_buckets(9) == (1, 2, 4, 8)
    assert runtime_token_bucket(1) == 1
    assert runtime_token_bucket(3) == 2
    assert runtime_token_bucket(33) == 32
    assert runtime_token_bucket(8192) == 8192
    assert runtime_token_bucket(9000) == 8192
    assert native_moe._tactic_token_bucket(33) == runtime_token_bucket(33)


@pytest.mark.parametrize(("topology", "ep_rank"), [("tp8", 0), ("ep8", 0), ("ep8", 7)])
def test_balanced_dummy_topk_is_unique_and_deterministic(
    topology: str, ep_rank: int
) -> None:
    shape = v4_moe_shape(topology, ep_rank)
    first_weights, first_ids = make_dummy_topk(
        65, shape, distribution="balanced", seed=0, device="cpu"
    )
    second_weights, second_ids = make_dummy_topk(
        65, shape, distribution="balanced", seed=999, device="cpu"
    )

    assert torch.equal(first_weights, second_weights)
    assert torch.equal(first_ids, second_ids)
    assert first_ids.dtype == torch.int32
    assert first_ids.min() >= 0
    assert first_ids.max() < shape.num_experts
    assert torch.all(torch.sort(first_ids, dim=-1).values.diff(dim=-1) != 0)


def test_random_dummy_topk_is_seeded_and_unique() -> None:
    shape = v4_moe_shape("ep8", 3)
    first_weights, first_ids = make_dummy_topk(
        16, shape, distribution="random", seed=7, device="cpu"
    )
    second_weights, second_ids = make_dummy_topk(
        16, shape, distribution="random", seed=7, device="cpu"
    )

    assert torch.equal(first_weights, second_weights)
    assert torch.equal(first_ids, second_ids)
    assert torch.all(torch.sort(first_ids, dim=-1).values.diff(dim=-1) != 0)


def _profile(ep_rank: int, winner: list[int], scores: dict[tuple[int, int], float]):
    ordered = sorted(scores.items(), key=lambda item: item[1])
    shape = v4_moe_shape("ep8", ep_rank)
    return {
        "topology": "ep8",
        "ep_rank": ep_rank,
        "local_expert_offset": shape.local_expert_offset,
        "num_tokens": 32,
        "shape": {
            "topology": shape.topology,
            "ep_rank": shape.ep_rank,
            "num_experts": shape.num_experts,
            "local_num_experts": shape.local_num_experts,
            "local_expert_offset": shape.local_expert_offset,
            "top_k": shape.top_k,
            "hidden_size": shape.hidden_size,
            "intermediate_size": shape.intermediate_size,
            "valid_hidden_size": shape.valid_hidden_size,
            "valid_intermediate_size": shape.valid_intermediate_size,
        },
        "device": {"uuid": f"gpu-{ep_rank}"},
        "winner": winner,
        "publishable": True,
        "local_assignments": 1,
        "winner_gap_percent": 1.0,
        "results": [
            {
                "tactic": list(tactic),
                "median_ms": score,
                "samples_ms": [score, score, score],
                "confirmed": True,
                "verification": {"passed": True},
            }
            for tactic, score in ordered
        ],
    }


def test_shared_table_uses_worst_rank_and_requires_every_ep_rank() -> None:
    profiles = []
    for rank in range(8):
        scores = {
            (8, 1): 1.0 + rank * 0.1,
            (16, 2): 1.5,
        }
        profiles.append(_profile(rank, [8, 1], scores))

    tables = aggregate_tables(profiles)
    shared = tables["shared_worst_rank"]["ep8"]["32"]
    assert shared["complete"] is True
    assert shared["verified_on_all_ranks"] is True
    assert shared["tactic"] == [16, 2]
    assert shared["worst_rank_median_ms"] == 1.5

    partial = aggregate_tables(profiles[:-1])
    assert partial["shared_worst_rank"]["ep8"]["32"]["complete"] is False

    profiles[-1]["device"]["uuid"] = profiles[0]["device"]["uuid"]
    duplicate_device = aggregate_tables(profiles)
    assert duplicate_device["shared_worst_rank"]["ep8"]["32"]["complete"] is False


def test_shared_table_requires_idle_rank_verification() -> None:
    profiles = [_profile(rank, [8, 1], {(8, 1): 1.0 + rank * 0.1}) for rank in range(8)]
    idle = profiles[-1]
    idle["local_assignments"] = 0
    idle["results"][0]["verification"]["passed"] = False

    shared = aggregate_tables(profiles)["shared_worst_rank"]["ep8"]["32"]
    assert shared["complete"] is True
    assert shared["tactic"] is None
    assert shared["verified_on_all_ranks"] is False
    assert shared["publishable"] is False


def test_shared_table_ignores_unconfirmed_and_failed_tactics() -> None:
    profiles = [
        _profile(rank, [8, 1], {(8, 1): 1.0, (16, 2): 2.0}) for rank in range(8)
    ]
    for profile in profiles:
        profile["results"].append(
            {
                "tactic": [32, 3],
                "median_ms": 0.1,
                "confirmed": False,
                "verification": {"passed": True},
            }
        )
    profiles[0]["results"][0]["verification"]["passed"] = False

    shared = aggregate_tables(profiles)["shared_worst_rank"]["ep8"]["32"]
    assert shared["tactic"] == [16, 2]
    assert shared["verified_on_all_ranks"] is True


def _report(profile, *, dso_sha: str = "dso", tuner_sha: str = "tuner"):
    return {
        "provenance": {
            "upstream_trtllm_commit": "upstream",
            "tactic_abi": "abi",
            "dso": {"sha256": dso_sha},
            "vendored_source_sha256": "source",
            "tuner_sha256": tuner_sha,
            "hardware": {
                "uuid": profile["device"]["uuid"],
                "name": "B200",
                "compute_capability": [10, 0],
                "driver_version": "driver",
            },
            "software": {
                "torch": "torch",
                "torch_cuda": "cuda",
                "flashinfer": {"runtime_version": "flashinfer"},
            },
            "model_config": {"sha256": "model"},
        },
        "config": {
            "topologies": ["ep8"],
            "m_values": [32],
            "ep_ranks": [profile["ep_rank"]],
            "routing_distribution": "random",
            "seed": 0,
            "weight_data": "random",
            "warmup": 2,
            "repeat": 10,
            "samples": 5,
            "confirm_top_k": 8,
            "confirm_rounds": 5,
            "verify_output": True,
            "use_cuda_graph": True,
        },
        "profiles": [profile],
    }


def test_merge_rejects_incompatible_or_duplicate_reports(tmp_path) -> None:
    profile = _profile(0, [8, 1], {(8, 1): 1.0, (16, 2): 1.5})
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_report(profile)))
    second.write_text(json.dumps(_report(profile)))

    with pytest.raises(ValueError, match="duplicate"):
        _merge_reports([first, second])

    incompatible = _report(_profile(1, [8, 1], {(8, 1): 1.0}), dso_sha="other")
    second.write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match="different DSO"):
        _merge_reports([first, second])

    incomplete = _report(_profile(1, [8, 1], {(8, 1): 1.0}))
    incomplete["config"]["m_values"] = [16, 32]
    second.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="incomplete profile matrix"):
        _merge_reports([second])

    uuid_mismatch = _report(_profile(1, [8, 1], {(8, 1): 1.0}))
    uuid_mismatch["provenance"]["hardware"]["uuid"] = "wrong-gpu"
    second.write_text(json.dumps(uuid_mismatch))
    with pytest.raises(ValueError, match="does not match source report"):
        _merge_reports([second])


def test_merge_rejects_different_routing_trace_identity(tmp_path) -> None:
    first_report = _report(_profile(0, [8, 1], {(8, 1): 1.0}))
    second_report = _report(_profile(1, [8, 1], {(8, 1): 1.0}))
    for report, trace_sha in ((first_report, "trace-a"), (second_report, "trace-b")):
        report["config"].update(
            {
                "routing_distribution": "trace",
                "routing_trace_manifest_sha256": trace_sha,
                "routing_trace_records": {
                    "32": {
                        "sha256": "record",
                        "layer_index": 3,
                        "role": "target_correction_bias",
                    }
                },
            }
        )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(first_report))
    second.write_text(json.dumps(second_report))

    with pytest.raises(ValueError, match="different DSO or tuning settings"):
        _merge_reports([first, second])


def _initial_merged_report(
    score_factory=None,
):
    profiles = []
    for rank in range(8):
        scores = (
            score_factory(rank)
            if score_factory is not None
            else {(8, 1): 1.0 + rank * 0.01, (16, 2): 1.2}
        )
        profile = _profile(rank, [8, 1], scores)
        # Plan construction must use these raw samples, not the persisted
        # summaries inherited from the phase-one report.
        for result in profile["results"]:
            result["median_ms"] = 999.0
        profiles.append(profile)
    report = _report(profiles[0], tuner_sha="phase-one-tuner")
    report["config"]["ep_ranks"] = list(range(8))
    report["profiles"] = profiles
    return report


def _confirmation_report(
    initial_report,
    plan,
    rank: int,
    *,
    tuner_sha: str = "phase-two-tuner",
    failed_verification: tuple[int, int] | None = None,
    rounds: int = 3,
    samples_per_round: int = 3,
):
    provenance = json.loads(json.dumps(initial_report["provenance"]))
    provenance["tuner_sha256"] = tuner_sha
    provenance["hardware"]["uuid"] = f"gpu-{rank}"
    entry = plan["entries"][0]
    rank_contract = entry["rank_contracts"][str(rank)]
    results = []
    for candidate_index, candidate in enumerate(entry["candidates"]):
        tactic = tuple(candidate["tactic"])
        score = 1.0 + rank * 0.01 + candidate_index * 0.2
        results.append(
            {
                "tactic": list(tactic),
                "median_ms": 777.0,
                "round_samples_ms": [
                    [score] * samples_per_round for _ in range(rounds)
                ],
                "verification": {"passed": tactic != failed_verification},
            }
        )
    confirmation_config = {
        "shared_top_k": plan["shared_top_k"],
        "rounds": rounds,
        "samples_per_round": samples_per_round,
        "warmup": initial_report["config"]["warmup"],
        "repeat": initial_report["config"]["repeat"],
        "use_cuda_graph": initial_report["config"]["use_cuda_graph"],
    }
    if initial_report["config"].get("routing_trace_manifest_sha256") is not None:
        confirmation_config["routing_trace_manifest_sha256"] = initial_report["config"][
            "routing_trace_manifest_sha256"
        ]
    return {
        "schema_version": 1,
        "kind": "trtllm_native_moe_shared_confirmation_rank",
        "initial_report_sha256": plan["initial_report_sha256"],
        "plan": json.loads(json.dumps(plan)),
        "plan_sha256": plan["plan_sha256"],
        "compatibility_identity": plan["compatibility_identity"],
        "confirmation_config": confirmation_config,
        "ep_rank": rank,
        "provenance": provenance,
        "profiles": [
            {
                "topology": "ep8",
                "ep_rank": rank,
                "local_expert_offset": rank_contract["local_expert_offset"],
                "num_tokens": 32,
                "device": {"uuid": f"gpu-{rank}"},
                "shape": rank_contract["shape"],
                "input_generator": rank_contract["input_generator"],
                "planned_tactics": [
                    candidate["tactic"] for candidate in entry["candidates"]
                ],
                "confirmation_order": [
                    [candidate["tactic"] for candidate in entry["candidates"]]
                    + [[-1, -1]]
                    for _ in range(rounds)
                ],
                "fallback_round_samples_ms": [
                    [2.0 + rank * 0.01] * samples_per_round for _ in range(rounds)
                ],
                "fallback_error": False,
                "results": results,
            }
        ],
    }


def test_shared_confirmation_plan_is_deterministic_minimax_from_raw_samples() -> None:
    initial = _initial_merged_report(
        lambda rank: {
            (16, 2): 1.5 if rank == 0 else 1.0,
            (8, 1): 1.5 if rank == 7 else 1.0,
            (32, 3): 2.0,
        }
    )
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )

    assert [candidate["tactic"] for candidate in plan["entries"][0]["candidates"]] == [
        [8, 1],
        [16, 2],
    ]
    assert plan == build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    assert plan["plan_sha256"] == _canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def test_shared_confirmation_plan_hash_detects_modification() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    plan["entries"][0]["candidates"][0]["tactic"] = [99, 99]

    with pytest.raises(ValueError, match="plan hash mismatch"):
        _validate_plan(plan)


def test_shared_confirmation_plan_binds_rank_contracts_and_distinct_gpus() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    entry = plan["entries"][0]
    rank7 = entry["rank_contracts"]["7"]
    assert rank7["shape"]["local_expert_offset"] == 7 * 48
    assert rank7["local_expert_offset"] == 7 * 48
    assert rank7["input_generator"]["schema_version"] == 1
    assert rank7["input_generator"]["case_seed"] == 32

    initial["profiles"][-1]["device"]["uuid"] = "gpu-0"
    incomplete = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )["entries"][0]
    assert incomplete["initial_complete"] is False
    assert incomplete["candidates"] == []


def test_trace_plan_requires_phase_one_input_contract() -> None:
    initial = _initial_merged_report()
    initial["config"].update(
        {
            "routing_distribution": "trace",
            "routing_trace_manifest_sha256": "trace",
            "routing_trace_records": {
                "32": {
                    "sha256": "record",
                    "layer_index": 3,
                    "role": "target_correction_bias",
                }
            },
        }
    )
    for profile in initial["profiles"]:
        profile["input_generator"] = tuner._input_generator_contract(
            v4_moe_shape("ep8", profile["ep_rank"]), 32, initial["config"]
        )
    complete = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )["entries"][0]
    assert complete["initial_complete"] is True

    initial["profiles"][-1]["input_generator"]["routing"][
        "trace_manifest_sha256"
    ] = "other"
    incomplete = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )["entries"][0]
    assert incomplete["initial_complete"] is False
    assert incomplete["candidates"] == []


def test_rank_confirmation_times_fallback_in_every_round(monkeypatch) -> None:
    calls = []
    timing_args = []

    monkeypatch.setattr(tuner, "_allocate_case", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        tuner,
        "_run_tactic",
        lambda shape, weights, case, tactic: calls.append(tactic),
    )

    def profile(call, *, warmup, repeat, samples):
        timing_args.append((warmup, repeat, samples))
        call()
        return [1.0] * samples

    monkeypatch.setattr(tuner, "_profile_eager", profile)
    monkeypatch.setattr(tuner, "_verification_reference", lambda *args: (object(), 1))
    monkeypatch.setattr(tuner, "_verify_tactic", lambda *args: {"passed": True})
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="B200", uuid="gpu", major=10, minor=0),
    )

    profile_row = tuner._confirm_shared_profile(
        v4_moe_shape("ep8", 0),
        32,
        [(8, 1), (16, 2)],
        object(),
        torch.device("cpu"),
        distribution="random",
        seed=0,
        warmup=2,
        repeat=10,
        samples=3,
        rounds=3,
        use_cuda_graph=False,
        input_generator={"schema_version": 1},
    )

    assert calls.count((-1, -1)) == 3
    assert len(profile_row["fallback_round_samples_ms"]) == 3
    assert all([-1, -1] in order for order in profile_row["confirmation_order"])
    assert timing_args == [(2, 10, 3)] * 9


def test_finalize_shared_confirmation_uses_raw_samples_and_allows_new_tuner() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    reports = [_confirmation_report(initial, plan, rank) for rank in range(8)]

    finalized = finalize_shared_confirmation(
        initial,
        reports,
        initial_report_sha256="initial",
    )

    entry = finalized["entries"][0]
    assert entry["winner"] == plan["entries"][0]["candidates"][0]["tactic"]
    assert entry["winner_worst_rank_median_ms"] == pytest.approx(1.07)
    assert entry["gap_percent"] > 1.0
    assert entry["fallback_worst_rank_median_ms"] == pytest.approx(2.07)
    assert entry["winner_vs_fallback_speedup_percent"] > 1.0
    assert entry["publishable"] is True
    assert finalized["confirmation_tuner_sha256"] == "phase-two-tuner"
    # A single-M experiment is publishable as a point, but is not a complete
    # 14-bucket runtime table.
    assert finalized["runtime_tables"] == {}
    assert finalized["runtime_table_publishable"] is False
    assert finalized["runtime_table_status"]["ep8"]["has_all_14_buckets"] is False


def test_single_trace_per_m_is_not_a_generic_runtime_table() -> None:
    initial = _initial_merged_report()
    initial["config"].update(
        {
            "routing_distribution": "trace",
            "routing_trace_manifest_sha256": "trace",
            "routing_trace_records": {
                "32": {
                    "sha256": "record",
                    "layer_index": 3,
                    "role": "target_correction_bias",
                }
            },
        }
    )
    for profile in initial["profiles"]:
        profile["input_generator"] = tuner._input_generator_contract(
            v4_moe_shape("ep8", profile["ep_rank"]), 32, initial["config"]
        )
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    reports = [_confirmation_report(initial, plan, rank) for rank in range(8)]

    finalized = finalize_shared_confirmation(
        initial,
        reports,
        initial_report_sha256="initial",
    )

    assert finalized["entries"][0]["publishable"] is True
    status = finalized["runtime_table_status"]["ep8"]
    assert status["eligible_for_generic_runtime_table"] is False
    assert status["reason"] == "one exact routing trace per M is exploratory-only"
    assert finalized["runtime_table_publishable"] is False
    assert finalized["runtime_tables"] == {}


def test_finalize_shared_confirmation_reports_missing_rank_and_failed_verify() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    failed_tactic = tuple(plan["entries"][0]["candidates"][0]["tactic"])
    reports = [
        _confirmation_report(
            initial,
            plan,
            rank,
            failed_verification=failed_tactic if rank == 0 else None,
        )
        for rank in range(7)
    ]

    finalized = finalize_shared_confirmation(
        initial,
        reports,
        initial_report_sha256="initial",
    )
    entry = finalized["entries"][0]
    assert entry["complete"] is False
    assert entry["publishable"] is False
    assert any(
        "missing confirmation ranks" in reason for reason in entry["diagnostics"]
    )
    failed = next(
        candidate
        for candidate in entry["candidates"]
        if candidate["tactic"] == list(failed_tactic)
    )
    assert failed["successful_on_all_ranks"] is False
    assert failed["failures"]["0"] == "output verification failed"


def test_finalize_shared_confirmation_rejects_sub_one_percent_gap() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    reports = [_confirmation_report(initial, plan, rank) for rank in range(8)]
    for rank, report in enumerate(reports):
        first_score = 1.0 + rank * 0.01
        report["profiles"][0]["results"][1]["round_samples_ms"] = [
            [first_score * 1.005] * 3,
            [first_score * 1.005] * 3,
            [first_score * 1.005] * 3,
        ]

    entry = finalize_shared_confirmation(
        initial,
        reports,
        initial_report_sha256="initial",
    )["entries"][0]
    assert entry["gap_percent"] == pytest.approx(0.5)
    assert entry["publishable"] is False
    assert any("below 1%" in reason for reason in entry["diagnostics"])


def test_finalize_requires_one_percent_speedup_over_worst_fallback() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    reports = [_confirmation_report(initial, plan, rank) for rank in range(8)]
    for rank, report in enumerate(reports):
        fallback = (1.0 + rank * 0.01) * 1.005
        report["profiles"][0]["fallback_round_samples_ms"] = [
            [fallback] * 3 for _ in range(3)
        ]

    entry = finalize_shared_confirmation(
        initial,
        reports,
        initial_report_sha256="initial",
    )["entries"][0]
    assert entry["gap_percent"] > 1.0
    assert entry["winner_vs_fallback_speedup_percent"] < 1.0
    assert entry["publishable"] is False
    assert any("worst fallback" in reason for reason in entry["diagnostics"])


def test_finalize_rejects_result_set_or_profile_contract_mismatch() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    reports = [_confirmation_report(initial, plan, rank) for rank in range(8)]
    reports[0]["profiles"][0]["results"].pop()
    with pytest.raises(ValueError, match="result tactics do not exactly match"):
        finalize_shared_confirmation(
            initial,
            reports,
            initial_report_sha256="initial",
        )

    reports = [_confirmation_report(initial, plan, rank) for rank in range(8)]
    reports[0]["profiles"][0]["shape"]["hidden_size"] = 1
    with pytest.raises(ValueError, match="shape/input contract"):
        finalize_shared_confirmation(
            initial,
            reports,
            initial_report_sha256="initial",
        )


def test_finalize_low_sample_confirmation_is_smoke_only() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    reports = [
        _confirmation_report(
            initial,
            plan,
            rank,
            rounds=2,
            samples_per_round=2,
        )
        for rank in range(8)
    ]

    entry = finalize_shared_confirmation(
        initial,
        reports,
        initial_report_sha256="initial",
    )["entries"][0]
    assert entry["sampling_sufficient_for_publication"] is False
    assert entry["publishable"] is False
    assert any("smoke-only" in reason for reason in entry["diagnostics"])


def test_finalize_shared_confirmation_rejects_duplicate_or_mixed_tuner_shards() -> None:
    initial = _initial_merged_report()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256="initial",
        shared_top_k=2,
    )
    reports = [_confirmation_report(initial, plan, rank) for rank in range(8)]

    with pytest.raises(ValueError, match="duplicate confirmation shard"):
        finalize_shared_confirmation(
            initial,
            reports + [reports[0]],
            initial_report_sha256="initial",
        )

    reports[-1]["provenance"]["tuner_sha256"] = "different-phase-two-tuner"
    with pytest.raises(ValueError, match="different tuner SHA256"):
        finalize_shared_confirmation(
            initial,
            reports,
            initial_report_sha256="initial",
        )


def test_finalize_shared_confirmation_cli_does_not_require_cuda(tmp_path) -> None:
    initial = _initial_merged_report()
    initial_path = tmp_path / "initial.json"
    initial_path.write_text(json.dumps(initial))
    initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    plan = build_shared_confirmation_plan(
        initial,
        initial_report_sha256=initial_sha,
        shared_top_k=2,
    )
    report_paths = []
    for rank in range(8):
        path = tmp_path / f"rank-{rank}.json"
        path.write_text(json.dumps(_confirmation_report(initial, plan, rank)))
        report_paths.append(path)
    output = tmp_path / "final.json"

    assert (
        main(
            [
                "--output",
                str(output),
                "--finalize-shared-from",
                str(initial_path),
                "--confirmation-reports",
                *(str(path) for path in report_paths),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["entries"][0]["publishable"] is True


def test_cli_defaults_to_ep8_and_phase_one_requires_model_config(
    tmp_path, capsys
) -> None:
    assert _DEFAULT_TOPOLOGIES == ("ep8",)
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "phase-one.json")])
    assert "phase-one tuning requires --model-config" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        lambda path, other: ["--merge", str(path)],
        lambda path, other: ["--confirm-shared-from", str(path)],
        lambda path, other: [
            "--finalize-shared-from",
            str(path),
            "--confirmation-reports",
            str(other),
        ],
        lambda path, other: ["--model-config", str(path)],
        lambda path, other: ["--routing-trace-manifest", str(path)],
    ],
)
def test_all_cli_modes_reject_output_input_alias(arguments, tmp_path, capsys) -> None:
    path = tmp_path / "same.json"
    other = tmp_path / "other.json"
    with pytest.raises(SystemExit):
        main(["--output", str(path), *arguments(path, other)])
    assert "--output must not overwrite an input path" in capsys.readouterr().err


def test_cli_rejects_output_overwriting_routing_trace_record(tmp_path, capsys) -> None:
    manifest_path = _write_routing_trace_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    record_path = tmp_path / manifest["records"]["4"]["path"]
    model_config = tmp_path / "config.json"
    model_config.write_text("{}")

    with pytest.raises(SystemExit):
        main(
            [
                "--output",
                str(record_path),
                "--routing-trace-manifest",
                str(manifest_path),
                "--model-config",
                str(model_config),
                "--m-values",
                "4",
            ]
        )
    assert "must not overwrite a routing trace record" in capsys.readouterr().err


def test_compatibility_identity_excludes_only_tuner_sha() -> None:
    initial = _initial_merged_report()
    changed_tuner = json.loads(json.dumps(initial["provenance"]))
    changed_tuner["tuner_sha256"] = "new"
    assert _compatibility_identity(
        initial["provenance"], initial["config"]
    ) == _compatibility_identity(changed_tuner, initial["config"])

    changed_dso = json.loads(json.dumps(changed_tuner))
    changed_dso["dso"]["sha256"] = "new-dso"
    assert _compatibility_identity(
        initial["provenance"], initial["config"]
    ) != _compatibility_identity(changed_dso, initial["config"])
