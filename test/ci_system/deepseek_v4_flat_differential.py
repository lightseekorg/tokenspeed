#!/usr/bin/env python3
"""Deterministic DeepSeek V4 radix-vs-flat full-model acceptance runner.

The scheduler layout is a compile-time choice, so the parent process starts one
fresh Python worker per scheduler wheel and per scenario.  Workers instantiate
the regular :class:`tokenspeed.Engine`; they do not mock the scheduler, model,
or cache backend.  Results are compared by generated token ID, never decoded
text or an aggregate quality score.

This file is intentionally importable without torch or a GPU.  Manifest and
comparison tests exercise the control plane on developer machines; model
workers import TokenSpeed only after their scheduler-specific ``PYTHONPATH``
has been installed by ``install_scheduler_variants.sh``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "ci.tokenspeed.io/deepseek-v4-flat-differential-v1"
REQUIRED_TOPOLOGIES = {
    "flash-dp4-ep",
    "pro-tp8",
    "pro-tp8-ep-dense-tp1",
}
REQUIRED_WORKLOAD_KINDS = {
    "cold",
    "full_prefix",
    "partial_prefix",
    "chunked",
    "graph_prefill",
    "mixed",
    "reuse",
}


class DifferentialGateError(RuntimeError):
    """A fail-closed acceptance-gate violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DifferentialGateError(message)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the checked-in differential manifest."""

    data = json.loads(path.read_text())
    _require(isinstance(data, dict), "manifest must be a JSON object")
    _require(data.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    seed = data.get("seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer"
    )

    variants = data.get("scheduler_variants")
    _require(isinstance(variants, dict), "scheduler_variants must be an object")
    _require(
        set(variants) == {"radix", "flat"}, "scheduler_variants must be radix and flat"
    )
    for name, expected_flat in (("radix", False), ("flat", True)):
        variant = variants[name]
        _require(
            isinstance(variant, dict), f"scheduler variant {name} must be an object"
        )
        _require(
            variant.get("flat_kvcache") is expected_flat,
            f"scheduler variant {name} has the wrong flat_kvcache expectation",
        )
        _require(
            isinstance(variant.get("python_path"), str) and variant["python_path"],
            f"scheduler variant {name} needs python_path",
        )

    synthetic = data.get("synthetic_1gpu")
    _require(isinstance(synthetic, dict), "synthetic_1gpu contract is required")
    _require(
        synthetic.get("runner") == "b200-1gpu", "synthetic gate must target one GPU"
    )
    _require(
        synthetic.get("scheduler_variant") == "flat",
        "synthetic gate must use the flat extension",
    )
    _require(
        synthetic.get("required_chain")
        == [
            "cpp_flat_scheduler",
            "python_flat_forward_bridge",
            "v4_backend_metadata",
            "v4_cache_writers_readers",
        ],
        "synthetic gate must name the real end-to-end chain",
    )
    _require(
        set(synthetic.get("physical_data", ()))
        == {
            "swa_kv",
            "compressed_kv",
            "compressor_state",
            "indexer_kv",
            "indexer_compressor_state",
        },
        "synthetic gate must cover all five V4 physical data classes",
    )
    _require(
        set(synthetic.get("acceptance_lengths", ())) == {"zero", "partial", "full"},
        "synthetic gate must cover exact MTP acceptance lengths",
    )
    _require(
        synthetic.get("overlap_depths") == [0, 1],
        "synthetic gate must cover overlap 0/1",
    )
    _require(
        synthetic.get("acceptance_injection_seam") == "scheduler_structured_completion",
        "synthetic gate must inject deterministic ready completions at the "
        "scheduler structured-completion seam",
    )
    _require(
        synthetic.get("gpu_test") == "test/runtime/test_deepseek_v4_flat_synthetic.py",
        "synthetic gate must point to the executable GPU test",
    )
    _require(
        synthetic.get("ci_task")
        == "test/ci/ut/ut-deepseek-v4-flat-synthetic-b200.yaml",
        "synthetic gate must point to the B200 CI task",
    )
    _require(
        synthetic.get("requires_page_sentinels") is True,
        "synthetic gate needs page sentinels",
    )
    _require(
        synthetic.get("status") == "implemented",
        "synthetic gate must remain fail-closed until its executable GPU task exists",
    )

    topologies = data.get("topologies")
    _require(isinstance(topologies, dict), "topologies must be an object")
    _require(
        set(topologies) == REQUIRED_TOPOLOGIES,
        f"topologies must be exactly {sorted(REQUIRED_TOPOLOGIES)}",
    )
    for topology_id, topology in topologies.items():
        _require(
            isinstance(topology, dict), f"topology {topology_id} must be an object"
        )
        _require(
            isinstance(topology.get("runner"), str) and topology["runner"],
            f"topology {topology_id} needs a runner",
        )
        kwargs = topology.get("engine_kwargs")
        _require(
            isinstance(kwargs, dict), f"topology {topology_id} needs engine_kwargs"
        )
        _require(
            isinstance(kwargs.get("model"), str), f"topology {topology_id} needs model"
        )
        _require(
            kwargs.get("seed") == seed,
            f"topology {topology_id} must use the manifest seed",
        )
        _require(
            kwargs.get("temperature", None) is None,
            f"topology {topology_id} must not set sampling temperature",
        )
        _require(
            kwargs.get("disable_kvstore") is True,
            f"topology {topology_id} must disable L2/L3 KVStore",
        )
        _require(
            kwargs.get("enable_memory_saver") is True,
            f"topology {topology_id} must enable reset validation",
        )
        if int(kwargs.get("data_parallel_size", 1)) > 1:
            _require(
                kwargs.get("load_balance_method") == "round_robin",
                f"DP topology {topology_id} must use deterministic round_robin dispatch",
            )

    scenario_sets = data.get("scenario_sets")
    _require(
        isinstance(scenario_sets, dict) and scenario_sets,
        "scenario_sets must be non-empty",
    )
    scenarios: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for set_name, entries in scenario_sets.items():
        _require(
            isinstance(set_name, str) and set_name,
            "scenario set names must be non-empty",
        )
        _require(
            isinstance(entries, list) and entries,
            f"scenario set {set_name} must be non-empty",
        )
        for entry in entries:
            _require(
                isinstance(entry, dict), f"scenario in {set_name} must be an object"
            )
            scenario_id = entry.get("id")
            _require(
                isinstance(scenario_id, str) and scenario_id,
                "scenario id must be non-empty",
            )
            _require(
                scenario_id not in seen_ids, f"duplicate scenario id {scenario_id}"
            )
            seen_ids.add(scenario_id)
            _validate_scenario(entry)
            scenarios.append(entry)

    spec_matrix = {
        (bool(s["prefix_caching"]), int(s["overlap_depth"]), int(s["mtp_steps"]))
        for s in scenarios
        if int(s["mtp_steps"]) > 0
    }
    required_spec_matrix = {
        (prefix, overlap, steps)
        for prefix in (False, True)
        for overlap in (0, 1)
        for steps in (1, 3)
    }
    _require(
        required_spec_matrix <= spec_matrix,
        "MTP matrix must cover prefix on/off x overlap 0/1 x steps 1/3",
    )
    _require(
        {s["execution_mode"] for s in scenarios} == {"eager", "graph"},
        "scenario matrix must cover eager and CUDA graph",
    )
    _require(
        {s["indexer_layout"] for s in scenarios} == {"fp8", "mxfp4"},
        "scenario matrix must cover FP8 and MXFP4 indexer layouts",
    )

    workloads = data.get("workloads")
    _require(isinstance(workloads, list) and workloads, "workloads must be non-empty")
    kinds = set()
    workload_ids = set()
    max_dp_size = max(
        int(topology["engine_kwargs"].get("data_parallel_size", 1))
        for topology in topologies.values()
    )
    for workload in workloads:
        _require(isinstance(workload, dict), "workload entries must be objects")
        workload_id = workload.get("id")
        kind = workload.get("kind")
        _require(
            isinstance(workload_id, str) and workload_id,
            "workload id must be non-empty",
        )
        _require(
            workload_id not in workload_ids, f"duplicate workload id {workload_id}"
        )
        workload_ids.add(workload_id)
        _require(kind in REQUIRED_WORKLOAD_KINDS, f"unsupported workload kind {kind!r}")
        kinds.add(kind)
        prompts = workload.get("prompts")
        _require(
            isinstance(prompts, list)
            and prompts
            and all(isinstance(p, str) and p for p in prompts),
            f"workload {workload_id} needs non-empty prompts",
        )
        prefix = workload.get("common_prefix", "")
        prefix_repeats = workload.get("common_prefix_repeats", 0)
        _require(
            isinstance(prefix, str),
            f"workload {workload_id} common_prefix must be a string",
        )
        _require(
            isinstance(prefix_repeats, int)
            and not isinstance(prefix_repeats, bool)
            and prefix_repeats >= 0,
            f"workload {workload_id} common_prefix_repeats must be >= 0",
        )
        repeat_counts = workload.get("prompt_repeat_counts", [1] * len(prompts))
        _require(
            isinstance(repeat_counts, list)
            and len(repeat_counts) == len(prompts)
            and all(
                isinstance(count, int) and not isinstance(count, bool) and count > 0
                for count in repeat_counts
            ),
            f"workload {workload_id} prompt_repeat_counts must align with prompts",
        )
        _require(
            isinstance(workload.get("max_new_tokens"), int)
            and workload["max_new_tokens"] > 0,
            f"workload {workload_id} needs max_new_tokens > 0",
        )
        if kind == "graph_prefill":
            _require(
                workload.get("requires_all_dp_extend") is True
                and workload.get("batched_dispatch") is True,
                "graph_prefill must declare its batched all-DP-extend contract",
            )
            _require(
                len(prompts) >= 2 * max_dp_size,
                "graph_prefill needs at least two long requests per DP rank",
            )
    _require(
        kinds == REQUIRED_WORKLOAD_KINDS, "workloads do not cover the required corpus"
    )
    reset_prompt = data.get("reset_probe_prompt")
    _require(
        isinstance(reset_prompt, str) and reset_prompt, "reset_probe_prompt is required"
    )
    return data


def _validate_scenario(scenario: dict[str, Any]) -> None:
    _require(
        scenario.get("execution_mode") in {"eager", "graph"}, "invalid execution_mode"
    )
    _require(
        type(scenario.get("prefix_caching")) is bool, "prefix_caching must be boolean"
    )
    _require(
        scenario.get("indexer_layout") in {"fp8", "mxfp4"}, "invalid indexer_layout"
    )
    _require(scenario.get("overlap_depth") in {0, 1}, "overlap_depth must be 0 or 1")
    _require(scenario.get("mtp_steps") in {0, 1, 3}, "mtp_steps must be 0, 1, or 3")


def selected_scenarios(
    manifest: dict[str, Any], scenario_sets: Iterable[str]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for name in scenario_sets:
        try:
            selected.extend(manifest["scenario_sets"][name])
        except KeyError as exc:
            raise DifferentialGateError(f"unknown scenario set {name!r}") from exc
    _require(bool(selected), "at least one scenario must be selected")
    return selected


def scenario_engine_kwargs(
    manifest: dict[str, Any], topology_id: str, scenario: dict[str, Any]
) -> dict[str, Any]:
    """Materialize one deterministic Engine configuration."""

    kwargs = copy.deepcopy(manifest["topologies"][topology_id]["engine_kwargs"])
    kwargs["enforce_eager"] = scenario["execution_mode"] == "eager"
    kwargs["enable_prefix_caching"] = bool(scenario["prefix_caching"])
    kwargs["disable_overlap_schedule"] = scenario["overlap_depth"] == 0
    kwargs["attention_use_fp4_indexer_cache"] = scenario["indexer_layout"] == "mxfp4"
    mtp_steps = int(scenario["mtp_steps"])
    if mtp_steps:
        kwargs["speculative_algorithm"] = "MTP"
        kwargs["speculative_num_steps"] = mtp_steps
    else:
        kwargs["speculative_algorithm"] = None
    return kwargs


def compare_variant_results(radix: dict[str, Any], flat: dict[str, Any]) -> None:
    """Fail on any deterministic token-ID drift or missing flat proof."""

    _require(radix.get("variant") == "radix", "radix worker result is mislabeled")
    _require(flat.get("variant") == "flat", "flat worker result is mislabeled")
    _require(radix.get("scenario") == flat.get("scenario"), "scenario mismatch")
    _require(radix.get("topology") == flat.get("topology"), "topology mismatch")
    _require(
        radix.get("token_ids") == flat.get("token_ids"),
        "radix/flat output_ids mismatch",
    )
    _require(
        radix.get("cached_tokens") == flat.get("cached_tokens"),
        "radix/flat cached_tokens mismatch",
    )
    _require(
        radix.get("acceptances") == flat.get("acceptances"),
        "radix/flat speculative acceptance mismatch",
    )
    radix_graph_proof = radix.get("graph_proof")
    flat_graph_proof = flat.get("graph_proof")
    for variant, graph_proof in (
        ("radix", radix_graph_proof),
        ("flat", flat_graph_proof),
    ):
        _require(
            isinstance(graph_proof, dict) and graph_proof.get("validated") is True,
            f"{variant} worker did not return validated CUDA graph evidence",
        )
    _require(
        radix_graph_proof.get("execution_mode")
        == flat_graph_proof.get("execution_mode"),
        "radix/flat CUDA graph evidence mode mismatch",
    )
    proof = flat.get("flat_proof")
    _require(isinstance(proof, dict), "flat worker did not return a proof")
    _require(
        isinstance(proof.get("fingerprint"), str) and len(proof["fingerprint"]) == 64,
        "flat plan fingerprint missing",
    )
    _require(
        proof.get("generation_delta") == 1,
        "flat reset did not advance exactly one generation",
    )
    _require(
        proof.get("all_pools_reset") is True, "flat pools did not return to baseline"
    )
    _require(
        proof.get("post_wake_token_exact") is True,
        "flat wake changed deterministic output_ids",
    )


def _sampling(max_new_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_k": 1,
        "max_new_tokens": max_new_tokens,
    }


def _normalize_response(response: Any) -> list[dict[str, Any]]:
    responses = response if isinstance(response, list) else [response]
    _require(
        responses and all(isinstance(item, dict) for item in responses),
        "Engine.generate returned an invalid response",
    )
    for item in responses:
        ids = item.get("output_ids")
        _require(
            isinstance(ids, list)
            and all(
                isinstance(token, int) and not isinstance(token, bool) for token in ids
            ),
            "Engine.generate response is missing integer output_ids",
        )
    return responses


def _run_workload(
    engine: Any, workload: dict[str, Any]
) -> tuple[list[list[int]], list[float], list[int]]:
    common = workload.get("common_prefix", "") * int(
        workload.get("common_prefix_repeats", 0)
    )
    repeat_counts = workload.get("prompt_repeat_counts", [1] * len(workload["prompts"]))
    prompts = [
        common + prompt * int(repeat)
        for prompt, repeat in zip(workload["prompts"], repeat_counts)
    ]
    sampling = _sampling(int(workload["max_new_tokens"]))
    kind = workload["kind"]
    responses: list[dict[str, Any]] = []
    if kind in {"cold", "full_prefix", "partial_prefix", "reuse", "chunked"}:
        for prompt in prompts:
            responses.extend(
                _normalize_response(
                    engine.generate(
                        prompt=prompt, sampling_params=sampling, stream=False
                    )
                )
            )
    elif kind == "graph_prefill":
        # AsyncLLM tokenizes and sends every item from one batch before awaiting
        # any response. Deterministic round-robin DP dispatch therefore lands two
        # multi-chunk extends on every rank, leaving a shared all-extend iteration
        # even if the first four requests reach their schedulers at different times.
        responses.extend(
            _normalize_response(
                engine.generate(
                    prompt=prompts,
                    sampling_params=[copy.deepcopy(sampling) for _ in prompts],
                    stream=False,
                )
            )
        )
    elif kind == "mixed":
        # Concurrent facade calls create one long-prefill/short-decode pressure
        # window without relying on sleeps or prompt-dependent acceptance.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [
                pool.submit(
                    engine.generate,
                    prompt=prompt,
                    sampling_params=sampling,
                    stream=False,
                )
                for prompt in prompts
            ]
            for future in futures:
                responses.extend(_normalize_response(future.result()))
    else:  # pragma: no cover - manifest validation makes this unreachable.
        raise DifferentialGateError(f"unsupported workload kind {kind!r}")

    token_ids = [list(response["output_ids"]) for response in responses]
    acceptances: list[float] = []
    cached_tokens: list[int] = []
    for response in responses:
        meta = response.get("meta_info") or {}
        value = meta.get("accept_draft_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            acceptances.append(float(value))
        cached = meta.get("cached_tokens", 0)
        _require(
            isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0,
            "Engine.generate returned invalid cached_tokens metadata",
        )
        cached_tokens.append(cached)
    return token_ids, acceptances, cached_tokens


def _internal_states(engine: Any) -> list[dict[str, Any]]:
    info = engine.get_server_info()
    states = info.get("internal_states") if isinstance(info, dict) else None
    _require(
        isinstance(states, list)
        and states
        and all(isinstance(state, dict) for state in states),
        "get_server_info.internal_states is missing",
    )
    return states


def _assert_runtime_mode(
    states: list[dict[str, Any]], *, expected_flat: bool, overlap_depth: int
) -> None:
    for rank, state in enumerate(states):
        _require(
            state.get("scheduler_flat_kvcache") is expected_flat,
            f"rank {rank} scheduler build does not match requested variant",
        )
        _require(
            state.get("overlap_schedule_depth") == overlap_depth,
            f"rank {rank} overlap depth was silently changed",
        )


def _assert_graph_evidence(
    states: list[dict[str, Any]],
    *,
    scenario: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the requested eager/graph route actually executed."""

    execution_mode = scenario["execution_mode"]
    expect_drafter = int(scenario["mtp_steps"]) > 0
    decode_replayed = False
    idle_replayed = False
    prefill_replayed = False
    for rank, state in enumerate(states):
        evidence = state.get("cuda_graph_evidence")
        _require(
            isinstance(evidence, dict),
            f"rank {rank} CUDA graph evidence is missing",
        )
        decode = evidence.get("decode")
        prefill = evidence.get("prefill")
        _require(
            isinstance(decode, dict) and isinstance(prefill, dict),
            f"rank {rank} CUDA graph evidence has an invalid shape",
        )
        captured_keys = decode.get("captured_keys")
        captured_buckets = prefill.get("captured_buckets")
        replayed_kinds = prefill.get("replayed_kinds")
        _require(
            isinstance(captured_keys, list)
            and isinstance(captured_buckets, list)
            and isinstance(replayed_kinds, list),
            f"rank {rank} CUDA graph evidence fields are invalid",
        )
        rank_decode_replayed = decode.get("decode_replayed") is True
        rank_idle_replayed = decode.get("idle_replayed") is True
        rank_prefill_replayed = "text" in replayed_kinds
        _require(
            evidence.get("drafter_present") is expect_drafter,
            f"rank {rank} drafter mode differs from the requested MTP scenario",
        )

        if execution_mode == "eager":
            _require(
                not captured_keys
                and not captured_buckets
                and not rank_decode_replayed
                and not rank_idle_replayed
                and not replayed_kinds,
                f"rank {rank} eager scenario unexpectedly used CUDA graph",
            )
            continue

        _require(bool(captured_keys), f"rank {rank} decode graph was not captured")
        _require(
            bool(captured_buckets),
            f"rank {rank} breakable prefill graph was not captured",
        )
        _require(
            rank_decode_replayed or rank_idle_replayed,
            f"rank {rank} never replayed decode or idle CUDA graph",
        )
        _require(
            rank_prefill_replayed,
            f"rank {rank} never replayed breakable prefill CUDA graph",
        )
        decode_replayed = decode_replayed or rank_decode_replayed
        idle_replayed = idle_replayed or rank_idle_replayed
        prefill_replayed = prefill_replayed or rank_prefill_replayed

    if execution_mode == "graph":
        _require(decode_replayed, "graph scenario never replayed target decode")
        _require(prefill_replayed, "graph scenario silently fell back for prefill")
        dp_size = int(topology["engine_kwargs"].get("data_parallel_size", 1))
        if dp_size > 1:
            _require(idle_replayed, "DP graph scenario never replayed an idle rank")

    return {
        "execution_mode": execution_mode,
        "validated": True,
        "decode_replayed": decode_replayed,
        "idle_replayed": idle_replayed,
        "prefill_replayed": prefill_replayed,
    }


def _flat_proof_before(
    states: list[dict[str, Any]],
) -> tuple[int, str, list[dict[str, Any]]]:
    fingerprints = {state.get("flat_plan_fingerprint") for state in states}
    _require(
        len(fingerprints) == 1, "flat plan fingerprint differs across attention ranks"
    )
    fingerprint = next(iter(fingerprints))
    _require(
        isinstance(fingerprint, str) and len(fingerprint) == 64,
        "flat plan fingerprint missing",
    )

    plans = [state.get("flat_kv_plan") for state in states]
    _require(all(isinstance(plan, dict) for plan in plans), "flat plan summary missing")
    canonical_plan = plans[0]
    _require(
        all(plan == canonical_plan for plan in plans),
        "flat pool capacity/schema differs across attention ranks",
    )
    pools = canonical_plan.get("pools")
    _require(
        isinstance(pools, list) and len(pools) >= 2,
        "V4 flat plan must use heterogeneous pools",
    )
    pool_ids = [pool.get("pool_id") for pool in pools]
    _require(
        len(pool_ids) == len(set(pool_ids)), "flat pools need independent page domains"
    )

    generations = {state.get("flat_kv_generation") for state in states}
    _require(len(generations) == 1, "flat generation differs across attention ranks")
    generation = next(iter(generations))
    _require(
        isinstance(generation, int) and not isinstance(generation, bool),
        "flat generation missing",
    )
    _require(
        all(state.get("flat_kv_quiescent") is True for state in states),
        "flat scheduler is not quiescent",
    )
    return generation, fingerprint, pools


def _assert_pools_reset(
    states: list[dict[str, Any]], pools: list[dict[str, Any]]
) -> int:
    expected_pool_ids = [pool.get("pool_id") for pool in pools]
    _require(
        all(isinstance(pool_id, str) and pool_id for pool_id in expected_pool_ids),
        "flat plan pool IDs must be non-empty strings",
    )
    _require(
        len(expected_pool_ids) == len(set(expected_pool_ids)),
        "flat plan contains duplicate pool IDs",
    )
    expected_capacity = {
        pool["pool_id"]: (pool["total_blocks"], pool["bytes_per_block"])
        for pool in pools
    }
    expected_pool_id_set = set(expected_pool_ids)
    generations = set()
    for rank, state in enumerate(states):
        generations.add(state.get("flat_kv_generation"))
        _require(
            state.get("flat_kv_quiescent") is True,
            f"rank {rank} is not quiescent after reset",
        )
        snapshots = state.get("flat_pool_snapshots")
        _require(
            isinstance(snapshots, list) and snapshots,
            f"rank {rank} pool snapshots missing",
        )
        snapshot_pool_ids = [snapshot.get("pool_id") for snapshot in snapshots]
        _require(
            all(isinstance(pool_id, str) and pool_id for pool_id in snapshot_pool_ids),
            f"rank {rank} returned an invalid pool ID",
        )
        _require(
            len(snapshot_pool_ids) == len(set(snapshot_pool_ids)),
            f"rank {rank} returned duplicate pool snapshots",
        )
        _require(
            set(snapshot_pool_ids) == expected_pool_id_set,
            f"rank {rank} pool snapshot set differs from the flat plan: "
            f"missing={sorted(expected_pool_id_set - set(snapshot_pool_ids))}, "
            f"extra={sorted(set(snapshot_pool_ids) - expected_pool_id_set)}",
        )
        for snapshot in snapshots:
            pool_id = snapshot.get("pool_id")
            _require(
                pool_id in expected_capacity,
                f"rank {rank} returned an unknown pool {pool_id!r}",
            )
            total, bytes_per_block = expected_capacity[pool_id]
            _require(
                snapshot.get("total_blocks") == total,
                f"rank {rank} pool capacity changed",
            )
            _require(
                snapshot.get("bytes_per_block") == bytes_per_block,
                f"rank {rank} pool layout changed",
            )
            _require(
                snapshot.get("free_blocks") == snapshot.get("usable_blocks"),
                f"rank {rank} pool {pool_id} leaked blocks",
            )
            for counter in (
                "active_blocks",
                "cached_evictable_blocks",
                "pinned_cached_blocks",
                "reserved_blocks",
            ):
                _require(
                    snapshot.get(counter) == 0,
                    f"rank {rank} pool {pool_id} retained {counter}",
                )
        _require(
            state.get("flat_arena_generation") == state.get("flat_kv_generation"),
            f"rank {rank} scheduler/arena generation mismatch",
        )
    _require(len(generations) == 1, "post-reset flat generation differs across ranks")
    generation = next(iter(generations))
    _require(
        isinstance(generation, int) and not isinstance(generation, bool),
        "post-reset generation missing",
    )
    return generation


def run_worker(
    manifest: dict[str, Any], topology_id: str, scenario: dict[str, Any], variant: str
) -> dict[str, Any]:
    expected_flat = bool(manifest["scheduler_variants"][variant]["flat_kvcache"])
    import tokenspeed_scheduler

    _require(
        bool(getattr(tokenspeed_scheduler, "FLAT_KVCACHE", False)) is expected_flat,
        f"loaded tokenspeed_scheduler extension is not the {variant} build",
    )
    from tokenspeed import Engine

    kwargs = scenario_engine_kwargs(manifest, topology_id, scenario)
    engine = Engine(**kwargs)
    try:
        initial_states = _internal_states(engine)
        _assert_runtime_mode(
            initial_states,
            expected_flat=expected_flat,
            overlap_depth=int(scenario["overlap_depth"]),
        )
        token_ids: dict[str, list[list[int]]] = {}
        acceptances: dict[str, list[float]] = {}
        cache_hits: dict[str, list[int]] = {}
        for workload in manifest["workloads"]:
            ids, rates, cached = _run_workload(engine, workload)
            token_ids[workload["id"]] = ids
            cache_hits[workload["id"]] = cached
            if rates:
                acceptances[workload["id"]] = rates

        if int(scenario["mtp_steps"]) > 0:
            _require(bool(acceptances), "MTP run did not report actual acceptance")
        if scenario["prefix_caching"]:
            for workload_id in ("full-prefix", "partial-prefix", "decode-page-reuse"):
                hits = cache_hits[workload_id]
                _require(
                    len(hits) >= 2 and max(hits[1:]) > 0,
                    f"prefix-enabled workload {workload_id} did not hit the cache",
                )
        else:
            _require(
                all(hit == 0 for hits in cache_hits.values() for hit in hits),
                "prefix-disabled scenario reported cached tokens",
            )

        graph_proof = _assert_graph_evidence(
            _internal_states(engine),
            scenario=scenario,
            topology=manifest["topologies"][topology_id],
        )

        proof = None
        if expected_flat:
            before_states = _internal_states(engine)
            generation_before, fingerprint, pools = _flat_proof_before(before_states)
            probe_sampling = _sampling(16)
            before_probe = _normalize_response(
                engine.generate(
                    prompt=manifest["reset_probe_prompt"],
                    sampling_params=probe_sampling,
                    stream=False,
                )
            )[0]["output_ids"]
            engine.release_memory_occupation(tags=["kv_cache"])
            _require(
                engine.is_sleeping() is True,
                "flat KV release did not enter sleeping state",
            )
            engine.resume_memory_occupation(tags=["kv_cache"])
            _require(
                engine.is_sleeping() is False, "flat KV resume did not wake the engine"
            )
            after_states = _internal_states(engine)
            generation_after = _assert_pools_reset(after_states, pools)
            after_probe = _normalize_response(
                engine.generate(
                    prompt=manifest["reset_probe_prompt"],
                    sampling_params=probe_sampling,
                    stream=False,
                )
            )[0]["output_ids"]
            proof = {
                "fingerprint": fingerprint,
                "generation_delta": generation_after - generation_before,
                "all_pools_reset": True,
                "post_wake_token_exact": before_probe == after_probe,
            }

        return {
            "variant": variant,
            "topology": topology_id,
            "scenario": scenario["id"],
            "token_ids": token_ids,
            "acceptances": acceptances,
            "cached_tokens": cache_hits,
            "graph_proof": graph_proof,
            "flat_proof": proof,
        }
    finally:
        engine.shutdown()


def _run_worker_process(
    *,
    repo_root: Path,
    manifest_path: Path,
    topology_id: str,
    scenario: dict[str, Any],
    variant: str,
    variant_config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    variant_path = (repo_root / variant_config["python_path"]).resolve()
    _require(
        variant_path.is_dir(), f"scheduler variant path does not exist: {variant_path}"
    )
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(variant_path) + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--manifest",
        str(manifest_path),
        "--topology",
        topology_id,
        "--scenario-id",
        scenario["id"],
        "--variant",
        variant,
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=repo_root, env=env, text=True)
    _require(completed.returncode == 0, f"{variant} worker failed for {scenario['id']}")
    _require(output_path.is_file(), f"{variant} worker did not write {output_path}")
    return json.loads(output_path.read_text())


def run_parent(
    manifest_path: Path, topology_id: str, scenario_sets: Iterable[str], repo_root: Path
) -> None:
    manifest = load_manifest(manifest_path)
    _require(topology_id in manifest["topologies"], f"unknown topology {topology_id!r}")
    selected = selected_scenarios(manifest, scenario_sets)
    with tempfile.TemporaryDirectory(prefix="deepseek-v4-flat-diff-") as temp_dir:
        temp = Path(temp_dir)
        for scenario in selected:
            results = {}
            for variant in ("radix", "flat"):
                results[variant] = _run_worker_process(
                    repo_root=repo_root,
                    manifest_path=manifest_path,
                    topology_id=topology_id,
                    scenario=scenario,
                    variant=variant,
                    variant_config=manifest["scheduler_variants"][variant],
                    output_path=temp / f"{scenario['id']}-{variant}.json",
                )
            compare_variant_results(results["radix"], results["flat"])
            print(
                f"[deepseek-v4-flat-differential] PASS {topology_id}/{scenario['id']}",
                flush=True,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--scenario-set", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scenario-id")
    parser.add_argument("--variant", choices=("radix", "flat"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = args.manifest.resolve()
    try:
        manifest = load_manifest(manifest_path)
        if args.worker:
            _require(args.scenario_id is not None, "worker requires --scenario-id")
            _require(args.variant is not None, "worker requires --variant")
            _require(args.output is not None, "worker requires --output")
            scenarios = [
                scenario
                for entries in manifest["scenario_sets"].values()
                for scenario in entries
                if scenario["id"] == args.scenario_id
            ]
            _require(len(scenarios) == 1, f"unknown scenario id {args.scenario_id!r}")
            result = run_worker(manifest, args.topology, scenarios[0], args.variant)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            return 0
        scenario_sets = args.scenario_set or list(manifest["scenario_sets"])
        run_parent(
            manifest_path, args.topology, scenario_sets, args.repo_root.resolve()
        )
        return 0
    except (DifferentialGateError, KeyError, ValueError) as exc:
        print(
            f"[deepseek-v4-flat-differential] FAIL: {exc}", file=sys.stderr, flush=True
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
