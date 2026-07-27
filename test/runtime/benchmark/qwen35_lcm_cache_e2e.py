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

"""Deterministic cache-hit and capacity comparison for Qwen3.5 Flat LCM.

The trace is compact: it stores boundary lengths and request counts, while this
module derives exact token IDs from a frozen integer generator. The same trace
therefore runs unchanged on a frozen-main Radix checkout, branch Radix, and
branch Flat LCM without depending on tokenizer behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping


_ARM_NAMES = ("main_radix", "branch_radix", "branch_flat_lcm")
_ROUND_NAMES = ("prime", "replay")
_MIN_RADIX_RESIDENT_HIT_RATE = 0.7
_MIN_FLAT_RESIDENT_HIT_RATE = 0.9
_MAX_RADIX_CLIFF_HIT_RATE = 0.2


def _positive_int(value, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _working_set_tokens(
    counts: list[int],
    prompt_lengths: list[int],
    boundary_indices: list[int],
) -> int:
    total = sum(
        count * length for count, length in zip(counts, prompt_lengths)
    )
    if boundary_indices:
        shared_lengths = [prompt_lengths[index] for index in boundary_indices]
        total -= sum(shared_lengths) - max(shared_lengths)
    return total


def _bucket_counts(
    target_tokens: int,
    prompt_lengths: list[int],
    boundary_indices: list[int],
) -> list[int]:
    """Cover every frozen boundary once, then add only long requests."""
    counts = [1] * len(prompt_lengths)
    minimum = _working_set_tokens(counts, prompt_lengths, boundary_indices)
    if target_tokens < minimum:
        raise ValueError(
            f"working set {target_tokens} cannot cover boundary buckets "
            f"requiring {minimum} tokens"
        )
    remaining = target_tokens - minimum
    bulk = max(range(len(prompt_lengths)), key=prompt_lengths.__getitem__)
    counts[bulk] += remaining // prompt_lengths[bulk]
    return counts


def _flat_capacity_model(
    spec: Mapping, flat_probe: Mapping
) -> tuple[int, int, int]:
    """Return safe tokens, State checkpoints/request, and admission parents."""
    geometry = flat_probe.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Flat probe is missing LCM geometry")
    block_tokens = _positive_int(
        spec["logical_block_tokens"], "logical_block_tokens"
    )
    if _positive_int(
        geometry.get("logical_block_tokens"), "geometry logical block tokens"
    ) != block_tokens:
        raise ValueError("trace and Flat geometry use different logical blocks")
    packings = geometry.get("cache_blocks_per_lcm_block")
    if not isinstance(packings, Mapping):
        raise ValueError("Flat geometry is missing group packing")
    full_packing = _positive_int(
        packings.get("full_attention"), "Full cache blocks per LCM block"
    )
    if full_packing != _positive_int(
        spec["full_cache_blocks_per_lcm_block"],
        "full_cache_blocks_per_lcm_block",
    ):
        raise ValueError("trace and Flat geometry use different Full packing")
    state_packings = [
        _positive_int(value, f"{name} cache blocks per LCM block")
        for name, value in packings.items()
        if str(name).startswith("linear_attention")
    ]
    state_group_count = _positive_int(
        spec["state_group_count"], "state_group_count"
    )
    if len(state_packings) != state_group_count:
        raise ValueError("trace and Flat geometry use different State group counts")

    capacity_bucket = str(spec["capacity_prompt_bucket"])
    try:
        capacity_spec = next(
            bucket
            for bucket in spec["buckets"]
            if bucket["name"] == capacity_bucket
        )
    except StopIteration as exc:
        raise ValueError(
            f"unknown capacity_prompt_bucket {capacity_bucket!r}"
        ) from exc
    prompt_pages = _positive_int(
        capacity_spec["pages"], f"{capacity_bucket} pages"
    )
    tail_tokens = int(capacity_spec.get("tail_tokens", 0))
    if not 0 <= tail_tokens < block_tokens:
        raise ValueError(
            f"{capacity_bucket} tail_tokens must be in [0, {block_tokens})"
        )
    prompt_tokens = prompt_pages * block_tokens + tail_tokens
    state_checkpoints_per_request = 1 if tail_tokens else 2
    num_parents = _positive_int(
        geometry.get("num_lcm_blocks"), "number of LCM blocks"
    )
    scheduled_tokens = _positive_int(
        spec["max_scheduled_tokens"], "max_scheduled_tokens"
    )
    decode_reserve_tokens = int(spec.get("decode_reserve_tokens", 0))
    if decode_reserve_tokens < 0:
        raise ValueError("decode_reserve_tokens must be >= 0")
    transient_blocks = (
        scheduled_tokens + decode_reserve_tokens + block_tokens - 1
    ) // block_tokens
    transient_parents = (
        transient_blocks + full_packing - 1
    ) // full_packing
    transient_parents += sum(
        (transient_blocks + packing - 1) // packing
        for packing in state_packings
    )
    resident_parents = num_parents - transient_parents
    if resident_parents <= 0:
        raise ValueError("LCM pool cannot hold one maximum admission chunk")

    def parents_needed(num_requests: int) -> int:
        full_blocks = num_requests * prompt_pages
        parents = (full_blocks + full_packing - 1) // full_packing
        for packing in state_packings:
            # Matching excludes the final prompt token. A page-aligned exact
            # replay therefore needs both the live frontier and its predecessor;
            # any tail makes the live frontier itself matchable.
            state_blocks = state_checkpoints_per_request * num_requests
            parents += (state_blocks + packing - 1) // packing
        return parents

    low = 0
    high = resident_parents * max([full_packing, *state_packings]) + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if parents_needed(middle) <= resident_parents:
            low = middle
        else:
            high = middle
    return low * prompt_tokens, state_checkpoints_per_request, transient_parents


def build_trace(
    spec: Mapping,
    *,
    radix_capacity_tokens: int,
    lcm_capacity_tokens: int,
) -> dict:
    """Resolve a compact trace template against measured arm capacities."""
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("unsupported trace schema")
    radix_capacity = _positive_int(radix_capacity_tokens, "radix capacity")
    lcm_capacity = _positive_int(lcm_capacity_tokens, "LCM capacity")
    if lcm_capacity <= radix_capacity:
        raise ValueError("LCM capacity must exceed Radix capacity")

    block_tokens = _positive_int(
        spec["logical_block_tokens"], "logical_block_tokens"
    )
    buckets = []
    for item in spec["buckets"]:
        pages = _positive_int(item["pages"], f"{item['name']} pages")
        tail_tokens = int(item.get("tail_tokens", 0))
        if not 0 <= tail_tokens < block_tokens:
            raise ValueError(
                f"{item['name']} tail_tokens must be in [0, {block_tokens})"
            )
        buckets.append(
            {
                "name": str(item["name"]),
                "pages": pages,
                "prompt_tokens": pages * block_tokens + tail_tokens,
            }
        )
    if not buckets:
        raise ValueError("trace needs at least one boundary bucket")
    prompt_lengths = [bucket["prompt_tokens"] for bucket in buckets]
    bucket_indices = {
        bucket["name"]: index for index, bucket in enumerate(buckets)
    }
    boundary_tree = [str(name) for name in spec["boundary_tree"]]
    if len(boundary_tree) < 2 or len(set(boundary_tree)) != len(boundary_tree):
        raise ValueError("boundary_tree must contain distinct boundary buckets")
    try:
        boundary_indices = [bucket_indices[name] for name in boundary_tree]
    except KeyError as exc:
        raise ValueError(f"boundary_tree references unknown bucket {exc.args[0]!r}")
    boundary_lengths = [prompt_lengths[index] for index in boundary_indices]
    if boundary_lengths != sorted(boundary_lengths):
        raise ValueError("boundary_tree buckets must be ordered shortest to longest")
    smaller_capacity = min(radix_capacity, lcm_capacity)

    no_pressure = _align_down(
        int(smaller_capacity * float(spec["no_pressure_fraction"])),
        block_tokens,
    )
    fixed_pressure = _align_down(
        int(smaller_capacity * float(spec["fixed_pressure_fraction"])),
        block_tokens,
    )
    cliff = _align_down((radix_capacity + lcm_capacity) // 2, block_tokens)
    if no_pressure > smaller_capacity // 2:
        raise ValueError("no-pressure working set must not exceed 50%")
    fixed_ratio = fixed_pressure / smaller_capacity
    if not 0.85 <= fixed_ratio <= 0.95:
        raise ValueError("fixed-pressure working set must be within 85%-95%")
    if not radix_capacity < cliff < lcm_capacity:
        raise ValueError("capacity-cliff working set must lie between arm capacities")

    phases = []
    for name, target in (
        ("no_pressure", no_pressure),
        ("fixed_pressure", fixed_pressure),
        ("capacity_cliff", cliff),
    ):
        counts = _bucket_counts(
            target,
            prompt_lengths,
            boundary_indices,
        )
        working_set = _working_set_tokens(
            counts,
            prompt_lengths,
            boundary_indices,
        )
        phases.append(
            {
                "name": name,
                "working_set_tokens": working_set,
                "request_counts": {
                    bucket["name"]: count
                    for bucket, count in zip(buckets, counts)
                },
            }
        )

    return {
        "schema_version": 1,
        "model": str(spec["model"]),
        "logical_block_tokens": block_tokens,
        "full_cache_blocks_per_lcm_block": _positive_int(
            spec["full_cache_blocks_per_lcm_block"],
            "full_cache_blocks_per_lcm_block",
        ),
        "seed": int(spec["seed"]),
        "token_id_min": int(spec["token_id_min"]),
        "token_id_max_exclusive": int(spec["token_id_max_exclusive"]),
        "boundary_tree": boundary_tree,
        "capacities": {
            "radix": radix_capacity,
            "flat_lcm": lcm_capacity,
        },
        "buckets": buckets,
        "phases": phases,
    }


def trace_sha256(trace: Mapping) -> str:
    encoded = json.dumps(
        trace, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _request_token_ids(trace: Mapping, request_id: str, length: int) -> list[int]:
    token_min = int(trace["token_id_min"])
    token_max = int(trace["token_id_max_exclusive"])
    if token_max <= token_min:
        raise ValueError("token_id_max_exclusive must exceed token_id_min")
    seed_material = f"{trace['seed']}:{request_id}".encode()
    state = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    span = token_max - token_min
    output = []
    for _ in range(length):
        state = (6364136223846793005 * state + 1442695040888963407) & (
            (1 << 64) - 1
        )
        output.append(token_min + state % span)
    return output


def iter_trace_requests(trace: Mapping) -> Iterator[dict]:
    buckets = {bucket["name"]: bucket for bucket in trace["buckets"]}
    boundary_tree = set(trace["boundary_tree"])
    for phase in trace["phases"]:
        for round_name in _ROUND_NAMES:
            for bucket_name, count in phase["request_counts"].items():
                bucket = buckets[bucket_name]
                for index in range(int(count)):
                    request_id = f"{phase['name']}/{bucket_name}/{index}"
                    token_source_id = (
                        f"{phase['name']}/boundary_tree"
                        if index == 0 and bucket_name in boundary_tree
                        else request_id
                    )
                    prompt_tokens = int(bucket["prompt_tokens"])
                    yield {
                        "phase": phase["name"],
                        "bucket": bucket_name,
                        "round": round_name,
                        "measured": True,
                        "request_id": request_id,
                        "prompt_tokens": prompt_tokens,
                        "input_ids": _request_token_ids(
                            trace, token_source_id, prompt_tokens
                        ),
                    }


def run_trace(
    trace: Mapping,
    *,
    generate: Callable[[list[int], Mapping], Mapping],
    flush_cache: Callable[[str], None],
) -> list[dict]:
    """Run the compact trace through an Engine-compatible response callback."""
    observations = []
    active_phase = None
    for request in iter_trace_requests(trace):
        if request["phase"] != active_phase:
            if active_phase is not None:
                flush_cache(request["phase"])
            active_phase = request["phase"]
        response = generate(request["input_ids"], request)
        meta = response["meta_info"]
        prompt_tokens = int(meta["prompt_tokens"])
        cached_tokens = int(meta.get("cached_tokens", 0))
        if prompt_tokens != request["prompt_tokens"]:
            raise ValueError(
                f"{request['request_id']}: runtime reported {prompt_tokens} prompt "
                f"tokens, expected {request['prompt_tokens']}"
            )
        if not 0 <= cached_tokens <= prompt_tokens:
            raise ValueError(
                f"{request['request_id']}: cached_tokens={cached_tokens} outside "
                f"[0, {prompt_tokens}]"
            )
        executed = int(
            meta.get("executed_prefill_tokens", prompt_tokens - cached_tokens)
        )
        observations.append(
            {
                key: request[key]
                for key in ("phase", "bucket", "round", "measured", "request_id")
            }
            | {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "executed_prefill_tokens": executed,
            }
        )
    return observations


def summarize_observations(observations: Iterable[Mapping]) -> dict:
    totals = {"buckets": {}, "phases": {}}
    for item in observations:
        if not item.get("measured", True):
            continue
        keys = {
            "buckets": f"{item['phase']}/{item['bucket']}/{item['round']}",
            "phases": f"{item['phase']}/{item['round']}",
        }
        for section, key in keys.items():
            row = totals[section].setdefault(
                key,
                {
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "executed_prefill_tokens": 0,
                },
            )
            for field in row:
                row[field] += int(item[field])
    for section in totals.values():
        for row in section.values():
            prompt_tokens = row["prompt_tokens"]
            row["cache_hit_rate"] = (
                row["cached_tokens"] / prompt_tokens if prompt_tokens else 0.0
            )
    return totals


def _validate_result(result: Mapping, expected_arm: str, trace_hash: str) -> None:
    if result.get("arm") != expected_arm:
        raise ValueError(f"result {expected_arm!r} has mismatched arm name")
    if result.get("trace_sha256") != trace_hash:
        raise ValueError("all arms must use the same trace")
    _positive_int(
        result.get("configured_cache_bytes"), "configured cache bytes"
    )
    _positive_int(
        result.get("allocated_cache_bytes"), "allocated cache bytes"
    )
    capacity = _positive_int(
        result.get("max_total_num_tokens"), "max_total_num_tokens"
    )
    physical = _positive_int(
        result.get("physical_token_capacity"), "physical_token_capacity"
    )
    if physical != capacity:
        raise ValueError(
            "scheduler capacity differs from physical token capacity; "
            "scheduler counts alone are not capacity evidence"
        )
    if result.get("capacity_source") in (None, "", "scheduler"):
        raise ValueError(
            "capacity must be derived from physical allocation, not scheduler state"
        )
    _positive_int(
        result.get("resident_prefix_capacity_tokens"),
        "resident prefix capacity",
    )


def compare_results(results: Mapping[str, Mapping]) -> dict:
    """Validate the three arms and return a JSON/Markdown-ready comparison."""
    if set(results) != set(_ARM_NAMES):
        raise ValueError(f"results must contain exactly {list(_ARM_NAMES)}")
    trace_hashes = {str(result.get("trace_sha256")) for result in results.values()}
    if len(trace_hashes) != 1:
        raise ValueError("all arms must use the same trace")
    trace_hash = trace_hashes.pop()
    for arm in _ARM_NAMES:
        _validate_result(results[arm], arm, trace_hash)

    configured = {
        int(result["configured_cache_bytes"]) for result in results.values()
    }
    if len(configured) != 1:
        raise ValueError("configured cache bytes differ across arms")

    main = results["main_radix"]
    branch_radix = results["branch_radix"]
    flat = results["branch_flat_lcm"]
    radix_capacity = max(
        int(main["resident_prefix_capacity_tokens"]),
        int(branch_radix["resident_prefix_capacity_tokens"]),
    )
    flat_capacity = int(flat["resident_prefix_capacity_tokens"])
    if flat_capacity <= radix_capacity:
        raise ValueError("Flat LCM capacity does not exceed Radix")
    working_sets = [result.get("working_sets") for result in results.values()]
    if any(item != working_sets[0] for item in working_sets[1:]):
        raise ValueError("working sets differ across arms")
    cliff = int(working_sets[0]["capacity_cliff"])
    if not radix_capacity < cliff < flat_capacity:
        raise ValueError("capacity-cliff working set is outside the capacity gap")

    bucket_sets = [
        set(result["summary"]["buckets"]) for result in results.values()
    ]
    if any(item != bucket_sets[0] for item in bucket_sets[1:]):
        raise ValueError("reported prefix-length buckets differ across arms")

    no_pressure_key = "no_pressure/replay"
    cliff_key = "capacity_cliff/replay"
    for arm in _ARM_NAMES:
        no_pressure_rate = float(
            results[arm]["summary"]["phases"][no_pressure_key]["cache_hit_rate"]
        )
        minimum = (
            _MIN_FLAT_RESIDENT_HIT_RATE
            if arm == "branch_flat_lcm"
            else _MIN_RADIX_RESIDENT_HIT_RATE
        )
        if no_pressure_rate < minimum:
            raise ValueError(
                f"{arm} no-pressure replay hit rate {no_pressure_rate:.6f} "
                f"is below {minimum:.1f}"
            )
    flat_cliff_rate = float(
        flat["summary"]["phases"][cliff_key]["cache_hit_rate"]
    )
    if flat_cliff_rate < _MIN_FLAT_RESIDENT_HIT_RATE:
        raise ValueError(
            f"Flat capacity-cliff replay hit rate {flat_cliff_rate:.6f} "
            f"is below {_MIN_FLAT_RESIDENT_HIT_RATE:.1f}"
        )
    for arm, result in (
        ("main_radix", main),
        ("branch_radix", branch_radix),
    ):
        radix_cliff_rate = float(
            result["summary"]["phases"][cliff_key]["cache_hit_rate"]
        )
        if radix_cliff_rate > _MAX_RADIX_CLIFF_HIT_RATE:
            raise ValueError(
                f"{arm} capacity-cliff replay hit rate "
                f"{radix_cliff_rate:.6f} exceeds "
                f"{_MAX_RADIX_CLIFF_HIT_RATE:.1f}; the workload did not "
                "demonstrate the capacity boundary"
            )

    buckets = {}
    for key in sorted(bucket_sets[0]):
        main_rate = float(main["summary"]["buckets"][key]["cache_hit_rate"])
        branch_rate = float(
            branch_radix["summary"]["buckets"][key]["cache_hit_rate"]
        )
        flat_rate = float(flat["summary"]["buckets"][key]["cache_hit_rate"])
        baseline = max(main_rate, branch_rate)
        if flat_rate < baseline:
            raise ValueError(
                f"Flat cache hit rate regressed for {key}: "
                f"{flat_rate:.6f} < {baseline:.6f}"
            )
        buckets[key] = {
            "main_radix": main_rate,
            "branch_radix": branch_rate,
            "branch_flat_lcm": flat_rate,
        }

    return {
        "trace_sha256": trace_hash,
        "configured_cache_bytes": configured.pop(),
        "capacity_ratio": flat_capacity / radix_capacity,
        "capacities": {
            arm: int(results[arm]["resident_prefix_capacity_tokens"])
            for arm in _ARM_NAMES
        },
        "addressable_logical_tokens": {
            arm: int(results[arm]["max_total_num_tokens"]) for arm in _ARM_NAMES
        },
        "allocated_cache_bytes": {
            arm: int(results[arm]["allocated_cache_bytes"]) for arm in _ARM_NAMES
        },
        "working_sets": working_sets[0],
        "buckets": buckets,
    }


def render_markdown(report: Mapping) -> str:
    lines = [
        "# Qwen3.5 LCM cache comparison",
        "",
        f"- Capacity ratio: {float(report['capacity_ratio']):.3f}",
        f"- Configured cache bytes: {int(report['configured_cache_bytes'])}",
        f"- Trace SHA-256: `{report['trace_sha256']}`",
        "",
        "| Bucket | main Radix | branch Radix | branch Flat LCM |",
        "|---|---:|---:|---:|",
    ]
    for key, rates in report["buckets"].items():
        lines.append(
            f"| {key} | {rates['main_radix']:.6f} | "
            f"{rates['branch_radix']:.6f} | "
            f"{rates['branch_flat_lcm']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _cache_storage(
    scheduler_info: Mapping,
    override: Mapping | None = None,
) -> dict:
    storage = override if override is not None else scheduler_info.get("cache_storage")
    if not isinstance(storage, Mapping):
        raise ValueError("runtime did not report cache_storage")
    return {
        "configured_cache_bytes": _positive_int(
            storage.get("configured_cache_bytes"), "configured cache bytes"
        ),
        "allocated_cache_bytes": _positive_int(
            storage.get("allocated_cache_bytes"), "allocated cache bytes"
        ),
        "physical_token_capacity": _positive_int(
            storage.get("physical_token_capacity"), "physical token capacity"
        ),
        "capacity_source": str(storage.get("capacity_source", "")),
        "geometry": storage.get("geometry", {}),
    }


def _default_engine_factory():
    from tokenspeed.runtime.entrypoints.engine import Engine

    return Engine


def probe_arm(
    *,
    arm: str,
    engine_args: Mapping,
    cache_storage_override: Mapping | None = None,
    engine_factory=None,
) -> dict:
    """Start one arm and capture its scheduler and physical cache report."""
    if arm not in _ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    if engine_factory is None:
        engine_factory = _default_engine_factory()
    engine = engine_factory(**dict(engine_args))
    try:
        scheduler_info = engine.scheduler_info
        return {
            "arm": arm,
            "max_total_num_tokens": _positive_int(
                scheduler_info.get("max_total_num_tokens"),
                "max_total_num_tokens",
            ),
        } | _cache_storage(scheduler_info, cache_storage_override)
    finally:
        engine.shutdown()


def build_trace_from_probes(spec: Mapping, probes: Mapping[str, Mapping]) -> dict:
    if set(probes) != set(_ARM_NAMES):
        raise ValueError(f"probes must contain exactly {list(_ARM_NAMES)}")
    configured = {
        int(probe["configured_cache_bytes"]) for probe in probes.values()
    }
    if len(configured) != 1:
        raise ValueError("configured cache bytes differ across arms")
    radix_capacity = max(
        int(probes["main_radix"]["physical_token_capacity"]),
        int(probes["branch_radix"]["physical_token_capacity"]),
    )
    capacity_bucket = str(spec["capacity_prompt_bucket"])
    flat_capacity, state_checkpoints, transient_parents = _flat_capacity_model(
        spec, probes["branch_flat_lcm"]
    )
    trace = build_trace(
        spec,
        radix_capacity_tokens=radix_capacity,
        lcm_capacity_tokens=flat_capacity,
    )
    replay_kind = "" if state_checkpoints == 1 else " exact replay"
    trace["capacity_model"] = {
        "radix": "runtime physical token capacity",
        "flat_lcm": (
            f"{capacity_bucket}{replay_kind} with "
            f"{'one' if state_checkpoints == 1 else 'two'} retained "
            f"checkpoint{'s' if state_checkpoints != 1 else ''} per State group "
            f"and {transient_parents}-parent admission reserve"
        ),
    }
    return trace


def run_arm(
    *,
    arm: str,
    trace: Mapping,
    engine_args: Mapping,
    cache_storage_override: Mapping | None = None,
    engine_factory=None,
    provenance: Mapping | None = None,
) -> dict:
    """Run a frozen trace against one engine arm and return its result JSON."""
    if arm not in _ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    if engine_factory is None:
        engine_factory = _default_engine_factory()
    scheduler_info = None
    storage = None
    observations = []
    for phase in trace["phases"]:
        engine = engine_factory(**dict(engine_args))
        try:
            phase_scheduler_info = engine.scheduler_info
            phase_storage = _cache_storage(
                phase_scheduler_info, cache_storage_override
            )
            if scheduler_info is None:
                scheduler_info = phase_scheduler_info
                storage = phase_storage
            elif (
                phase_scheduler_info.get("max_total_num_tokens")
                != scheduler_info.get("max_total_num_tokens")
                or phase_storage != storage
            ):
                raise ValueError("cache geometry changed between trace phases")

            phase_trace = dict(trace)
            phase_trace["phases"] = [phase]

            def generate(input_ids, _request):
                return engine.generate(
                    input_ids=input_ids,
                    sampling_params={"temperature": 0, "max_new_tokens": 1},
                )

            observations.extend(
                run_trace(
                    phase_trace,
                    generate=generate,
                    flush_cache=lambda _phase: None,
                )
            )
        finally:
            engine.shutdown()

    if scheduler_info is None or storage is None:
        raise ValueError("trace needs at least one phase")
    capacity_key = "flat_lcm" if arm == "branch_flat_lcm" else "radix"
    resident_capacity = _positive_int(
        trace["capacities"][capacity_key],
        f"{capacity_key} resident prefix capacity",
    )
    return {
        "arm": arm,
        "trace_sha256": trace_sha256(trace),
        "max_total_num_tokens": _positive_int(
            scheduler_info.get("max_total_num_tokens"),
            "max_total_num_tokens",
        ),
        "resident_prefix_capacity_tokens": resident_capacity,
        "capacity_model": trace.get("capacity_model", {}).get(capacity_key, ""),
        **storage,
        "working_sets": {
            phase["name"]: int(phase["working_set_tokens"])
            for phase in trace["phases"]
        },
        "summary": summarize_observations(observations),
        "provenance": dict(provenance or _git_provenance()),
    }


def _git_provenance() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"git_commit": commit, "git_dirty": dirty}


def _read_json(value: str) -> dict:
    if value.lstrip().startswith("{"):
        return json.loads(value)
    return json.loads(pathlib.Path(value).read_text())


def _write_json(path: str, value: Mapping) -> None:
    pathlib.Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe")
    probe.add_argument("--arm", choices=_ARM_NAMES, required=True)
    probe.add_argument("--engine-args-json", required=True)
    probe.add_argument("--cache-storage-json")
    probe.add_argument("--output", required=True)

    build = commands.add_parser("build-trace")
    build.add_argument(
        "--spec",
        default=str(pathlib.Path(__file__).with_name("qwen35_lcm_cache_trace.json")),
    )
    build.add_argument("--main-probe", required=True)
    build.add_argument("--branch-radix-probe", required=True)
    build.add_argument("--flat-probe", required=True)
    build.add_argument("--output", required=True)

    run = commands.add_parser("run")
    run.add_argument("--arm", choices=_ARM_NAMES, required=True)
    run.add_argument("--trace", required=True)
    run.add_argument("--engine-args-json", required=True)
    run.add_argument("--cache-storage-json")
    run.add_argument("--output", required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--main-result", required=True)
    compare.add_argument("--branch-radix-result", required=True)
    compare.add_argument("--flat-result", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--markdown", required=True)
    return parser


def main(argv=None, *, engine_factory=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe":
        _write_json(
            args.output,
            probe_arm(
                arm=args.arm,
                engine_args=_read_json(args.engine_args_json),
                cache_storage_override=(
                    _read_json(args.cache_storage_json)
                    if args.cache_storage_json
                    else None
                ),
                engine_factory=engine_factory,
            ),
        )
        return 0
    if args.command == "build-trace":
        probes = {
            "main_radix": _read_json(args.main_probe),
            "branch_radix": _read_json(args.branch_radix_probe),
            "branch_flat_lcm": _read_json(args.flat_probe),
        }
        _write_json(
            args.output,
            build_trace_from_probes(_read_json(args.spec), probes),
        )
        return 0
    if args.command == "run":
        _write_json(
            args.output,
            run_arm(
                arm=args.arm,
                trace=_read_json(args.trace),
                engine_args=_read_json(args.engine_args_json),
                cache_storage_override=(
                    _read_json(args.cache_storage_json)
                    if args.cache_storage_json
                    else None
                ),
                engine_factory=engine_factory,
            ),
        )
        return 0

    results = {
        "main_radix": _read_json(args.main_result),
        "branch_radix": _read_json(args.branch_radix_result),
        "branch_flat_lcm": _read_json(args.flat_result),
    }
    report = compare_results(results)
    _write_json(args.output, report)
    pathlib.Path(args.markdown).write_text(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
