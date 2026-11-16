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

"""Explicit cost rules for dense matmul, attention and collective traffic."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections import Counter
from dataclasses import asdict, dataclass, replace
from math import prod
from pathlib import Path

from tokenspeed_kernel.op_profile import (
    ComputeComponent,
    CostRegistry,
    OpCost,
    OpRecord,
    TensorDescription,
    _profile_include_search_roots,
    _profile_observed_configs,
    _profile_validate_startup_observation,
)


def _eager_host_control_input(
    op_name,
    implementation,
    source_sha256,
    name,
    *,
    device,
    dtype,
    shape,
    stride,
    storage_offset,
    logical_bytes,
) -> bool:
    """Identify the reviewed prefill descriptor that never reaches CUDA."""
    return (
        op_name == "conv.causal_conv1d_prefill"
        and implementation == "triton_causal_conv1d_prefill"
        and source_sha256
        == "c42179c6e4c51d784ad9dc259094d9b5177b2ac121eae67874dd7cf966b22d5e"
        and name == "prefill_metadata.cu_seqlens_cpu"
        and device == "cpu"
        and dtype == "torch.int64"
        and tuple(shape) == (2,)
        and tuple(stride) == (1,)
        and storage_offset == 0
        and logical_bytes == 16
    )


def _include_search_error(observed: dict, baseline: dict, reviewed: list) -> str | None:
    """Validate the same closed, ordered positive and negative include search."""
    previous = None
    for snapshot in (baseline, observed):
        try:
            search = snapshot["include_search"]
            roots = {item["name"]: item["root"] for item in snapshot["modules"]}
            entries = _profile_include_search_roots(
                roots,
                snapshot["configured_roots"],
                snapshot["environment"],
                snapshot["working_directory"],
            )
            candidates = snapshot["include_candidates"]
            if (
                search.get("schema") != "tokenspeed.gmma_include_search/v1"
                or search.get("complete") is not True
                or search.get("entries") != entries
                or len(candidates) != len(entries)
            ):
                return "complete ordered GMMA include search unavailable"
            identity, positives = [], 0
            for entry, item in zip(entries, candidates):
                path = str(Path(entry["root"]) / "cute/arch/mma_sm90_gmma.hpp")
                if (
                    any(item.get(key) != value for key, value in entry.items())
                    or item.get("path") != path
                    or item.get("unavailable_reason")
                ):
                    return "unrepresented or unavailable GMMA include candidate"
                if item.get("presence") == "present_regular_file":
                    if item.get("sha256") not in reviewed:
                        return "unreviewed resolved GMMA include candidate"
                    positives += 1
                elif item.get("presence") == "proved_absent_ENOENT":
                    if (
                        item.get("sha256") is not None
                        or item.get("exception_type") != "FileNotFoundError"
                        or item.get("errno") != 2
                        or item.get("exception_path") != path
                    ):
                        return "invalid path-specific GMMA absence proof"
                else:
                    return "unavailable typed GMMA include candidate"
                identity.append(
                    {
                        key: item.get(key)
                        for key in (
                            "root",
                            "origin",
                            "path",
                            "presence",
                            "sha256",
                            "bytes",
                            "inode",
                            "device",
                            "mtime_ns",
                            "ctime_ns",
                            "exception_type",
                            "errno",
                            "exception_path",
                        )
                    }
                )
            if not positives:
                return "no reviewed applicable GMMA include candidate"
            if previous is not None and identity != previous:
                return "GMMA include presence/order/identity changed since baseline"
            previous = identity
        except (KeyError, TypeError, ValueError):
            return "complete ordered GMMA include search unavailable"
    return None


def register_selected_engine_resolver(
    registry: CostRegistry,
    *,
    definitions: bytes,
    expected_sha256: str,
    supplementary: bytes | None = None,
    supplementary_sha256: str | None = None,
) -> None:
    """Use caller-reviewed immutable guards after an exact native scope join.

    This is an offline cost provider, never a trace-selected import or runtime
    execution selector. The explicit caller owns semantic review of the guard
    bytes, including source invariants and the full artifact candidate universe.
    Only normalization components change; canonical useful FLOPs and IO come
    from the existing rule on this replay's live metadata.

    supplementary is optional caller-reviewed provenance for an immutable older
    capture. Its exact run/rank/process/capture binding is mandatory; it never
    mutates that capture or claims historical module-byte identity. Without it,
    source contracts consume the collector's automatic final-flush provenance.
    Reviewed bindings default to Graph. An explicit attribution_modes entry
    can enable the same guard checks over eager runtime/activity rows.
    """
    digest = hashlib.sha256(definitions).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Reviewed engine guard SHA256 mismatch")
    document = json.loads(definitions)
    if document.get("schema") != "tokenspeed.reviewed_engine_guards/v1":
        raise ValueError("Unsupported reviewed engine guard schema")
    if (supplementary is None) != (supplementary_sha256 is None):
        raise ValueError("Supplementary evidence requires explicit reviewed SHA256")
    supplements = []
    if supplementary is not None:
        if hashlib.sha256(supplementary).hexdigest() != supplementary_sha256:
            raise ValueError("Reviewed supplementary provenance SHA256 mismatch")
        supplement = json.loads(supplementary)
        if supplement.get("schema") != "tokenspeed.engine_source_supplement/v1":
            raise ValueError("Unsupported engine source supplement schema")
        supplements = supplement["bindings"]
        for item in supplements:
            identity = item.get("identity", {})
            if (
                not isinstance(identity.get("capture_sha256"), str)
                or len(identity["capture_sha256"]) != 64
                or not isinstance(identity.get("run_id"), str)
                or type(identity.get("rank")) is not int
                or type(identity.get("replay_identity", {}).get("process_id"))
                is not int
            ):
                raise ValueError(
                    "Supplement requires exact run/rank/process/capture identity"
                )
    rules = document["bindings"]
    for rule in rules:
        modes = rule.get("attribution_modes", ["cuda_graph_node_ids"])
        if (
            not isinstance(modes, list)
            or not modes
            or any(
                mode not in ("cuda_graph_node_ids", "eager_nvtx_gpu_projection")
                for mode in modes
            )
            or len(set(modes)) != len(modes)
        ):
            raise ValueError(
                "Unsupported or duplicate reviewed native attribution mode"
            )
        divisor = rule.get("tensor_alignment_divisor", 1)
        if type(divisor) is not int or divisor < 1 or divisor > 256 or 256 % divisor:
            raise ValueError("Alignment guard must be provable from modulo256")
    if len({rule["contract_id"] for rule in rules}) != len(rules):
        raise ValueError("Duplicate reviewed engine contract")

    def matches(actual, expected):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and matches(actual[key], value)
                for key, value in expected.items()
            )
        if isinstance(expected, (tuple, list)):
            return (
                isinstance(actual, (tuple, list))
                and len(actual) == len(expected)
                and all(matches(a, b) for a, b in zip(actual, expected))
            )
        return type(actual) is type(expected) and actual == expected

    def tensor_signature(op):
        return {
            "inputs": {name: asdict(value) for name, value in op.inputs.items()},
            "outputs": {name: asdict(value) for name, value in op.outputs.items()},
            "static_traits": op.static_traits,
            "autocast_enabled": op.autocast_enabled,
        }

    def fingerprint(item):
        kernel = item.get("kernel")
        if not isinstance(kernel, dict):
            return None
        keys = (
            "name",
            "grid",
            "block",
            "registers_per_thread",
            "static_shared_memory",
            "dynamic_shared_memory",
            "cluster",
        )
        selected = {key: kernel.get(key) for key in keys}
        if any(
            value is None or isinstance(value, list) and any(v is None for v in value)
            for value in selected.values()
        ):
            return None
        return json.dumps(selected, sort_keys=True)

    def memory_conflicts(actual, expected):
        # Missing raw fields are not counterevidence to a reviewed source role.
        memory = actual.get("memory") or {}
        conflicts = [
            f"{key}={memory[key]!r} outside reviewed values {allowed!r}"
            for key, allowed in expected.get("memory_values_if_known", {}).items()
            if memory.get(key) is not None
            and not any(matches(memory[key], value) for value in allowed)
        ]
        for keys in expected.get("memory_equal_if_known", []):
            observed = [memory[key] for key in keys if memory.get(key) is not None]
            if observed and any(not matches(value, observed[0]) for value in observed):
                conflicts.append(f"reviewed equal memory fields {keys!r} differ")
        return conflicts

    def native_matches(actual, case):
        kernels = [
            item for item in actual if item.get("table") == "CUPTI_ACTIVITY_KIND_KERNEL"
        ]
        copies = [
            item for item in actual if item.get("table") == "CUPTI_ACTIVITY_KIND_MEMCPY"
        ]
        supports = case.get("native_support_activities", [])
        expected_copies = [
            item
            for item in supports
            if item.get("table") == "CUPTI_ACTIVITY_KIND_MEMCPY"
        ]
        if (
            len(kernels) + len(copies) != len(actual)
            or len(expected_copies) != len(supports)
            or len(copies) != sum(item["count"] for item in expected_copies)
        ):
            return False
        remaining = list(copies)
        for expected in expected_copies:
            required = {
                "table": expected["table"],
                "memory": expected.get("memory", {}),
            }
            for _ in range(expected["count"]):
                selected = next(
                    (
                        item
                        for item in remaining
                        if matches(item, required)
                        and not memory_conflicts(item, expected)
                    ),
                    None,
                )
                if selected is None:
                    return False
                remaining.remove(selected)
        observed = [fingerprint(item) for item in kernels]
        if any(value is None for value in observed):
            return False
        if case.get("source_instruction_invariant") is True:
            # A reviewed fixed function/artifact source contract may prove the
            # engine independently of launch dimensions. Retain every actual
            # resource field in the proof, and still require the full role set.
            return Counter(item["kernel"]["name"] for item in kernels) == Counter(
                item["name"] for item in case["native_launches"]
            )
        return Counter(observed) == Counter(
            json.dumps(item, sort_keys=True) for item in case["native_launches"]
        )

    def source_provenance(rule, context, libraries, worker):
        guards = rule.get("source_provenance")
        if guards is None:
            return None, None
        observed = libraries.get("semantic_provenance")
        origin = "same_process_collector_final_flush"
        matching = [item for item in supplements if matches(context, item["identity"])]
        if len(matching) > 1:
            return None, "multiple supplementary provenance bindings match capture"
        if matching:
            observed = matching[0]["provenance"]
            origin = "caller_reviewed_same_environment_supplement"
        if (
            not isinstance(observed, dict)
            or observed.get("schema") != "tokenspeed.fp8_source_provenance/v1"
            or observed.get("process_id") != worker.get("process_id")
        ):
            return None, "same-process semantic source/cache provenance unavailable"
        sources = {}
        roots = {
            item.get("name"): item.get("root") for item in observed.get("modules", [])
        }
        for item in observed.get("source_files", []):
            key = (item.get("module"), item.get("relative_path"))
            if key in sources:
                return None, "duplicate installed semantic source identity"
            sources[key] = item
        for expected in guards["source_files"]:
            item = sources.get((expected["module"], expected["relative_path"]))
            if (
                item is None
                or item.get("unavailable_reason")
                or item.get("sha256") not in expected["sha256s"]
                or not roots.get(expected["module"])
                or Path(item["path"])
                != Path(roots[expected["module"]]) / expected["relative_path"]
            ):
                return (
                    None,
                    "reviewed semantic API/mainloop dependency source unavailable or changed",
                )
        baseline = observed.get("baseline")
        include_error = _include_search_error(
            observed, baseline or {}, guards["primitive_sha256s"]
        )
        if include_error:
            return None, include_error
        # Only explicitly closed environment search segments are accounted for.
        # Other include flags/overrides still require their exact reviewed value.
        environment = observed.get("environment", {})
        include_variables = {"CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"}
        overrides = {
            key: value
            for key, value in environment.items()
            if value
            and ("INCLUDE" in key or "FLAGS" in key or key == "CPATH")
            and key not in include_variables
        }
        reviewed_overrides = {
            key: value
            for key, value in guards.get("compiler_overrides", {}).items()
            if key not in include_variables
        }
        if overrides != reviewed_overrides:
            return None, "unreviewed compiler/include override"
        policy = guards["generation_policy"]
        compiler_proofs = observed.get("compilers", {})
        actual_api = {
            item.get("sha256")
            for item in observed.get("loaded_extensions", [])
            if not item.get("unavailable_reason")
        }
        if not set(policy["api_object_sha256s"]) <= actual_api:
            return None, "reviewed loaded API object identity unavailable"
        for key, relative in guards.get("configured_roots", {}).items():
            expected_root = (
                [str(Path(roots["flashinfer"]) / item) for item in relative]
                if isinstance(relative, list)
                else str(Path(roots["flashinfer"]) / relative)
            )
            if observed.get("configured_roots", {}).get(key) != expected_root:
                return (
                    None,
                    "actual loaded API include/mainloop root differs from reviewed root",
                )
        certificate = observed.get("generation_certificate") if matching else None
        if certificate is not None:
            if (
                certificate.get("evidence_sha256")
                not in policy["retrospective_evidence_sha256s"]
                or certificate.get("run_id") != context["run_id"]
                or certificate.get("source_contract_sha256")
                != guards["source_contract_sha256"]
                or certificate.get("workflow_assumptions")
                != policy["workflow_assumptions"]
            ):
                return None, "unreviewed retrospective current-generation certificate"
            lower, upper = certificate["start_unix_ns"], certificate["end_unix_ns"]
            worker_pids = certificate["worker_pids"]
        else:
            source_context = observed.get("source_context") or {}
            workers = observed.get("workers", [])
            if (
                not isinstance(baseline, dict)
                or baseline.get("process_id") != observed["process_id"]
                or source_context != baseline.get("source_context")
                or source_context.get("run_id") != context["run_id"]
                or source_context.get("rank") != context["rank"]
                or observed.get("run_id") != context["run_id"]
                or observed.get("rank") != context["rank"]
                or source_context.get("profile_config_sha256")
                != context.get("profile_config_sha256")
                or not context.get("profile_config_sha256")
                or not workers
            ):
                return None, "same-run startup/source-generation baseline unavailable"
            observation = source_context.get("startup_observation")
            try:
                body = _profile_validate_startup_observation(
                    observation, run_id=context["run_id"]
                )
                expected_config_hashes = {
                    hashlib.sha256(
                        json.dumps(
                            config,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode()
                    ).hexdigest()
                    for config in _profile_observed_configs(
                        observation, run_id=context["run_id"]
                    )
                }
                if (
                    source_context["profile_config_sha256"]
                    not in expected_config_hashes
                ):
                    return (
                        None,
                        "expanded config differs from before-Engine observation",
                    )
            except (ValueError, TypeError, KeyError):
                return None, "validated before-Engine startup observation unavailable"
            if observation["sha256"] != context.get("startup_observation_sha256"):
                return None, "capture and source-generation observation differ"
            lower, upper = body["realtime_after_ns"], observed.get("finished_unix_ns")
            if not (
                type(upper) is int
                and lower
                <= baseline.get("started_unix_ns", 0)
                <= observed.get("started_unix_ns", 0)
                <= upper
            ):
                return None, "startup/source snapshot clock order is unavailable"
            worker_pids = []
            worker_identity = {}
            for worker in workers:
                pid = worker.get("process_id")
                raw = worker.get("process_identity") or {}
                clock = body.get("clock_identity") or {}
                if (
                    type(pid) is not int
                    or pid <= 0
                    or pid in worker_pids
                    or raw.get("unavailable_reason")
                    or raw.get("process_id") != pid
                    or raw.get("stat_process_id") != pid
                    or not raw.get("namespace_pids")
                    or raw["namespace_pids"][-1] != pid
                    or any(
                        not clock.get(key) or raw.get(key) != clock[key]
                        for key in ("boot_id", "time_namespace", "pid_namespace")
                    )
                    or raw.get("monotonic_ns", 0) < body["monotonic_ns"]
                    or not lower
                    <= raw.get("realtime_before_ns", 0)
                    <= raw.get("realtime_after_ns", 0)
                    <= upper
                    or (
                        pid != body["process_id"]
                        and body["process_id"]
                        not in raw.get("ancestor_process_ids", [])
                    )
                ):
                    return None, "same-run worker PID/ancestry/clock domain unavailable"
                worker_pids.append(pid)
                worker_identity[pid] = (
                    raw.get("stat_process_id"),
                    raw.get("start_ticks"),
                    raw.get("boot_id"),
                    raw.get("time_namespace"),
                    raw.get("pid_namespace"),
                )
            initial_workers = {
                worker.get("process_id"): worker.get("process_identity") or {}
                for worker in baseline.get("workers", [])
            }
            if set(initial_workers) != set(worker_pids) or any(
                (
                    initial_workers[pid].get("stat_process_id"),
                    initial_workers[pid].get("start_ticks"),
                    initial_workers[pid].get("boot_id"),
                    initial_workers[pid].get("time_namespace"),
                    initial_workers[pid].get("pid_namespace"),
                )
                != identity
                for pid, identity in worker_identity.items()
            ):
                return None, "worker identity changed since source baseline"
            # Source/API/compiler bytes must predate the genuine before-import
            # observation and remain unchanged through final flush.
            for name in (
                "source_files",
                "compilers",
                "configured_roots",
                "modules",
                "loaded_extensions",
                "startup_driver_source",
            ):
                before, after = baseline.get(name), observed.get(name)
                if before != after:
                    return None, f"{name} changed since source-generation baseline"
            driver = observed.get("startup_driver_source") or {}
            if any(
                driver.get(key) != body["driver_source"].get(key)
                for key in ("path", "sha256", "bytes")
            ):
                return None, "startup driver source identity unavailable or changed"
            material = (
                observed.get("source_files", [])
                + observed.get("loaded_extensions", [])
                + list(compiler_proofs.values())
                + [driver]
            )
            material += [
                item
                for item in observed.get("include_candidates", [])
                if item.get("presence") == "present_regular_file"
            ]
            if any(
                item.get("unavailable_reason")
                or type(item.get("ctime_ns")) is not int
                or item["ctime_ns"] >= lower
                for item in material
            ):
                return (
                    None,
                    "source/API/compiler/include lifetime does not predate before-Engine observation",
                )
        if (
            worker["process_id"] not in worker_pids
            or type(lower) is not int
            or type(upper) is not int
            or lower >= upper
        ):
            return None, "current-generation worker/time identity unavailable"

        def generated(item, cache, expected):
            compiler = compiler_proofs.get(expected["name"], {})
            if (
                compiler.get("unavailable_reason")
                or compiler.get("sha256")
                not in policy["compiler_sha256s"][expected["name"]]
            ):
                return False
            created = item.get("birthtime_ns", item.get("ctime_ns"))
            if type(created) is not int or not lower <= created <= upper:
                return False
            if not certificate:
                baseline_cache = next(
                    (
                        c
                        for c in observed["baseline"].get("caches", [])
                        if c["name"] == expected["name"]
                    ),
                    None,
                )
                initial = next(
                    (
                        c
                        for c in (baseline_cache or {}).get("candidates", [])
                        if c["relative_path"] == item["relative_path"]
                    ),
                    None,
                )
                # Under the explicit isolated/compiler-only writer policy, a
                # stable warmup artifact may predate collector construction but
                # must still be born in this worker lifetime. DeepJIT additionally
                # supplies its actual producer command below.
                if initial is not None and any(
                    initial.get(key) != item.get(key)
                    for key in ("sha256", "inode", "device", "mtime_ns", "ctime_ns")
                ):
                    return False
            if expected["name"] == "deep_gemm":
                parent = str(Path(item["relative_path"]).parent)
                meta = next(
                    (
                        c.get("metadata")
                        for c in cache["candidates"]
                        if c["relative_path"] == f"{parent}/meta.json"
                    ),
                    None,
                )
                if not isinstance(meta, dict):
                    return False
                command = shlex.split(meta.get("command", ""))
                try:
                    producer = int(Path(command[1]).parent.name.split("-")[0])
                    output = command.index("--output-file")
                    options = command[2:output] + command[output + 2 :]
                except (ValueError, IndexError):
                    return False
                if (
                    producer not in worker_pids
                    or command[0] not in (compiler["path"], compiler["requested"])
                    or meta.get("compiler_info", {}).get("path")
                    not in (compiler["path"], compiler["requested"])
                    or meta.get("config", {}).get("python_library_root")
                    != roots["deep_gemm"]
                    or meta.get("config", {}).get("include_dirs")
                    != [str(Path(roots["deep_gemm"]) / "include")]
                    or meta.get("compiler_options") != policy["deep_compiler_options"]
                    or options
                    != policy["deep_command_options"][:-1]
                    + [str(Path(roots["deep_gemm"]) / "include")]
                ):
                    return False
            return True

        cache_proof = []
        for expected in guards["caches"]:
            compiler = compiler_proofs.get(expected["name"], {})
            if (
                compiler.get("unavailable_reason")
                or compiler.get("sha256")
                not in policy["compiler_sha256s"][expected["name"]]
            ):
                return None, "actual selected compiler identity is not reviewed"
            caches = [
                item
                for item in observed.get("caches", [])
                if item.get("name") == expected["name"]
            ]
            if (
                len(caches) != 1
                or caches[0].get("complete") is not True
                or caches[0].get("unavailable_reason")
            ):
                return (
                    None,
                    "complete compatible disk-cache candidate evidence unavailable",
                )
            variable = (
                "DG_JIT_CACHE_DIR"
                if expected["name"] == "deep_gemm"
                else "TRTLLM_DG_CACHE_DIR"
            )
            configured = environment.get(variable)
            if configured is None and observed.get("user_home"):
                configured = str(
                    Path(observed["user_home"])
                    / (".dj" if expected["name"] == "deep_gemm" else ".tensorrt_llm")
                )
            if (
                configured is None
                or Path(caches[0]["root"]) != Path(configured) / "cache"
            ):
                return (
                    None,
                    "cache candidate root differs from actual active selected API root/cache",
                )
            if not certificate:
                before_cache = next(
                    (
                        item
                        for item in baseline.get("caches", [])
                        if item.get("name") == expected["name"]
                    ),
                    None,
                )
                if (
                    not before_cache
                    or before_cache.get("complete") is not True
                    or before_cache.get("unavailable_reason")
                    or before_cache.get("root") != caches[0]["root"]
                ):
                    return None, "compatible cache baseline unavailable or changed"
                final_by_path = {
                    item["relative_path"]: item for item in caches[0]["candidates"]
                }
                if any(
                    item["relative_path"].endswith(".cubin")
                    and any(
                        final_by_path.get(item["relative_path"], {}).get(key)
                        != item.get(key)
                        for key in ("sha256", "inode", "device", "mtime_ns", "ctime_ns")
                    )
                    for item in before_cache.get("candidates", [])
                ):
                    return (
                        None,
                        "completed cache artifact removed or replaced since baseline",
                    )
            candidates = [
                item
                for item in caches[0]["candidates"]
                if item["relative_path"].endswith(".cubin")
                and any(
                    item["relative_path"].split("/")[0].startswith(prefix)
                    for prefix in expected["key_prefixes"]
                )
            ]
            for item in candidates:
                route = "reviewed_preexisting_bytes"
                if item.get("unavailable_reason"):
                    return None, "unreadable compatible cache artifact"
                if item.get("sha256") not in expected["cubin_sha256s"]:
                    if not generated(item, caches[0], expected):
                        return (
                            None,
                            "unknown reused cache bytes without trusted current-generation lineage",
                        )
                    route = "reviewed_current_generation_lineage"
                cache_proof.append(
                    dict(
                        cache=expected["name"],
                        root=caches[0]["root"],
                        sha256=item["sha256"],
                        relative_path=item["relative_path"],
                        route=route,
                    )
                )
        # The reviewed source invariant covers newly generated code, including
        # NVRTC's in-memory runtimeCache path. Disk census does not prove module
        # identity or exhaust that in-memory cache, and is never described as such.
        return (
            dict(
                kind=origin,
                supplementary_sha256=supplementary_sha256 if matching else None,
                source_contract_sha256=guards["source_contract_sha256"],
                source_module_roots=roots,
                provenance_sha256=hashlib.sha256(
                    json.dumps(observed, sort_keys=True).encode()
                ).hexdigest(),
                compatible_disk_candidates=cache_proof,
                current_generation="validated worker/time/compiler/source lineage; reviewed invariant includes NVRTC memory path",
                workflow_assumptions=policy["workflow_assumptions"],
                runtime_cache_inventory="not_observed",
            ),
            None,
        )

    def resolve(op, cost, context):
        candidates = [
            rule
            for rule in rules
            if rule["op_name"] == op.op_name
            and rule["implementation"] == op.implementation
        ]
        if not candidates or not cost.useful_flops:
            return None
        mode = context.get("attribution_mode", "cuda_graph_node_ids")
        eager = mode == "eager_nvtx_gpu_projection"
        device_tensors = [
            value
            for name, value in op.inputs.items()
            if not (
                eager
                and _eager_host_control_input(
                    op.op_name,
                    op.implementation,
                    op.scalar_args.get("profile_source_sha256"),
                    name,
                    device=value.device,
                    dtype=value.dtype,
                    shape=value.shape,
                    stride=value.stride,
                    storage_offset=value.storage_offset,
                    logical_bytes=value.logical_bytes,
                )
            )
        ] + list(op.outputs.values())
        worker = context.get("eager_step_identity" if eager else "replay_identity", {})
        base = {
            key: context.get(key)
            for key in (
                "capture_sha256",
                "replay_identity",
                "run_id",
                "rank",
                "scope_identity",
                "source_node_ids",
                "native_device_uuids",
            )
        }
        if eager:
            for key in ("capture_sha256", "replay_identity", "source_node_ids"):
                del base[key]
            base.update(
                attribution_mode=mode,
                eager_step_identity=worker,
                eager_sidecar=context["eager_sidecar"],
                source_report_sha256=context["source_report_sha256"],
            )
        base["definitions_sha256"] = digest
        failures = []
        accepted = []
        for rule in candidates:
            failure = None
            source_proof = None
            libraries = context.get("loaded_library_manifest")
            native = context.get("activity_bindings", [])
            signature = tensor_signature(op)
            identities = {(item["table"], item.get("source_rowid")) for item in native}
            if mode not in rule.get("attribution_modes", ["cuda_graph_node_ids"]):
                failure = "execution path has no reviewed native engine contract"
            elif not context.get("complete_join"):
                failure = (
                    context.get("incomplete_join_reason")
                    or "incomplete exact native scope/source-node join"
                )
            elif (
                not native
                or any(
                    type(item.get("source_rowid")) is not int
                    or item["source_rowid"] < 1
                    for item in native
                )
                or len(identities) != len(native)
            ):
                failure = "missing or duplicate native activity identity"
            elif eager and any(
                type(item.get("runtime_rowid")) is not int
                or item["runtime_rowid"] < 1
                or item.get("process_id") != worker.get("process_id")
                or item.get("thread_id") != worker.get("thread_id")
                for item in native
            ):
                failure = "missing exact eager runtime activity identity"
            elif not eager and any(
                type(item.get("node")) is not int
                or item.get("source") not in context["source_node_ids"]
                for item in native
            ):
                failure = "missing or duplicate native activity identity"
            elif len({value.device for value in device_tensors}) != 1 or any(
                not value.device.startswith("cuda:") for value in device_tensors
            ):
                failure = "boundary tensors do not share one CUDA device"
            elif (
                op.scalar_args.get("profile_source_sha256")
                not in rule["source_sha256s"]
            ):
                failure = "unreviewed or missing implementation source SHA"
            elif rule.get("tensor_alignment_divisor") is not None and any(
                value.pointer_mod256 is None
                or value.pointer_mod256 % rule["tensor_alignment_divisor"]
                for value in device_tensors
            ):
                failure = "required actual tensor alignment unavailable or unmatched"
            elif not matches(op.scalar_args, rule.get("scalar_args", {})):
                failure = (
                    "source branch, alias, flags or scalar guards differ or are missing"
                )
            elif any(
                name not in op.scalar_args.get("profile_none_arguments", ())
                for name in rule.get("none_arguments", [])
            ):
                failure = "required None argument mode not observed"
            elif any(
                name not in op.scalar_args.get("profile_argument_names", ())
                or name in op.scalar_args.get("profile_none_arguments", ())
                for name in rule.get("present_arguments", [])
            ):
                failure = "required tensor/output argument mode not observed"
            elif not any(
                matches(signature, case["boundary"]) and native_matches(native, case)
                for case in rule["cases"]
            ):
                conflicts = [
                    conflict
                    for case in rule["cases"]
                    if matches(signature, case["boundary"])
                    for expected in case.get("native_support_activities", [])
                    for item in native
                    if item.get("table") == expected["table"]
                    for conflict in memory_conflicts(item, expected)
                ]
                failure = (
                    "observed copy/source-binding conflict: " + "; ".join(conflicts)
                    if conflicts
                    else "actual boundary or complete native launch/resource role set differs"
                )
            elif (
                not isinstance(libraries, dict)
                or libraries.get("schema") != "tokenspeed.loaded_libraries/v1"
                or libraries.get("process_id") != worker.get("process_id")
                or libraries.get("unavailable_reason")
            ):
                failure = "same-process run-owned loaded-library manifest unavailable"
            else:
                device = context.get("hardware", {}).get("device") or {}
                native_ids = {
                    (item.get("kernel") or item.get("memory") or {}).get("device_id")
                    for item in native
                }
                context_ids = {
                    (item.get("kernel") or item.get("memory") or {}).get("context_id")
                    for item in native
                }
                observed = (context.get("hardware", {}).get("selection") or {}).get(
                    "observed", {}
                )
                observed_uuid = (
                    str(observed.get("uuid", ""))
                    .lower()
                    .removeprefix("gpu-")
                    .replace("-", "")
                )
                device_map = context.get("native_device_uuids", {})
                native_uuid = (
                    device_map.get(next(iter(native_ids)), {}).get("uuid")
                    if len(native_ids) == 1
                    else None
                )
                if (
                    None in native_ids
                    or len(native_ids) != 1
                    or None in context_ids
                    or len(context_ids) != 1
                    or not native_uuid
                    or native_uuid != observed_uuid
                ):
                    failure = "native process/context/device UUID does not match captured worker hardware"
                if device.get("compute_capability") != rule["compute_capability"]:
                    failure = "reviewed device compute capability not observed"
                if (
                    len(native_ids) == 1
                    and device_map.get(next(iter(native_ids)), {}).get(
                        "compute_capability"
                    )
                    != rule["compute_capability"]
                ):
                    failure = (
                        "native device compute capability unavailable or unmatched"
                    )
                actual_libs = [
                    item
                    for item in libraries["libraries"]
                    if any(
                        Path(item["path"]).name.startswith(prefix)
                        for prefix in rule["library_prefixes"]
                    )
                ]
                if any(
                    item.get("unavailable_reason") or not item.get("sha256")
                    for item in actual_libs
                ) or Counter(item.get("sha256") for item in actual_libs) != Counter(
                    rule["library_sha256s"]
                ):
                    failure = "actual relevant library candidate universe differs or is unreadable"
                actual_sources = {
                    item.get("sha256")
                    for item in libraries.get("source_files", [])
                    if not item.get("unavailable_reason")
                }
                if not set(rule.get("source_file_sha256s", [])) <= actual_sources:
                    failure = "reviewed installed source-file proof unavailable"
                if failure is None:
                    source_proof, failure = source_provenance(
                        rule, context, libraries, worker
                    )
            if failure:
                failures.append(f"{rule['contract_id']}: {failure}")
                continue
            components = []
            for part in rule["components"]:
                numerator, denominator = part["numerator"], part["denominator"]
                if (
                    type(numerator) is not int
                    or type(denominator) is not int
                    or numerator < 0
                    or denominator < 1
                    or cost.useful_flops * numerator % denominator
                ):
                    raise ValueError("Reviewed component partition is not integral")
                components.append(
                    asdict(
                        ComputeComponent(
                            cost.useful_flops * numerator // denominator,
                            part["compute_type"],
                            rule["contract_id"],
                        )
                    )
                )
            # OpCost validates the exact component sum when the registry binds it.
            accepted.append(
                {
                    **base,
                    "contract_id": rule["contract_id"],
                    "status": "bound",
                    "semantic_sources": document["semantic_sources"],
                    "engine_evidence_level": rule["engine_evidence_level"],
                    "module_identity": rule["module_identity"],
                    "boundary_signature": {**signature, "scalar_args": op.scalar_args},
                    "activity_bindings": native,
                    "reviewed_noncompute_support": next(
                        case.get("native_support_activities", [])
                        for case in rule["cases"]
                        if matches(signature, case["boundary"])
                        and native_matches(native, case)
                    ),
                    "artifact_candidates": [
                        item
                        for item in libraries["libraries"]
                        if any(
                            Path(item["path"]).name.startswith(prefix)
                            for prefix in rule["library_prefixes"]
                        )
                    ],
                    "source_provenance": source_proof,
                    "components": components,
                }
            )
        if len(accepted) > 1:
            raise ValueError("Conflicting selected-engine contracts match one Op")
        if accepted:
            return accepted[0]
        return {
            **base,
            "status": "unavailable",
            "unavailable_reason": "; ".join(failures),
        }

    registry.register_engine_resolver(resolve)


def _matmul_compute_type(op: OpRecord, *operands: TensorDescription) -> str | None:
    """Use an explicit implementation contract or captured execution decision.

    Operand storage dtype alone does not identify Tensor Core/SIMT execution.
    Dynamic metadata takes precedence over the implementation's fixed traits.
    """
    if (
        op.autocast_enabled
        or not operands
        or any(not tensor.device.startswith("cuda:") for tensor in operands)
    ):
        return None
    compute_type = op.scalar_args.get(
        "compute_type", op.static_traits.get("compute_type")
    )
    if compute_type is None:
        return None
    if not isinstance(compute_type, str) or not compute_type.endswith(
        ("_tensor", "_simt")
    ):
        raise ValueError("Explicit compute_type must identify precision and engine")
    return compute_type


@dataclass(frozen=True)
class DenseMatmulCost:
    """Plain dense 2D GEMM cost using explicit argument roles.

    No padding, quantization, fusion or implicit workspace is modeled. HBM
    traffic remains unknown; logical traffic assumes each operand read once.
    """

    lhs: str
    rhs: str
    output: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            lhs = op.inputs[self.lhs]
            rhs = op.inputs[self.rhs]
            output = op.outputs[self.output]
        except KeyError as exc:
            raise ValueError(f"Missing GEMM tensor role: {exc}") from exc
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError("DenseMatmulCost requires 2D operands")

        m, k = lhs.shape
        rhs_k, n = rhs.shape
        if k != rhs_k or output.shape != (m, n):
            raise ValueError("GEMM shapes do not match")

        flops = 2 * m * n * k
        return OpCost(
            useful_flops=flops,
            executed_flops=flops,
            logical_read_bytes=lhs.logical_bytes + rhs.logical_bytes,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=None,
            rule="dense_mm.v1",
            assumptions=(
                "unquantized dense A@B without padding or fusion",
                "each operand read once; output written once",
                "logical bytes do not measure HBM traffic",
            ),
            compute_type=_matmul_compute_type(op, lhs, rhs),
        )


@dataclass(frozen=True)
class _DenseProjectionCost:
    """Cost of A @ B.transpose(-1, -2) for registered torch_mm and torch_bmm."""

    ndim: int
    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            lhs = op.inputs["A"]
            rhs = op.inputs["B"]
            output = op.outputs["return"]
        except KeyError as exc:
            raise ValueError(f"Missing dense projection tensor role: {exc}") from exc
        if len(lhs.shape) != self.ndim or len(rhs.shape) != self.ndim:
            raise ValueError("Dense projection operand ranks do not match the rule")
        batch_shape = lhs.shape[:-2]
        if lhs.dtype != rhs.dtype or batch_shape != rhs.shape[:-2]:
            raise ValueError("Dense projection operands must match dtype and batch")

        m, k = lhs.shape[-2:]
        n, rhs_k = rhs.shape[-2:]
        if k != rhs_k or output.shape != (*batch_shape, m, n):
            raise ValueError(
                "Dense projection shapes do not match A @ B.transpose(-1, -2)"
            )

        batch_size = lhs.shape[0] if self.ndim == 3 else 1
        read_bytes = lhs.logical_bytes + rhs.logical_bytes
        read_bytes += sum(
            op.inputs[name].logical_bytes
            for name in ("bias", "alpha")
            if name in op.inputs
        )
        assumptions = (
            "useful dense matmul FLOPs only; bias, alpha and dtype conversions excluded",
            "output dtype does not select the compute peak",
            "compute peak requires an explicit implementation contract or execution decision",
            "logical IO does not establish physical HBM traffic",
        )
        estimated_hbm_bytes = None
        if self.hbm_policy == "logical_io_once":
            estimated_hbm_bytes = read_bytes + output.logical_bytes
            assumptions += (
                "HBM estimate assumes one operand read and one final output write; "
                "cache reuse, reloads, intermediate buffers and workspace excluded",
            )
        return OpCost(
            useful_flops=2 * batch_size * m * n * k,
            executed_flops=None,
            logical_read_bytes=read_bytes,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=estimated_hbm_bytes,
            rule=f"dense_projection_{self.ndim}d.v1",
            assumptions=assumptions,
            compute_type=_matmul_compute_type(op, lhs, rhs),
        )


@dataclass(frozen=True)
class BlockScaledMatmulCost:
    """E4M3 A[M,K] @ B[N,K].T with FP32 1x128 / 128x128 scales.

    The scope starts after activation quantization and includes any scale layout
    conversion performed by the selected GEMM. Only boundary IO is estimated.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            lhs, rhs = (op.inputs[name] for name in ("A", "B"))
            lhs_scale, rhs_scale = (
                op.inputs[name] for name in ("A_scales", "B_scales")
            )
            output = op.outputs["return"]
        except KeyError as exc:
            raise ValueError(f"Missing block-scaled GEMM tensor: {exc}") from exc
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError("Block-scaled GEMM requires 2D operands")
        m, k = lhs.shape
        n, rhs_k = rhs.shape
        if (
            min(m, n, k) < 1
            or k != rhs_k
            or k % 128
            or n % 64
            or output.shape != (m, n)
            or lhs_scale.shape != (m, k // 128)
            or rhs_scale.shape != ((n + 127) // 128, k // 128)
        ):
            raise ValueError("Block-scaled GEMM shapes or 128-wide scales disagree")
        if (
            lhs.dtype != "torch.float8_e4m3fn"
            or rhs.dtype != lhs.dtype
            or output.dtype != "torch.bfloat16"
            or lhs_scale.dtype != "torch.float32"
            or rhs_scale.dtype != "torch.float32"
            or any(t.device != lhs.device for t in (rhs, lhs_scale, rhs_scale, output))
        ):
            raise ValueError(
                "Expected colocated E4M3 operands, FP32 scales and BF16 output"
            )
        reads = sum(t.logical_bytes for t in (lhs, rhs, lhs_scale, rhs_scale))
        return OpCost(
            useful_flops=2 * m * n * k,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=(
                reads + output.logical_bytes
                if self.hbm_policy == "logical_io_once"
                else None
            ),
            rule="fp8_block_mm.v1",
            assumptions=(
                "useful matmul FLOPs; quantization, scale arithmetic and padding excluded",
                "activation quantization occurs before this GEMM scope",
                "FP8 storage does not establish the selected matrix engine; a reviewed mainloop/source-universe contract is required",
                "boundary IO read/write once; temporary scale layouts and reloads excluded",
                "logical IO does not establish physical HBM traffic",
            ),
            compute_type=None,
            compute_components=(
                ComputeComponent(
                    2 * m * n * k,
                    None,
                    missing_reason="FP8 matrix mainloop engine unavailable: reviewed source/fallback universe or matching selected-engine proof required",
                ),
            ),
        )


@dataclass(frozen=True)
class FP8TokenGroupQuantCost:
    """FP16/BF16 to E4M3 with one FP32 scale per 128 input elements.

    Counts per-element rescaling only, not amax comparisons or scale formation.
    Canonical scale copies are inside the op, but boundary IO excludes their
    intermediate buffers, just as it excludes GEMM's temporary scale layouts.
    Bound implementations enforce 128-wide groups; output scale geometry is
    checked below without collecting the same group size again.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        granularity = op.scalar_args.get(
            "granularity", op.static_traits.get("granularity")
        )
        scale_encoding = op.scalar_args.get(
            "scale_encoding", op.static_traits.get("scale_encoding")
        )
        if (
            granularity not in {"token_group", "token_group_128"}
            or scale_encoding != "float32"
        ):
            raise ValueError(
                "FP8 quantization cost requires 128-wide groups and FP32 scales"
            )
        try:
            x = op.inputs["x"]
            values, scales = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing FP8 quantization tensor: {exc}") from exc
        if (
            len(x.shape) != 2
            or min(x.shape) < 1
            or x.shape[-1] % 128
            or values.shape != x.shape
            or scales.shape != (x.shape[0], x.shape[1] // 128)
            or x.dtype not in {"torch.float16", "torch.bfloat16"}
            or values.dtype != "torch.float8_e4m3fn"
            or scales.dtype != "torch.float32"
            or any(t.device != x.device for t in (values, scales))
        ):
            raise ValueError(
                "FP8 quantization requires matching values and [M,K/128] FP32 scales"
            )
        writes = values.logical_bytes + scales.logical_bytes
        return OpCost(
            useful_flops=prod(x.shape),
            executed_flops=None,
            logical_read_bytes=x.logical_bytes,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                x.logical_bytes + writes
                if self.hbm_policy == "logical_io_once"
                else None
            ),
            rule="fp8_token_group_128.v1",
            assumptions=(
                "one FP32 rescale per input element; amax, scale formation and FP8 conversion excluded",
                "boundary input read and FP8 values/FP32 scales write once",
                "temporary scale layout copies, padded scale storage and physical reloads excluded",
                "FP8 output does not select a Tensor Core compute peak",
            ),
            compute_type=(
                "fp32_simt"
                if x.device.startswith("cuda:") and not op.autocast_enabled
                else None
            ),
        )


@dataclass(frozen=True)
class MLANoPEPrologueCost:
    """Native-cache absorbed MLA: assemble query tails and store new latent rows.

    Counts touched slices, never the allocated cache capacity or unchanged
    absorbed query channels. Masked writes, RoPE and quantized caches require
    separate rules; no device mask or slot values are read back here.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        if any(name.startswith(("rotary.", "expanded.")) for name in op.inputs):
            raise ValueError(
                "MLA NoPE prologue cost excludes RoPE and expanded attention"
            )
        if "cache.write_mask" in op.inputs:
            raise ValueError("MLA prologue masked write count is unavailable")
        try:
            query, q_pe, latent = (
                op.inputs[name] for name in ("query", "q_pe", "latent_cache")
            )
            cache, slots = (
                op.inputs[f"cache.{name}"] for name in ("kv_cache", "slots")
            )
            output = op.outputs["return.query"]
        except KeyError as exc:
            raise ValueError(f"Missing MLA NoPE prologue tensor: {exc}") from exc
        if len(query.shape) != 3 or len(q_pe.shape) != 3 or len(latent.shape) != 2:
            raise ValueError("MLA prologue requires query/q_pe[T,H,D] and latent[T,D]")
        tokens, heads, width = query.shape
        if (
            min(tokens, heads, width) < 1
            or q_pe.shape[:2] != (tokens, heads)
            or not 0 <= q_pe.shape[-1] < width
            or latent.shape != (tokens, width)
            or output.shape != query.shape
            or slots.shape != (tokens,)
            or slots.dtype not in {"torch.int32", "torch.int64"}
            or len(cache.shape) not in (3, 4)
            or cache.shape[-2:] != (1, width)
            or prod(cache.shape[:-2]) < tokens
        ):
            raise ValueError("MLA prologue input, cache and output geometry disagree")
        if (
            query.dtype not in {"torch.float16", "torch.bfloat16"}
            or any(t.dtype != query.dtype for t in (q_pe, latent, output))
            or cache.dtype not in {query.dtype, "torch.bfloat16"}
            or any(
                t.device != query.device for t in (q_pe, latent, output, cache, slots)
            )
        ):
            raise ValueError(
                "MLA NoPE prologue requires colocated native FP16/BF16 tensors"
            )
        cache_row_bytes = cache.logical_bytes // prod(cache.shape[:-2])
        reads = q_pe.logical_bytes + latent.logical_bytes + slots.logical_bytes
        writes = q_pe.logical_bytes + tokens * cache_row_bytes
        return OpCost(
            useful_flops=0,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="mla_nope_prologue.v1",
            assumptions=(
                "NoPE copies query tail channels and stores one latent row per token",
                "absorbed query prefix and previously cached tokens are not accessed",
                "slot indices included; cache capacity and index arithmetic excluded",
                "casts, sanitization comparisons and data movement have no model FLOPs",
                "logical IO estimate assumes one read/write of touched slices, excluding cache effects",
            ),
        )


@dataclass(frozen=True)
class MLACacheCopyCost:
    """Scatter/gather split latent and RoPE rows through an existing cache.

    Both entry points return None. Tensor roles determine reads and writes;
    only the rows selected by loc are accessed, not the whole cache capacity.
    """

    hbm_policy: str
    scatter: bool

    def __call__(self, op: OpRecord) -> OpCost:
        if self.scatter and "write_mask" in op.inputs:
            raise ValueError("MLA cache masked write count is unavailable")
        try:
            cache, slots, nope, rope = (
                op.inputs[name]
                for name in ("kv_buffer", "loc", "cache_k_nope", "cache_k_rope")
            )
        except KeyError as exc:
            raise ValueError(f"Missing MLA cache copy tensor: {exc}") from exc
        if (
            len(slots.shape) != 1
            or slots.dtype not in {"torch.int32", "torch.int64"}
            or slots.stride != (1,)
            or len(cache.shape) not in (3, 4)
            or min(cache.shape) < 1
            or cache.shape[-2] != 1
            or cache.stride[-1] != 1
        ):
            raise ValueError("MLA copy requires slot indices and native cache rows")
        rows = slots.shape[0]
        if (
            any(
                len(t.shape) not in (2, 3)
                or t.shape[0] != rows
                or prod(t.shape[1:-1]) != 1
                or t.stride[-1] != 1
                for t in (nope, rope)
            )
            or cache.shape[-1] != nope.shape[-1] + rope.shape[-1]
        ):
            raise ValueError("MLA cache copy input/output row geometry disagrees")
        if any(t.device != cache.device for t in (slots, nope, rope)):
            raise ValueError("MLA cache copy requires colocated tensors")
        cache_bytes = rows * (cache.logical_bytes // prod(cache.shape[:-2]))
        split_bytes = nope.logical_bytes + rope.logical_bytes
        reads, writes = (
            (split_bytes, cache_bytes) if self.scatter else (cache_bytes, split_bytes)
        )
        reads += slots.logical_bytes
        return OpCost(
            useful_flops=0,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule=f"mla_cache_{'scatter' if self.scatter else 'gather'}.v1",
            assumptions=(
                "one combined cache row per slot lookup, counting each tensor's storage dtype",
                "destination arguments are writes only; untouched rows and page gaps excluded",
                "slot indices included; repeated slots counted per access without reading their values",
                "casts, sanitization comparisons and index arithmetic have no model FLOPs",
                "logical IO estimate excludes physical reloads and cache effects",
            ),
        )


@dataclass(frozen=True)
class _FlashMLACost:
    """Adapt FlashMLA metadata to the common attention matmul count.

    QK and PV share latent cache storage: the value is a prefix of the key.
    The logical IO estimate reads each request's visible cache rows once,
    shared across query heads and positions; split/combine workspace is excluded.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        if op.scalar_args.get("is_fp8_kvcache") is not False or any(
            name in op.inputs
            for name in (
                "indices",
                "attn_sink",
                "extra_k_cache",
                "extra_indices_in_kvcache",
                "topk_length",
                "extra_topk_length",
            )
        ):
            raise ValueError(
                "Paged MLA cost supports dense, native-cache attention only"
            )
        lengths = op.tensor_values.get("cache_seqlens")
        if lengths is None:
            raise ValueError(
                "cache_seqlens snapshot unavailable; CUDA lengths require NvtxTimeline and a small contiguous integer vector"
            )
        causal = op.scalar_args.get("causal")
        value_dim = op.scalar_args.get("head_dim_v")
        if type(causal) is not bool or type(value_dim) is not int or value_dim < 1:
            raise ValueError("Paged MLA requires causal and head_dim_v metadata")
        try:
            q, cache, table, seq = (
                op.inputs[name]
                for name in ("q", "k_cache", "block_table", "cache_seqlens")
            )
            output, lse = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing paged MLA tensor: {exc}") from exc
        if len(q.shape) != 4 or len(cache.shape) != 4 or len(table.shape) != 2:
            raise ValueError(
                "Paged MLA requires q[B,Q,H,D], cache[P,S,1,D] and block_table[B,P]"
            )
        batch, query_length, heads, key_dim = q.shape
        pages, page_size, kv_heads, cache_dim = cache.shape
        if (
            min(*q.shape, *cache.shape) < 1
            or kv_heads != 1
            or cache_dim != key_dim
            or value_dim > key_dim
            or table.shape[0] != batch
            or seq.shape != (batch,)
            or output.shape != (batch, query_length, heads, value_dim)
            or lse.shape != (batch, heads, query_length)
            or len(lengths) != batch
        ):
            raise ValueError("Paged MLA shapes or visible lengths disagree")
        if (
            q.dtype not in {"torch.float16", "torch.bfloat16"}
            or cache.dtype != q.dtype
            or output.dtype != q.dtype
            or lse.dtype != "torch.float32"
            or table.dtype not in {"torch.int32", "torch.int64"}
            or seq.dtype not in {"torch.int32", "torch.int64"}
            or any(t.device != q.device for t in (cache, table, seq, output, lse))
        ):
            raise ValueError(
                "Paged MLA requires colocated matching native tensors and integer metadata"
            )
        flops = cache_rows = used_pages = 0
        mask = "causal_lower_right" if causal else "full"
        for length in lengths:
            if type(length) is not int or not 0 <= length <= page_size * table.shape[1]:
                raise ValueError("Paged MLA shapes or visible lengths disagree")
            flops += AttentionMatmulCost._matmul_flops(
                query_heads=heads,
                query_length=query_length,
                kv_length=length,
                qk_head_dim=key_dim,
                value_head_dim=value_dim,
                mask=mask,
            )
            cache_rows += length
            used_pages += (length + page_size - 1) // page_size
        cache_bytes = cache_rows * cache.logical_bytes // (pages * page_size)
        table_bytes = used_pages * (4 if table.dtype == "torch.int32" else 8)
        reads = q.logical_bytes + cache_bytes + table_bytes + seq.logical_bytes
        writes = output.logical_bytes + lse.logical_bytes
        return OpCost(
            useful_flops=flops,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="flashmla_decode.v1",
            assumptions=(
                "useful QK/PV matmuls with lower-right causal alignment and actual visible lengths",
                "single-token decode uses every visible KV row, without causal halving",
                "softmax, scaling, split/combine arithmetic and padding excluded from useful FLOPs",
                "each request's visible latent rows read once, shared by QK/PV and all query heads/positions",
                "HBM estimate excludes split/combine workspace, tile reloads and cross-request cache reuse",
                "cache/table allocation capacity is not traffic; only referenced logical rows/pages counted",
            ),
            compute_type=_matmul_compute_type(op, q, cache),
        )


@dataclass(frozen=True)
class _VarlenAttentionCost:
    """Adapt packed Q/K/V and cumulative lengths to the shared QK/PV count."""

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        if (
            op.scalar_args.get("window_size") != (-1, -1)
            or op.scalar_args.get("attention_chunk") != 0
            or any(
                name in op.inputs
                for name in (
                    "qv",
                    "q_descale",
                    "k_descale",
                    "v_descale",
                    "sinks",
                )
            )
        ):
            raise ValueError(
                "Variable-length cost supports full or causal native attention only"
            )
        causal = op.scalar_args.get("causal")
        if type(causal) is not bool:
            raise ValueError("Variable-length attention requires causal metadata")
        try:
            q, k, v = (op.inputs[name] for name in ("q", "k", "v"))
            output = (
                op.outputs["return"]
                if "return" in op.outputs
                else op.outputs["return[0]"]
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing variable-length attention tensor: {exc}"
            ) from exc
        if (
            any(len(t.shape) != 3 or min(t.shape) < 1 for t in (q, k, v))
            or k.shape[:2] != v.shape[:2]
            or q.shape[-1] != k.shape[-1]
            or q.shape[1] % k.shape[1]
            or output.shape != (*q.shape[:2], v.shape[-1])
            or q.dtype not in {"torch.float16", "torch.bfloat16"}
            or any(t.dtype != q.dtype or t.device != q.device for t in (k, v, output))
        ):
            raise ValueError(
                "Variable-length attention requires matching native packed Q/K/V and output"
            )
        lengths = []
        metadata_bytes = 0
        for axis, total in (("q", q.shape[0]), ("k", k.shape[0])):
            name = f"cu_seqlens_{axis}"
            values = op.tensor_values.get(name)
            if values is None:
                raise ValueError(
                    f"{name} snapshot unavailable; CUDA lengths require NvtxTimeline"
                )
            description = op.inputs.get(name)
            if (
                description is None
                or description.shape != (len(values),)
                or description.dtype not in {"torch.int32", "torch.int64"}
                or len(values) < 2
                or any(type(value) is not int for value in values)
                or values[0] != 0
                or values[-1] != total
                or any(left > right for left, right in zip(values, values[1:]))
            ):
                raise ValueError(
                    "Cumulative attention lengths do not partition the packed tensors"
                )
            metadata_bytes += description.logical_bytes
            capacities = tuple(right - left for left, right in zip(values, values[1:]))
            name = f"seqused_{axis}"
            used = op.inputs.get(name)
            if used is not None:
                values = op.tensor_values.get(name)
                if (
                    values is None
                    or used.shape != (len(capacities),)
                    or used.dtype != "torch.int32"
                    or used.device != q.device
                    or len(values) != len(capacities)
                    or any(
                        type(length) is not int or not 0 <= length <= capacity
                        for length, capacity in zip(values, capacities, strict=True)
                    )
                ):
                    raise ValueError(
                        f"{name} live lengths unavailable or out of bounds"
                    )
                metadata_bytes += used.logical_bytes
                lengths.append(values)
            else:
                lengths.append(capacities)
        if len(lengths[0]) != len(lengths[1]):
            raise ValueError(
                "Query and KV cumulative lengths have different request counts"
            )
        flops = sum(
            AttentionMatmulCost._matmul_flops(
                query_heads=q.shape[1],
                query_length=query_length,
                kv_length=kv_length,
                qk_head_dim=q.shape[2],
                value_head_dim=v.shape[2],
                mask="causal_lower_right" if causal else "full",
            )
            for query_length, kv_length in zip(*lengths, strict=True)
        )
        reads = q.logical_bytes + k.logical_bytes + v.logical_bytes + metadata_bytes
        writes = sum(t.logical_bytes for t in op.outputs.values())
        return OpCost(
            useful_flops=flops,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="attention_varlen.v1",
            assumptions=(
                "packed native Q/K/V with exact per-request lengths and lower-right causal alignment",
                "useful QK/PV work only; softmax, scaling, softcap and padding excluded",
                "logical boundary IO counted once, including offsets, optional live lengths and returned LSE",
                "HBM estimate excludes tiling reloads, intermediate workspace and cache effects",
            ),
            compute_type=_matmul_compute_type(op, q, k, v),
        )


@dataclass(frozen=True)
class AttentionStateMergeCost:
    """Merge two partial attention outputs and their FP32 log-sum-exp values.

    Counts the weighted value sum with normalized per-row weights: 3*N useful
    FLOPs. Weight formation and LSE arithmetic are excluded, including the
    Triton implementation's per-element normalization division.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            out_a, lse_a, out_b, lse_b = (
                op.inputs[name] for name in ("out_a", "lse_a", "out_b", "lse_b")
            )
            output, lse = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing attention merge tensor: {exc}") from exc
        if (
            len(out_a.shape) != 3
            or min(out_a.shape) < 1
            or any(t.shape != out_a.shape for t in (out_b, output))
            or any(t.shape != out_a.shape[:2] for t in (lse_a, lse_b, lse))
        ):
            raise ValueError("Attention merge requires values[T,H,D] and LSE[T,H]")
        if (
            out_a.dtype not in {"torch.float16", "torch.bfloat16"}
            or any(t.dtype != out_a.dtype for t in (out_b, output))
            or any(t.dtype != "torch.float32" for t in (lse_a, lse_b, lse))
            or any(t.device != out_a.device for t in (out_b, output, lse_a, lse_b, lse))
        ):
            raise ValueError(
                "Attention merge requires colocated FP16/BF16 values and FP32 LSE"
            )
        reads = sum(t.logical_bytes for t in (out_a, lse_a, out_b, lse_b))
        writes = output.logical_bytes + lse.logical_bytes
        return OpCost(
            useful_flops=3 * prod(output.shape),
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="attention_merge_state.v1",
            assumptions=(
                "two multiplies and one add per output value with normalized row weights",
                "weight formation, normalization division, exp/log and LSE arithmetic excluded",
                "both partial values and LSEs read once; merged values and LSE written once",
                "in-place output is a write to existing inputs, not an additional read",
                "FP32 weighted value arithmetic selects the FP32 SIMT reference peak",
                "logical IO estimate excludes physical reloads and cache effects",
            ),
            compute_type=(
                "fp32_simt"
                if out_a.device.startswith("cuda:") and not op.autocast_enabled
                else None
            ),
        )


@dataclass(frozen=True)
class RMSNormCost:
    """Weighted RMSNorm with FP32 arithmetic, optionally fusing two row norms.

    The parallel implementation returns None; its output arguments describe
    writes, including when an output aliases its corresponding input.
    """

    hbm_policy: str
    parallel: bool

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            tensors = (
                [
                    tuple(
                        op.inputs[f"{role}{i}"]
                        for role in ("input", "weight", "output")
                    )
                    for i in (1, 2)
                ]
                if self.parallel
                else [(op.inputs["input"], op.inputs["weight"], op.outputs["return"])]
            )
        except KeyError as exc:
            raise ValueError(f"Missing RMSNorm tensor: {exc}") from exc
        first = tensors[0][0]
        elements = reads = writes = 0
        for x, weight, output in tensors:
            if (
                len(x.shape) not in ((2,) if self.parallel else (2, 3))
                or not all(dim > 0 for dim in x.shape)
                or weight.shape != (x.shape[-1],)
                or output.shape != x.shape
                or x.shape[:-1] != first.shape[:-1]
            ):
                raise ValueError(
                    "RMSNorm requires matching rows, output and last-dimension weight"
                )
            if first.dtype not in {"torch.float16", "torch.bfloat16"} or any(
                t.dtype != first.dtype or t.device != first.device
                for t in (x, weight, output)
            ):
                raise ValueError(
                    "RMSNorm requires colocated matching FP16/BF16 tensors"
                )
            elements += prod(x.shape)
            reads += x.logical_bytes + weight.logical_bytes
            writes += output.logical_bytes
        return OpCost(
            useful_flops=4 * elements,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="rmsnorm_fused_parallel.v1" if self.parallel else "rmsnorm.v1",
            assumptions=(
                "4*N nominal RMSNorm FLOPs: square, sum, normalize and weight multiply",
                "per-row scalars, rsqrt, dtype conversions and reduction padding excluded",
                "FP32 arithmetic selects the FP32 SIMT reference peak, not the storage dtype peak",
                "one input/weight read and output write; in-place output is not an extra input",
                "logical IO estimate excludes physical reloads and cache effects",
            ),
            compute_type=(
                "fp32_simt"
                if first.device.startswith("cuda:") and not op.autocast_enabled
                else None
            ),
        )


@dataclass(frozen=True)
class CausalConv1dPrefillCost:
    """Depthwise convolution over packed x[channels, tokens].

    Channels include every packed Q/K/V projection. The state estimate counts
    one history row per sequence, never the capacity of the allocated pool.
    Device masks are not read back: initialized, nonempty sequences are assumed
    for the IO estimate; first chunks and padded slots can overestimate traffic.
    The Triton implementation accumulates in FP32. MFU uses the FP32 SIMT FMA
    reference peak, including when half-precision products are rounded before
    accumulation; it does not measure the mixed instruction pipeline's occupancy.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            x, weight = op.inputs["x"], op.inputs["weight"]
            boundaries = op.inputs["query_start_loc"]
            output = op.outputs["return"]
        except KeyError as exc:
            raise ValueError(f"Missing causal convolution tensor: {exc}") from exc
        if len(x.shape) != 2 or len(weight.shape) != 2:
            raise ValueError("Causal convolution requires x[C,T] and weight[C,W]")
        channels, tokens = x.shape
        weight_channels, width = weight.shape
        if channels != weight_channels or width < 1 or output.shape != x.shape:
            raise ValueError(
                "Causal convolution input, weight and output shapes differ"
            )
        if len(boundaries.shape) != 1 or boundaries.shape[0] < 1:
            raise ValueError("Causal convolution requires sequence boundaries")

        state_bytes = 0
        state = op.inputs.get("conv_states")
        if state is not None:
            if (
                len(state.shape) != 3
                or state.shape[1] != channels
                or state.shape[2] < width - 1
                or not prod(state.shape)
            ):
                raise ValueError("Causal convolution state shape does not match")
            element_bytes = state.logical_bytes // prod(state.shape)
            sequences = boundaries.shape[0] - 1
            state_bytes = sequences * channels * (width - 1) * element_bytes
        read_bytes = x.logical_bytes + weight.logical_bytes + state_bytes
        if "bias" in op.inputs:
            read_bytes += op.inputs["bias"].logical_bytes
        write_bytes = output.logical_bytes + state_bytes
        scalar_dtypes = {"torch.float16", "torch.bfloat16", "torch.float32"}
        compute_type = (
            "fp32_simt"
            if x.device.startswith("cuda:")
            and weight.device == x.device
            and x.dtype in scalar_dtypes
            and weight.dtype in scalar_dtypes
            and not op.autocast_enabled
            else None
        )
        return OpCost(
            useful_flops=2 * tokens * channels * width,
            executed_flops=None,
            logical_read_bytes=read_bytes,
            logical_write_bytes=write_bytes,
            estimated_hbm_bytes=(
                read_bytes + write_bytes
                if self.hbm_policy == "logical_io_once"
                else None
            ),
            rule="causal_conv1d_prefill.v1",
            assumptions=(
                "nominal depthwise convolution FLOPs; bias and SiLU excluded",
                "packed channels include all Q/K/V projections",
                "one input/weight read, output write and history read/write per sequence",
                "history estimate assumes initialized nonempty sequences; device masks unknown",
                "cache reuse, tile reloads and indexing metadata excluded from HBM estimate",
                "FP32 accumulation normalized to the FP32 SIMT FMA reference peak",
                "FP16/BF16 products may round before accumulation; MFU is not instruction-pipeline utilization",
            ),
            compute_type=compute_type,
        )


@dataclass(frozen=True)
class GatedDeltaPrefillCost:
    """Packed gated-delta scan with a value-major [N,H,V,K] state.

    Counts the equivalent recurrence, not the executed chunk factorization:
    state decay, state/key product, value correction, rank-one update and readout.
    Normalization, gate activations and input scaling are excluded. Mixed-precision
    chunk kernels do not establish one compute peak from Q/K/V storage dtypes.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            q, k, v = (op.inputs[name] for name in ("q", "k", "v"))
            state = op.inputs["initial_state"]
            output, final_state = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing gated-delta tensor: {exc}") from exc
        if len(q.shape) != 3 or k.shape != q.shape or len(v.shape) != 3:
            raise ValueError("Gated-delta scan requires packed q/k/v[T,H,D]")
        tokens, heads, key_dim = q.shape
        if v.shape[:2] != (tokens, heads) or output.shape != v.shape:
            raise ValueError("Gated-delta value/output shapes do not match")
        value_dim = v.shape[2]
        if (
            len(state.shape) != 4
            or state.shape[1:] != (heads, value_dim, key_dim)
            or final_state.shape != state.shape
        ):
            raise ValueError("Gated-delta scan requires value-major state[N,H,V,K]")
        reads = sum(
            tensor.logical_bytes
            for name, tensor in op.inputs.items()
            if name
            in {"q", "k", "v", "g", "beta_logits", "A_log", "dt_bias", "initial_state"}
        )
        writes = output.logical_bytes + final_state.logical_bytes
        return OpCost(
            useful_flops=tokens * heads * (7 * key_dim * value_dim + value_dim),
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="gated_delta_prefill.v1",
            assumptions=(
                "equivalent recurrence FLOPs; chunk factorization and recomputation excluded",
                "per token/head: K*V decay, 2*K*V prediction, V subtraction, 2*K*V update, 2*K*V readout",
                "normalization, nonlinear gates and featurewise beta scaling excluded",
                "one boundary input read and output write; intermediate chunk buffers excluded",
                "mixed-precision implementation; storage dtype does not select one compute peak",
            ),
        )


@dataclass(frozen=True)
class CausalConv1dUpdateCost:
    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        if (
            op.cost_rule == "causal_conv1d_update.v3"
            and "output_state_indices" in op.inputs
        ):
            return self._output_bank_cost(op)
        try:
            x, weight, state = (
                op.inputs[name] for name in ("x", "weight", "conv_state")
            )
            if (
                len(x.shape) != 2
                or "parent_indices" in op.inputs
                or "output_state_indices" in op.inputs
                or "intermediate_conv_window" in op.inputs
            ):
                raise ValueError("Only ordinary live decode convolution is modeled")
            batch, channels = x.shape
            width = weight.shape[1]
            if (
                weight.shape[0] != channels
                or state.shape[-2] != channels
                or state.shape[-1] < width - 1
            ):
                raise ValueError("Convolution history geometry mismatch")
            if "conv_state_indices" in op.inputs:
                indices = op.tensor_values.get("conv_state_indices")
                if (
                    indices is None
                    or len(indices) != batch
                    or any(index < 0 for index in indices)
                ):
                    raise ValueError(
                        "Actual nonpadded convolution row indices unavailable"
                    )
            from math import prod

            element = state.logical_bytes // prod(state.shape)
            history = batch * channels * (width - 1) * element
            reads = x.logical_bytes + weight.logical_bytes + history
            reads += sum(
                op.inputs[name].logical_bytes
                for name in ("bias", "cache_seqlens", "conv_state_indices")
                if name in op.inputs
            )
            writes = op.outputs["return"].logical_bytes + history
            return OpCost(
                2 * batch * channels * width,
                None,
                reads,
                writes,
                reads + writes if self.hbm_policy == "logical_io_once" else None,
                (
                    "causal_conv1d_update.v3"
                    if op.cost_rule == "causal_conv1d_update.v3"
                    else "causal_conv1d_update.v2"
                ),
                (
                    "nominal useful depthwise convolution products; SiLU and indexing excluded",
                    "ordinary one-token decode: W-1 initialized history elements per live channel read/write once, not the allocated pool",
                    "logical IO policy, not physical HBM traffic; mixed operand-product/accumulation instruction precision remains unproven",
                ),
                compute_type=None,
                compute_components=_conv_components(op, 2 * batch * channels * width),
            )
        except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
            raise ValueError(
                f"Convolution profile metadata unavailable: {error}"
            ) from error

    def _output_bank_cost(self, op: OpRecord) -> OpCost:
        """Count one-token decode with live rows or the declared pad sentinel.

        Replay binding clears capture-time integers before filling this op's
        snapshots. Missing, stale or dropped snapshots therefore stay unknown.
        A padded source returns before data/output access; negative destinations
        suppress only live-row history writes.
        """
        if (
            op.op_name != "conv.causal_conv1d_update"
            or op.implementation != "triton_causal_conv1d_update"
        ):
            raise ValueError("Output-bank convolution requires the Triton update leaf")
        if any(
            name in op.inputs
            for name in (
                "parent_indices",
                "intermediate_conv_window",
                "num_accepted_tokens",
                "cache_seqlens",
            )
        ):
            raise ValueError(
                "Only ordinary one-token output-bank convolution is modeled"
            )
        if (
            type(op.scalar_args.get("pad_slot_id")) is not int
            or op.scalar_args["pad_slot_id"] != -1
        ):
            raise ValueError("Output-bank convolution requires recorded pad_slot_id=-1")
        try:
            x, weight, state, sources, destinations = (
                op.inputs[name]
                for name in (
                    "x",
                    "weight",
                    "conv_state",
                    "conv_state_indices",
                    "output_state_indices",
                )
            )
            output = op.outputs["return"]
        except KeyError as error:
            raise ValueError(
                f"Output-bank convolution tensor unavailable: {error}"
            ) from error
        if len(x.shape) != 2 or len(weight.shape) != 2 or len(state.shape) != 3:
            raise ValueError(
                "Output-bank convolution requires x[B,D], weight[D,K], state[P,D,S]"
            )
        batch, channels = x.shape
        pages, state_channels, state_length = state.shape
        weight_channels, width = weight.shape
        if (
            min(batch, channels, pages) < 1
            or width not in {2, 3, 4}
            or weight_channels != channels
            or state_channels != channels
            or state_length < width - 1
            or output.shape != x.shape
            or output.dtype != x.dtype
            or output.stride != x.stride
        ):
            raise ValueError(
                "Output-bank convolution geometry or return metadata mismatch"
            )
        if any(
            tensor.dtype not in {"torch.float16", "torch.bfloat16", "torch.float32"}
            or tensor.device != x.device
            for tensor in (x, weight, state, output)
        ):
            raise ValueError(
                "Output-bank convolution requires colocated floating tensors"
            )
        bias = op.inputs.get("bias")
        if bias is not None and (
            bias.shape != (channels,)
            or bias.dtype not in {"torch.float16", "torch.bfloat16", "torch.float32"}
            or bias.device != x.device
        ):
            raise ValueError("Output-bank convolution bias metadata mismatch")
        values = {}
        for name, tensor, shape in (
            ("conv_state_indices", sources, (batch,)),
            ("output_state_indices", destinations, (batch, 1)),
        ):
            integers = op.tensor_values.get(name)
            if (
                tensor.shape != shape
                or tensor.dtype not in {"torch.int32", "torch.int64"}
                or tensor.device != x.device
                or integers is None
                or len(integers) != batch
                or any(type(value) is not int for value in integers)
            ):
                raise ValueError(
                    f"Actual output-bank convolution {name} snapshot unavailable or invalid"
                )
            values[name] = integers
        source_rows = values["conv_state_indices"]
        destination_rows = values["output_state_indices"]
        if any(row != -1 and not 0 <= row < pages for row in source_rows):
            raise ValueError(
                "Output-bank convolution source is neither pad_slot_id nor a valid page"
            )
        live_rows = [
            (source, destination)
            for source, destination in zip(source_rows, destination_rows)
            if source != -1
        ]
        if any(destination >= pages for _, destination in live_rows):
            raise ValueError(
                "Output-bank convolution destination exceeds the state pool"
            )
        source_counts = {}
        for source, _ in live_rows:
            source_counts[source] = source_counts.get(source, 0) + 1
        written_rows = set()
        for source, destination in live_rows:
            if destination < 0:
                continue
            if destination in written_rows or (
                destination in source_counts
                and (destination != source or source_counts[destination] > 1)
            ):
                raise ValueError(
                    "Output-bank convolution has colliding writes or cross-row history dependencies"
                )
            written_rows.add(destination)
        state_element_bytes = state.logical_bytes // prod(state.shape)
        history_per_row = channels * (width - 1) * state_element_bytes
        live = len(live_rows)
        reads = sources.logical_bytes
        if live:
            reads += (
                x.logical_bytes // batch * live
                + weight.logical_bytes
                + live * history_per_row
                + (bias.logical_bytes if bias is not None else 0)
                + destinations.logical_bytes // batch * live
            )
        writes = (
            output.logical_bytes // batch * live + len(written_rows) * history_per_row
        )
        return OpCost(
            useful_flops=2 * live * channels * width,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="causal_conv1d_update.v3",
            assumptions=(
                "nominal useful depthwise convolution products; "
                "bias, SiLU and indexing arithmetic excluded",
                "one-token non-tree output-bank decode; pad_slot_id=-1 reads only its source index and leaves output/history unchanged; negative live destinations suppress history writes",
                "one logical input, weight, bias, index and source-history read; "
                "activation and nonnegative destination history written once",
                "aliased x still has a logical read and write; "
                "state-pool capacity and stride gaps excluded",
                "logical IO policy, not physical HBM traffic; operand-product/"
                "accumulation instruction precision remains unproven",
            ),
            compute_type=None,
            compute_components=_conv_components(op, 2 * live * channels * width),
        )


def _conv_components(op: OpRecord, flops: int) -> tuple[ComputeComponent, ...]:
    proven = (
        op.scalar_args.get("compiled_contract_id") == "causal_conv1d_update.triton.v1"
        and op.scalar_args.get("compiled_target") in {"90", "90a"}
        and op.scalar_args.get("compiled_bf16_mul_fp32_add") is True
        and bool(op.scalar_args.get("compiled_kernel_hash"))
        and all(
            op.inputs[name].dtype == "torch.bfloat16"
            for name in ("x", "weight", "conv_state")
        )
        and not op.autocast_enabled
    )
    return tuple(
        ComputeComponent(
            flops // 2,
            key if proven else None,
            "causal_conv1d_update.triton.v1" if proven else None,
            (
                None
                if proven
                else "Conv actual launch-returned SM90 BF16-mul/FP32-add artifact proof unavailable"
            ),
        )
        for key in ("bf16_simt", "fp32_simt")
    )


@dataclass(frozen=True)
class SmallOpCost:
    """Exact small source boundaries; comparisons/indices have zero model FLOPs."""

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            rule, a = op.cost_rule, op.scalar_args
            components = None
            assumptions = (
                "logical boundary tensor regions once; not measured HBM traffic",
            )
            if rule == "silu_mul.v1":
                x, y = op.inputs["x"], op.outputs["return"]
                if (
                    len(x.shape) != 2
                    or x.shape[-1] % 2
                    or x.dtype != "torch.bfloat16"
                    or y.shape != (x.shape[0], x.shape[1] // 2)
                    or y.dtype != x.dtype
                    or a.get("limit") is not None
                    or op.autocast_enabled
                    or not x.device.startswith("cuda:")
                    or y.device != x.device
                    or any(
                        alias != "return=out" for alias in a.get("output_aliases", ())
                    )
                ):
                    raise ValueError(
                        "Only unclamped CUDA BF16 SwiGLU with a separate output is modeled"
                    )
                f = 2 * prod(y.shape)
                reads, writes = x.logical_bytes, y.logical_bytes
                engine = (
                    "fp32_simt"
                    if a.get("activation_engine_contract")
                    == "flashinfer.silu.float.b2.v1"
                    else None
                )
                components = (
                    ComputeComponent(
                        f,
                        engine,
                        a.get("activation_engine_contract"),
                        (
                            None
                            if engine
                            else "SwiGLU exact installed source and loaded activation artifact proof unavailable"
                        ),
                    ),
                )
                assumptions += (
                    "two useful gate/up products; sigmoid SFU/division implementation excluded; local J from actual x",
                )
            elif rule == "vocab_shard_embedding.v1":
                weight, ids, y = (
                    op.inputs["weight"],
                    op.inputs["indices"],
                    op.outputs["return"],
                )
                values = op.tensor_values.get("indices")
                begin, end = a["org_range"]
                added_begin, added_end = a["added_range"]
                padding = a["num_org_padding"]
                if (
                    len(weight.shape) != 2
                    or len(ids.shape) != 1
                    or values is None
                    or len(values) != prod(ids.shape)
                    or y.shape != (*ids.shape, weight.shape[1])
                    or weight.dtype != y.dtype
                    or not weight.device.startswith("cuda:")
                    or weight.device != ids.device
                    or weight.device != y.device
                    or weight.stride[-1] != 1
                    or min(begin, added_begin, padding) < 0
                    or end < begin
                    or added_end < added_begin
                ):
                    raise ValueError(
                        "Actual vocabulary shard indices/layout unavailable"
                    )
                offset = added_begin - (end - begin) - padding
                rows = {
                    v - begin if begin <= v < end else v - offset
                    for v in values
                    if begin <= v < end or added_begin <= v < added_end
                }
                if any(not 0 <= r < weight.shape[0] for r in rows):
                    raise ValueError("Vocabulary row outside recorded table")
                f = 0
                reads = ids.logical_bytes + len(rows) * weight.shape[1] * (
                    weight.logical_bytes // prod(weight.shape)
                )
                writes = y.logical_bytes
                assumptions += (
                    "actual unique local table-row working set; invalid/nonlocal rows write zero; allocated table capacity excluded",
                )
            elif rule == "distributed_argmax.v1":
                logits, y, ids = (
                    op.inputs["logits"],
                    op.outputs["return[0]"],
                    op.outputs["return[1]"],
                )
                m, n = logits.shape
                if (
                    y.shape != (m,)
                    or ids.shape != (m,)
                    or ids.dtype != "torch.int64"
                    or a["state.world_size"] < 1
                ):
                    raise ValueError("Distributed argmax output/rank metadata mismatch")
                f, reads, writes = (
                    0,
                    logits.logical_bytes,
                    y.logical_bytes + ids.logical_bytes,
                )
                assumptions += (
                    "comparisons/integer IDs are zero model floating arithmetic; NVLS peer/synchronization IO not inferred from buffer capacity",
                )
                if a["state.world_size"] > 1:
                    return OpCost(
                        f,
                        None,
                        reads,
                        writes,
                        (
                            reads + writes
                            if self.hbm_policy == "logical_io_once"
                            else None
                        ),
                        rule,
                        assumptions
                        + (
                            "logical_io_once counts only boundary logits and returned values/IDs; NVLS protocol, synchronization workspace and reload physical traffic excluded",
                            "comm_unavailable_reason: distributed argmax NVLS traffic unmodeled; no AllReduce payload inferred",
                        ),
                    )
            elif rule in {
                "integer_mask_ids.v1",
                "integer_special_mask.v1",
                "mask_word.v1",
            }:
                f = 0
                if rule == "integer_mask_ids.v1":
                    ids = op.inputs["input_ids"]
                    invalid, safe = op.outputs["return[0]"], op.outputs["return[1]"]
                    if (
                        len(ids.shape) != 1
                        or ids.dtype not in {"torch.int32", "torch.int64"}
                        or invalid.shape != ids.shape
                        or invalid.dtype != "torch.bool"
                        or safe.shape != ids.shape
                        or safe.dtype != ids.dtype
                        or a["vocab_size"] < 1
                    ):
                        raise ValueError(
                            "Safe-ID source tensor geometry/dtype mismatch"
                        )
                    reads = ids.logical_bytes
                    writes = sum(t.logical_bytes for t in op.outputs.values())
                elif rule == "integer_special_mask.v1":
                    ids, special, y = (
                        op.inputs["input_ids"],
                        op.inputs["special_token_ids"],
                        op.outputs["return"],
                    )
                    if (
                        len(ids.shape) != 1
                        or len(special.shape) != 1
                        or y.shape != ids.shape
                        or ids.dtype not in {"torch.int32", "torch.int64"}
                        or special.dtype != "torch.int32"
                        or y.dtype != "torch.bool"
                        or a["vocab_size"] < 1
                    ):
                        raise ValueError(
                            "Special-ID mask source tensor geometry/dtype mismatch"
                        )
                    reads = (
                        op.inputs["input_ids"].logical_bytes
                        + op.inputs["special_token_ids"].logical_bytes
                    )
                    writes = op.outputs["return"].logical_bytes
                else:
                    word, invalid = op.inputs["word"], op.inputs["invalid"]
                    if (
                        len(word.shape) != 2
                        or invalid.shape != (word.shape[0],)
                        or invalid.dtype != "torch.bool"
                        or word.device != invalid.device
                    ):
                        raise ValueError(
                            "Word-mask source tensor geometry/dtype mismatch"
                        )
                    reads = (
                        op.inputs["word"].logical_bytes
                        + op.inputs["invalid"].logical_bytes
                    )
                    writes = op.inputs["word"].logical_bytes
                    if "return=word" not in a.get("output_aliases", ()):
                        raise ValueError(
                            "Mask-word source must return its in-place input"
                        )
            elif rule == "longcat_residual_add.v1":
                x, shared, y = (
                    op.inputs["output"],
                    op.inputs["shared"],
                    op.outputs["return"],
                )
                if (
                    len(x.shape) != 2
                    or x.shape != shared.shape
                    or x.shape != y.shape
                    or x.dtype != shared.dtype
                    or x.dtype != y.dtype
                    or x.dtype != "torch.bfloat16"
                    or x.device != shared.device
                    or x.device != y.device
                    or a.get("output_aliases")
                    or a.get("combine_order") not in {"rank", "slot"}
                    or op.autocast_enabled
                ):
                    raise ValueError(
                        "Routed/shared add requires matching BF16 partial tensors and no alias"
                    )
                f, reads, writes = (
                    prod(x.shape),
                    x.logical_bytes + shared.logical_bytes,
                    y.logical_bytes,
                )
                engine = (
                    "fp32_simt"
                    if a.get("pytorch_add_engine_contract")
                    == "torch.bf16_add.sm90.vector8.b2.v1"
                    and x.shape == (1, 4096)
                    else None
                )
                components = (
                    ComputeComponent(
                        f,
                        engine,
                        a.get("pytorch_add_engine_contract"),
                        (
                            None
                            if engine
                            else "BF16 add exact loaded SM90 vector8 source/artifact proof unavailable"
                        ),
                    ),
                )
                assumptions += (
                    "one add per actual local element; PyTorch BF16 storage does not establish arithmetic engine",
                )
            elif rule == "sampling_broadcast.v1":
                tensor = op.inputs["tensor"]
                ranks = a["ranks"]
                if len(set(ranks)) != len(ranks) or a["src"] not in ranks:
                    raise ValueError("Sampling broadcast source/group mismatch")
                traffic = CollectiveTraffic(
                    "broadcast",
                    tensor.logical_bytes,
                    len(ranks),
                    "ranks=" + ",".join(str(rank) for rank in ranks),
                )
                return OpCost(
                    0,
                    None,
                    tensor.logical_bytes,
                    tensor.logical_bytes,
                    None,
                    rule,
                    (
                        "sampling support outside Model; Broadcast normalized payload uses its own NCCL-tests factor1, never AllReduce factor",
                        "backend transport/placement unproven; no communication capacity normalization",
                    ),
                    communication=traffic,
                )
            elif rule == "sampling_support.v1":
                f = 0
                if op.op_name == "sampling.verify_chain_greedy":
                    reads = sum(
                        op.inputs[n].logical_bytes
                        for n in ("candidates", "target_predict")
                    )
                    writes = sum(
                        op.inputs[n].logical_bytes
                        for n in ("predicts", "accept_index", "accept_token_num")
                    )
                elif op.op_name == "sampling.increment_accept_lengths":
                    reads = writes = op.inputs["accept_length"].logical_bytes
                elif op.op_name == "sampling.apply_stop_conditions":
                    # The slot-indexed source pool is not the working set.
                    return OpCost(
                        0,
                        None,
                        op.inputs["req_pool_indices"].logical_bytes,
                        0,
                        None,
                        rule,
                        (
                            "named active sampling support outside Model; no model floating arithmetic",
                            "slot-indexed stop-state IO unavailable; recorded bytes only a known index subtotal, never pool capacity",
                        ),
                    )
                else:
                    raise ValueError("Unsupported named sampling support source")
                assumptions += (
                    "named active sampling support outside Model; integer verification/update contributes zero model FLOPs",
                )
            else:
                raise ValueError("Unknown small-op contract")
            return OpCost(
                f,
                None,
                reads,
                writes,
                reads + writes if self.hbm_policy == "logical_io_once" else None,
                rule,
                assumptions,
                compute_components=components,
            )
        except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
            raise ValueError(f"Small-op metadata unavailable: {error}") from error


def register_compute_costs(registry: CostRegistry, *, hbm_policy: str) -> None:
    """Register costs for opted-in compute, quantization and MLA cache writes.

    Args:
        registry: Cost registry receiving the named rules.
        hbm_policy: 'unavailable' or the rough estimate 'logical_io_once'.

    Compute types require explicit implementation contracts/host decisions.
    Missing evidence leaves MFU unavailable; storage/output dtype never selects
    a compute engine or a compatible peak.
    Projection rules count matrix products only; bias and conversion are excluded.
    RMSNorm uses nominal FP32 arithmetic counts, not its storage precision peak.
    Registers the rules in place and returns None.
    """
    if hbm_policy not in {"unavailable", "logical_io_once"}:
        raise ValueError(f"Unsupported HBM policy: {hbm_policy}")
    from tokenspeed_kernel.profile_attention_moe import register_graph_costs

    register_graph_costs(registry, hbm_policy=hbm_policy)
    for rule in (
        "silu_mul.v1",
        "vocab_shard_embedding.v1",
        "distributed_argmax.v1",
        "integer_mask_ids.v1",
        "integer_special_mask.v1",
        "mask_word.v1",
        "longcat_residual_add.v1",
        "sampling_support.v1",
        "sampling_broadcast.v1",
    ):
        registry.register(rule, SmallOpCost(hbm_policy))
    for name, ndim in (("torch_mm.v1", 2), ("torch_bmm.v1", 3)):
        registry.register(
            name,
            _DenseProjectionCost(ndim=ndim, hbm_policy=hbm_policy),
        )
    registry.register("fp8_block_mm.v1", BlockScaledMatmulCost(hbm_policy=hbm_policy))
    registry.register("fp8_token_group_128.v1", FP8TokenGroupQuantCost(hbm_policy))
    registry.register("mla_nope_prologue.v1", MLANoPEPrologueCost(hbm_policy))
    for mode, scatter in (("scatter", True), ("gather", False)):
        registry.register(
            f"mla_cache_{mode}.v1", MLACacheCopyCost(hbm_policy, scatter=scatter)
        )
    registry.register("flashmla_decode.v1", _FlashMLACost(hbm_policy))
    registry.register("attention_varlen.v1", _VarlenAttentionCost(hbm_policy))
    registry.register("attention_merge_state.v1", AttentionStateMergeCost(hbm_policy))
    for name, parallel in (("rmsnorm.v1", False), ("rmsnorm_fused_parallel.v1", True)):
        registry.register(name, RMSNormCost(hbm_policy=hbm_policy, parallel=parallel))
    registry.register("causal_conv1d_update.v2", CausalConv1dUpdateCost(hbm_policy))
    registry.register("causal_conv1d_update.v3", CausalConv1dUpdateCost(hbm_policy))
    registry.register(
        "causal_conv1d_prefill.v1", CausalConv1dPrefillCost(hbm_policy=hbm_policy)
    )
    registry.register(
        "gated_delta_prefill.v1", GatedDeltaPrefillCost(hbm_policy=hbm_policy)
    )


@dataclass(frozen=True)
class CollectiveTraffic:
    """NCCL-tests normalized traffic, not measured link or HBM bytes.

    total_bytes is the full AllReduce/ReduceScatter input or AllGather output,
    already including element size. group identifies the participating ranks.
    """

    collective: str
    total_bytes: int
    world_size: int
    group: str

    def __post_init__(self) -> None:
        if self.collective not in {
            "all_reduce_sum",
            "reduce_scatter_sum",
            "all_gather",
            "broadcast",
        }:
            raise ValueError(f"Unsupported collective: {self.collective}")
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if type(self.total_bytes) is not int or self.total_bytes < 0:
            raise ValueError("total_bytes must be a nonnegative integer byte count")
        if not self.group.strip():
            raise ValueError("Communication group identity is required")

    @property
    def bus_bytes(self) -> float:
        """Return per-rank equivalent bytes; divide by seconds to obtain busbw."""
        if self.collective == "broadcast":
            return float(self.total_bytes if self.world_size > 1 else 0)
        phases = 2 if self.collective == "all_reduce_sum" else 1
        return self.total_bytes * phases * (self.world_size - 1) / self.world_size


@dataclass(frozen=True)
class CollectiveCost:
    """Describe one rank's synchronous collective using its full-array tensor.

    collective selects all_reduce_sum, reduce_scatter_sum or all_gather.
    tensor and tensor_source (input/output) identify the full input for reduction
    or full output for gathering; in-place output buffers can be input arguments.
    world_size is the participating group size, not the machine's GPU count.
    group must uniquely identify that group within the profile session.
    """

    collective: str
    tensor: str
    tensor_source: str
    world_size: int
    group: str

    def __post_init__(self) -> None:
        # Validate group metadata before a runtime tensor is available.
        CollectiveTraffic(
            collective=self.collective,
            total_bytes=0,
            world_size=self.world_size,
            group=self.group,
        )
        if self.tensor_source not in {"input", "output"}:
            raise ValueError("tensor_source must be input or output")

    def __call__(self, op: OpRecord) -> OpCost:
        tensors = op.inputs if self.tensor_source == "input" else op.outputs
        if self.tensor not in tensors:
            raise ValueError(f"Missing collective full-array tensor: {self.tensor}")
        tensor = tensors[self.tensor]
        numel = prod(tensor.shape)
        if self.collective != "all_reduce_sum" and numel % self.world_size:
            raise ValueError("Full collective array must split evenly across ranks")

        return self._from_bytes(tensor.logical_bytes)

    def _from_bytes(self, total_bytes: int) -> OpCost:
        """Shared boundary accounting, including a grouped tuple of tensors."""
        shard_bytes = total_bytes // self.world_size
        read_bytes = shard_bytes if self.collective == "all_gather" else total_bytes
        write_bytes = (
            shard_bytes if self.collective == "reduce_scatter_sum" else total_bytes
        )
        return OpCost(
            useful_flops=0,
            executed_flops=None,
            logical_read_bytes=read_bytes,
            logical_write_bytes=write_bytes,
            estimated_hbm_bytes=None,
            rule="nccl_collective.v1",
            assumptions=(
                "total bytes already include dtype element size",
                "NCCL-tests per-rank normalized bus bytes, not physical link traffic",
                "equal partitions for ReduceScatter and AllGather",
                "reduction arithmetic excluded from model useful FLOPs",
                "collective work must join the timed stream before scope exit",
            ),
            communication=CollectiveTraffic(
                collective=self.collective,
                total_bytes=total_bytes,
                world_size=self.world_size,
                group=self.group,
            ),
        )


def _collective_group(op: OpRecord) -> tuple[tuple[int, ...], str]:
    ranks = op.scalar_args.get("ranks")
    if (
        not isinstance(ranks, tuple)
        or not ranks
        or any(type(r) is not int for r in ranks)
    ):
        raise ValueError("AllReduce requires the participating group ranks")
    if len(set(ranks)) != len(ranks):
        raise ValueError("AllReduce group ranks must be unique")
    name = op.scalar_args.get("group_name")
    return ranks, f"nccl:{name!r}:{ranks}"


@dataclass(frozen=True)
class _NcclAllReduceCost:
    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        ranks, group = _collective_group(op)
        if op.scalar_args.get("is_sum") is not True:
            raise ValueError("AllReduce cost currently models SUM only")
        tensors = [
            tensor
            for name, tensor in op.inputs.items()
            if name in {"tensor", "first", "second"} or name.startswith("tensor[")
        ]
        if not tensors:
            raise ValueError("Missing AllReduce input tensor")
        # Tuple and grouped-pair entries run their reductions in one outer scope.
        total_bytes = sum(t.logical_bytes for t in tensors) if len(ranks) > 1 else 0
        cost = CollectiveCost(
            "all_reduce_sum", "tensor", "input", len(ranks), group
        )._from_bytes(total_bytes)
        if self.hbm_policy == "logical_io_once":
            cost = replace(
                cost,
                estimated_hbm_bytes=cost.logical_read_bytes + cost.logical_write_bytes,
                assumptions=cost.assumptions
                + (
                    "HBM estimate counts one input read and output write; protocol buffers and reloads excluded",
                ),
            )
        return cost


@dataclass(frozen=True)
class AllReduceRMSNormCost:
    """TRT-LLM SUM + residual + RMSNorm, or its unfused fallback.

    Both IPC and MNNVL normalize the full result on every rank, including
    two-shot: only the collective reduction is sharded. Quantization and
    reduce-scattered residual variants require separate cost rules.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        ranks, group = _collective_group(op)
        fused = op.scalar_args.get("fused")
        if type(fused) is not bool:
            raise ValueError("AllReduce RMSNorm requires the actual fusion decision")
        if op.scalar_args.get("block_quant_fp8", False):
            raise ValueError("Quantized AllReduce RMSNorm cost is not modeled")
        if op.scalar_args.get("residual_reduce_scattered", False):
            raise ValueError("Reduce-scattered residual RMSNorm cost is not modeled")
        try:
            x, residual, weight = (
                op.inputs[name] for name in ("input_tensor", "residual", "weight")
            )
            output, residual_out = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing AllReduce RMSNorm tensor: {exc}") from exc
        if (
            len(x.shape) != 2
            or min(x.shape) < 1
            or any(t.shape != x.shape for t in (residual, output, residual_out))
            or weight.shape != (x.shape[1],)
            or any(
                t.dtype != x.dtype or t.device != x.device
                for t in (residual, weight, output, residual_out)
            )
        ):
            raise ValueError(
                "AllReduce RMSNorm requires matching [T,H] tensors and weight[H]"
            )
        reads = x.logical_bytes + residual.logical_bytes + weight.logical_bytes
        writes = output.logical_bytes + residual_out.logical_bytes
        if op.scalar_args.get("has_partial_norm_out", False):
            partial = op.outputs.get("return[3]")
            rank = op.scalar_args.get("rank")
            if type(rank) is not int or not 0 <= rank < len(ranks):
                raise ValueError("Partial RMSNorm output requires the group-local rank")
            rows = x.shape[0] // len(ranks) + int(rank < x.shape[0] % len(ranks))
            if (
                partial is None
                or partial.shape != (rows, x.shape[1])
                or partial.dtype != x.dtype
                or partial.device != x.device
            ):
                raise ValueError("Partial RMSNorm output shape or dtype does not match")
            # This fused pattern writes only partial_norm_out, not the allocated
            # norm_out buffer. The fallback instead returns a view of norm_out.
            if fused:
                writes = residual_out.logical_bytes + partial.logical_bytes
        assumptions = (
            "full RMSNorm epilogue on every rank for both one-shot and two-shot",
            "5*T*H useful FLOPs: residual add plus RMSNorm; per-row scalars and rsqrt excluded",
            "collective reduction arithmetic excluded from model useful FLOPs",
            "normalized AllReduce bus bytes use the original input dtype and full tensor size",
            "bandwidth and MFU divide by the whole scope window, not a communication-only time",
            "logical IO counts boundary tensors once; protocol buffers, cache reuse and reloads excluded",
            "FP32 RMSNorm arithmetic uses the FP32 SIMT reference peak; residual rounding may differ",
        )
        if not fused:
            assumptions += (
                "unfused NCCL and PyTorch epilogue; intermediate HBM traffic is not modeled",
            )
        return OpCost(
            useful_flops=5 * prod(x.shape),
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes
                if fused and self.hbm_policy == "logical_io_once"
                else None
            ),
            rule=(
                "all_reduce_residual_rmsnorm.v1"
                if fused
                else "all_reduce_residual_rmsnorm_unfused.v1"
            ),
            assumptions=assumptions,
            compute_type=(
                "fp32_simt"
                if x.device.startswith("cuda:")
                and x.dtype in {"torch.float16", "torch.bfloat16", "torch.float32"}
                and not op.autocast_enabled
                else None
            ),
            communication=CollectiveTraffic(
                "all_reduce_sum", x.logical_bytes, len(ranks), group
            ),
        )


def register_collective_costs(registry: CostRegistry, *, hbm_policy: str) -> None:
    """Register SUM and residual/RMSNorm costs using recorded process-group ranks."""
    if hbm_policy not in {"unavailable", "logical_io_once"}:
        raise ValueError(f"Unsupported HBM policy: {hbm_policy}")
    registry.register("nccl_all_reduce_sum.v1", _NcclAllReduceCost(hbm_policy))
    registry.register(
        "all_reduce_residual_rmsnorm.v1", AllReduceRMSNormCost(hbm_policy)
    )


@dataclass(frozen=True)
class AttentionMatmulCost:
    """Fixed-length attention cost for explicit [batch, heads, sequence, dim] roles.

    q/k/v/output name recorded tensor arguments and return paths. mask selects
    full, causal_upper_left or causal_lower_right alignment for unpadded sequences.
    Compute type is inferred from Q/K/V GPU dtypes; unknown precision stays None.
    Counts useful QK^T and PV FLOPs with an FMA counted as two. GQA uses query heads;
    scaling, softmax, projections, dropout, padding and recompute are excluded.
    """

    q: str
    k: str
    v: str
    output: str
    mask: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            q = op.inputs[self.q]
            k = op.inputs[self.k]
            v = op.inputs[self.v]
            output = op.outputs[self.output]
        except KeyError as exc:
            raise ValueError(f"Missing attention tensor role: {exc}") from exc
        if any(len(tensor.shape) != 4 for tensor in (q, k, v, output)):
            raise ValueError(
                "AttentionMatmulCost requires [batch, heads, sequence, dim] tensors"
            )

        batch_size, query_heads, query_length, qk_head_dim = q.shape
        key_batch, kv_heads, kv_length, key_head_dim = k.shape
        value_batch, value_heads, value_length, value_head_dim = v.shape
        if not (
            batch_size == key_batch == value_batch
            and qk_head_dim == key_head_dim
            and kv_heads == value_heads
            and kv_heads > 0
            and query_heads % kv_heads == 0
            and kv_length == value_length
            and output.shape == (batch_size, query_heads, query_length, value_head_dim)
        ):
            raise ValueError("Attention tensor shapes do not match")

        dimensions = (
            batch_size,
            query_heads,
            query_length,
            kv_length,
            qk_head_dim,
            value_head_dim,
        )
        if any(type(dim) is not int or dim < 1 for dim in dimensions):
            raise ValueError("Attention dimensions must be positive integers")

        flops = batch_size * self._matmul_flops(
            query_heads=query_heads,
            query_length=query_length,
            kv_length=kv_length,
            qk_head_dim=qk_head_dim,
            value_head_dim=value_head_dim,
            mask=self.mask,
        )
        return OpCost(
            useful_flops=flops,
            executed_flops=None,
            logical_read_bytes=q.logical_bytes + k.logical_bytes + v.logical_bytes,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=None,
            rule="attention_matmuls.v1",
            assumptions=(
                f"useful QK^T and PV only; mask={self.mask}",
                "FMA counts as two; GQA uses query heads; QK and V dimensions are separate",
                "softmax, scale, projections, padding and recompute excluded",
                "logical tensor IO does not establish HBM traffic or executed FLOPs",
            ),
            compute_type=_matmul_compute_type(op, q, k, v),
        )

    @staticmethod
    def _matmul_flops(
        *,
        query_heads: int,
        query_length: int,
        kv_length: int,
        qk_head_dim: int,
        value_head_dim: int,
        mask: str,
    ) -> int:
        """Count one request's QK/PV work for explicit tensors or a cache view."""
        triangle = min(query_length, kv_length)
        if mask == "full":
            pairs = query_length * kv_length
        elif mask == "causal_upper_left":
            # Row i sees min(i + 1, kv_length) keys.
            pairs = (
                triangle * (triangle + 1) // 2 + (query_length - triangle) * kv_length
            )
        elif mask == "causal_lower_right":
            # Row i sees max(0, kv_length - query_length + i + 1) keys.
            pairs = triangle * (2 * kv_length - triangle + 1) // 2
        else:
            raise ValueError(f"Unsupported attention mask: {mask}")
        return 2 * query_heads * pairs * (qk_head_dim + value_head_dim)
