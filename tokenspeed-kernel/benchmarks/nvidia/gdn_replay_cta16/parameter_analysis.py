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


"""Audit real JIT variants and the benchmark-to-kernel parameter boundary."""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import adapter
import torch
from tokenspeed_kernel._triton import triton


def audit_jit_reuse(
    kernel: triton.JITFunction, launches: Sequence[Callable[[], Any]]
) -> dict[str, int]:
    """Count real process-local JIT variants across a launch sequence.

    Args:
        kernel: The JIT entrypoint used by every launch.
        launches: Nonempty sequence; the first launch establishes the warm cache.

    Returns:
        Variant counts after warmup and after all launches, and their difference.
    """
    if not launches:
        raise ValueError("at least one warmup launch is required")
    launches[0]()
    torch.cuda.synchronize()
    before = sum(len(cache[0]) for cache in kernel.device_caches.values())
    for launch in launches[1:]:
        launch()
    torch.cuda.synchronize()
    after = sum(len(cache[0]) for cache in kernel.device_caches.values())
    return {
        "warm_variants": before,
        "variant_count": after,
        "new_variants": after - before,
    }


def analyze(request: dict[str, Any]) -> dict[str, Any]:
    """Verify runtime-scalar reuse across the configured serving workload scope.

    This reports the real JIT/compiler evidence and the adapter's argument
    boundary. Numerical correctness remains the separate independent verifier.
    It does not claim to prove arbitrary Python source safe through name matching.

    Args:
        request: Kernel-gen request containing the candidate, parameter contract,
            and workloads to audit.

    Returns:
        Analysis status, parameter violations, and real compiler-cache evidence.
        Unsupported audit scopes report unknown evidence explicitly.
    """
    torch.cuda.set_device(0)
    candidate_path = Path(request["candidate"]["path"])
    module, settings = adapter._candidate(candidate_path)
    kernel = module._gdn_replay_commit_persistent_kernel
    if not isinstance(kernel, triton.JITFunction):
        return {
            "status": "unknown",
            "message": "parameter audit requires a direct Triton/Gluon JIT entrypoint",
            "kernel_analysis": {"variant_count": None},
            "violations": [],
            "warnings": [],
        }
    declarations = request["parameter_contract"]["parameters"]
    params = {parameter.name: parameter for parameter in kernel.params}
    runtime_map = {
        "batch": "B",
        "tokens": "T",
        "layers": "L",
        "accepted_lengths": "accepted_length",
    }
    violations = []
    for name, argument in runtime_map.items():
        if argument not in params:
            violations.append(
                {
                    "kind": "missing_runtime_argument",
                    "message": f"{name} is not supplied through {argument}",
                }
            )
        elif params[argument].is_constexpr:
            violations.append(
                {
                    "kind": "runtime_constexpr",
                    "message": f"workload parameter {name} is constexpr through {argument}",
                }
            )
        elif name != "accepted_lengths" and not params[argument].do_not_specialize:
            violations.append(
                {
                    "kind": "runtime_specialization",
                    "message": f"{argument} still specializes on runtime scalar values/alignment",
                }
            )
    if (
        "PAYLOAD_LAYER_STRIDE" not in params
        or params["PAYLOAD_LAYER_STRIDE"].is_constexpr
        or not params["PAYLOAD_LAYER_STRIDE"].do_not_specialize
    ):
        violations.append(
            {
                "kind": "derived_stride_specialization",
                "message": "payload layer stride must remain an unspecialized runtime scalar",
            }
        )
    harness_names = {
        name for name, spec in declarations.items() if spec["source"] == "harness"
    }
    leaked = harness_names.intersection(params)
    if leaked:
        violations.append(
            {
                "kind": "harness_parameter",
                "message": f"harness metadata appears in kernel arguments: {sorted(leaked)}",
            }
        )
    workloads = request["workloads"]
    if not workloads:
        return {
            "status": "failed",
            "message": "analysis request has no workloads",
            "kernel_analysis": {"variant_count": None},
            "violations": violations,
            "warnings": [],
        }
    models = {tuple(workload["params"]["geometry"]) for workload in workloads}
    if len(models) != 1:
        return {
            "status": "unknown",
            "message": "this fixed-model experiment requires separate audits for distinct model geometries",
            "kernel_analysis": {"variant_count": None},
            "violations": violations,
            "warnings": [],
        }
    geometry = next(iter(models))
    for name, value in zip(("H", "HV", "K", "V"), geometry, strict=True):
        if name not in params:
            violations.append(
                {
                    "kind": "missing_model_argument",
                    "message": f"fixed model dimension {name}={value} is absent",
                }
            )
    # The task's tokens contract permits a bounded token-bucket specialization.
    # All configured windows and the T3/T4 serving audit share bucket4.
    bucket = triton.next_power_of_2(workloads[0]["params"]["tokens"])
    if any(triton.next_power_of_2(w["params"]["tokens"]) != bucket for w in workloads):
        return {
            "status": "unknown",
            "message": "audit each permitted token bucket separately",
            "kernel_analysis": {"variant_count": None},
            "violations": violations,
            "warnings": [],
        }
    audit_params = [dict(w["params"]) for w in workloads]
    audit_params.append(
        {
            **audit_params[0],
            "batch": 3,
            "layers": 2,
            "tokens": 3,
            "accepted_lengths": [0, 1, 2, 3],
        }
    )
    audit_params.append(
        {
            **audit_params[0],
            "batch": 2,
            "layers": 1,
            "tokens": 4,
            "accepted_lengths": [1, 4],
        }
    )

    def launch(values):
        payload, parameters, pool, arguments = adapter._inputs(values)
        adapter._launch(module, settings, payload, parameters, **arguments)
        torch.cuda.synchronize()

    from functools import partial

    evidence = audit_jit_reuse(
        kernel, [partial(launch, values) for values in audit_params]
    )
    if evidence["new_variants"]:
        violations.append(
            {
                "kind": "compile_cache_growth",
                "message": f"runtime B/T/L/stride/acceptance sweep created {evidence['new_variants']} new compiled variants after warmup",
            }
        )
    analysis = {
        "function": "Accepted-prefix GDN recurrence over layer/request/value-head work; pool rows read and committed through GPU metadata.",
        "data_flow": "The adapter supplies payload, gate parameters, GPU pool addresses/strides, read/write rows, accepted-length tensor and runtime B/T/L/payload stride. Fixed model dimensions and explicit kernel settings are compile-time. Harness seed constructs tensor data; timing/order/verification metadata are not kernel arguments.",
        "runtime_parameters": runtime_map,
        "derived_runtime_parameters": ["PAYLOAD_LAYER_STRIDE"],
        "compile_time_parameters": [p.name for p in kernel.params if p.is_constexpr],
        "parameter_sources": declarations,
        "variant_count": evidence["variant_count"],
        "variant_evidence": {
            **evidence,
            "scope": "configured BF16/model/token-bucket workloads plus runtime B3/T3/L2 and B2/T4/L1 audit",
            "audited_launches": len(audit_params),
            "token_bucket": bucket,
        },
        "model_geometry": list(geometry),
        "settings": settings,
        "fixed_evaluator_cta_bound": adapter.MAX_CTAS,
        "source_sha256": hashlib.sha256(
            (candidate_path / "kernel.py").read_bytes()
        ).hexdigest(),
    }
    return {
        "status": "completed",
        "kernel_analysis": analysis,
        "violations": violations,
        "warnings": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(json.loads(args.request.read_text()))
    except Exception as exc:
        result = {
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "kernel_analysis": {"variant_count": None},
            "violations": [],
            "warnings": [],
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
