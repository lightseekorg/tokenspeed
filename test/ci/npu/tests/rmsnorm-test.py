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

"""Accuracy and latency of the NPU RMSNorm kernel against an FP32 reference.

Covers the three call shapes of ``tokenspeed_kernel_npu.ops.layernorm.rmsnorm``:
the plain normalization, the preallocated ``out`` path, and the fused
add-residual path that also updates the residual in place.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_cli import parse_mode, run_modes

if TYPE_CHECKING:
    import torch

WARMUP = 10
ITERS = 50
SAMPLES = 5
EPS = 1e-6
ATOL = 2e-2
RTOL = 2e-2


def emit(phase, event, **fields):
    record = {
        "schema_version": 1,
        "operator": "rmsnorm",
        "phase": phase,
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def cases():
    planned = []
    for dtype in ("bfloat16", "float32"):
        for rows, hidden in ((1, 1024), (17, 4096), (257, 2048)):
            planned.append(
                {
                    "name": f"plain_r{rows}_h{hidden}_{dtype}",
                    "rows": rows,
                    "hidden": hidden,
                    "dtype": dtype,
                    "residual": False,
                    "prealloc": False,
                }
            )
        planned.append(
            {
                "name": f"prealloc_r33_h1024_{dtype}",
                "rows": 33,
                "hidden": 1024,
                "dtype": dtype,
                "residual": False,
                "prealloc": True,
            }
        )
        planned.append(
            {
                "name": f"add_residual_r17_h1024_{dtype}",
                "rows": 17,
                "hidden": 1024,
                "dtype": dtype,
                "residual": True,
                "prealloc": False,
            }
        )
    return planned


def configs():
    return [
        {"name": "decode_r1_h4096", "rows": 1, "hidden": 4096, "residual": False},
        {"name": "decode_r1_h8192", "rows": 1, "hidden": 8192, "residual": False},
        {"name": "batch_r256_h4096", "rows": 256, "hidden": 4096, "residual": False},
        {
            "name": "prefill_r4096_h4096",
            "rows": 4096,
            "hidden": 4096,
            "residual": False,
        },
        {
            "name": "add_residual_r4096_h4096",
            "rows": 4096,
            "hidden": 4096,
            "residual": True,
        },
        {
            "name": "add_residual_r256_h8192",
            "rows": 256,
            "hidden": 8192,
            "residual": True,
        },
    ]


class RMSNormTests:
    def __init__(self):
        import torch
        import torch_npu  # noqa: F401
        from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

        self.torch = torch
        self.rmsnorm = rmsnorm

    def reference_rmsnorm(self, x, weight, eps):
        """Mirror the kernel contract in FP32, then return the input dtype."""
        torch = self.torch
        x_float = x.float()
        normed = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + eps
        )
        return (normed * weight.float()).to(x.dtype)

    def run_case(self, case) -> bool:
        torch = self.torch
        dtype = getattr(torch, case["dtype"])
        rows, hidden = case["rows"], case["hidden"]
        x = torch.randn(rows, hidden, device="npu", dtype=dtype)
        weight = torch.randn(hidden, device="npu", dtype=dtype)
        residual = torch.randn_like(x) if case["residual"] else None
        expected_input = x if residual is None else x + residual
        expected = self.reference_rmsnorm(expected_input, weight, EPS)

        if case["residual"]:
            residual_before = residual.clone()
            result, residual_out = self.rmsnorm(x, weight, EPS, residual=residual)
            if residual_out is not residual:
                print(
                    f"[rmsnorm] FAIL ({case['name']}): residual was not updated in place"
                )
                return False
            try:
                torch.testing.assert_close(
                    residual, x + residual_before, atol=0, rtol=0
                )
            except AssertionError as exc:
                print(f"[rmsnorm] FAIL ({case['name']}): residual sum mismatch: {exc}")
                return False
        elif case["prealloc"]:
            out = torch.empty_like(x)
            result = self.rmsnorm(x, weight, EPS, out=out)
            if result is not out:
                print(
                    f"[rmsnorm] FAIL ({case['name']}): preallocated out was not reused"
                )
                return False
        else:
            result = self.rmsnorm(x, weight, EPS)

        try:
            torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)
        except AssertionError as exc:
            print(f"[rmsnorm] FAIL ({case['name']}): {exc}")
            return False

        print(
            f"[rmsnorm] {case['name']}: PASS (rows={rows}, hidden={hidden}, "
            f"dtype={case['dtype']}, residual={case['residual']}, "
            f"prealloc={case['prealloc']})"
        )
        return True

    def run_accuracy(self) -> int:
        planned = cases()
        completed = 0
        passed = 0
        current = None
        emit(
            "accuracy",
            "started",
            expected_cases=len(planned),
            case_names=[c["name"] for c in planned],
        )
        try:
            if not self.torch.npu.is_available():
                raise RuntimeError("No NPU device is available")
            self.torch.npu.set_device(0)
            for case in planned:
                current = case["name"]
                emit("accuracy", "case_started", case=case, dtype=case["dtype"])
                ok = self.run_case(case)
                completed += 1
                passed += int(ok)
                emit(
                    "accuracy",
                    "case_completed",
                    case=case,
                    dtype=case["dtype"],
                    status="passed" if ok else "failed",
                )
                if not ok:
                    raise AssertionError(f"Accuracy mismatch in case {current}")
        except Exception as exc:  # pragma: no cover - device dependent
            traceback.print_exc()
            emit(
                "accuracy",
                "failed",
                expected_cases=len(planned),
                completed_cases=completed,
                passed_cases=passed,
                failed_cases=completed - passed,
                current_case=current,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            print(f"[rmsnorm] FAIL: {exc}", flush=True)
            return 1
        emit(
            "accuracy",
            "completed",
            expected_cases=len(planned),
            completed_cases=completed,
            passed_cases=passed,
            failed_cases=completed - passed,
        )
        ok = passed == len(planned)
        print("[rmsnorm] PASS" if ok else "[rmsnorm] FAIL", flush=True)
        return 0 if ok else 1

    def measure(self, op, iters: int, samples: int, config_name: str) -> list[float]:
        for _ in range(WARMUP):
            op()
        self.torch.npu.synchronize()
        per_call_ms = []
        for sample_index in range(samples):
            start = self.torch.npu.Event(enable_timing=True)
            end = self.torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                op()
            end.record()
            self.torch.npu.synchronize()
            value = start.elapsed_time(end) / iters
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid NPU timing: {value}")
            per_call_ms.append(value)
            emit(
                "performance",
                "sample",
                config=config_name,
                sample_index=sample_index,
                latency_ms=value,
            )
        return per_call_ms

    def run_config(self, config) -> None:
        torch = self.torch
        rows, hidden = config["rows"], config["hidden"]
        dtype = torch.bfloat16
        x = torch.randn(rows, hidden, device="npu", dtype=dtype)
        weight = torch.randn(hidden, device="npu", dtype=dtype)
        residual = torch.randn_like(x) if config["residual"] else None
        metadata = {"dtype": "bfloat16", "residual": config["residual"]}

        emit(
            "performance",
            "config_started",
            config=config,
            warmup=WARMUP,
            iterations=ITERS,
            expected_samples=SAMPLES,
            **metadata,
        )

        if residual is None:

            def op() -> None:
                self.rmsnorm(x, weight, EPS)

        else:
            residual_arg = residual

            def op() -> None:
                self.rmsnorm(x, weight, EPS, residual=residual_arg)

        samples = self.measure(op, ITERS, SAMPLES, config["name"])
        median = statistics.median(samples)
        print(
            f"[rmsnorm-perf] {config['name']}: rows={rows} hidden={hidden} "
            f"dtype=bfloat16 residual={config['residual']} iters={ITERS} "
            f"samples={SAMPLES}"
        )
        print("    per-sample ms: " + ", ".join(f"{value:.4f}" for value in samples))
        print(f"    median latency: {median:.4f} ms/call")
        emit(
            "performance",
            "config_completed",
            config=config,
            warmup=WARMUP,
            iterations=ITERS,
            sample_count=len(samples),
            samples_ms=samples,
            median_latency_ms=median,
            threshold_ms=None,
            performance_verdict="not_evaluated",
            **metadata,
        )

    def run_performance(self) -> int:
        planned = configs()
        completed = 0
        current = None
        emit(
            "performance",
            "started",
            expected_configs=len(planned),
            config_names=[c["name"] for c in planned],
        )
        try:
            if not self.torch.npu.is_available():
                raise RuntimeError("No NPU device is available")
            self.torch.npu.set_device(0)
            for config in planned:
                current = config["name"]
                self.run_config(config)
                completed += 1
        except Exception as exc:  # pragma: no cover - device dependent
            traceback.print_exc()
            emit(
                "performance",
                "failed",
                expected_configs=len(planned),
                completed_configs=completed,
                current_config=current,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            print(f"[rmsnorm-perf] FAIL: {exc}", flush=True)
            return 1
        emit(
            "performance",
            "completed",
            expected_configs=len(planned),
            completed_configs=completed,
        )
        print("[rmsnorm-perf] PASS", flush=True)
        return 0


def main(argv=None) -> int:
    mode = parse_mode(argv, __doc__)
    suite = RMSNormTests()
    return run_modes(mode, suite.run_accuracy, suite.run_performance)


if __name__ == "__main__":
    sys.exit(main())
