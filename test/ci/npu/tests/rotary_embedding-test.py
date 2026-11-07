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

"""Accuracy and latency of the NPU RoPE kernel against a real-valued reference.

Runs ``tokenspeed_kernel_npu.ops.rotary_embedding.apply_rope`` for both rotary
layouts the runtime selects between: NeoX (``half``, adjacent dims of each
half rotate) and interleaved (``interleave``, adjacent dim pairs rotate). The
reference is computed in FP64 from the same ``cos_sin_cache``, so a layout swap
or an off-by-one in the frequency table shows up as a mismatch.
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
ROPE_BASE = 10000.0
ATOL = 2e-2
RTOL = 2e-2


def emit(phase, event, **fields):
    record = {
        "schema_version": 1,
        "operator": "rotary_embedding",
        "phase": phase,
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def cases():
    planned = []
    for dtype in ("bfloat16", "float32"):
        for head_size in (64, 128):
            for is_neox in (True, False):
                planned.append(
                    {
                        "name": (f"neox{int(is_neox)}_hd{head_size}_" f"t17_{dtype}"),
                        "tokens": 17,
                        "q_heads": 8,
                        "kv_heads": 2,
                        "head_size": head_size,
                        "is_neox": is_neox,
                        "dtype": dtype,
                        "separate_out": False,
                    }
                )
        planned.append(
            {
                "name": f"separate_out_neox1_hd128_t17_{dtype}",
                "tokens": 17,
                "q_heads": 8,
                "kv_heads": 2,
                "head_size": 128,
                "is_neox": True,
                "dtype": dtype,
                "separate_out": True,
            }
        )
    return planned


def configs():
    return [
        {"name": "decode_t1_hd128", "tokens": 1, "heads": 16, "head_size": 128},
        {"name": "decode_t16_hd128", "tokens": 16, "heads": 16, "head_size": 128},
        {"name": "prefill_t2048_hd128", "tokens": 2048, "heads": 16, "head_size": 128},
        {"name": "prefill_t4096_hd128", "tokens": 4096, "heads": 16, "head_size": 128},
        {"name": "prefill_t4096_hd64", "tokens": 4096, "heads": 32, "head_size": 64},
        {"name": "prefill_t8192_hd64", "tokens": 8192, "heads": 32, "head_size": 64},
    ]


class RotaryEmbeddingTests:
    def __init__(self):
        import torch
        import torch_npu  # noqa: F401
        from tokenspeed_kernel_npu.ops.rotary_embedding import apply_rope

        self.torch = torch
        self.apply_rope = apply_rope

    def cos_sin_cache(self, positions, head_size):
        """Build the frequency table the kernel indexes by position."""
        torch = self.torch
        rows = int(positions.max()) + 1
        inverse = ROPE_BASE ** (
            -torch.arange(0, head_size, 2, device=positions.device, dtype=torch.float32)
            / head_size
        )
        angles = torch.outer(
            torch.arange(rows, device=positions.device, dtype=torch.float32), inverse
        )
        return torch.cat([angles.cos(), angles.sin()], dim=-1)

    def reference(self, x, positions, cache, head_size, is_neox):
        """Rotate with the half-split or interleaved layout in FP32."""
        torch = self.torch
        tokens = x.shape[0]
        heads = x.shape[1] // head_size
        selected = cache[positions.long()]
        cos = selected[:, : head_size // 2].unsqueeze(1)
        sin = selected[:, head_size // 2 :].unsqueeze(1)
        values = x.float().reshape(tokens, heads, head_size)
        if is_neox:
            pairs = values.reshape(tokens, heads, 2, head_size // 2)
            first, second = pairs[..., 0, :], pairs[..., 1, :]
            rotated = torch.stack(
                [first * cos - second * sin, first * sin + second * cos], dim=-2
            )
        else:
            pairs = values.reshape(tokens, heads, head_size // 2, 2)
            first, second = pairs[..., 0], pairs[..., 1]
            rotated = torch.stack(
                [first * cos - second * sin, first * sin + second * cos], dim=-1
            )
        return rotated.reshape(tokens, heads * head_size).to(x.dtype)

    def run_case(self, case) -> bool:
        torch = self.torch
        dtype = getattr(torch, case["dtype"])
        head_size = case["head_size"]
        tokens = case["tokens"]
        positions = torch.randint(0, 4096, (tokens,), dtype=torch.int64, device="npu")
        q = torch.randn(tokens, case["q_heads"] * head_size, device="npu", dtype=dtype)
        k = torch.randn(tokens, case["kv_heads"] * head_size, device="npu", dtype=dtype)
        cache = self.cos_sin_cache(positions, head_size)
        expected_q = self.reference(q, positions, cache, head_size, case["is_neox"])
        expected_k = self.reference(k, positions, cache, head_size, case["is_neox"])

        q_in, k_in = q.clone(), k.clone()
        if case["separate_out"]:
            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            self.apply_rope(
                positions=positions,
                q=q_in,
                k=k_in,
                head_size=head_size,
                cos_sin_cache=cache,
                is_neox=case["is_neox"],
                q_rope_out=q_out,
                k_rope_out=k_out,
            )
            if not torch.equal(q_in, q) or not torch.equal(k_in, k):
                print(
                    f"[rotary_embedding] FAIL ({case['name']}): "
                    "separate outputs also overwrote the inputs"
                )
                return False
            actual_q, actual_k = q_out, k_out
        else:
            # No output buffers: the kernel must rotate Q and K in place.
            self.apply_rope(
                positions=positions,
                q=q_in,
                k=k_in,
                head_size=head_size,
                cos_sin_cache=cache,
                is_neox=case["is_neox"],
            )
            actual_q, actual_k = q_in, k_in

        try:
            torch.testing.assert_close(actual_q, expected_q, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(actual_k, expected_k, atol=ATOL, rtol=RTOL)
        except AssertionError as exc:
            print(f"[rotary_embedding] FAIL ({case['name']}): {exc}")
            return False

        print(
            f"[rotary_embedding] {case['name']}: PASS (tokens={tokens}, "
            f"q_heads={case['q_heads']}, kv_heads={case['kv_heads']}, "
            f"head_size={head_size}, is_neox={case['is_neox']}, dtype={case['dtype']})"
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
            print(f"[rotary_embedding] FAIL: {exc}", flush=True)
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
        print(
            "[rotary_embedding] PASS" if ok else "[rotary_embedding] FAIL", flush=True
        )
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
        tokens = config["tokens"]
        heads = config["heads"]
        head_size = config["head_size"]
        dtype = torch.bfloat16
        metadata = {"dtype": "bfloat16", "is_neox": True}

        emit(
            "performance",
            "config_started",
            config=config,
            warmup=WARMUP,
            iterations=ITERS,
            expected_samples=SAMPLES,
            **metadata,
        )

        positions = torch.arange(tokens, device="npu", dtype=torch.int64) % 4096
        cache = self.cos_sin_cache(positions, head_size)
        q = torch.randn(tokens, heads * head_size, device="npu", dtype=dtype)
        k = torch.randn(tokens, heads * head_size, device="npu", dtype=dtype)
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        def op() -> None:
            self.apply_rope(
                positions=positions,
                q=q,
                k=k,
                head_size=head_size,
                cos_sin_cache=cache,
                is_neox=True,
                q_rope_out=q_out,
                k_rope_out=k_out,
            )

        samples = self.measure(op, ITERS, SAMPLES, config["name"])
        median = statistics.median(samples)
        print(
            f"[rotary_embedding-perf] {config['name']}: tokens={tokens} "
            f"heads={heads} head_size={head_size} dtype=bfloat16 is_neox=True "
            f"iters={ITERS} samples={SAMPLES}"
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
            print(f"[rotary_embedding-perf] FAIL: {exc}", flush=True)
            return 1
        emit(
            "performance",
            "completed",
            expected_configs=len(planned),
            completed_configs=completed,
        )
        print("[rotary_embedding-perf] PASS", flush=True)
        return 0


def main(argv=None) -> int:
    mode = parse_mode(argv, __doc__)
    suite = RotaryEmbeddingTests()
    return run_modes(mode, suite.run_accuracy, suite.run_performance)


if __name__ == "__main__":
    sys.exit(main())
