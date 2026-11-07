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

"""Accuracy and latency of the NPU paged MHA kernels against an FP32 reference.

Covers the three runtime entry points in ``tokenspeed_kernel_npu.ops.mha``:
causal variable-length prefill, paged extend (cached prefix plus new tokens),
and paged decode with one query per request.
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
ATOL = 2e-2
RTOL = 2e-2


def emit(phase, event, **fields):
    record = {
        "schema_version": 1,
        "operator": "mha",
        "phase": phase,
        "event": event,
        **fields,
    }
    print("NPU_RESULT_JSON=" + json.dumps(record, allow_nan=False), flush=True)


def cases():
    planned = []
    # FIA accepts only the half-precision families, so cover FP16 alongside BF16.
    for dtype in ("bfloat16", "float16"):
        for head_size in (64, 128):
            planned.append(
                {
                    "name": f"prefill_hd{head_size}_{dtype}",
                    "mode": "prefill",
                    "head_size": head_size,
                    "q_heads": 8,
                    "kv_heads": 2,
                    "dtype": dtype,
                    "page_size": 0,
                    "batch": 0,
                    "query_lengths": [3, 5, 9],
                    "cache_lengths": [],
                }
            )
            planned.append(
                {
                    "name": f"extend_hd{head_size}_{dtype}",
                    "mode": "extend",
                    "head_size": head_size,
                    "q_heads": 8,
                    "kv_heads": 2,
                    "dtype": dtype,
                    "page_size": 128,
                    "batch": 3,
                    "query_lengths": [2, 3, 5],
                    "cache_lengths": [5, 70, 257],
                }
            )
            planned.append(
                {
                    "name": f"decode_hd{head_size}_{dtype}",
                    "mode": "decode",
                    "head_size": head_size,
                    "q_heads": 8,
                    "kv_heads": 2,
                    "dtype": dtype,
                    "page_size": 64,
                    "batch": 4,
                    "query_lengths": [1, 1, 1, 1],
                    "cache_lengths": [1, 8, 129, 300],
                }
            )
    return planned


def configs():
    return [
        {
            "name": "decode_b1_kv2048",
            "mode": "decode",
            "batch": 1,
            "q_lengths": 1,
            "cache_length": 2048,
            "page_size": 128,
            "q_heads": 16,
            "kv_heads": 2,
            "head_size": 128,
        },
        {
            "name": "decode_b64_kv2048",
            "mode": "decode",
            "batch": 64,
            "q_lengths": 1,
            "cache_length": 2048,
            "page_size": 128,
            "q_heads": 16,
            "kv_heads": 2,
            "head_size": 128,
        },
        {
            "name": "decode_b128_kv4096",
            "mode": "decode",
            "batch": 128,
            "q_lengths": 1,
            "cache_length": 4096,
            "page_size": 128,
            "q_heads": 16,
            "kv_heads": 2,
            "head_size": 128,
        },
        {
            "name": "prefill_t2048",
            "mode": "prefill",
            "batch": 1,
            "q_lengths": 2048,
            "cache_length": 0,
            "page_size": 0,
            "q_heads": 16,
            "kv_heads": 2,
            "head_size": 128,
        },
        {
            "name": "prefill_t4096_b4",
            "mode": "prefill",
            "batch": 4,
            "q_lengths": 1024,
            "cache_length": 0,
            "page_size": 0,
            "q_heads": 16,
            "kv_heads": 2,
            "head_size": 128,
        },
        {
            "name": "extend_t1024_kv4096_b4",
            "mode": "extend",
            "batch": 4,
            "q_lengths": 256,
            "cache_length": 4096,
            "page_size": 128,
            "q_heads": 16,
            "kv_heads": 2,
            "head_size": 128,
        },
    ]


class MhaTests:
    def __init__(self):
        import torch
        import torch_npu  # noqa: F401
        from tokenspeed_kernel_npu.ops.mha import (
            mha_decode_with_kvcache,
            mha_extend_with_kvcache,
            mha_prefill,
        )

        self.torch = torch
        self.mha_prefill = mha_prefill
        self.mha_extend = mha_extend_with_kvcache
        self.mha_decode = mha_decode_with_kvcache

    def attention(self, q, k, v, *, causal):
        """Reference attention in FP32 over [tokens, heads, head_size]."""
        torch = self.torch
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(
            q.shape[-1]
        )
        if causal:
            q_positions = (
                k.shape[0] - q.shape[0] + torch.arange(q.shape[0], device=q.device)
            )
            k_positions = torch.arange(k.shape[0], device=q.device)
            allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            scores = scores.masked_fill(
                ~allowed.unsqueeze(0), torch.finfo(torch.float32).min
            )
        probabilities = torch.softmax(scores, dim=-1).to(q.dtype)
        return torch.einsum("hqk,khd->qhd", probabilities.float(), v.float()).to(
            q.dtype
        )

    def paged_rows(self, cache, table, batch, length):
        pages = cache[table[batch].long()].reshape(-1, cache.shape[-2], cache.shape[-1])
        return pages[:length]

    def run_prefill_case(self, case, tokens_per_seq):
        torch = self.torch
        dtype = getattr(torch, case["dtype"])
        head_size = case["head_size"]
        lengths = list(tokens_per_seq)
        total = sum(lengths)
        cpu_cumulative = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        cumulative = torch.tensor(cpu_cumulative, dtype=torch.int32, device="npu")
        q = torch.randn(total, case["q_heads"], head_size, dtype=dtype, device="npu")
        k = torch.randn(total, case["kv_heads"], head_size, dtype=dtype, device="npu")
        v = torch.randn_like(k)

        output = self.mha_prefill(q, k, v, cumulative, cpu_cumulative, max(lengths))
        expected, start = [], 0
        for length in lengths:
            expected.append(
                self.attention(
                    q[start : start + length],
                    k[start : start + length],
                    v[start : start + length],
                    causal=True,
                )
            )
            start += length
        return output, torch.cat(expected)

    def run_cached_case(self, case, query_lengths, cache_lengths, page_size):
        torch = self.torch
        dtype = getattr(torch, case["dtype"])
        head_size = case["head_size"]
        batch = len(query_lengths)
        total_q = sum(query_lengths)
        cumulative_q = torch.tensor(
            [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="npu",
        )
        cumulative_kv = torch.tensor(
            [0, *torch.tensor(cache_lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="npu",
        )
        live_lengths = torch.tensor(cache_lengths, dtype=torch.int32, device="npu")
        pages = max(
            1,
            max((length + page_size - 1) // page_size for length in cache_lengths),
        )
        table = torch.arange(batch * pages, dtype=torch.int32, device="npu").reshape(
            batch, pages
        )
        total_pages = int(table.max()) + 1
        q = torch.randn(total_q, case["q_heads"], head_size, dtype=dtype, device="npu")
        k_cache = torch.randn(
            total_pages,
            page_size,
            case["kv_heads"],
            head_size,
            dtype=dtype,
            device="npu",
        )
        v_cache = torch.randn_like(k_cache)

        if case["mode"] == "extend":
            output = self.mha_extend(
                q,
                cumulative_q,
                cumulative_kv,
                k_cache,
                v_cache,
                table,
                live_lengths,
                max(query_lengths),
                page_size,
                is_causal=True,
            )
        else:
            output = self.mha_decode(
                q,
                k_cache,
                v_cache,
                table,
                live_lengths,
                max_seqlen_k=page_size * pages,
            )

        expected, start = [], 0
        for index, query_length in enumerate(query_lengths):
            length = cache_lengths[index]
            expected.append(
                self.attention(
                    q[start : start + query_length],
                    self.paged_rows(k_cache, table, index, length),
                    self.paged_rows(v_cache, table, index, length),
                    causal=case["mode"] == "extend",
                )
            )
            start += query_length
        return output, torch.cat(expected)

    def run_case(self, case) -> bool:
        torch = self.torch
        if case["mode"] == "prefill":
            output, expected = self.run_prefill_case(case, case["query_lengths"])
        else:
            output, expected = self.run_cached_case(
                case,
                case["query_lengths"],
                case["cache_lengths"],
                case["page_size"],
            )
        try:
            torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)
        except AssertionError as exc:
            print(f"[mha] FAIL ({case['name']}): {exc}")
            return False
        print(
            f"[mha] {case['name']}: PASS (mode={case['mode']}, "
            f"head_size={case['head_size']}, q_heads={case['q_heads']}, "
            f"kv_heads={case['kv_heads']}, dtype={case['dtype']})"
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
            print(f"[mha] FAIL: {exc}", flush=True)
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
        print("[mha] PASS" if ok else "[mha] FAIL", flush=True)
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

    def build_perf_op(self, config):
        torch = self.torch
        dtype = torch.bfloat16
        head_size = config["head_size"]
        q_heads, kv_heads = config["q_heads"], config["kv_heads"]
        page_size = config["page_size"]
        batch = config["batch"]
        query_lengths = [config["q_lengths"]] * batch
        total_q = sum(query_lengths)

        if config["mode"] == "prefill":
            cumulative = torch.tensor(
                [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
                dtype=torch.int32,
                device="npu",
            )
            q = torch.randn(total_q, q_heads, head_size, dtype=dtype, device="npu")
            k = torch.randn(total_q, kv_heads, head_size, dtype=dtype, device="npu")
            v = torch.randn_like(k)
            cpu_cumulative = [0, *torch.tensor(query_lengths).cumsum(0).tolist()]
            max_length = max(query_lengths)

            def prefill_op() -> None:
                self.mha_prefill(q, k, v, cumulative, cpu_cumulative, max_length)

            return prefill_op

        cache_lengths = [config["cache_length"]] * batch
        pages = (config["cache_length"] + page_size - 1) // page_size
        table = torch.arange(batch * pages, dtype=torch.int32, device="npu").reshape(
            batch, pages
        )
        live_lengths = torch.tensor(cache_lengths, dtype=torch.int32, device="npu")
        cumulative_q = torch.tensor(
            [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="npu",
        )
        cumulative_kv = torch.tensor(
            [0, *torch.tensor(cache_lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="npu",
        )
        q = torch.randn(total_q, q_heads, head_size, dtype=dtype, device="npu")
        k_cache = torch.randn(
            batch * pages, page_size, kv_heads, head_size, dtype=dtype, device="npu"
        )
        v_cache = torch.randn_like(k_cache)

        if config["mode"] == "extend":
            max_q = max(query_lengths)

            def extend_op() -> None:
                self.mha_extend(
                    q,
                    cumulative_q,
                    cumulative_kv,
                    k_cache,
                    v_cache,
                    table,
                    live_lengths,
                    max_q,
                    page_size,
                    is_causal=True,
                )

            return extend_op

        def decode_op() -> None:
            self.mha_decode(
                q, k_cache, v_cache, table, live_lengths, max_seqlen_k=page_size * pages
            )

        return decode_op

    def run_config(self, config) -> None:
        metadata = {
            "dtype": "bfloat16",
            "mode": config["mode"],
            "page_size": config["page_size"],
        }
        emit(
            "performance",
            "config_started",
            config=config,
            warmup=WARMUP,
            iterations=ITERS,
            expected_samples=SAMPLES,
            **metadata,
        )
        op = self.build_perf_op(config)
        samples = self.measure(op, ITERS, SAMPLES, config["name"])
        median = statistics.median(samples)
        print(
            f"[mha-perf] {config['name']}: mode={config['mode']} "
            f"batch={config['batch']} q_len={config['q_lengths']} "
            f"cache_len={config['cache_length']} page_size={config['page_size']} "
            f"q_heads={config['q_heads']} kv_heads={config['kv_heads']} "
            f"head_size={config['head_size']} iters={ITERS} samples={SAMPLES}"
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
            print(f"[mha-perf] FAIL: {exc}", flush=True)
            return 1
        emit(
            "performance",
            "completed",
            expected_configs=len(planned),
            completed_configs=completed,
        )
        print("[mha-perf] PASS", flush=True)
        return 0


def main(argv=None) -> int:
    mode = parse_mode(argv, __doc__)
    suite = MhaTests()
    return run_modes(mode, suite.run_accuracy, suite.run_performance)


if __name__ == "__main__":
    sys.exit(main())
