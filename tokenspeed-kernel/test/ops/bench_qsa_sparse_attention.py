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

"""Benchmark direct-slot QSA sparse attention against FlashInfer FA2.

MTP3 contributes three draft rows in addition to the base query, for four
query rows per live request. Cases carry both batch size and query length so
``rows=128`` cannot be confused with either ``batch_size=128`` decode or the
production ``batch_size=32, MTP=3`` shape.

Example:
    PYTHONPATH=tokenspeed-kernel/python python \
      tokenspeed-kernel/test/ops/bench_qsa_sparse_attention.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.attention import qsa_sparse_attention
from tokenspeed_kernel.platform import ArchVersion, current_platform

HEAD_DIM = 256
NUM_Q_HEADS = 6
NUM_KV_HEADS = 1
SELECTED_WIDTH = 2051
DEFAULT_CACHE_SLOTS = 262144
DEFAULT_CASES = (
    (1, 1),
    (1, 4),
    (4, 4),
    (8, 4),
    (16, 4),
    (32, 4),
    (64, 4),
)
BACKENDS = {
    "cute_dsl": "cute_dsl_blackwell_qsa_sparse_attention",
    "flashinfer_fa2": "flashinfer_fa2_qsa_sparse_attention",
}


def _case_name(batch_size: int, max_seqlen_q: int) -> str:
    suffix = "decode" if max_seqlen_q == 1 else f"q{max_seqlen_q}"
    return f"bs{batch_size}_{suffix}"


def _parse_cases(specification: str) -> tuple[tuple[int, int], ...]:
    cases = []
    for item in specification.split(","):
        batch_text, query_text = item.split("x", maxsplit=1)
        batch_size, max_seqlen_q = int(batch_text), int(query_text)
        if batch_size < 1 or max_seqlen_q < 1:
            raise ValueError("batch size and max_seqlen_q must be positive")
        cases.append((batch_size, max_seqlen_q))
    return tuple(cases)


def _make_inputs(
    rows: int,
    cache_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(2026 + rows)
    q = torch.randn(
        rows,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k_cache = (
        torch.randn(
            cache_slots,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    v_cache = (
        torch.randn(
            cache_slots,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    selected = torch.randint(
        1,
        cache_slots,
        (rows, SELECTED_WIDTH),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    selected[:, -1] = -1
    return q, k_cache, v_cache, selected


def _call(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    backend: str,
    max_seqlen_q: int,
) -> Callable[[], torch.Tensor]:
    q, k_cache, v_cache, selected = inputs
    override = BACKENDS[backend]

    def run() -> torch.Tensor:
        return qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            selected,
            scale=HEAD_DIM**-0.5,
            max_seqlen_q=max_seqlen_q,
            k_scale=1.0,
            v_scale=1.0,
            override=override,
        )

    return run


def _eager_time_us(
    call: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples), samples


def _graph_time_us(
    call: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, list[float]]:
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples), samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=",".join(f"{batch}x{query}" for batch, query in DEFAULT_CASES),
        help="comma-separated batch_size x max_seqlen_q pairs, for example 32x4",
    )
    parser.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help=f"comma-separated choices from {','.join(BACKENDS)}",
    )
    parser.add_argument("--cache-slots", type=int, default=DEFAULT_CACHE_SLOTS)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    platform = current_platform()
    if platform.arch_version != ArchVersion(10, 0):
        raise RuntimeError("the CuTe DSL QSA specialization requires NVIDIA SM100")

    cases = _parse_cases(args.cases)
    backends = tuple(value for value in args.backends.split(",") if value)
    unknown = set(backends) - BACKENDS.keys()
    if unknown:
        raise ValueError(f"unknown backends: {sorted(unknown)}")

    results: list[dict[str, object]] = []
    for batch_size, max_seqlen_q in cases:
        rows = batch_size * max_seqlen_q
        inputs = _make_inputs(rows, args.cache_slots)
        outputs: dict[str, torch.Tensor] = {}
        latencies: dict[tuple[str, str], float] = {}
        for backend in backends:
            call = _call(inputs, backend, max_seqlen_q)
            try:
                outputs[backend] = call()
            except (ImportError, RuntimeError) as error:
                result = {
                    "case": _case_name(batch_size, max_seqlen_q),
                    "batch_size": batch_size,
                    "max_seqlen_q": max_seqlen_q,
                    "rows": rows,
                    "backend": backend,
                    "status": "unsupported",
                    "reason": str(error),
                }
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                continue
            result = {
                "case": _case_name(batch_size, max_seqlen_q),
                "batch_size": batch_size,
                "max_seqlen_q": max_seqlen_q,
                "rows": rows,
                "backend": backend,
                "status": "ok",
            }
            if args.mode in ("eager", "both"):
                latency, samples = _eager_time_us(
                    call,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                latencies[(backend, "eager")] = latency
                result["eager_us"] = round(latency, 3)
                result["eager_samples_us"] = [round(sample, 3) for sample in samples]
            if args.mode in ("graph", "both"):
                latency, samples = _graph_time_us(
                    call,
                    args.warmup,
                    args.iterations,
                    args.repeats,
                )
                latencies[(backend, "graph")] = latency
                result["graph_us"] = round(latency, 3)
                result["graph_samples_us"] = [round(sample, 3) for sample in samples]
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        if "cute_dsl" in outputs and "flashinfer_fa2" in outputs:
            torch.testing.assert_close(
                outputs["cute_dsl"].float(),
                outputs["flashinfer_fa2"].float(),
                rtol=3.5e-2,
                atol=3.5e-2,
            )
        for mode in ("eager", "graph"):
            cute_key = ("cute_dsl", mode)
            flashinfer_key = ("flashinfer_fa2", mode)
            if cute_key in latencies and flashinfer_key in latencies:
                speedup = latencies[flashinfer_key] / latencies[cute_key]
                comparison = {
                    "case": _case_name(batch_size, max_seqlen_q),
                    "batch_size": batch_size,
                    "max_seqlen_q": max_seqlen_q,
                    "rows": rows,
                    "mode": mode,
                    "cute_dsl_vs_flashinfer_fa2": round(speedup, 4),
                }
                results.append(comparison)
                print(json.dumps(comparison, sort_keys=True), flush=True)

    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
