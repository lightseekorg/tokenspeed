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

"""MLA Prefill Benchmark: TokenSpeed MLA vs FlashInfer.

Benchmarks MLA prefill latency across backends on Blackwell GPUs.
Matches the use cases from the TokenSpeed README:

    use case 1: batch_size=1, seqlen_qo=8K,  seqlen_kv=8K
    use case 2: batch_size=1, seqlen_qo=8K,  seqlen_kv=32K
    use case 3: batch_size=1, seqlen_qo=8K,  seqlen_kv=64K
    use case 4: batch_size=4, seqlen_qo=512,  seqlen_kv=80K
    use case 5: batch_size=4, seqlen_qo=1024, seqlen_kv=80K

Usage:
    # Benchmark all available backends
    python bench_mla_prefill.py

    # Only TokenSpeed MLA
    python bench_mla_prefill.py --backends tokenspeed

    # Custom shapes
    python bench_mla_prefill.py \
        --batch-sizes 1,4 --seqlen-qos 8192 --seqlen-kvs 8192,32768,65536

    # FP8 input
    python bench_mla_prefill.py --dtype fp8

    # Export results
    python bench_mla_prefill.py --export csv --output mla_prefill_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PrefillShape:
    batch_size: int
    seqlen_q: int
    seqlen_kv: int
    h_q: int
    h_k: int
    d_qk: int
    d_v: int


@dataclass
class BenchmarkResult:
    backend: str
    dtype: str
    batch_size: int
    seqlen_q: int
    seqlen_kv: int
    h_q: int
    h_k: int
    d_qk: int
    d_v: int
    is_causal: bool
    median_latency_us: float
    p90_latency_us: float
    p99_latency_us: float
    min_latency_us: float
    max_latency_us: float
    warmup_iters: int
    bench_iters: int
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Default shapes (from TokenSpeed README)
# ---------------------------------------------------------------------------

# DeepSeek V3/V4 MLA config:
# h_q=128, h_k=1, d_qk=192 (qk_nope_head_dim=128 + qk_rope_head_dim=64)
# But prefill uses h_q > h_k with GQA-style, so h_q=128, h_k=1 for MLA
# Actually in MLA prefill, the shapes are:
#   Q: [sum(q_lens), h_q, d_qk]
#   K: [sum(kv_lens), h_k, d_qk]
#   V: [sum(kv_lens), h_k, d_v]
# With h_q = num_heads * h_r, h_k = 1 for MLA
# For DeepSeek V3: num_heads=128, kv_lora_rank=512, qk_rope_head_dim=64
# So h_q=128, h_k=1, d_qk=192, d_v=128 is for the compressed MLA representation
# But the TokenSpeed README uses h_q=128, d_qk=192 for the Q/K shapes

DEFAULT_PREFILL_SHAPES = [
    # (batch_size, seqlen_q, seqlen_kv, h_q, h_k)
    (1, 8192, 8192, 128, 1),
    (1, 8192, 32768, 128, 1),
    (1, 8192, 65536, 128, 1),
    (4, 512, 81920, 128, 1),
    (4, 1024, 81920, 128, 1),
]

# MLA dimensions (DeepSeek V3)
H_Q = 128
H_K = 1
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
D_QK = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
D_V = 128


# ---------------------------------------------------------------------------
# Timing utilities (shared with decode benchmark)
# ---------------------------------------------------------------------------


def benchmark_fn(fn, warmup_iters: int = 10, bench_iters: int = 50) -> list[float]:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

    for i in range(bench_iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    latencies = [
        s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)  # ms -> us
    ]
    return sorted(latencies)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    w = rank - float(lo)
    return float(sorted_vals[lo]) * (1.0 - w) + float(sorted_vals[hi]) * w


def randn_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    source_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    return torch.randn(shape, dtype=source_dtype, device=device).to(dtype)


# ---------------------------------------------------------------------------
# Backend: TokenSpeed MLA Prefill
# ---------------------------------------------------------------------------


def bench_tokenspeed_mla_prefill(
    shape: PrefillShape,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    softmax_scale: float,
    is_causal: bool,
) -> BenchmarkResult | None:
    try:
        from tokenspeed_mla import tokenspeed_mla_prefill
    except ImportError:
        print("  [SKIP] tokenspeed_mla not available", flush=True)
        return None

    B = shape.batch_size
    total_q = B * shape.seqlen_q
    total_kv = B * shape.seqlen_kv
    h_q, h_k = shape.h_q, shape.h_k
    d_qk, d_v = shape.d_qk, shape.d_v

    q = randn_tensor((total_q, h_q, d_qk), dtype, device)
    k = randn_tensor((total_kv, h_k, d_qk), dtype, device)
    v = randn_tensor((total_kv, h_k, d_v), dtype, device)
    seq_lens = torch.full((B,), shape.seqlen_kv, dtype=torch.int32, device=device)
    cum_seq_lens = (
        torch.arange(B + 1, dtype=torch.int32, device=device) * shape.seqlen_kv
    )
    cum_seq_lens_q = (
        torch.arange(B + 1, dtype=torch.int32, device=device) * shape.seqlen_q
    )

    # Warmup compile
    print(f"    tokenspeed_mla_prefill: compiling...", end="", flush=True)

    def run():
        return tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens,
            cum_seq_lens=cum_seq_lens,
            max_seq_len=shape.seqlen_kv,
            batch_size=B,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            return_lse=False,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=shape.seqlen_q,
            enable_pdl=False,
        )

    out = run()
    torch.cuda.synchronize()
    del out
    print(" done", flush=True)

    latencies = benchmark_fn(run, warmup_iters, bench_iters)

    return BenchmarkResult(
        backend="tokenspeed_mla",
        dtype=str(dtype),
        batch_size=B,
        seqlen_q=shape.seqlen_q,
        seqlen_kv=shape.seqlen_kv,
        h_q=h_q,
        h_k=h_k,
        d_qk=d_qk,
        d_v=d_v,
        is_causal=is_causal,
        median_latency_us=percentile(latencies, 50),
        p90_latency_us=percentile(latencies, 90),
        p99_latency_us=percentile(latencies, 99),
        min_latency_us=latencies[0],
        max_latency_us=latencies[-1],
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


# ---------------------------------------------------------------------------
# Backend: FlashInfer TRT-LLM ragged attention (DeepSeek)
# ---------------------------------------------------------------------------


def bench_flashinfer_ragged_prefill(
    shape: PrefillShape,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    softmax_scale: float,
    is_causal: bool,
) -> BenchmarkResult | None:
    try:
        from flashinfer.prefill import trtllm_ragged_attention_deepseek
    except ImportError:
        print("  [SKIP] flashinfer.prefill not available", flush=True)
        return None

    if dtype not in (torch.bfloat16, torch.float16):
        print(
            f"  [SKIP] flashinfer prefill only supports bf16/fp16, got {dtype}",
            flush=True,
        )
        return None

    B = shape.batch_size
    total_q = B * shape.seqlen_q
    total_kv = B * shape.seqlen_kv
    h_q, h_k = shape.h_q, shape.h_k
    d_qk, d_v = shape.d_qk, shape.d_v

    q = randn_tensor((total_q, h_q, d_qk), dtype, device)
    k = randn_tensor((total_kv, h_k, d_qk), dtype, device)
    v = randn_tensor((total_kv, h_k, d_v), dtype, device)

    cum_seq_lens_q = (
        torch.arange(B + 1, dtype=torch.int32, device=device) * shape.seqlen_q
    )
    cum_seq_lens_kv = (
        torch.arange(B + 1, dtype=torch.int32, device=device) * shape.seqlen_kv
    )
    seq_lens = torch.full((B,), shape.seqlen_kv, dtype=torch.int32, device=device)

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    def run():
        return trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace,
            seq_lens=seq_lens,
            max_q_len=shape.seqlen_q,
            max_kv_len=shape.seqlen_kv,
            bmm1_scale=softmax_scale,
            bmm2_scale=1.0,
            o_sf_scale=-1.0,
            batch_size=B,
            window_left=-1,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            enable_pdl=False,
            is_causal=is_causal,
            return_lse=False,
        )

    print(f"    flashinfer: compiling...", end="", flush=True)
    run()
    torch.cuda.synchronize()
    print(" done", flush=True)

    latencies = benchmark_fn(run, warmup_iters, bench_iters)

    return BenchmarkResult(
        backend="flashinfer",
        dtype=str(dtype),
        batch_size=B,
        seqlen_q=shape.seqlen_q,
        seqlen_kv=shape.seqlen_kv,
        h_q=h_q,
        h_k=h_k,
        d_qk=d_qk,
        d_v=d_v,
        is_causal=is_causal,
        median_latency_us=percentile(latencies, 50),
        p90_latency_us=percentile(latencies, 90),
        p99_latency_us=percentile(latencies, 99),
        min_latency_us=latencies[0],
        max_latency_us=latencies[-1],
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


# ---------------------------------------------------------------------------
# Table formatting (shared with decode benchmark)
# ---------------------------------------------------------------------------


def format_results_table(results: list[BenchmarkResult]) -> str:
    if not results:
        return "No results."

    from collections import defaultdict

    groups: dict[tuple, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        key = (r.batch_size, r.seqlen_q, r.seqlen_kv, r.h_q)
        groups[key].append(r)

    lines: list[str] = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    lines.append(f"MLA Prefill Benchmark | GPU: {gpu_name} | dtype: {results[0].dtype}")
    lines.append(
        f"h_q={results[0].h_q}, h_k={results[0].h_k}, "
        f"d_qk={results[0].d_qk}, d_v={results[0].d_v}"
    )
    lines.append(f"causal={results[0].is_causal}")
    lines.append("")

    for key in sorted(groups.keys()):
        B, sq, sk, hq = key
        group = sorted(groups[key], key=lambda r: r.median_latency_us)
        lines.append(f"B={B}, seqlen_q={sq}, seqlen_kv={sk}, h_q={hq}")

        header = (
            f"{'Backend':<25} {'p50(us)':>10} {'p90(us)':>10} "
            f"{'p99(us)':>10} {'min(us)':>10} {'max(us)':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for r in group:
            lines.append(
                f"{r.backend:<25} {r.median_latency_us:>10.1f} "
                f"{r.p90_latency_us:>10.1f} {r.p99_latency_us:>10.1f} "
                f"{r.min_latency_us:>10.1f} {r.max_latency_us:>10.1f}"
            )
        lines.append("")

    return "\n".join(lines)


def format_speedup_table(
    results: list[BenchmarkResult],
    baseline: str = "flashinfer",
) -> str:
    if not results:
        return "No results."

    from collections import defaultdict

    groups: dict[tuple, dict[str, BenchmarkResult]] = defaultdict(dict)
    for r in results:
        key = (r.batch_size, r.seqlen_q, r.seqlen_kv, r.h_q)
        groups[key][r.backend] = r

    lines: list[str] = []
    lines.append(f"MLA Prefill Speedup vs {baseline}")
    lines.append("")

    backends = sorted(set(r.backend for r in results) - {baseline})
    header = f"{'Shape':<40}" + "".join(f" {b:>12}" for b in backends)
    lines.append(header)
    lines.append("-" * len(header))

    for key in sorted(groups.keys()):
        B, sq, sk, hq = key
        shape_str = f"B={B},sq={sq},sk={sk},hq={hq}"
        baseline_r = groups[key].get(baseline)
        if baseline_r is None:
            continue
        row = f"{shape_str:<40}"
        for b in backends:
            r = groups[key].get(b)
            if r is not None:
                speedup = baseline_r.median_latency_us / r.median_latency_us
                row += f" {speedup:>11.2f}x"
            else:
                row += f" {'N/A':>12}"
        lines.append(row)

    return "\n".join(lines)


def export_csv(results: list[BenchmarkResult], path: str) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].__dataclass_fields__.keys()),
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)


def export_json(results: list[BenchmarkResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BACKEND_REGISTRY = {
    "tokenspeed": bench_tokenspeed_mla_prefill,
    "flashinfer": bench_flashinfer_ragged_prefill,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLA Prefill Benchmark: TokenSpeed vs FlashInfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["tokenspeed", "flashinfer"],
        choices=list(BACKEND_REGISTRY.keys()),
        help="Backends to benchmark (default: all)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp8"],
        default="bf16",
        help="Input dtype (default: bf16)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--seqlen-qos",
        type=str,
        default=None,
        help="Comma-separated query sequence lengths",
    )
    parser.add_argument(
        "--seqlen-kvs",
        type=str,
        default=None,
        help="Comma-separated KV sequence lengths",
    )
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Disable causal masking (default: causal=True)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Warmup iterations (default: 5, prefill is expensive)",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=20,
        help="Benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "--export",
        choices=["csv", "json"],
        default=None,
        help="Export format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mla_prefill_benchmark",
        help="Output file path",
    )
    parser.add_argument(
        "--speedup-baseline",
        type=str,
        default="flashinfer",
        help="Baseline backend for speedup comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)

    dtype_map = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
    dtype = dtype_map[args.dtype]
    softmax_scale = 1.0 / math.sqrt(D_QK)
    is_causal = not args.no_causal

    # Build shapes
    if args.batch_sizes or args.seqlen_qos or args.seqlen_kvs:
        batch_sizes = (
            [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else [1]
        )
        seqlen_qos = (
            [int(x) for x in args.seqlen_qos.split(",")] if args.seqlen_qos else [8192]
        )
        seqlen_kvs = (
            [int(x) for x in args.seqlen_kvs.split(",")] if args.seqlen_kvs else [8192]
        )
        shapes = [
            PrefillShape(
                batch_size=B,
                seqlen_q=sq,
                seqlen_kv=sk,
                h_q=H_Q,
                h_k=H_K,
                d_qk=D_QK,
                d_v=D_V,
            )
            for B in batch_sizes
            for sq in seqlen_qos
            for sk in seqlen_kvs
        ]
    else:
        shapes = [
            PrefillShape(
                batch_size=B,
                seqlen_q=sq,
                seqlen_kv=sk,
                h_q=hq,
                h_k=hk,
                d_qk=D_QK,
                d_v=D_V,
            )
            for B, sq, sk, hq, hk in DEFAULT_PREFILL_SHAPES
        ]

    print("=" * 80)
    print("MLA Prefill Benchmark")
    print("=" * 80)
    print(f"GPU:          {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"dtype:        {args.dtype}")
    print(f"Backends:     {args.backends}")
    print(f"Causal:       {is_causal}")
    print(f"Shapes:       {len(shapes)} configurations")
    print(f"Warmup:       {args.warmup_iters} iters")
    print(f"Benchmark:    {args.bench_iters} iters")
    print(f"softmax_scale: {softmax_scale:.6f}")
    print("=" * 80)

    all_results: list[BenchmarkResult] = []

    for shape in shapes:
        print(
            f"\n--- B={shape.batch_size}, sq={shape.seqlen_q}, "
            f"sk={shape.seqlen_kv}, h_q={shape.h_q} ---"
        )

        for backend_name in args.backends:
            bench_fn = BACKEND_REGISTRY[backend_name]
            result = bench_fn(
                shape=shape,
                dtype=dtype,
                device=device,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
                softmax_scale=softmax_scale,
                is_causal=is_causal,
            )
            if result is not None:
                all_results.append(result)
                print(
                    f"    => p50={result.median_latency_us:.1f}us, "
                    f"p90={result.p90_latency_us:.1f}us, "
                    f"p99={result.p99_latency_us:.1f}us"
                )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(format_results_table(all_results))

    baseline_backends = set(r.backend for r in all_results)
    if args.speedup_baseline in baseline_backends and len(baseline_backends) > 1:
        print("\n" + "=" * 80)
        print(format_speedup_table(all_results, baseline=args.speedup_baseline))

    if args.export == "csv":
        out_path = args.output if args.output.endswith(".csv") else f"{args.output}.csv"
        export_csv(all_results, out_path)
        print(f"\nResults exported to {out_path}")
    elif args.export == "json":
        out_path = (
            args.output if args.output.endswith(".json") else f"{args.output}.json"
        )
        export_json(all_results, out_path)
        print(f"\nResults exported to {out_path}")


if __name__ == "__main__":
    main()
