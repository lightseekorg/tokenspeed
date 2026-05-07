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

"""MLA Decode Benchmark: TokenSpeed MLA vs FlashMLA vs FlashInfer MLA.

Benchmarks MLA decode latency across multiple backends on Blackwell GPUs.
Produces a unified comparison table (ASCII + CSV).

Usage:
    # Benchmark all available backends with default shapes
    python bench_mla_decode.py

    # Only TokenSpeed MLA
    python bench_mla_decode.py --backends tokenspeed

    # TokenSpeed vs FlashMLA
    python bench_mla_decode.py --backends tokenspeed flash_mla

    # Custom shapes
    python bench_mla_decode.py --batch-sizes 1,4,8 --seq-lens-kv 8192,32768,81920

    # FP8 input
    python bench_mla_decode.py --dtype fp8

    # Export results
    python bench_mla_decode.py --export csv --output mla_decode_results.csv
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
class DecodeShape:
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_heads: int
    qk_nope_head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int


@dataclass
class BenchmarkResult:
    backend: str
    dtype: str
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_heads: int
    qk_nope_head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    median_latency_us: float
    p90_latency_us: float
    p99_latency_us: float
    min_latency_us: float
    max_latency_us: float
    warmup_iters: int
    bench_iters: int
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Default decode shapes
# ---------------------------------------------------------------------------

DEFAULT_DECODE_SHAPES = [
    # (batch_size, seq_len_q, seq_len_kv, num_heads)
    (1, 1, 8192, 16),
    (1, 1, 32768, 16),
    (1, 1, 81920, 16),
    (4, 1, 8192, 16),
    (4, 1, 32768, 16),
    (4, 1, 81920, 16),
    (1, 1, 8192, 32),
    (1, 1, 32768, 32),
    (1, 1, 81920, 32),
    (4, 1, 8192, 32),
    (4, 1, 32768, 32),
    (4, 1, 81920, 32),
    (4, 4, 81920, 16),
    (4, 4, 81920, 32),
    (8, 1, 81920, 16),
    (8, 1, 81920, 32),
    (16, 1, 81920, 16),
    (16, 1, 81920, 32),
]

# DeepSeek V3 MLA config
QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
PAGE_SIZE = 64


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def benchmark_fn(fn, warmup_iters: int = 10, bench_iters: int = 50) -> list[float]:
    """Run fn multiple times and return latencies in microseconds."""
    # Warmup
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


def make_block_tables(
    batch_size: int,
    num_pages_per_seq: int,
    device: torch.device,
) -> torch.Tensor:
    """Create one contiguous physical page range per request."""
    return (
        torch.arange(
            batch_size * num_pages_per_seq,
            dtype=torch.int32,
            device=device,
        )
        .reshape(batch_size, num_pages_per_seq)
        .contiguous()
    )


def randn_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    source_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    return torch.randn(shape, dtype=source_dtype, device=device).to(dtype)


# ---------------------------------------------------------------------------
# Backend: TokenSpeed MLA Decode
# ---------------------------------------------------------------------------


def bench_tokenspeed_mla_decode(
    shape: DecodeShape,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    softmax_scale: float,
) -> BenchmarkResult | None:
    try:
        from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode
    except ImportError:
        print("  [SKIP] tokenspeed_mla not available", flush=True)
        return None

    from tokenspeed_mla.mla_decode import _get_split_kv_and_workspace_size

    B, q_len, H = shape.batch_size, shape.seq_len_q, shape.num_heads
    kv_lora_rank = shape.kv_lora_rank
    qk_rope_head_dim = shape.qk_rope_head_dim
    D_qk = kv_lora_rank + qk_rope_head_dim
    page_size = shape.page_size
    max_seq_len = shape.seq_len_kv

    q_dtype = dtype

    # Allocate inputs
    query = randn_tensor((B, q_len, H, D_qk), q_dtype, device)
    num_pages = (max_seq_len + page_size - 1) // page_size
    num_pages_total = B * num_pages
    kv_cache = randn_tensor((num_pages_total, page_size, D_qk), q_dtype, device)
    block_tables = make_block_tables(B, num_pages, device)
    seq_lens = torch.full((B,), max_seq_len, dtype=torch.int32, device=device)

    # Workspace
    num_sms = get_num_sm(device)
    # fold_sq logic
    mma_m_tile = 128
    fold_sq = H < mma_m_tile and H * q_len <= mma_m_tile
    H_eff = H * q_len if fold_sq else H
    q_len_eff = 1 if fold_sq else q_len
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B, q_len_eff, H_eff, kv_lora_rank, num_sms
    )
    workspace_buffer = torch.zeros(
        max(workspace_size, 1), dtype=torch.int8, device=device
    )

    # Warmup + JIT compile (first call triggers compile)
    def run():
        tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            softmax_scale=softmax_scale,
            enable_pdl=False,
        )

    # Extra warmup for JIT
    print(
        "    tokenspeed_mla: compiling (may take a while on first run)...",
        end="",
        flush=True,
    )
    run()
    torch.cuda.synchronize()
    print(" done", flush=True)

    latencies = benchmark_fn(run, warmup_iters, bench_iters)

    return BenchmarkResult(
        backend="tokenspeed_mla",
        dtype=str(dtype),
        batch_size=B,
        seq_len_q=q_len,
        seq_len_kv=max_seq_len,
        num_heads=H,
        qk_nope_head_dim=shape.qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
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
# Backend: FlashMLA
# ---------------------------------------------------------------------------


def bench_flash_mla_decode(
    shape: DecodeShape,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    softmax_scale: float,
) -> BenchmarkResult | None:
    try:
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    except ImportError:
        print("  [SKIP] flash_mla not available", flush=True)
        return None

    B, q_len, H = shape.batch_size, shape.seq_len_q, shape.num_heads
    kv_lora_rank = shape.kv_lora_rank
    qk_rope_head_dim = shape.qk_rope_head_dim
    D_qk = kv_lora_rank + qk_rope_head_dim
    page_size = shape.page_size
    max_seq_len = shape.seq_len_kv

    # FlashMLA only supports BF16
    if dtype != torch.bfloat16:
        print(f"  [SKIP] flash_mla only supports bf16, got {dtype}", flush=True)
        return None

    # Allocate inputs. FlashMLA expects q as [B, q_len, H, D] and k_cache as
    # [num_pages, page_size, 1, D].
    query = randn_tensor((B, q_len, H, D_qk), dtype, device)
    num_pages = (max_seq_len + page_size - 1) // page_size
    num_pages_total = B * num_pages
    kv_cache = randn_tensor((num_pages_total, page_size, 1, D_qk), dtype, device)
    block_tables = make_block_tables(B, num_pages, device)
    seq_lens = torch.full((B,), max_seq_len, dtype=torch.int32, device=device)

    # FlashMLA metadata
    cache_seqlens = seq_lens
    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, q_len * H, 1)

    def run():
        flash_mla_with_kvcache(
            q=query,
            k_cache=kv_cache,
            block_table=block_tables,
            cache_seqlens=cache_seqlens,
            head_dim_v=kv_lora_rank,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=True,
        )

    print(f"    flash_mla: compiling...", end="", flush=True)
    run()
    torch.cuda.synchronize()
    print(" done", flush=True)

    latencies = benchmark_fn(run, warmup_iters, bench_iters)

    return BenchmarkResult(
        backend="flash_mla",
        dtype=str(dtype),
        batch_size=B,
        seq_len_q=q_len,
        seq_len_kv=max_seq_len,
        num_heads=H,
        qk_nope_head_dim=shape.qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
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
# Backend: FlashInfer MLA
# ---------------------------------------------------------------------------


def bench_flashinfer_mla_decode(
    shape: DecodeShape,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    softmax_scale: float,
) -> BenchmarkResult | None:
    try:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
    except ImportError:
        print("  [SKIP] flashinfer.mla not available", flush=True)
        return None

    B, q_len, H = shape.batch_size, shape.seq_len_q, shape.num_heads
    kv_lora_rank = shape.kv_lora_rank
    qk_rope_head_dim = shape.qk_rope_head_dim
    D_qk = kv_lora_rank + qk_rope_head_dim
    page_size = shape.page_size
    max_seq_len = shape.seq_len_kv

    # FlashInfer MLA on Blackwell — check availability
    if dtype not in (torch.bfloat16, torch.float16):
        print(
            f"  [SKIP] flashinfer.mla only supports bf16/fp16, got {dtype}",
            flush=True,
        )
        return None

    # Allocate inputs
    query = randn_tensor((B, q_len, H, D_qk), dtype, device)
    query_flat = query.reshape(B * q_len, H, D_qk)
    q_nope = query_flat[..., :kv_lora_rank].contiguous()
    q_pe = query_flat[..., kv_lora_rank:].contiguous()
    num_pages = (max_seq_len + page_size - 1) // page_size
    num_pages_total = B * num_pages
    kv_cache = randn_tensor((num_pages_total, page_size, D_qk), dtype, device)
    ckv_cache = kv_cache[:, :, :kv_lora_rank].contiguous()
    kpe_cache = kv_cache[:, :, kv_lora_rank:].contiguous()
    block_tables = make_block_tables(B, num_pages, device)
    kv_indices = block_tables.reshape(-1).contiguous()
    seq_lens = torch.full((B,), max_seq_len, dtype=torch.int32, device=device)
    qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * q_len
    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * num_pages

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="auto")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        seq_lens,
        H,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        True,
        softmax_scale,
        dtype,
        dtype,
    )

    def run():
        wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False)

    print(f"    flashinfer_mla: compiling...", end="", flush=True)
    run()
    torch.cuda.synchronize()
    print(" done", flush=True)

    latencies = benchmark_fn(run, warmup_iters, bench_iters)

    return BenchmarkResult(
        backend="flashinfer_mla",
        dtype=str(dtype),
        batch_size=B,
        seq_len_q=q_len,
        seq_len_kv=max_seq_len,
        num_heads=H,
        qk_nope_head_dim=shape.qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
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
# Backend: TRT-LLM Gen MLA decode (via FlashInfer wrapper)
# ---------------------------------------------------------------------------


def bench_trtllm_mla_decode(
    shape: DecodeShape,
    dtype: torch.dtype,
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    softmax_scale: float,
) -> BenchmarkResult | None:
    try:
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    except ImportError:
        print("  [SKIP] trtllm_mla (flashinfer.mla) not available", flush=True)
        return None

    B, q_len, H = shape.batch_size, shape.seq_len_q, shape.num_heads
    kv_lora_rank = shape.kv_lora_rank
    qk_rope_head_dim = shape.qk_rope_head_dim
    D_qk = kv_lora_rank + qk_rope_head_dim
    page_size = shape.page_size
    max_seq_len = shape.seq_len_kv

    if dtype not in (torch.bfloat16, torch.float16):
        print(f"  [SKIP] trtllm_mla only supports bf16/fp16, got {dtype}", flush=True)
        return None

    # Allocate
    query = randn_tensor((B, q_len, H, D_qk), dtype, device)
    num_pages = (max_seq_len + page_size - 1) // page_size
    num_pages_total = B * num_pages
    kv_cache = randn_tensor((num_pages_total, 1, page_size, D_qk), dtype, device)
    block_tables = make_block_tables(B, num_pages, device)
    seq_lens = torch.full((B,), max_seq_len, dtype=torch.int32, device=device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    def run():
        trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=shape.qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=softmax_scale,
        )

    print(f"    trtllm_mla: compiling...", end="", flush=True)
    run()
    torch.cuda.synchronize()
    print(" done", flush=True)

    latencies = benchmark_fn(run, warmup_iters, bench_iters)

    return BenchmarkResult(
        backend="trtllm_mla",
        dtype=str(dtype),
        batch_size=B,
        seq_len_q=q_len,
        seq_len_kv=max_seq_len,
        num_heads=H,
        qk_nope_head_dim=shape.qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
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
# Table formatting
# ---------------------------------------------------------------------------


def format_results_table(results: list[BenchmarkResult]) -> str:
    if not results:
        return "No results."

    # Group by shape
    from collections import defaultdict

    groups: dict[tuple, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        key = (r.batch_size, r.seq_len_q, r.seq_len_kv, r.num_heads)
        groups[key].append(r)

    lines: list[str] = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    dtype_str = results[0].dtype if results else "N/A"
    cfg = results[0]
    lines.append(f"MLA Decode Benchmark | GPU: {gpu_name} | dtype: {dtype_str}")
    lines.append(
        "qk_nope_head_dim="
        f"{cfg.qk_nope_head_dim}, kv_lora_rank={cfg.kv_lora_rank}, "
        f"qk_rope_head_dim={cfg.qk_rope_head_dim}, page_size={cfg.page_size}"
    )
    lines.append("")

    for key in sorted(groups.keys()):
        B, q_len, kv_len, H = key
        group = sorted(groups[key], key=lambda r: r.median_latency_us)
        lines.append(f"B={B}, q_len={q_len}, kv_len={kv_len}, H={H}")

        header = (
            f"{'Backend':<20} {'p50(us)':>10} {'p90(us)':>10} "
            f"{'p99(us)':>10} {'min(us)':>10} {'max(us)':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for r in group:
            lines.append(
                f"{r.backend:<20} {r.median_latency_us:>10.1f} "
                f"{r.p90_latency_us:>10.1f} {r.p99_latency_us:>10.1f} "
                f"{r.min_latency_us:>10.1f} {r.max_latency_us:>10.1f}"
            )
        lines.append("")

    return "\n".join(lines)


def format_speedup_table(
    results: list[BenchmarkResult],
    baseline: str = "flash_mla",
) -> str:
    """Format a speedup table relative to a baseline backend."""
    if not results:
        return "No results."

    from collections import defaultdict

    groups: dict[tuple, dict[str, BenchmarkResult]] = defaultdict(dict)
    for r in results:
        key = (r.batch_size, r.seq_len_q, r.seq_len_kv, r.num_heads)
        groups[key][r.backend] = r

    lines: list[str] = []
    lines.append(f"MLA Decode Speedup vs {baseline}")
    lines.append("")

    backends = sorted(set(r.backend for r in results) - {baseline})
    header = f"{'Shape':<30}" + "".join(f" {b:>12}" for b in backends)
    lines.append(header)
    lines.append("-" * len(header))

    for key in sorted(groups.keys()):
        B, q_len, kv_len, H = key
        shape_str = f"B={B},q={q_len},kv={kv_len},H={H}"
        baseline_r = groups[key].get(baseline)
        if baseline_r is None:
            continue
        row = f"{shape_str:<30}"
        for b in backends:
            r = groups[key].get(b)
            if r is not None:
                speedup = baseline_r.median_latency_us / r.median_latency_us
                row += f" {speedup:>11.2f}x"
            else:
                row += f" {'N/A':>12}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


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
    "tokenspeed": bench_tokenspeed_mla_decode,
    "flash_mla": bench_flash_mla_decode,
    "flashinfer_mla": bench_flashinfer_mla_decode,
    "trtllm_mla": bench_trtllm_mla_decode,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MLA Decode Benchmark: " "TokenSpeed vs FlashMLA vs FlashInfer vs TRT-LLM"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python bench_mla_decode.py
  python bench_mla_decode.py --backends tokenspeed flash_mla
  python bench_mla_decode.py --dtype fp8
  python bench_mla_decode.py --batch-sizes 1,4,8,16 --seq-lens-kv 8192,81920
  python bench_mla_decode.py --export csv --output results.csv
""",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["tokenspeed", "flash_mla", "flashinfer_mla", "trtllm_mla"],
        choices=list(BACKEND_REGISTRY.keys()),
        help="Backends to benchmark (default: all available)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp8"],
        default="bf16",
        help="Input dtype (default: bf16)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes (default: from standard shapes)",
    )
    parser.add_argument(
        "--seq-lens-kv",
        type=str,
        default=None,
        help="Comma-separated KV sequence lengths (default: from standard shapes)",
    )
    parser.add_argument(
        "--num-heads",
        type=str,
        default=None,
        help="Comma-separated num_heads values (default: 16,32)",
    )
    parser.add_argument(
        "--qk-nope-head-dim",
        type=int,
        default=QK_NOPE_HEAD_DIM,
        help=f"QK non-RoPE head dim (default: {QK_NOPE_HEAD_DIM})",
    )
    parser.add_argument(
        "--kv-lora-rank",
        type=int,
        default=KV_LORA_RANK,
        help=f"KV LoRA rank (default: {KV_LORA_RANK})",
    )
    parser.add_argument(
        "--qk-rope-head-dim",
        type=int,
        default=QK_ROPE_HEAD_DIM,
        help=f"QK RoPE head dim (default: {QK_ROPE_HEAD_DIM})",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=PAGE_SIZE,
        help=f"Page size (default: {PAGE_SIZE})",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--export",
        choices=["csv", "json"],
        default=None,
        help="Export format for results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mla_decode_benchmark",
        help="Output file path (without extension for json)",
    )
    parser.add_argument(
        "--speedup-baseline",
        type=str,
        default="flash_mla",
        help="Baseline backend for speedup comparison (default: flash_mla)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8": torch.float8_e4m3fn,
    }
    dtype = dtype_map[args.dtype]
    softmax_scale = 1.0 / math.sqrt(args.qk_nope_head_dim + args.qk_rope_head_dim)

    # Build shapes
    if args.batch_sizes or args.seq_lens_kv or args.num_heads:
        batch_sizes = (
            [int(x) for x in args.batch_sizes.split(",")]
            if args.batch_sizes
            else [1, 4]
        )
        seq_lens_kv = (
            [int(x) for x in args.seq_lens_kv.split(",")]
            if args.seq_lens_kv
            else [8192, 32768, 81920]
        )
        num_heads_list = (
            [int(x) for x in args.num_heads.split(",")] if args.num_heads else [16, 32]
        )
        shapes = []
        for B in batch_sizes:
            for kv_len in seq_lens_kv:
                for H in num_heads_list:
                    shapes.append(
                        DecodeShape(
                            batch_size=B,
                            seq_len_q=1,
                            seq_len_kv=kv_len,
                            num_heads=H,
                            qk_nope_head_dim=args.qk_nope_head_dim,
                            kv_lora_rank=args.kv_lora_rank,
                            qk_rope_head_dim=args.qk_rope_head_dim,
                            page_size=args.page_size,
                        )
                    )
    else:
        shapes = [
            DecodeShape(
                batch_size=B,
                seq_len_q=q_len,
                seq_len_kv=kv_len,
                num_heads=H,
                qk_nope_head_dim=args.qk_nope_head_dim,
                kv_lora_rank=args.kv_lora_rank,
                qk_rope_head_dim=args.qk_rope_head_dim,
                page_size=args.page_size,
            )
            for B, q_len, kv_len, H in DEFAULT_DECODE_SHAPES
        ]

    print("=" * 80)
    print("MLA Decode Benchmark")
    print("=" * 80)
    print(f"GPU:       {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"dtype:     {args.dtype}")
    print(f"Backends:  {args.backends}")
    print(f"Shapes:    {len(shapes)} configurations")
    print(f"Warmup:    {args.warmup_iters} iters")
    print(f"Benchmark: {args.bench_iters} iters")
    print(f"softmax_scale: {softmax_scale:.6f}")
    print("=" * 80)

    all_results: list[BenchmarkResult] = []

    for shape in shapes:
        print(
            f"\n--- B={shape.batch_size}, q={shape.seq_len_q}, "
            f"kv={shape.seq_len_kv}, H={shape.num_heads} ---"
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
            )
            if result is not None:
                all_results.append(result)
                print(
                    f"    => p50={result.median_latency_us:.1f}us, "
                    f"p90={result.p90_latency_us:.1f}us, "
                    f"p99={result.p99_latency_us:.1f}us"
                )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(format_results_table(all_results))

    # Speedup table (if baseline exists)
    baseline_backends = set(r.backend for r in all_results)
    if args.speedup_baseline in baseline_backends and len(baseline_backends) > 1:
        print("\n" + "=" * 80)
        print(format_speedup_table(all_results, baseline=args.speedup_baseline))

    # Export
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
