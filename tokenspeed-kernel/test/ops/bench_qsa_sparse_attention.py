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

"""Measure the CuTe QSA kernel with the TP4 local head geometry.

Example (run from the repository root):
    python tokenspeed-kernel/test/ops/bench_qsa_sparse_attention.py \
        --cache-dtype bf16 --seq-len 65536 --max-context-len 262144

CUDA events bracket attention graph nodes. The interval includes device-side
dispatch/event gaps; it is not the kernel activity duration. Selection, cache
generation and eviction are outside the interval. ``--source`` accepts an
unchanged source snapshot without importing unrelated optional kernel packages.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch


def make_inputs(
    rows: int,
    seq_len: int,
    max_context_len: int,
    cache_dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate paged KV and 512 selected four-token groups per query row."""
    if not 2048 <= seq_len <= max_context_len:
        raise ValueError("require 2048 <= seq_len <= max_context_len")
    torch.manual_seed(seed)
    page_size = 256
    num_pages = (max_context_len + page_size - 1) // page_size
    cache_slots = (num_pages + 1) * page_size
    query = torch.randn(rows, 6, 256, device="cuda", dtype=torch.bfloat16)
    key = (
        torch.randn(cache_slots, 1, 256, device="cuda", dtype=torch.bfloat16) * 0.25
    ).to(cache_dtype)
    value = (torch.randn_like(key, dtype=torch.bfloat16) * 0.25).to(cache_dtype)
    pages = torch.randperm(num_pages, device="cuda", dtype=torch.int32) + 1
    slots = torch.full((rows, 2051), -1, device="cuda", dtype=torch.int32)
    for row in range(rows):
        groups = torch.randperm(seq_len // 4, device="cuda", dtype=torch.int32)[:512]
        logical = (
            groups[:, None] * 4 + torch.arange(4, device="cuda", dtype=torch.int32)
        ).flatten()
        slots[row, :2048] = (
            pages[logical // page_size] * page_size + logical % page_size
        )
        tail = torch.arange(seq_len // 4 * 4, seq_len, device="cuda", dtype=torch.int32)
        slots[row, 2048 : 2048 + tail.numel()] = (
            pages[tail // page_size] * page_size + tail % page_size
        )
    return query, key, value, slots


def reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Compute attention in FP32 over exactly the positive selected slots."""
    safe = slots.clamp_min(0).long()
    keys = key[safe, 0].float()
    values = value[safe, 0].float()
    scores = torch.einsum("rhd,rsd->rhs", query.float(), keys) * 0.0625
    scores.masked_fill_(slots[:, None, :] <= 0, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1).nan_to_num()
    return torch.einsum("rhs,rsd->rhd", probabilities, values)


def measure(
    call: Callable[[], torch.Tensor],
    cache_state: str,
    iterations: int,
    samples: int,
) -> list[float]:
    """Return event-interval microseconds per call, excluding cold eviction.

    Use a CUDA activity trace for kernel execution duration: especially for
    single-call cold samples, the event interval also includes device gaps.
    """
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    eviction = (
        torch.empty(
            torch.cuda.get_device_properties(0).L2_cache_size * 2,
            device="cuda",
            dtype=torch.uint8,
        )
        if cache_state == "cold"
        else None
    )
    count = iterations if eviction is None else 1
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if eviction is not None:
            eviction.zero_()
        start.record()
        for _ in range(count):
            call()
        end.record()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(samples):
        graph.replay()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1000.0 / count)
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dtype", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--max-context-len", type=int, default=262144)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--cache-state", choices=("warm", "cold"), default="warm")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "python/tokenspeed_kernel/thirdparty/cute_dsl/qsa_sparse.py",
    )
    args = parser.parse_args()
    if min(args.rows, args.iterations, args.samples) < 1:
        parser.error("rows, iterations and samples must be positive")
    spec = importlib.util.spec_from_file_location("qsa_benchmark_source", args.source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dtype = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[args.cache_dtype]
    query, key, value, slots = make_inputs(
        args.rows, args.seq_len, args.max_context_len, dtype, args.seed
    )

    def call() -> torch.Tensor:
        return module.kernel(
            query,
            key,
            value,
            slots,
            scale=0.0625,
            max_seqlen_q=1,
            k_scale=None,
            v_scale=None,
        )

    actual = call()
    expected = reference(query, key, value, slots)
    torch.testing.assert_close(actual.float(), expected, rtol=3.5e-2, atol=3.5e-2)
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        call()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    values = measure(call, args.cache_state, args.iterations, args.samples)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "rows": args.rows,
                "seq_len": args.seq_len,
                "max_context_len": args.max_context_len,
                "cache_dtype": args.cache_dtype,
                "cache_state": args.cache_state,
                "timing_scope": "cuda_graph_event_interval",
                "samples": args.samples,
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
                "max_abs_error": (actual.float() - expected).abs().max().item(),
            }
        )
    )


if __name__ == "__main__":
    main()
