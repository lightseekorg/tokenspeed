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

"""Benchmark Qwen gated-residual glue at production shapes.

Example:
    PYTHONPATH=tokenspeed-kernel/python python \
      tokenspeed-kernel/test/ops/bench_hyperconnection.py --mode both
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch
from tokenspeed_kernel import (
    gated_residual_combine,
    gated_residual_mix,
    grouped_gemma_rmsnorm,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320
WIDE = HC_COUNT * HIDDEN_SIZE
DEFAULT_ROWS = (0, 1, 4, 8, 16, 24, 32, 128, 512, 2048, 8192)
BACKENDS = {
    "default": None,
    "triton": "triton_hyperconnection_mix",
    "persistent": "triton_persistent_hyperconnection_mix",
    "cute_dsl": "cute_dsl_hyperconnection_mix",
}


def _iterations(rows: int) -> int:
    if rows <= 32:
        return 200
    if rows <= 512:
        return 50
    if rows <= 2048:
        return 20
    return 10


def _event_time(call: Callable[[], object], iterations: int) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _graph_time(call: Callable[[], object], iterations: int) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _backend_supported(name: str, rows: int, dtype: torch.dtype) -> bool:
    if name == "persistent":
        return (
            current_platform().is_nvidia
            and dtype in (torch.bfloat16, torch.float16)
            and 1 <= rows <= 16
        )
    if name == "cute_dsl":
        return (
            current_platform().is_blackwell
            and dtype is torch.bfloat16
            and 1 <= rows <= 32
            and KernelRegistry.get().get_by_name("cute_dsl_hyperconnection_mix")
            is not None
        )
    return rows > 0


def _case(rows: int, dtype: torch.dtype, backend: str) -> Callable[[], torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(2026 + rows)
    residual = torch.randn(rows, WIDE, dtype=dtype, device="cuda", generator=generator)
    norm_weight = (
        torch.randn(WIDE, dtype=dtype, device="cuda", generator=generator) * 0.02
    )
    projection = (
        torch.randn(
            LOWRANK + HC_COUNT,
            WIDE,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    up = (
        torch.randn(
            WIDE,
            LOWRANK,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    override = BACKENDS[backend]

    def chain() -> torch.Tensor:
        normalized = grouped_gemma_rmsnorm(residual, norm_weight, HIDDEN_SIZE, 1e-6)
        mixed, inject = gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            override=override,
        )
        return gated_residual_combine(mixed, residual, inject, HC_COUNT, HIDDEN_SIZE)

    return chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", default=",".join(map(str, DEFAULT_ROWS)), help="comma-separated T"
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="default")
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/ROCm device is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    rows_values = tuple(int(value) for value in args.rows.split(","))
    results = []
    for rows in rows_values:
        if not _backend_supported(args.backend, rows, dtype):
            results.append(
                {"rows": rows, "backend": args.backend, "status": "unsupported"}
            )
            continue
        chain = _case(rows, dtype, args.backend)
        iterations = _iterations(rows)
        result: dict[str, object] = {
            "rows": rows,
            "dtype": args.dtype,
            "backend": args.backend,
            "iterations": iterations,
        }
        if args.mode in ("eager", "both"):
            result["eager_us"] = round(_event_time(chain, iterations), 3)
        if args.mode in ("graph", "both"):
            result["graph_us"] = round(_graph_time(chain, iterations), 3)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
