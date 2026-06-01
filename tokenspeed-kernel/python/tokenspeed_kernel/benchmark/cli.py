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

from __future__ import annotations

import argparse
import json

import torch
from tokenspeed_kernel.benchmark.config import BenchmarkConfig
from tokenspeed_kernel.benchmark.report import format_report
from tokenspeed_kernel.benchmark.result import export_results
from tokenspeed_kernel.benchmark.runner import BenchmarkRunner
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.profiling import ProfilingConfig
from tokenspeed_kernel.registry import load_builtin_kernels

_DTYPE_SELECTIONS: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": Platform.get().fp8e4m3fn.dtype,
}


def _parse_shapes(raw: str | None) -> list[dict] | None:
    if raw is None:
        return None
    obj = json.loads(raw)
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list) and all(isinstance(item, dict) for item in obj):
        return obj
    raise ValueError("--shapes must be a JSON object or list of objects")


def _parse_op(raw: str | None) -> tuple[str, str] | None:
    if raw is None:
        return None
    if "." not in raw:
        raise ValueError("--op must be in family.mode format, e.g. gemm.mm")
    family, mode = raw.split(".", 1)
    return family, mode


def _parse_proton_config(args: argparse.Namespace) -> ProfilingConfig | None:
    has_overrides = any(
        getattr(args, key) is not None
        for key in (
            "proton_output",
            "proton_data",
            "proton_backend",
            "proton_mode",
            "proton_hook",
            "proton_output_format",
        )
    )
    if not has_overrides:
        return None

    hook = args.proton_hook
    if hook == "none":
        hook = None

    return ProfilingConfig(
        output=args.proton_output or "profile",
        data=args.proton_data or "trace",
        backend=args.proton_backend,
        mode=args.proton_mode,
        hook=hook if hook is not None else None,
        output_format=args.proton_output_format or "",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark registered kernels")
    parser.add_argument("kernel_name", nargs="?", help="Benchmark a specific kernel")
    parser.add_argument("--op", help="Benchmark all kernels for family.mode")
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPE_SELECTIONS),
        default="bf16",
        help="Benchmark dtype",
    )
    parser.add_argument(
        "--dtype-role",
        required=True,
        help="Tensor role whose storage dtype is selected by --dtype",
    )
    parser.add_argument(
        "--shapes",
        help="JSON object or list of shape objects override",
    )
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=None,
        help="Run numerics verification alongside benchmarking",
    )
    verify_group.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip numerics verification",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument(
        "--no-cuda-events",
        action="store_true",
        help="Use CPU wall time instead of CUDA events",
    )
    parser.add_argument(
        "--proton",
        action="store_true",
        help="Enable Proton profiling for the benchmark run",
    )
    parser.add_argument(
        "--proton-output",
        help="Proton output path prefix",
    )
    parser.add_argument(
        "--proton-data",
        choices=["tree", "trace"],
        help="Proton data mode",
    )
    parser.add_argument(
        "--proton-backend",
        choices=["cupti", "roctracer"],
        help="Proton activity backend",
    )
    parser.add_argument(
        "--proton-mode",
        choices=["pcsampling", "periodic_flushing"],
        help="Proton profiling mode",
    )
    parser.add_argument(
        "--proton-hook",
        choices=["triton", "none"],
        help="Proton launch hook",
    )
    parser.add_argument(
        "--proton-output-format",
        choices=["hatchet", "hatchet_msgpack", "chrome_trace"],
        help="Proton output format override",
    )
    parser.add_argument("--export", help="Export benchmark results as JSON")
    args = parser.parse_args(argv)

    load_builtin_kernels()
    dtype = _DTYPE_SELECTIONS[args.dtype]
    op_filter = _parse_op(args.op)
    shapes = _parse_shapes(args.shapes)
    proton_config = _parse_proton_config(args)

    config = BenchmarkConfig(
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        verify=True if args.verify is None else args.verify,
        use_cuda_events=not args.no_cuda_events,
        proton_profile=args.proton or proton_config is not None,
        proton_config=proton_config,
    )
    runner = BenchmarkRunner(config)

    if args.kernel_name is not None:
        results = runner.benchmark_kernel(
            args.kernel_name, shapes=shapes, dtype=dtype, dtype_role=args.dtype_role
        )
    elif op_filter is not None:
        assert op_filter is not None
        family, mode = op_filter
        results = runner.benchmark_op(
            family, mode, shapes=shapes, dtype=dtype, dtype_role=args.dtype_role
        )
    else:
        results = runner.benchmark_all(dtype=dtype, dtype_role=args.dtype_role)

    print(format_report(results))

    if args.export is not None:
        export_results(results, args.export)

    return 0
