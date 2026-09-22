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

"""Paired CUDA-graph HC mix benchmark against saved pre-change sources.

Use --baseline-kind=dispatch with both --baseline-file (hc_fused.py) and
--baseline-dispatch-file (ops/residual/cute_fused.py) to compare static and
dynamic rows through their actual wrappers. The small-kernel baseline retains
the original T<=16 CuTe / larger-T Triton comparison. Captures and timing pairs
alternate A/B order; JSON includes every sample and per-capture medians so
capture-to-capture variation is visible separately from within-capture jitter.
"""

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from unittest import mock

import torch
from cuda.bindings.driver import CUstream
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.ops.residual.cute_fused import cute_fused_hyperconnection_mix
from tokenspeed_kernel.ops.residual.triton import triton_hyperconnection_mix
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.registry import KernelRegistry


def capture(function, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [function() for _ in range(calls)]
    return graph, outputs


def elapsed(graph, calls):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / calls


def baseline_call(kernel_class, x, w, u, enable_pdl):
    rows = x.shape[0]
    active = torch.empty((16, 320), device=x.device, dtype=x.dtype)
    epochs = torch.zeros((w.shape[0] + 63) // 64, device=x.device, dtype=torch.int64)
    compiled = None

    def run():
        nonlocal compiled
        if rows > 16:
            return triton_hyperconnection_mix(x, w, u, 4, 2560, 320, 1.0, True)
        out = torch.empty((rows, 2560), device=x.device, dtype=x.dtype)
        inject = (
            torch.empty((rows, 4), device=x.device, dtype=x.dtype)
            if w.shape[0] == 324
            else None
        )
        values = (
            x.unsqueeze(-1),
            w.unsqueeze(-1),
            u.unsqueeze(-1),
            active,
            epochs,
            out,
            out if inject is None else inject,
        )
        operands = tuple(
            from_dlpack(value, assumed_align=16).mark_layout_dynamic(
                leading_dim=leading
            )
            for value, leading in zip(values, (1, 1, 1, 1, 0, 1, 1))
        )
        stream = CUstream(torch.cuda.current_stream().cuda_stream)
        if compiled is None:
            compiled = cute_ext.compile(
                kernel_class(rows, w.shape[0], 16, enable_pdl, 1.0, True),
                *operands,
                stream,
            )
        compiled(*operands, stream)
        return out, inject

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument(
        "--baseline-kind", choices=["small-kernel", "dispatch"], required=True
    )
    parser.add_argument("--baseline-dispatch-file", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], required=True)
    parser.add_argument(
        "--projection-rows", type=int, choices=[320, 324], required=True
    )
    parser.add_argument("--pdl", choices=["on", "off"], required=True)
    parser.add_argument("--captures", type=int, required=True)
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    if min(*args.rows, args.captures, args.calls, args.repeats) < 1:
        parser.error("rows and measurement counts must be positive")
    spec = importlib.util.spec_from_file_location("_hc_baseline", args.baseline_file)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    baseline_dispatch = None
    if args.baseline_kind == "dispatch":
        if args.baseline_dispatch_file is None:
            parser.error("dispatch baseline requires --baseline-dispatch-file")
        dispatch_spec = importlib.util.spec_from_file_location(
            "_hc_baseline_dispatch", args.baseline_dispatch_file
        )
        baseline_dispatch = importlib.util.module_from_spec(dispatch_spec)
        sys.modules[dispatch_spec.name] = baseline_dispatch
        # Keep the saved wrapper's registration out of the live registry. Its
        # kernel, plan selection and workspace management execute unchanged.
        with mock.patch.object(KernelRegistry, "_instance", KernelRegistry()):
            dispatch_spec.loader.exec_module(baseline_dispatch)
        baseline_dispatch.FusedGatedResidualKernel = original.FusedGatedResidualKernel
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.manual_seed(173)
    enable_pdl = args.pdl == "on"
    pdl_enabled(enable_pdl)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    w = torch.randn((args.projection_rows, 10240), device="cuda", dtype=dtype) * 0.01
    u = torch.randn((10240, 320), device="cuda", dtype=dtype) * 0.01
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
                "dtype": args.dtype,
                "baseline_kind": args.baseline_kind,
                "pdl": enable_pdl,
                "calls": args.calls,
                "captures": args.captures,
                "repeats": args.repeats,
                "boundary": "whole GPU operator, warmed CUDA graph; allocation/JIT excluded",
            }
        ),
        flush=True,
    )
    for rows in args.rows:
        x = torch.randn((rows, 10240), device="cuda", dtype=dtype)

        def after():
            return cute_fused_hyperconnection_mix(x, w, u, 4, 2560, 320, 1.0, True)

        measurements = []
        for capture_id in range(args.captures):
            # Both implementations get fresh workspace for a fresh capture
            # stream. Alternate capture order as well as timing order to
            # expose allocation/graph-placement variation on both sides.
            if baseline_dispatch is None:
                before = baseline_call(
                    original.FusedGatedResidualKernel, x, w, u, enable_pdl
                )
            else:

                def before():
                    return baseline_dispatch.cute_fused_hyperconnection_mix(
                        x, w, u, 4, 2560, 320, 1.0, True
                    )

            if capture_id % 2:
                new_graph, new_outputs = capture(after, args.calls)
                old_graph, old_outputs = capture(before, args.calls)
            else:
                old_graph, old_outputs = capture(before, args.calls)
                new_graph, new_outputs = capture(after, args.calls)
            for _ in range(20):
                old_graph.replay()
                new_graph.replay()
            torch.cuda.synchronize()
            tolerance = (
                0.0
                if baseline_dispatch is not None or rows <= 16
                else (0.04 if dtype == torch.bfloat16 else 0.008)
            )
            torch.testing.assert_close(
                new_outputs[-1], old_outputs[-1], rtol=tolerance, atol=tolerance
            )
            old_times, new_times = [], []
            for repeat in range(args.repeats):
                if (repeat + capture_id) % 2:
                    new_times.append(elapsed(new_graph, args.calls))
                    old_times.append(elapsed(old_graph, args.calls))
                else:
                    old_times.append(elapsed(old_graph, args.calls))
                    new_times.append(elapsed(new_graph, args.calls))
            measurements.append(
                {
                    "baseline_us": statistics.median(old_times),
                    "candidate_us": statistics.median(new_times),
                    "baseline_samples_us": old_times,
                    "candidate_samples_us": new_times,
                }
            )
        ratio = math.exp(
            statistics.mean(
                math.log(item["candidate_us"] / item["baseline_us"])
                for item in measurements
            )
        )
        print(
            json.dumps(
                {
                    "rows": rows,
                    "change_pct": (ratio - 1) * 100,
                    "bitwise_equal": tolerance == 0.0,
                    "capture_medians": measurements,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
