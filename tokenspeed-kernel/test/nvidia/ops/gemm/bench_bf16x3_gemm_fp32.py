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

"""CUDA graph benchmark of BF16 rows against an FP32 weight.

Compares ``decode_gemv(x, w, weight_split=...)`` with the FP32 Torch product
``x.float() @ w.T`` at IEEE precision, per call and against FP64. Run with
PYTHONPATH=tokenspeed-kernel/python from the repository root, for example::

    python tokenspeed-kernel/test/nvidia/ops/gemm/bench_bf16x3_gemm_fp32.py \
        --n 256 --k 4096 --rows 17,64,256,2048,8192
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv, decode_gemv_weight_split


def _graph_us(fn, calls: int, repeats: int) -> float:
    """Median microseconds per call over ``repeats`` replays of ``calls``."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            fn()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / calls)
    return statistics.median(times)


def _ulp_row(y, exact):
    """Largest |y - exact| in FP32 units of the row's largest |exact|."""
    row_max = exact.abs().amax(dim=1, keepdim=True).float()
    spacing = torch.nextafter(row_max, torch.full_like(row_max, torch.inf)) - row_max
    return ((y.double() - exact).abs() / spacing.double()).max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--rows", type=str, required=True)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    weight = torch.randn(args.n, args.k, device="cuda") * 0.02
    pieces = decode_gemv_weight_split(weight)
    if pieces is None:
        raise SystemExit("no registered kernel takes a split of this weight here")
    for m in (int(value) for value in args.rows.split(",")):
        x = torch.randn(m, args.k, device="cuda").to(torch.bfloat16)
        out = torch.empty(m, args.n, device="cuda")
        exact = x.double() @ weight.double().t()
        split_us = _graph_us(
            lambda: decode_gemv(x, weight, out, weight_split=pieces),
            args.calls,
            args.repeats,
        )
        split_ulp = _ulp_row(out, exact)
        torch_us = _graph_us(
            lambda: torch.mm(x.float(), weight.t(), out=out), args.calls, args.repeats
        )
        torch_ulp = _ulp_row(out, exact)
        print(
            json.dumps(
                {
                    "m": m,
                    "n": args.n,
                    "k": args.k,
                    "bf16x3_us": round(split_us, 2),
                    "fp32_torch_us": round(torch_us, 2),
                    "bf16x3_max_ulp_row": round(split_ulp, 2),
                    "fp32_torch_max_ulp_row": round(torch_ulp, 2),
                }
            )
        )


if __name__ == "__main__":
    main()
