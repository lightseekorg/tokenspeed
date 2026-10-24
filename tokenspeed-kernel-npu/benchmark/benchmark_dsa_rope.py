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

"""Paired synthetic benchmark: full CANN indexer-RoPE chain versus selected path."""

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch
import torch_npu
from tokenspeed_kernel_npu.ops.dsa import (
    _composite_interleave_rope,
    interleave_rope,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--tokens",
        "--rows",
        dest="tokens",
        type=int,
        nargs="+",
        default=[1, 8, 64, 65, 128, 256, 512, 1024],
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--layout", choices=("contiguous", "outer-strided"), required=True
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens + args.heads + [args.samples, args.repetitions]) < 1:
        parser.error("sizes and iteration counts must be positive")
    if any(h not in (1, 4, 8, 16, 32, 64) for h in args.heads):
        parser.error("the fused layouts support heads 1, 4, 8, 16, 32 and 64")
    torch.npu.set_device(0)
    torch.manual_seed(421)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    capture_stream = torch.npu.Stream()
    stream_limit = torch.npu.get_stream_limit(torch.npu.current_stream())
    # Explicitly preserve the caller's core limits on the graph capture stream.
    torch.npu.set_stream_limit(
        capture_stream,
        cube_num=int(stream_limit["cube_core_num"]),
        vector_num=int(stream_limit["vector_core_num"]),
    )
    result = {
        "config": vars(args) | {"output": str(args.output)},
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(0),
            "python": platform.python_version(),
            "capture_stream_limit": torch.npu.get_stream_limit(capture_stream),
        },
        "cases": [],
    }
    for rows in args.tokens:
        for heads in args.heads:
            shape = (rows, 128) if heads == 1 else (rows, heads, 128)
            base = torch.randn(
                (*shape[:-1], 256 if args.layout == "outer-strided" else 128),
                device="npu",
                dtype=dtype,
            )
            x = base[..., :128]
            positions = torch.arange(rows, device="npu") * 97 % 8192
            table = torch.randn(8192, 64, device="npu", dtype=dtype)
            functions = {
                "cann": lambda: _composite_interleave_rope(
                    x, positions, table, rope_dim=64
                ),
                "optimized": lambda: interleave_rope(x, positions, table, rope_dim=64),
            }
            reference = functions["cann"]()
            assert torch.equal(functions["optimized"](), reference)
            runners, graphs, retained = {}, {}, {}
            for name, fn in functions.items():
                for _ in range(10):
                    fn()
                torch.npu.synchronize()
                if args.mode == "graph":
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=capture_stream):
                        outputs = [fn() for _ in range(args.repetitions)]
                    # Hold capture allocations for the entire measurement.
                    retained[name] = outputs
                    graphs[name] = graph
                    runners[name] = graph.replay
                else:

                    def replay(fn=fn):
                        return [fn() for _ in range(args.repetitions)]

                    runners[name] = replay
                for _ in range(10):
                    runners[name]()
            torch.npu.synchronize()
            samples = {name: [] for name in functions}
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            for i in range(args.samples):
                order = ("cann", "optimized") if i % 2 == 0 else ("optimized", "cann")
                for name in order:
                    start.record()
                    runners[name]()
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        start.elapsed_time(end) * 1000 / args.repetitions
                    )
            for outputs in retained.values():
                assert all(torch.equal(out, reference) for out in outputs)
            case = {
                "tokens": rows,
                "heads": heads,
                "layout": args.layout,
                "input_stride": list(x.stride()),
                "samples_us": samples,
            }
            case["summary_us"] = {
                name: {
                    "median": statistics.median(times),
                    "mean": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                }
                for name, times in samples.items()
            }
            baseline, optimized = (
                case["summary_us"][n]["median"] for n in ("cann", "optimized")
            )
            case["median_saved_us"] = baseline - optimized
            case["median_reduction_percent"] = (baseline - optimized) / baseline * 100
            result["cases"].append(case)
            print(
                json.dumps({k: v for k, v in case.items() if k != "samples_us"}),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
