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

"""Benchmark TokenSpeed MLA decode V1 (M128) and V2 (SM107 M256) on the k3 cases.

Problem: H=96 q heads, qk_nope 128 / qk_rope 64 / v 128, kv_lora_rank 512
(D_qk = 576), FP8 Q/KV and FP8 output, page size 64, CUDA graph timing with
optional cold L2 (KV rotation). The 16 cases are B in {1, 4, 16, 64} x
Q in {4, 8} x KV in {64k, 128k}.

  V1: tokenspeed_mla/mla_decode_fp8.py      (M128, 2-CTA cluster; SM100/103/107)
  V2: tokenspeed_mla/mla_decode_fp8_sm107.py (M256, 4-CTA cluster; SM107 only)

V2 is reported as N/A on GPUs other than SM107 (e.g. Blackwell SM100/SM103).

Usage (run from the repository root, inside the Python environment that has
torch, nvidia-cutlass-dsl and tvm-ffi installed):

  # All 16 cases, V1 and V2, cold L2 (KV rotation), results in a JSON file.
  PYTHONPATH=tokenspeed-mla/python python tokenspeed-mla/scripts/bench_k3_mla_decode.py \\
      --backends v1 v2 --cache cold --output /tmp/k3_rubin.json

  # Blackwell (SM100/SM103): only V1 can run; V2 is printed as N/A.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tokenspeed-mla/python python \\
      tokenspeed-mla/scripts/bench_k3_mla_decode.py \\
      --backends v1 v2 --cache cold --output /tmp/k3_blackwell.json

  # A subset of cases (indices 0-15 follow the table order: B, then KV, then Q).
  ... --case-index 0 --case-index 15 --cache warm --repeats 5 --output /tmp/x.json

Arguments:
  --backends     One or more of v1, v2.
  --cache        cold: rotate Q/KV so every call misses L2; warm: reuse buffers.
  --output       JSON file with per-case median/percentile latency and error.
  --case-index   Run only the given case index (repeatable); default is all 16.
  --repeats      Timed repeats per case, each of 20 graph replays (default 15).

Output: one table row per case with the median latency in microseconds. Every
result is checked against an FP32 reference on the first, middle and last
request (relative RMSE).
"""

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_sm107_mla_decode import (  # noqa: E402
    Case,
    Settings,
    _measure,
    check_output,
    make_inputs,
    reference_mla,
)
from bench_sm107_pr6177 import prepare_tokenspeed, unsupported_reason  # noqa: E402

BACKENDS = {"v1": "tokenspeed-m128", "v2": "tokenspeed-sm107"}
HEADS = 96
# (batch, kv_len, q_len) of k3-problem.txt.
K3_CASES = [
    Case(batch, kv_k * 1024, HEADS, q_len)
    for batch, kv_k, q_len in (
        (1, 64, 4),
        (1, 64, 8),
        (1, 128, 4),
        (1, 128, 8),
        (4, 64, 4),
        (4, 64, 8),
        (4, 128, 4),
        (4, 128, 8),
        (16, 64, 4),
        (16, 64, 8),
        (16, 128, 4),
        (16, 128, 8),
        (64, 64, 4),
        (64, 64, 8),
        (64, 128, 4),
        (64, 128, 8),
    )
]


def run_case(case, settings, backends):
    import torch

    capability = torch.cuda.get_device_capability()
    query, kv, tables, lengths = make_inputs(case, settings, "cuda")
    indices = sorted({0, case.batch // 2, case.batch - 1})
    expected = reference_mla(query, kv, tables, lengths, indices, 192**-0.5)
    results = {}
    for name in backends:
        backend = BACKENDS[name]
        reason = unsupported_reason(backend, capability)
        if reason is not None:
            results[name] = dict(status="unsupported", reason=reason)
            continue
        workspace = torch.empty(
            settings.workspace_mib * 1024**2, dtype=torch.int8, device=query.device
        )
        out = torch.empty(
            (case.batch, case.q_len, case.heads, 512),
            dtype=query.dtype,
            device=query.device,
        )
        run, details = prepare_tokenspeed(
            backend, case, settings, tables, lengths, workspace, out
        )
        run(query, kv)
        torch.cuda.synchronize()
        errors = check_output(out[indices], expected, "fp8-out")
        timing = _measure(
            run,
            query,
            kv,
            out,
            expected,
            indices,
            case.batch * case.kv_len * 576,
            settings,
        )
        results[name] = dict(status="measured", **details, **errors, **timing)
        del run, workspace, out
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", choices=list(BACKENDS), nargs="+", required=True)
    parser.add_argument("--cache", choices=("cold", "warm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-index", type=int, action="append")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    if args.case_index is not None and any(
        i < 0 or i >= len(K3_CASES) for i in args.case_index
    ):
        parser.error(f"case-index must be between 0 and {len(K3_CASES) - 1}")

    import torch

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7)):
        parser.error("This benchmark requires SM100, SM103 or SM107")
    props = torch.cuda.get_device_properties(0)
    gpu = f"{props.name} (sm_{capability[0]}{capability[1]}, {props.multi_processor_count} SMs)"
    settings = Settings(
        dtype="fp8-out",
        cache=args.cache,
        page_size=64,
        variable_kv=False,
        enable_pdl=False,
        warmup=5,
        repeats=args.repeats,
        graph_iters=20,
        workspace_mib=256,
        seed=42,
    )
    print(f"GPU: {gpu}; cache={args.cache}; median latency in us", flush=True)
    header = "".join(f"{name:>10s}" for name in args.backends)
    print(f"{'B':>3s} {'KV':>5s} {'Q':>2s} |{header}", flush=True)
    records = []
    for index, case in enumerate(K3_CASES):
        if args.case_index is not None and index not in args.case_index:
            continue
        results = run_case(case, settings, args.backends)
        cells = "".join(
            (
                f"{results[n]['median_us']:10.1f}"
                if results[n]["status"] == "measured"
                else f"{'N/A':>10s}"
            )
            for n in args.backends
        )
        print(
            f"{case.batch:3d} {case.kv_len // 1024:4d}k {case.q_len:2d} |{cells}",
            flush=True,
        )
        records.append(dict(case=asdict(case), results=results))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f"{json.dumps(dict(gpu=gpu, settings=asdict(settings), records=records), indent=2)}\n"
        )


if __name__ == "__main__":
    main()
