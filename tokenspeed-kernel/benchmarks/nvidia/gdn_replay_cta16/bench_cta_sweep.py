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

"""Isolated CUDA-graph CTA sweep of the immutable current GDN champion."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenspeed_kernel.platform import max_active_ctas, pdl_enabled
from utils import assert_no_triton_compile

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[3]
OUT: Path | None = None


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def emit(phase, **fields):
    value = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase, **fields}
    save("status.json", value)
    print(json.dumps(value), flush=True)


def launch(module, payload, parameters, arguments, ctas):
    # The official adapter also launches this JIT entry directly. Changing
    # only the grid bypasses the snapshot's Python-only 16-CTA validation;
    # it leaves every byte of the actual GPU implementation unchanged.
    h, hv, k, v = arguments["geometry"]
    return module._gdn_replay_commit_persistent_kernel[(ctas,)](
        payload,
        parameters,
        arguments["state_addresses"],
        arguments["state_row_strides"],
        arguments["read_indices"],
        arguments["write_indices"],
        arguments["accepted_length"],
        arguments["accepted_length"].numel(),
        arguments["draft_token_num"],
        payload.shape[0],
        payload.stride(0),
        H=h,
        HV=hv,
        K=k,
        V=v,
        TOKEN_BLOCK=4,
        GATE_BLOCK=32,
        STATE_DTYPE_CODE=0,
        NUM_BUFFERS=2,
        SPT_K=8,
        ENABLE_PDL=True,
        num_warps=8,
        maxnreg=168,
        launch_pdl=True,
    )


def validate(module, ctas):
    cases = load(
        ROOT / "tokenspeed-kernel/test/nvidia/ops/test_gdn_replay_pipeline.py",
        "cta_sweep_test_cases",
    )
    rows = []
    for batch in (5, 32, 64):
        case = cases._make_case(
            batch=batch,
            tokens=4,
            layers=5,
            state_dtype=torch.bfloat16,
            row_padding=3,
            token_capacity=batch * 4 + 3,
            geometry=(4, 12, 128, 128),
        )
        arguments = dict(
            state_addresses=case.addresses,
            state_row_strides=case.strides,
            read_indices=case.read,
            write_indices=case.write,
            accepted_length=case.accepted,
            draft_token_num=4,
            geometry=case.geometry,
            state_dtype=case.pool.dtype,
        )
        pristine = case.storage.clone()
        for pattern in ("mixed0to4", "full4"):
            if pattern == "full4":
                case.accepted.fill_(4)
            for count in ctas:
                case.storage.copy_(pristine)
                initial = case.storage.clone()
                launch(module, case.payload, case.parameters, arguments, count)
                torch.cuda.synchronize()
                cases._check_result(case, initial)
                rows.append(
                    {
                        "batch": batch,
                        "layers": 5,
                        "ctas": count,
                        "accepted": pattern,
                        "passed": True,
                    }
                )
    save(
        "correctness.json",
        {
            "contract": "Original repository BF16 rtol=2e-2, atol=1e-3; untouched rows exact. Odd row padding, invalid rows, in-place rows, uneven CTA work counts.",
            "checks": rows,
        },
    )
    emit("correctness_passed", checks=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument(
        "--ctas", nargs="+", type=int, choices=(16, 24, 32), required=True
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument(
        "--patterns", nargs="+", choices=("full4", "mixed"), required=True
    )
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global OUT
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if sorted(args.ctas) != [16, 24, 32]:
        raise ValueError("Compare all three explicit budgets: 16, 24, and 32")
    if (OUT / "results.json").exists():
        raise FileExistsError("Preserve the existing benchmark result")
    digest = hashlib.sha256((SOURCE / "kernel.py").read_bytes()).hexdigest()
    torch.cuda.set_device(args.gpu)
    pdl_enabled(True)
    module = load(SOURCE / "kernel.py", "gdn_champion_cta_sweep")
    adapter = load(
        ROOT / "tokenspeed-kernel/benchmarks/nvidia/gdn_replay_cta16/adapter.py",
        "cta_sweep_original_adapter",
    )
    device = torch.cuda.get_device_properties(args.gpu)
    save(
        "manifest.json",
        {
            "arguments": {**vars(args), "output": str(args.output)},
            "candidate": "grouped-replay",
            "source_sha256": digest,
            "geometry": [4, 12, 128, 128],
            "T": 4,
            "dtype": "bfloat16",
            "num_buffers": 2,
            "compute_warps": 8,
            "max_registers": 168,
            "pdl": True,
            "scope": "Single kernel only. Unrestricted SM placement. No model or draft. Same official task data generator and CUDA-graph timing protocol; only CTA grid changes.",
            "device": device.name,
            "sm_count": device.multi_processor_count,
            "versions": {
                p: importlib.metadata.version(p) for p in ("torch", "tokenspeed-triton")
            },
        },
    )
    validate(module, args.ctas)
    records = []
    rng = random.Random(20261103)
    jit_reuse = True
    for batch in args.batches:
        for pattern in args.patterns:
            accepted = [4] if pattern == "full4" else [1, 2, 3, 4]
            payload, parameters, pool, arguments = adapter._inputs(
                dict(
                    seed=args.seed,
                    batch=batch,
                    layers=args.layers,
                    tokens=4,
                    geometry=[4, 12, 128, 128],
                    accepted_lengths=accepted,
                )
            )
            initial_sources = pool[:, : batch + 1].clone()
            launch(module, payload, parameters, arguments, 16)
            torch.cuda.synchronize()
            expected = pool[:, batch + 1 :].clone()
            graphs, resources = {}, {}
            for count in args.ctas:
                pool[:, batch + 1 :].fill_(float("nan"))
                with assert_no_triton_compile(
                    module._gdn_replay_commit_persistent_kernel
                ):
                    binary = launch(module, payload, parameters, arguments, count)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    pool[:, batch + 1 :], expected, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    pool[:, : batch + 1], initial_sources, rtol=0, atol=0
                )
                resources[count] = dict(
                    registers=binary.n_regs,
                    spills=binary.n_spills,
                    shared_bytes=binary.metadata.shared,
                    threads=binary.metadata.num_warps * 32,
                    resident_ctas_per_sm=max_active_ctas(binary),
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    launch(module, payload, parameters, arguments, count)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    launch(module, payload, parameters, arguments, count)
                stream.synchronize()
                graphs[count] = graph
            emit("timing", batch=batch, pattern=pattern, layers=args.layers)
            samples = {count: [] for count in args.ctas}
            for block in range(args.blocks):
                order = list(args.ctas)
                rng.shuffle(order)
                for count in order:
                    for _ in range(4):
                        graphs[count].replay()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        graphs[count].replay()
                    end.record()
                    end.synchronize()
                    samples[count].append(start.elapsed_time(end) / args.iterations)
            for count in args.ctas:
                values = samples[count]
                row = dict(
                    batch=batch,
                    layers=args.layers,
                    tokens=4,
                    accepted=pattern,
                    ctas=count,
                    median_ms=statistics.median(values),
                    min_ms=min(values),
                    max_ms=max(values),
                    samples_ms=values,
                    resources=resources[count],
                )
                records.append(row)
                emit("result", **row)
            del graphs, initial_sources, expected, pool, payload, parameters, arguments
            torch.cuda.empty_cache()
            save(
                "results.json",
                {
                    "scope": "Single kernel; fixed input CUDA-graph replay; original tolerances; no concurrent work",
                    "source_sha256": digest,
                    "jit_reuse_passed": jit_reuse,
                    "results": records,
                },
            )
    lines = [
        "# Isolated GDN commit CTA sweep",
        "",
        f"Grouped replay snapshot, unchanged kernel source. Synthetic geometry H4/HV12/K128/V128, {args.layers} GDN layers, T=4, BF16, two buffers, eight compute warps, 168-register limit, PDL enabled.",
        "",
        f"CUDA-graph timing: {args.blocks} randomized blocks × {args.iterations} replays, four excluded warmups per block. Values are medians. Only launch grid changes; one CTA fits per SM. Full4 means all accepted lengths equal four; mixed means repeated 1/2/3/4. No model, draft, or Nsight tracing.",
        "",
        "| BS | Acceptance | 16 CTA (ms) | 24 CTA (ms) | 32 CTA (ms) | 24 vs 16 reduction | 32 vs 16 reduction |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for batch in args.batches:
        for pattern in args.patterns:
            rows = {
                x["ctas"]: x
                for x in records
                if x["batch"] == batch and x["accepted"] == pattern
            }
            a, b, c = (rows[n]["median_ms"] for n in (16, 24, 32))
            lines.append(
                f"| {batch} | {pattern} | {a:.3f} | {b:.3f} | {c:.3f} | {(1-b/a)*100:.1f}% | {(1-c/a)*100:.1f}% |"
            )
    lines += [
        "",
        "18 independent repository-reference checks passed. Every destination was poisoned before each full-size launch; outputs matched the 16-CTA reference bitwise and source rows were unchanged. Changing CTA count reused the compiled kernel. Per-configuration register, spill, and occupancy measurements are in results.json.",
        "",
        "These are isolated kernel timings and do not establish decode overlap performance.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    emit("complete", report="REPORT.md")


if __name__ == "__main__":
    main()
