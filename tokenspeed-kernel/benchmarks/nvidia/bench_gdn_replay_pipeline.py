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

"""Measure cross-work replay prefetch, alone and beside synthetic BF16 GEMMs.

Inputs and outputs are allocated before capture. State sources are disjoint
from destinations, so every replay starts from the same state. This is a hot
cache experiment; the GEMMs are a concurrency probe, not a real MTP model.
"""

import argparse
import json
import random
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from tokenspeed_kernel.ops.attention.gdn import gdn_replay_commit
from tokenspeed_kernel.ops.attention.gdn.gluon_nvidia import (
    launch_gdn_replay_commit_persistent,
)
from tokenspeed_kernel.platform import pdl_enabled


def _capture(
    commit: Callable[[], Any] | None,
    gemm: Callable[[], Any] | None,
    *,
    order: str,
    main: torch.cuda.Stream,
    side: torch.cuda.Stream,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        if commit is None:
            assert gemm is not None
            gemm()
        elif gemm is None:
            commit()
        elif order == "serial":
            gemm()
            commit()
        else:
            side.wait_stream(main)
            if order == "draft_first":
                gemm()
            with torch.cuda.stream(side):
                commit()
            if order == "commit_first":
                gemm()
            main.wait_stream(side)
    return graph


def _measure(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    iterations: int,
    blocks: int,
) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {name: [] for name in graphs}
    names = list(graphs)
    rng = random.Random(481)
    for _ in range(blocks):
        rng.shuffle(names)
        for name in names:
            graph = graphs[name]
            for _ in range(5):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / iterations)
    return samples


def _run_batch(args: argparse.Namespace, batch: int) -> list[dict[str, Any]]:
    torch.manual_seed(731)
    layers, tokens = args.layers, args.tokens
    h, hv, k, v = args.key_heads, args.value_heads, 128, 128
    state_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.state_dtype
    ]
    width = h * k + hv * v + 2 * hv
    payload = torch.randn(
        layers, batch * tokens, width, device="cuda", dtype=torch.bfloat16
    )
    parameters = torch.randn(layers, 2, hv, device="cuda") * 0.1
    pool = (
        torch.randn(layers, 2 * batch + 1, hv, v, k, device="cuda", dtype=state_dtype)
        * 0.02
    )
    read = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).repeat(
        layers, 1
    )
    write = read + batch
    accepted = torch.full((batch,), tokens, device="cuda", dtype=torch.int32)
    kwargs = dict(
        state_addresses=torch.tensor(
            [pool[i].data_ptr() for i in range(layers)],
            device="cuda",
            dtype=torch.uint64,
        ),
        state_row_strides=torch.full(
            (layers,), pool.stride(1), device="cuda", dtype=torch.int64
        ),
        read_indices=read,
        write_indices=write,
        accepted_length=accepted,
        draft_token_num=tokens,
        geometry=(h, hv, k, v),
        state_dtype=state_dtype,
    )
    original = partial(
        gdn_replay_commit, payload, parameters, solution="triton", **kwargs
    )
    original()
    expected = pool[:, batch + 1 :].clone()
    weight = torch.randn(
        args.gemm_size, args.gemm_size, device="cuda", dtype=torch.bfloat16
    )
    x = torch.randn(batch * tokens, args.gemm_size, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)

    def gemm(repeats: int) -> None:
        for _ in range(repeats):
            torch.mm(x, weight, out=y)

    commits: dict[str, Callable[[], Any]] = {"original": original}
    resources: dict[str, dict[str, int]] = {}
    configurations: dict[str, dict[str, int]] = {}
    for warps in args.num_warps:
        for budget in args.cta_budgets:
            for buffers in (1, 2):
                name = f"w{warps}_cta{budget}_buf{buffers}"
                commits[name] = partial(
                    launch_gdn_replay_commit_persistent,
                    payload,
                    parameters,
                    max_ctas=budget,
                    num_buffers=buffers,
                    num_warps=warps,
                    max_registers=args.max_registers,
                    **kwargs,
                )
                pool[:, batch + 1 :].fill_(float("nan"))
                kernel = commits[name]()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    pool[:, batch + 1 :], expected, rtol=0, atol=0
                )
                resources[name] = dict(
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,
                )
                configurations[name] = dict(
                    num_warps=warps, max_ctas=budget, num_buffers=buffers
                )
    print(f"B={batch}: exact state agreement for all configurations", flush=True)
    for _ in range(4):
        gemm(max(args.gemm_repeats))
    torch.cuda.synchronize()
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    main.wait_stream(torch.cuda.current_stream())
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, commit in commits.items():
        graphs[f"commit/{name}"] = _capture(
            commit, None, order="serial", main=main, side=side
        )
    for repeats in args.gemm_repeats:
        draft = partial(gemm, repeats)
        graphs[f"gemm/{repeats}"] = _capture(
            None, draft, order="serial", main=main, side=side
        )
        graphs[f"serial/{repeats}"] = _capture(
            original, draft, order="serial", main=main, side=side
        )
        for name, commit in commits.items():
            for order in args.orders:
                graphs[f"overlap/{repeats}/{name}/{order}"] = _capture(
                    commit, draft, order=order, main=main, side=side
                )
    samples = _measure(
        graphs, iterations=args.iterations, blocks=args.measurement_blocks
    )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    rows = []
    for repeats in args.gemm_repeats:
        serial = medians[f"serial/{repeats}"]
        for name in commits:
            for order in args.orders:
                key = f"overlap/{repeats}/{name}/{order}"
                elapsed = medians[key]
                row = dict(
                    batch=batch,
                    layers=layers,
                    tokens=tokens,
                    state_dtype=args.state_dtype,
                    name=name,
                    order=order,
                    gemm_repeats=repeats,
                    commit_us=medians[f"commit/{name}"],
                    gemm_us=medians[f"gemm/{repeats}"],
                    original_commit_us=medians["commit/original"],
                    original_serial_us=serial,
                    combined_us=elapsed,
                    improvement_vs_serial_pct=(serial - elapsed) / serial * 100,
                    commit_samples_us=samples[f"commit/{name}"],
                    combined_samples_us=samples[key],
                    **configurations.get(name, {}),
                    **resources.get(name, {}),
                )
                rows.append(row)
                print(
                    f"B={batch} gemms={repeats} {name} {order}: "
                    f"commit={row['commit_us']:.2f}us combined={elapsed:.2f}us "
                    f"serial={serial:.2f}us gain={row['improvement_vs_serial_pct']:.2f}%",
                    flush=True,
                )
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for name, graph in graphs.items():
            torch.cuda.nvtx.range_push(f"B{batch}/{name}")
            graph.replay()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--cta-budgets", type=int, nargs="+", required=True)
    parser.add_argument("--gemm-repeats", type=int, nargs="+", required=True)
    parser.add_argument("--num-warps", type=int, nargs="+", required=True)
    parser.add_argument("--max-registers", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--key-heads", type=int, required=True)
    parser.add_argument("--value-heads", type=int, required=True)
    parser.add_argument("--gemm-size", type=int, required=True)
    parser.add_argument("--state-dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument(
        "--orders", nargs="+", choices=("commit_first", "draft_first"), required=True
    )
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--measurement-blocks", type=int, required=True)
    parser.add_argument(
        "--profile", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.batches
            + args.cta_budgets
            + args.gemm_repeats
            + [args.iterations, args.measurement_blocks]
        )
        <= 0
    ):
        parser.error(
            "batches, CTA budgets, GEMM repeats, iterations, and blocks must be positive"
        )
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_properties(0)
    report: dict[str, Any] = dict(
        arch=f"sm{device.major}{device.minor}",
        sm_count=device.multi_processor_count,
        torch_version=torch.__version__,
        pdl=pdl_enabled(),
        geometry=[args.key_heads, args.value_heads, 128, 128],
        gemm_size=args.gemm_size,
        num_warps=args.num_warps,
        max_registers=args.max_registers,
        iterations=args.iterations,
        measurement_blocks=args.measurement_blocks,
        cache_mode="hot",
        rows=[],
    )
    for batch in args.batches:
        report["rows"].extend(_run_batch(args, batch))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
