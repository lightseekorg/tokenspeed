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

"""Controlled overlap of saved GDN and actual mHC kernels on packed SMs.

This is a fixed-work probe, not a full MTP model. A 120 us device delay stands
in for the other draft operations before its cluster-16 mHC. No arithmetic or
mHC tactic is changed between placement choices. Capture includes the green
stream so kernel-node execution contexts are fixed correctly.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from placement import PLACEMENTS, CommitPlacement
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.residual import gated_residual_mix
from tokenspeed_kernel.platform import pdl_enabled


@gluon.jit
def draft_delay(scratch, nanoseconds):
    result = gl.inline_asm_elementwise(
        "{ .reg .u64 begin, now, elapsed, limit; .reg .pred wait; "
        "cvt.u64.u32 limit, $1; mov.u64 begin, %globaltimer; "
        "delay_loop: mov.u64 now, %globaltimer; sub.u64 elapsed, now, begin; "
        "setp.lt.u64 wait, elapsed, limit; @wait bra delay_loop; mov.u32 $0, 0; }",
        constraints="=r,r",
        args=[nanoseconds],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.store(scratch, result)


@gluon.jit
def record_smids(output):
    sm = gl.inline_asm_elementwise(
        "mov.u32 $0, %smid;",
        constraints="=r",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.store(output + gl.program_id(0), sm)


def make_inputs(batch, layers, acceptance):
    h, hv, k, v, t = 4, 12, 128, 128, 4
    payload = torch.randn(
        layers, batch * t, h * k + hv * v + 2 * hv, device="cuda", dtype=torch.bfloat16
    )
    parameters = torch.randn(layers, 2, hv, device="cuda") * 0.1
    pool = (
        torch.randn(
            layers, 2 * batch + 1, hv, v, k, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    read = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).repeat(
        layers, 1
    )
    accepted = (
        torch.full((batch,), t, device="cuda", dtype=torch.int32)
        if acceptance == "full4"
        else torch.arange(batch, device="cuda", dtype=torch.int32) % t + 1
    )
    args = dict(
        state_addresses=torch.tensor(
            [pool[i].data_ptr() for i in range(layers)],
            device="cuda",
            dtype=torch.uint64,
        ),
        state_row_strides=torch.full(
            (layers,), pool.stride(1), device="cuda", dtype=torch.int64
        ),
        read_indices=read,
        write_indices=read + batch,
        accepted_length=accepted,
        draft_token_num=t,
        geometry=(h, hv, k, v),
        state_dtype=pool.dtype,
    )
    return payload, parameters, pool, args


def hc_inputs(rows, projection):
    return (
        torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16),
        torch.randn(projection, 10240, device="cuda", dtype=torch.bfloat16) * 0.01,
        torch.randn(10240, 320, device="cuda", dtype=torch.bfloat16) * 0.01,
    )


def hc(values):
    return gated_residual_mix(
        *values,
        4,
        2560,
        320,
        projection_rows=values[1].shape[0],
        projection_scale=0.25,
        weights_independent=True,
        override="cute_fused_hyperconnection_mix",
        solution=None,
    )


def make_graph(placement, inputs, values, scratch, *, kind, delay_ns):
    payload, parameters, pool, arguments = inputs
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    main.wait_stream(torch.cuda.current_stream())
    side.wait_stream(torch.cuda.current_stream())
    # HC occupancy is cached per capture stream, not only per tactic.
    if kind in {"joint", "draft"}:
        with torch.cuda.stream(side):
            for item in values:
                hc(item)
            draft_delay[(1,)](scratch, delay_ns, num_warps=4)
        side.synchronize()
    if kind in {"joint", "commit"}:
        with torch.cuda.stream(main):
            placement.launch(payload, parameters, **arguments)
        main.synchronize()
    graph = torch.cuda.CUDAGraph()
    outputs = []
    with torch.cuda.graph(graph, stream=main):
        if kind in {"joint", "draft"}:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                outputs.append(hc(values[0]))
                draft_delay[(1,)](scratch, delay_ns, num_warps=4)
                for item in values[1:]:
                    outputs.append(hc(item))
        if kind in {"joint", "commit"}:
            placement.launch(payload, parameters, **arguments)
        if kind in {"joint", "draft"}:
            main.wait_stream(side)
    torch.cuda.synchronize()
    return graph, outputs


def measure(graphs, *, blocks, iterations):
    names = list(graphs)
    rng = random.Random(1717)
    samples = {name: [] for name in names}
    for _ in range(blocks):
        rng.shuffle(names)
        for name in names:
            graph = graphs[name]
            for _ in range(4):
                graph.replay()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / iterations)
    return samples


def run(args):
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(731)
    pdl_enabled(True)
    placements = {
        name: CommitPlacement(spec=PLACEMENTS[name]) for name in args.placements
    }
    records = []
    profile_work = []
    for batch in args.batches:
        big, small, tail = (
            hc_inputs(batch * 4, 324),
            hc_inputs(batch, 324),
            hc_inputs(batch, 320),
        )
        values = [big, small, tail, small, small, tail, small, small, tail]
        expected_hc = [
            tuple(x.clone() if x is not None else None for x in hc(item))
            for item in values
        ]
        scratch = torch.empty((), device="cuda", dtype=torch.int32)
        for acceptance in args.acceptance:
            inputs = make_inputs(batch, args.layers, acceptance)
            payload, parameters, pool, arguments = inputs
            placements[args.placements[0]].launch(payload, parameters, **arguments)
            torch.cuda.synchronize()
            expected = pool[:, batch + 1 :].clone()
            graphs = {}
            owners = []
            outputs = {}
            placement_records = {}
            for name, placement in placements.items():
                smids = torch.empty(1024, device="cuda", dtype=torch.int32)
                with placement.ordered():
                    record_smids[(1024,)](smids, num_warps=4)
                torch.cuda.synchronize()
                placement_records[name] = {
                    **placement.metadata(),
                    "probe_smids": smids.unique().cpu().tolist(),
                }
                for kind in ["commit", "joint"]:
                    graph, result = make_graph(
                        placement,
                        inputs,
                        values,
                        scratch,
                        kind=kind,
                        delay_ns=args.delay_ns,
                    )
                    graphs[f"{name}/{kind}"] = graph
                    outputs[f"{name}/{kind}"] = result
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        pool[:, batch + 1 :], expected, rtol=0, atol=0
                    )
                    for actual, wanted in zip(result, expected_hc):
                        for a, b in zip(actual, wanted):
                            if a is not None:
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
            draft_graph, draft_outputs = make_graph(
                placements[args.placements[0]],
                inputs,
                values,
                scratch,
                kind="draft",
                delay_ns=args.delay_ns,
            )
            graphs["draft"] = draft_graph
            samples = measure(graphs, blocks=args.blocks, iterations=args.iterations)
            for name, placement in placements.items():
                record = {
                    "batch": batch,
                    "layers": args.layers,
                    "acceptance": acceptance,
                    "placement": name,
                    "placement_metadata": placement_records[name],
                    "commit_us": statistics.median(samples[f"{name}/commit"]),
                    "joint_us": statistics.median(samples[f"{name}/joint"]),
                    "draft_us": statistics.median(samples["draft"]),
                    "samples_us": {
                        kind: samples[f"{name}/{kind}"] for kind in ["commit", "joint"]
                    },
                    "correctness": "bitwise saved-GDN state and mHC output agreement",
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ["samples_us", "placement_metadata"]
                        }
                    ),
                    flush=True,
                )
            profile_work.append(
                (
                    batch,
                    acceptance,
                    graphs,
                    inputs,
                    values,
                    outputs,
                    draft_outputs,
                    scratch,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": __doc__,
                "arguments": vars(args) | {"output": str(args.output)},
                "results": records,
            },
            indent=2,
        )
        + "\n"
    )
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        for batch, acceptance, graphs, *owners in profile_work:
            for name in args.placements:
                with torch.cuda.nvtx.range(
                    f"placement={name} bs={batch} acceptance={acceptance}"
                ):
                    for _ in range(args.profile_iterations):
                        graphs[f"{name}/joint"].replay()
                    torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    # Graphs hold references to green contexts; destroy graph owners first.
    profile_work.clear()
    graphs.clear()
    del graph
    for placement in placements.values():
        placement.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--placements", nargs="+", choices=tuple(PLACEMENTS), required=True
    )
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument(
        "--acceptance", nargs="+", choices=["full4", "mixed"], required=True
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--delay-ns", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--profile-iterations", type=int, required=True)
    run(parser.parse_args())
