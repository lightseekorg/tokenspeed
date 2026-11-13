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

"""Time the DFlash2 draft layer (one launch a layer) on 8 GPUs at 8 rows.

Five layers' weights rotate, as a draft forward streams them from HBM, and
the fifth launch is the last layer's (with the draft's final norm). The
launches replay in a CUDA graph with the step bumped once per five launches
inside it, as serving bumps it once per draft forward: the launches then
alternate their peer copy. The time is the mean of the five layers. Then one
traced launch (layer 0) prints each stage's first start, last ready, last
computed and last end (us from the launch's first stamp).

    python tokenspeed-kernel/benchmarks/amd/gfx950/muse_spark/monokernel/draft_layer_bench.py [--prefix 8000] [--trace]
"""

import argparse
import socket
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, LAYERS, LAUNCHES, REPS, ROWS = 8, 5, 20, 20, 8
# Whole draft forwards: the step bumps once per LAYERS launches.
assert LAUNCHES % LAYERS == 0


def worker(rank: int, port: int, args) -> None:
    from tokenspeed_kernel.benchmark import muse_spark_monokernel_gluon as gluon_inputs
    from tokenspeed_kernel.ops.communication._iris import (
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel import trace
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import draft_layer

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=900),
    )
    device = torch.device(f"cuda:{rank}")
    ctx = get_or_create_context(heap_size=256 << 20)
    group = IrisGroup(ctx, dist.group.WORLD, rank, device)
    # The last layer's plan (with the final norm) needs the larger scratch.
    last_plan = draft_layer.layer_plan(ROWS, True)
    scratch = ctx.zeros((last_plan.scratch_pairs,), dtype=torch.int64)
    group.finish_prepare()
    layers = [
        gluon_inputs.draft_layer_inputs(args.prefix + i, rank, ROWS, device)[1:]
        for i in range(LAYERS)
    ]
    # The last layer stores the final norm's whole rows in place of its
    # output partial.
    weights, rt = layers[-1]
    rt = dict(
        rt,
        out=None,
        final_gain=torch.ones(gluon_inputs.OWNED, device=device),
        normed=torch.zeros(
            ROWS, gluon_inputs.HIDDEN, dtype=torch.bfloat16, device=device
        ),
    )
    layers[-1] = (weights, rt)
    step = torch.full((1,), 1, dtype=torch.int32, device=device)

    def launch(i, trace_buf=None):
        # The kernel's peer copy is (step * LAYERS + layer) & 1: one bump a
        # forward makes every launch's copy differ from the launch before.
        if i % LAYERS == 0:
            step.add_(1)
        weights, rt = layers[i % LAYERS]
        draft_layer.launch_gluon_muse_spark_draft_layer_gfx950(
            weights,
            **rt,
            scratch=scratch,
            step=step,
            peer_bases=group.bases,
            layer=i % LAYERS,
            trace_buf=trace_buf,
            compile_only=False,
        )

    # Compile both binaries (the last layer's is another) before the capture.
    for i in range(LAYERS):
        launch(i)
    torch.cuda.synchronize()
    dist.barrier()
    graph, bump = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch(0)
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(graph):
        for i in range(LAUNCHES):
            launch(i)
    with torch.cuda.graph(bump):
        for i in range(LAUNCHES // LAYERS):
            step.add_(1)
    times, bumps = [], []
    for _ in range(REPS):
        for g, out in ((graph, times), (bump, bumps)):
            dist.barrier()
            torch.cuda.synchronize()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            out.append(a.elapsed_time(b) * 1e3 / LAUNCHES)
    mine = torch.tensor([statistics.median(times), statistics.median(bumps)])
    every = [torch.zeros(2) for _ in range(WORLD)]
    dist.all_gather(every, mine)
    if rank == 0:
        per = [x[0].item() for x in every]
        b = statistics.median(x[1].item() for x in every)
        print(
            f"draft layer, 8 GPUs, {ROWS} rows, prefix {args.prefix}: "
            f"{max(per) - b:.1f} us a launch, the mean of {LAYERS} layers (ranks "
            + " ".join(f"{v:.1f}" for v in per)
            + f", step bump {b:.1f})",
            flush=True,
        )
    if args.trace:
        # The traced launch (LAUNCHES % LAYERS == 0) is layer 0's.
        plan = draft_layer.layer_plan(ROWS, False)
        buf = torch.zeros(
            (plan.grid, int(trace.EVENTS), 8), dtype=torch.int64, device=device
        )
        dist.barrier()
        torch.cuda.synchronize()
        launch(LAUNCHES, buf)
        torch.cuda.synchronize()
        tb = buf.cpu()
        if rank == 0:
            t0 = tb[tb > 0].min().item()
            us = lambda v: (v - t0) * trace.TICK_NS / 1e3  # noqa: E731
            print("  stage     start   ready  computed    end  (us)", flush=True)
            for i, st in enumerate(plan.stages):
                pts = []
                for point, reduce in (
                    (0, torch.min),
                    (1, torch.max),
                    (2, torch.max),
                    (3, torch.max),
                ):
                    v = tb[:, 4 * i + point, :]
                    v = v[v > 0]
                    pts.append(us(reduce(v).item()) if v.numel() else float("nan"))
                print(
                    f"  {st.name:7s} {pts[0]:7.1f} {pts[1]:7.1f} {pts[2]:8.1f} {pts[3]:7.1f}",
                    flush=True,
                )
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=int, default=8000)
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
