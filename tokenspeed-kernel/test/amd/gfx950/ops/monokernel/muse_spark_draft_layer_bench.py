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

Five layers' weights rotate, as a draft forward streams them from HBM; the
launches replay in a CUDA graph with the step bumped inside it. Then one
traced launch prints each stage's first start, last ready, last computed and
last end (us from the launch's first stamp).

    python muse_spark_draft_layer_bench.py [--prefix 8000] [--trace]
"""

import argparse
import os
import socket
import statistics
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORLD, LAYERS, LAUNCHES, REPS, ROWS = 8, 5, 20, 20, 8


def worker(rank: int, port: int, args) -> None:
    import test_muse_spark_draft_layer_gfx950 as t
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
    plan = t.plan(ROWS, True)
    scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
    group.finish_prepare()
    layers = [
        t.layer_inputs(args.prefix + i, rank, ROWS, device)[1:] for i in range(LAYERS)
    ]
    step = torch.full((1,), 1, dtype=torch.int32, device=device)

    def launch(i, trace_buf=None):
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

    for i in range(3):
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
        for i in range(LAUNCHES):
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
            f"{max(per) - b:.1f} us a launch (ranks "
            + " ".join(f"{v:.1f}" for v in per)
            + f", step bump {b:.1f})",
            flush=True,
        )
    if args.trace:
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
