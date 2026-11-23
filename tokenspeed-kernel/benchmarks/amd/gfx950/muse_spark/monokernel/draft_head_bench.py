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

"""Time the DFlash2 draft head (one launch a draft step) on 8 GPUs.

The launches replay in a CUDA graph with the step bumped inside it; the
LM-head shard (414 MB a rank) streams from HBM each launch. Then one traced
launch prints each stage's first start, last ready, last computed and last
end (us from the launch's first stamp).

    python tokenspeed-kernel/benchmarks/amd/gfx950/muse_spark/monokernel/draft_head_bench.py [--trace]
"""

import argparse
import socket
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, LAUNCHES, REPS = 8, 20, 20


def worker(rank: int, port: int, args) -> None:
    from tokenspeed_kernel.benchmark import muse_spark_monokernel as inputs
    from tokenspeed_kernel.ops.communication._iris import (
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel import trace
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import draft_head

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
    plan = draft_head.head_plan(draft_head.STEPS)
    scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
    group.finish_prepare()
    x, hidden_proj, pred, succ, anchor = inputs.draft_head_shared(
        7, draft_head.STEPS, device
    )
    weights = draft_head.pack_draft_head(
        {
            "lm_head": inputs.draft_head_shard(7, rank, device),
            "hidden_proj": hidden_proj,
            "pred_codebook": pred,
            "succ_codebook": succ,
        },
        rank,
        inputs.HEAD_SCALE,
        inputs.HEAD_SOFTCAP,
    )
    outs = dict(
        anchor=torch.empty(1, dtype=torch.int64, device=device),
        cand_ids=torch.empty(
            draft_head.STEPS, inputs.HEAD_TOPK, dtype=torch.int64, device=device
        ),
        unary=torch.empty(
            draft_head.STEPS, inputs.HEAD_TOPK, dtype=torch.float32, device=device
        ),
        scores=torch.empty(
            draft_head.STEPS,
            inputs.HEAD_TOPK,
            inputs.HEAD_TOPK,
            dtype=torch.float32,
            device=device,
        ),
    )
    first = anchor.to(torch.int32)
    step = torch.full((1,), 1, dtype=torch.int32, device=device)

    def launch(trace_buf=None):
        step.add_(1)
        draft_head.launch_gluon_muse_spark_draft_head_gfx950(
            weights,
            x=x,
            first_token=first,
            **outs,
            scratch=scratch,
            step=step,
            peer_bases=group.bases,
            trace_buf=trace_buf,
            compile_only=False,
        )

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    dist.barrier()
    graph, bump = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(graph):
        for _ in range(LAUNCHES):
            launch()
    with torch.cuda.graph(bump):
        for _ in range(LAUNCHES):
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
        per = [v[0].item() for v in every]
        b = statistics.median(v[1].item() for v in every)
        print(
            f"draft head, 8 GPUs, {draft_head.STEPS} steps: {max(per) - b:.1f} us a launch "
            "(ranks " + " ".join(f"{v:.1f}" for v in per) + f", step bump {b:.1f})",
            flush=True,
        )
    if args.trace:
        buf = torch.zeros(
            (plan.grid, int(trace.EVENTS), 8), dtype=torch.int64, device=device
        )
        dist.barrier()
        torch.cuda.synchronize()
        launch(buf)
        torch.cuda.synchronize()
        tb = buf.cpu()
        if rank == 0:
            t0 = tb[tb > 0].min().item()
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
                    pts.append(
                        (reduce(v).item() - t0) * trace.TICK_NS / 1e3
                        if v.numel()
                        else float("nan")
                    )
                print(
                    f"  {st.name:7s} {pts[0]:7.1f} {pts[1]:7.1f} {pts[2]:8.1f} {pts[3]:7.1f}",
                    flush=True,
                )
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
