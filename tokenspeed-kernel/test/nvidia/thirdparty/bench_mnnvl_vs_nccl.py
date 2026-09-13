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

"""Where should the fused MNNVL all-reduce hand over to NCCL?

Sweeps token count and times, per shape:
  * ``nccl``          -- ``dist.all_reduce`` + a separate torch RMSNorm, i.e.
                         what the unfused path actually costs end to end;
  * ``mnnvl_oneshot`` -- fused kernel with one-shot forced for every shape;
  * ``mnnvl_twoshot`` -- fused kernel with two-shot forced for every shape;
  * ``ipc_lamport``   -- the single-node IPC workspace, for reference when run
                         on one node (skipped automatically if unavailable).

Rank 0 prints a table plus the crossover token count, so the dispatch
thresholds can be set from measurement instead of assumption.

Launch (world size 2, 4 or 8; multi-node is the interesting case)::

    srun --ntasks=8 --ntasks-per-node=4 python bench_mnnvl_vs_nccl.py
"""

from __future__ import annotations

import os
import statistics

import torch
import torch.distributed as dist

H, EPS = 7168, 1e-6
MAXTOK = 2048
TOKENS = [
    int(t)
    for t in os.environ.get(
        "BENCH_TOKENS",
        "1,2,4,8,16,32,64,128,129,192,256,384,512,768,1024,1536,2048",
    ).split(",")
]
WARMUP, ITERS = int(os.environ.get("BENCH_WARMUP", 10)), int(
    os.environ.get("BENCH_ITERS", 50)
)


def _spans_nodes() -> bool:
    """True when the world spans hosts. Cross-node CUDA-IPC workspace creation
    fails AND poisons the CUDA context ('invalid resource handle' on the next
    allocation), so it must be skipped outright rather than caught."""
    import socket

    names = [None] * dist.get_world_size()
    dist.all_gather_object(names, socket.gethostname())
    return len(set(names)) > 1


def _setup():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)


def _time_us(fn) -> float:
    """Median per-call microseconds; barriers keep ranks in step between reps."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / ITERS)
        dist.barrier()
    return statistics.median(samples)


def main() -> None:
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        AllReduceFusionPattern,
        trtllm_allreduce_fusion,
        trtllm_create_ipc_workspace_for_all_reduce_fusion,
        trtllm_create_mnnvl_workspace_for_all_reduce_fusion,
    )

    rank, world, dev = _setup()
    p0 = rank == 0

    # One strategy-independent allocation serves both forced variants.
    mnnvl_ws = trtllm_create_mnnvl_workspace_for_all_reduce_fusion(
        rank, world, MAXTOK, H, group=dist.group.WORLD
    )
    ipc_ws = None
    if _spans_nodes():
        if p0:
            print("# ipc lamport workspace skipped: world spans nodes")
    else:
        _, ipc_ws = trtllm_create_ipc_workspace_for_all_reduce_fusion(
            rank, world, MAXTOK, H, group=dist.group.WORLD
        )

    torch.manual_seed(4242)
    weight = torch.randn(H, dtype=torch.bfloat16, device=dev).contiguous()

    if p0:
        print(f"# world={world} hidden={H} dtype=bf16 iters={ITERS} (median of 5)")
        head = (
            f"{'tokens':>7} {'MiB':>8} {'nccl+norm':>11} {'oneshot':>9} {'twoshot':>9}"
        )
        if ipc_ws is not None:
            head += f" {'ipc':>9}"
        head += f"  {'best':>13}"
        print(head)
        print("-" * len(head))

    rows = []
    for ntok in TOKENS:
        x = (torch.randn(ntok, H, dtype=torch.bfloat16, device=dev) * 0.1).contiguous()
        residual = (
            torch.randn(ntok, H, dtype=torch.bfloat16, device=dev) * 0.1
        ).contiguous()
        norm_out, residual_out = torch.empty_like(x), torch.empty_like(residual)
        mib = ntok * H * 2 / 1024 / 1024

        def nccl_path(buf=x, res=residual):
            acc = buf.clone()
            dist.all_reduce(acc)
            acc = acc + res
            var = acc.float().pow(2).mean(-1, keepdim=True)
            return (acc.float() * torch.rsqrt(var + EPS) * weight.float()).to(buf.dtype)

        def fused(ws, oneshot, buf=x, res=residual):
            def run():
                trtllm_allreduce_fusion(
                    allreduce_in=buf,
                    world_size=world,
                    world_rank=rank,
                    token_num=buf.shape[0],
                    hidden_dim=H,
                    workspace_ptrs=ws,
                    launch_with_pdl=False,
                    trigger_completion_at_end=True,
                    fp32_acc=False,
                    use_oneshot=oneshot,
                    pattern_code=AllReduceFusionPattern.kARResidualRMSNorm,
                    residual_in=res,
                    residual_out=residual_out,
                    norm_out=norm_out,
                    rms_gamma=weight,
                    rms_eps=EPS,
                )

            return run

        t_nccl = _time_us(nccl_path)
        t_one = _time_us(fused(mnnvl_ws, True))
        t_two = _time_us(fused(mnnvl_ws, False))
        t_ipc = float("nan")
        if ipc_ws is not None:
            t_ipc = _time_us(fused(ipc_ws, ntok <= 128))

        cands = {"nccl": t_nccl, "oneshot": t_one, "twoshot": t_two}
        if ipc_ws is not None:
            cands["ipc"] = t_ipc
        best = min((v, k) for k, v in cands.items() if v == v)
        rows.append((ntok, t_nccl, t_one, t_two, t_ipc, best[1]))

        if p0:

            def f(v):
                return "     -   " if v != v else f"{v:9.1f}"

            line = f"{ntok:>7} {mib:8.2f} {f(t_nccl):>11} {f(t_one)} {f(t_two)}"
            if ipc_ws is not None:
                line += f" {f(t_ipc)}"
            line += f"  {best[1]:>13}"
            print(line)

    if p0:
        print()
        fused_wins = [r[0] for r in rows if r[5] != "nccl"]
        nccl_wins = [r[0] for r in rows if r[5] == "nccl"]
        print(f"# fused wins at tokens: {fused_wins}")
        print(f"# nccl  wins at tokens: {nccl_wins}")
        if fused_wins and nccl_wins:
            crossover = min(t for t in nccl_wins if t > max(fused_wins, default=0))
            print(f"# RECOMMENDATION: dispatch to NCCL at token_num >= {crossover}")
        elif not nccl_wins:
            print("# RECOMMENDATION: fused path wins across the swept range")
        else:
            print("# RECOMMENDATION: nccl wins across the swept range")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
