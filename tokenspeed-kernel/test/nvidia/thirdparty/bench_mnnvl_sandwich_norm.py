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

"""What does fusing a pre/post ("sandwich") norm boundary into the MNNVL
all-reduce save?

Per token count, a CUDA graph of ``BENCH_OPS`` boundaries is timed for:
  * ``chain`` -- the plain MNNVL all-reduce, ``rmsnorm`` of the sum and
                 ``rmsnorm`` with the residual and the scale pair;
  * ``fused`` -- ``allreduce_sandwich_rmsnorm``, one launch per boundary.
Rank 0 prints microseconds per boundary and the saving.

Launch (world size 2, 4, 8 or 16)::

    torchrun --nproc-per-node=4 bench_mnnvl_sandwich_norm.py
"""

from __future__ import annotations

import os
import statistics

import torch
import torch.distributed as dist

HIDDEN = int(os.environ.get("BENCH_HIDDEN", 4096))
EPS = 1e-5
PAIR = (0.25, 0.97)
TOKENS = [
    int(t)
    for t in os.environ.get("BENCH_TOKENS", "1,8,32,128,512,1024,2048").split(",")
]
OPS = int(os.environ.get("BENCH_OPS", 20))
REPS = int(os.environ.get("BENCH_REPS", 30))


def _boundaries(tokens, dev, fused):
    from tokenspeed_kernel.ops.communication import allreduce_sandwich_rmsnorm, trtllm
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm
    from tokenspeed_kernel.thirdparty.cuda.trtllm import (
        AllReduceFusionPattern,
        trtllm_allreduce_fusion,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    mnnvl = trtllm._manager_for_group(dist.group.WORLD).mnnvl_workspace
    x = torch.randn(tokens, HIDDEN, device=dev).to(torch.bfloat16)
    residual = torch.randn(tokens, HIDDEN, device=dev).to(torch.bfloat16)
    post = torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
    weight = torch.ones(HIDDEN, device=dev, dtype=torch.bfloat16)
    s = torch.empty_like(x)

    def one():
        if fused:
            out = allreduce_sandwich_rmsnorm(
                x,
                residual,
                post,
                weight,
                rank,
                dist.group.WORLD,
                eps=EPS,
                x_scale=PAIR[0],
                residual_scale=PAIR[1],
            )
            assert out is not None
            return
        trtllm_allreduce_fusion(
            allreduce_in=x,
            world_size=world,
            world_rank=rank,
            token_num=tokens,
            hidden_dim=HIDDEN,
            workspace_ptrs=mnnvl,
            trigger_completion_at_end=True,
            fp32_acc=False,
            pattern_code=AllReduceFusionPattern.kAllReduce,
            use_oneshot=mnnvl.resolve_use_oneshot(tokens, None, HIDDEN),
            allreduce_out=s,
        )
        a = rmsnorm(s, post, EPS, enable_pdl=None)
        rmsnorm(
            a,
            weight,
            EPS,
            residual=residual,
            enable_pdl=None,
            round_residual_sum_bf16=True,
            x_scale=PAIR[0],
            residual_scale=PAIR[1],
        )

    return one


def _time(one) -> float:
    for _ in range(3):
        one()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(OPS):
            one()
    samples = []
    for _ in range(REPS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        dist.barrier()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / OPS)
    return statistics.median(samples)


def main() -> None:
    from tokenspeed_kernel.ops.communication import trtllm

    lrank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lrank)
    dist.init_process_group("nccl")
    dev = torch.device("cuda", lrank)
    rank = dist.get_rank()
    armed = trtllm.ensure_workspace_initialized(
        rank=rank, group=dist.group.WORLD, max_token_num=max(TOKENS), hidden_dim=HIDDEN
    )
    if not armed or trtllm._manager_for_group(dist.group.WORLD).mnnvl_workspace is None:
        raise SystemExit("no MNNVL workspace on this fabric")
    rows = []
    for tokens in TOKENS:
        chain = _time(_boundaries(tokens, dev, fused=False))
        fused = _time(_boundaries(tokens, dev, fused=True))
        rows.append((tokens, chain, fused))
    if rank == 0:
        print(
            f"world {dist.get_world_size()} hidden {HIDDEN}, us per boundary ({OPS} per graph)"
        )
        print(f"{'tokens':>7} {'chain':>9} {'fused':>9} {'saved':>9}")
        for tokens, chain, fused in rows:
            print(f"{tokens:>7} {chain:>9.2f} {fused:>9.2f} {chain - fused:>9.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
