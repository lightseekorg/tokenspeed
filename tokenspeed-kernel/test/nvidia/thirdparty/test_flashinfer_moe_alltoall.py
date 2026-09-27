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

"""Distributed round-trip checks for the FlashInfer MoE transport.

Run with torchrun; each worker needs one GPU in a shared NVLink domain.
"""

import os

import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.flashinfer.moe_alltoall import FlashInferMoeAlltoAll


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend="nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    top_k, hidden, experts = 16, 3584, world * 56
    transport = FlashInferMoeAlltoAll(
        group=dist.group.WORLD,
        max_tokens=128,
        hidden_size=hidden,
        top_k=top_k,
        num_experts=experts,
        dtype=torch.bfloat16,
        weights_dtype=torch.bfloat16,
    )
    print(f"rank={rank} transport initialized", flush=True)

    for prequantized in (False, True):
        for capacity in (4, 1, 128, 5, 2):
            count = rank % (capacity + 1)
            if rank == world - 1:
                count = capacity
            x = (
                torch.arange(
                    count * hidden, device="cuda", dtype=torch.float32
                ).reshape(count, hidden)
                % 17
                / 16
            ).to(torch.bfloat16)
            ids = (
                (
                    torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
                    + rank * 53
                    + torch.arange(count, device="cuda", dtype=torch.int32)[:, None]
                    * 137
                )
                % experts
            ).contiguous()
            weights = (
                (
                    torch.arange(1, top_k + 1, device="cuda", dtype=torch.float32)[
                        None, :
                    ].expand(count, top_k)
                    / 136
                )
                .to(torch.bfloat16)
                .contiguous()
            )
            dispatched = x
            if prequantized:
                packed = (
                    (
                        torch.arange(
                            count * hidden // 2, device="cuda", dtype=torch.int32
                        )
                        + rank * 31
                    )
                    .remainder(256)
                    .to(torch.uint8)
                    .reshape(count, hidden // 2)
                )
                scales = (
                    (
                        torch.arange(
                            count * hidden // 16, device="cuda", dtype=torch.int32
                        )
                        + rank * 7
                    )
                    .remainder(256)
                    .to(torch.uint8)
                    .reshape(count, hidden // 16)
                )
                dispatched = (packed, scales)
                # Both payloads contribute so missing or misrouted scale rows fail.
                x = (
                    packed.float().repeat_interleave(2, dim=-1)
                    + scales.float().repeat_interleave(16, dim=-1)
                ).to(torch.bfloat16)
            expected = torch.zeros_like(x, dtype=torch.float32)
            for owner in range(world):
                mask = (ids >= owner * 56) & (ids < (owner + 1) * 56)
                factor = torch.where(mask, weights.float(), 0).sum(-1, keepdim=True) * (
                    owner + 1
                )
                expected += (x.float() * factor).to(torch.bfloat16).float()
            expected = expected.to(torch.bfloat16)

            def round_trip():
                recv, recv_ids, recv_weights, offset = transport.dispatch(
                    dispatched, ids, weights, capacity
                )
                if prequantized:
                    recv = (
                        recv[0].float().repeat_interleave(2, dim=-1)
                        + recv[1].float().repeat_interleave(16, dim=-1)
                    ).to(torch.bfloat16)
                mask = (recv_ids >= rank * 56) & (recv_ids < (rank + 1) * 56)
                factor = torch.where(mask, recv_weights.float(), 0).sum(
                    -1, keepdim=True
                ) * (rank + 1)
                partial = (recv.float() * factor).to(torch.bfloat16)
                return transport.combine(partial, count, capacity, offset)

            for _ in range(2):
                actual = round_trip()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = round_trip()
            for _ in range(3):
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            dist.barrier()
            print(
                f"rank={rank} prequantized={prequantized} capacity={capacity} local_tokens={count} eager+graph passed",
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        print("ALLTOALL_ROUNDTRIP_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
