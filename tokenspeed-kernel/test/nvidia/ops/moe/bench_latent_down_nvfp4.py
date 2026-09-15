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

"""Eight-rank correctness and paired down-projection input-preparation benchmark.

Launch with torchrun on two four-GPU nodes. Results use the slowest rank;
the baseline is the exact mailbox / all_gather_inner path from main, followed
by FlashInfer fp4_quantize. No model weights or sampling parameters are changed.
"""

import argparse
import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from flashinfer import fp4_quantize
from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map
from tokenspeed_kernel.ops.communication.triton import all_gather_inner, create_state
from tokenspeed_kernel.ops.moe.activation import Nvfp4Activation
from tokenspeed_kernel.ops.moe.latent_down import KimiK3LatentDownOp
from tokenspeed_kernel.ops.moe.latent_down_nvfp4 import KimiK3Nvfp4DownOp
from tokenspeed_kernel.platform import pdl_enabled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--tokens",
        default="1,4,8,9,16,32,64,128,256,512,1024,1279,1280,1281,2048,4096,8192",
    )
    ap.add_argument("--replays", type=int, default=1000)
    ap.add_argument("--delay-cycles", type=int, default=200000)
    ap.add_argument("--pdl", type=int, choices=(0, 1), default=1)
    ap.add_argument("--mode", choices=("all", "auto"), default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    group = dist.group.WORLD
    assert dist.get_world_size() == 8, "this gate requires TP8"
    assert all(
        gather_fabric_map()
    ), "all eight ranks must pass the fabric allocation probe"
    pdl_enabled(bool(args.pdl))
    widths = [int(x) for x in args.tokens.split(",")]
    max_m = max(1280, *widths)
    scale = torch.tensor(128.0, device=device, dtype=torch.float32)
    mailboxes, ops = [], []
    for slot in range(2):
        mailbox = KimiK3LatentDownOp.initialize(
            group=group,
            hidden_size=7168,
            latent_size=3584,
            device=device,
            block_index=slot,
            layer_count=2,
            model_scope="nvfp4_input_gate",
            max_m=1280,
        )
        assert mailbox is not None, "multicast down path must be live, not replicated"
        op = KimiK3Nvfp4DownOp.initialize(
            mailbox, scale, group=group, max_m=max_m, mode=args.mode
        )
        assert op is not None and op.mode == args.mode
        mailboxes.append(mailbox)
        ops.append(op)
    state = create_state(
        group=group,
        rank_in_group=rank,
        attnres_max_numel=0,
        attnres_max_rows=0,
        max_tokens=max_m,
        hidden_size=3584,
        device=device,
        max_numel=0,
        max_bytes=0,
    )
    torch.manual_seed(112)
    hidden = torch.randn((max_m, 7168), device=device, dtype=torch.bfloat16)
    torch.manual_seed(137 + rank)
    weight = torch.randn((448, 7168), device=device, dtype=torch.bfloat16) / 64
    rows = []
    for m in widths:
        x = hidden[:m]

        def baseline(slot=0):
            if m <= 1280:
                latent = mailboxes[slot](x, weight)
            else:
                local = torch.mm(x, weight.t())
                latent = all_gather_inner(state, local, tp_hidden_dim=3584, safe=False)
            return fp4_quantize(
                latent, scale, is_sf_swizzled_layout=False, enable_pdl=bool(args.pdl)
            )

        def candidate(slot=0):
            if ops[slot].handles(m):
                return ops[slot](x, weight)
            data, sf = baseline(slot)
            return Nvfp4Activation(data, sf.view(torch.float8_e4m3fn), (m, 3584), scale)

        ref, ref_sf = baseline()
        ref, ref_sf = ref.clone(), ref_sf.flatten().clone()
        dist.barrier()
        actual = candidate()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.data, ref, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.scales.view(torch.uint8).flatten(), ref_sf, rtol=0, atol=0
        )
        dist.barrier()

        # Alternate the payload while each peer takes a turn arriving late.
        # Identical graph inputs alone cannot expose stale-generation reads.
        for late_rank in range(8):
            x.neg_()
            delayed_ref, delayed_sf = baseline(late_rank % 2)
            delayed_ref = delayed_ref.clone()
            delayed_sf = delayed_sf.flatten().clone()
            dist.barrier()
            if rank == late_rank:
                torch.cuda._sleep(args.delay_cycles)
            delayed = candidate(late_rank % 2)
            torch.cuda.synchronize()
            torch.testing.assert_close(delayed.data, delayed_ref, rtol=0, atol=0)
            torch.testing.assert_close(
                delayed.scales.view(torch.uint8).flatten(),
                delayed_sf,
                rtol=0,
                atol=0,
            )
            dist.barrier()

        graphs = []
        # One replay is a whole two-slot rotation, as in the model. Report
        # per-projection time; do not benchmark an unsupported one-slot reuse.
        for fn in (
            lambda: (baseline(0), baseline(1)),
            lambda: (candidate(0), candidate(1)),
        ):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn()
            graphs.append((graph, output))

        timings = [[], []]
        for repeat in range(6):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                graph = graphs[index][0]
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / args.replays / 2, device=device
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                timings[index].append(elapsed.item())
        for fused in graphs[1][1]:
            torch.testing.assert_close(fused.data, ref, rtol=0, atol=0)
            torch.testing.assert_close(
                fused.scales.view(torch.uint8).flatten(), ref_sf, rtol=0, atol=0
            )
        row = {
            "m": m,
            "fused_path": ops[0].handles(m),
            "bitwise_equal": True,
            "delayed_peers_checked": 8,
            "projections_per_replay": 2,
            "main_us": timings[0],
            "fused_us": timings[1],
        }
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "mode": args.mode,
                        "pdl": bool(args.pdl),
                        "replays": args.replays,
                        "rows": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
        del graphs
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
