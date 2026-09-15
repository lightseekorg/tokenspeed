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


"""TP8 controlled experiment: quantizer organization versus mailbox fusion."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from flashinfer import fp4_quantize
from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map
from tokenspeed_kernel.ops.communication.triton import all_gather_inner, create_state
from tokenspeed_kernel.ops.moe.latent_down import KimiK3LatentDownOp
from tokenspeed_kernel.ops.moe.latent_down_nvfp4 import (
    KimiK3Nvfp4DownOp,
    _mailbox_geometry,
)
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import (
    PLAIN,
)
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input import (
    launch as scalar_launch,
)
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input_cooperative import (
    launch as cooperative_launch,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pdl", type=int, choices=(0, 1), required=True)
    ap.add_argument("--replays", type=int, required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--ctas", required=True)
    args = ap.parse_args()
    widths = [int(value) for value in args.tokens.split(",")]
    grids = [int(value) for value in args.ctas.split(",")]
    assert widths and min(widths) > 0 and max(widths) <= 8192
    assert grids and min(grids) > 0 and len(grids) == len(set(grids))
    max_m = max(1280, *widths)
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    group = dist.group.WORLD
    assert dist.get_world_size() == 8 and all(gather_fabric_map())
    use_pdl = bool(args.pdl)
    pdl_enabled(use_pdl)
    scale = torch.tensor(128.0, device=device, dtype=torch.float32)
    mailboxes, ops, output_buffers = [], [], []
    for slot in range(2):
        mailbox = KimiK3LatentDownOp.initialize(
            group=group,
            hidden_size=7168,
            latent_size=3584,
            device=device,
            block_index=slot,
            layer_count=2,
            model_scope="cooperative_quant_gate",
            max_m=1280,
        )
        assert mailbox is not None
        op = KimiK3Nvfp4DownOp.initialize(
            mailbox, scale, group=group, max_m=1280, mode="mailbox"
        )
        assert op is not None and op.mode == "mailbox"
        mailboxes.append(mailbox)
        ops.append(op)
        output_buffers.append(
            (
                torch.empty((max_m, 1792), device=device, dtype=torch.uint8),
                torch.empty((max_m, 224), device=device, dtype=torch.uint8),
            )
            if max_m > 1280
            else (op.workspace.data, op.workspace.scales)
        )
    state = (
        create_state(
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
        if max_m > 1280
        else None
    )
    torch.manual_seed(112)
    hidden = torch.randn((max_m, 7168), device=device, dtype=torch.bfloat16)
    torch.manual_seed(137 + rank)
    weight = torch.randn((448, 7168), device=device, dtype=torch.bfloat16) / 64
    rows = []
    for m in widths:
        x = hidden[:m]
        if rank == 0:
            print(json.dumps({"m": m, "stage": "fixture"}), flush=True)

        def latent_input(slot):
            if m <= 1280:
                return mailboxes[slot](x, weight)
            return all_gather_inner(
                state, torch.mm(x, weight.t()), tp_hidden_dim=3584, safe=False
            )

        fixture = latent_input(0).clone()
        torch.cuda.synchronize()
        dist.barrier()

        # These configurations never alter producer geometry or PDL policy.
        variants = [
            ("main", "main", 0, 0),
            ("separate-old", "separate-old", 16, 0),
            ("q-fi", "q-fi", 0, 0),
            ("q-old", "q-old", 16, 0),
        ]
        if m <= 1280:
            variants.insert(1, ("old-fused", "old-fused", 16, 0))
        for values in (8, 4, 2):
            for ctas in grids:
                kinds = ("fused", "separate", "q") if m <= 1280 else ("separate", "q")
                for kind in kinds:
                    variants.append((f"{kind}-v{values}-c{ctas}", kind, values, ctas))

        def execute(spec, slot):
            name, kind, values, ctas = spec
            op, mb = ops[slot], mailboxes[slot]
            if kind == "main":
                return fp4_quantize(
                    latent_input(slot),
                    scale,
                    is_sf_swizzled_layout=False,
                    enable_pdl=use_pdl,
                )
            if kind == "q-fi":
                return fp4_quantize(
                    fixture, scale, is_sf_swizzled_layout=False, enable_pdl=use_pdl
                )
            if kind == "old-fused":
                result = op(x, weight)
                return result.data, result.scales.view(torch.uint8)
            if kind == "fused":
                mb._slot.gemm_by_m[m](
                    x, weight, mb._slot.mailbox, mb._slot.multicast_ptr
                )
                source = mb._slot.mailbox
            elif kind.startswith("q"):
                source = fixture
            else:
                source = latent_input(slot)
            data, sf = (buffer[:m] for buffer in output_buffers[slot])
            if kind.startswith("q"):
                # fp4_quantize allocates distinct outputs for every captured
                # call. Match that working set, especially when 16 large
                # outputs no longer fit in L2; do not compare it to two reused
                # custom buffers. Allocations are not replayed inside the graph.
                data, sf = torch.empty_like(data), torch.empty_like(sf)
            if kind in ("separate-old", "q-old"):
                old_ctas, old_threads = _mailbox_geometry(m)
                scalar_launch(
                    source,
                    data,
                    sf,
                    scale,
                    op.workspace.signals,
                    hidden=3584,
                    m=m,
                    mode=PLAIN,
                    rank=0,
                    world=1,
                    data_mc=0,
                    scale_mc=0,
                    ctas=old_ctas,
                    threads=old_threads,
                )
            else:
                cooperative_launch(
                    source,
                    data,
                    sf,
                    scale,
                    hidden=3584,
                    m=m,
                    values=values,
                    ctas=ctas,
                    threads=128,
                    mailbox=kind == "fused",
                    use_pdl=use_pdl,
                )
            return data, sf

        def check(actual, expected):
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(
                actual[1].flatten(), expected[1].flatten(), rtol=0, atol=0
            )

        expected = tuple(t.clone() for t in execute(variants[0], 0))
        q_spec = next(spec for spec in variants if spec[1] == "q-fi")
        q_expected = tuple(t.clone() for t in execute(q_spec, 0))
        # Eager references use slot zero, as does the first checked variant.
        # Finish consumption/rearming on every peer before reusing that slot;
        # local clone/synchronize alone cannot prevent another rank lapping it.
        # Timed graphs retain their original two-slot rotation without barriers.
        torch.cuda.synchronize()
        dist.barrier()
        for spec in variants:
            if rank == 0:
                print(
                    json.dumps({"m": m, "stage": "check", "variant": spec[0]}),
                    flush=True,
                )
            check(execute(spec, 0), q_expected if spec[1].startswith("q") else expected)
            torch.cuda.synchronize()
            dist.barrier()
            if spec[1] in ("fused", "separate"):
                for late in range(8):
                    x.neg_()
                    ref = tuple(t.clone() for t in execute(variants[0], late % 2))
                    dist.barrier()
                    if rank == late:
                        torch.cuda._sleep(200000)
                    check(execute(spec, late % 2), ref)
                    torch.cuda.synchronize()
                    dist.barrier()
        graphs = []
        if rank == 0:
            print(json.dumps({"m": m, "stage": "capture"}), flush=True)
        for spec in variants:
            for _ in range(3):
                execute(spec, 0)
                execute(spec, 1)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            # Quantizer-only controls are shorter than a host graph launch.
            # Amortize the host gap without changing the full-pipeline graph.
            operations = 16 if spec[1].startswith("q") else 2
            with torch.cuda.graph(graph):
                output = tuple(execute(spec, i % 2) for i in range(operations))
            assert len({result[0].data_ptr() for result in output}) == operations
            assert len({result[1].data_ptr() for result in output}) == operations
            graphs.append((graph, output, operations))
        timings = [[] for _ in variants]
        if rank == 0:
            print(json.dumps({"m": m, "stage": "timing"}), flush=True)
        for repeat in range(args.rounds):
            order = (
                range(len(variants))
                if repeat % 2 == 0
                else reversed(range(len(variants)))
            )
            for index in order:
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(args.replays):
                    graphs[index][0].replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / args.replays / graphs[index][2],
                    device=device,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                timings[index].append(elapsed.item())
        # Buffers are deliberately shared across variants. Replay each graph
        # before checking it, rather than reading an overwritten borrowed view.
        for index, spec in enumerate(variants):
            graphs[index][0].replay()
            torch.cuda.synchronize()
            reference = q_expected if spec[1].startswith("q") else expected
            for result in graphs[index][1]:
                check(result, reference)
            dist.barrier()
        row = {
            "m": m,
            "baseline_collective": "mailbox" if m <= 1280 else "all_gather_inner",
            "bitwise_equal": True,
            "delayed_peers_checked": 8,
            "projections_per_replay": 2,
            "variants": [
                {
                    "name": s[0],
                    "kind": s[1],
                    "values_per_lane": s[2],
                    "ctas": s[3],
                    "operations_per_replay": g[2],
                    "distinct_output_buffers": len(
                        {result[0].data_ptr() for result in g[1]}
                    ),
                    "us": t,
                }
                for s, t, g in zip(variants, timings, graphs)
            ],
        }
        rows.append(row)
        if rank == 0:
            result = {
                "pdl": use_pdl,
                "replays": args.replays,
                "rounds": args.rounds,
                "ctas": grids,
                "rows": rows,
                "timing": "slowest rank, per operation, hot two-slot CUDA Graph",
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({"m": m, "pass": True, "variants": len(variants)}),
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
