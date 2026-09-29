# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Exact EP8 comparison between two Petit packages selected with PYTHONPATH."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import runpy
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("gpt_oss_120b", "dsv4"), required=True)
    parser.add_argument("--action", choices=("write", "check"), required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--tokens", nargs="+", type=int, required=True)
    parser.add_argument(
        "--routing",
        nargs="+",
        choices=("balanced", "skewed", "empty-destination"),
        required=True,
    )
    parser.add_argument("--resource-report", type=Path)
    args = parser.parse_args()
    import os

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=10))
    assert dist.get_world_size() == 8
    benchmark_path = (
        Path(__file__).resolve().parents[5]
        / "benchmarks/amd/gfx950/ops/bench_gluon_petit_megamoe.py"
    )
    bench = runpy.run_path(str(benchmark_path))
    petit = load_petit_kernel()
    gpt = args.profile == "gpt_oss_120b"
    experts, topk, dim, padded = (128, 4, 2880, 3072) if gpt else (384, 6, 7168, 7168)
    config = petit.MegaMoeConfig(
        world_size=8,
        num_experts=experts,
        topk=topk,
        model_dim=dim,
        activation=petit.MegaMoeActivation.mxfp4,
        activation_function=(
            petit.MegaMoeActivationFunction.swiglu
            if gpt
            else petit.MegaMoeActivationFunction.silu
        ),
        stages=petit.MegaMoeStages.two_stage,
        inter_dim=3072,
        has_bias=gpt,
    )
    torch.manual_seed(20260926 + rank)
    w1, w2, s1, s2, b1, b2 = bench["build_petit_weights"](
        local_experts=experts // 8,
        hidden_size=padded,
        intermediate_size=3072,
        device=device,
    )
    # Nonzero packed biases exercise both projection loads.
    b1.normal_(0, 0.1)
    b2.normal_(0, 0.1)
    heap = petit.create_vmm_symmetric_heap(8)
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    mega = importlib.import_module("lib.moe.rocm.mega_moe.mega_moe_two_stage_kernel")
    records = []
    seen = set()
    for m in args.tokens:
        views = config.input_views(heap, max(m, 1))
        for routing in args.routing:
            torch.manual_seed(1000 + m * 8 + rank)
            hidden = torch.randn((m, dim), dtype=torch.bfloat16, device=device)
            route = torch.arange(m * topk, device=device, dtype=torch.int32).reshape(
                m, topk
            )
            if routing == "balanced":
                ids = (route + rank * topk) % experts
            elif routing == "skewed":
                ids = route % topk
            else:
                ids = (route + rank * topk) % (experts - experts // 8)
            weights = torch.rand((m, topk), device=device)
            weights /= weights.sum(dim=1, keepdim=True)
            inputs = petit.MegaMoeInputViews(
                views.tokens[:m], views.scales[:m], ids, weights
            )
            out = torch.empty((m, padded), dtype=torch.bfloat16, device=device)

            def run() -> torch.Tensor:
                config.quantize(hidden, out=inputs)
                return config.run(
                    heap,
                    w1,
                    w2,
                    s1,
                    s2,
                    m,
                    w13_bias=b1 if gpt else None,
                    w2_bias=b2 if gpt else None,
                    out=out,
                    inputs=inputs,
                )

            dist.barrier()
            expected = run().clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(run(), expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            assert torch.isfinite(expected).all()
            if m:
                assert torch.count_nonzero(expected) > 0
            path = args.reference_dir / f"{args.profile}-{routing}-{m}-rank{rank}.pt"
            if args.action == "write":
                torch.save(expected.cpu(), path)
            else:
                torch.testing.assert_close(
                    expected.cpu(), torch.load(path, weights_only=True), rtol=0, atol=0
                )
            del graph
            for name in ("MegaMoEStage1", "MegaMoEStage2", "MegaMoECombine"):
                jit = vars(mega)[name]
                for kernel in jit.device_caches[rank][0].values():
                    key = kernel.hash
                    if key in seen:
                        continue
                    seen.add(key)
                    asm = kernel.asm["amdgcn"]
                    row = {
                        "m": m,
                        "routing": routing,
                        "rank": rank,
                        "kernel": name,
                        "hash": kernel.hash,
                        "shared": kernel.metadata.shared,
                        "num_warps": kernel.metadata.num_warps,
                    }
                    for field in (
                        "vgpr_count",
                        "sgpr_count",
                        "vgpr_spill_count",
                        "sgpr_spill_count",
                    ):
                        row[field] = int(re.search(rf"\.{field}:\s+(\d+)", asm)[1])
                    for opcode in (
                        "buffer_load_dword",
                        "buffer_load_dwordx2",
                        "buffer_load_dwordx4",
                        "ds_bpermute_b32",
                        "buffer_atomic_or",
                    ):
                        row[opcode] = len(
                            re.findall(rf"^\s*{opcode}\s", asm, re.MULTILINE)
                        )
                    row["mfma"] = len(re.findall(r"^\s*v_mfma", asm, re.MULTILINE))
                    records.append(row)
                    if args.resource_report and rank == 0:
                        assembly_dir = args.resource_report.parent / "assembly"
                        assembly_dir.mkdir(parents=True, exist_ok=True)
                        (assembly_dir / f"{name}-{kernel.hash}.s").write_text(asm)
            if rank == 0:
                print(
                    f"{args.action}: {args.profile} M={m} {routing}: exact eager/graph",
                    flush=True,
                )
    if args.resource_report:
        path = args.resource_report.with_name(
            f"{args.resource_report.stem}.rank{rank}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
