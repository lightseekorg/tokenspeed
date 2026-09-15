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

"""Checkpoint-free TP8 reproduction of the documented complete-tail boundaries.

Run in a matching GB300/CUTLASS environment. This new, functionally named
harness is not the historical numerical certificate; rerun the final PR head.
Three rounds are a bounded smoke, never formal performance acceptance.
"""

import argparse
import json
import math
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from fused_rs_up_ag_admission import (
    _query_cluster_admission,
    cap_smem_precheck,
)
from fused_rs_up_ag_inputs import _make_layer_inputs, _semantic_reference
from fused_rs_up_ag_statistics import paired_statistics
from tokenspeed_kernel.ops.communication.fused_rs_up_projection import (
    prepare_fused_rs_up_projection,
)
from tokenspeed_kernel.ops.communication.fused_rs_up_projection_config import (
    fused_rs_tail_ht_config,
    fused_rs_up_projection_config,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
    MNNVLCuteDSLHTFinalizeAllReduceRMSNorm,
    MNNVLCuteDSLHTFinalizeTuning,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import (
    BoundSharedRs,
    SharedRsWorkspace,
    _vote,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    BoundSymmetricUpProjection,
    SymmetricUpProjectionOverlapTuning,
    allocate_symmetric_up_projection_output,
)
from tokenspeed_kernel.ops.communication.shared_rs_contract import SharedRsTuning
from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_shared_down_projection
from up_projection_resources import collect_plan_resources


class TailArm:
    """Keep four outputs and all workspace/HT owners alive on one explicit stream."""

    def __init__(self, layers, producers, m, device, fused, mode):
        self.layers, self.producers = layers, producers
        self.mode, self.fused = mode, fused
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.device = device
        self.partials = [torch.empty_like(layer.shared_partial) for layer in layers]
        self.ht = MNNVLCuteDSLHTFinalizeAllReduceRMSNorm.initialize(
            group=dist.group.WORLD,
            hidden_size=3584,
            top_k=16,
            rms_eps=1e-5,
            candidate_min_tokens=1280,
            candidate_max_tokens=8192,
            tuning_routes=(
                MNNVLCuteDSLHTFinalizeTuning(
                    **fused_rs_tail_ht_config(10 if fused else 7)
                ),
            ),
        )
        self.workspace = SharedRsWorkspace.allocate(dist.group.WORLD, m, device)
        # Bind the actual persistent HT result without running an unadmitted up GEMM.
        latent = self.ht._backend._output[:m]
        self.rs = None
        if not fused:
            self.rs = BoundSharedRs.prepare(
                self.workspace,
                m,
                SharedRsTuning(64, 1024, 4, "row"),
                tuple(x for layer in layers for x in (layer.prefix, layer.up_weight)),
            )
        self.plans = []
        for layer in layers:
            output = allocate_symmetric_up_projection_output(
                dist.group.WORLD, m, device=device
            )
            if fused:
                plan = prepare_fused_rs_up_projection(
                    latent,
                    layer.up_weight,
                    layer.prefix,
                    self.workspace,
                    output,
                    residual_is_replicated=True,
                )
            else:
                tuning = fused_rs_up_projection_config(m)["tuning"]
                tuning.pop("reduce_vectors")
                tuning["addend_stages"] = 0
                plan = BoundSymmetricUpProjection.prepare(
                    latent,
                    layer.up_weight,
                    self.rs.output_view,
                    layer.prefix,
                    output,
                    residual_is_replicated=True,
                    tuning=SymmetricUpProjectionOverlapTuning(**tuning),
                    skip_entry_sync=True,
                )
            self.plans.append(plan)

    def __call__(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise ValueError("complete tail must run on its prepared stream")
        results = []
        for slot, (layer, plan) in enumerate(zip(self.layers, self.plans)):
            if self.mode == "tail":
                partial = layer.shared_partial
            else:
                destination = (
                    plan.input_view if self.mode == "direct" else self.partials[slot]
                )
                partial = kimi3_shared_down_projection(
                    *self.producers[slot], out=destination, solution="torch"
                )
            if self.fused:
                if self.mode != "direct":
                    plan.stage(partial)
            else:
                self.rs.stage(partial)
                self.rs.run()
            latent = self.ht(
                layer.gemm2_output,
                layer.expert_weights,
                layer.expanded_idx,
                layer.gamma,
            )
            bound = plan.plan if self.fused else plan
            if latent.data_ptr() != bound.inputs[0].data_ptr():
                raise RuntimeError("HT changed its prebound output pointer")
            results.append(plan())
        return results


def admit(arm):
    """Inspect actual CUBIN before the first fused producer launch on every rank."""
    records, error = [], None
    try:
        for wrapped in arm.plans:
            plan = wrapped.plan
            resources = collect_plan_resources(plan)
            ptx = plan.compiled.__ptx__
            if type(ptx) is bytes:
                ptx = ptx.decode()
            cubin = plan.compiled.__cubin__
            precheck = cap_smem_precheck(resources, ptx, plan.max_active_clusters)
            capacity = _query_cluster_admission(
                cubin,
                precheck["ptx_entry"],
                plan.max_active_clusters,
                precheck["dynamic_smem_upper_bound_bytes"],
            )
            records.append(
                dict(resources=resources, precheck=precheck, capacity=capacity)
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _vote(dist.group.WORLD, ("fused-rs-up-ag-binary-admission", arm.mode), error)
    return records


def checked_equal(actual, expected, label):
    error = None
    if not all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(actual, expected)
    ):
        error = f"bitwise mismatch: {label}"
    _vote(dist.group.WORLD, ("fused-rs-up-ag-bitwise", label), error)


def time_pair(baseline, candidate, rounds, device):
    samples = [[], []]
    graphs = (baseline, candidate)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for _ in range(8):
        baseline.replay()
        candidate.replay()
    torch.cuda.synchronize()
    dist.barrier()
    for round_index in range(rounds):
        for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
            start.record()
            for _ in range(20):
                graphs[index].replay()
            end.record()
            end.synchronize()
            local = torch.tensor(
                [start.elapsed_time(end) * 1000 / (20 * 4)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(local, op=dist.ReduceOp.MAX)
            samples[index].append(local.item())
    return paired_statistics(*samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", required=True, type=int, choices=(4096, 8192))
    parser.add_argument("--scope", required=True, choices=("primary", "producer"))
    parser.add_argument("--rounds", required=True, type=int, choices=(3, 31))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    if dist.get_world_size() != 8:
        raise ValueError("this benchmark requires one TP8 MNNVL group")
    device = torch.device("cuda", torch.cuda.current_device())
    rank, m = dist.get_rank(), args.tokens
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        layers = [_make_layer_inputs(rank, m, slot, device) for slot in range(4)]
        generator = torch.Generator(device=device).manual_seed(87000 + rank)
        producers = (
            [
                (
                    torch.randn(
                        m, 768, device=device, dtype=torch.bfloat16, generator=generator
                    )
                    * 0.03,
                    torch.randn(
                        7168,
                        768,
                        device=device,
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    * 0.03,
                )
                for _ in range(4)
            ]
            if args.scope == "producer"
            else []
        )
        mode = "tail" if args.scope == "primary" else "copied"
        arms = {
            "materialized": TailArm(layers, producers, m, device, False, mode),
            "fused_copied": TailArm(layers, producers, m, device, True, mode),
        }
        if args.scope == "producer":
            arms["fused_direct"] = TailArm(layers, producers, m, device, True, "direct")
        admission = {name: admit(arm) for name, arm in arms.items() if arm.fused}
        eager = {name: [x.clone() for x in arm()] for name, arm in arms.items()}
        for name, outputs in eager.items():
            checked_equal(outputs, eager["materialized"], name)
        # Independent reference retains the original bounds, never a looser rerun gate.
        reference_errors = []
        for slot, layer in enumerate(layers):
            if producers:
                partial = kimi3_shared_down_projection(
                    *producers[slot],
                    out=torch.empty_like(layer.shared_partial),
                    solution="torch",
                )
                layer = replace(layer, shared_partial=partial)
            reference = _semantic_reference(layer, rank).float()
            actual = eager["fused_copied"][slot].float()
            difference = actual - reference
            absolute = difference.abs().max().item()
            relative_l2 = (difference.norm() / reference.norm().clamp_min(1e-12)).item()
            error = None
            if (
                not math.isfinite(absolute)
                or not math.isfinite(relative_l2)
                or absolute > 0.046875 + 0.01 * reference.abs().max().item()
                or relative_l2 > 0.006
            ):
                error = "original complete-tail reference bounds exceeded"
            _vote(dist.group.WORLD, ("fused-rs-up-ag-reference", slot), error)
            reference_errors.append(
                dict(max_absolute=absolute, relative_l2=relative_l2)
            )
        graphs, outputs = {}, {}
        for name, arm in arms.items():
            for _ in range(8):
                arm()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs[name] = arm()
            graph.replay()
            checked_equal(outputs[name], eager[name], name + "-eager-graph")
            graphs[name] = graph
        # A/B/A changes every slot's replicated residual; graph pointers remain fixed.
        saved_prefixes = [layer.prefix.clone() for layer in layers]
        flags = torch.empty(128, len(arms) - 1, 4, device=device, dtype=torch.bool)
        changed = torch.empty(127, len(arms), 4, device=device, dtype=torch.bool)
        previous = {
            name: [x.clone() for x in values] for name, values in outputs.items()
        }
        comparison = torch.empty((m, 7168), device=device, dtype=torch.bool)
        for generation in range(128):
            for layer, saved in zip(layers, saved_prefixes):
                layer.prefix.copy_(saved)
                if generation % 2:
                    layer.prefix.add_(0.03125)
            for graph in graphs.values():
                graph.replay()
            for arm_index, name in enumerate(list(arms)[1:]):
                for slot in range(4):
                    torch.eq(
                        outputs[name][slot].view(torch.int16),
                        outputs["materialized"][slot].view(torch.int16),
                        out=comparison,
                    )
                    torch.all(comparison, out=flags[generation, arm_index, slot])
            for arm_index, name in enumerate(arms):
                for slot in range(4):
                    if generation:
                        torch.ne(
                            outputs[name][slot].view(torch.int16),
                            previous[name][slot].view(torch.int16),
                            out=comparison,
                        )
                        torch.any(
                            comparison, out=changed[generation - 1, arm_index, slot]
                        )
                    previous[name][slot].copy_(outputs[name][slot])
        passed = flags.all().item() and changed.all().item()
        error = None if passed else "continuous replay mismatch or stale output"
        _vote(dist.group.WORLD, ("fused-rs-up-ag-continuous", 128), error)
        for layer, saved in zip(layers, saved_prefixes):
            layer.prefix.copy_(saved)
        for name, graph in graphs.items():
            graph.replay()
            checked_equal(outputs[name], eager[name], name + "-restored-A")
        pairs = [("materialized", "fused_copied")]
        if args.scope == "producer":
            pairs += [
                ("fused_copied", "fused_direct"),
                ("materialized", "fused_direct"),
            ]
        comparisons = {
            f"{a}_to_{b}": time_pair(graphs[a], graphs[b], args.rounds, device)
            for a, b in pairs
        }
        local = dict(
            rank=rank,
            admission=admission,
            reference_errors=reference_errors,
            continuous_checks=flags.numel() + changed.numel(),
            continuous_failures=0,
        )
        ranks = [None] * 8
        dist.all_gather_object(ranks, local)
        if rank == 0:
            result = dict(
                schema="fused-shared-rs-up-projection-ag-rerun",
                M=m,
                scope=args.scope,
                rounds=args.rounds,
                graph_layers=4,
                warmup=8,
                iters=20,
                instrumented=False,
                latency_rank_reduction="max",
                smoke_only=args.rounds == 3,
                runtime_enabled=False,
                config=fused_rs_up_projection_config(m),
                comparisons=comparisons,
                ranks=ranks,
                note="New harness smoke/paired rerun, not a replacement for full historical numerical certification.",
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
