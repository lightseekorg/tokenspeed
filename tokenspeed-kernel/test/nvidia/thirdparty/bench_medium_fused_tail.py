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

"""TP8 checkpoint-free medium-M complete-tail tuning with a matched BT prefix.

The denominator is BT-only plus the original addmm/full-width AllReduce #2,
not unchanged main. Both arms include the same shared-input copy before BT.
One process invocation measures one geometry/M; a performance loss is data,
not an exception. Correctness or admission failure produces a failed receipt
and a nonzero exit. Source hashes and allocation metadata belong in the caller's
manifest. No model text, request bodies or device tensor contents are exported.
"""

import argparse
import json
import os
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from fused_rs_up_ag_admission import _query_cluster_admission, cap_smem_precheck
from fused_rs_up_ag_inputs import (
    _local_finalize_reference,
    _make_layer_inputs,
    _rank_ordered_sum,
    _rmsnorm_reference,
    _semantic_reference,
)
from fused_rs_up_ag_statistics import paired_statistics, performance_gate
from medium_fused_tail_contract import (
    BT_TUNING,
    REFERENCE_ATOL,
    REFERENCE_L2,
    REFERENCE_RTOL,
    measurement_config,
    reference_passes,
)
from tokenspeed_kernel.ops.communication.fused_rs_up_projection_config import (
    fused_rs_up_projection_config,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
    MNNVLCuteDSLBTFinalizeTuning,
    MNNVLCuteDSLFinalizeAllReduceRMSNorm,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_fused_rs_up_projection import (
    BoundFusedRsUpProjection,
    FusedRsUpProjectionTuning,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import (
    SharedRsWorkspace,
    _vote,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    allocate_symmetric_up_projection_output,
)
from tokenspeed_kernel.ops.communication.multimem import (
    multimem_all_reduce_staged,
    multimem_prealloc,
    multimem_stage,
)
from up_projection_resources import collect_plan_resources


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument(
        "--implementation", required=True, choices=("inherited", "medium")
    )
    parser.add_argument("--tile-m", required=True, type=int)
    parser.add_argument("--tile-n", required=True, type=int)
    parser.add_argument("--two-cta", required=True, type=int, choices=(0, 1))
    parser.add_argument("--cluster-m", required=True, type=int)
    parser.add_argument("--cluster-n", required=True, type=int)
    parser.add_argument("--cluster-cap", required=True, type=int)
    parser.add_argument("--scheduler-type", required=True)
    parser.add_argument("--ab-stages", type=int, default=0)
    parser.add_argument("--c-stages", required=True, type=int)
    parser.add_argument("--addend-stages", required=True, type=int)
    parser.add_argument("--reduce-vectors", required=True, type=int)
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--replays", required=True, type=int)
    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--rank-skew-cycles", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def make_tuning(args):
    if args.cluster_cap < 0 or args.rank_skew_cycles < 0:
        raise ValueError("cluster cap and skew cycles must be nonnegative")
    values = {
        "tile_m": args.tile_m,
        "tile_n": args.tile_n,
        "two_cta": bool(args.two_cta),
        "cluster_m": args.cluster_m,
        "cluster_n": args.cluster_n,
        "c_stages": args.c_stages,
        "addend_stages": args.addend_stages,
        "reduce_vectors": args.reduce_vectors,
        "cluster_cap": None if args.cluster_cap == 0 else args.cluster_cap,
        "scheduler_type": args.scheduler_type,
        "ab_stages": args.ab_stages,
    }
    if args.implementation == "medium":
        from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection import (
            MediumFusedRsUpProjectionTuning,
        )

        return MediumFusedRsUpProjectionTuning(**values)
    if (
        (args.tile_m, args.tile_n, args.two_cta, args.cluster_m, args.cluster_n)
        != (256, 128, 1, 2, 1)
        or args.cluster_cap != 0
        or args.ab_stages != 0
        or args.scheduler_type != "static_persistent"
        or (args.c_stages, args.addend_stages, args.reduce_vectors) != (0, 1, 4)
    ):
        raise ValueError(
            "inherited control is the frozen 256x128/2CTA/group4/auto-C geometry"
        )
    # Only the tuning dictionary is reused: the qualified large-M facade is
    # deliberately not called, and no HT config is imported into this BT test.
    return FusedRsUpProjectionTuning(**fused_rs_up_projection_config(8192)["tuning"])


class TailPair:
    """Hold identical BT state and four independently live outputs per arm."""

    def __init__(self, layers, args, tuning, device):
        self.layers, self.m, self.device = layers, args.tokens, device
        self.stream = torch.cuda.current_stream(device).cuda_stream
        self.group = dist.group.WORLD
        self.group_name, self.rank = self.group.group_name, dist.get_rank()
        if not multimem_prealloc(1024, (7168,), self.group_name):
            raise RuntimeError("original shared multimem staging is unavailable")
        self.bt = MNNVLCuteDSLFinalizeAllReduceRMSNorm.initialize(
            group=self.group,
            hidden_size=3584,
            top_k=16,
            rms_eps=1e-5,
            candidate_min_tokens=256,
            candidate_max_tokens=1024,
            tuning_routes=(MNNVLCuteDSLBTFinalizeTuning(**BT_TUNING),),
        )
        # The real BT result, not a scratch latent or separately changed prefix.
        first = layers[0]
        latent = self.bt(
            first.gemm2_output, first.expert_weights, first.expanded_idx, first.gamma
        )
        self.workspace = SharedRsWorkspace.allocate(self.group, self.m + 32, device)
        self.raw_guard = self.workspace.state.comm_buff[self.m :]
        self.raw_guard.fill_(53.0)
        self.plans, self.output_owners, self.output_guards = [], [], []
        for layer in layers:
            owner = allocate_symmetric_up_projection_output(
                self.group, self.m + 32, device=device
            )
            output = replace(owner, tensor=owner.tensor[: self.m])
            self.output_owners.append(owner)
            self.output_guards.append(owner.tensor[self.m :])
            self.output_guards[-1].fill_(61.0)
            if args.implementation == "medium":
                from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection import (
                    prepare_medium_fused_rs_up_projection,
                )

                plan = prepare_medium_fused_rs_up_projection(
                    latent,
                    layer.up_weight,
                    layer.prefix,
                    self.workspace,
                    output,
                    residual_is_replicated=True,
                    tuning=tuning,
                )
            else:
                plan = BoundFusedRsUpProjection.prepare_fused(
                    latent,
                    layer.up_weight,
                    layer.prefix,
                    self.workspace,
                    output,
                    residual_is_replicated=True,
                    tuning=tuning,
                )
            self.plans.append(plan)

    def _check_stream(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise ValueError("tail execution must use its explicitly prepared stream")

    def baseline(self):
        self._check_stream()
        outputs = []
        for layer in self.layers:
            shared = multimem_stage(layer.shared_partial, self.group_name, 1024)
            if shared is None:
                raise RuntimeError("baseline shared stage is unavailable")
            latent = self.bt(
                layer.gemm2_output,
                layer.expert_weights,
                layer.expanded_idx,
                layer.gamma,
            )
            start = self.rank * 896
            owner = shared[:, start : start + 896]
            owner.add_(layer.prefix[:, start : start + 896])
            owner.addmm_(latent, layer.up_weight.t())
            outputs.append(multimem_all_reduce_staged(shared, self.group_name).clone())
        return outputs

    def candidate(self):
        self._check_stream()
        outputs = []
        for layer, plan in zip(self.layers, self.plans):
            plan.stage(layer.shared_partial)
            latent = self.bt(
                layer.gemm2_output,
                layer.expert_weights,
                layer.expanded_idx,
                layer.gamma,
            )
            if latent.data_ptr() != plan.inputs[0].data_ptr():
                raise RuntimeError("BT changed its bound output pointer")
            outputs.append(plan())
        return outputs

    def guard_checks(self):
        return bool(torch.all(self.raw_guard == 53.0).item()) and all(
            bool(torch.all(guard == 61.0).item()) for guard in self.output_guards
        )


def checked(condition, label):
    _vote(dist.group.WORLD, ("medium-tail-check", label), None if condition else label)


def equal_outputs(actual, expected):
    return all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(actual, expected)
    )


def reference_errors(actual, reference):
    difference = actual.float() - reference.float()
    absolute = difference.abs().max().item()
    relative_l2 = (difference.norm() / reference.float().norm().clamp_min(1e-12)).item()
    maximum = reference.abs().max().item()
    return {
        "max_absolute": absolute,
        "relative_l2": relative_l2,
        "reference_max": maximum,
        "passed": reference_passes(absolute, relative_l2, maximum),
    }


def independent_fp32_reference(layer, rank):
    """FP32 sum/shared/up reference, keeping the defined BF16 routed boundary."""
    routed = _rmsnorm_reference(
        _rank_ordered_sum(_local_finalize_reference(layer)), layer.gamma
    )
    shared = layer.shared_partial.float()
    dist.all_reduce(shared)
    up = torch.zeros_like(shared)
    up[:, rank * 896 : (rank + 1) * 896].copy_(
        torch.mm(routed.float(), layer.up_weight.float().t())
    )
    dist.all_reduce(up)
    return shared + up + layer.prefix.float()


def admit(pair, implementation):
    records = []
    for slot, plan in enumerate(pair.plans):
        resources, error, admission = None, None, None
        try:
            resources = collect_plan_resources(plan)
            if resources["driver"].get("status") != "available":
                raise ValueError(
                    "compiled CUBIN resources must be available before launch"
                )
            if implementation == "inherited":
                ptx = plan.compiled.__ptx__
                if type(ptx) is bytes:
                    ptx = ptx.decode()
                check = cap_smem_precheck(resources, ptx, plan.max_active_clusters)
                admission = _query_cluster_admission(
                    plan.compiled.__cubin__,
                    check["ptx_entry"],
                    plan.max_active_clusters,
                    check["dynamic_smem_upper_bound_bytes"],
                )
            else:
                # The medium binder performs actual CUBIN capacity validation
                # collectively before returning; preserve its numeric receipt.
                admission = getattr(plan, "admission_records", None)
        except Exception as exc:
            error = type(exc).__name__
        _vote(pair.group, ("medium-tail-binary-admission", slot), error)
        # Symbol names, arbitrary messages and absolute paths are not exported.
        driver = {
            key: value
            for key, value in resources["driver"].items()
            if type(value) in (int, float, bool)
        }
        records.append(
            {
                "resources": resources["static"],
                "artifacts": resources["artifacts"],
                "driver": driver,
                "max_active_clusters": plan.max_active_clusters,
                "admission": admission,
            }
        )
    return records


def capture_pair(pair, stream, warmup):
    eager, graphs, outputs = {}, {}, {}
    for name, function in (("baseline", pair.baseline), ("candidate", pair.candidate)):
        eager[name] = [value.clone() for value in function()]
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(pair.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs[name] = function()
        graph.replay()
        checked(equal_outputs(outputs[name], eager[name]), name + "-eager-graph")
        graphs[name] = graph
    return eager, graphs, outputs


def check_changed_inputs(pair, graphs, outputs, eager, generations, skew_cycles):
    saved = [
        {
            key: getattr(layer, key).clone()
            for key in ("gemm2_output", "shared_partial", "prefix")
        }
        for layer in pair.layers
    ]
    distinct = 0
    for generation in range(generations):
        for slot, layer in enumerate(pair.layers):
            for key, value in saved[slot].items():
                getattr(layer, key).copy_(value)
            if generation % 2:
                layer.gemm2_output.neg_()
                layer.shared_partial.neg_()
                layer.prefix.add_(0.125)
        for name, graph in graphs.items():
            if skew_cycles and pair.rank == generation % 8:
                torch.cuda._sleep(skew_cycles)
            graph.replay()
            if generation % 2 == 0:
                checked(equal_outputs(outputs[name], eager[name]), name + "-restored-A")
            else:
                changed = all(
                    not torch.equal(a, b) for a, b in zip(outputs[name], eager[name])
                )
                checked(changed, name + "-changed-B")
                distinct += int(changed)
        if generation == 1:
            graph_values = {
                name: [value.clone() for value in values]
                for name, values in outputs.items()
            }
            for name in graphs:
                function = pair.baseline if name == "baseline" else pair.candidate
                checked(
                    equal_outputs(function(), graph_values[name]),
                    name + "-changed-eager-graph",
                )
            for slot, layer in enumerate(pair.layers):
                reference = independent_fp32_reference(layer, pair.rank)
                for name in graphs:
                    checked(
                        reference_errors(graph_values[name][slot], reference)["passed"],
                        name + "-changed-reference",
                    )
        checked(pair.guard_checks(), "raw-output-guard")
    for slot, layer in enumerate(pair.layers):
        for key, value in saved[slot].items():
            getattr(layer, key).copy_(value)
    for name, graph in graphs.items():
        graph.replay()
        checked(equal_outputs(outputs[name], eager[name]), name + "-final-A")
    return {
        "generations": generations,
        "changed_checks": distinct,
        "ABA": True,
        "rank_skew": skew_cycles > 0,
    }


def check_markers(pair, graphs, outputs):
    saved = [
        {
            key: getattr(layer, key).clone()
            for key in ("gemm2_output", "shared_partial", "prefix")
        }
        for layer in pair.layers
    ]
    rows = torch.arange(pair.m, device=pair.device, dtype=torch.float32)[:, None]
    columns = torch.arange(7168, device=pair.device, dtype=torch.float32)[None, :]
    errors = []
    for case in ("zero", "rank_row_column", "positive_negative"):
        for slot, layer in enumerate(pair.layers):
            layer.gemm2_output.zero_()
            layer.shared_partial.zero_()
            layer.prefix.zero_()
            if case == "rank_row_column":
                layer.shared_partial.copy_(
                    (pair.rank + 1) / 128 + (rows % 16) / 2048 + (columns % 8) / 1024
                )
                layer.prefix.copy_((columns // 896) / 32 + (rows % 8) / 128 + slot / 64)
            elif case == "positive_negative":
                sign = 1 if pair.rank % 2 else -1
                layer.shared_partial.copy_(
                    sign * (1 / 8 + (rows % 16) / 2048 + (columns % 8) / 1024)
                )
                layer.prefix.fill_(0.125 + slot / 64)
        for graph in graphs.values():
            graph.replay()
        for slot, layer in enumerate(pair.layers):
            reference = independent_fp32_reference(layer, pair.rank)
            for name in graphs:
                error = reference_errors(outputs[name][slot], reference)
                checked(error["passed"], name + "-" + case)
                errors.append({"case": case, "slot": slot, "arm": name, **error})
        checked(pair.guard_checks(), "marker-guards")
    for slot, layer in enumerate(pair.layers):
        for key, value in saved[slot].items():
            getattr(layer, key).copy_(value)
    return errors


def time_pair(graphs, args, device):
    samples = {name: [] for name in graphs}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.warmup):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    for round_index in range(args.rounds):
        order = (
            ("baseline", "candidate")
            if round_index % 2 == 0
            else ("candidate", "baseline")
        )
        for name in order:
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            latency = torch.tensor(
                [start.elapsed_time(end) * 1000 / (args.replays * 4)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(latency, op=dist.ReduceOp.MAX)
            samples[name].append(latency.item())
    return paired_statistics(samples["baseline"], samples["candidate"])


def run(args, result, device):
    tuning, error = None, None
    try:
        tuning = make_tuning(args)
    except Exception as exc:
        error = type(exc).__name__
    _vote(
        dist.group.WORLD,
        (
            "medium-tail-tuning",
            args.implementation,
            None if tuning is None else asdict(tuning),
        ),
        error,
    )
    result["tuning"] = asdict(tuning)
    _vote(
        dist.group.WORLD,
        (
            "medium-tail-config",
            result["measurement"],
            result["tuning"],
            args.implementation,
        ),
        None,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        rank = dist.get_rank()
        layers = [
            _make_layer_inputs(rank, args.tokens, slot, device) for slot in range(4)
        ]
        checked(
            len({x.up_weight.data_ptr() for x in layers}) == 4, "independent-weights"
        )
        pair = TailPair(layers, args, tuning, device)
        result["phase"] = "binary_admission"
        admission = admit(pair, args.implementation)
        result["phase"] = "correctness"
        eager, graphs, outputs = capture_pair(pair, stream, args.warmup)
        errors = []
        for slot, layer in enumerate(layers):
            references = {
                "old_semantics": _semantic_reference(layer, rank),
                "fp32": independent_fp32_reference(layer, rank),
            }
            for name in graphs:
                for reference_name, reference in references.items():
                    error = reference_errors(eager[name][slot], reference)
                    checked(error["passed"], name + "-" + reference_name)
                    errors.append(
                        {
                            "arm": name,
                            "reference": reference_name,
                            "slot": slot,
                            **error,
                        }
                    )
        replay = check_changed_inputs(
            pair, graphs, outputs, eager, args.generations, args.rank_skew_cycles
        )
        markers = check_markers(pair, graphs, outputs)
        result["phase"] = "performance"
        stats = time_pair(graphs, args, device)
        checked(pair.guard_checks(), "final-guards")
        records = [None] * dist.get_world_size()
        dist.all_gather_object(
            records,
            {
                "rank": rank,
                "admission": admission,
                "reference_errors": errors,
                "marker_errors": markers,
                "replay": replay,
                "guards": True,
            },
        )
        result.update(
            phase="complete",
            correctness_passed=True,
            statistics=stats,
            ranks=records,
            local_gain_screen_passed=performance_gate(stats, 1.01),
            stable_gain_accepted=False,
        )


def main():
    args = parse_args()
    measurement = measurement_config(
        args.tokens, args.rounds, args.warmup, args.replays, 4, args.generations
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() != 8:
        raise ValueError("medium fused tail requires one TP8 group")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "schema": "medium-fused-tail-v1",
        "measurement": measurement,
        "implementation": args.implementation,
        "bt_tuning": BT_TUNING,
        "reference_bounds": {
            "atol": REFERENCE_ATOL,
            "rtol": REFERENCE_RTOL,
            "relative_l2": REFERENCE_L2,
        },
        "correctness_passed": False,
        "phase": "prepare",
    }
    try:
        run(args, result, device)
    except Exception as exc:
        result["failure_type"] = type(exc).__name__
        raise
    finally:
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
