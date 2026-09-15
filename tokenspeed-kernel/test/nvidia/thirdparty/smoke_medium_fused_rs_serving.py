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

"""Checkpoint-free TP8 live-pointer smoke for the medium serving adapter.

Example arguments: --tokens 256 832 1024 --generations 3
--rank-skew-cycles 10000 --output result.json. A 128-generation run can name
every serving-profile bucket. This tests two chained layers, two independent
graphs per bucket, real producer out=, live pointer/content A/B/A, rank skew,
guarded workspace/output reuse and pre-launch rejection. It neither measures
tail latency nor qualifies model quality or serving TTFT.
"""

import argparse
import json
import os
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from fused_rs_up_ag_reference import equal_outputs, reference_errors
from test_fused_rs_up_projection_serving import check_all_ranks
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving import (
    MediumFusedRsUpProjectionServing,
)
from tokenspeed_kernel.ops.communication.medium_fused_rs_up_projection_serving_config import (
    MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS,
    medium_fused_rs_serving_config,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import (
    SharedRsWorkspace,
    _vote,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    allocate_symmetric_up_projection_output,
)
from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_shared_down_projection


def validate_options(tokens, generations, rank_skew_cycles):
    if (
        len(tokens) < 2
        or len(set(tokens)) != len(tokens)
        or any(
            type(m) is not int or m not in MEDIUM_FUSED_RS_SERVING_CANDIDATE_TOKENS
            for m in tokens
        )
    ):
        raise ValueError("name at least two distinct supported medium serving buckets")
    if type(generations) is not int or generations < 3:
        raise ValueError("at least three generations are required for A/B/A")
    if type(rank_skew_cycles) is not int or rank_skew_cycles < 0:
        raise ValueError("rank skew cycles must be a nonnegative integer")


def make_operands(m, slot, device, layers=2):
    replicated = torch.Generator(device=device).manual_seed(9231 + m + 101 * slot)
    local = torch.Generator(device=device).manual_seed(
        18001 + m + 103 * slot + 997 * dist.get_rank()
    )
    latent = [
        torch.randn(m, 3584, device=device, dtype=torch.bfloat16, generator=replicated)
        for _ in range(layers)
    ]
    residual = (
        torch.randn(m, 7168, device=device, dtype=torch.bfloat16, generator=replicated)
        * 0.02
    )
    producers = [
        (
            torch.randn(m, 768, device=device, dtype=torch.bfloat16, generator=local)
            * 0.03,
            torch.randn(7168, 768, device=device, dtype=torch.bfloat16, generator=local)
            * 0.03,
        )
        for _ in range(layers)
    ]
    return {"latent": latent, "residual": residual, "producers": producers}


def run_layers(operands, adapters, weights, snapshot_outputs=False):
    prefix, results = operands["residual"], []
    m = prefix.shape[0]
    for layer, adapter in enumerate(adapters):
        shared = kimi3_shared_down_projection(
            *operands["producers"][layer], out=adapter.input_view(m), solution="torch"
        )
        prefix = adapter(operands["latent"][layer], weights[layer], prefix, shared)
        # Correctness-only snapshots preserve intermediates across pool reuse.
        # They are not part of the serving implementation or a performance run.
        results.append(prefix.clone() if snapshot_outputs else prefix)
    return results


def old_reference(latent, weight, residual, producer):
    shared = kimi3_shared_down_projection(
        *producer, out=torch.empty_like(residual), solution="torch"
    )
    owner = slice(dist.get_rank() * 896, (dist.get_rank() + 1) * 896)
    shared[:, owner].add_(residual[:, owner])
    shared[:, owner].addmm_(latent, weight.t())
    dist.all_reduce(shared)
    return shared


def numerical_checks(actual, operands, weights, label):
    records, prefix = [], operands["residual"]
    for layer in range(len(weights)):
        reference = old_reference(
            operands["latent"][layer],
            weights[layer],
            prefix,
            operands["producers"][layer],
        )
        error = reference_errors(actual[layer], reference)
        check_all_ranks(error["passed"], label + "-old-semantic-reference")
        records.append({"layer": layer, **error})
        # Test each layer against its actual incoming residual; graph equality
        # independently verifies the complete two-layer chain end to end.
        prefix = actual[layer]
    return records


def update_inputs(operands, saved, phase):
    for tensor, source in zip(operands["latent"], saved["latent"]):
        tensor.copy_(source)
        if phase:
            tensor.neg_()
    operands["residual"].copy_(saved["residual"])
    if phase:
        operands["residual"].add_(0.125)
    for producer, activation in zip(operands["producers"], saved["activations"]):
        producer[0].copy_(activation)
        if phase:
            producer[0].neg_()


def make_case(
    m, slot, adapters, weights, warmup, stream, device, snapshot_outputs=False
):
    operands = make_operands(m, slot, device, len(adapters))
    warm_latent = {tensor.data_ptr() for tensor in warmup["latent"]}
    warm_residual = {warmup["residual"].data_ptr()}
    check_all_ranks(
        all(tensor.data_ptr() not in warm_latent for tensor in operands["latent"]),
        "live latent aliases warmup",
    )
    check_all_ranks(
        operands["residual"].data_ptr() not in warm_residual,
        "live residual aliases warmup",
    )
    check_all_ranks(
        adapters[0].output.tensor.data_ptr() not in warm_residual,
        "chained residual aliases warmup",
    )
    saved = {
        "latent": [x.clone() for x in operands["latent"]],
        "residual": operands["residual"].clone(),
        "activations": [producer[0].clone() for producer in operands["producers"]],
    }
    expected, errors = [], []
    for phase in (0, 1):
        update_inputs(operands, saved, phase)
        observed = [
            x.clone() for x in run_layers(operands, adapters, weights, snapshot_outputs)
        ]
        errors.append(
            numerical_checks(observed, operands, weights, "phase-" + str(phase))
        )
        expected.append(observed)
    check_all_ranks(
        all(not torch.equal(a, b) for a, b in zip(expected[0], expected[1])),
        "changed inputs did not change layer outputs",
    )
    update_inputs(operands, saved, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = run_layers(operands, adapters, weights, snapshot_outputs)
    graph.replay()
    check_all_ranks(
        equal_outputs(outputs, expected[0]), "capture used stale warmup pointers"
    )
    return {
        "m": m,
        "slot": slot,
        "operands": operands,
        "saved": saved,
        "graph": graph,
        "outputs": outputs,
        "expected": expected,
        "errors": errors,
    }


def rejection_checks(workspace, adapters, weights, operands, m, stream):
    fresh = type(adapters[0])(workspace, adapters[0].output.tensor.shape[0])
    rejected_capture = False
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            fresh(
                operands["latent"][0],
                weights[0],
                operands["residual"],
                fresh.input_view(m),
            )
    except RuntimeError as exc:
        rejected_capture = "compilation must finish during warmup" in str(exc)
    check_all_ranks(
        rejected_capture and not fresh._plans,
        "capture without warmup was not rejected before compilation",
    )
    # Every rank changes storage identity, while retaining bitwise weight data.
    changed_weight = weights[0].clone()
    rejected_weight = False
    try:
        adapters[0](
            operands["latent"][0],
            changed_weight,
            operands["residual"],
            adapters[0].input_view(m),
        )
    except (ValueError, RuntimeError) as exc:
        rejected_weight = "weight changed" in str(exc)
    check_all_ranks(
        rejected_weight, "changed weight identity was not collectively rejected"
    )
    return {
        "capture_without_warmup_rejected": True,
        "changed_weight_identity_rejected": True,
    }


def guard_checks(workspace, guards, capacity):
    return bool(torch.all(workspace.state.comm_buff[capacity:] == 53.0).item()) and all(
        bool(torch.all(guard == 61.0).item()) for guard in guards
    )


def run(args, device, adapter_type, profile, output_pool_type=None):
    capacity = max(args.tokens)
    workspace = SharedRsWorkspace.allocate(dist.group.WORLD, capacity + 32, device)
    workspace.state.comm_buff[capacity:].fill_(53.0)
    pooled = output_pool_type is not None
    layers = 4 if pooled else 2
    pool = output_pool_type(workspace, capacity) if pooled else None
    adapters = (
        [pool.bind_layer(i) for i in range(layers)]
        if pooled
        else [adapter_type(workspace, capacity) for _ in range(layers)]
    )
    owners, guards = [], []
    # Add a physical guard after the logical output without modifying adapter
    # code or its profile. All descriptors still address the exact logical view.
    for adapter in adapters[:2]:
        owner = allocate_symmetric_up_projection_output(
            dist.group.WORLD, capacity + 32, device=device
        )
        adapter.output = replace(owner, tensor=owner.tensor[:capacity])
        owner.tensor[capacity:].fill_(61.0)
        owners.append(owner)
        guards.append(owner.tensor[capacity:])
    if pooled:
        # Retain two guarded physical owners, and use the real pool selector
        # to bind four independent launch caches to these alternating slots.
        pool.outputs = tuple(adapter.output for adapter in adapters[:2])
        adapters = [pool.bind_layer(i) for i in range(layers)]
        check_all_ranks(
            adapters[0].output is adapters[2].output
            and adapters[1].output is adapters[3].output,
            "parity output sharing missing",
        )
    check_all_ranks(
        adapters[0].output.tensor.data_ptr() != adapters[1].output.tensor.data_ptr(),
        "layer outputs alias",
    )
    local = torch.Generator(device=device).manual_seed(5201 + 997 * dist.get_rank())
    weights = [
        torch.randn(896, 3584, device=device, dtype=torch.bfloat16, generator=local)
        * 0.015
        for _ in range(layers)
    ]
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    warmups, cases = {}, []
    with torch.cuda.stream(stream):
        for m in args.tokens:
            warmup = make_operands(m, 99, device, layers)
            warmups[m] = warmup
            # Warm each layer using independent warmup residual, so even the
            # second layer's real chained residual has a different pointer.
            for layer, adapter in enumerate(adapters):
                shared = kimi3_shared_down_projection(
                    *warmup["producers"][layer],
                    out=adapter.input_view(m),
                    solution="torch"
                )
                adapter(
                    warmup["latent"][layer], weights[layer], warmup["residual"], shared
                )
            for slot in (0, 1):
                cases.append(
                    make_case(
                        m, slot, adapters, weights, warmup, stream, device, pooled
                    )
                )
        rejections = rejection_checks(
            workspace,
            adapters,
            weights,
            warmups[args.tokens[0]],
            args.tokens[0],
            stream,
        )
        flags = []
        sequence = list(range(len(cases))) + list(reversed(range(len(cases)))) + [0]
        for generation in range(args.generations):
            phase = generation % 2
            for index in sequence:
                case = cases[index]
                update_inputs(case["operands"], case["saved"], phase)
                if args.rank_skew_cycles and dist.get_rank() == generation % 8:
                    torch.cuda._sleep(args.rank_skew_cycles)
                case["graph"].replay()
                # Materialize a boolean before reusing persistent layer storage;
                # do not compare old aliased outputs after another graph runs.
                for actual, expected in zip(case["outputs"], case["expected"][phase]):
                    flags.append(torch.all(actual == expected))
            check_all_ranks(
                guard_checks(workspace, guards, capacity),
                "workspace/output guard changed",
            )
        # End on restored A even when the requested generation count is even.
        case = cases[0]
        update_inputs(case["operands"], case["saved"], 0)
        case["graph"].replay()
        check_all_ranks(
            equal_outputs(case["outputs"], case["expected"][0]),
            "final restored A mismatch",
        )
        combined = torch.stack(flags)
        dist.all_reduce(combined, op=dist.ReduceOp.MIN)
        check_all_ranks(
            bool(combined.all().item()),
            "live-pointer/content or output-lifetime replay mismatch",
        )
        return {
            "rank": dist.get_rank(),
            "passed": True,
            "tokens": args.tokens,
            "layers": layers,
            "pooled_outputs": pooled,
            "correctness_only_intermediate_snapshots": pooled,
            "graph_slots_per_bucket": 2,
            "generations": args.generations,
            "checks_per_rank": len(flags),
            "all_replay_flags": combined.cpu().tolist(),
            "real_shared_producer_out": True,
            "warmup_and_graph_operand_pointers_distinct": True,
            "two_layers_share_raw_workspace": True,
            "per_layer_outputs_survive_next_layer_workspace_reuse": not pooled,
            "guards": True,
            "rank_skew": args.rank_skew_cycles > 0,
            "rejections": rejections,
            "cases": [
                {"m": case["m"], "slot": case["slot"], "numerical": case["errors"]}
                for case in cases
            ],
            "profile": {str(m): asdict(profile(m)) for m in args.tokens},
            "admission": {
                str(m): adapters[0]._plans[m].capacity_records for m in args.tokens
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--rank-skew-cycles", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_options(args.tokens, args.generations, args.rank_skew_cycles)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() != 8:
        raise ValueError("medium serving adapter acceptance requires TP8")
    _vote(
        dist.group.WORLD,
        ("medium-serving-smoke", args.tokens, args.generations, args.rank_skew_cycles),
        None,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    record = {
        "schema": "medium-fused-serving-adapter-v1",
        "passed": False,
        "full_model_or_ttft_qualified": False,
        "performance_measured": False,
    }
    try:
        local = run(
            args,
            torch.device("cuda", torch.cuda.current_device()),
            MediumFusedRsUpProjectionServing,
            medium_fused_rs_serving_config,
        )
        ranks = [None] * 8
        dist.all_gather_object(ranks, local)
        record.update(passed=True, ranks=ranks)
    except Exception as exc:
        record["failure_type"] = type(exc).__name__
        raise
    finally:
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
