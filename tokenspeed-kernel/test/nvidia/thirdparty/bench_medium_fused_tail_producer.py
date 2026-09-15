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

"""Real shared producer plus medium BT tail: original AR2, fused copied/direct.

Use exactly the CLI tuning/sampling arguments accepted by bench_medium_fused_tail.
All three pairwise comparisons include BF16 [M,768] x [7168,768] down projection
for each of four independent layers. Copied/direct share the identical compiled
plan and execute sequentially; untimed correctness snapshots precede every reuse.
Three/five rounds and three generations are smoke only. Formal sampling uses
31 rounds, eight warmups, 20 replays and 128 changed-input generations; two
independent batches and a trace remain separate acceptance requirements.
"""

import json
import os
from dataclasses import asdict, replace
from datetime import timedelta

import torch
import torch.distributed as dist
from bench_medium_fused_tail import (
    TailPair,
    admit,
    checked,
    equal_outputs,
    independent_fp32_reference,
    make_tuning,
    parse_args,
    reference_errors,
    time_pair,
)
from fused_rs_up_ag_inputs import _make_layer_inputs, _semantic_reference
from fused_rs_up_ag_statistics import performance_gate
from medium_fused_tail_contract import (
    BT_TUNING,
    REFERENCE_ATOL,
    REFERENCE_L2,
    REFERENCE_RTOL,
)
from medium_fused_tail_producer_contract import (
    ARMS,
    COMPARISONS,
    producer_measurement_config,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import _vote
from tokenspeed_kernel.ops.communication.multimem import (
    multimem_all_reduce_staged,
    multimem_stage,
)
from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_shared_down_projection


class ProducerTailPair(TailPair):
    """Same BT/fused plan, independent producer data and guarded ordinary outputs."""

    def __init__(self, layers, args, tuning, device):
        super().__init__(layers, args, tuning, device)
        self.producers, self.ordinary_owners, self.ordinary_guards = [], [], []
        for slot, layer in enumerate(layers):
            generator = torch.Generator(device=device).manual_seed(
                87000 + 101 * self.rank + 997 * slot + self.m
            )
            activation = (
                torch.randn(
                    self.m,
                    768,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                * 0.03
            ).contiguous()
            weight = (
                torch.randn(
                    7168, 768, device=device, dtype=torch.bfloat16, generator=generator
                )
                * 0.03
            ).contiguous()
            self.producers.append((activation, weight))
            ordinary = torch.empty(
                self.m + 32, 7168, device=device, dtype=torch.bfloat16
            )
            ordinary[self.m :].fill_(47.0)
            self.ordinary_owners.append(ordinary)
            self.ordinary_guards.append(ordinary[self.m :])
            layer.shared_partial = ordinary[: self.m]
        checked(
            len({activation.data_ptr() for activation, _ in self.producers}) == 4,
            "independent-producer-activations",
        )
        checked(
            len({weight.data_ptr() for _, weight in self.producers}) == 4,
            "independent-producer-weights",
        )

    def run_mode(self, mode):
        """Execute producer then one complete layer before proceeding to the next."""
        self._check_stream()
        if mode not in ARMS:
            raise ValueError("producer mode must be baseline, copied or direct")
        outputs = []
        for slot, (layer, plan) in enumerate(zip(self.layers, self.plans)):
            destination = plan.input_view if mode == "direct" else layer.shared_partial
            partial = kimi3_shared_down_projection(
                *self.producers[slot], out=destination, solution="torch"
            )
            if partial.data_ptr() != destination.data_ptr():
                raise RuntimeError("producer did not honor its exact out= destination")
            if mode == "baseline":
                staged = multimem_stage(partial, self.group_name, 1024)
                if staged is None:
                    raise RuntimeError("original shared multimem stage unavailable")
            elif mode == "copied":
                plan.stage(partial)
            latent = self.bt(
                layer.gemm2_output,
                layer.expert_weights,
                layer.expanded_idx,
                layer.gamma,
            )
            if mode == "baseline":
                start = self.rank * 896
                owner = staged[:, start : start + 896]
                owner.add_(layer.prefix[:, start : start + 896])
                owner.addmm_(latent, layer.up_weight.t())
                outputs.append(
                    multimem_all_reduce_staged(staged, self.group_name).clone()
                )
            else:
                if latent.data_ptr() != plan.inputs[0].data_ptr():
                    raise RuntimeError("BT changed its bound latent output")
                outputs.append(plan())
        return outputs

    def guard_checks(self):
        return super().guard_checks() and all(
            bool(torch.all(guard == 47.0).item()) for guard in self.ordinary_guards
        )


def snapshot(values):
    """Snapshot persistent results before another arm can overwrite them."""
    return [value.clone() for value in values]


def check_producers(pair, label):
    """Require identical cuBLAS producer math for ordinary and symmetric outputs."""
    records = []
    for slot, (layer, plan, producer) in enumerate(
        zip(pair.layers, pair.plans, pair.producers)
    ):
        ordinary = kimi3_shared_down_projection(
            *producer, out=layer.shared_partial, solution="torch"
        )
        expected = ordinary.clone()
        symmetric = kimi3_shared_down_projection(
            *producer, out=plan.input_view, solution="torch"
        )
        checked(
            torch.equal(expected.view(torch.int16), symmetric.view(torch.int16)),
            label + "-producer-copied-direct",
        )
        reference = torch.mm(producer[0].float(), producer[1].float().t())
        error = reference_errors(expected, reference)
        checked(error["passed"], label + "-producer-fp32")
        records.append({"slot": slot, "copied_direct_bitwise": True, **error})
    checked(pair.guard_checks(), label + "-producer-guards")
    return records


def make_references(pair):
    """Independent complete-tail references include the actual producer output."""
    references = []
    for layer, producer in zip(pair.layers, pair.producers):
        partial = kimi3_shared_down_projection(
            *producer, out=layer.shared_partial, solution="torch"
        )
        reference_layer = replace(layer, shared_partial=partial)
        references.append(
            {
                "old_semantics": _semantic_reference(reference_layer, pair.rank),
                "fp32": independent_fp32_reference(reference_layer, pair.rank),
            }
        )
    return references


def check_references(observations, references, label):
    errors = []
    for name, values in observations.items():
        for slot, actual in enumerate(values):
            for reference_name, reference in references[slot].items():
                error = reference_errors(actual, reference)
                checked(error["passed"], label + "-" + name + "-" + reference_name)
                errors.append(
                    {"arm": name, "slot": slot, "reference": reference_name, **error}
                )
    checked(
        equal_outputs(observations["copied"], observations["direct"]),
        label + "-fused-copied-direct",
    )
    return errors


def capture(pair, stream, warmup):
    eager, graphs, outputs = {}, {}, {}
    for name in ARMS:
        eager[name] = snapshot(pair.run_mode(name))
        for _ in range(warmup):
            pair.run_mode(name)
        torch.cuda.synchronize(pair.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs[name] = pair.run_mode(name)
        graph.replay()
        observed = snapshot(outputs[name])
        checked(equal_outputs(observed, eager[name]), name + "-eager-graph")
        graphs[name] = graph
    checked(equal_outputs(eager["copied"], eager["direct"]), "captured-copied-direct")
    return eager, graphs, outputs


def replay_observations(pair, graphs, outputs, generation, skew_cycles):
    observed = {}
    for name, graph in graphs.items():
        if skew_cycles and pair.rank == generation % 8:
            torch.cuda._sleep(skew_cycles)
        graph.replay()
        observed[name] = snapshot(outputs[name])
    return observed


def save_inputs(pair):
    return [
        dict(
            activation=producer[0].clone(),
            weight=producer[1].clone(),
            gemm2_output=layer.gemm2_output.clone(),
            prefix=layer.prefix.clone(),
        )
        for layer, producer in zip(pair.layers, pair.producers)
    ]


def restore_inputs(pair, saved):
    for slot, (layer, producer) in enumerate(zip(pair.layers, pair.producers)):
        producer[0].copy_(saved[slot]["activation"])
        producer[1].copy_(saved[slot]["weight"])
        layer.gemm2_output.copy_(saved[slot]["gemm2_output"])
        layer.prefix.copy_(saved[slot]["prefix"])


def check_replay(pair, graphs, outputs, eager, generations, skew_cycles):
    saved = save_inputs(pair)
    producer_errors, changed_reference = [], []
    changed_checks = 0
    for generation in range(generations):
        restore_inputs(pair, saved)
        if generation % 2:
            for layer, producer in zip(pair.layers, pair.producers):
                producer[0].neg_()
                producer[1].add_(0.00390625)
                layer.gemm2_output.neg_()
                layer.prefix.add_(0.125)
        if generation == 1:
            producer_errors = check_producers(pair, "changed-B")
        observed = replay_observations(pair, graphs, outputs, generation, skew_cycles)
        checked(
            equal_outputs(observed["copied"], observed["direct"]),
            "replayed-copied-direct",
        )
        for name in ARMS:
            if generation % 2 == 0:
                checked(
                    equal_outputs(observed[name], eager[name]), name + "-restored-A"
                )
            else:
                changed = all(
                    not torch.equal(a, b) for a, b in zip(observed[name], eager[name])
                )
                checked(changed, name + "-changed-B")
                changed_checks += int(changed)
        if generation == 1:
            changed_reference = check_references(
                observed, make_references(pair), "changed-B"
            )
            for name in ARMS:
                checked(
                    equal_outputs(snapshot(pair.run_mode(name)), observed[name]),
                    name + "-changed-eager-graph",
                )
        checked(pair.guard_checks(), "replay-guards")
    restore_inputs(pair, saved)
    final = replay_observations(pair, graphs, outputs, 0, 0)
    for name in ARMS:
        checked(equal_outputs(final[name], eager[name]), name + "-final-A")
    return {
        "generations": generations,
        "changed_checks": changed_checks,
        "ABA": True,
        "rank_skew": skew_cycles > 0,
        "producer_errors": producer_errors,
        "changed_reference_errors": changed_reference,
    }


def check_markers(pair, graphs, outputs):
    saved = save_inputs(pair)
    rows = torch.arange(pair.m, device=pair.device, dtype=torch.float32)
    columns = torch.arange(7168, device=pair.device, dtype=torch.float32)
    records = []
    for case in ("zero", "rank_row_column", "positive_negative"):
        for slot, (layer, producer) in enumerate(zip(pair.layers, pair.producers)):
            producer[0].zero_()
            producer[1].zero_()
            layer.gemm2_output.zero_()
            layer.prefix.zero_()
            if case != "zero":
                # One nonzero inner-product coordinate makes rank/row/column
                # ownership independently visible without GEMM accumulation noise.
                rank_factor = (
                    (pair.rank + 1) / 8
                    if case == "rank_row_column"
                    else (1 if pair.rank % 2 else -1)
                )
                producer[0][:, 0].copy_(rank_factor * (1 + (rows % 8) / 16))
                producer[1][:, 0].copy_((1 + (columns % 8) / 16) / 8)
                layer.prefix.copy_(
                    (columns[None, :] // 896) / 32
                    + (rows[:, None] % 8) / 128
                    + slot / 64
                )
        producer_errors = check_producers(pair, case)
        observed = replay_observations(pair, graphs, outputs, 0, 0)
        errors = check_references(observed, make_references(pair), case)
        checked(pair.guard_checks(), case + "-marker-guards")
        records.append(
            {
                "case": case,
                "producer_errors": producer_errors,
                "reference_errors": errors,
            }
        )
    restore_inputs(pair, saved)
    return records


def run(args, result, device):
    tuning, error = None, None
    try:
        tuning = make_tuning(args)
    except Exception as exc:
        error = type(exc).__name__
    _vote(
        dist.group.WORLD,
        ("medium-producer-tuning", None if tuning is None else asdict(tuning)),
        error,
    )
    result["tuning"] = asdict(tuning)
    _vote(
        dist.group.WORLD,
        (
            "medium-producer-config",
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
        pair = ProducerTailPair(layers, args, tuning, device)
        result["phase"] = "binary_admission"
        admission = admit(pair, args.implementation)
        result["phase"] = "correctness"
        producers = check_producers(pair, "initial-A")
        eager, graphs, outputs = capture(pair, stream, args.warmup)
        reference = check_references(eager, make_references(pair), "initial-A")
        replay = check_replay(
            pair, graphs, outputs, eager, args.generations, args.rank_skew_cycles
        )
        markers = check_markers(pair, graphs, outputs)
        result["phase"] = "performance"
        comparisons = {}
        for baseline, candidate in COMPARISONS:
            comparisons[baseline + "_to_" + candidate] = time_pair(
                {"baseline": graphs[baseline], "candidate": graphs[candidate]},
                args,
                device,
            )
        checked(pair.guard_checks(), "final-producer-guards")
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(
            ranks,
            {
                "rank": rank,
                "admission": admission,
                "producer_errors": producers,
                "reference_errors": reference,
                "replay": replay,
                "markers": markers,
                "guards": True,
            },
        )
        result.update(
            phase="complete",
            correctness_passed=True,
            comparisons=comparisons,
            ranks=ranks,
            direct_gain_screen_passed=performance_gate(
                comparisons["copied_to_direct"], 1.01
            ),
            overall_gain_screen_passed=performance_gate(
                comparisons["baseline_to_direct"], 1.01
            ),
            stable_gain_accepted=False,
        )


def main():
    args = parse_args()
    measurement = producer_measurement_config(
        args.tokens, args.rounds, args.warmup, args.replays, args.generations
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    if dist.get_world_size() != 8:
        raise ValueError("producer-inclusive medium tail requires TP8")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "schema": "medium-fused-tail-producer-v1",
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
