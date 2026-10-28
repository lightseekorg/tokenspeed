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

"""Reproduce prefill DP skew using real C++ scheduler plans.

Model mode reports estimates under an explicit cost model, not GPU speedups.
CUDA mode runs a K3-shaped explicit-QKV MLA component with a synchronization
per DP round. It excludes model projections, KDA, MoE, cache transfers and HTTP.
Run CUDA mode with torchrun, one process per exclusively allocated GPU.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from tokenspeed_scheduler import (
    AttnKind,
    CacheGroupConfig,
    ExecutionEvent,
    ForwardEvent,
    PrefillComputeConfig,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def workloads(world: int, repeats: int):
    # Every rank gets the same total work, with different FIFO order. This
    # isolates per-round imbalance from unequal work assigned to a rank.
    cases = {
        "uniform": [(0, 8192)] * 4,
        "mixed_context": [(0, 8192), (0, 32768), (0, 16384), (0, 65536)],
        "prefix_skew": [(0, 8192), (65536, 8192), (4096, 8192), (32768, 8192)],
    }
    return {
        name: [
            (items[rank % len(items) :] + items[: rank % len(items)]) * repeats
            for rank in range(world)
        ]
        for name, items in cases.items()
    }


def submit(scheduler, request_id, tokens):
    request = RequestSpec()
    request.request_id = request_id
    request.tokens = tokens
    scheduler.submit_requests([request])


def complete(scheduler, request_id, finish):
    done = ForwardEvent.ExtendResult()
    done.request_id = request_id
    done.tokens = [2147483646] if finish else []
    scheduler.advance(ExecutionEvent().add_event(done))
    if finish:
        event = ForwardEvent.Finish()
        event.request_id = request_id
        scheduler.advance(ExecutionEvent().add_event(event))


def plans(queue, compute, token_limit):
    """Build scheduler plans after warming the queue's cached prefixes.

    Args:
        queue: Ordered (cached_prefix_tokens, new_tokens) request lengths.
        compute: PrefillComputeConfig, or None for token-only scheduling.
        token_limit: Maximum input tokens per scheduler plan.

    Returns:
        A tuple (schedule, planning_ns). The schedule contains the
        (query_tokens, prefix_tokens) pair for each actual scheduled forward
        after prefix warming. planning_ns is the total scheduler planning
        time in nanoseconds, excluding prefix warming and completion handling.
    """
    cfg = SchedulerConfig()
    cfg.prefix_granularity = 128
    cfg.max_scheduled_tokens = token_limit
    cfg.max_batch_size = 1
    cfg.num_device_pages = 1 + sum(c + q for c, q in queue) // 128 + 128
    cfg.num_snapshot_pages = 1
    cfg.disable_l2_cache = True
    cfg.prefill_compute = compute
    cfg.cache_groups = [
        CacheGroupConfig("mla", 128, cfg.num_device_pages, AttnKind.Full),
        CacheGroupConfig("kda", 0, cfg.num_device_pages, AttnKind.State),
    ]
    scheduler = Scheduler(cfg)
    for i, (c, _) in enumerate(queue):
        if c:
            rid = f"seed-{i}"
            submit(scheduler, rid, list(range(i * 100000, i * 100000 + c)))
            end = 0
            while end < c:
                batch = scheduler.next_execution_plan().forward[0]
                end = batch.extend_prefix_lens[0] + batch.input_lengths[0]
                complete(scheduler, rid, end == c)
    for i, (c, q) in enumerate(queue):
        submit(scheduler, str(i), list(range(i * 100000, i * 100000 + c + q)))
    result = []
    completed = 0
    scheduled_tokens = 0
    planning_ns = 0
    while completed < len(queue):
        started = time.perf_counter_ns()
        plan = scheduler.next_execution_plan()
        planning_ns += time.perf_counter_ns() - started
        if plan.aborts or len(plan.forward) != 1:
            raise RuntimeError(f"No progress: {plan.aborts}")
        batch = plan.forward[0]
        (q,) = batch.input_lengths
        (c,) = batch.extend_prefix_lens
        (rid,) = batch.request_ids
        expected_c, expected_q = queue[int(rid)]
        if c < expected_c or c + q > expected_c + expected_q:
            raise RuntimeError("Unexpected prefix miss or duplicate scheduled tokens")
        result.append((q, c))
        scheduled_tokens += q
        done = c + q == expected_c + expected_q
        complete(scheduler, rid, done)
        completed += int(done)
    if scheduled_tokens != sum(q for _, q in queue):
        raise RuntimeError(
            "Baseline and candidate must compute exactly the same token workload"
        )
    return result, planning_ns


def summarize(rank_costs):
    rounds = list(itertools.zip_longest(*rank_costs, fillvalue=0.0))
    critical = sum(max(row) for row in rounds)
    useful = sum(sum(row) for row in rounds)
    world = len(rank_costs)
    return {
        "rounds": len(rounds),
        "critical_path_ms": critical / 1e6,
        "compute_utilization": useful / (world * critical),
        "mean_round_max_over_mean": statistics.mean(
            max(row) * world / sum(row) for row in rounds
        ),
    }


def nonnegative_fit(samples):
    x = np.array([[q, q * (c + (q + 1) / 2), 1] for q, c, _ in samples], dtype=float)
    y = np.array([ns for _, _, ns in samples])
    scale = np.linalg.norm(x, axis=0)
    normalized = x / scale
    candidates = []
    for mask in range(1, 8):
        columns = [i for i in range(3) if mask & (1 << i)]
        weights = np.linalg.lstsq(normalized[:, columns], y, rcond=None)[0]
        if np.all(weights >= 0):
            fit = np.zeros(3)
            fit[columns] = weights / scale[columns]
            candidates.append((float(np.linalg.norm(x @ fit - y)), fit))
    _, weights = min(candidates, key=lambda item: item[0])
    if weights[0] == 0 and weights[1] == 0:
        raise RuntimeError(
            "Calibration has no length-dependent cost; inspect the samples"
        )
    error = float(np.median(np.abs(x @ weights - y) / y))
    return weights.tolist(), error


class CUDAAttention:
    def __init__(self, max_tokens, heads, qk_dim, v_dim, device):
        import torch
        from tokenspeed_kernel.ops.attention.mla.tokenspeed_mla import (
            tokenspeed_mla_prefill,
        )

        self.torch = torch
        self.kernel = tokenspeed_mla_prefill
        self.q = torch.randn(
            (max_tokens, heads, qk_dim), device=device, dtype=torch.bfloat16
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn(
            (max_tokens, heads, v_dim), device=device, dtype=torch.bfloat16
        )
        self.scale = qk_dim**-0.5
        self.metadata = {}

    def prepare(self, q, c):
        if (q, c) not in self.metadata:
            self.metadata[q, c] = tuple(
                self.torch.tensor(values, device=self.q.device, dtype=self.torch.int32)
                for values in ([c + q], [0, c + q], [0, q])
            )

    def run(self, q, c):
        lengths, cu_k, cu_q = self.metadata[q, c]
        return self.kernel(
            self.q[c : c + q],
            self.k[: c + q],
            self.v[: c + q],
            lengths,
            cu_k,
            c + q,
            1,
            self.scale,
            is_causal=True,
            return_lse=False,
            cum_seq_lens_q=cu_q,
            max_seq_len_q=q,
            enable_pdl=False,
            out=None,
        )

    def measure(self, q, c, repetitions):
        self.prepare(q, c)
        for _ in range(3):
            self.run(q, c)
        self.torch.cuda.synchronize()
        begin = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        values = []
        for _ in range(repetitions):
            begin.record()
            self.run(q, c)
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end) * 1e6)
        return statistics.median(values)

    def check_chunk_equivalence(self):
        for q, c in ((512, 2048), (256, 2048), (256, 2304)):
            self.prepare(q, c)
        reference = self.run(512, 2048).clone()
        actual = self.torch.cat([self.run(256, 2048), self.run(256, 2304)])
        self.torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)


def run_cuda(attention, schedules, iterations):
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    local = schedules[rank]
    rounds = max(map(len, schedules))
    for q, c in set(local):
        attention.prepare(q, c)
        attention.run(q, c)
    torch.cuda.synchronize()
    dist.barrier()
    dummy = torch.zeros(1, device=attention.q.device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    trials = []
    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        costs = []
        for step in range(rounds):
            begin.record()
            if step < len(local):
                attention.run(*local[step])
            end.record()
            # One common synchronization per DP round, including idle ranks.
            dist.all_reduce(dummy)
            torch.cuda.synchronize()
            costs.append(begin.elapsed_time(end) * 1e6 if step < len(local) else 0.0)
        elapsed = time.perf_counter_ns() - start
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {"elapsed_ns": elapsed, "costs": costs})
        summary = summarize([row["costs"] for row in gathered])
        summary["wall_ms"] = max(row["elapsed_ns"] for row in gathered) / 1e6
        trials.append(summary)
    return {key: statistics.median(row[key] for row in trials) for key in trials[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("model", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--token-limit", type=int, default=8192)
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--coefficients", type=float, nargs=3, default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--qk-dim", type=int, default=192)
    parser.add_argument("--v-dim", type=int, default=128)
    args = parser.parse_args()
    if min(args.ranks, args.repeats, args.token_limit, args.iterations) <= 0:
        parser.error("ranks, repeats, token-limit and iterations must be positive")
    rank = 0
    calibration = None
    attention = None
    if args.mode == "cuda":
        import torch
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        args.ranks = dist.get_world_size()
        torch.manual_seed(0)
        attention = CUDAAttention(73728, args.heads, args.qk_dim, args.v_dim, "cuda")
        attention.check_chunk_equivalence()
        samples = [
            (q, c, attention.measure(q, c, 10))
            for q in (512, 2048, 8192)
            for c in (0, 16384, 65536)
        ]
        gathered = [None] * args.ranks
        dist.all_gather_object(gathered, samples)
        weights, fit_error = nonnegative_fit(
            list(itertools.chain.from_iterable(gathered))
        )
        calibration = {
            "samples": gathered,
            "median_relative_error": fit_error,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        }
    else:
        if args.coefficients is None:
            parser.error(
                "model mode requires explicit --coefficients TOKEN_NS PAIR_NS REQUEST_NS"
            )
        weights = args.coefficients
    unit = PrefillComputeConfig(1, *weights)
    unit.validate()
    budget_ns = (
        args.budget_ms * 1e6
        if args.budget_ms is not None
        else unit.cost(args.token_limit, 2 * args.token_limit)
    )
    compute = PrefillComputeConfig(budget_ns, *weights)
    compute.validate()
    report = {
        "mode": args.mode,
        "scope": (
            "cost-model estimate"
            if attention is None
            else "synchronized MLA GPU component; not full-model serving"
        ),
        "ranks": args.ranks,
        "budget_ns": budget_ns,
        "coefficients_ns": weights,
        "token_limit": args.token_limit,
        "calibration": calibration,
        "cases": {},
    }
    for name, queues in workloads(args.ranks, args.repeats).items():
        result = {}
        for label, policy in (("token", None), ("compute", compute)):
            planned = [plans(queue, policy, args.token_limit) for queue in queues]
            schedules = [item[0] for item in planned]
            metrics = summarize(
                [[compute.cost(q, c) for q, c in schedule] for schedule in schedules]
            )
            metrics["planner_us_per_round"] = (
                sum(item[1] for item in planned) / sum(map(len, schedules)) / 1000
            )
            if attention is not None:
                metrics["gpu"] = run_cuda(attention, schedules, args.iterations)
            result[label] = metrics
        result["modeled_speedup"] = (
            result["token"]["critical_path_ms"] / result["compute"]["critical_path_ms"]
        )
        if attention is not None:
            result["gpu_component_speedup"] = (
                result["token"]["gpu"]["wall_ms"] / result["compute"]["gpu"]["wall_ms"]
            )
        report["cases"][name] = result
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if attention is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
