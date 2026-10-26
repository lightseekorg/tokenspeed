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

"""EAGLE-Harness device policy: its rounds against EAGLE-4's decisions,
recorded on synthetic cost data and scripted rounds, and at its exact
boundaries (test_eagle_harness_round.py drives it through the drafter's round
protocol)."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import re
import struct
import sys
import time
from dataclasses import dataclass

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.drafter.eagle_harness import device_policy
from tokenspeed.runtime.execution.drafter.eagle_harness.config import (
    ChainWidthEntry,
    ChainWidthTable,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.cost import (
    BatchCost,
    IntervalCost,
    load_cost_data,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.device_policy import (
    DevicePolicy,
)

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the eagle_harness policy kernels are NVIDIA-only",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="needs an NVIDIA GPU",
)

_DEVICE = torch.device("cuda")
_FRONTIER_ROWS = 1024
# The cached length of frontier rows past the live ones (never read).
_STALE_LENGTH = 2**40 + 7
_TARGET_WIDTHS = [0, 7, 15, 31, 63]
_DRAFT_WIDTHS = [0, 1, 3, 7]
_CHAIN_DEPTH = 15
_CHAIN_LLM_WIDTHS = [0, 1, 3, 5, 7, 9, 11, 13, 15]


def _write_cost_dir(
    root: str,
    batch_sizes: tuple[int, ...],
    target_widths: list[int],
    draft_widths: list[int],
    *,
    bs_slope: float,
    cliff: float,
) -> None:
    """Three context intervals per batch size; cost grows with width, batch
    size and context, and at batch 4 and up jumps by ``cliff`` from width 6."""
    draft_slope = 0.01 if len(draft_widths) > 1 else 0.0
    for bs in batch_sizes:
        for model, widths, base, slope in (
            ("model_llm", target_widths, 4000.0, 0.012),
            ("model_slm", draft_widths, 170.0, draft_slope),
        ):
            path = os.path.join(root, "cost", f"bs_{bs}", model)
            os.makedirs(path, exist_ok=True)
            for interval in range(3):
                v0 = base * (1 + 0.15 * (bs - 1)) * (1 + 0.05 * interval)
                values = [
                    round(
                        v0
                        * (
                            1
                            + slope * w * (1 + bs_slope * (bs - 1))
                            + (cliff if w >= 5 and bs >= 4 else 0.0)
                        ),
                        4,
                    )
                    for w in widths
                ]
                data = {
                    "value": values,
                    "index": widths,
                    "ratio": [round(v / values[0], 4) for v in values],
                    "cache_length_step": 512,
                }
                with open(
                    os.path.join(path, f"model_interval_{interval}.json"), "w"
                ) as f:
                    json.dump(data, f)


def _rounds(
    seed: int, num_rounds: int, topk: int, depth: int, verify: int, max_count: int
):
    """Per round: context length, each depth's work deltas, node count and the
    ranked verify scores."""
    rng = random.Random(seed)
    for _ in range(num_rounds):
        cache_len = rng.randint(64, 1600)
        deltas = []
        for _ in range(depth):
            base = rng.uniform(0.5, 4.0)
            steps = sorted((rng.uniform(0, 0.15) for _ in range(topk)), reverse=True)
            values = [base]
            for step in steps:
                values.append(values[-1] + step)
            deltas.append(torch.tensor(values, dtype=torch.float32).tolist())
        draft_count = rng.randint(2, max_count)
        scores = sorted((rng.uniform(0.0, 1.0) for _ in range(verify)), reverse=True)
        yield cache_len, deltas, draft_count, scores


def _chain_rounds(seed: int, num_rounds: int):
    rng = random.Random(seed)
    for _ in range(num_rounds):
        cache_len = rng.randint(64, 1600)
        deltas = []
        for _ in range(_CHAIN_DEPTH):
            base = rng.uniform(0.5, 4.0)
            deltas.append([base, base + rng.uniform(0.0, 0.6)])
        yield cache_len, deltas


def _policy(
    cost_data: dict[int, BatchCost],
    *,
    topk: int,
    depth: int,
    verify: int,
    intensity: str = "high_intensity",
    moe: bool = False,
    exec_mode: int = 0,
    config: str = "/nonexistent",
    table: ChainWidthTable = (),
) -> DevicePolicy:
    return DevicePolicy(
        cost_data,
        verify_num=verify,
        node_top_k=topk,
        tree_depth=depth,
        compute_intensity=intensity,
        is_moe_target=moe,
        exec_mode=exec_mode,
        benchmark_config=config,
        file_group=None,
        chain_width_table=table,
        device=_DEVICE,
    )


@dataclass(frozen=True)
class _Scenario:
    """A recorded tree scenario: cost data (batch sizes 1, 2, 4 at a gentle
    slope, or 1, 2, 4, 8 steep, every tier), the policy and its rounds (every
    round but each fourth sampled when ``sampled``; benchmark batch 24)."""

    steep: bool
    topk: int
    depth: int
    verify: int
    max_count: int
    intensity: str
    moe: bool
    exec_mode: int
    sampled: bool


_HIGH = _Scenario(False, 4, 8, 16, 40, "high_intensity", False, 0, False)
_SCENARIOS = {
    "high": _HIGH,
    "low_moe": dataclasses.replace(_HIGH, intensity="low_intensity", moe=True),
    "exec_mode_1": dataclasses.replace(_HIGH, exec_mode=1),
    "sampled": dataclasses.replace(_HIGH, sampled=True),
    "k8_d6": dataclasses.replace(
        _HIGH, steep=True, topk=8, depth=6, verify=64, max_count=100
    ),
}

# EAGLE-4's decisions per (scenario, batch size) and round, as _Outcome.recorded orders them.
# fmt: off
_RECORDED = {
    ('high', 1): [
        (16, 1, 4, 16, 16), (32, 3, 4, 16, 16), (32, 3, 4, 16, 16), (32, 5, 4, 16, 16),
        (32, 5, 4, 16, 16), (32, 3, 4, 11, 8), (32, 3, 4, 16, 8), (32, 3, 4, 16, 16),
        (32, 7, -1, 16, 16), (32, 7, -1, 16, 16), (32, 3, 4, 10, 8), (32, 5, 4, 16, 8),
        (32, 3, 4, 16, 16), (32, 5, 4, 16, 16), (32, 7, -1, 16, 16), (32, 7, -1, 16, 16),
        (32, 3, 4, 16, 16), (32, 7, -1, 16, 16), (32, 7, -1, 10, 8), (32, 7, -1, 10, 8),
        (32, 7, -1, 9, 8), (32, 7, -1, 15, 8), (32, 3, 4, 16, 8), (32, 3, 4, 2, 1),
    ],
    ('high', 4): [
        (16, 1, 4, 14, 8), (16, 1, 4, 16, 8), (16, 3, 4, 16, 16), (16, 1, 4, 16, 16),
        (16, 1, 4, 16, 16), (16, 5, 4, 16, 16), (16, 1, 4, 12, 8), (16, 3, 4, 6, 1),
        (16, 5, 4, 15, 1), (16, 1, 4, 16, 8), (16, 1, 4, 16, 16), (16, 1, 4, 16, 16),
        (16, 1, 4, 16, 16), (16, 3, 4, 8, 8), (16, 7, -1, 3, 1), (16, 1, 4, 8, 1),
        (16, 3, 4, 16, 8), (16, 1, 4, 5, 1), (16, 1, 4, 16, 1), (16, 1, 4, 15, 8),
        (16, 5, 4, 16, 8), (16, 1, 4, 16, 16), (16, 5, 4, 16, 16), (16, 3, 4, 16, 16),
    ],
    ('low_moe', 1): [
        (16, 1, 4, 16, 16), (32, 3, 4, 16, 16), (32, 3, 4, 16, 16), (32, 5, 4, 16, 16),
        (32, 3, 4, 16, 16), (32, 3, 4, 11, 8), (32, 3, 4, 16, 8), (32, 3, 4, 16, 16),
        (32, 7, -1, 16, 16), (32, 7, -1, 16, 16), (32, 3, 4, 10, 8), (32, 5, 4, 16, 8),
        (32, 3, 4, 16, 16), (32, 5, 4, 16, 16), (32, 7, -1, 16, 16), (32, 5, 4, 16, 16),
        (32, 3, 4, 16, 16), (32, 3, 4, 16, 16), (32, 5, 4, 10, 8), (32, 7, -1, 10, 8),
        (32, 7, -1, 9, 8), (32, 7, -1, 15, 8), (32, 3, 4, 16, 8), (32, 3, 4, 2, 1),
    ],
    ('low_moe', 4): [
        (16, 1, 4, 14, 8), (16, 1, 4, 16, 8), (16, 3, 4, 16, 16), (16, 1, 4, 16, 16),
        (16, 1, 4, 16, 16), (16, 3, 4, 16, 16), (16, 1, 4, 12, 8), (16, 3, 4, 6, 1),
        (16, 5, 4, 15, 1), (16, 1, 4, 16, 8), (16, 1, 4, 16, 16), (16, 1, 4, 16, 16),
        (16, 1, 4, 16, 16), (16, 3, 4, 8, 8), (16, 3, 4, 3, 1), (16, 1, 4, 8, 1),
        (16, 3, 4, 16, 8), (16, 1, 4, 5, 1), (16, 1, 4, 16, 1), (16, 1, 4, 15, 8),
        (16, 5, 4, 16, 8), (16, 1, 4, 16, 16), (16, 5, 4, 16, 16), (16, 3, 4, 16, 16),
    ],
    ('exec_mode_1', 4): [
        (16, 1, 1, 14, 8), (16, 1, 1, 16, 8), (16, 1, 1, 16, 8), (16, 1, 1, 16, 8),
        (16, 1, 1, 16, 8), (16, 1, 1, 16, 8), (16, 1, 1, 12, 8), (16, 1, 1, 6, 8),
        (16, 1, 1, 15, 8), (16, 1, 1, 16, 8), (16, 1, 1, 16, 8), (16, 1, 1, 16, 8),
        (16, 1, 1, 16, 8), (16, 1, 1, 8, 8), (16, 1, 1, 3, 8), (16, 1, 1, 8, 8),
        (16, 1, 1, 16, 8), (16, 1, 1, 5, 8), (16, 1, 1, 16, 8), (16, 1, 1, 15, 8),
        (16, 1, 1, 16, 8), (16, 1, 1, 16, 8), (16, 1, 1, 16, 8), (16, 1, 1, 16, 8),
    ],
    ('exec_mode_1', 8): [
        (16, 1, 4, 16, 16), (16, 3, 4, 11, 8), (16, 1, 4, 16, 8), (16, 3, 4, 16, 16),
        (16, 5, 4, 16, 16), (16, 5, 4, 16, 16), (16, 1, 4, 16, 16), (16, 5, 4, 16, 16),
        (16, 1, 4, 16, 16), (16, 3, 4, 16, 16), (16, 1, 4, 2, 1), (16, 1, 4, 16, 1),
        (16, 5, 4, 16, 16), (16, 5, 4, 16, 16), (16, 3, 4, 16, 16), (16, 3, 4, 16, 16),
        (16, 1, 4, 7, 1), (16, 5, 4, 12, 1), (16, 1, 4, 2, 1), (16, 3, 4, 10, 1),
        (16, 3, 4, 11, 8), (16, 3, 4, 7, 1), (16, 1, 4, 16, 1), (16, 1, 4, 7, 1),
    ],
    ('sampled', 1): [
        (16, 7, -1, 16, 16), (32, 1, 4, 16, 16), (32, 3, 4, 16, 16), (32, 5, 4, 16, 16),
        (32, 3, 4, 16, 16), (32, 5, 4, 11, 8), (32, 3, 4, 16, 8), (32, 3, 4, 16, 16),
        (32, 3, 4, 16, 16), (32, 5, 4, 16, 16), (32, 7, -1, 10, 8), (32, 5, 4, 16, 8),
        (32, 3, 4, 16, 16), (32, 3, 4, 16, 16), (32, 5, 4, 16, 16), (32, 7, -1, 16, 16),
        (32, 7, -1, 16, 16), (32, 3, 4, 16, 16), (32, 5, 4, 10, 8), (32, 7, -1, 10, 8),
        (32, 7, -1, 9, 8), (32, 7, -1, 15, 8), (32, 7, -1, 16, 8), (32, 3, 4, 2, 1),
    ],
    ('sampled', 4): [
        (16, 7, -1, 14, 8), (16, 1, 4, 16, 8), (16, 1, 4, 16, 16), (16, 1, 4, 16, 16),
        (16, 3, 4, 16, 16), (16, 1, 4, 16, 16), (16, 3, 4, 12, 8), (16, 3, 4, 6, 1),
        (16, 1, 4, 15, 1), (16, 3, 4, 16, 8), (16, 1, 4, 16, 16), (16, 1, 4, 16, 16),
        (16, 1, 4, 16, 16), (16, 1, 4, 8, 8), (16, 3, 4, 3, 1), (16, 1, 4, 8, 1),
        (16, 5, 4, 16, 8), (16, 3, 4, 5, 1), (16, 1, 4, 16, 1), (16, 1, 4, 15, 8),
        (16, 1, 4, 16, 8), (16, 3, 4, 16, 16), (16, 1, 4, 16, 16), (16, 3, 4, 16, 16),
    ],
    ('k8_d6', 2): [
        (64, 5, -1, 61, 32), (64, 5, -1, 14, 8), (64, 5, -1, 64, 8), (64, 5, -1, 64, 64),
        (64, 5, -1, 10, 8), (64, 5, -1, 64, 8), (64, 1, 8, 25, 16), (64, 5, -1, 64, 16),
        (64, 3, 8, 64, 64), (64, 5, -1, 62, 32), (64, 5, -1, 38, 32), (64, 3, 8, 64, 32),
        (64, 5, -1, 64, 64), (64, 5, -1, 8, 8), (64, 5, -1, 64, 8), (64, 5, -1, 17, 16),
        (64, 5, -1, 60, 16), (64, 5, -1, 26, 16), (64, 5, -1, 24, 16), (64, 5, -1, 64, 16),
        (64, 5, -1, 17, 16), (64, 5, -1, 51, 16), (64, 5, -1, 64, 32), (64, 5, -1, 10, 8),
    ],
    ('k8_d6', 4): [
        (64, 5, -1, 64, 64), (64, 1, 4, 41, 32), (64, 1, 4, 41, 32), (64, 1, 4, 41, 32),
        (64, 1, 4, 23, 16), (64, 1, 4, 2, 1), (64, 1, 4, 41, 1), (64, 1, 4, 10, 8),
        (64, 1, 4, 26, 8), (64, 1, 4, 34, 16), (64, 1, 4, 41, 32), (64, 1, 4, 41, 32),
        (64, 1, 4, 21, 16), (64, 1, 4, 39, 16), (64, 1, 4, 41, 32), (64, 1, 4, 26, 16),
        (64, 1, 4, 41, 16), (32, 1, 4, 41, 32), (32, 1, 4, 33, 32), (32, 1, 4, 41, 32),
        (32, 1, 4, 41, 32), (32, 1, 4, 41, 32), (32, 1, 4, 41, 32), (32, 1, 4, 10, 8),
    ],
    ('k8_d6', 8): [
        (64, 1, 8, 41, 32), (64, 1, 4, 41, 32), (64, 1, 4, 41, 32), (64, 1, 4, 41, 32),
        (64, 1, 4, 41, 32), (64, 1, 4, 41, 32), (64, 1, 4, 41, 32), (64, 1, 4, 41, 32),
        (64, 1, 4, 41, 32), (64, 1, 4, 41, 32), (64, 1, 4, 22, 16), (64, 1, 4, 13, 8),
        (64, 1, 4, 41, 8), (64, 1, 4, 14, 8), (64, 1, 4, 41, 8), (64, 1, 4, 41, 32),
        (32, 1, 4, 41, 32), (32, 1, 4, 41, 32), (32, 1, 4, 41, 32), (32, 1, 4, 27, 16),
        (32, 1, 4, 41, 16), (32, 1, 4, 41, 32), (32, 1, 4, 10, 8), (32, 1, 4, 41, 8),
    ],
}
# Per batch size, per round: the depth EAGLE-4's chain-depth rule stopped the chain at.
_CHAIN_RECORDED = {
    1: [2, 2, 2, 4, 4, 6, 4, 2, 2, 2, 2, 8, 2, 8, 2, 6, 6, 12, 2, 2, 4, 4, 2, 4],
    2: [4, 8, 8, 4, 2, 4, 4, 2, 2, 2, 6, 2, 6, 4, 8, 4, 6, 4, 4, 2, 2, 6, 6, 8],
    4: [2, 2, 2, 2, 2, 4, 2, 6, 2, 4, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 6],
    8: [2, 2, 2, 4, 2, 2, 2, 4, 6, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 4, 4, 2, 4],
}
# fmt: on


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@dataclass
class _Round:
    """One round's inputs (the device outputs the policy reads, scripted)."""

    bs: int
    frontier_sum: int
    sampled: bool
    deltas: torch.Tensor  # fp32 [tree_depth, K + 1]: each depth's work deltas
    top_scores: torch.Tensor  # fp32 [rows, verify_num]: each row's ranked scores
    pool_size: int | None = None  # caps the ranked count (recorded rounds)


@dataclass
class _Outcome:
    past_verify_num: int
    widths: list[int]
    cumu_starts: list[int]
    work_sizes: list[int]
    go: list[int]
    last_depth: int
    stop_width: int
    cumu_count: int
    verify_count: int
    flat: int
    verify_num: int

    def recorded(self) -> tuple[int, int, int, int, int]:
        return (
            self.past_verify_num,
            self.last_depth,
            self.stop_width,
            self.verify_count,
            self.verify_num,
        )


def _slot_rows(policy: DevicePolicy) -> torch.Tensor:
    """A round's per-depth slots: widths, cumu_starts, work_sizes, go."""
    return torch.stack(
        [policy.widths, policy.cumu_starts, policy.work_sizes, policy.go]
    )


def _assert_round(policy: DevicePolicy, o: _Outcome) -> None:
    """A live round with every decision launched: the node pool holds the
    root's K children from node 1, then each drafted depth's K children per
    expanded node (1 to K of them) right after the previous depth's, up to
    the pool size; the pinned go mirror holds every depth's go flag; and
    deltas_depth names every depth whose slot asks for work deltas."""
    k = policy.node_top_k
    end = 1
    for d in range(o.last_depth + 1):
        assert o.cumu_starts[d] == end and 1 <= o.widths[d] <= k
        end += k * o.widths[d]
    assert o.cumu_count == end
    for d in range(1, policy.tree_depth + 1):
        assert policy.go_ready(d) == bool(o.go[d])
    assert all(policy.deltas_depth(d) for d, n in enumerate(o.work_sizes) if n)


class _DeviceDrafter:
    """Drives a DevicePolicy as the drafter does, launching every decision
    (no-ops past the stop), eagerly or from a graph captured once: inputs in
    persistent buffers, decide(d) before depth d drafts, each drafted depth's
    scripted work deltas queued when its slot asks for them."""

    def __init__(self, policy: DevicePolicy, max_bs: int) -> None:
        self.policy = policy
        k, verify_num = policy.node_top_k, policy.verify_num_limit
        self.bs = 0
        self.live_rows = torch.zeros((1,), dtype=torch.int32, device=_DEVICE)
        # Column 0 of a [rows, 2] buffer: a strided frontier, padded as the drafter's.
        rows = torch.zeros((_FRONTIER_ROWS, 2), dtype=torch.int64, device=_DEVICE)
        self._frontier_rows = rows
        self.frontier = rows[:, 0]
        # Splits each round's cached length sum over its live rows.
        self._split = random.Random(0)
        self.sampled = torch.zeros((1,), dtype=torch.bool, device=_DEVICE)
        self.deltas = torch.zeros((policy.tree_depth, k + 1), device=_DEVICE)
        self.work_deltas = torch.zeros((k + 1,), device=_DEVICE)
        # Exactly [max_bs, verify_num]: the ranking reads verify_count columns of bs rows.
        self.top_scores = torch.zeros((max_bs, verify_num), device=_DEVICE)
        self.pool_size = torch.zeros((1,), dtype=torch.int32, device=_DEVICE)
        self.replace_pool_size = torch.zeros((1,), dtype=torch.bool, device=_DEVICE)
        # The round's launches captured once (None: launched eagerly).
        self.graph: torch.cuda.CUDAGraph | None = None

    def load(self, r: _Round) -> None:
        self.bs = r.bs
        self.live_rows.fill_(r.bs)
        cuts = sorted(self._split.randint(0, r.frontier_sum) for _ in range(r.bs - 1))
        lengths = [b - a for a, b in zip([0] + cuts, cuts + [r.frontier_sum])]
        self._frontier_rows.fill_(_STALE_LENGTH)
        self._frontier_rows[: r.bs, 0] = torch.tensor(
            lengths[: r.bs], dtype=torch.int64
        )
        # Some rounds pass a frontier of exactly the live rows.
        exact = self._split.random() < 0.3
        self.frontier = self._frontier_rows[: r.bs if exact else _FRONTIER_ROWS, 0]
        self.sampled.fill_(r.sampled)
        self.deltas.copy_(r.deltas)
        self.top_scores.zero_()
        rows, cols = r.top_scores.shape
        self.top_scores[:rows, :cols].copy_(r.top_scores)
        self.replace_pool_size.fill_(r.pool_size is not None)
        self.pool_size.fill_(r.pool_size or 0)

    def capture(self) -> None:
        """Capture the round's launches; later rounds replay them."""
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.cuda.graph(self.graph, stream=stream):
            self._launch()
        torch.cuda.current_stream().wait_stream(stream)

    def step(self) -> None:
        """A whole round."""
        self.policy.next_round(self.bs)
        if self.graph is None:
            self._launch()
        else:
            self.graph.replay()

    def _launch(self) -> None:
        policy = self.policy
        policy.begin_round(self.live_rows, self.frontier, self.sampled)
        for d in range(policy.tree_depth - 1):
            # Depth d's work deltas, when it drafted and its slot asks.
            queued = (policy.go[d] > 0) & (policy.work_sizes[d] > 0)
            self.work_deltas.copy_(
                torch.where(queued, self.deltas[d], self.work_deltas)
            )
            policy.decide(d + 1, self.work_deltas)
        # A recorded pool size caps the ranked count (the device pool holds at least verify_num nodes).
        capped = torch.minimum(policy.verify_count, self.pool_size)
        policy.verify_count.copy_(
            torch.where(self.replace_pool_size, capped, policy.verify_count)
        )
        policy.select_verify(self.top_scores)

    def outcome(self) -> _Outcome:
        p = self.policy
        return _Outcome(
            int(p.past_verify_num.item()),
            *_slot_rows(p).tolist(),
            int(p._state[device_policy._LAST_DEPTH].item()),
            int(p._state[device_policy._STOP_WIDTH].item()),
            *(int(t.item()) for t in (p.cumu_count, p.verify_count)),
            int(p._state[device_policy._FLAT].item()),
            int(p.verify_num.item()),
        )


@pytest.mark.parametrize("name, batch_size", sorted(_RECORDED))
def test_device_replays_recorded_tree_rounds(
    tmp_path, name: str, batch_size: int
) -> None:
    s = _SCENARIOS[name]
    batch_sizes, bs_slope, cliff = (
        ((1, 2, 4, 8), 2.0, 0.3) if s.steep else ((1, 2, 4), 0.2, 0.0)
    )
    _write_cost_dir(
        str(tmp_path),
        batch_sizes,
        _TARGET_WIDTHS,
        _DRAFT_WIDTHS,
        bs_slope=bs_slope,
        cliff=cliff,
    )
    config = tmp_path / "benchmark_config.json"
    config.write_text('{"batch_size": 24}')
    policy = _policy(
        load_cost_data(str(tmp_path / "cost")),
        topk=s.topk,
        depth=s.depth,
        verify=s.verify,
        intensity=s.intensity,
        moe=s.moe,
        exec_mode=s.exec_mode,
        config=str(config),
    )
    device = _DeviceDrafter(policy, batch_size)
    got = []
    rounds = _rounds(batch_size, 24, s.topk, s.depth, s.verify, s.max_count)
    for i, (cache_len, deltas, draft_count, scores) in enumerate(rounds):
        # From round 3 on, the round replays from a graph (the device reads every size).
        if i == 3:
            device.capture()
        top = torch.tensor([scores] * batch_size)
        sampled = s.sampled and i % 4 != 3
        device.load(
            _Round(
                batch_size,
                cache_len * batch_size,
                sampled,
                torch.tensor(deltas),
                top,
                draft_count,
            )
        )
        device.step()
        outcome = device.outcome()
        _assert_round(policy, outcome)
        got.append(outcome.recorded())
    assert got == _RECORDED[name, batch_size]


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("batch_size", sorted(_CHAIN_RECORDED))
def test_device_replays_recorded_chain_rounds(
    tmp_path, batch_size: int, sampled: bool
) -> None:
    """Recorded chain rounds; sampled (all but each fourth), a round stops at the plan its rule set."""
    _write_cost_dir(
        str(tmp_path), (1, 2, 4, 8), _CHAIN_LLM_WIDTHS, [0], bs_slope=0.3, cliff=0.08
    )
    chain = _policy(
        load_cost_data(str(tmp_path / "cost")),
        topk=1,
        depth=_CHAIN_DEPTH,
        verify=16,
        intensity="chain_intensity",
    )
    device = _DeviceDrafter(chain, batch_size)
    got, want = [], []
    last = _CHAIN_DEPTH - 1
    plan = last
    for i, (cache_len, deltas) in enumerate(_chain_rounds(batch_size, 24)):
        if i == 3:
            device.capture()
        top = torch.zeros((batch_size, 16))
        sampled_round = sampled and i % 4 != 3
        device.load(
            _Round(
                batch_size,
                cache_len * batch_size,
                sampled_round,
                torch.tensor(deltas),
                top,
            )
        )
        device.step()
        o = device.outcome()
        assert o.verify_count == o.verify_num == o.last_depth + 2
        _assert_round(chain, o)
        got.append(o.last_depth)
        stop = _CHAIN_RECORDED[batch_size][i]
        if sampled_round:
            want.append(plan)
            plan = stop if stop <= plan else min(plan + 2, last)
        else:
            want.append(stop)
    assert got == want


# Widths 4 from 4 requests, 2 from 8, 3 from 300 and the full 16 from 400.
_CHAIN_TABLE = (
    ChainWidthEntry(4, 4),
    ChainWidthEntry(8, 2),
    ChainWidthEntry(300, 3),
    ChainWidthEntry(400, 16),
)


def _chain_policy(tmp_path, table: ChainWidthTable) -> DevicePolicy:
    """A 15-step chain on the chain cost data (batch sizes 1, 2, 4, 8),
    serving up to 512 requests."""
    cost_dir = tmp_path / "cost"
    if not cost_dir.exists():
        _write_cost_dir(
            str(tmp_path),
            (1, 2, 4, 8),
            _CHAIN_LLM_WIDTHS,
            [0],
            bs_slope=0.3,
            cliff=0.08,
        )
    return _policy(
        load_cost_data(str(cost_dir)),
        topk=1,
        depth=_CHAIN_DEPTH,
        verify=16,
        intensity="chain_intensity",
        table=table,
    )


def _table_outcome(width: int) -> _Outcome:
    """A width table round's slots and counts: depths 1..width - 2 draft at
    width one, depth 1 also when a stop at depth 0 leaves it undrafted."""
    last = width - 2
    slots = [(1, d + 1, 0, int(d == 0 or d <= last)) for d in range(max(last, 1) + 1)]
    slots += [(0, 0, 0, 0)] * (_CHAIN_DEPTH + 1 - len(slots))
    widths, starts, work, go = (list(row) for row in zip(*slots))
    stop_width = -1 if last == _CHAIN_DEPTH - 1 else 1
    return _Outcome(
        16, widths, starts, work, go, last, stop_width, width, width, 0, width
    )


def _poll_round(device: _DeviceDrafter) -> None:
    """A round as the drafter polls it: depth 1 always, each next depth
    (its decision) only once the pinned go flag says it drafts."""
    policy = device.policy
    policy.next_round(device.bs)
    policy.begin_round(device.live_rows, device.frontier, device.sampled)
    depth, deadline = 1, time.monotonic() + 600
    while True:
        queued = (policy.go[depth - 1] > 0) & (policy.work_sizes[depth - 1] > 0)
        device.work_deltas.copy_(
            torch.where(queued, device.deltas[depth - 1], device.work_deltas)
        )
        policy.decide(depth, device.work_deltas)
        if depth == policy.tree_depth - 1:
            break
        while (go := policy.go_ready(depth + 1)) is None:
            assert time.monotonic() < deadline, "the decision never landed"
        if not go:
            break
        depth += 1


@pytest.mark.parametrize("protocol", ["eager", "graph", "poll"])
def test_chain_width_table_rounds(tmp_path, protocol: str) -> None:
    """A chain round the width table covers drafts and verifies its width with
    no decision (eager: every decision launched; graph: the round captured
    once at 512 rows; poll: the drafter's protocol), whatever its work deltas;
    a round below the table decides as without one, bit for bit, on the
    1..512 batch tables. Two rounds in three are sampled: a table round leaves
    the sampled plan to the rule's rounds."""
    rule = _DeviceDrafter(_chain_policy(tmp_path, ()), 512)
    device = _DeviceDrafter(_chain_policy(tmp_path, _CHAIN_TABLE), 512)
    rng = random.Random(protocol)
    sizes = [1, 2, 3, 4, 7, 8, 9, 256, 299, 300, 399, 400, 512]
    widths = {bs: None if bs < 4 else 4 if bs < 8 else 2 for bs in sizes}
    widths.update({bs: 3 if bs < 400 else 16 for bs in sizes if bs >= 300})
    seen = set()
    for i, (cache_len, deltas) in enumerate(_chain_rounds(7, 60)):
        bs = rng.choice(sizes)
        sampled = i % 3 != 2
        r = _Round(
            bs, cache_len * bs, sampled, torch.tensor(deltas), torch.zeros(1, 16)
        )
        if widths[bs] is None:
            rule.load(r)
            rule.step()
            expected = rule.outcome()
        else:
            expected = _table_outcome(widths[bs])
        device.load(r)
        if protocol == "graph" and i == 3:
            device.capture()
        if protocol == "poll":
            _poll_round(device)
        else:
            device.step()
        outcome = device.outcome()
        assert outcome == expected, (i, bs)
        if protocol == "poll":
            # The flags the drafter polls: up to the depth after the last drafted one.
            for d in range(2, max(outcome.last_depth + 1, 2) + 1):
                assert device.policy.go_ready(d) == bool(outcome.go[d])
        else:
            _assert_round(device.policy, outcome)
        seen.add(widths[bs])
    assert seen == {None, 2, 3, 4, 16}


def test_chain_width_table_beyond_the_chain_is_refused(tmp_path) -> None:
    for table in ((ChainWidthEntry(4, 17),), (ChainWidthEntry(4, 1),)):
        with pytest.raises(ValueError, match="outside 2..16"):
            _chain_policy(tmp_path, table)
    _write_cost_dir(
        str(tmp_path / "tree"),
        (1, 2, 4),
        _TARGET_WIDTHS,
        _DRAFT_WIDTHS,
        bs_slope=0.2,
        cliff=0.0,
    )
    with pytest.raises(ValueError, match="trees run no chain width table"):
        _policy(
            load_cost_data(str(tmp_path / "tree" / "cost")),
            topk=4,
            depth=8,
            verify=16,
            table=_CHAIN_TABLE,
        )


@pytest.mark.parametrize(
    "widths, exec_mode, match",
    [
        ([0, 63, 7], 0, "strictly increasing"),
        ([0, 7, 7, 15], 0, "strictly increasing"),
        ([0, 31, 63], 1, "within the verify budget 16"),
    ],
)
def test_cost_widths_a_round_cannot_verify_are_refused(
    tmp_path, widths: list[int], exec_mode: int, match: str
) -> None:
    """Widths out of order, or a batch cut wider than the verify budget, would
    write past the verify tree."""
    _write_cost_dir(
        str(tmp_path), (1, 2, 4), widths, _DRAFT_WIDTHS, bs_slope=0.2, cliff=0.0
    )
    with pytest.raises(ValueError, match=match):
        _policy(
            load_cost_data(str(tmp_path / "cost")),
            topk=4,
            depth=8,
            verify=16,
            exec_mode=exec_mode,
        )


def test_chain_cost_widths_the_stop_test_does_not_price_are_refused(tmp_path) -> None:
    """The chain stop test prices widths 2, 4, ..., 16 by position."""
    _write_cost_dir(
        str(tmp_path),
        (1, 2, 4),
        [0, 1, 2, 3, 4, 5, 6, 7, 15],
        [0],
        bs_slope=0.3,
        cliff=0.0,
    )
    with pytest.raises(ValueError, match="chain cost data needs widths 1, 2, 4"):
        _policy(
            load_cost_data(str(tmp_path / "cost")),
            topk=1,
            depth=_CHAIN_DEPTH,
            verify=16,
            intensity="chain_intensity",
        )


def _ratio_13_model_ratio() -> float:
    """A model cost ratio m whose tiny batch size ratio
    max(1.0, 1.0 / ((20 / m + 1) / 2)) is exactly 1.3."""
    m = 20 / (2 / 1.3 - 1)
    while (ratio := max(1.0, 1.0 / ((20 / m + 1) / 2))) != 1.3:
        m = math.nextafter(m, math.inf if ratio < 1.3 else -math.inf)
    return m


_W4 = [0, 1, 3, 7]
# name: (cost index, cost ratio, model cost ratio, policy changes, rounds), one cost interval.
_BOUNDARIES = {
    # A sampled first round sizes unranked (verify_count 7 is a measured width).
    "unranked_first_round": (
        _W4,
        [1.0, 1.05, 1.1, 1.2],
        20.0,
        dict(depth=3, verify=7),
        "4s 4 4 4sf 4f",
    ),
    # A stop at depth 3 plans the next sampled round there; plans grow by two up to the last depth.
    "sampled_plan_cap": (
        [0, 1, 3, 7, 15],
        [1.0, 1.05, 1.1, 1.2, 1.3],
        20.0,
        {},
        "4sx3 4s 4s 4s",
    ),
    # The cliff look-ahead at depth 2, which only a stop triggers.
    "chain_cliff_after_stop": (
        None,
        [1.0, 1.0, 1.0, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5],
        2.0,
        {},
        "2t2 " * 6,
    ),
    "chain_cliff_step_exactly_015": (
        None,
        [1.0, 0.1, 0.15, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35],
        10.0,
        {},
        "4t2 " * 6,
    ),
    "model_ratio_exactly_5": (_W4, [1.0, 1.05, 1.1, 1.2], 5.0, {}, "4"),
    "model_ratio_exactly_100": (_W4, [1.0, 1.05, 1.1, 1.2], 100.0, {}, "4"),
    "cost_state_exactly_085": (_W4, [1.0, 1.0, 1.0, 0.85], 20.0, {}, "1"),
    "cost_state_exactly_070_moe": (
        _W4,
        [1.0, 1.0, 1.0, 0.70],
        20.0,
        dict(moe=True),
        "1",
    ),
    "cost_state_exactly_4": (_W4, [1.0, 1.0, 1.0, 4.0], 20.0, {}, "4"),
    # Widths 1 and 3 cost exactly 0.04 apart.
    "cost_delta_exactly_004": (_W4, [1.0, 0.04, 0.08, 1.1], 20.0, {}, "4f 4f"),
    # A tiny batch: decisions past depth 5 compare against ratio 1.3 exactly.
    "batch_size_ratio_exactly_13": (
        _W4,
        [1.0, 1.0, 1.1, 1.2],
        _ratio_13_model_ratio(),
        dict(depth=10),
        "1 " + "1t7 " * 6,
    ),
    # Tiny rounds, then one clearing tiny mid-round: its stopping decision reads the new flag.
    "verify_count_after_tiny_turns": (
        [0, 1, 3, 7, 15, 31, 63],
        [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5],
        20.0,
        dict(topk=8, depth=6, verify=64),
        "1f " * 16 + "4 1",
    ),
    # 32 measured widths: the widest is reset only by its comparison with width 20.
    "verify_reset_by_a_late_width": (
        list(range(32)),
        [1.0 + 0.1 * w for w in range(32)],
        20.0,
        dict(topk=8, depth=6, verify=64),
        "1 1",
    ),
}
# fmt: off
# Per round, as _RECORDED's: EAGLE-4's decisions.
_BOUNDARY_EXPECTED = {
    'batch_size_ratio_exactly_13': [
        (16, 9, -1, 16, 8), (32, 9, -1, 16, 8), (32, 9, -1, 16, 8), (32, 7, 2, 16, 8),
        (32, 7, 2, 16, 8), (32, 9, -1, 16, 8), (32, 7, 2, 16, 8),
    ],
    'chain_cliff_after_stop': [
        (16, 5, -1, 7, 7), (16, 5, -1, 7, 7), (16, 2, 1, 4, 4), (16, 2, 1, 4, 4),
        (16, 5, -1, 7, 7), (16, 2, 1, 4, 4),
    ],
    'chain_cliff_step_exactly_015': [
        (16, 5, -1, 7, 7), (16, 5, -1, 7, 7), (16, 2, 1, 4, 4), (16, 2, 1, 4, 4),
        (16, 5, -1, 7, 7), (16, 2, 1, 4, 4),
    ],
    'cost_delta_exactly_004': [
        (16, 4, -1, 16, 4), (16, 4, -1, 16, 4),
    ],
    'cost_state_exactly_070_moe': [
        (16, 4, -1, 16, 8),
    ],
    'cost_state_exactly_085': [
        (16, 4, -1, 16, 8),
    ],
    'cost_state_exactly_4': [
        (16, 4, -1, 16, 4),
    ],
    'model_ratio_exactly_100': [
        (16, 4, -1, 16, 8),
    ],
    'model_ratio_exactly_5': [
        (16, 4, -1, 16, 8),
    ],
    'sampled_plan_cap': [
        (16, 4, -1, 16, 16), (16, 3, 2, 15, 8), (16, 4, -1, 16, 8), (16, 4, -1, 16, 16),
    ],
    'unranked_first_round': [
        (16, 2, -1, 7, 4), (16, 2, -1, 7, 4), (16, 2, -1, 7, 4), (16, 2, -1, 7, 4),
        (16, 2, -1, 7, 2),
    ],
    'verify_count_after_tiny_turns': [
        (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2),
        (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2),
        (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2),
        (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2), (64, 5, -1, 64, 2),
        (64, 5, -1, 64, 2), (16, 5, -1, 64, 64),
    ],
    'verify_reset_by_a_late_width': [
        (64, 5, -1, 64, 31), (64, 5, -1, 64, 32),
    ],
}
# Per "t" scenario: the compared delta's index and fp32 bits, then each round's delta 0 bits around the threshold.
_ON_THRESHOLD = {
    'batch_size_ratio_exactly_13': (2, 0x3EA83586, [0x329B5FFB, 0x329B5FFB, 0x329B5FFC, 0x329B5FFC, 0x329B5FFB, 0x329B5FFC]),
    'chain_cliff_after_stop': (1, 0x4013C362, [0x346CBFB1, 0x346CBFB1, 0x346CBFB2, 0x346CBFB2, 0x346CBFB1, 0x346CBFB2]),
    'chain_cliff_step_exactly_015': (1, 0x3F2C63F2, [0x333EDFA4, 0x333EDFA4, 0x333EDFA5, 0x333EDFA5, 0x333EDFA4, 0x333EDFA5]),
}
# fmt: on


@pytest.mark.parametrize("name", sorted(_BOUNDARIES))
def test_rounds_at_exact_boundaries(name: str) -> None:
    """Literal bounds and stability comparisons on their exact values. A
    round ``<bs>[s][f][x<d>|t<d>]`` is sampled, has flat ranked scores, no gain
    at depth d's decision or that decision on its threshold (large gains
    else); a threshold's depth 0 work delta is bisected on EAGLE-4's
    double-precision test."""
    index, ratio, model_ratio, changes, rounds = _BOUNDARIES[name]
    chain = index is None
    target = IntervalCost(
        value=[1.0] * len(ratio), ratio=ratio, index=index or _CHAIN_LLM_WIDTHS
    )
    draft = IntervalCost(value=[1.0, 1.0], ratio=[1.0, 1.2], index=[0, 1])
    batch = BatchCost(
        intervals=[(target, draft)], cache_length_step=1000, model_ratio=[model_ratio]
    )
    costs = dict.fromkeys((1, 2, 4), batch)
    if chain:
        policy = _policy(costs, topk=1, depth=6, verify=16, intensity="chain_intensity")
    else:
        policy = _policy(costs, **(dict(topk=2, depth=5, verify=16) | changes))
    device = _DeviceDrafter(policy, 4)
    if name == "verify_reset_by_a_late_width":
        # The previous round's published scores.
        published = [10.0 * w for w in range(64)]
        published[31] = published[20] + 1.0
        policy._ranked[:64] = torch.tensor(published)
        policy._state[device_policy._RANKED_COUNT] = 64
    k = policy.node_top_k
    on_index, on_bits, on_roots = _ON_THRESHOLD.get(name, (0, 0, []))
    roots, got = iter(on_roots), []
    for token in rounds.split():
        bs, flags, kind, gains_depth = re.fullmatch(
            r"(\d+)([sf]*)(?:([xt])(\d+))?", token
        ).groups()
        # Row d holds the work deltas the decision at depth d + 1 reads.
        rows = [[0.0] + [1e6] * k for _ in range(policy.tree_depth)]
        if kind == "x":
            rows[int(gains_depth) - 1] = [1.0] * (k + 1)
        if kind == "t":
            row = rows[int(gains_depth) - 1]
            row[0], row[on_index] = _f32(next(roots)), _f32(on_bits)
        top = torch.rand(
            (int(bs), policy.verify_num_limit),
            generator=torch.Generator().manual_seed(1),
        )
        top = top.sort(dim=-1, descending=True).values
        top[:, 1:] = 1e-3 if "f" in flags else top[:, 1:]
        top[:, 0] = 1.001
        device.load(
            _Round(int(bs), int(bs) * 500, "s" in flags, torch.tensor(rows), top)
        )
        device.step()
        got.append(device.outcome().recorded())
    assert got == _BOUNDARY_EXPECTED[name]


def test_exec_mode_1_reads_the_benchmark_past_one_request(tmp_path) -> None:
    _write_cost_dir(
        str(tmp_path), (1, 2, 4), _TARGET_WIDTHS, _DRAFT_WIDTHS, bs_slope=0.2, cliff=0.0
    )
    config = tmp_path / "benchmark_config.json"
    policy = _policy(
        load_cost_data(str(tmp_path / "cost")),
        topk=4,
        depth=8,
        verify=16,
        exec_mode=1,
        config=str(config),
    )
    policy.next_round(1)
    with pytest.raises(FileNotFoundError, match="benchmark client writes"):
        policy.next_round(2)
    # Line-JSON writers store a one-element list.
    config.write_text('[{"batch_size": 24}]\n')
    policy.next_round(2)
    assert int(policy._state[device_policy._BENCH_BATCH_SIZE].item()) == 24


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
