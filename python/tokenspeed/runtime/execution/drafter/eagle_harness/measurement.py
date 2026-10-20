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

"""EAGLE-Harness cost measurement (exec mode 2), engine side.

The measurement driver writes ``<run dir>/config/measure_config.json``
(``{"batch_size": n}``) and sends ``n`` identical greedy requests.
Every decode round then drafts the full depth at one draft width and
verifies at one verify width, both cycled round-robin from a counter over the
batch size's sweep (:func:`draft_widths`, :func:`verify_widths`;
:class:`MeasurePolicy`), and commits exactly one fixed token per request,
so the context grows by one token a round. Rank 0 times each round's draft
depths (one draft step per sample, wall time between drained GPUs and a TP
barrier) and the next forward's verify (target forward through the accept,
events recorded inside the decode graph), and appends the samples
:mod:`cost` processes.

Only the rounds of a batch at the configured size count, and each prefill
starts them afresh. Every rank counts them; once the batch has its
:data:`MEASURED_ROUNDS` verifies, every request commits an EOS token
instead, which ends the batch at the same round on every rank.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from tokenspeed.runtime.execution.drafter.eagle_harness.config import (
    first_rank_value,
    load_run_config,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.cost import (
    MeasureRecord,
    save_records,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.device_policy import (
    write_round_slots,
)

__all__ = [
    "FORCED_TOKEN",
    "MEASURED_ROUNDS",
    "CostMeasurement",
    "MeasurePolicy",
    "draft_widths",
    "verify_widths",
]

# The token every request commits each measured round.
FORCED_TOKEN = 888
# Verify (and draft) samples a batch records before it ends.
MEASURED_ROUNDS = 512


def draft_widths(batch_size: int, compute_intensity: str, node_top_k: int) -> list[int]:
    """The draft widths a measurement sweeps at this batch size, up to
    ``node_top_k`` (wider ones are not measurable and are skipped)."""
    if compute_intensity == "chain_intensity":
        widths = [1]
    elif batch_size <= 32:
        widths = [1, 2, 4, 8]
    else:
        widths = [1, 2, 4] if batch_size <= 64 else [1, 2]
    return [w for w in widths if w <= node_top_k]


def verify_widths(batch_size: int, compute_intensity: str) -> list[int]:
    """The verify widths a measurement sweeps at this batch size, widest first
    (the order a measurement cycles through them)."""
    if compute_intensity == "chain_intensity":
        return [16, 14, 12, 10, 8, 6, 4, 2, 1]
    high = compute_intensity == "high_intensity"
    if batch_size <= 2:
        return [64, 32, 16, 8, 1]
    if batch_size <= 8:
        return [64, 32, 16, 8, 1] if high else [32, 16, 8, 4, 1]
    return [32, 16, 8, 4, 1] if high else [16, 8, 4, 2, 1]


class MeasurePolicy:
    """Exec mode 2 decisions, taken on the host into device slots laid out as
    :class:`DevicePolicy`'s: every depth at the round's draft width (the first at the
    previous round's, as EAGLE-4 decides the first-step width before
    the round's width), no early stop, every request verified at the round's
    verify width. The device steps of a round have nothing to decide."""

    def __init__(
        self,
        *,
        verify_num: int,
        node_top_k: int,
        tree_depth: int,
        compute_intensity: str,
        device: torch.device,
    ) -> None:
        self.verify_num_limit = verify_num
        self.node_top_k = node_top_k
        self.tree_depth = tree_depth
        self.compute_intensity = compute_intensity
        self.counter = 0
        self.draft_width = node_top_k
        self.previous_draft_width = node_top_k
        self.verify_width_now = verify_num
        # Per depth: width, pool start, work list size, go (DevicePolicy's layout).
        self._slots = torch.zeros((4, tree_depth + 1), dtype=torch.int32, device=device)
        self.widths, self.cumu_starts, self.work_sizes, _ = self._slots.unbind(0)
        self._counts = torch.zeros((4,), dtype=torch.int32, device=device)
        self.past_verify_num, self.cumu_count, self.verify_count, self.verify_num = (
            self._counts.split(1)
        )

    def next_widths(self, batch_size: int, measured_batch_size: int) -> None:
        """Advance the counter and pick the round's widths, verifying at most
        the measured batch's widest verify width."""
        self.counter += 1
        ci = self.compute_intensity
        drafts = draft_widths(batch_size, ci, self.node_top_k)
        limit = min(max(verify_widths(measured_batch_size, ci)), self.verify_num_limit)
        verifies = [w for w in verify_widths(batch_size, ci) if w <= limit]
        self.previous_draft_width = self.draft_width
        self.draft_width = drafts[self.counter % len(drafts)]
        self.verify_width_now = verifies[self.counter % len(verifies)]

    def write_round(self) -> None:
        """The round's slots at the current widths."""
        widths = [self.previous_draft_width]
        widths += [self.draft_width] * (self.tree_depth - 2)
        self._write(widths, self.verify_width_now)

    def write_full_round(self) -> None:
        """Slots of a round drafting every depth at full width."""
        widths = [self.node_top_k] * (self.tree_depth - 1)
        self._write(widths, self.verify_num_limit)

    def _write(self, widths: list[int], verify_num: int) -> None:
        start = write_round_slots(self._slots, self.node_top_k, widths, lambda d, s: 0)
        verify_count = min(self.verify_num_limit, start)
        # The tree holds only the ranked nodes; the verify forward runs every row anyway.
        verify_num = min(verify_num, verify_count)
        self._counts.copy_(
            torch.tensor(
                [self.verify_num_limit, start, verify_count, verify_num],
                dtype=torch.int32,
            )
        )

    def go_ready(self, depth: int) -> bool:
        return depth < self.tree_depth

    def deltas_depth(self, depth: int) -> bool:
        return False

    def begin_round(
        self, live_rows: torch.Tensor, frontier: torch.Tensor, sampled: torch.Tensor
    ) -> None:
        """The round's slots are written before it."""

    def decide(self, depth: int, work_deltas: torch.Tensor) -> None:
        """Measured widths are decided before the round."""

    def select_verify(self, top_scores: torch.Tensor) -> None:
        """Measured rounds verify at the width written before the round."""


class CostMeasurement:
    """Exec mode 2 sampling: the run's parameters, the round timers and the
    recorded samples (rank 0 keeps and writes them). The TP ranks read the
    driver's config through ``file_group`` (``first_rank_value``)."""

    def __init__(
        self,
        *,
        run_dir: str,
        tree_depth: int,
        tp_group,
        file_group: dist.ProcessGroup | None,
        rank: int,
        eos_token: int,
        device: torch.device,
    ) -> None:
        self.run_dir = run_dir
        self.config_path = os.path.join(run_dir, "config", "measure_config.json")
        self.tree_depth = tree_depth
        self.tp_group = tp_group
        self.file_group = file_group
        self._barrier = torch.zeros((1,), dtype=torch.int32, device=device)
        self.recorder = rank == 0
        # The batch size the driver measures (None before it wrote one).
        self.measured_batch_size: int | None = None
        self.records: list[MeasureRecord] = []
        # Committed token of every request, read in-graph: FORCED_TOKEN, then EOS once finished.
        self.commit_token = torch.full(
            (1,), FORCED_TOKEN, dtype=torch.int32, device=device
        )
        self.eos_token = eos_token
        # Recorded inside the decode graph around the target forward and verify.
        self.verify_start = torch.cuda.Event(enable_timing=True, external=True)
        self.verify_end = torch.cuda.Event(enable_timing=True, external=True)
        self.draft_start = torch.cuda.Event(enable_timing=True)
        self.draft_end = torch.cuda.Event(enable_timing=True)
        # (verify width, context) of the tree the next forward verifies.
        self.pending_verify: tuple[int, int] | None = None
        # Rounds of the batch counted on every rank (rank 0 also keeps their samples).
        self.verify_rounds = 0
        self.draft_rounds = 0
        self.written = False

    def on_extend(self) -> None:
        """A prefill: reload the measured batch size and start the batch afresh."""
        path = self.config_path
        self.measured_batch_size = first_rank_value(
            lambda: (
                load_run_config(path)["batch_size"] if os.path.exists(path) else None
            ),
            self.file_group,
        )
        self.records = []
        self.pending_verify = None
        self.verify_rounds = 0
        self.draft_rounds = 0
        self.written = False
        self.commit_token.fill_(FORCED_TOKEN)

    @property
    def finished(self) -> bool:
        """Whether the batch has its rounds (the same on every rank)."""
        return self.verify_rounds >= MEASURED_ROUNDS

    def recording(self, batch_size: int) -> bool:
        return batch_size == self.measured_batch_size and not self.finished

    def _drain(self) -> None:
        if self.tp_group is not None:
            # Device group: the control plane's CPU collectives run on the event-loop thread.
            dist.all_reduce(self._barrier, group=self.tp_group)
        torch.cuda.synchronize()

    def record_verify(self, batch_size: int) -> None:
        """The verify this forward ran, if it verified a measured round."""
        pending, self.pending_verify = self.pending_verify, None
        if pending is None or not self.recording(batch_size):
            return
        self.verify_rounds += 1
        if self.finished:
            self.commit_token.fill_(self.eos_token)
        if self.recorder:
            self.verify_end.synchronize()
            elapsed = int(self.verify_start.elapsed_time(self.verify_end) * 1000)
            self._append(elapsed, 1, pending[0], pending[1], "model_llm", batch_size)

    def begin_draft(self) -> None:
        self._drain()
        self.draft_start.record()

    def end_draft(
        self, batch_size: int, draft_width: int, verify_width: int, cache_len: int
    ) -> None:
        self._drain()
        self.draft_end.record()
        if not self.recording(batch_size) or self.draft_rounds >= MEASURED_ROUNDS:
            return
        self.draft_rounds += 1
        self.pending_verify = (verify_width, cache_len)
        if self.recorder:
            self.draft_end.synchronize()
            elapsed = int(self.draft_start.elapsed_time(self.draft_end) * 1000)
            self._append(
                elapsed,
                self.tree_depth - 1,
                draft_width,
                cache_len,
                "model_slm",
                batch_size,
            )

    def _append(
        self,
        elapsed: int,
        count: int,
        width: int,
        cache_len: int,
        category: str,
        batch_size: int,
    ) -> None:
        if elapsed <= 0:
            raise RuntimeError(f"{category} sample of {elapsed} us")
        self.records.append(
            MeasureRecord(elapsed, count, width, cache_len, category, batch_size)
        )

    def complete(self) -> bool:
        """Whether rank 0 holds the finished batch's samples, not written yet."""
        return self.recorder and self.finished and not self.written

    def write_model_records(self) -> str:
        """Write the batch's raw stage samples under ``test/bs_<n>``; returns that directory."""
        test_dir = os.path.join(self.run_dir, "test", f"bs_{self.measured_batch_size}")
        save_records(test_dir, self.records)
        self.written = True
        return test_dir
