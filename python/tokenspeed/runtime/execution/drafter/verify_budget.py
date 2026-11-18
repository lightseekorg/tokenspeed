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

"""Host-side verification budget selection from measured graph-step costs.

The caller owns calibration and request feedback. No tensors, device readbacks,
or distributed operations belong in this policy.
"""

import math
from bisect import bisect_left, insort
from collections import defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from operator import mul
from statistics import median_high


def expected_prefix_tokens(rates: Sequence[Sequence[float]]) -> list[float]:
    """Expected output for each additional query after the request anchors."""
    gains = sorted((p for row in rates for p in accumulate(row, mul)), reverse=True)
    return list(accumulate(gains, initial=float(len(rates))))


class VerifyBudgetPlanner:
    """Query costs shared by confidence and acceptance-history policies.

    Baselines measure complete decode graphs. Completed live rounds add a
    bounded correction to the whole curve, including unvisited query buckets.
    Keep nine samples per bucket; each Q with ten samples in the last 5,000
    rounds contributes its median to a nonnegative per-query correction.
    """

    def __init__(self, width: int):
        if width < 2:
            raise ValueError("confidence requires a draft window")
        self.width = width
        self.baselines: dict[tuple[int, int], float] = {}
        self.timings: dict[tuple[int, int], deque[float]] = defaultdict(
            lambda: deque(maxlen=9)
        )
        self.corrections: deque[tuple[int, float]] = deque(maxlen=5000)
        self.corrections_by_query: dict[int, list[float]] = defaultdict(list)

    def _validate_timing(self, batch: int, queries: int, milliseconds: float) -> None:
        if (
            not batch <= queries <= batch * self.width
            or batch < 1
            or not (math.isfinite(milliseconds) and milliseconds > 0)
        ):
            raise ValueError("invalid confidence timing sample")

    def set_baseline(self, batch: int, queries: int, milliseconds: float) -> None:
        self._validate_timing(batch, queries, milliseconds)
        self.baselines[batch, queries] = milliseconds

    def observe(self, batch: int, queries: int, milliseconds: float) -> None:
        self._validate_timing(batch, queries, milliseconds)
        self.timings[batch, queries].append(milliseconds)
        if (batch, queries) in self.baselines:
            # Keep the bounded window sorted per Q, so selection only reads medians.
            if len(self.corrections) == self.corrections.maxlen:
                old_q, old_value = self.corrections[0]
                old = self.corrections_by_query[old_q]
                old.pop(bisect_left(old, old_value))
            value = (milliseconds - self.baselines[batch, queries]) / queries
            self.corrections.append((queries, value))
            insort(self.corrections_by_query[queries], value)

    def select(
        self,
        batch: int,
        rates: Sequence[Sequence[float]],
        *,
        capacities: Collection[int] | None = None,
    ) -> int:
        if batch < 1 or len(rates) > batch:
            raise ValueError("invalid confidence batch")
        if any(len(row) != self.width - 1 for row in rates):
            raise ValueError("confidence width mismatch")
        rewards = expected_prefix_tokens(
            [
                [min(1, max(0, p)) if math.isfinite(p) else 0 for p in row]
                for row in rates
            ]
        )
        medians = [
            v[len(v) // 2] for v in self.corrections_by_query.values() if len(v) >= 10
        ]
        correction = max(0, median_high(medians)) if medians else 0
        best_q, best_rate = batch, -1
        for key in sorted(self.baselines.keys() | self.timings.keys()):
            b, q = key
            if b != batch or (capacities is not None and q not in capacities):
                continue
            cost = self.baselines[key] + correction * q if key in self.baselines else 0
            if key in self.timings:
                cost = max(cost, median_high(self.timings[key]))
            rate = rewards[min(q - len(rates), len(rewards) - 1)] / cost
            if rate > best_rate:
                best_q, best_rate = q, rate
        return best_q


@dataclass(frozen=True)
class VerifyBudgetPolicy:
    enabled: bool
    min_batch_size: int
    verify_width: int
    min_gain: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.min_batch_size) is not int or self.min_batch_size < 1:
            raise ValueError("min_batch_size must be a positive integer")
        if type(self.verify_width) is not int or self.verify_width < 2:
            raise ValueError("verify_width must include an anchor and a draft token")
        if not math.isfinite(self.min_gain) or not 0 <= self.min_gain < 1:
            raise ValueError("min_gain must be finite and in [0, 1)")

    def admits(self, *, batch_size: int, graph_batch_size: int | None) -> bool:
        """Check admission before reading scores or agreeing across TP ranks.

        Padded batches currently retain native verification. Their cost and
        reward cannot be inferred from a fully occupied graph measurement.
        """
        return (
            self.enabled
            and batch_size >= self.min_batch_size
            and graph_batch_size == batch_size
        )

    def select(
        self,
        scores: Sequence[Sequence[float]],
        *,
        graph_batch_size: int | None,
        native_step_ms: float,
        adaptive_step_ms: Mapping[int, float],
        policy_ms: float,
    ) -> int:
        """Return the physical query budget, including one anchor per request.

        Costs must represent this batch and the selected context bucket and
        include draft, verification, and sampling. Add separately measured host
        policy overhead through policy_ms. Startup wall-clock samples already
        include policy and agreement, so use policy_ms=0 for those samples.
        Missing measurements or non-finite feedback retain full width.
        Full width denotes native execution, not the adaptive full-width graph.
        """
        if not self.admits(batch_size=len(scores), graph_batch_size=graph_batch_size):
            return len(scores) * self.verify_width
        return self.select_distributed(
            (scores,),
            graph_batch_size=graph_batch_size,
            native_step_ms=native_step_ms,
            adaptive_step_ms=adaptive_step_ms,
            policy_ms=policy_ms,
        )

    def select_distributed(
        self,
        rank_scores: Sequence[Sequence[Sequence[float]]],
        *,
        graph_batch_size: int | None,
        native_step_ms: float,
        adaptive_step_ms: Mapping[int, float],
        policy_ms: float,
    ) -> int:
        """Choose a common per-rank capacity using the sum of rank rewards.

        Each rank allocates its own capacity among its own requests. Costs must
        measure the synchronized distributed step, not isolated rank execution.
        Real request counts may differ; padded rows contribute no reward.
        """
        if not rank_scores:
            raise ValueError("distributed verification requires rank scores")
        sizes = [len(scores) for scores in rank_scores]
        bs = max(sizes)
        padded = bs if graph_batch_size is None else graph_batch_size
        if padded < bs:
            raise ValueError("graph batch must cover every rank")
        full = padded * self.verify_width
        if not self.enabled or bs < self.min_batch_size or graph_batch_size is None:
            return full
        if any(
            len(row) != self.verify_width - 1
            for scores in rank_scores
            for row in scores
        ):
            raise ValueError("confidence width must match the draft window")
        if not math.isfinite(native_step_ms) or native_step_ms <= 0:
            return full
        if not math.isfinite(policy_ms) or policy_ms < 0:
            raise ValueError("policy cost must be finite and nonnegative")
        if any(
            not math.isfinite(v)
            for scores in rank_scores
            for row in scores
            for v in row
        ):
            return full
        for q, cost in adaptive_step_ms.items():
            if type(q) is not int or not padded <= q <= full:
                raise ValueError(
                    "query capacity must fit the active verification window"
                )
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("adaptive step costs must be finite and positive")

        # Costs describe the shared physical graph; rewards count only live
        # requests. Padding retains a physical anchor, but earns no reward.
        # An idle rank contributes neither anchors nor draft gains.
        rewards = [float(sum(sizes))] * (full + 1)
        for scores in rank_scores:
            probabilities = []
            for row in scores:
                converted = []
                for score in row:
                    exp = math.exp(-abs(score))
                    converted.append(1 / (1 + exp) if score >= 0 else exp / (1 + exp))
                probabilities.append(converted)
            prefix = expected_prefix_tokens(probabilities)
            for q in range(padded, full + 1):
                rewards[q] += prefix[min(q - padded, len(prefix) - 1)] - len(scores)
        native_rate = rewards[-1] / native_step_ms
        best_q = full
        best_rate = native_rate * (1 + self.min_gain)
        for q, step_ms in sorted(adaptive_step_ms.items(), reverse=True):
            if q == full:
                continue
            rate = rewards[q] / (step_ms + policy_ms)
            if rate > best_rate:
                best_q, best_rate = q, rate
        return best_q
