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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


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

        Costs must cover this batch and context and include draft, verification,
        and sampling. Adaptive costs exclude the separately measured host policy
        overhead. Missing measurements or non-finite feedback retain full width.
        Full width denotes native execution, not the adaptive full-width graph.
        """
        bs = len(scores)
        full = bs * self.verify_width
        if not self.admits(batch_size=bs, graph_batch_size=graph_batch_size):
            return full
        if any(len(row) != self.verify_width - 1 for row in scores):
            raise ValueError("confidence width must match the draft window")
        if not math.isfinite(native_step_ms) or native_step_ms <= 0:
            return full
        if not math.isfinite(policy_ms) or policy_ms < 0:
            raise ValueError("policy cost must be finite and nonnegative")
        if any(not math.isfinite(v) for row in scores for v in row):
            return full
        for q, cost in adaptive_step_ms.items():
            if type(q) is not int or not bs <= q <= full:
                raise ValueError(
                    "query capacity must fit the active verification window"
                )
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("adaptive step costs must be finite and positive")

        gains = []
        for row in scores:
            survival = 1.0
            for score in row:
                # Stable sigmoid without overflow at either end.
                exp = math.exp(-abs(score))
                survival *= 1 / (1 + exp) if score >= 0 else exp / (1 + exp)
                gains.append(survival)
        gains.sort(reverse=True)
        rewards = [float(bs)]
        for gain in gains:
            rewards.append(rewards[-1] + gain)
        native_rate = rewards[-1] / native_step_ms
        best_q = full
        best_rate = native_rate * (1 + self.min_gain)
        for q, step_ms in sorted(adaptive_step_ms.items(), reverse=True):
            if q == full:
                continue
            rate = rewards[q - bs] / (step_ms + policy_ms)
            if rate > best_rate:
                best_q, best_rate = q, rate
        return best_q
