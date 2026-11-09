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

"""Completed acceptance history for the existing confidence budget planner."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptanceForecast:
    """Conditional rates for full and observed-prefix-only budget comparisons."""

    generation: int
    rates: list[float]
    observed_rates: list[float]
    censored: bool


@dataclass(frozen=True)
class _AcceptanceSample:
    queries: int
    accepted: int
    survival: tuple[float | None, ...]


class AcceptanceHistory:
    """Smooth prefix survival without treating a budget limit as a rejection.

    Args:
        width: Maximum verify width, including the target token.
        observation_weight: Weight of each completed observation in the EMA.
    """

    def __init__(self, width: int, *, observation_weight: float):
        if width < 2 or not 0 < observation_weight <= 1:
            raise ValueError("acceptance history requires a window and an EMA weight")
        self.width = width
        self.observation_weight = observation_weight
        self._survival: list[float | None] = [None] * (width - 1)
        self._samples: dict[int, _AcceptanceSample] = {}

    def observe(self, generation: int, *, queries: int, accepted: int) -> None:
        """Record pre-stop acceptance for this candidate generation.

        A rejection identifies every later prefix as unsuccessful. When the
        entire query budget was accepted, longer prefixes remain unobserved.
        """
        if not 1 <= accepted <= queries <= self.width:
            raise ValueError("acceptance must lie inside the verified prefix")
        for index, previous in enumerate(self._survival):
            position = index + 1
            if accepted == queries and position >= queries:
                continue
            value = float(position < accepted)
            self._survival[index] = (
                value
                if previous is None
                else previous + self.observation_weight * (value - previous)
            )
        self._samples[generation] = _AcceptanceSample(
            queries, accepted, tuple(self._survival)
        )
        self._samples = {
            age: sample
            for age, sample in self._samples.items()
            if age >= generation - 3
        }

    def forecast(
        self, model_rates: list[float], *, maximum_generation: int
    ) -> AcceptanceForecast | None:
        """Use the newest completed sample no newer than the allowed generation.

        Observed prefixes use their empirical EMA survival. A budget-limited
        sample extrapolates its unobserved tail with conditional model rates.
        The observed-only curve sets that tail to zero, allowing the caller to
        measure whether the same cost model actually favors a larger budget.
        """
        if len(model_rates) != self.width - 1:
            raise ValueError("confidence rates must match the verify width")
        ages = [age for age in self._samples if age <= maximum_generation]
        if not ages:
            return None
        generation = max(ages)
        sample = self._samples[generation]
        censored = sample.accepted == sample.queries and sample.queries < self.width
        rates, observed_rates = [], []
        previous = 1.0
        for index, model in enumerate(model_rates):
            probability = min(1.0, max(0.0, model)) if math.isfinite(model) else 0.0
            tail = censored and index + 1 >= sample.queries
            empirical = sample.survival[index]
            survival = (
                previous * probability
                if tail or empirical is None
                else min(previous, max(0.0, empirical))
            )
            conditional = survival / previous if previous > 0 else 0.0
            rates.append(conditional)
            observed_rates.append(0.0 if tail else conditional)
            previous = survival
        return AcceptanceForecast(generation, rates, observed_rates, censored)
