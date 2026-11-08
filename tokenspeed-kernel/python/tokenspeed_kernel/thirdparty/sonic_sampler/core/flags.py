# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from enum import IntFlag, auto, unique
from operator import lshift

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import MAX_K

# The most top-k log-probabilities a request may ask for.
MAX_TOP_LP: int = 5


@unique
class Indicator(IntFlag):

    # Grammar Enablement -> 0x1.
    GRAMMAR = auto()

    # Repetition (Multiplicative) -> 0x2.
    MULTIPLICATIVE = auto()

    # Repetition (Frequency) -> 0x4.
    FREQUENCY = auto()

    # Repetition (Presence) -> 0x8.
    PRESENCE = auto()

    # Logit Bias -> 0x10.
    BIAS = auto()

    # Temperature -> 0x20.
    TEMPERATURE = auto()

    # Top-1 -> 0x40.
    GREEDY = auto()

    # Top-K -> 0x80.
    TOP_K = auto()

    # Top-P -> 0x100.
    TOP_P = auto()

    # Min-P -> 0x200.
    MIN_P = auto()

    # Output Top-K Log-Probs -> 0x400.
    TOP_K_LOGPROBS = auto()

    # Stochastic Verify over Greedy Drafting -> 0x800.
    STOCHASTIC_VERIFY_GREEDY_DRAFT = auto()

    @classmethod
    def stochastic(cls) -> Indicator:

        return (
            Indicator.TEMPERATURE | Indicator.TOP_K | Indicator.TOP_P | Indicator.MIN_P
        )

    @classmethod
    def from_params(
        cls,
        grammar: bool = False,
        multiplicative: float = 1.0,
        frequency: float = 0.0,
        presence: float = 0.0,
        logit_bias: bool = False,
        temperature: float = 1.0,
        top_k: int = MAX_K,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k_logprobs: int = 0,
        greedy_draft: bool = False,
    ) -> Indicator:

        flags = [
            grammar,
            multiplicative != 1.0,
            frequency != 0.0,
            presence != 0.0,
            logit_bias,
            temperature != 1.0,
            (greedy := (top_k == 1)),
            1 < top_k < MAX_K,
            not greedy and (0.0 < top_p < 1.0),
            not greedy and (0.0 < min_p < 1.0),
            not greedy and (0 < top_k_logprobs <= MAX_TOP_LP),
            not greedy and greedy_draft,
        ]

        positions = range(len(flags))
        bits = map(int, flags)

        return cls(sum(map(lshift, bits, positions)))
