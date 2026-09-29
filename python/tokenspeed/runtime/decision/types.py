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

"""Types for decision-style requests (Jev-like typed decisions).

A decision request asks the model to choose among enumerated candidates
rather than generate prose — routing, classification, policy checks,
ranking. An adapter compiles it into a low-level ``ScoreCall`` (the
``/v1/score`` contract); the low-level readout guarantees every label is
scored, independent of top-k logprobs coverage. See docs/design/scoring.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# One independent Yes/No judgment per candidate; candidates cannot see each
# other. Scores of different candidates are not comparable beyond argmax-of-Yes.
STYLE_POINTWISE_YESNO = "pointwise_yesno"
# One joint choice among all candidates visible together, read as A/B/C...
# labels at a single answer boundary.
STYLE_FUSED_CHOICE = "fused_choice"

ALL_STYLES = (STYLE_POINTWISE_YESNO, STYLE_FUSED_CHOICE)


@dataclass(frozen=True)
class DecisionRequest:
    """High-level decision request; ``adapter`` selects the model-family
    adapter that owns the prompt scaffold and label vocabulary.

    All behavior-selecting fields are explicit — there are no defaults.
    """

    # Shared context + question every candidate judgment sees.
    query: str
    # Candidate answer texts, in response order.
    candidates: list[str]
    # One of ALL_STYLES.
    style: str
    # Adapter registry name (``get_decision_adapter``).
    adapter: str
    # True: label-restricted softmax scores; False: raw logprobs.
    apply_softmax: bool

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("DecisionRequest.query must be non-empty.")
        if not self.candidates:
            raise ValueError("DecisionRequest.candidates must be non-empty.")
        if self.style not in ALL_STYLES:
            raise ValueError(
                f"Unknown decision style {self.style!r}; expected one of "
                f"{ALL_STYLES}."
            )


@dataclass(frozen=True)
class ScoreCall:
    """The low-level ``/v1/score`` call an adapter compiles a request into.

    Mirrors ``Engine.score``: ``items`` are scored independently against
    the shared ``query`` (SIS execution); columns follow ``label_token_ids``
    order.
    """

    query: str
    items: list[str]
    label_token_ids: list[int]
    apply_softmax: bool


@dataclass(frozen=True)
class DecisionResult:
    """The decision extracted from the raw score rows.

    ``probabilities`` is populated only when the request had
    ``apply_softmax=True``: per-candidate Yes probability for
    ``pointwise_yesno``, per-label probability for ``fused_choice``. These
    are single-forward readouts, not calibrated correctness probabilities.
    """

    # Selected candidate text and its index in ``candidates``.
    answer: str
    answer_index: int
    # Raw score rows as returned by the engine (one per candidate for
    # pointwise; a single row for fused_choice).
    scores: list[list[float]]
    probabilities: list[float] | None
