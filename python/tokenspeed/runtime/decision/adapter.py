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

"""Model-family adapters for decision requests.

The adapter is the single owner of everything model-specific about a
decision request: the prompt scaffold (how candidates are presented), the
label vocabulary (which strings score each judgment), and the translation
of raw score rows back into a decision. The engine below it only knows
the ``ScoreCall`` contract; see docs/design/scoring.md.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tokenspeed.runtime.decision.types import (
    STYLE_FUSED_CHOICE,
    STYLE_POINTWISE_YESNO,
    DecisionRequest,
    DecisionResult,
    ScoreCall,
)

# Label vocabularies. Every label must encode to exactly one token with the
# served model's tokenizer — enforced at compile time by _label_token_ids.
YES_NO_LABELS = ("Yes", "No")
LETTER_LABELS = tuple(string.ascii_uppercase)


def _label_token_ids(labels: Sequence[str], tokenizer, family: str) -> list[int]:
    """Resolve label strings to single token ids, failing loudly otherwise.

    A label that tokenizes to more than one token would silently break the
    readout (scores would gather only its first token's probability), so
    this is a hard error, not a clamp.
    """
    token_ids = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Decision adapter {family!r}: label {label!r} encodes to "
                f"{len(ids)} tokens ({ids}) with the served tokenizer; every "
                "scoring label must be exactly one token. Pick a different "
                "label or tokenize client-side and call the score API with "
                "explicit label_token_ids."
            )
        token_ids.append(ids[0])
    return token_ids


@runtime_checkable
class DecisionAdapter(Protocol):
    """Compiles a ``DecisionRequest`` into a ``ScoreCall`` and extracts the
    decision from the returned score rows."""

    family: str

    def compile(self, req: DecisionRequest, tokenizer) -> ScoreCall:
        """Build the score call: item texts, label ids, softmax flag."""
        ...

    def extract(
        self, req: DecisionRequest, score_rows: list[list[float]]
    ) -> DecisionResult:
        """Translate raw score rows (one per item, columns in label order)
        into the selected candidate and optional probabilities."""
        ...


class GenericDecisionAdapter:
    """Family-agnostic adapter: Open-Jev-style Yes/No pointwise judgments
    and A/B/C fused choice.

    Registered under every supported family name for v1; a family with
    genuinely different scaffold or label needs gets its own subclass
    rather than a branch here.
    """

    def __init__(self, family: str = "generic") -> None:
        self.family = family

    def compile(self, req: DecisionRequest, tokenizer) -> ScoreCall:
        if req.style == STYLE_POINTWISE_YESNO:
            items = [
                f"Proposed answer: {candidate}\n"
                "Is this proposed answer correct? Answer Yes or No."
                for candidate in req.candidates
            ]
            labels = list(YES_NO_LABELS)
        elif req.style == STYLE_FUSED_CHOICE:
            if len(req.candidates) > len(LETTER_LABELS):
                raise ValueError(
                    f"fused_choice supports at most {len(LETTER_LABELS)} "
                    f"candidates, got {len(req.candidates)}."
                )
            options = "\n".join(
                f"{LETTER_LABELS[i]}) {candidate}"
                for i, candidate in enumerate(req.candidates)
            )
            letters = LETTER_LABELS[: len(req.candidates)]
            items = [
                f"{options}\n"
                "Choose the single correct option above. Your entire response "
                "must be exactly one letter from: "
                f"{', '.join(letters)}. Do not include any other words, "
                "punctuation, or explanation."
            ]
            labels = list(letters)
        else:
            # DecisionRequest.__post_init__ already rejects unknown styles.
            raise AssertionError(f"unreachable style {req.style!r}")

        return ScoreCall(
            query=req.query,
            items=items,
            label_token_ids=_label_token_ids(labels, tokenizer, self.family),
            apply_softmax=req.apply_softmax,
        )

    def extract(
        self, req: DecisionRequest, score_rows: list[list[float]]
    ) -> DecisionResult:
        if req.style == STYLE_POINTWISE_YESNO:
            if len(score_rows) != len(req.candidates):
                raise ValueError(
                    f"Expected {len(req.candidates)} score rows, got "
                    f"{len(score_rows)}."
                )
            # Column 0 is Yes (label order fixed by compile()).
            yes_scores = [row[0] for row in score_rows]
            answer_index = max(range(len(yes_scores)), key=yes_scores.__getitem__)
            probabilities = yes_scores if req.apply_softmax else None
        else:  # fused_choice
            if len(score_rows) != 1:
                raise ValueError(
                    f"fused_choice expects exactly 1 score row, got "
                    f"{len(score_rows)}."
                )
            row = score_rows[0]
            answer_index = max(range(len(row)), key=row.__getitem__)
            probabilities = row if req.apply_softmax else None

        return DecisionResult(
            answer=req.candidates[answer_index],
            answer_index=answer_index,
            scores=score_rows,
            probabilities=probabilities,
        )
