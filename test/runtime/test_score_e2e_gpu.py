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

"""End-to-end Score API tests against a small model on GPU.

Pins the full vertical: Engine.score reads every declared label at the
answer boundary (no top-k omission), apply_softmax semantics, the
decision adapter path, the raw /generate passthrough, and the
score-only contract rejection.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="score e2e requires a GPU"
)

_MODEL = os.environ.get("TOKENSPEED_SCORE_TEST_MODEL", "Qwen/Qwen2-0.5B-Instruct")

_QUERY = (
    "Context:\nThe customer supplied an order ID and wants to know whether "
    "that order has shipped.\n\n"
    "Question: Which action should the agent take next?\n"
)
_ITEM_CORRECT = (
    "Proposed answer: Query the order-status service.\n"
    "Is this proposed answer correct? Answer Yes or No."
)
_ITEM_WRONG = (
    "Proposed answer: Tell the customer a joke.\n"
    "Is this proposed answer correct? Answer Yes or No."
)


@pytest.fixture(scope="module")
def engine_and_tokenizer():
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=_MODEL,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        log_level="error",
    )
    try:
        yield engine, AutoTokenizer.from_pretrained(_MODEL)
    finally:
        engine.shutdown()


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == 1, f"{text!r} is not a single token: {ids}"
    return ids[0]


def test_score_reads_every_label_and_normalizes(engine_and_tokenizer):
    engine, tokenizer = engine_and_tokenizer
    label_ids = [_single_token_id(tokenizer, t) for t in ("Yes", "No")]

    out = engine.score(
        query=_QUERY,
        items=[_ITEM_CORRECT, _ITEM_WRONG],
        label_token_ids=label_ids,
        apply_softmax=True,
    )

    rows = out["scores"]
    assert len(rows) == 2
    for row in rows:
        assert len(row) == 2
        assert all(0.0 < v < 1.0 for v in row)
        assert sum(row) == pytest.approx(1.0, rel=1e-6)
    # Sanity, not a quality gate: a sensible model prefers the correct action.
    assert rows[0][0] > rows[1][0]


def test_score_raw_logprobs_match_softmax_argmax(engine_and_tokenizer):
    engine, tokenizer = engine_and_tokenizer
    label_ids = [_single_token_id(tokenizer, t) for t in ("Yes", "No")]

    raw = engine.score(
        query=_QUERY,
        items=[_ITEM_CORRECT],
        label_token_ids=label_ids,
        apply_softmax=False,
    )["scores"][0]
    norm = engine.score(
        query=_QUERY,
        items=[_ITEM_CORRECT],
        label_token_ids=label_ids,
        apply_softmax=True,
    )["scores"][0]

    assert all(v <= 0.0 for v in raw)
    assert (raw[0] > raw[1]) == (norm[0] > norm[1])
    # The normalized row is the label-restricted softmax of the raw row.
    exps = [math.exp(v) for v in raw]
    assert norm[0] == pytest.approx(exps[0] / sum(exps), rel=1e-4)


def test_decision_pointwise_end_to_end(engine_and_tokenizer):
    from tokenspeed.runtime.decision import (
        STYLE_POINTWISE_YESNO,
        DecisionRequest,
    )

    engine, _ = engine_and_tokenizer
    result = engine.decision(
        DecisionRequest(
            query=_QUERY,
            candidates=[
                "Query the order-status service.",
                "Tell the customer a joke.",
            ],
            style=STYLE_POINTWISE_YESNO,
            adapter="generic",
            apply_softmax=True,
        )
    )
    assert result.answer_index in (0, 1)
    assert (
        result.answer
        == [
            "Query the order-status service.",
            "Tell the customer a joke.",
        ][result.answer_index]
    )
    assert len(result.probabilities) == 2
    # Same sanity direction as the raw score test.
    assert result.answer == "Query the order-status service."


def test_generate_passthrough_attaches_scores(engine_and_tokenizer):
    engine, tokenizer = engine_and_tokenizer
    label_ids = [_single_token_id(tokenizer, t) for t in ("Yes", "No")]

    out = engine.generate(
        prompt=_QUERY + _ITEM_CORRECT,
        sampling_params={
            "max_new_tokens": 0,
            "score_label_token_ids": label_ids,
            "score_apply_softmax": True,
        },
    )
    row = out["scores"]
    assert len(row) == 2
    assert sum(row) == pytest.approx(1.0, rel=1e-6)


def test_score_rejects_decode_budget(engine_and_tokenizer):
    engine, tokenizer = engine_and_tokenizer
    label_ids = [_single_token_id(tokenizer, t) for t in ("Yes", "No")]

    with pytest.raises(Exception, match="max_new_tokens=0"):
        engine.generate(
            prompt=_QUERY,
            sampling_params={
                "max_new_tokens": 4,
                "score_label_token_ids": label_ids,
            },
        )
