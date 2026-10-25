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

"""The capturable grammar's host node writes the eager walk's masks.

Real xgrammar matchers on a small vocabulary, behind the engine host's
grammar entries (``grammar_test_utils``). The eager producer walks the
draft chain position by position; the host node, C++ that calls the
matchers without the GIL, fills it with one matcher call and resets only
the rows it wrote before. Both must give the same bits on every bound row,
over steps whose batches change.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import xgrammar as xgr

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the host node builds against the CUDA runtime",
)

from test.runtime.grammar_test_utils import host_grammar  # noqa: E402

from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.grammar.capturable_grammar import (  # noqa: E402
    CapturableGrammarExecutor,
    _fill_eager_bitmask,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)

# 70 tokens, so a mask row is three int32 words; the grammar's tokens sit in
# all three. Token 0 ends the grammar.
EOS, A, B, AB, BRACE, C, BC, JUNK = 0, 1, 33, 34, 40, 65, 66, 7
VOCAB = [f"t{i}" for i in range(70)]
for _id, _text in ((EOS, "<eos>"), (A, "a"), (B, "b"), (AB, "ab")):
    VOCAB[_id] = _text
for _id, _text in ((BRACE, "}"), (C, "c"), (BC, "bc")):
    VOCAB[_id] = _text
W = 4


@pytest.fixture(scope="module")
def compiled():
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[EOS])
    return xgr.GrammarCompiler(info).compile_grammar('root ::= "a" ("b" | "c")* "}"')


def _grammar(compiled, accepted):
    return host_grammar(compiled, accepted)


# (accepted so far, drafts; column 0 is the last accepted token) per request.
# Request 0 is a completing prefill, request 1 an incomplete one (no output).
STEPS = [
    [
        ((), None),
        ((), None),
        ((A,), (A, B, C, BRACE)),  # every draft fits
        ((A,), (A, B, JUNK, C)),  # rejected at position 2
        ((A, B), (B, BRACE, EOS, B)),  # ends at position 2
        ((), (0, A, 10**6, B)),  # out of the vocabulary at position 2
        ((A, BRACE), (BRACE, EOS, A, B)),  # ends at position 1
        ((A, BRACE, EOS), (EOS, A, B, C)),  # already terminated
        (None, (A, B, C, BRACE)),  # no grammar
    ],
    [
        (None, None),
        ((A,), None),
        (None, (A, B, C, BRACE)),
        ((A, B), (B, C, BC, BRACE)),
        ((A,), (A, AB, B, C)),  # rejected at position 1
        (None, (0, A, 10**6, B)),
        ((), (0, A, B, BRACE)),
        ((A, C, C), (C, BRACE, EOS, EOS)),
        ((A,), (A, BRACE, EOS, A)),
    ],
    [(None, None)] * 2 + [(None, (A, B, C, BRACE))] * 7,
]


def _batch(compiled, step):
    grammars = [None if a is None else _grammar(compiled, a) for a, _ in step]
    drafts = torch.tensor(
        [d if d is not None else (0,) * W for _, d in step], dtype=torch.int32
    )
    return grammars, drafts


def _eager_masks(grammars, drafts, layout, monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None),
    )
    bs = len(grammars)
    buffers = SimpleNamespace(
        candidates_cpu_buf=torch.zeros((bs, W), dtype=torch.int32),
        vocab_mask_spec_cpu_buf=torch.zeros((bs * W, 3), dtype=torch.int32),
        vocab_mask_spec_buf=torch.zeros((bs * W, 3), dtype=torch.int32),
    )
    decodes = drafts[layout.num_extends :].flatten()
    _fill_eager_bitmask(grammars, bs, buffers, W, True, decodes, layout)
    return buffers.vocab_mask_spec_buf


def _host_fill(executor, grammars, layout, *, drafts=None):
    """One execution of the host node for this batch. With nothing sampled
    (accept lengths are zero) the previous batch's matchers stay put."""
    if drafts is not None:
        executor.candidates[: len(grammars)] = drafts
    executor.add_batch(
        grammars=grammars,
        bs=len(grammars),
        has_candidates=drafts is not None,
        output_layout=layout,
    )
    executor.host.launch(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return executor.host_bitmask


def _next_masks(grammars):
    masks = torch.zeros((len(grammars), 3), dtype=torch.int32)
    for i, grammar in enumerate(grammars):
        if grammar is not None and not grammar.is_terminated():
            grammar.fill_vocab_mask(masks, i)
    return masks


def test_draft_chain_masks_match_the_eager_walk_across_steps(compiled, monkeypatch):
    layout = ForwardOutputLayout(2, 1, 7, W)
    executor = CapturableGrammarExecutor(
        max_bs=len(STEPS[0]), vocab_size=len(VOCAB), max_tokens_per_req=W
    )
    for step in STEPS:
        grammars, drafts = _batch(compiled, step)
        expected = _eager_masks(
            [None if a is None else _grammar(compiled, a) for a, _ in step],
            drafts,
            layout,
            monkeypatch,
        )
        before = _next_masks(grammars)
        assert torch.equal(
            _host_fill(executor, grammars, layout, drafts=drafts), expected
        )
        # The walk leaves every matcher where it was.
        assert torch.equal(_next_masks(grammars), before)
    assert (executor.host_bitmask == -1).all()


def test_draft_past_the_vocabulary_ends_the_walk(monkeypatch):
    # Almost every token fits, so the mask sets the bits past the 70-token
    # vocabulary in its last word too; drafts 80 and 95 are still rejected.
    compiled = xgr.GrammarCompiler(
        xgr.TokenizerInfo(VOCAB, stop_token_ids=[EOS])
    ).compile_grammar('root ::= [a-z0-9]* "}"')
    step = [((), (A, B, 80, C)), ((), (A, 95, B, C))]
    layout = ForwardOutputLayout(0, 0, len(step), W)
    executor = CapturableGrammarExecutor(
        max_bs=len(step), vocab_size=len(VOCAB), max_tokens_per_req=W
    )
    grammars, drafts = _batch(compiled, step)
    expected = _eager_masks(_batch(compiled, step)[0], drafts, layout, monkeypatch)
    assert torch.equal(_host_fill(executor, grammars, layout, drafts=drafts), expected)


def test_single_row_masks_reset_rows_left_by_a_previous_step(compiled):
    executor = CapturableGrammarExecutor(
        max_bs=3, vocab_size=len(VOCAB), max_tokens_per_req=1
    )
    layout = ForwardOutputLayout(0, 0, 3, 1)
    for accepted in ([(A,), (), (A, B)], [None, (A, BRACE), None]):
        grammars = [None if a is None else _grammar(compiled, a) for a in accepted]
        expected = _next_masks(grammars)
        expected[[g is None for g in grammars]] = -1
        assert torch.equal(_host_fill(executor, grammars, layout), expected)
