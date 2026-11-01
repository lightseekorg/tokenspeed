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

"""Both grammar runtimes write the masks of a position-by-position walk.

Real xgrammar matchers on a small vocabulary, behind the engine host's
grammar entries (``grammar_test_utils``). The reference walks the draft
chain position by position; the C++ fill, which calls the matchers without
the GIL, fills it with one matcher call and resets only the rows it wrote
before. It runs as the CUDA host node or, without one, before the forward.
Both must give the reference's bits on every bound row, over steps whose
batches change.
"""

from __future__ import annotations

import ctypes
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import xgrammar as xgr

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from test.runtime.grammar_test_utils import host_grammar  # noqa: E402

from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.grammar.capturable_grammar import (  # noqa: E402
    CapturableGrammarExecutor,
    EagerGrammarBuffers,
    GrammarStepInputs,
    setup_grammar_step,
)
from tokenspeed.runtime.grammar.grammar_host import (  # noqa: E402
    load_grammar_host_ext,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
RUNTIMES = [
    pytest.param(
        "host_node",
        marks=pytest.mark.skipif(
            torch.version.hip is not None,
            reason="the host node builds against the CUDA runtime",
        ),
    ),
    "eager",
]

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
# Requests 0 and 1 are prefills; request 1 completes only in some layouts.
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


def _reference_masks(grammars, drafts, layout):
    """Each position's mask after the drafts before it, walked one token at
    a time; rows the walk does not reach allow everything."""
    masks = torch.full((len(grammars) * W, 3), -1, dtype=torch.int32)
    for i, grammar in enumerate(grammars):
        if grammar is None or grammar.is_terminated():
            continue
        advanced = 0
        positions = layout.output_width(i)
        for pos in range(positions):
            if grammar.is_terminated():
                break
            grammar.fill_vocab_mask(masks, i * W + pos)
            if pos + 1 == positions or not grammar.try_accept_token(
                int(drafts[i, pos + 1])
            ):
                break
            advanced += 1
        if advanced:
            grammar.rollback(advanced)
    return masks


def _runtime(kind, max_bs, tokens_per_req):
    runtime_cls = (
        CapturableGrammarExecutor if kind == "host_node" else EagerGrammarBuffers
    )
    return runtime_cls(
        max_bs=max_bs, vocab_size=len(VOCAB), max_tokens_per_req=tokens_per_req
    )


def _fill(runtime, grammars, layout, *, drafts=None):
    """One forward's fill of this batch: the bound rows of the bitmask. With
    nothing sampled (accept lengths are zero) the host node leaves the
    previous batch's matchers where they were; without a host node the
    forward's own setup fills, or clears when no request has a grammar."""
    bs = len(grammars)
    has_candidates = drafts is not None
    if isinstance(runtime, EagerGrammarBuffers):
        return _setup(runtime, grammars, layout, drafts=drafts).vocab_mask.cpu()
    if has_candidates:
        runtime.candidates[:bs] = drafts
    runtime.add_batch(
        grammars=grammars,
        bs=bs,
        has_candidates=has_candidates,
        output_layout=layout,
    )
    runtime.host.launch(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    bitmask = runtime.host_bitmask
    return (
        bitmask[: bs * runtime.max_tokens_per_req] if has_candidates else bitmask[:bs]
    )


def _setup(runtime, grammars, layout, *, drafts=None):
    """The eager runtime's setup of one forward: the sampling info it binds.
    The commit advances the matchers, so setup returns no completion."""
    # The forward's input: prefill tokens, then the decode rows' drafts.
    input_ids = torch.full((5,), JUNK, dtype=torch.int32)
    if drafts is not None:
        input_ids = torch.cat([input_ids, drafts[layout.num_extends :].flatten()])
    input_ids = input_ids.cuda()
    # Queued work keeps the drafts' copy to the host pending until setup waits for it.
    torch.cuda._sleep(5_000_000)
    info = SimpleNamespace()
    completion = setup_grammar_step(
        sampling_info=info,
        bs=len(grammars),
        is_spec_decode=drafts is not None,
        grammar_inputs=GrammarStepInputs(grammars) if any(grammars) else None,
        grammar_runtime=runtime,
        input_ids_buf=input_ids,
        grammar_backend="xgrammar",
        output_layout=layout,
    )
    assert completion is None
    torch.cuda.synchronize()
    return info


def _next_masks(grammars):
    masks = torch.zeros((len(grammars), 3), dtype=torch.int32)
    for i, grammar in enumerate(grammars):
        if grammar is not None and not grammar.is_terminated():
            grammar.fill_vocab_mask(masks, i)
    return masks


@pytest.mark.parametrize("prefill_outputs", [1, 2])
@pytest.mark.parametrize("kind", RUNTIMES)
def test_draft_chain_masks_match_the_reference_walk_across_steps(
    compiled, kind, prefill_outputs
):
    layout = ForwardOutputLayout(2, prefill_outputs, 7, W)
    runtime = _runtime(kind, max_bs=len(STEPS[0]), tokens_per_req=W)
    for step in STEPS:
        grammars, drafts = _batch(compiled, step)
        expected = _reference_masks(_batch(compiled, step)[0], drafts, layout)
        before = _next_masks(grammars)
        assert torch.equal(_fill(runtime, grammars, layout, drafts=drafts), expected)
        # The walk leaves every matcher where it was.
        assert torch.equal(_next_masks(grammars), before)
    if kind == "host_node":
        assert (runtime.host_bitmask == -1).all()


@pytest.mark.parametrize("kind", RUNTIMES)
def test_draft_past_the_vocabulary_ends_the_walk(kind):
    # Almost every token fits, so the mask sets the bits past the 70-token
    # vocabulary in its last word too; drafts 80 and 95 are still rejected.
    compiled = xgr.GrammarCompiler(
        xgr.TokenizerInfo(VOCAB, stop_token_ids=[EOS])
    ).compile_grammar('root ::= [a-z0-9]* "}"')
    step = [((), (A, B, 80, C)), ((), (A, 95, B, C))]
    layout = ForwardOutputLayout(0, 0, len(step), W)
    runtime = _runtime(kind, max_bs=len(step), tokens_per_req=W)
    grammars, drafts = _batch(compiled, step)
    expected = _reference_masks(_batch(compiled, step)[0], drafts, layout)
    assert torch.equal(_fill(runtime, grammars, layout, drafts=drafts), expected)


@pytest.mark.parametrize("kind", RUNTIMES)
def test_single_row_masks_reset_rows_left_by_a_previous_step(compiled, kind):
    runtime = _runtime(kind, max_bs=3, tokens_per_req=1)
    layout = ForwardOutputLayout(0, 0, 3, 1)
    for accepted in ([(A,), (), (A, B)], [None, (A, BRACE), None]):
        grammars = [None if a is None else _grammar(compiled, a) for a in accepted]
        expected = _next_masks(grammars)
        expected[[g is None for g in grammars]] = -1
        assert torch.equal(_fill(runtime, grammars, layout), expected)


@pytest.mark.parametrize("kind", RUNTIMES)
def test_one_runtime_serves_verify_plain_and_grammar_free_steps(compiled, kind):
    # A runtime sized for verify also binds plain steps at one row per request.
    runtime = _runtime(kind, max_bs=3, tokens_per_req=W)
    verify = STEPS[0][2:5]
    verify_layout = ForwardOutputLayout(0, 0, 3, W)
    plain_layout = ForwardOutputLayout(2, 2, 0, W)
    for _ in range(2):
        grammars, drafts = _batch(compiled, verify)
        expected = _reference_masks(_batch(compiled, verify)[0], drafts, verify_layout)
        assert torch.equal(
            _fill(runtime, grammars, verify_layout, drafts=drafts), expected
        )
        grammars = [_grammar(compiled, a) for a in ((A, B), (A,))]
        assert torch.equal(
            _fill(runtime, grammars, plain_layout), _next_masks(grammars)
        )
        # Nothing a grammar batch wrote is left in the bound rows.
        none = [None] * 3
        assert (_fill(runtime, none, verify_layout, drafts=drafts) == -1).all()
        assert (_fill(runtime, none[:2], plain_layout) == -1).all()


# Entries a C++ matcher may still call while the process runs.
_FAILING_ENTRIES: list = []


def _matcher_that_fills_nothing():
    """A grammar whose entries answer state reads but write no mask."""
    i32, u64, i64, usize = (
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.c_size_t,
    )
    entries = {
        "is_terminated": ctypes.CFUNCTYPE(i32, u64)(lambda handle: 0),
        "accept_token": ctypes.CFUNCTYPE(i32, u64, i32)(lambda handle, token: 1),
        "fill_bitmask": ctypes.CFUNCTYPE(i32, u64, usize, i64, i64, i32)(
            lambda *args: -1
        ),
        "traverse_draft_tree": ctypes.CFUNCTYPE(
            i32, u64, usize, usize, usize, i64, usize, i64
        )(lambda *args: -1),
        "retain": ctypes.CFUNCTYPE(i32, u64)(lambda handle: 0),
        "free": ctypes.CFUNCTYPE(None, u64)(lambda handle: None),
    }
    _FAILING_ENTRIES.append(entries)
    ext = load_grammar_host_ext()
    native = ext.Entries(
        {
            name: ctypes.cast(entry, ctypes.c_void_p).value
            for name, entry in entries.items()
        }
    )
    matcher = ext.Matcher(native, 1, [EOS])
    return SimpleNamespace(finished=False, native_matcher=lambda: matcher)


@pytest.mark.parametrize("verify", [False, True])
def test_without_a_host_node_a_mask_the_matcher_did_not_fill_fails_the_forward(
    verify,
):
    # The host node cannot raise; the eager fill must not sample from every token instead.
    runtime = _runtime("eager", max_bs=1, tokens_per_req=W)
    drafts = torch.tensor([(A, B, C, BRACE)], dtype=torch.int32) if verify else None
    with pytest.raises(RuntimeError, match="did not fill its mask"):
        _setup(
            runtime,
            [_matcher_that_fills_nothing()],
            ForwardOutputLayout(0, 0, 1, W),
            drafts=drafts,
        )
