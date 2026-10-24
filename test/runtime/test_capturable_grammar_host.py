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

"""The capturable grammar's host node, with real xgrammar matchers behind
the engine host's grammar entries (``grammar_test_utils``).

The node is C++ that runs on the CUDA callback thread: it must finish
without the GIL, advance each forward's matchers exactly once (from the
next forward's node or from post_process), and read the sampled tokens at
the output layout's offsets.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch
import xgrammar as xgr

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the host node builds against the CUDA runtime",
)

from test.runtime.grammar_test_utils import handle_known, host_grammar  # noqa: E402

from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.grammar.capturable_grammar import (  # noqa: E402
    CapturableGrammarExecutor,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)

# 70 tokens, so a mask row is three int32 words. Token 0 ends the grammar.
EOS, A, B, C, BRACE = 0, 1, 33, 65, 40
VOCAB = [f"t{i}" for i in range(70)]
for _id, _text in ((EOS, "<eos>"), (A, "a"), (B, "b"), (C, "c"), (BRACE, "}")):
    VOCAB[_id] = _text


@pytest.fixture(scope="module")
def compile_grammar():
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(VOCAB, stop_token_ids=[EOS]))
    return compiler.compile_grammar


def _grammar(compiled, accepted=()):
    return host_grammar(compiled, accepted)


def _next_mask(grammar):
    mask = torch.zeros((1, 3), dtype=torch.int32)
    grammar.fill_vocab_mask(mask, 0)
    return mask[0]


def _executor(max_bs, tokens_per_req=1):
    return CapturableGrammarExecutor(
        max_bs=max_bs, vocab_size=len(VOCAB), max_tokens_per_req=tokens_per_req
    )


def _launch(executor):
    executor.host.launch(torch.cuda.current_stream().cuda_stream)


def test_host_node_advances_and_fills_without_the_gil(compile_grammar):
    compiled = compile_grammar('root ::= "a" ("b" | "c")* "}"')
    executor = _executor(max_bs=1)
    layout = ForwardOutputLayout(0, 0, 1, 1)
    grammar = _grammar(compiled)
    first = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
    _launch(executor)
    torch.cuda.synchronize()
    executor.output_tokens[0] = A
    executor.accept_lengths[0] = 1
    executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)

    done_device = torch.ones(1, dtype=torch.int32, device="cuda")
    done_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    done = done_host.numpy()
    interval = sys.getswitchinterval()
    # Nothing in the loop releases the GIL, and no switch is forced.
    sys.setswitchinterval(1000.0)
    try:
        # The node runs after the launch returns, while this thread spins.
        torch.cuda._sleep(20_000_000)
        _launch(executor)
        done_host.copy_(done_device, non_blocking=True)
        deadline = time.perf_counter() + 2.0
        while done[0] == 0 and time.perf_counter() < deadline:
            pass
        completed = bool(done[0])
    finally:
        sys.setswitchinterval(interval)
    torch.cuda.synchronize()

    assert completed
    assert first.wait(0) and first.terminated_at() == [-1]
    assert torch.equal(executor.host_bitmask[0], _next_mask(_grammar(compiled, [A])))


def test_grammar_steps_replay_in_a_cuda_graph(compile_grammar):
    compiled = compile_grammar('root ::= "a" ("b" | "c")* "}"')
    bs = 2
    executor = _executor(max_bs=bs)
    layout = ForwardOutputLayout(0, 0, bs, 1)
    sampled = torch.zeros(bs, dtype=torch.int32, device="cuda")
    accept = torch.ones(bs, dtype=torch.int32, device="cuda")
    seen_mask = torch.zeros(bs, 3, dtype=torch.int32, device="cuda")

    def step():
        executor.schedule_fill(None, candidate_start=0)
        executor.wait_bitmask()
        seen_mask.copy_(executor.bitmask[:bs])
        executor.schedule_post_sampler(sampled, accept)

    def add(grammars):
        return executor.add_batch(grammars=grammars, bs=bs, output_layout=layout)

    # As ForwardStepRunner captures: an eager warmup, then the capture.
    add([None] * bs)
    step()
    torch.cuda.synchronize()
    add([None] * bs)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.synchronize()
    executor.reset_state()
    assert len(executor.host) == 0

    def replay(grammars, token):
        completion = add(grammars)
        sampled.fill_(token)
        graph.replay()
        torch.cuda.synchronize()
        assert len(executor.host) == 0
        return completion, seen_mask.cpu()

    grammar = _grammar(compiled)
    allow_all = torch.full((3,), -1, dtype=torch.int32)
    first, mask = replay([grammar, None], A)
    assert torch.equal(mask[0], _next_mask(_grammar(compiled)))
    assert torch.equal(mask[1], allow_all)
    assert not first.wait(0)

    # The next forward advances the matcher by the sampled token and resets
    # its mask row.
    second, mask = replay([None, None], B)
    assert (mask == -1).all()
    assert first.wait(0) and first.terminated_at() == [-1, -1]
    assert second.wait(0) and second.terminated_at() == [-1, -1]
    assert torch.equal(_next_mask(grammar), _next_mask(_grammar(compiled, [A])))

    third, mask = replay([None, grammar], BRACE)
    assert torch.equal(mask[1], _next_mask(_grammar(compiled, [A])))
    fourth, mask = replay([None, grammar], EOS)
    assert third.terminated_at() == [-1, -1]
    assert torch.equal(mask[1], _next_mask(_grammar(compiled, [A, BRACE])))
    replay([None, None], A)
    assert fourth.terminated_at() == [-1, 0]
    assert grammar.is_terminated()


def test_host_advance_and_host_node_advance_once(compile_grammar):
    # Two "b"s still allow a third, three allow only "c": an extra advance shows.
    compiled = compile_grammar('root ::= "b" "b" "b" "c" "}"')
    executor = _executor(max_bs=1)
    layout = ForwardOutputLayout(0, 0, 1, 1)
    grammar = _grammar(compiled)
    first = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
    _launch(executor)
    torch.cuda.synchronize()
    assert not first.wait(0.001)

    # No next forward came in time: post_process advances on the host, and
    # the node that pops the next batch leaves this one alone.
    first.advance_on_host([B], [1])
    assert first.terminated_at() == [-1]
    executor.output_tokens[0] = B
    executor.accept_lengths[0] = 1
    second = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
    _launch(executor)
    torch.cuda.synchronize()
    assert torch.equal(executor.host_bitmask[0], _next_mask(_grammar(compiled, [B])))

    # The node advances the second batch; a later host advance does nothing.
    executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
    _launch(executor)
    torch.cuda.synchronize()
    second.advance_on_host([B], [1])
    assert torch.equal(_next_mask(grammar), _next_mask(_grammar(compiled, [B, B])))


def test_host_advance_racing_the_host_node_advances_once(compile_grammar):
    # A long advance, so the two sides overlap: one advance of ``width`` "b"s
    # leaves a "b" allowed, a second one leaves only "c".
    width = 1024
    compiled = compile_grammar(f'root ::= "b"{{{2 * width}}} "c"')
    executor = _executor(max_bs=1, tokens_per_req=width)
    layout = ForwardOutputLayout(0, 0, 1, width)
    executor.output_tokens[:] = B
    executor.accept_lengths[0] = width
    once = _next_mask(_grammar(compiled, [B] * width))
    grammar = _grammar(compiled)
    previous = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
    _launch(executor)
    for step in range(40):
        # As in serving, a forward's node has run before its post_process.
        torch.cuda.synchronize()
        advanced, grammar = grammar, _grammar(compiled)
        batch = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
        if step % 2:
            # The host advances first; otherwise the node is mid-advance.
            torch.cuda._sleep(2_000_000)
        _launch(executor)
        previous.advance_on_host([B] * width, [width])
        assert previous.terminated_at() == [-1]
        torch.cuda.synchronize()
        assert torch.equal(_next_mask(advanced), once)
        previous = batch


def test_the_host_node_fills_after_a_host_advance_it_waited_for(compile_grammar):
    # Mid-group a matcher allows "b" or "c"; only a whole group allows "}".
    width = 1024
    compiled = compile_grammar(f'root ::= ("b"{{{width - 1}}} "c")* "}}"')
    executor = _executor(max_bs=1, tokens_per_req=width)
    layout = ForwardOutputLayout(0, 0, 1, width)
    group = [B] * (width - 1) + [C]
    executor.output_tokens[:] = torch.tensor(group)
    executor.accept_lengths[0] = width
    whole = _next_mask(_grammar(compiled, group))
    # As in serving, consecutive forwards share the request's matcher.
    grammar = _grammar(compiled)
    previous = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
    _launch(executor)
    for step in range(40):
        torch.cuda.synchronize()
        batch = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
        # The node comes at a later point of the host's advance each step.
        torch.cuda._sleep(50_000 * step)
        _launch(executor)
        previous.advance_on_host(group, [width])
        torch.cuda.synchronize()
        assert torch.equal(executor.host_bitmask[0], whole)
        previous = batch


def test_a_queued_host_node_outlives_its_executor(compile_grammar):
    import gc

    from tokenspeed.runtime.grammar import capturable_grammar

    compiled = compile_grammar('root ::= "a" "}"')
    executor = _executor(max_bs=1)
    executor.add_batch(
        grammars=[_grammar(compiled)],
        bs=1,
        output_layout=ForwardOutputLayout(0, 0, 1, 1),
    )
    torch.cuda._sleep(20_000_000)
    _launch(executor)
    del executor
    gc.collect()
    torch.cuda.synchronize()
    host_bitmask = capturable_grammar._LIVE_HOSTS[-1][1]
    assert torch.equal(host_bitmask[0], _next_mask(_grammar(compiled)))


def test_a_queued_batch_keeps_the_matcher_past_its_release(compile_grammar):
    compiled = compile_grammar('root ::= "a" ("b" | "c")* "}"')
    executor = _executor(max_bs=1)
    layout = ForwardOutputLayout(0, 0, 1, 1)
    grammar = _grammar(compiled)
    first = executor.add_batch(grammars=[grammar], bs=1, output_layout=layout)
    # The request ends while its forward is queued: that forward's node
    # still fills from the matcher, and the next one still advances it.
    grammar.release()
    _launch(executor)
    torch.cuda.synchronize()
    assert torch.equal(executor.host_bitmask[0], _next_mask(_grammar(compiled)))
    executor.output_tokens[0] = A
    executor.accept_lengths[0] = 1
    executor.add_batch(grammars=[None], bs=1, output_layout=layout)
    _launch(executor)
    torch.cuda.synchronize()
    assert first.wait(0) and first.terminated_at() == [-1]
    assert torch.equal(_next_mask(grammar), _next_mask(_grammar(compiled, [A])))
    del first
    assert handle_known(grammar.handle)
    # The next add drops the batch the node is done with, and the last hold.
    executor.add_batch(grammars=[None], bs=1, output_layout=layout)
    assert not handle_known(grammar.handle)


@pytest.mark.parametrize("on_host", [False, True])
def test_advance_reads_tokens_at_the_output_layout_offsets(compile_grammar, on_host):
    compiled = compile_grammar('root ::= "a" "b" "c" "}"')
    # Three completing prefills (the second one's token is not to advance),
    # an incomplete one (no output) and three width-3 decodes.
    layout = ForwardOutputLayout(4, 3, 3, 3)
    executor = _executor(max_bs=7, tokens_per_req=3)
    accepted = ([], [], [A], [], [A], [A], [A, B, C])
    grammars = [_grammar(compiled, tokens) for tokens in accepted]
    # Reading another request's tokens, or reading on past a rejected token,
    # changes what a matcher accepts.
    tokens = [A, A, B, B, C, BRACE, C, B, C, BRACE, EOS, A]
    lengths = [1, 1, 1, 0, 2, 3, 3]
    batch = executor.add_batch(
        grammars=grammars,
        bs=7,
        advance_mask=[True, False, True, True, True, True, True],
        output_layout=layout,
    )
    _launch(executor)
    torch.cuda.synchronize()
    if on_host:
        batch.advance_on_host(tokens, lengths)
    else:
        executor.output_tokens[: len(tokens)] = torch.tensor(tokens)
        executor.accept_lengths[:7] = torch.tensor(lengths)
        executor.add_batch(grammars=[None] * 7, bs=7, output_layout=layout)
        _launch(executor)
        torch.cuda.synchronize()
    # The last decode ends at its second token and leaves the third.
    assert batch.terminated_at() == [-1, -1, -1, -1, -1, -1, 1]
    assert grammars[6].is_terminated()
    for grammar, now in zip(grammars, ([A], [], [A, B], [], [A, B, C], [A])):
        assert torch.equal(_next_mask(grammar), _next_mask(_grammar(compiled, now)))


def test_batches_are_checked_against_their_layout(compile_grammar):
    compiled = compile_grammar('root ::= "a" "}"')
    executor = _executor(max_bs=2)
    with pytest.raises(IndexError):
        executor.add_batch(
            grammars=[None, _grammar(compiled)],
            bs=2,
            output_layout=ForwardOutputLayout(0, 0, 1, 1),
        )
    with pytest.raises(ValueError):
        executor.add_batch(
            grammars=[None] * 3, bs=3, output_layout=ForwardOutputLayout(0, 0, 3, 1)
        )
    assert len(executor.host) == 0
    # A batch without grammars completes as it is added.
    idle = executor.add_batch(
        grammars=[None, None], bs=2, output_layout=ForwardOutputLayout(0, 0, 1, 1)
    )
    assert idle.wait(0) and idle.terminated_at() == [-1, -1]
