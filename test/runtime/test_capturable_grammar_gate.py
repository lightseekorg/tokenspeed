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

"""The gated hostfunc and the grammar steps that skip it, in CUDA graphs.

A skipped execution must return without the GIL, and the captured grammar
pipeline must run its build exactly for the steps that have grammar work:
a batch with a grammar, and the step after it, which advances those
matchers and resets their mask rows.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.grammar.capturable_grammar import (  # noqa: E402
    CapturableGrammarExecutor,
)
from tokenspeed.runtime.utils.hostfunc import (  # noqa: E402
    launch_hostfunc,
    make_hostfunc_gate,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _capture(fn):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    return graph


def test_gate_decides_each_execution_in_launch_order():
    gate = make_hostfunc_gate()
    calls = []
    graph = _capture(lambda: launch_hostfunc(calls.append, "replay", gate=gate))
    assert len(gate) == 0

    for run in (True, False, False, True):
        gate.push(run)
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    assert calls == ["replay", "replay"]
    assert len(gate) == 0

    # Eager launches pop the same FIFO; an empty gate runs, as does no gate.
    gate.push(False)
    launch_hostfunc(calls.append, "eager", gate=gate)
    launch_hostfunc(calls.append, "empty", gate=gate)
    launch_hostfunc(calls.append, "ungated", gate=None)
    torch.cuda.synchronize()
    assert calls == ["replay", "replay", "empty", "ungated"]

    gate.push(True)
    gate.clear()
    assert len(gate) == 0


def test_skipped_execution_does_not_take_the_gil():
    gate = make_hostfunc_gate()
    calls = []
    done_device = torch.ones(1, dtype=torch.int32, device="cuda")
    done_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    done = done_host.numpy()

    def body():
        # The callback fires after replay() returns, while this thread spins.
        torch.cuda._sleep(20_000_000)
        launch_hostfunc(calls.append, 1, gate=gate)
        done_host.copy_(done_device, non_blocking=True)

    graph = _capture(body)

    def completes_while_holding_the_gil(run):
        done[0] = 0
        gate.push(run)
        interval = sys.getswitchinterval()
        # Nothing in the loop releases the GIL, and no switch is forced.
        sys.setswitchinterval(1000.0)
        try:
            graph.replay()
            deadline = time.perf_counter() + 2.0
            while done[0] == 0 and time.perf_counter() < deadline:
                pass
            completed = bool(done[0])
        finally:
            sys.setswitchinterval(interval)
        torch.cuda.synchronize()
        return completed

    assert completes_while_holding_the_gil(False)
    assert calls == []
    # The check is live: a run needs the GIL, so it waits for the sync.
    assert not completes_while_holding_the_gil(True)
    assert calls == [1]


class _Grammar:
    """Matcher double: masks its row with its tag, records accepted tokens."""

    finished = False

    def __init__(self, tag):
        self.tag = tag
        self.accepted = []

    def is_terminated(self):
        return False

    def accept_token(self, token):
        self.accepted.append(token)

    def fill_vocab_mask(self, mask, row):
        mask[row] = self.tag


def test_grammar_build_runs_only_for_steps_with_grammar_work(monkeypatch):
    bs = 2
    executor = CapturableGrammarExecutor(
        max_bs=bs, vocab_size=64, max_tokens_per_req=1, device="cuda"
    )
    layout = ForwardOutputLayout(0, 0, bs, 1)
    builds = []
    build = executor.build

    def counted_build():
        builds.append(True)
        build()

    monkeypatch.setattr(executor, "build", counted_build)
    sampled = torch.zeros(bs, dtype=torch.int32, device="cuda")
    accept = torch.ones(bs, dtype=torch.int32, device="cuda")
    seen_mask = torch.zeros(bs, 2, dtype=torch.int32, device="cuda")

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
    graph = _capture(step)
    executor.reset_state()
    builds.clear()

    def replay(grammars, token):
        completion = add(grammars)
        sampled.fill_(token)
        graph.replay()
        torch.cuda.synchronize()
        assert len(executor.gate) == 0
        return completion, seen_mask[:, 0].tolist()

    grammar = _Grammar(5)
    first, mask = replay([grammar, None], 7)
    assert (len(builds), mask) == (1, [5, -1])
    assert not first.event.is_set()

    # The next step runs without a grammar: it advances the matcher by the
    # sampled token and resets its mask row.
    second, mask = replay([None, None], 9)
    assert (len(builds), mask) == (2, [-1, -1])
    assert grammar.accepted == [7]
    assert first.event.is_set() and first.terminated_at == [-1, -1]
    assert second.event.is_set()

    # Nothing pending: the build is skipped.
    third, mask = replay([None, None], 11)
    assert (len(builds), mask) == (2, [-1, -1])
    assert third.event.is_set() and third.terminated_at == [-1, -1]
    assert grammar.accepted == [7]

    _, mask = replay([None, grammar], 13)
    assert (len(builds), mask) == (3, [-1, 5])
    assert grammar.accepted == [7]
