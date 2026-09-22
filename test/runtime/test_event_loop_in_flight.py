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

"""CPU-only tests for the event loop's in-flight commit queue helpers.

These exercise ``_dispatch_depends_on_pending_commit`` (the single registry
of overlap-breaking dependencies) and ``_drain_in_flight`` with fakes, so no
model, CUDA context, or transfer backend is created.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402


def _predicate_loop(*, eager_grammar_buffers=None):
    return SimpleNamespace(_uses_eager_grammar=eager_grammar_buffers is not None)


def test_handoff_shaped_batch_without_pd_keeps_overlap() -> None:
    loop = _predicate_loop()
    op = SimpleNamespace(num_extends=lambda: 0)

    assert not EventLoop._dispatch_depends_on_pending_commit(loop, op, None)


def test_eager_grammar_batch_depends_on_pending_commit() -> None:
    op = SimpleNamespace(num_extends=lambda: 1)
    grammar_inputs = object()

    # Eager grammar reads matcher state that only advances at commit.
    eager = _predicate_loop(eager_grammar_buffers=object())
    assert EventLoop._dispatch_depends_on_pending_commit(eager, op, grammar_inputs)
    # No grammar in the batch: overlap kept.
    assert not EventLoop._dispatch_depends_on_pending_commit(eager, op, None)
    # Capturable grammar (no eager buffers) advances in-graph: overlap kept.
    capturable = _predicate_loop(eager_grammar_buffers=None)
    assert not EventLoop._dispatch_depends_on_pending_commit(
        capturable, op, grammar_inputs
    )


class _DrainHarness:
    """Only the state read by ``EventLoop._drain_in_flight``."""

    def __init__(self) -> None:
        self.committed: list[object] = []

    def _commit_forward_results(self, forward_op, results):
        self.committed.append(forward_op)
        return [f"change-{forward_op}"]


def test_drain_in_flight_commits_oldest_first() -> None:
    loop = _DrainHarness()
    in_flight = deque([("op0", None), ("op1", None)])

    request_changes = EventLoop._drain_in_flight(loop, in_flight)

    assert not in_flight
    assert loop.committed == ["op0", "op1"]
    assert request_changes == ["change-op0", "change-op1"]


def test_per_request_reads_precede_the_dependent_drain() -> None:
    """Every per-request lookup on rid_to_state happens before the drain.

    A drained commit can finish a request and pop it from rid_to_state while
    its id is still in the planned forward_op, so sampling params, grammar
    state, ngram inputs and the batch log's context lengths must all be read
    first. Pinned on the statement order of EventLoop.event_loop.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(EventLoop.event_loop))
    calls = [
        node.func.attr
        for node in sorted(
            (
                n
                for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            ),
            key=lambda n: (n.lineno, n.col_offset),
        )
    ]
    drain = calls.index("_dispatch_depends_on_pending_commit")
    for reader in (
        "_gather_sampling_params",
        "_gather_grammar_state",
        "log_dispatch",
    ):
        assert calls.index(reader) < drain, f"{reader} runs after the drain"


def test_request_context_length_sums_prompt_and_output() -> None:
    state = SimpleNamespace(input_length=1000, output_length=24)
    loop = SimpleNamespace(output_processor=SimpleNamespace(rid_to_state={"r": state}))

    assert EventLoop._request_context_length(loop, "r") == 1024


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
