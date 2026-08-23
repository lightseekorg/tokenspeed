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
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor  # noqa: E402


def _predicate_loop(*, kv_transfer=None, eager_grammar_buffers=None):
    return SimpleNamespace(
        kv_transfer=kv_transfer,
        model_executor=SimpleNamespace(eager_grammar_buffers=eager_grammar_buffers),
    )


def test_pd_handoff_batch_depends_on_pending_commit() -> None:
    # kv_transfer.execute needs the final chunk's bootstrap token, which only
    # lands at commit — but only for the P-side handoff batch (no extends).
    prefill_executor = object.__new__(DisaggPrefillExecutor)
    loop = _predicate_loop(kv_transfer=prefill_executor)
    handoff_op = SimpleNamespace(num_extends=lambda: 0)
    extend_op = SimpleNamespace(num_extends=lambda: 1)

    assert EventLoop._dispatch_depends_on_pending_commit(loop, handoff_op, None)
    assert not EventLoop._dispatch_depends_on_pending_commit(loop, extend_op, None)


def test_handoff_shaped_batch_without_pd_keeps_overlap() -> None:
    loop = _predicate_loop(kv_transfer=None)
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

    def _commit_forward_results(self, forward_op, results, on_first_token):
        self.committed.append(forward_op)
        return [f"change-{forward_op}"]


def test_drain_in_flight_commits_oldest_first() -> None:
    loop = _DrainHarness()
    in_flight = deque([("op0", None, None), ("op1", None, None)])

    request_changes = EventLoop._drain_in_flight(loop, in_flight)

    assert not in_flight
    assert loop.committed == ["op0", "op1"]
    assert request_changes == ["change-op0", "change-op1"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
