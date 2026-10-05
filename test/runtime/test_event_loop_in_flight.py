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
of overlap-breaking dependencies), ``_drain_in_flight`` and the pipeline's
commit-side token broadcast with fakes, so no model, CUDA context, or
transfer backend is created.
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


def _pp_loop(monkeypatch, *, is_last_pp_rank: bool, broadcast):
    """A loop on a two-stage pipeline whose gloo broadcast is ``broadcast``."""
    import tokenspeed.runtime.engine.event_loop as event_loop_module

    monkeypatch.setattr(
        event_loop_module,
        "pg_manager",
        SimpleNamespace(get_process_group=lambda backend, group: ("gloo", group)),
    )
    monkeypatch.setattr(event_loop_module.dist, "broadcast_object_list", broadcast)
    mapping = SimpleNamespace(
        has_pp=True, pp_group=(3, 7), is_last_pp_rank=is_last_pp_rank
    )
    return SimpleNamespace(server_args=SimpleNamespace(mapping=mapping))


def test_pp_broadcast_adopts_the_last_stage_tokens_candidates_logprobs_and_flags(
    monkeypatch,
) -> None:
    """A stage before the last commits the last stage's sampled tokens, the
    drafter's candidate rows (next_input_ids) -- every rank's scheduler folds
    them into the final chunk's result, which the remote decode carries --
    both logprob vectors, which every rank's output processor records on its
    request state, and the NaN guard's per-request flags, so every stage
    aborts (or finishes) the same requests: only the last stage audits
    logits and prompt logprobs, and the stage's own flags (zero here, the
    guard found nothing in its placeholder outputs) would commit as healthy a
    request the last stage aborts. The prompt-logprob plan the flat vector
    follows is the stage's own (mirrored) one, so it stays in place."""
    from_last_stage = (
        ["sampled"],
        ["lengths"],
        ["candidates"],
        ["output logprobs"],
        ["prompt logprobs"],
        ["nan flags"],
    )

    def broadcast(payload, src, group):
        assert src == 7 and group == ("gloo", (3, 7))
        assert payload == [None]
        payload[0] = from_last_stage

    loop = _pp_loop(monkeypatch, is_last_pp_rank=False, broadcast=broadcast)
    results = SimpleNamespace(
        output_tokens="placeholder",
        output_lengths="placeholder",
        next_input_ids=None,
        output_logprobs=None,
        input_token_logprobs=None,
        output_nan_flags="this stage's clean flags",
        input_logprob_plan="this stage's plan",
    )

    EventLoop._pp_broadcast_output_tokens(loop, forward_op=None, results=results)

    assert results.output_tokens == ["sampled"]
    assert results.output_lengths == ["lengths"]
    assert results.next_input_ids == ["candidates"]
    assert results.output_logprobs == ["output logprobs"]
    assert results.input_token_logprobs == ["prompt logprobs"]
    assert results.output_nan_flags == ["nan flags"]
    assert results.input_logprob_plan == "this stage's plan"


def test_pp_broadcast_sends_the_last_stage_results_unchanged(monkeypatch) -> None:
    sent = []
    loop = _pp_loop(
        monkeypatch,
        is_last_pp_rank=True,
        broadcast=lambda payload, src, group: sent.append(payload[0]),
    )
    results = SimpleNamespace(
        output_tokens="sampled",
        output_lengths="lengths",
        next_input_ids="candidates",
        output_logprobs="output logprobs",
        input_token_logprobs="prompt logprobs",
        output_nan_flags="nan flags",
    )

    EventLoop._pp_broadcast_output_tokens(loop, forward_op=None, results=results)

    assert sent == [
        (
            "sampled",
            "lengths",
            "candidates",
            "output logprobs",
            "prompt logprobs",
            "nan flags",
        )
    ]
    assert (results.output_tokens, results.next_input_ids) == ("sampled", "candidates")
    assert results.input_token_logprobs == "prompt logprobs"
    assert results.output_nan_flags == "nan flags"


def test_pp_broadcast_precedes_commit_post_processing() -> None:
    """The adopted tokens/candidates must be in place before the output
    processor reads the result. Pinned on the statement order of
    EventLoop._commit_forward_results."""
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(EventLoop._commit_forward_results))
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
    assert calls.index("_pp_broadcast_output_tokens") < calls.index(
        "post_process_forward_op"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
