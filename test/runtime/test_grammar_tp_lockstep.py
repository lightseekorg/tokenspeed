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

"""Attention TP ranks admit a grammar request in the same iteration.

Each rank compiles and caches grammars on its own, so one rank can hold a
compiled grammar while another is still compiling it (or no longer caches
it). With attention TP > 1 a cache hit is therefore admitted through the
queue, whose all_gather admits it only once every rank is ready, and a
failure the group agreed on aborts the request on every rank, even one whose
own compile has just finished. A grammar ready on every rank but invalid on
one of them (a timeout marker cached there only) aborts the request on every
rank too. A single rank keeps admitting a cache hit at once.

Pure CPU; the peer rank is a stub of torch.distributed.all_gather_object, or
a second rank in a thread.
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    RequestState,
)
from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    BaseGrammarBackend,
    BaseGrammarObject,
)
from tokenspeed.runtime.grammar.grammar_manager import GrammarManager  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import (  # noqa: E402
    SamplingParams,
)

SCHEMA = '{"type": "object"}'
KEY = ("json", SCHEMA)


class _Grammar(BaseGrammarObject):
    def copy(self):
        return _Grammar()


class _Backend(BaseGrammarBackend):
    def init_value_impl(self, key, require_reasoning: bool = False):
        return _Grammar()


class _Peer:
    """The other attention TP rank: its ready and failed index sets and its
    invalid grammars by index."""

    def __init__(self):
        self.ready: set[int] = set()
        self.failed: set[int] = set()
        self.invalid: dict = {}

    def all_gather_object(self, gathered, obj, group=None):
        gathered[0] = obj
        gathered[1] = (set(self.ready), set(self.failed), dict(self.invalid))


class _Group:
    """all_gather_object across rank threads, one barrier per call."""

    def __init__(self, size: int):
        self.slots = [None] * size
        self.barrier = threading.Barrier(size, timeout=10)
        self.local = threading.local()

    def all_gather_object(self, gathered, obj, group=None):
        self.slots[self.local.rank] = obj
        self.barrier.wait()
        gathered[:] = self.slots
        self.barrier.wait()


def _manager(sync_size: int) -> GrammarManager:
    args = SimpleNamespace(
        grammar_backend="none",
        grammar_compile_timeout_secs=30.0,
        grammar_compile_max_retries=1,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=[0])),
    )
    manager = GrammarManager(args, tokenizer=None, vocab_size=32)
    manager.grammar_backend = _Backend()
    manager.grammar_sync_size = sync_size
    manager.grammar_sync_group = object() if sync_size > 1 else None
    return manager


def _state() -> RequestState:
    return RequestState(
        prompt_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=8, json_schema=SCHEMA),
        stream=False,
        tokenizer=None,
        computes_prompt_logprobs=False,
    )


def _compile(manager: GrammarManager) -> None:
    """Puts KEY's compiled grammar in this rank's cache."""
    manager.grammar_backend.get_cached_or_future_value(KEY)[0].result(timeout=5)


class TestGrammarTpLockstep(unittest.TestCase):
    def setUp(self):
        self.peer = _Peer()
        patcher = mock.patch(
            "torch.distributed.all_gather_object", self.peer.all_gather_object
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _queue(self, manager: GrammarManager) -> RequestState:
        state = _state()
        self.assertFalse(manager.process_req_with_grammar(state))
        manager.add_to_queue(SimpleNamespace(request_id="r"), state, None)
        return state

    def test_single_rank_admits_a_cache_hit_at_once(self):
        manager = _manager(1)
        _compile(manager)
        state = _state()
        self.assertTrue(manager.process_req_with_grammar(state))
        self.assertIsInstance(state.grammar, _Grammar)

    def test_cache_hit_waits_for_the_other_rank(self):
        manager = _manager(2)
        _compile(manager)
        state = self._queue(manager)

        self.assertEqual(manager.get_ready_grammar_requests(), [])  # peer compiling
        self.assertEqual(len(manager), 1)

        self.peer.ready = {0}
        promoted = manager.get_ready_grammar_requests()
        self.assertEqual([entry[1] for entry in promoted], [state])
        self.assertFalse(state.finished)
        self.assertIsInstance(state.grammar, _Grammar)
        self.assertEqual(len(manager), 0)

    def test_cached_failure_is_aborted_with_the_group(self):
        manager = _manager(2)
        manager.grammar_backend.cache_invalid(KEY, "bad schema")
        state = self._queue(manager)

        self.peer.ready = {0}
        self.assertEqual(len(manager.get_ready_grammar_requests()), 1)
        reason = state.finished_reason.to_json()
        self.assertEqual(reason["type"], "abort")
        self.assertIn("bad schema", reason["message"])

    def test_group_failure_aborts_a_compile_done_here(self):
        manager = _manager(2)
        _compile(manager)
        state = self._queue(manager)

        self.peer.failed = {0}  # the other rank's compile timed out
        self.assertEqual(len(manager.get_ready_grammar_requests()), 1)
        self.assertIsNone(state.grammar)
        self.assertIn("timed out", state.finished_reason.to_json()["message"])
        # This rank's compiled grammar stays cached.
        value, hit = manager.grammar_backend.get_cached_or_future_value(KEY)
        self.assertTrue(hit)
        self.assertFalse(value.is_invalid)


class TestGrammarTpOutcome(unittest.TestCase):
    """Ranks 0 and 1 in threads: one holds KEY's compiled grammar, the other a
    timeout marker for it (rank 0, then rank 1). Both see a cache hit and
    are ready at once; both must finish the request the same way."""

    def _run(self, managers):
        group = _Group(len(managers))
        outcomes = [None] * len(managers)

        def rank(r):
            group.local.rank = r
            manager, state = managers[r], _state()
            if not manager.process_req_with_grammar(state):
                manager.add_to_queue(SimpleNamespace(request_id="r"), state, None)
                manager.get_ready_grammar_requests()
            reason = state.finished_reason
            outcomes[r] = (len(manager), reason and reason.to_json())

        with mock.patch("torch.distributed.all_gather_object", group.all_gather_object):
            threads = [threading.Thread(target=rank, args=(r,)) for r in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)

        self.assertIsNotNone(outcomes[0])
        self.assertEqual(outcomes[0], outcomes[1])
        return outcomes[0]

    def _marker_and_compiled(self, timeouts: int):
        marked, compiled = _manager(2), _manager(2)
        for _ in range(timeouts):
            marked.grammar_backend.record_compile_timeout(
                KEY, "Grammar compilation timed out after 30.0s", 30.0, max_retries=1
            )
        _compile(compiled)
        return marked, compiled

    def test_transient_marker_on_one_rank_aborts_on_both(self):
        for managers in (
            self._marker_and_compiled(1),
            self._marker_and_compiled(1)[::-1],
        ):
            queued, reason = self._run(list(managers))
            self.assertEqual(queued, 0)
            self.assertEqual(reason["type"], "abort")
            self.assertIn("timed out", reason["message"])
            self.assertNotIn("gave up", reason["message"])

    def test_permanent_marker_on_one_rank_aborts_on_both(self):
        for managers in (
            self._marker_and_compiled(2),
            self._marker_and_compiled(2)[::-1],
        ):
            queued, reason = self._run(list(managers))
            self.assertEqual(queued, 0)
            self.assertIn("gave up after 1 retries", reason["message"])


if __name__ == "__main__":
    unittest.main()
