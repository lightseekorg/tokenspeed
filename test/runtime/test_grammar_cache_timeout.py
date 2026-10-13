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

"""Regression tests for BaseGrammarBackend's compile-timeout cache.

These tests pin the contract that GrammarManager relies on:

1. Permanent failures (``cache_invalid``) stay sticky forever.
2. Transient timeouts (``record_compile_timeout``) cache a TTL'd marker;
   while it's still valid, a still-running compile that finishes late must
   NOT be allowed to clobber the marker.
3. After the TTL expires, a new request triggers a fresh compile and the
   successful result becomes the new cached value (timeout marker is
   replaced, not left to poison the key).
4. Repeated timeouts escalate: after ``max_retries`` consecutive timeouts
   the marker becomes permanent (no TTL) — at that point the compiler is
   consistently broken for this key.
5. ``compile_started_at`` measures from when the worker actually begins
   running, not from when the future was submitted, so executor queueing
   delay doesn't eat into the timeout budget.
6. The base API exposes ``is_invalid`` / ``expired`` attributes on every
   value so callers don't need isinstance checks.

Pure CPU; no model, tokenizer, or GPU required.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)


class _FakeGrammar(BaseGrammarObject):
    """Trivial successful compile result; copy() returns self for cheap equality."""

    def copy(self):
        return self


class _GatedBackend(BaseGrammarBackend):
    """Backend whose ``init_value_impl`` blocks on an externally-controlled
    Event so tests can drive the worker thread's lifecycle precisely."""

    def __init__(self):
        super().__init__()
        self.compile_started = threading.Event()
        self.compile_can_finish = threading.Event()
        self.compile_count = 0

    def init_value_impl(self, key, require_reasoning: bool = False):
        self.compile_count += 1
        self.compile_started.set()
        self.compile_can_finish.wait()
        return _FakeGrammar()


class GrammarBackendCacheTests(unittest.TestCase):
    KEY = ("json", "schema-A")

    # --- attribute interface (no isinstance) ---

    def test_is_invalid_attribute_split(self):
        """Successful and failed grammars share the is_invalid attribute so
        callers don't need isinstance to branch."""
        ok = _FakeGrammar()
        bad = InvalidGrammarObject("nope")
        self.assertFalse(ok.is_invalid)
        self.assertTrue(bad.is_invalid)
        self.assertFalse(ok.expired)
        self.assertFalse(bad.expired)  # no expires_at -> permanent

    # --- permanent failures ---

    def test_cache_invalid_is_permanent(self):
        b = _GatedBackend()
        b.cache_invalid(self.KEY, "bad regex")
        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertTrue(hit)
        self.assertTrue(v.is_invalid)
        self.assertEqual(v.error_message, "bad regex")
        self.assertIsNone(v.expires_at)
        # Even after a long sleep it stays invalid.
        time.sleep(0.05)
        self.assertFalse(v.expired)

    # --- transient timeouts (TTL) ---

    def test_timeout_marker_sticks_within_ttl(self):
        """While the TTL'd marker is fresh, a slow compile that completes
        late must not be able to overwrite it. Otherwise the abort message
        we returned to the user would be silently contradicted by the next
        request seeing a real grammar."""
        b = _GatedBackend()
        # Start a compile and let it block.
        fut = b.executor.submit(b.init_value, self.KEY)
        self.assertTrue(b.compile_started.wait(timeout=1.0))

        # Mid-compile: cache a transient timeout marker.
        b.record_compile_timeout(self.KEY, "timed out", ttl_secs=2.0, max_retries=2)

        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertTrue(hit)
        self.assertTrue(v.is_invalid)
        self.assertFalse(v.expired)
        self.assertEqual(v.error_message, "timed out")

        # Now let the slow compile finish — it must NOT clobber the marker.
        b.compile_can_finish.set()
        fut.result(timeout=1.0)

        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertTrue(hit, "expected cache hit, got miss")
        self.assertTrue(
            v.is_invalid, "late finish wrongly un-invalidated the cache marker"
        )
        self.assertEqual(v.error_message, "timed out")

    def test_marker_expires_and_recovers(self):
        """After the TTL, a fresh compile is dispatched and the success path
        replaces the marker — the key is not poisoned forever."""
        b = _GatedBackend()
        # Cache a short-lived timeout marker (no in-flight compile this time).
        b.record_compile_timeout(self.KEY, "timed out", ttl_secs=0.1, max_retries=2)
        time.sleep(0.15)

        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertFalse(hit, "expired marker should evict on lookup")
        # The returned value is a Future from the executor; the worker is
        # already running and blocked on compile_can_finish.
        self.assertTrue(b.compile_started.wait(timeout=1.0))
        b.compile_can_finish.set()
        result = v.result(timeout=1.0)
        self.assertFalse(result.is_invalid)
        self.assertEqual(b.compile_count, 1, "exactly one fresh compile expected")

        # Subsequent lookups see the real grammar.
        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertTrue(hit)
        self.assertFalse(v.is_invalid)

    def test_repeated_timeouts_escalate_to_permanent(self):
        """After max_retries+1 consecutive timeouts the marker becomes
        permanent (no TTL) so we stop hitting the compiler in a loop."""
        b = _GatedBackend()
        # max_retries=2 → first 2 timeouts get a TTL, the 3rd is permanent.
        for _ in range(2):
            b.record_compile_timeout(
                self.KEY, "timed out", ttl_secs=0.01, max_retries=2
            )
            v, hit = b.get_cached_or_future_value(self.KEY)
            self.assertTrue(hit)
            self.assertTrue(v.is_invalid)
            self.assertIsNotNone(v.expires_at, "first 2 timeouts must have a TTL")
            time.sleep(0.02)  # let TTL expire so the next call sees a miss path

        # 3rd timeout escalates to permanent.
        b.record_compile_timeout(self.KEY, "timed out", ttl_secs=0.01, max_retries=2)
        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertTrue(hit)
        self.assertTrue(v.is_invalid)
        self.assertIsNone(v.expires_at, "post-escalation marker must be permanent")
        self.assertIn("gave up after", v.error_message)

    def test_successful_compile_clearstimeout_history(self):
        """A successful compile after some transient timeouts resets the
        retry counter, so a future timeout cycle gets the full retry budget
        again."""
        b = _GatedBackend()
        b.record_compile_timeout(self.KEY, "t1", ttl_secs=0.01, max_retries=2)
        time.sleep(0.02)
        b.record_compile_timeout(self.KEY, "t2", ttl_secs=0.01, max_retries=2)
        self.assertEqual(b.timeout_history[self.KEY].count, 2)
        time.sleep(0.02)

        # Trigger a fresh compile that succeeds.
        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertFalse(hit)
        b.compile_can_finish.set()
        v.result(timeout=1.0)

        # Successful compile must have cleared the per-key history; a
        # subsequent timeout is also a no-op here (cache holds a valid
        # grammar, so record_compile_timeout drops the spurious mark).
        self.assertNotIn(self.KEY, b.timeout_history)
        b.record_compile_timeout(self.KEY, "t3", ttl_secs=0.01, max_retries=2)
        self.assertNotIn(
            self.KEY,
            b.timeout_history,
            "spurious timeout against a valid cached grammar must not increment",
        )
        v, hit = b.get_cached_or_future_value(self.KEY)
        self.assertTrue(hit)
        self.assertFalse(v.is_invalid, "valid grammar must not be clobbered by timeout")

    # --- compile_started_at excludes executor queueing ---

    def test_started_at_records_worker_start_not_submit(self):
        """compile_started_at must reflect when the worker actually begins
        running init_value_impl, so an arbitrary executor queueing delay
        doesn't count toward the timeout budget."""
        # Use a single-thread executor and pre-fill it with a blocker so the
        # second submission must wait in the queue.
        from concurrent.futures import ThreadPoolExecutor

        b = _GatedBackend()
        b.executor.shutdown(wait=False)
        b.executor = ThreadPoolExecutor(max_workers=1)

        block_release = threading.Event()
        b.executor.submit(block_release.wait)  # occupy the only worker

        submit_ts = time.monotonic()
        future = b.executor.submit(b.init_value, self.KEY)

        # While queued, no entry exists yet -> compile_started_at is None.
        for _ in range(5):
            if b.compile_started.is_set():
                self.fail("worker must not have started while queue is blocked")
            self.assertIsNone(b.compile_started_at(self.KEY))
            time.sleep(0.02)

        queue_delay = time.monotonic() - submit_ts
        self.assertGreater(
            queue_delay,
            0.05,
            "test setup failed: expected measurable queue delay before unblocking",
        )

        # Release the blocker so the worker can pick up the real compile.
        block_release.set()
        self.assertTrue(b.compile_started.wait(timeout=1.0))

        started_at = b.compile_started_at(self.KEY)
        self.assertIsNotNone(started_at)
        # started_at must reflect the worker start, well after submit_ts +
        # queue_delay. Slack the bound a bit but require it to be later than
        # the submit instant by at least the observed queue delay.
        self.assertGreaterEqual(
            started_at,
            submit_ts + queue_delay - 0.02,
            "started_at should be past submit_ts + queue delay",
        )

        # After the compile finishes, started_at must REMAIN set — callers
        # use it to compute compile-only elapsed time, and a worker that
        # finishes between a future.done() check and the elapsed check must
        # not silently revert callers to a wall-clock-from-submit fallback
        # (that was a real bug — see GrammarManager's done()-recheck).
        b.compile_can_finish.set()
        future.result(timeout=1.0)
        self.assertEqual(b.compile_started_at(self.KEY), started_at)


class GrammarManagerTimeoutTests(unittest.TestCase):
    """End-to-end timeout behavior in GrammarManager.get_ready_grammar_requests:

    queued-but-not-yet-started compiles must still time out (otherwise a
    wedged or saturated executor could let a request wait forever), and
    once the worker starts the per-compile budget begins fresh.
    """

    KEY = ("json", "queued-forever")

    def _make_manager(self, *, compile_timeout: float = 0.05, max_retries: int = 2):
        # Build a GrammarManager without going through ServerArgs (the parent
        # imports too much). Subclass it to plug in our gated backend.
        from tokenspeed.runtime.grammar.grammar_manager import (
            GrammarManager as RealGrammarManager,
        )

        backend = _GatedBackend()
        # Saturate the executor so no worker is available — submitted
        # init_value calls will sit in the queue, never running.
        n_workers = backend.executor._max_workers
        block_release = threading.Event()
        for _ in range(n_workers):
            backend.executor.submit(block_release.wait)

        mgr = RealGrammarManager.__new__(RealGrammarManager)
        mgr.server_args = None
        mgr.grammar_queue = []
        mgr.compile_timeout_secs = compile_timeout
        mgr.compile_max_retries = max_retries
        mgr.grammar_backend = backend
        mgr.grammar_sync_group = None
        mgr.grammar_sync_size = 1
        return mgr, backend, block_release

    def _fake_state(self, key=KEY):
        from tokenspeed.runtime.engine.generation_output_processor import RequestState
        from tokenspeed.runtime.sampling.sampling_params import SamplingParams

        sp = SamplingParams(json_schema='{"type":"object"}')
        s = RequestState(
            prompt_input_ids=[0],
            sampling_params=sp,
            stream=False,
            tokenizer=None,
            computes_prompt_logprobs=True,
            eos_token_ids=[1],
        )
        s.rid = "test-rid"
        return s

    def test_queued_compile_times_out_via_queued_ts(self):
        """A request whose worker NEVER starts (executor saturated) must
        still time out within ~compile_timeout_secs of being queued."""
        mgr, backend, block_release = self._make_manager(compile_timeout=0.05)
        try:
            state = self._fake_state()
            value, hit = backend.get_cached_or_future_value(self.KEY)
            self.assertFalse(hit)
            state.grammar = value  # the Future
            state.grammar_key = self.KEY
            mgr.add_to_queue(spec=object(), state=state, bootstrap_info=object())

            # compile_started_at must be None (worker is queued, not running).
            self.assertIsNone(backend.compile_started_at(self.KEY))

            # First poll — within the budget, no failure.
            promoted = mgr.get_ready_grammar_requests()
            self.assertEqual(promoted, [])

            # Wait past the compile timeout; worker still hasn't started.
            time.sleep(0.07)
            self.assertIsNone(backend.compile_started_at(self.KEY))

            promoted = mgr.get_ready_grammar_requests()
            self.assertEqual(len(promoted), 1)
            _, promoted_state, _ = promoted[0]
            self.assertIs(promoted_state, state)
            self.assertTrue(state.finished)
            self.assertIn("timed out", state.to_abort_message)
        finally:
            block_release.set()

    def test_completion_at_timeout_boundary_admits_not_aborts(self):
        """Race: the worker finishes the compile in the same poll iteration
        where the elapsed check would otherwise mark it timed out. The
        request must be admitted (compile succeeded) — not aborted."""
        mgr, backend, block_release = self._make_manager(compile_timeout=0.05)
        try:
            state = self._fake_state()
            value, hit = backend.get_cached_or_future_value(self.KEY)
            self.assertFalse(hit)
            state.grammar = value
            state.grammar_key = self.KEY
            mgr.add_to_queue(spec=object(), state=state, bootstrap_info=object())

            # Let the worker start compiling, then sleep past the timeout
            # budget WITHOUT completing the compile yet.
            block_release.set()
            self.assertTrue(backend.compile_started.wait(timeout=1.0))
            time.sleep(0.07)  # past compile_timeout_secs from started_at

            # NOW finish the compile, *before* the next poll. From the poll's
            # perspective, the future is done at the moment we'd otherwise
            # decide "timed out" — must admit, not abort.
            backend.compile_can_finish.set()
            value.result(timeout=1.0)

            promoted = mgr.get_ready_grammar_requests()
            self.assertEqual(len(promoted), 1)
            _, promoted_state, _ = promoted[0]
            self.assertIs(promoted_state, state)
            self.assertFalse(
                state.finished,
                f"expected admission, got abort: {state.to_abort_message!r}",
            )
            self.assertFalse(state.grammar.is_invalid)
        finally:
            block_release.set()

    def test_compile_budget_resets_on_worker_start(self):
        """If the request waits in the queue almost to the timeout and only
        then the worker starts, we get a fresh compile budget — total max
        wait is ~2 * compile_timeout_secs, not 1."""
        mgr, backend, block_release = self._make_manager(compile_timeout=0.10)
        try:
            state = self._fake_state()
            value, hit = backend.get_cached_or_future_value(self.KEY)
            self.assertFalse(hit)
            state.grammar = value
            state.grammar_key = self.KEY
            mgr.add_to_queue(spec=object(), state=state, bootstrap_info=object())

            # Sit just under the queue-wait budget, then unblock the executor
            # so the worker starts running our compile.
            time.sleep(0.07)
            block_release.set()
            self.assertTrue(backend.compile_started.wait(timeout=1.0))

            # We're past the queue budget, but compile just started → must
            # NOT be marked failed yet.
            promoted = mgr.get_ready_grammar_requests()
            self.assertEqual(promoted, [], "compile budget should reset on start")
            self.assertFalse(state.finished)

            # Within the fresh compile budget, complete the compile.
            backend.compile_can_finish.set()
            value.result(timeout=1.0)
            promoted = mgr.get_ready_grammar_requests()
            self.assertEqual(len(promoted), 1)
            self.assertFalse(state.finished, "successful compile should not abort")
            self.assertFalse(state.grammar.is_invalid)
        finally:
            block_release.set()


if __name__ == "__main__":
    unittest.main()
