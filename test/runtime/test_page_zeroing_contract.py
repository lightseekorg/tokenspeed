"""Contract for physical cache-page sanitization.

``ModelExecutor.zero_cache_pages`` runs the scheduler's page-reuse list through
the target pool and any stateful draft pool. Only pools that alias recurrent
state and KV bytes need this; pure-attention pools do not, so the page list is
safely ignored for them instead of crashing the engine at startup.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")


class _DefaultStream:
    """Records the streams it was asked to wait on, in call order."""

    def __init__(self, trace):
        self._trace = trace

    def wait_stream(self, stream):
        self._trace.append(("wait", stream))


class ZeroCachePagesContractTest(unittest.TestCase):
    @staticmethod
    def _fake(pool, draft_pool, trace):
        return types.SimpleNamespace(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=draft_pool,
            device="cpu",
            default_stream=_DefaultStream(trace),
            execution_stream="execution-stream",
        )

    @classmethod
    def _call(cls, pool, page_ids, draft_pool=None):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        fake = cls._fake(pool, draft_pool, trace=[])
        return ModelExecutor.zero_cache_pages(fake, page_ids)

    def test_empty_page_list_is_a_noop(self):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        fake = self._fake(types.SimpleNamespace(), None, trace)
        self.assertIsNone(ModelExecutor.zero_cache_pages(fake, []))
        self.assertEqual(trace, [], "an empty list places no fence either")

    def test_zeroing_orders_itself_behind_the_execution_stream(self):
        # The pages' previous owner may still be writing them from a forward
        # in flight on the execution stream: the default stream waits on it
        # before the pool touches a byte, and the function places that wait
        # itself rather than relying on a caller-side fence.
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_pages=lambda page_ids: trace.append(("zero", list(page_ids))),
        )
        fake = self._fake(pool, None, trace)
        self.assertIsNone(ModelExecutor.zero_cache_pages(fake, [4, 5]))
        self.assertEqual(trace, [("wait", "execution-stream"), ("zero", [4, 5])])

    def test_pure_attention_pool_ignores_page_reuse_list(self):
        # No zero_pages and not flagged as state-aliasing -> skip, no raise.
        pool = types.SimpleNamespace(requires_page_zeroing=False)
        self.assertIsNone(self._call(pool, [1, 2, 3]))

    def test_missing_flag_defaults_to_skip(self):
        pool = types.SimpleNamespace()  # attribute absent entirely
        self.assertIsNone(self._call(pool, [1, 2, 3]))

    def test_state_aliasing_pool_without_impl_fails_loudly(self):
        # Declares it needs zeroing but forgot to implement it -> tripwire.
        pool = types.SimpleNamespace(requires_page_zeroing=True)
        with self.assertRaises(RuntimeError):
            self._call(pool, [1, 2, 3])

    def test_pool_with_impl_is_invoked(self):
        seen = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_pages=lambda page_ids: seen.append(list(page_ids)),
        )
        # device="cpu" -> returns None after invoking zero_pages.
        self.assertIsNone(self._call(pool, [4, 5]))
        self.assertEqual(seen, [[4, 5]])

    def test_group_aware_pool_is_invoked(self):
        seen = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_new_blocks=lambda pages: seen.append(dict(pages)),
        )
        pages = {"full": [4, 5], "state": [9]}
        self.assertIsNone(self._call(pool, pages))
        self.assertEqual(seen, [pages])

    def test_stateful_draft_pool_zeros_its_group_subset(self):
        target_seen = []
        draft_seen = []
        target = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_new_blocks=lambda pages: target_seen.append(dict(pages)),
        )
        draft = types.SimpleNamespace(
            requires_page_zeroing=True,
            arena=types.SimpleNamespace(
                cache_group_specs=(
                    types.SimpleNamespace(group_id="history"),
                    types.SimpleNamespace(group_id="state"),
                ),
            ),
            zero_new_blocks=lambda pages: draft_seen.append(dict(pages)),
        )
        pages = {"history": [4], "state": [9], "target_only": [12]}

        self.assertIsNone(self._call(target, pages, draft))
        self.assertEqual(target_seen, [pages])
        self.assertEqual(draft_seen, [{"history": [4], "state": [9]}])


if __name__ == "__main__":
    unittest.main()
