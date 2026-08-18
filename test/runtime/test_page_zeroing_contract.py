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


class ZeroCachePagesContractTest(unittest.TestCase):
    @staticmethod
    def _call(pool, page_ids, draft_pool=None, attn_backend=None):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        fake = types.SimpleNamespace(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=draft_pool,
            attn_backend=attn_backend
            or types.SimpleNamespace(drop_deferred_on_pages=lambda pages: None),
            device="cpu",
        )
        return ModelExecutor.zero_cache_pages(fake, page_ids)

    def test_empty_page_list_is_a_noop(self):
        pool = types.SimpleNamespace()
        self.assertIsNone(self._call(pool, []))

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

    def test_mapping_pages_condemn_deferred_work_before_zeroing(self):
        order = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_new_blocks=lambda pages: order.append("zero"),
        )
        backend = types.SimpleNamespace(
            drop_deferred_on_pages=lambda pages: order.append("drop"),
        )
        self.assertIsNone(self._call(pool, {"g": [1]}, attn_backend=backend))
        self.assertEqual(order, ["drop", "zero"])
