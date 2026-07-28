"""Contract for flat-cache physical-page sanitization.

``ModelExecutor.zero_flat_cache_pages`` runs the scheduler's page-reuse list
through the KV pool's ``zero_pages``. Only pools that alias recurrent-state and
KV bytes in one slab need this; pure-attention pools (MHA, and Inkling whose
conv state lives in a separate pool) do not, so the page list is safely
ignored for them instead of crashing the engine at startup.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")


class ZeroFlatCachePagesContractTest(unittest.TestCase):
    @staticmethod
    def _call(pool, page_ids):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        fake = types.SimpleNamespace(token_to_kv_pool=pool, device="cpu")
        return ModelExecutor.zero_flat_cache_pages(fake, page_ids)

    def test_empty_page_list_is_a_noop(self):
        pool = types.SimpleNamespace()
        self.assertIsNone(self._call(pool, []))

    def test_pure_attention_pool_ignores_page_reuse_list(self):
        # No zero_pages and not flagged as state-aliasing -> skip, no raise.
        pool = types.SimpleNamespace(flat_kv_requires_page_zeroing=False)
        self.assertIsNone(self._call(pool, [1, 2, 3]))

    def test_missing_flag_defaults_to_skip(self):
        pool = types.SimpleNamespace()  # attribute absent entirely
        self.assertIsNone(self._call(pool, [1, 2, 3]))

    def test_state_aliasing_pool_without_impl_fails_loudly(self):
        # Declares it needs zeroing but forgot to implement it -> tripwire.
        pool = types.SimpleNamespace(flat_kv_requires_page_zeroing=True)
        with self.assertRaises(RuntimeError):
            self._call(pool, [1, 2, 3])

    def test_pool_with_impl_is_invoked(self):
        seen = []
        pool = types.SimpleNamespace(
            flat_kv_requires_page_zeroing=True,
            zero_pages=lambda page_ids: seen.append(list(page_ids)),
        )
        # device="cpu" -> returns None after invoking zero_pages.
        self.assertIsNone(self._call(pool, [4, 5]))
        self.assertEqual(seen, [[4, 5]])


if __name__ == "__main__":
    unittest.main()
