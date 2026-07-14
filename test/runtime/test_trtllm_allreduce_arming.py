"""TrtllmAllReduceBackend.configure_tp_groups: the one arming policy point
(token window bounded by _MAX_ONESHOT_BYTES, group dedupe, size-1 skip)."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class ConfigureTpGroupsTest(unittest.TestCase):
    def setUp(self):
        try:
            from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
                TrtllmAllReduceBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.backend = TrtllmAllReduceBackend.__new__(TrtllmAllReduceBackend)
        self.calls = []
        self.backend.configure_group = lambda **kw: self.calls.append(kw) or True

    @staticmethod
    def _mapping(rank, attn_group, moe_group):
        return SimpleNamespace(
            rank=rank,
            attn=SimpleNamespace(tp_group=attn_group),
            moe=SimpleNamespace(tp_ep_group=moe_group),
        )

    def test_identical_groups_armed_once(self):
        # TP4 with moe-tp4: attn and moe share one group -> one workspace.
        m = self._mapping(1, (0, 1, 2, 3), (0, 1, 2, 3))
        self.backend.configure_tp_groups(m, hidden_size=2880)
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["rank"], 1)
        self.assertEqual(call["group"], (0, 1, 2, 3))
        self.assertEqual(call["hidden_dim"], 2880)
        # 2 MB of bf16 rows: (2 * 1024 * 1024) // (2880 * 2) = 364.
        self.assertEqual(call["max_token_num"], 364)

    def test_distinct_groups_armed_each(self):
        m = self._mapping(2, (0, 1, 2, 3), (2, 3))
        self.backend.configure_tp_groups(m, hidden_size=4096)
        self.assertEqual(sorted(c["group"] for c in self.calls), [(0, 1, 2, 3), (2, 3)])
        by_group = {c["group"]: c for c in self.calls}
        self.assertEqual(by_group[(0, 1, 2, 3)]["rank"], 2)
        self.assertEqual(by_group[(2, 3)]["rank"], 0)

    def test_size_one_groups_skipped(self):
        m = self._mapping(0, (0,), (0,))
        self.backend.configure_tp_groups(m, hidden_size=4096)
        self.assertEqual(self.calls, [])

    def test_window_floor_is_one_token(self):
        m = self._mapping(0, (0, 1), (0, 1))
        self.backend.configure_tp_groups(m, hidden_size=10**9)
        self.assertEqual(self.calls[0]["max_token_num"], 1)
