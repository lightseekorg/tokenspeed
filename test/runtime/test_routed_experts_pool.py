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

"""Tests for the R3 (Rollout Routing Replay) slot-indexed routing pool.

Covers :class:`RoutedExpertsPool` (store/gather by KV slot, reserved slot 0,
sizing) and :class:`RoutedExpertsCapturer` (per-forward capture/commit, the
prefix-hit retrieval property, no-op when inactive, and the TP/EP row-count
misalignment guard). Pure CPU tensors — no GPU or kernel stack needed.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch  # noqa: E402

from tokenspeed.runtime.cache.routed_experts_pool import (  # noqa: E402
    ROUTING_UNSET,
    RoutedExpertsCapturer,
    RoutedExpertsPool,
    build_routed_experts_capturer,
    get_global_routed_experts_capturer,
    set_global_routed_experts_capturer,
)


class TestRoutedExpertsPool(unittest.TestCase):
    def _pool(self, size=16, num_moe_layers=3, top_k=2):
        return RoutedExpertsPool(size=size, num_moe_layers=num_moe_layers, top_k=top_k)

    def test_shape_and_reserved_slot(self):
        pool = self._pool(size=16, num_moe_layers=3, top_k=2)
        # +1 reserved padding row at index 0.
        self.assertEqual(tuple(pool.buffer.shape), (17, 3, 2))
        self.assertEqual(pool.buffer.dtype, torch.int32)
        # Everything starts unset.
        self.assertTrue(torch.all(pool.buffer == ROUTING_UNSET))

    def test_store_and_gather_roundtrip(self):
        pool = self._pool()
        loc = torch.tensor([1, 5, 9])
        ids_l0 = torch.tensor([[3, 7], [1, 2], [4, 4]], dtype=torch.int32)
        pool.store_layer(0, loc, ids_l0)
        got = pool.gather_layer(0, loc)
        self.assertTrue(torch.equal(got, ids_l0))

    def test_gather_all_layers_shape(self):
        pool = self._pool(num_moe_layers=3, top_k=2)
        loc = torch.tensor([2, 4])
        pool.store_layer(0, loc, torch.tensor([[1, 1], [2, 2]], dtype=torch.int32))
        pool.store_layer(2, loc, torch.tensor([[7, 8], [9, 0]], dtype=torch.int32))
        allrouting = pool.gather(loc)
        self.assertEqual(tuple(allrouting.shape), (2, 3, 2))
        # Layer 1 was never written → still unset.
        self.assertTrue(torch.all(allrouting[:, 1, :] == ROUTING_UNSET))
        self.assertTrue(
            torch.equal(allrouting[:, 0, :], torch.tensor([[1, 1], [2, 2]]))
        )
        self.assertTrue(
            torch.equal(allrouting[:, 2, :], torch.tensor([[7, 8], [9, 0]]))
        )

    def test_reserved_slot_zero_does_not_alias(self):
        pool = self._pool()
        # Writing real slots must never touch the reserved padding row 0.
        pool.store_layer(
            0,
            torch.tensor([1, 2, 3]),
            torch.tensor([[5, 5], [6, 6], [7, 7]], dtype=torch.int32),
        )
        self.assertTrue(torch.all(pool.buffer[0] == ROUTING_UNSET))

    def test_store_rejects_row_mismatch(self):
        pool = self._pool()
        with self.assertRaises(ValueError):
            pool.store_layer(
                0, torch.tensor([1, 2]), torch.tensor([[1, 1]], dtype=torch.int32)
            )

    def test_store_rejects_topk_mismatch(self):
        pool = self._pool(top_k=2)
        with self.assertRaises(ValueError):
            pool.store_layer(
                0, torch.tensor([1]), torch.tensor([[1, 2, 3]], dtype=torch.int32)
            )

    def test_store_rejects_bad_layer(self):
        pool = self._pool(num_moe_layers=3)
        with self.assertRaises(ValueError):
            pool.store_layer(3, torch.tensor([1]), torch.tensor([[1, 1]]))

    def test_reset(self):
        pool = self._pool()
        pool.store_layer(
            0, torch.tensor([1]), torch.tensor([[9, 9]], dtype=torch.int32)
        )
        pool.reset()
        self.assertTrue(torch.all(pool.buffer == ROUTING_UNSET))

    def test_num_bytes(self):
        pool = self._pool(size=100, num_moe_layers=4, top_k=8)
        # (100 + 1) * 4 * 8 * 4 bytes (int32)
        self.assertEqual(pool.num_bytes, 101 * 4 * 8 * 4)

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            RoutedExpertsPool(size=-1, num_moe_layers=1, top_k=1)
        with self.assertRaises(ValueError):
            RoutedExpertsPool(size=1, num_moe_layers=0, top_k=1)
        with self.assertRaises(ValueError):
            RoutedExpertsPool(size=1, num_moe_layers=1, top_k=0)


class TestRoutedExpertsCapturer(unittest.TestCase):
    def _setup(self, num_moe_layers=2, top_k=2):
        pool = RoutedExpertsPool(size=16, num_moe_layers=num_moe_layers, top_k=top_k)
        return pool, RoutedExpertsCapturer(pool)

    def test_capture_commit_writes_pool(self):
        pool, cap = self._setup(num_moe_layers=2, top_k=2)
        loc = torch.tensor([3, 4, 5])
        cap.begin_forward(loc)
        self.assertTrue(cap.active)
        cap.capture(0, torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32))
        cap.capture(1, torch.tensor([[7, 8], [9, 0], [1, 1]], dtype=torch.int32))
        cap.commit()
        self.assertFalse(cap.active)
        got = pool.gather(loc)
        self.assertTrue(
            torch.equal(got[:, 0, :], torch.tensor([[1, 2], [3, 4], [5, 6]]))
        )
        self.assertTrue(
            torch.equal(got[:, 1, :], torch.tensor([[7, 8], [9, 0], [1, 1]]))
        )

    def test_prefix_hit_retrieval_property(self):
        # The core R3 property: routing written for a set of slots is retrievable
        # later purely from those slots — exactly what a prefix-cache hit does
        # when it reuses the KV slots of a shared prefix.
        pool, cap = self._setup(num_moe_layers=2, top_k=2)
        prefix_slots = torch.tensor([10, 11, 12, 13])
        cap.begin_forward(prefix_slots)
        cap.capture(
            0, torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.int32)
        )
        cap.capture(
            1, torch.tensor([[5, 5], [6, 6], [7, 7], [8, 8]], dtype=torch.int32)
        )
        cap.commit()

        # A later request whose prefix hit reuses slots 11..13 recovers routing
        # with no recompute, consistent with the reused KV.
        reused = torch.tensor([11, 12, 13])
        routing = pool.gather(reused)
        self.assertTrue(
            torch.equal(routing[:, 0, :], torch.tensor([[2, 2], [3, 3], [4, 4]]))
        )
        self.assertTrue(
            torch.equal(routing[:, 1, :], torch.tensor([[6, 6], [7, 7], [8, 8]]))
        )

    def test_capture_noop_when_inactive(self):
        pool, cap = self._setup()
        # No begin_forward → capture must be a no-op (hook stays installed cheaply).
        cap.capture(0, torch.tensor([[1, 2]], dtype=torch.int32))
        cap.commit()
        self.assertTrue(torch.all(pool.buffer == ROUTING_UNSET))

    def test_begin_forward_none_disables(self):
        pool, cap = self._setup()
        cap.begin_forward(None)
        self.assertFalse(cap.active)
        cap.capture(0, torch.tensor([[1, 2]], dtype=torch.int32))
        cap.commit()
        self.assertTrue(torch.all(pool.buffer == ROUTING_UNSET))

    def test_tp_row_misalignment_is_skipped(self):
        # Under TP/EP, topk_ids rows are the global all-gathered token set, not
        # the rank-local slots. Skip rather than corrupt, and count it.
        pool, cap = self._setup(top_k=2)
        loc = torch.tensor([1, 2, 3])  # 3 local slots
        cap.begin_forward(loc)
        # 6 global rows (e.g. TP=2 all-gather) — must be skipped.
        cap.capture(0, torch.tensor([[i, i] for i in range(6)], dtype=torch.int32))
        cap.commit()
        self.assertEqual(cap.skipped_misaligned, 1)
        self.assertTrue(torch.all(pool.buffer == ROUTING_UNSET))

    def test_capture_detaches_from_source(self):
        # commit must persist a snapshot even if the source tensor is mutated
        # after capture (topk_ids is transient inside the MoE forward).
        pool, cap = self._setup(num_moe_layers=1, top_k=2)
        loc = torch.tensor([1, 2])
        src = torch.tensor([[1, 1], [2, 2]], dtype=torch.int32)
        cap.begin_forward(loc)
        cap.capture(0, src)
        src.fill_(0)  # overwrite the source before commit
        cap.commit()
        self.assertTrue(
            torch.equal(pool.gather_layer(0, loc), torch.tensor([[1, 1], [2, 2]]))
        )


class TestCaptureInOrder(unittest.TestCase):
    """capture_in_order assigns MoE layers by per-forward invocation order."""

    def _setup(self, num_moe_layers=3, top_k=2):
        pool = RoutedExpertsPool(size=16, num_moe_layers=num_moe_layers, top_k=top_k)
        return pool, RoutedExpertsCapturer(pool)

    def test_sequential_layer_assignment(self):
        pool, cap = self._setup(num_moe_layers=3, top_k=2)
        loc = torch.tensor([1, 2])
        cap.begin_forward(loc)
        cap.capture_in_order(torch.tensor([[0, 0], [0, 0]], dtype=torch.int32))  # -> 0
        cap.capture_in_order(torch.tensor([[1, 1], [1, 1]], dtype=torch.int32))  # -> 1
        cap.capture_in_order(torch.tensor([[2, 2], [2, 2]], dtype=torch.int32))  # -> 2
        cap.commit()
        got = pool.gather(loc)
        self.assertTrue(torch.equal(got[:, 0, :], torch.zeros(2, 2, dtype=torch.int32)))
        self.assertTrue(torch.equal(got[:, 1, :], torch.ones(2, 2, dtype=torch.int32)))
        self.assertTrue(
            torch.equal(got[:, 2, :], torch.full((2, 2), 2, dtype=torch.int32))
        )

    def test_counter_resets_each_forward(self):
        pool, cap = self._setup(num_moe_layers=2, top_k=1)
        loc = torch.tensor([5])
        cap.begin_forward(loc)
        cap.capture_in_order(torch.tensor([[9]], dtype=torch.int32))  # layer 0
        cap.commit()
        # New forward → counter back to 0, layer 0 overwritten.
        cap.begin_forward(loc)
        cap.capture_in_order(torch.tensor([[3]], dtype=torch.int32))  # layer 0 again
        cap.commit()
        self.assertTrue(torch.equal(pool.gather_layer(0, loc), torch.tensor([[3]])))

    def test_overflow_is_skipped_and_counted(self):
        pool, cap = self._setup(num_moe_layers=2, top_k=1)
        loc = torch.tensor([1])
        cap.begin_forward(loc)
        cap.capture_in_order(torch.tensor([[1]], dtype=torch.int32))  # layer 0
        cap.capture_in_order(torch.tensor([[2]], dtype=torch.int32))  # layer 1
        cap.capture_in_order(torch.tensor([[3]], dtype=torch.int32))  # overflow
        cap.commit()
        self.assertEqual(cap.skipped_overflow, 1)
        self.assertTrue(torch.equal(pool.gather_layer(0, loc), torch.tensor([[1]])))
        self.assertTrue(torch.equal(pool.gather_layer(1, loc), torch.tensor([[2]])))

    def test_inactive_is_noop(self):
        pool, cap = self._setup()
        cap.capture_in_order(torch.tensor([[1, 2]], dtype=torch.int32))
        cap.commit()
        self.assertTrue(torch.all(pool.buffer == ROUTING_UNSET))


class TestBuildFromConfig(unittest.TestCase):
    """build_routed_experts_capturer derives size/layers/top_k from config."""

    @staticmethod
    def _model_config(*, num_hidden_layers, num_experts_per_tok):
        text = types.SimpleNamespace()
        if num_hidden_layers is not None:
            text.num_hidden_layers = num_hidden_layers
        if num_experts_per_tok is not None:
            text.num_experts_per_tok = num_experts_per_tok
        hf = types.SimpleNamespace(text_config=text)
        return types.SimpleNamespace(hf_config=hf)

    def test_moe_config_builds_capturer(self):
        server_args = types.SimpleNamespace(device="cpu")
        mc = self._model_config(num_hidden_layers=6, num_experts_per_tok=4)
        cap = build_routed_experts_capturer(server_args, mc, size=32)
        self.assertIsInstance(cap, RoutedExpertsCapturer)
        self.assertEqual(tuple(cap.pool.buffer.shape), (33, 6, 4))

    def test_dense_config_returns_none(self):
        server_args = types.SimpleNamespace(device="cpu")
        mc = self._model_config(num_hidden_layers=6, num_experts_per_tok=None)
        self.assertIsNone(build_routed_experts_capturer(server_args, mc, size=32))

    def test_reads_from_hf_config_when_no_text_config(self):
        server_args = types.SimpleNamespace(device="cpu")
        hf = types.SimpleNamespace(num_hidden_layers=4, num_experts_per_tok=2)
        mc = types.SimpleNamespace(hf_config=hf)
        cap = build_routed_experts_capturer(server_args, mc, size=8)
        self.assertEqual(tuple(cap.pool.buffer.shape), (9, 4, 2))


class TestReturnRoutedExpertsFlag(unittest.TestCase):
    """The request-level flag defaults off and propagates through __getitem__.

    io_struct is imported lazily so the kernel-free pool tests above still run
    in environments where the full engine import chain is unavailable.
    """

    def test_flag_defaults_off_and_propagates(self):
        from tokenspeed.runtime.engine.io_struct import GenerateReqInput

        self.assertFalse(GenerateReqInput(text="hi").return_routed_experts)

        obj = GenerateReqInput(text=["a", "b"], return_routed_experts=True)
        obj.normalize_batch_and_arguments()
        for i in range(2):
            self.assertTrue(obj[i].return_routed_experts)


class TestGlobalCapturer(unittest.TestCase):
    def tearDown(self):
        set_global_routed_experts_capturer(None)

    def test_global_accessor_roundtrip(self):
        self.assertIsNone(get_global_routed_experts_capturer())
        pool = RoutedExpertsPool(size=4, num_moe_layers=1, top_k=1)
        cap = RoutedExpertsCapturer(pool)
        set_global_routed_experts_capturer(cap)
        self.assertIs(get_global_routed_experts_capturer(), cap)
        set_global_routed_experts_capturer(None)
        self.assertIsNone(get_global_routed_experts_capturer())


if __name__ == "__main__":
    unittest.main(verbosity=2)
