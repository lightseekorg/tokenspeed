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

"""Stdlib-only ownership and gate tests; adapter/native tests own byte recovery."""

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

# utils.__init__ imports the full accelerator stack. Load this stdlib-only
# module directly so the gate tests also run on a host without PyTorch/NPU.
_runtime = Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime"


def load_module(name, source):
    spec = importlib.util.spec_from_file_location(name, _runtime / source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_module = load_module(
    "ascend_saver_state_under_test", "utils/ascend_memory_saver_state.py"
)
AllocationPolicy = _module.AllocationPolicy
PoolKey = _module.PoolKey
OwnedPool = _module.OwnedPool
PoolState = _module.PoolState
validate_memory_saver_scope = _module.validate_memory_saver_scope

WEIGHTS = PoolKey("weights", AllocationPolicy.CPU_BACKUP, "target")
KV = PoolKey("kv_cache", AllocationPolicy.DISCARD, "decode")
GRAPH = PoolKey("acl_graph", AllocationPolicy.CPU_BACKUP, "decode")
WEIGHTS_DISCARD = PoolKey("weights", AllocationPolicy.DISCARD, "target")


@dataclass
class FakePool:
    id: tuple[int, int]


class FakeBackend:
    def __init__(self):
        self.created = []
        self.fail_resume = False

    def make_pool(self, key):
        self.created.append(key)
        return OwnedPool(object(), FakePool((0, len(self.created))))

    def pause(self, tag):
        pass

    def resume(self, tag):
        if self.fail_resume:
            raise RuntimeError("injected H2D failure")


class PoolStateTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.state = PoolState(
            backend=self.backend,
            make_pool=self.backend.make_pool,
            synchronize=lambda: None,
        )

    def test_partial_resume_stays_blocked_even_before_first_allocation(self):
        self.state.pause(None)
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        for key in (WEIGHTS, KV, GRAPH):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                with self.state.region(key):
                    self.fail("paused empty tag became usable")
        for tag in ("weights", "kv_cache"):
            self.state.resume(tag)
            with self.assertRaises(RuntimeError):
                self.state.assert_active()
        with self.state.region(WEIGHTS):
            pass
        self.state.resume("acl_graph")
        self.state.assert_active()
        self.state.pause(None)
        self.state.resume(None)
        self.state.assert_active()

    def test_resume_failure_blocks_use_until_successful_retry(self):
        with self.state.region(GRAPH):
            pass
        self.state.pause("acl_graph")
        self.backend.fail_resume = True
        with self.assertRaisesRegex(RuntimeError, "H2D"):
            self.state.resume("acl_graph")
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        self.backend.fail_resume = False
        self.state.resume("acl_graph")
        self.state.assert_active()

    def test_pause_rejects_live_capture_and_scope_exception_cleans_gate(self):
        with self.assertRaisesRegex(ValueError, "capture failed"):
            with self.state.region(GRAPH):
                with self.assertRaises(RuntimeError):
                    self.state.pause(None)
                raise ValueError("capture failed")
        self.state.pause("acl_graph")
        self.state.resume("acl_graph")
        self.state.assert_active()

    def test_region_cannot_nest_a_different_pool(self):
        with self.state.region(WEIGHTS):
            with self.assertRaisesRegex(RuntimeError, "Cannot nest"):
                with self.state.region(KV):
                    self.fail("a nested pool changed allocator ownership")
        with self.state.region(KV):
            pass

    def test_same_tag_keeps_distinct_policy_pools(self):
        with self.state.region(WEIGHTS) as backup:
            pass
        with self.state.region(WEIGHTS_DISCARD) as discard:
            self.assertNotEqual(backup.pool.id, discard.pool.id)
        self.assertEqual(self.backend.created, [WEIGHTS, WEIGHTS_DISCARD])

    def test_one_tag_can_share_or_separate_graph_pools_by_scope(self):
        shared = PoolKey("acl_graph", AllocationPolicy.DISCARD, "joint")
        target = PoolKey("acl_graph", AllocationPolicy.DISCARD, "target")
        draft = PoolKey("acl_graph", AllocationPolicy.DISCARD, "draft")
        with self.state.region(shared) as first:
            shared_pool = self.state.graph_pool(owned=first, requested=None)
        with self.state.region(shared) as second:
            self.assertIs(first, second)
            self.assertEqual(
                self.state.graph_pool(owned=second, requested=shared_pool), shared_pool
            )
        with self.state.region(target) as target_pool:
            with self.assertRaisesRegex(ValueError, "not the managed pool"):
                self.state.graph_pool(owned=target_pool, requested=shared_pool)
        with self.state.region(draft) as draft_pool:
            self.assertNotEqual(target_pool.pool.id, draft_pool.pool.id)
        self.state.pause("acl_graph")
        self.assertEqual(self.backend.created, [shared, target, draft])
        with self.assertRaises(RuntimeError):
            self.state.assert_active()

    def test_resident_storage_does_not_create_or_borrow_a_managed_pool(self):
        resident = PoolKey("acl_graph", AllocationPolicy.REMAIN, "draft")
        with self.state.region(GRAPH) as graph:
            managed_pool = graph.pool.id
        with self.state.region(resident) as graph:
            self.assertIsNone(graph)
            self.assertIsNone(self.state.graph_pool(owned=graph, requested=None))
            with self.assertRaisesRegex(ValueError, "cannot use a managed pool"):
                self.state.graph_pool(owned=graph, requested=managed_pool)
        self.state.pause("acl_graph")
        self.assertEqual(self.backend.created, [GRAPH])

    def test_parameter_and_draft_restore_policies_are_explicit(self):
        for has_draft, weight, draft, valid in (
            (False, "cpu_backup", None, True),
            (False, "discard", None, True),
            (False, None, None, False),
            (True, "discard", "retain", True),
            (True, "discard", "refresh", True),
            (True, "discard", None, False),
        ):
            with self.subTest(has_draft=has_draft, weight=weight, draft=draft):
                args = dict(
                    has_draft_weights=has_draft,
                    weight_policy=weight,
                    draft_policy=draft,
                )
                if valid:
                    validate_memory_saver_scope(**args)
                else:
                    with self.assertRaises(ValueError):
                        validate_memory_saver_scope(**args)


if __name__ == "__main__":
    unittest.main()
