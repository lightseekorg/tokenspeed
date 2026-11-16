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

"""CPU behavior tests; the absent NPU allocator is represented by a byte store."""

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


Mapping = load_module("mapping_under_test", "distributed/mapping.py").Mapping
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


class FakeNpu:
    def __init__(self):
        self.next_id = 1
        self.content = {}
        self.saved = {}
        self.fail_resume = False

    def make_pool(self, key):
        pool = FakePool((0, self.next_id))
        self.next_id += 1
        self.content[key] = bytearray(b"initial")
        return OwnedPool(object(), pool)

    def pause(self, tag):
        for key, data in list(self.content.items()):
            if tag is None or key.tag == tag:
                if key.policy is AllocationPolicy.CPU_BACKUP:
                    self.saved[key] = bytes(data)
                self.content[key] = None

    def resume(self, tag):
        if self.fail_resume:
            raise RuntimeError("injected H2D failure")
        for key, data in list(self.content.items()):
            if (tag is None or key.tag == tag) and data is None:
                self.content[key] = bytearray(self.saved.get(key, bytes(7)))


class PoolStateTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeNpu()
        self.state = PoolState(
            backend=self.backend,
            make_pool=self.backend.make_pool,
            synchronize=lambda: None,
        )

    def test_weights_restore_kv_discard_and_partial_resume_stays_blocked(self):
        with self.state.region(WEIGHTS):
            self.backend.content[WEIGHTS][:] = b"weights"
        with self.state.region(KV):
            self.backend.content[KV][:] = b"history"
        self.state.pause(None)
        self.state.resume("weights")
        self.assertEqual(self.backend.content[WEIGHTS], b"weights")
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        self.state.resume("kv_cache")
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        self.state.resume("acl_graph")
        self.state.assert_active()
        self.assertEqual(self.backend.content[KV], bytes(7))

    def test_region_reentry_reuses_pool_and_preserves_updated_content(self):
        with self.state.region(WEIGHTS) as first:
            self.backend.content[WEIGHTS][:] = b"changed"
        with self.state.region(WEIGHTS) as second:
            self.assertIs(first, second)
        self.assertEqual(self.backend.content[WEIGHTS], b"changed")

    def test_allocation_blocked_while_paused_even_before_first_allocation(self):
        self.state.pause("weights")
        with self.assertRaises(RuntimeError):
            with self.state.region(WEIGHTS):
                self.fail("unmapped pool became available")
        self.state.resume("weights")
        with self.state.region(WEIGHTS):
            pass

    def test_pause_all_before_any_pool_blocks_all_three_tags(self):
        self.state.pause(None)
        with self.assertRaises(RuntimeError):
            self.state.assert_active()
        for key in (WEIGHTS, KV, GRAPH):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                with self.state.region(key):
                    self.fail("paused empty tag became usable")
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
        self.assertEqual(self.backend.content[GRAPH], b"initial")

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

    def test_explicit_shared_graph_pool_preserved_foreign_pool_rejected(self):
        with self.state.region(GRAPH) as owned:
            selected = self.state.graph_pool(owned=owned, requested=None)
            self.assertEqual(
                self.state.graph_pool(owned=owned, requested=selected),
                selected,
            )
            with self.assertRaises(ValueError):
                self.state.graph_pool(owned=owned, requested=(0, 99))

    def test_same_tag_distinct_policies_both_release(self):
        with self.state.region(WEIGHTS) as backup:
            pass
        with self.state.region(WEIGHTS_DISCARD) as discard:
            self.assertNotEqual(backup.pool.id, discard.pool.id)
        self.state.pause("weights")
        self.state.resume("weights")
        self.assertEqual(self.backend.content[WEIGHTS], b"initial")
        self.assertEqual(self.backend.content[WEIGHTS_DISCARD], bytes(7))

    def test_one_tag_can_share_or_separate_graph_pools_by_scope(self):
        shared = PoolKey("acl_graph", AllocationPolicy.DISCARD, "joint")
        target = PoolKey("acl_graph", AllocationPolicy.DISCARD, "target")
        draft = PoolKey("acl_graph", AllocationPolicy.DISCARD, "draft")
        with self.state.region(shared) as first:
            shared_pool = self.state.graph_pool(owned=first, requested=None)
        with self.state.region(shared) as second:
            self.assertEqual(
                self.state.graph_pool(owned=second, requested=shared_pool), shared_pool
            )
        with self.state.region(target) as target_pool:
            with self.assertRaisesRegex(ValueError, "not the managed pool"):
                self.state.graph_pool(owned=target_pool, requested=shared_pool)
        with self.state.region(draft) as draft_pool:
            self.assertNotEqual(target_pool.pool.id, draft_pool.pool.id)
        self.state.pause("acl_graph")
        self.assertEqual(self.backend.content, dict.fromkeys((shared, target, draft)))

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
        self.assertEqual(self.backend.content, {GRAPH: None})

    def test_unsupported_deployments_fail_before_device_work(self):
        valid = dict(
            mapping=Mapping(),
            emulate_rank_zero=False,
            disaggregation_mode="null",
            has_draft_weights=False,
            speculative_draft_tree=False,
            speculative_topk=1,
            speculative_sampling=False,
            weight_policy="cpu_backup",
            draft_policy=None,
        )
        for size in (1, 2, 4, 8):
            for policy in ("cpu_backup", "discard"):
                with self.subTest(size=size, policy=policy):
                    validate_memory_saver_scope(
                        **(
                            valid
                            | {
                                "mapping": Mapping(world_size=size),
                                "weight_policy": policy,
                            }
                        )
                    )
        for key, value in [
            ("mapping", Mapping(world_size=3)),
            ("mapping", Mapping(world_size=2, nnodes=2)),
            ("mapping", Mapping(world_size=2, attn_tp_size=1)),
            ("mapping", Mapping(world_size=2, dense_tp_size=1)),
            ("mapping", Mapping(world_size=2, moe_ep_size=2)),
            ("mapping", Mapping(world_size=2, vision_tp_size=1)),
            ("mapping", Mapping(world_size=2, pp_size=2)),
            ("mapping", Mapping(world_size=2, attn_qcp_size=2)),
            ("mapping", Mapping(world_size=2, attn_kvp_size=2)),
            ("emulate_rank_zero", True),
            ("disaggregation_mode", "decode"),
            ("speculative_draft_tree", True),
            ("weight_policy", None),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_memory_saver_scope(**(valid | {key: value}))

    def test_draft_discard_requires_policy_and_keeps_chain_requirements(self):
        args = dict(
            mapping=Mapping(),
            emulate_rank_zero=False,
            disaggregation_mode="null",
            has_draft_weights=True,
            speculative_draft_tree=False,
            speculative_topk=1,
            speculative_sampling=False,
            weight_policy="discard",
            draft_policy="retain",
        )
        validate_memory_saver_scope(**args)
        validate_memory_saver_scope(**(args | {"draft_policy": "refresh"}))
        validate_memory_saver_scope(
            **(args | {"has_draft_weights": False, "draft_policy": None})
        )
        for key, value in [
            ("draft_policy", None),
            ("speculative_topk", 2),
            ("speculative_sampling", True),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_memory_saver_scope(**(args | {key: value}))


if __name__ == "__main__":
    unittest.main()
