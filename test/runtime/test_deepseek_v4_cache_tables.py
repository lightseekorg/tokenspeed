from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace


def _load_module():
    path = (
        pathlib.Path(__file__).parents[2]
        / "python/tokenspeed/runtime/deepseek_v4_cache_tables.py"
    )
    spec = importlib.util.spec_from_file_location("_v4_cache_tables", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_tables = _load_module()


class DeepseekV4CacheTableSourceTest(unittest.TestCase):
    def test_flat_requires_explicit_base_for_every_group(self):
        source = _tables.resolve_deepseek_v4_cache_table_source(
            paged_tables=None,
            paged_base_offsets=None,
            flat_tables={"history": object(), "state": object()},
            flat_base_offsets={"history": [0], "state": [7]},
        )
        self.assertEqual(source.kind, "flat")
        self.assertEqual(set(source.tables), {"history", "state"})
        self.assertEqual(source.base_offsets["state"], [7])

    def test_flat_missing_or_extra_base_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "table/base group mismatch"):
            _tables.resolve_deepseek_v4_cache_table_source(
                paged_tables=None,
                paged_base_offsets=None,
                flat_tables={"history": object(), "state": object()},
                flat_base_offsets={"history": [0]},
            )
        with self.assertRaisesRegex(RuntimeError, "requires both"):
            _tables.resolve_deepseek_v4_cache_table_source(
                paged_tables=None,
                paged_base_offsets=None,
                flat_tables={"history": object()},
                flat_base_offsets=None,
            )

    def test_radix_and_flat_cannot_be_injected_together(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            _tables.resolve_deepseek_v4_cache_table_source(
                paged_tables={"history": object()},
                paged_base_offsets=None,
                flat_tables={"history": object()},
                flat_base_offsets={"history": [0]},
            )

    def test_radix_keeps_missing_full_history_base_compatibility(self):
        source = _tables.resolve_deepseek_v4_cache_table_source(
            paged_tables={"history": object()},
            paged_base_offsets=None,
            flat_tables=None,
            flat_base_offsets=None,
        )
        self.assertEqual(source.kind, "radix")
        self.assertEqual(source.base_offsets, {})

    def test_required_source_rejects_empty_abi(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            _tables.resolve_deepseek_v4_cache_table_source(
                paged_tables=None,
                paged_base_offsets=None,
                flat_tables=None,
                flat_base_offsets=None,
                require_source=True,
            )
        with self.assertRaisesRegex(RuntimeError, "at least one"):
            _tables.resolve_deepseek_v4_cache_table_source(
                paged_tables=None,
                paged_base_offsets=None,
                flat_tables={},
                flat_base_offsets={},
            )

    def test_legacy_loc_mirror_requires_one_stride1_matching_group(self):
        safe = SimpleNamespace(
            group_id="legacy",
            family="history",
            retention="full_history",
            table_layout="absolute",
            entry_stride_tokens=1,
            block_size_tokens=16,
        )
        compressed = SimpleNamespace(
            group_id="v4.compressed",
            family="history",
            retention="full_history",
            table_layout="absolute",
            entry_stride_tokens=4,
            block_size_tokens=256,
        )

        self.assertEqual(
            _tables.legacy_flat_loc_group_id([safe], legacy_page_size=16),
            "legacy",
        )
        self.assertIsNone(
            _tables.legacy_flat_loc_group_id([compressed], legacy_page_size=16)
        )
        self.assertIsNone(
            _tables.legacy_flat_loc_group_id([safe, compressed], legacy_page_size=16)
        )

    def test_v4_flat_loc_policy_preserves_owner_local_group_sets(self):
        shared = SimpleNamespace(group_id="shared.history")
        target_state = SimpleNamespace(group_id="target.state")
        draft_only = SimpleNamespace(group_id="draft.state")
        plan = SimpleNamespace(plan_fingerprint="same-plan")
        target_pool = SimpleNamespace(
            flat_memory_plan=plan,
            scheduler_group_specs=(shared, target_state, draft_only),
            paged_cache_group_specs=(shared, target_state),
        )
        draft_pool = SimpleNamespace(
            flat_memory_plan=plan,
            paged_cache_group_specs=(shared, draft_only),
        )
        backend = SimpleNamespace(requires_group_keyed_cache_locs=True)

        policy = _tables.resolve_deepseek_v4_flat_loc_policy(
            target_backend=backend,
            target_pool=target_pool,
            draft_backend=backend,
            draft_pool=draft_pool,
            speculative_algorithm="MTP",
        )

        self.assertIsNotNone(policy)
        self.assertEqual(policy.target_group_ids, ("shared.history", "target.state"))
        self.assertEqual(policy.draft_group_ids, ("shared.history", "draft.state"))

    def test_v4_flat_spec_cannot_borrow_target_page_domain(self):
        shared = SimpleNamespace(group_id="shared.history")
        plan = SimpleNamespace(plan_fingerprint="same-plan")
        target_pool = SimpleNamespace(
            flat_memory_plan=plan,
            scheduler_group_specs=(shared,),
            paged_cache_group_specs=(shared,),
        )
        backend = SimpleNamespace(requires_group_keyed_cache_locs=True)

        with self.assertRaisesRegex(RuntimeError, "group-keyed cache locations"):
            _tables.resolve_deepseek_v4_flat_loc_policy(
                target_backend=SimpleNamespace(requires_group_keyed_cache_locs=False),
                target_pool=target_pool,
                draft_backend=None,
                draft_pool=None,
                speculative_algorithm=None,
            )

        with self.assertRaisesRegex(RuntimeError, "draft backend"):
            _tables.resolve_deepseek_v4_flat_loc_policy(
                target_backend=backend,
                target_pool=target_pool,
                draft_backend=None,
                draft_pool=None,
                speculative_algorithm="MTP",
            )

        with self.assertRaisesRegex(RuntimeError, "group-keyed draft"):
            _tables.resolve_deepseek_v4_flat_loc_policy(
                target_backend=backend,
                target_pool=target_pool,
                draft_backend=SimpleNamespace(requires_group_keyed_cache_locs=False),
                draft_pool=SimpleNamespace(
                    flat_memory_plan=plan,
                    paged_cache_group_specs=(shared,),
                ),
                speculative_algorithm="MTP",
            )

        with self.assertRaisesRegex(RuntimeError, "shared plan"):
            _tables.resolve_deepseek_v4_flat_loc_policy(
                target_backend=backend,
                target_pool=target_pool,
                draft_backend=backend,
                draft_pool=SimpleNamespace(
                    flat_memory_plan=SimpleNamespace(plan_fingerprint="different-plan"),
                    paged_cache_group_specs=(shared,),
                ),
                speculative_algorithm="MTP",
            )


if __name__ == "__main__":
    unittest.main()
