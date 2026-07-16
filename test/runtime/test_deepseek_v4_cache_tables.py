from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


def _load_module():
    path = (
        pathlib.Path(__file__).parents[2]
        / "python/tokenspeed/runtime/deepseek_v4_cache_tables.py"
    )
    with mock.patch.dict(sys.modules):
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

    def test_invalid_flat_source_shapes_fail_closed(self):
        cases = (
            (
                "table/base group mismatch",
                None,
                {"history": object(), "state": object()},
                {"history": [0]},
                False,
            ),
            ("requires both", None, {"history": object()}, None, False),
            (
                "exactly one",
                {"history": object()},
                {"history": object()},
                {"history": [0]},
                False,
            ),
            ("missing", None, None, None, True),
            ("at least one", None, {}, {}, False),
        )
        for message, paged, flat, bases, required in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    _tables.resolve_deepseek_v4_cache_table_source(
                        paged_tables=paged,
                        paged_base_offsets=None,
                        flat_tables=flat,
                        flat_base_offsets=bases,
                        require_source=required,
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
        cases = (
            ("target backend", False, None, None, None, "group-keyed cache locations"),
            ("missing draft backend", True, "MTP", None, None, "draft backend"),
            ("draft backend", True, "MTP", False, "same-plan", "group-keyed draft"),
            ("plan mismatch", True, "MTP", True, "different-plan", "shared plan"),
        )

        for (
            name,
            target_group_keyed,
            algorithm,
            draft_group_keyed,
            draft_fingerprint,
            message,
        ) in cases:
            with self.subTest(name=name):
                shared = SimpleNamespace(group_id="shared.history")
                target_plan = SimpleNamespace(plan_fingerprint="same-plan")
                target_pool = SimpleNamespace(
                    flat_memory_plan=target_plan,
                    scheduler_group_specs=(shared,),
                    paged_cache_group_specs=(shared,),
                )
                target_backend = SimpleNamespace(
                    requires_group_keyed_cache_locs=target_group_keyed
                )
                draft_backend = (
                    SimpleNamespace(requires_group_keyed_cache_locs=draft_group_keyed)
                    if draft_group_keyed is not None
                    else None
                )
                draft_pool = (
                    SimpleNamespace(
                        flat_memory_plan=SimpleNamespace(
                            plan_fingerprint=draft_fingerprint
                        ),
                        paged_cache_group_specs=(shared,),
                    )
                    if draft_group_keyed is not None
                    else None
                )

                with self.assertRaisesRegex(RuntimeError, message):
                    _tables.resolve_deepseek_v4_flat_loc_policy(
                        target_backend=target_backend,
                        target_pool=target_pool,
                        draft_backend=draft_backend,
                        draft_pool=draft_pool,
                        speculative_algorithm=algorithm,
                    )


if __name__ == "__main__":
    unittest.main()
