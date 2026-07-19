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
        / "python/tokenspeed/runtime/flat_cache_tables.py"
    )
    with mock.patch.dict(sys.modules):
        spec = importlib.util.spec_from_file_location("_v4_cache_tables", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


_tables = _load_module()
_SPECS = tuple(SimpleNamespace(group_id=group_id) for group_id in ("a", "b"))


def _backend(*, flat=True, paged=True, group_keyed=True):
    return SimpleNamespace(
        uses_flat_cache_groups=flat,
        uses_paged_cache_groups=paged,
        requires_group_keyed_cache_locs=group_keyed,
    )


def _pool(*, plan=None, owner_specs=_SPECS, scheduler_specs=_SPECS):
    return SimpleNamespace(
        flat_memory_plan=plan,
        paged_cache_group_specs=owner_specs,
        scheduler_group_specs=scheduler_specs,
    )


class DeepseekV4CacheTableSourceTest(unittest.TestCase):
    def test_binding_is_selected_once_from_scheduler_and_pool(self):
        cases = (
            ("radix build", False, _backend(), object(), "radix", (), False),
            ("dual flat", True, _backend(), object(), "flat", ("a", "b"), True),
            (
                "flat only",
                True,
                _backend(paged=False, group_keyed=False),
                None,
                "flat",
                ("a", "b"),
                False,
            ),
        )
        for name, active, backend, plan, kind, groups, group_keyed in cases:
            with self.subTest(name=name):
                binding = _tables.resolve_cache_table_binding(
                    backend=backend,
                    pool=_pool(plan=plan),
                    flat_scheduler_active=active,
                )
                self.assertEqual(
                    (binding.kind, binding.group_ids, binding.group_keyed_cache_locs),
                    (kind, groups, group_keyed),
                )

    def test_owner_view_reuses_only_the_same_planned_union(self):
        view = _tables.FlatCacheTableOwnerView(("target",))
        tables = {"target": object(), "draft": object()}
        bases = {"target": object(), "draft": object()}

        def source(generation, source_tables=tables):
            return _tables.CacheTableSource(
                kind="flat",
                tables=source_tables,
                base_offsets=bases,
                planned=True,
                generation=generation,
            )

        union = source(3)
        first = view.bind(union, owner="target")
        self.assertIs(view.bind(union, owner="target"), first)
        self.assertIsNot(view.bind(source(4), owner="target"), first)
        self.assertIsNot(
            view.bind(
                source(4, {"target": object(), "draft": object()}), owner="target"
            ),
            first,
        )
        self.assertEqual(tuple(first.tables), ("target",))
        self.assertIs(first.tables["target"], tables["target"])
        self.assertIs(first.base_offsets["target"], bases["target"])

    def test_flat_source_schema_is_atomic_and_group_complete(self):
        tables = {"history": object(), "state": object()}
        bases = {"history": [0], "state": [7]}
        source = _tables.resolve_cache_table_source(
            paged_tables=None,
            paged_base_offsets=None,
            flat_tables=tables,
            flat_base_offsets=bases,
        )
        self.assertEqual(source.kind, "flat")
        self.assertIs(source.tables, tables)
        self.assertIs(source.base_offsets, bases)

        cases = (
            ("table/base group mismatch", None, tables, {"history": [0]}, False),
            ("requires both", None, {"history": object()}, None, False),
            ("exactly one", {"history": object()}, tables, bases, False),
            ("missing", None, None, None, True),
            ("at least one", None, {}, {}, False),
        )
        for message, paged, flat, offsets, required in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                _tables.resolve_cache_table_source(
                    paged_tables=paged,
                    paged_base_offsets=None,
                    flat_tables=flat,
                    flat_base_offsets=offsets,
                    require_source=required,
                )

    def test_speculative_bindings_preserve_owner_local_domains(self):
        shared = SimpleNamespace(group_id="shared.history")
        target = SimpleNamespace(group_id="target.state")
        draft = SimpleNamespace(group_id="draft.state")
        plan = SimpleNamespace(plan_fingerprint="same-plan")
        union = (shared, target, draft)
        target_pool = _pool(
            plan=plan, owner_specs=(shared, target), scheduler_specs=union
        )
        draft_pool = _pool(
            plan=plan, owner_specs=(shared, draft), scheduler_specs=union
        )
        backend = _backend()
        target_binding = _tables.resolve_cache_table_binding(
            backend=backend,
            pool=target_pool,
            flat_scheduler_active=True,
        )
        draft_binding = _tables.resolve_cache_table_binding(
            backend=backend,
            pool=draft_pool,
            flat_scheduler_active=True,
        )
        _tables.validate_speculative_flat_bindings(
            target_pool=target_pool,
            target_binding=target_binding,
            draft_pool=draft_pool,
            draft_binding=draft_binding,
            speculative_algorithm="MTP",
        )
        self.assertEqual(target_binding.group_ids, ("shared.history", "target.state"))
        self.assertEqual(draft_binding.group_ids, ("shared.history", "draft.state"))

    def test_speculative_bindings_reject_ambiguous_page_domains(self):
        cases = (
            ("target backend", False, None, None, "group-keyed cache locations"),
            ("missing draft backend", True, None, None, "draft backend"),
            ("draft backend", True, False, "same-plan", "group-keyed cache"),
            ("plan mismatch", True, True, "different-plan", "canonical plan object"),
        )
        shared = SimpleNamespace(group_id="shared.history")
        target_plan = SimpleNamespace(plan_fingerprint="same-plan")
        target_pool = _pool(
            plan=target_plan, owner_specs=(shared,), scheduler_specs=(shared,)
        )
        for name, target_keyed, draft_keyed, draft_fingerprint, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, message):
                target_binding = _tables.resolve_cache_table_binding(
                    backend=_backend(group_keyed=target_keyed),
                    pool=target_pool,
                    flat_scheduler_active=True,
                )
                draft_pool = (
                    _pool(
                        plan=SimpleNamespace(plan_fingerprint=draft_fingerprint),
                        owner_specs=(shared,),
                        scheduler_specs=(shared,),
                    )
                    if draft_keyed is not None
                    else None
                )
                draft_binding = (
                    _tables.resolve_cache_table_binding(
                        backend=_backend(group_keyed=draft_keyed),
                        pool=draft_pool,
                        flat_scheduler_active=True,
                    )
                    if draft_pool is not None
                    else None
                )
                _tables.validate_speculative_flat_bindings(
                    target_pool=target_pool,
                    target_binding=target_binding,
                    draft_pool=draft_pool,
                    draft_binding=draft_binding,
                    speculative_algorithm="MTP",
                )


if __name__ == "__main__":
    unittest.main()
