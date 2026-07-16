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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _ROOT / "python" / "tokenspeed" / "runtime" / "configs"
_SCHEDULER_UTILS = (
    _ROOT / "python" / "tokenspeed" / "runtime" / "engine" / "scheduler_utils.py"
)
_EVENT_LOOP = _ROOT / "python" / "tokenspeed" / "runtime" / "engine" / "event_loop.py"
_V4_BACKEND = (
    _ROOT
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "backends"
    / "deepseek_v4.py"
)
_CUDA_WRAPPER = (
    _ROOT / "python" / "tokenspeed" / "runtime" / "execution" / "cuda_graph_wrapper.py"
)


def _load(mod_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_flat_source_selector():
    tree = ast.parse(_CUDA_WRAPPER.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CudaGraphWrapper"
    )
    selector = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "_uses_flat_table_source"
    )
    seam = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ClassDef(
                    name="SourceSelector",
                    bases=[],
                    keywords=[],
                    body=[selector],
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
    )
    namespace = {}
    exec(compile(seam, str(_CUDA_WRAPPER), "exec"), namespace)
    return namespace["SourceSelector"]._uses_flat_table_source


class _ValueConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _SchedulerConfig:
    class Role:
        P = "prefill"
        D = "decode"
        Fused = "fused"

    @property
    def uses_structured_flat_admission(self) -> bool:
        return bool(
            getattr(self, "enable_structured_flat_kv_completion", False)
            and getattr(self, "flat_block_pools", ())
        )


def _install_scheduler_stubs() -> None:
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.device = lambda value: SimpleNamespace(type=value)
    sys.modules["torch"] = torch

    scheduler = types.ModuleType("tokenspeed_scheduler")
    scheduler.Cache = SimpleNamespace(
        WriteBackDoneEvent=type("WriteBackDoneEvent", (), {}),
        PrefetchDoneEvent=type("PrefetchDoneEvent", (), {}),
        LoadBackDoneEvent=type("LoadBackDoneEvent", (), {}),
    )
    scheduler.ExecutionEvent = type("ExecutionEvent", (), {})
    scheduler.ForwardEvent = SimpleNamespace(
        Abort=type("Abort", (), {}),
        ExtendResult=type("ExtendResult", (), {}),
        Finish=type("Finish", (), {}),
        UpdateReserveNumTokens=type("UpdateReserveNumTokens", (), {}),
    )
    scheduler.FlatBlockPoolConfig = _ValueConfig
    scheduler.PagedCacheGroupConfig = _ValueConfig
    scheduler.PagedCacheGroupFamily = SimpleNamespace(
        History="history",
        State="state",
    )
    scheduler.PagedCachePrefixRole = SimpleNamespace(
        HistoryAnchor="history_anchor",
        ContinuationState="continuation_state",
        None_="none",
    )
    scheduler.PagedCacheRetention = SimpleNamespace(
        FullHistory="full_history",
        SlidingWindow="sliding_window",
    )
    scheduler.PagedCacheTableLayout = SimpleNamespace(
        Absolute="absolute",
        BoundedWindow="bounded_window",
    )
    scheduler.PrefixCacheAdjunctSpec = _ValueConfig
    scheduler.RequestSpec = _ValueConfig
    scheduler.SchedulerConfig = _SchedulerConfig
    sys.modules["tokenspeed_scheduler"] = scheduler


_install_scheduler_stubs()
_contract = _load(
    "tokenspeed.runtime.configs.flat_kv_contract",
    _CONFIGS_DIR / "flat_kv_contract.py",
)
_paged = _load(
    "paged_cache_spec_v4_scheduler_bridge_test",
    _CONFIGS_DIR / "paged_cache_spec.py",
)
sys.modules["tokenspeed.runtime.configs.paged_cache_spec"] = _paged
_plan = _load(
    "flat_memory_plan_v4_scheduler_bridge_test",
    _CONFIGS_DIR / "flat_memory_plan.py",
)
_bridge = _load("scheduler_utils_v4_plan_bridge_test", _SCHEDULER_UTILS)


def _component(owner: str, group_id: str, name: str, nbytes: int):
    return _plan.FlatComponentTensorPlan(
        owner=owner,
        group_id=group_id,
        layer=0,
        component=name,
        dtype="uint8",
        shape_per_block=(nbytes,),
        stride_bytes=(1,),
        alignment_bytes=1,
        bytes_per_block=nbytes,
    )


def _build_pool():
    target_shared = _paged.PagedCacheGroupSpec(
        "shared",
        "full_history",
        4,
        2,
        None,
        pool_id="pool.shared",
        owner_mask=_paged.CACHE_OWNER_TARGET,
        required_producer_domain_mask=1,
    )
    draft_shared = _paged.PagedCacheGroupSpec(
        "shared",
        "full_history",
        4,
        2,
        None,
        pool_id="pool.shared",
        owner_mask=_paged.CACHE_OWNER_DRAFT,
        required_producer_domain_mask=4,
    )
    draft_only = _paged.PagedCacheGroupSpec(
        "draft.state",
        "sliding_window",
        2,
        1,
        8,
        family="state",
        pool_id="pool.draft",
        prefix_role="continuation_state",
        table_layout="bounded_window",
        owner_mask=_paged.CACHE_OWNER_DRAFT,
        required_producer_domain_mask=8,
    )
    plan = _plan.build_v4_flat_memory_plan(
        max_total_tokens=64,
        target_group_specs=[target_shared],
        target_group_page_counts={"shared": 5},
        target_components=[_component("target", "shared", "kv", 16)],
        draft_group_specs=[draft_shared, draft_only],
        draft_group_page_counts={"shared": 7, "draft.state": 3},
        draft_components=[
            _component("draft", "shared", "kv", 8),
            _component("draft", "draft.state", "state", 4),
        ],
    )
    pool_by_id = {pool.pool_id: pool for pool in plan.pools}
    return SimpleNamespace(
        paged_cache_group_specs=plan.target_owner_group_specs,
        paged_cache_group_page_counts={"shared": 5},
        scheduler_group_specs=plan.scheduler_group_specs,
        scheduler_group_page_counts={
            spec.group_id: pool_by_id[spec.pool_id].total_blocks
            for spec in plan.scheduler_group_specs
        },
        flat_memory_plan=plan,
    )


class TestV4SchedulerPlanBridge(unittest.TestCase):
    def test_v4_backend_declares_flat_and_spec_capability(self):
        tree = ast.parse(_V4_BACKEND.read_text())
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DeepseekV4AttentionBackend"
        )
        assignments = {
            target.id: node.value
            for node in backend.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance((target := node.targets[0]), ast.Name)
        }
        for field in (
            "uses_flat_cache_groups",
            "flat_spec_capable",
            "requires_group_keyed_cache_locs",
        ):
            self.assertIn(field, assignments)
            self.assertIsInstance(assignments[field], ast.Constant)
            self.assertIs(assignments[field].value, True)

    def test_dual_source_v4_selects_flat_only_for_arena_plan(self):
        select_flat = _load_flat_source_selector()
        backend = SimpleNamespace(
            uses_flat_cache_groups=True,
            uses_paged_cache_groups=True,
        )

        self.assertFalse(select_flat(SimpleNamespace(flat_memory_plan=None), backend))
        self.assertTrue(
            select_flat(SimpleNamespace(flat_memory_plan=object()), backend)
        )
        self.assertTrue(
            select_flat(
                SimpleNamespace(flat_memory_plan=None),
                SimpleNamespace(
                    uses_flat_cache_groups=True,
                    uses_paged_cache_groups=False,
                ),
            )
        )

    def test_scheduler_uses_union_specs_and_physical_pool_capacities(self):
        pool = _build_pool()
        groups = {
            group.group_id: group for group in _bridge.pool_to_paged_cache_groups(pool)
        }
        pools = _bridge.pool_to_flat_block_pools(pool)

        self.assertEqual(set(groups), {"shared", "draft.state"})
        self.assertEqual(groups["shared"].total_pages, 7)
        self.assertEqual(groups["shared"].block_size, 8)
        self.assertEqual(groups["shared"].pool_id, "pool.shared")
        self.assertEqual(groups["shared"].required_producer_domain_mask, 1 | 4)
        self.assertEqual(
            groups["draft.state"].table_layout,
            "bounded_window",
        )
        self.assertEqual(
            [(item.pool_id, item.total_blocks, item.bytes_per_block) for item in pools],
            [("pool.draft", 3, 4), ("pool.shared", 7, 24)],
        )

    def test_make_config_carries_canonical_flat_pools(self):
        pool = _build_pool()
        groups = _bridge.pool_to_paged_cache_groups(pool)
        pools = _bridge.pool_to_flat_block_pools(pool)
        config = _bridge.make_config(
            num_device_pages=0,
            max_scheduled_tokens=16,
            max_batch_size=2,
            page_size=8,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            prefetch_threshold=4,
            role="null",
            paged_cache_groups=groups,
            flat_block_pools=pools,
        )

        self.assertEqual(config.paged_cache_groups, groups)
        self.assertEqual(config.flat_block_pools, pools)
        self.assertEqual(config.num_device_pages, 0)
        self.assertFalse(config.enable_structured_flat_kv_completion)

        structured_config = _bridge.make_config(
            num_device_pages=0,
            max_scheduled_tokens=16,
            max_batch_size=2,
            page_size=8,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            prefetch_threshold=4,
            role="null",
            paged_cache_groups=groups,
            flat_block_pools=pools,
            enable_structured_flat_kv_completion=True,
        )
        self.assertTrue(structured_config.enable_structured_flat_kv_completion)
        self.assertTrue(structured_config.uses_structured_flat_admission)

        with self.assertRaisesRegex(ValueError, "only device page authority"):
            _bridge.make_config(
                num_device_pages=8,
                max_scheduled_tokens=16,
                max_batch_size=2,
                page_size=8,
                num_host_pages=0,
                disable_l2_cache=True,
                enable_l3_storage=False,
                prefetch_threshold=4,
                role="null",
                paged_cache_groups=groups,
                flat_block_pools=pools,
            )

    def test_legacy_pool_has_no_explicit_flat_pool_config(self):
        legacy_spec = _paged.PagedCacheGroupSpec(
            "legacy",
            "full_history",
            4,
            1,
            None,
        )
        legacy_pool = SimpleNamespace(
            paged_cache_group_specs=(legacy_spec,),
            paged_cache_group_page_counts={"legacy": 9},
        )

        groups = _bridge.pool_to_paged_cache_groups(legacy_pool)
        self.assertEqual([group.group_id for group in groups], ["legacy"])
        self.assertEqual(groups[0].pool_id, "default")
        self.assertEqual(_bridge.pool_to_flat_block_pools(legacy_pool), [])

        config = _bridge.make_config(
            num_device_pages=9,
            max_scheduled_tokens=16,
            max_batch_size=2,
            page_size=4,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            prefetch_threshold=4,
            role="null",
            paged_cache_groups=groups,
        )
        self.assertEqual(config.num_device_pages, 9)

    def test_event_loop_passes_flat_pools_to_scheduler_config(self):
        tree = ast.parse(_EVENT_LOOP.read_text())
        make_config_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "make_config"
        ]
        self.assertEqual(len(make_config_calls), 1)
        keywords = {
            keyword.arg: keyword.value for keyword in make_config_calls[0].keywords
        }
        self.assertIn("flat_block_pools", keywords)
        self.assertIsInstance(keywords["flat_block_pools"], ast.Name)
        self.assertEqual(keywords["flat_block_pools"].id, "flat_block_pools")
        self.assertIn("enable_structured_flat_kv_completion", keywords)
        structured = keywords["enable_structured_flat_kv_completion"]
        self.assertIsInstance(structured, ast.Call)
        self.assertIsInstance(structured.func, ast.Name)
        self.assertEqual(structured.func.id, "bool")
        self.assertEqual(len(structured.args), 1)
        self.assertIsInstance(structured.args[0], ast.Name)
        self.assertEqual(structured.args[0].id, "flat_block_pools")
        num_device_pages = keywords["num_device_pages"]
        self.assertIsInstance(num_device_pages, ast.IfExp)
        self.assertIsInstance(num_device_pages.test, ast.Name)
        self.assertEqual(num_device_pages.test.id, "flat_block_pools")
        self.assertIsInstance(num_device_pages.body, ast.Constant)
        self.assertEqual(num_device_pages.body.value, 0)

    def test_scheduler_observability_uses_native_config_authority(self):
        self.assertEqual(_bridge.scheduler_backend_identity(True), ("flat", "ON"))
        self.assertEqual(_bridge.scheduler_backend_identity(False), ("radix", "OFF"))

        self.assertEqual(
            _bridge.scheduler_admission_path(
                flat_kvcache_ext=True,
                config=SimpleNamespace(uses_structured_flat_admission=True),
            ),
            "structured-flat",
        )
        self.assertEqual(
            _bridge.scheduler_admission_path(
                flat_kvcache_ext=True,
                config=SimpleNamespace(uses_structured_flat_admission=False),
            ),
            "legacy-flat-compat",
        )
        self.assertEqual(
            _bridge.scheduler_admission_path(
                flat_kvcache_ext=False,
                config=object(),
            ),
            "radix",
        )

    def test_forward_bridge_fails_before_allocating_past_plan_width(self):
        forward_op = SimpleNamespace(
            flat_block_tables={"shared": [[1, 2, 3]]},
        )

        with self.assertRaisesRegex(ValueError, "plan requires 3"):
            _bridge.flat_block_tables_from_forward_op(
                forward_op,
                device="cpu",
                num_reqs=1,
                max_cols_by_group={"shared": 2},
            )

    def test_forward_bridge_requires_the_planned_group_union(self):
        forward_op = SimpleNamespace(
            flat_block_tables={"shared": [[1, 2]]},
        )

        with self.assertRaisesRegex(ValueError, "missing=.*draft.state"):
            _bridge.flat_block_tables_from_forward_op(
                forward_op,
                device="cpu",
                num_reqs=1,
                max_cols_by_group={"shared": 2, "draft.state": 1},
            )


if __name__ == "__main__":
    unittest.main()
