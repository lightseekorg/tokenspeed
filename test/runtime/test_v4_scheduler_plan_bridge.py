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

import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _ROOT / "python" / "tokenspeed" / "runtime" / "configs"
_SCHEDULER_UTILS = (
    _ROOT / "python" / "tokenspeed" / "runtime" / "engine" / "scheduler_utils.py"
)


def _load(mod_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


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


with mock.patch.dict(sys.modules):
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


def _make_scheduler_config(**overrides):
    pool = _build_pool()
    groups = _bridge.pool_to_paged_cache_groups(pool)
    pools = _bridge.pool_to_flat_block_pools(pool)
    kwargs = dict(
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
    kwargs.update(overrides)
    return _bridge.make_config(**kwargs), groups, pools


class TestV4SchedulerPlanBridge(unittest.TestCase):
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
        config, groups, pools = _make_scheduler_config()

        self.assertEqual(config.paged_cache_groups, groups)
        self.assertEqual(config.flat_block_pools, pools)
        self.assertEqual(config.num_device_pages, 0)

    def test_make_config_structured_completion_truth_matrix(self):
        cases = (
            ("default disabled", {}, False),
            (
                "explicitly enabled",
                {"enable_structured_flat_kv_completion": True},
                True,
            ),
        )

        for name, overrides, expected in cases:
            with self.subTest(name=name):
                config, _, _ = _make_scheduler_config(**overrides)
                self.assertEqual(
                    config.enable_structured_flat_kv_completion,
                    expected,
                )
                self.assertEqual(config.uses_structured_flat_admission, expected)

    def test_make_config_rejects_nonzero_device_page_authority(self):
        with self.assertRaisesRegex(ValueError, "only device page authority"):
            _make_scheduler_config(num_device_pages=8)


if __name__ == "__main__":
    unittest.main()
