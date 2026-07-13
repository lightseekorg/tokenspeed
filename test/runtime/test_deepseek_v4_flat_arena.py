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
import contextlib
import importlib.util
import logging
import math
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _ROOT / "python" / "tokenspeed" / "runtime" / "configs"
_V4_CACHE_FILE = (
    _ROOT
    / "python"
    / "tokenspeed"
    / "runtime"
    / "layers"
    / "attention"
    / "kv_cache"
    / "deepseek_v4.py"
)
_REGISTRY_FILE = (
    _ROOT / "python" / "tokenspeed" / "runtime" / "layers" / "attention" / "registry.py"
)


def _load(mod_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeDType:
    def __init__(self, name: str, itemsize: int):
        self.name = name
        self.itemsize = itemsize

    def __repr__(self) -> str:
        return f"torch.{self.name}"


class _FakeTensorSlice:
    def __init__(self, parent: "_FakeTensor", index: int):
        self._parent = parent
        self._index = index
        self.shape = parent.shape[1:]
        self.dtype = parent.dtype
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize

    def zero_(self):
        self._parent.zero_pages.add(self._index)
        return self


class _FakeTensor:
    _next_ptr = 4096

    def __init__(self, shape, dtype, device, *, zeroed: bool):
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.device = device
        self.ndim = len(self.shape)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize
        self.zero_pages = set(range(self.shape[0])) if zeroed else set()
        self.all_zero = zeroed
        self._ptr = _FakeTensor._next_ptr
        _FakeTensor._next_ptr += 4096

    def __getitem__(self, index: int):
        return _FakeTensorSlice(self, index)

    def zero_(self):
        self.zero_pages = set(range(self.shape[0]))
        self.all_zero = True
        return self

    def stride(self, dim: int | None = None):
        strides = []
        running = 1
        for size in reversed(self.shape):
            strides.append(running)
            running *= size
        strides = tuple(reversed(strides))
        return strides if dim is None else strides[dim]

    def data_ptr(self) -> int:
        return self._ptr


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.uint8 = _FakeDType("uint8", 1)
        self.float32 = _FakeDType("float32", 4)
        self.int32 = _FakeDType("int32", 4)
        self.int64 = _FakeDType("int64", 8)
        self.bool = _FakeDType("bool", 1)
        self.Tensor = _FakeTensor
        self.dtype = _FakeDType
        self.allocations: list[_FakeTensor] = []
        self._utils = SimpleNamespace(
            _element_size=lambda dtype: dtype.itemsize,
        )

    def empty(self, shape, *, dtype, device):
        tensor = _FakeTensor(shape, dtype, device, zeroed=False)
        self.allocations.append(tensor)
        return tensor

    def zeros(self, shape, *, dtype, device):
        tensor = _FakeTensor(shape, dtype, device, zeroed=True)
        self.allocations.append(tensor)
        return tensor

    @staticmethod
    def is_tensor(value) -> bool:
        return isinstance(value, _FakeTensor)


class _FakeMemorySaverAdapter:
    @classmethod
    def create(cls, *, enable: bool):
        instance = cls()
        instance.enabled = enable
        return instance

    @contextlib.contextmanager
    def region(self, **kwargs):
        del kwargs
        yield


class _FakeBaseTokenToKVPool:
    def __init__(
        self,
        *,
        size,
        dtype,
        device,
        max_batch_size,
        max_context_len,
        page_size,
        rank,
    ):
        del max_batch_size, max_context_len
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.rank = rank


def _install_stubs() -> _FakeTorch:
    fake_torch = _FakeTorch()
    sys.modules["torch"] = fake_torch

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.prod = math.prod
    sys.modules["numpy"] = fake_numpy

    ops = types.ModuleType("tokenspeed.runtime.layers.attention.deepseek_v4_ops")
    ops.deepseek_v4_compressed_slot_mapping = lambda **kwargs: kwargs.get("out")
    sys.modules[ops.__name__] = ops

    base = types.ModuleType("tokenspeed.runtime.layers.attention.kv_cache.base")
    base.BaseTokenToKVPool = _FakeBaseTokenToKVPool
    sys.modules[base.__name__] = base

    utils = types.ModuleType("tokenspeed.runtime.utils")
    utils.get_colorful_logger = logging.getLogger
    sys.modules[utils.__name__] = utils

    common = types.ModuleType("tokenspeed.runtime.utils.common")
    common.ceil_div = lambda numer, denom: -(-numer // denom)
    sys.modules[common.__name__] = common

    memory = types.ModuleType("tokenspeed.runtime.utils.torch_memory_saver_adapter")
    memory.TorchMemorySaverAdapter = _FakeMemorySaverAdapter
    sys.modules[memory.__name__] = memory
    return fake_torch


_torch = _install_stubs()
_paged = _load(
    "tokenspeed.runtime.configs.paged_cache_spec",
    _CONFIGS_DIR / "paged_cache_spec.py",
)
_spec = _load(
    "tokenspeed.runtime.configs.deepseek_v4_cache_spec",
    _CONFIGS_DIR / "deepseek_v4_cache_spec.py",
)
_plan = _load(
    "tokenspeed.runtime.configs.flat_memory_plan",
    _CONFIGS_DIR / "flat_memory_plan.py",
)
_cache = _load("deepseek_v4_flat_arena_test", _V4_CACHE_FILE)


def _hf_config():
    return SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )


def _layout(layer_indices=(0, 1, 2)):
    return _cache.deepseek_v4_cache_layout_from_config(
        _hf_config(),
        page_size=256,
        use_fp4_indexer_cache=True,
        layer_indices=layer_indices,
    )


def _build_plan(**overrides):
    kwargs = dict(
        target_layout=_layout(),
        target_hf_config=_hf_config(),
        target_layer_num=3,
        target_max_live_requests=2,
        target_max_context_len=512,
        max_scheduled_tokens=128,
        max_total_tokens=1024,
        draft_layout=_layout((0, 1)),
        draft_hf_config=_hf_config(),
        draft_layer_num=2,
        draft_max_live_requests=4,
        draft_max_context_len=512,
    )
    kwargs.update(overrides)
    return _cache.build_deepseek_v4_flat_memory_plan(**kwargs)


def _capture_cols(
    spec,
    *,
    context_len: int,
    chunk_tokens: int,
    verify_width: int,
    overlap: int,
) -> int:
    protected = (overlap + 1) * verify_width
    if spec.retention == "sliding_window":
        retained = min(spec.sliding_window_tokens, context_len)
        prefill_carry = overlap * chunk_tokens
        return (
            math.ceil((retained + prefill_carry + protected) / spec.block_size_tokens)
            + 1
        )
    return math.ceil((context_len + protected) / spec.block_size_tokens)


def _export_cols(
    spec,
    *,
    context_len: int,
    chunk_tokens: int,
    verify_width: int,
    overlap: int,
) -> int:
    protected = (overlap + 1) * verify_width
    if spec.retention == "sliding_window":
        retained = min(spec.sliding_window_tokens, context_len)
        in_flight_prefill = (overlap + 1) * chunk_tokens
        return (
            math.ceil(
                (retained + in_flight_prefill + protected) / spec.block_size_tokens
            )
            + 1
        )
    return math.ceil((context_len + protected) / spec.block_size_tokens)


class TestDeepseekV4FlatComponentSchemas(unittest.TestCase):
    def test_layout_produces_every_owner_component_plane(self):
        layout = _layout()
        specs = _spec.build_v4_cache_specs(
            _hf_config(),
            layer_ratio=layout.layer_ratio,
        )
        components = _cache.deepseek_v4_flat_component_plans(
            layout=layout,
            specs=specs,
            layer_num=3,
            owner="target",
        )

        identities = {
            (component.layer, component.component) for component in components
        }
        self.assertEqual(
            identities,
            {
                (0, "swa_kv"),
                (1, "swa_kv"),
                (1, "compressed_kv"),
                (1, "compressor_state"),
                (1, "indexer_kv"),
                (1, "indexer_state"),
                (2, "swa_kv"),
                (2, "compressed_kv"),
                (2, "compressor_state"),
            },
        )
        bytes_by_group = {
            group_id: sum(
                component.bytes_per_block
                for component in components
                if component.group_id == group_id
            )
            for group_id in {component.group_id for component in components}
        }
        self.assertEqual(
            bytes_by_group,
            _cache._deepseek_v4_cache_group_page_bytes(layout, specs, 3),
        )

    def test_target_draft_plan_uses_shared_max_capacity_and_combined_bytes(self):
        plan = _build_plan()
        pools = {pool.pool_id: pool for pool in plan.pools}
        scheduler_specs = {spec.group_id: spec for spec in plan.scheduler_group_specs}

        self.assertEqual(plan.max_total_tokens, 1024)
        self.assertEqual(
            scheduler_specs["v4.swa_kv"].owner_mask,
            _spec.CACHE_OWNER_TARGET | _spec.CACHE_OWNER_DRAFT,
        )
        swa_pool = pools["v4.swa"]
        self.assertEqual(
            {tensor.owner for tensor in swa_pool.tensors},
            {"target", "draft"},
        )
        self.assertEqual(
            swa_pool.bytes_per_block,
            sum(tensor.bytes_per_block for tensor in swa_pool.tensors),
        )
        target_swa_count = _paged.compute_paged_cache_group_page_counts(
            [
                spec
                for spec in plan.target_owner_group_specs
                if spec.group_id == "v4.swa_kv"
            ],
            max_live_requests=2,
            max_scheduled_tokens=128,
            max_total_tokens=1024,
            max_context_len=512,
        )["v4.swa_kv"]
        draft_swa_count = _paged.compute_paged_cache_group_page_counts(
            [
                spec
                for spec in plan.draft_owner_group_specs
                if spec.group_id == "v4.swa_kv"
            ],
            max_live_requests=4,
            max_scheduled_tokens=128,
            max_total_tokens=1024,
            max_context_len=512,
        )["v4.swa_kv"]
        self.assertEqual(swa_pool.total_blocks, max(target_swa_count, draft_swa_count))

    def test_plan_rejects_a_partial_scheduler_page(self):
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            _cache.build_deepseek_v4_flat_memory_plan(
                target_layout=_layout(),
                target_hf_config=_hf_config(),
                target_layer_num=3,
                target_max_live_requests=2,
                target_max_context_len=512,
                max_scheduled_tokens=128,
                max_total_tokens=1023,
            )

    def test_profile_returns_the_same_canonical_plan_consumed_by_arena(self):
        expected = _build_plan()
        allocation_count = len(_torch.allocations)
        actual = _cache.profile_deepseek_v4_flat_memory_plan(
            target_layout=_layout(),
            target_hf_config=_hf_config(),
            target_layer_num=3,
            target_max_live_requests=2,
            target_max_context_len=512,
            max_scheduled_tokens=128,
            available_cache_memory_bytes=expected.device_cache_total_bytes,
            draft_layout=_layout((0, 1)),
            draft_hf_config=_hf_config(),
            draft_layer_num=2,
            draft_max_live_requests=4,
            draft_max_context_len=512,
            max_total_tokens_cap=1024,
        )

        self.assertEqual(actual, expected)
        self.assertEqual(len(_torch.allocations), allocation_count)


class TestV4FlatMemoryAccounting(unittest.TestCase):
    def test_accounting_fields_are_exact_and_owner_local(self):
        plan = _build_plan(
            target_max_live_requests=4,
            target_max_graph_bs=3,
            draft_max_graph_bs=2,
            decode_input_tokens=8,
            overlap_schedule_depth=1,
        )
        target_specs = {spec.group_id: spec for spec in plan.target_owner_group_specs}
        draft_specs = {spec.group_id: spec for spec in plan.draft_owner_group_specs}
        table_plans = {item.group_id: item for item in plan.group_table_plans}

        self.assertEqual(set(table_plans), set(plan.scheduler_group_specs_by_id))
        for group_id, item in table_plans.items():
            with self.subTest(group_id=group_id):
                expected_target_capture = (
                    _capture_cols(
                        target_specs[group_id],
                        context_len=512,
                        chunk_tokens=128,
                        verify_width=8,
                        overlap=1,
                    )
                    if group_id in target_specs
                    else 0
                )
                expected_draft_capture = (
                    _capture_cols(
                        draft_specs[group_id],
                        context_len=512,
                        chunk_tokens=128,
                        verify_width=8,
                        overlap=1,
                    )
                    if group_id in draft_specs
                    else 0
                )
                owner_export_cols = [
                    _export_cols(
                        spec,
                        context_len=512,
                        chunk_tokens=128,
                        verify_width=8,
                        overlap=1,
                    )
                    for spec in (
                        target_specs.get(group_id),
                        draft_specs.get(group_id),
                    )
                    if spec is not None
                ]
                self.assertEqual(item.target_capture_cols, expected_target_capture)
                self.assertEqual(item.draft_capture_cols, expected_draft_capture)
                self.assertEqual(item.max_export_cols, max(owner_export_cols))
                self.assertEqual(item.max_live_descriptor_cols, max(owner_export_cols))

        expected_graph = 4 * (
            3
            * sum(
                table_plans[group_id].target_capture_cols + 1
                for group_id in target_specs
            )
            + 2
            * sum(
                table_plans[group_id].draft_capture_cols + 1 for group_id in draft_specs
            )
        )
        expected_forward = (
            4
            * 2
            * 4
            * (
                sum(item.max_export_cols for item in table_plans.values())
                + len(table_plans)
            )
        )
        expected_pool_metadata = sum(
            pool.total_blocks * _plan.V4_CPU_POOL_METADATA_BYTES_PER_BLOCK_ESTIMATE
            + _plan.V4_CPU_POOL_FIXED_BYTES_ESTIMATE
            for pool in plan.pools
        )
        expected_request_metadata = (
            _plan.V4_CPU_BLOCK_REF_BYTES_ESTIMATE
            * 4
            * sum(item.max_live_descriptor_cols for item in table_plans.values())
        )

        self.assertEqual(plan.forward_buffer_depth, 2)
        self.assertEqual(plan.target_graph_batch_rows, 3)
        self.assertEqual(plan.draft_graph_batch_rows, 2)
        self.assertEqual(plan.max_scheduled_batch_rows, 4)
        self.assertEqual(plan.graph_metadata_bytes, expected_graph)
        self.assertEqual(plan.forward_input_bytes, expected_forward)
        self.assertEqual(plan.cpu_forward_staging_bytes, expected_forward)
        self.assertEqual(
            plan.cpu_forward_export_bytes,
            expected_forward
            + 2 * len(table_plans) * _plan.V4_CPU_EXPORT_GROUP_HEADER_BYTES_ESTIMATE,
        )
        self.assertEqual(
            plan.cpu_pool_metadata_bytes_estimate,
            expected_pool_metadata,
        )
        self.assertEqual(
            plan.cpu_request_metadata_bytes_estimate,
            expected_request_metadata,
        )
        self.assertEqual(
            plan.device_cache_total_bytes,
            plan.payload_bytes + expected_graph + expected_forward,
        )
        self.assertEqual(
            plan.cpu_cache_metadata_total_bytes,
            plan.cpu_pool_metadata_bytes_estimate
            + plan.cpu_request_metadata_bytes_estimate
            + plan.cpu_forward_export_bytes
            + plan.cpu_forward_staging_bytes,
        )

    def test_fingerprint_preserves_owner_specific_graph_rows(self):
        common = dict(
            target_max_live_requests=4,
            draft_layout=_layout(),
            draft_layer_num=3,
            draft_max_live_requests=4,
        )
        target_heavy = _build_plan(
            target_max_graph_bs=3,
            draft_max_graph_bs=2,
            **common,
        )
        draft_heavy = _build_plan(
            target_max_graph_bs=2,
            draft_max_graph_bs=3,
            **common,
        )

        self.assertEqual(
            target_heavy.graph_metadata_bytes,
            draft_heavy.graph_metadata_bytes,
        )
        self.assertNotEqual(
            target_heavy.plan_fingerprint,
            draft_heavy.plan_fingerprint,
        )

    def test_accounting_is_monotonic_in_token_capacity(self):
        small = _build_plan(max_total_tokens=512)
        large = _build_plan(max_total_tokens=1024)

        self.assertLess(small.payload_bytes, large.payload_bytes)
        self.assertLess(
            small.cpu_pool_metadata_bytes_estimate,
            large.cpu_pool_metadata_bytes_estimate,
        )
        self.assertEqual(small.graph_metadata_bytes, large.graph_metadata_bytes)
        self.assertEqual(small.forward_input_bytes, large.forward_input_bytes)
        self.assertLess(small.device_cache_total_bytes, large.device_cache_total_bytes)

    def test_forward_depth_tracks_overlap_lifetime(self):
        single = _build_plan(decode_input_tokens=8, overlap_schedule_depth=0)
        overlapped = _build_plan(decode_input_tokens=8, overlap_schedule_depth=1)

        self.assertEqual(single.forward_buffer_depth, 1)
        self.assertEqual(overlapped.forward_buffer_depth, 2)
        self.assertGreaterEqual(
            overlapped.forward_input_bytes,
            2 * single.forward_input_bytes,
        )
        self.assertEqual(
            overlapped.cpu_forward_staging_bytes,
            overlapped.forward_input_bytes,
        )

    def test_width_guard_rejects_a_buffer_one_column_too_small(self):
        plan = _build_plan()
        group = plan.group_table_plans[0]

        _paged.require_flat_table_cols(
            group_id=group.group_id,
            purpose="export",
            actual_cols=group.max_export_cols,
            required_cols=group.max_export_cols,
        )
        with self.assertRaisesRegex(ValueError, "one column too small"):
            _paged.require_flat_table_cols(
                group_id=group.group_id,
                purpose="one column too small",
                actual_cols=group.max_export_cols - 1,
                required_cols=group.max_export_cols,
            )

    def test_profile_reserves_graph_and_forward_device_bytes(self):
        capped = _build_plan(
            target_max_live_requests=4,
            target_max_graph_bs=3,
            draft_max_graph_bs=2,
            decode_input_tokens=8,
            overlap_schedule_depth=1,
        )
        available = capped.device_cache_total_bytes - 1
        self.assertGreater(available, capped.payload_bytes)

        actual = _cache.profile_deepseek_v4_flat_memory_plan(
            target_layout=_layout(),
            target_hf_config=_hf_config(),
            target_layer_num=3,
            target_max_live_requests=4,
            target_max_context_len=512,
            target_max_graph_bs=3,
            max_scheduled_tokens=128,
            available_cache_memory_bytes=available,
            draft_layout=_layout((0, 1)),
            draft_hf_config=_hf_config(),
            draft_layer_num=2,
            draft_max_live_requests=4,
            draft_max_context_len=512,
            draft_max_graph_bs=2,
            max_total_tokens_cap=1024,
            decode_input_tokens=8,
            overlap_schedule_depth=1,
        )

        self.assertLess(actual.max_total_tokens, capped.max_total_tokens)
        self.assertLessEqual(actual.device_cache_total_bytes, available)


class TestV4FlatArenaSet(unittest.TestCase):
    def setUp(self):
        _torch.allocations.clear()
        self.plan = _build_plan()
        self.arena = _cache.V4FlatArenaSet(
            self.plan,
            device="cuda:0",
            enable_memory_saver=False,
        )

    def _component_tensors(self):
        return tuple(
            self.arena.tensor(
                component.owner,
                component.group_id,
                component.layer,
                component.component,
            )
            for pool in self.plan.pools
            for component in pool.tensors
        )

    def test_arena_allocates_plan_shapes_and_only_initializes_null_pages(self):
        self.assertEqual(self.arena.arena_generation, 0)
        self.assertEqual(
            len(_torch.allocations),
            sum(len(pool.tensors) for pool in self.plan.pools),
        )
        pool_by_group = {
            spec.group_id: next(
                pool for pool in self.plan.pools if pool.pool_id == spec.pool_id
            )
            for spec in self.plan.scheduler_group_specs
        }
        for pool in self.plan.pools:
            for component in pool.tensors:
                with self.subTest(
                    pool=pool.pool_id,
                    owner=component.owner,
                    component=component.component,
                ):
                    tensor = self.arena.tensor(
                        component.owner,
                        component.group_id,
                        component.layer,
                        component.component,
                    )
                    self.assertEqual(
                        tensor.shape,
                        (pool.total_blocks, *component.shape_per_block),
                    )
                    self.assertEqual(tensor.zero_pages, {0})
                    self.assertFalse(tensor.all_zero)
                    self.assertEqual(
                        tensor.shape[0],
                        pool_by_group[component.group_id].total_blocks,
                    )
        self.assertEqual(self.arena.get_kv_size_bytes(), self.plan.payload_bytes)

    def test_wake_repair_advances_once_and_resets_every_component_null_page(self):
        tensors = self._component_tensors()
        for tensor in tensors:
            tensor.zero_pages.clear()

        self.assertEqual(self.arena.repair_after_wake(expected_generation=1), 1)
        self.assertEqual(self.arena.arena_generation, 1)
        self.assertTrue(all(tensor.zero_pages == {0} for tensor in tensors))

        self.assertEqual(self.arena.repair_after_wake(), 2)
        self.assertEqual(self.arena.arena_generation, 2)

    def test_wake_repair_rejects_scheduler_generation_drift_before_mutation(self):
        tensors = self._component_tensors()
        for tensor in tensors:
            tensor.zero_pages.clear()

        with self.assertRaisesRegex(RuntimeError, "generation mismatch"):
            self.arena.repair_after_wake(expected_generation=2)

        self.assertEqual(self.arena.arena_generation, 0)
        self.assertTrue(all(not tensor.zero_pages for tensor in tensors))

    def test_target_and_draft_pools_are_nonallocating_views_of_one_arena(self):
        allocation_count = len(_torch.allocations)
        common = dict(
            size=self.plan.max_total_tokens,
            model_dtype=_torch.float32,
            device="cuda:0",
            enable_memory_saver=False,
            max_context_len=512,
            page_size=256,
            rank=0,
            hf_config=_hf_config(),
            max_scheduled_tokens=128,
            flat_arena_set=self.arena,
        )
        target = _cache.DeepseekV4TokenToKVPool(
            layout=_layout(),
            layer_num=3,
            max_batch_size=2,
            cache_owner="target",
            **common,
        )
        draft = _cache.DeepseekV4TokenToKVPool(
            layout=_layout((0, 1)),
            layer_num=2,
            max_batch_size=4,
            cache_owner="draft",
            **common,
        )

        self.assertEqual(len(_torch.allocations), allocation_count)
        self.assertIs(target.flat_arena_set, draft.flat_arena_set)
        self.assertEqual(target.arena_generation, 0)
        self.assertEqual(draft.arena_generation, 0)
        self.arena.repair_after_wake()
        self.assertEqual(target.arena_generation, 1)
        self.assertEqual(draft.arena_generation, 1)
        self.assertEqual(target.scheduler_group_specs, self.plan.scheduler_group_specs)
        self.assertEqual(target.owner_group_specs, self.plan.target_owner_group_specs)
        self.assertEqual(draft.owner_group_specs, self.plan.draft_owner_group_specs)
        self.assertEqual(
            target.flat_capture_cols_by_group,
            {
                item.group_id: item.target_capture_cols
                for item in self.plan.group_table_plans
                if item.target_capture_cols > 0
            },
        )
        self.assertEqual(
            draft.flat_capture_cols_by_group,
            {
                item.group_id: item.draft_capture_cols
                for item in self.plan.group_table_plans
                if item.draft_capture_cols > 0
            },
        )
        self.assertEqual(
            target.flat_max_export_cols_by_group,
            {
                item.group_id: item.max_export_cols
                for item in self.plan.group_table_plans
            },
        )
        self.assertEqual(target.paged_cache_group_specs, target.owner_group_specs)
        self.assertEqual(draft.paged_cache_group_specs, draft.owner_group_specs)
        self.assertIs(
            target.get_swa_kv_buffer(0),
            self.arena.tensor("target", "v4.swa_kv", 0, "swa_kv"),
        )
        self.assertIs(
            draft.get_swa_kv_buffer(0),
            self.arena.tensor("draft", "v4.swa_kv", 0, "swa_kv"),
        )
        self.assertIsNot(target.get_swa_kv_buffer(0), draft.get_swa_kv_buffer(0))
        self.assertEqual(
            target.get_swa_kv_buffer(0).shape[0],
            draft.get_swa_kv_buffer(0).shape[0],
        )

    def test_radix_constructor_keeps_independent_legacy_allocations(self):
        _torch.allocations.clear()
        pool = _cache.DeepseekV4TokenToKVPool(
            size=256,
            model_dtype=_torch.float32,
            layout=_layout(),
            layer_num=3,
            device="cuda:0",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=512,
            page_size=256,
            rank=0,
            hf_config=_hf_config(),
            max_scheduled_tokens=128,
        )

        self.assertIsNone(pool.flat_arena_set)
        self.assertIsNone(pool.arena_generation)
        self.assertGreater(len(_torch.allocations), 0)
        self.assertTrue(pool.get_swa_kv_buffer(0).all_zero)
        self.assertFalse(pool.get_compressor_state_buffer(1).all_zero)

    def test_flat_debug_metrics_use_pool_snapshots_not_legacy_group_allocators(self):
        target = _cache.DeepseekV4TokenToKVPool(
            size=self.plan.max_total_tokens,
            model_dtype=_torch.float32,
            layout=_layout(),
            layer_num=3,
            device="cuda:0",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=512,
            page_size=256,
            rank=0,
            hf_config=_hf_config(),
            max_scheduled_tokens=128,
            cache_owner="target",
            flat_arena_set=self.arena,
        )

        snapshots = [
            SimpleNamespace(
                pool_id=pool.pool_id,
                usable_blocks=pool.total_blocks - 1,
                free_blocks=pool.total_blocks - 2,
                active_blocks=1,
                cached_evictable_blocks=0,
                reserved_blocks=0,
            )
            for pool in self.plan.pools
        ]

        class _Scheduler:
            @staticmethod
            def flat_pool_snapshots():
                return snapshots

            @staticmethod
            def paged_cache_group_total_pages(group_id):
                raise AssertionError(f"legacy group allocator used for {group_id}")

        target.bind_paged_cache_scheduler(_Scheduler())
        with self.assertLogs(_cache.logger, level="DEBUG") as captured:
            target.maybe_log_paged_cache_group_pages()

        self.assertIn("flat cache state group pages", captured.output[0])


class TestV4RegistryFlatGate(unittest.TestCase):
    def test_registry_keeps_radix_profiler_and_gates_shared_arena_on_flat_build(self):
        tree = ast.parse(_REGISTRY_FILE.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_attn_components"
        )
        called_names = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("scheduler_ext_flat_kvcache", called_names)
        self.assertIn("profile_deepseek_v4_flat_memory_plan", called_names)
        self.assertIn("profile_deepseek_v4_max_num_pages", called_names)
        self.assertIn("V4FlatArenaSet", called_names)

        pool_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DeepseekV4TokenToKVPool"
        ]
        owners = {
            keyword.value.value
            for call in pool_calls
            for keyword in call.keywords
            if keyword.arg == "cache_owner" and isinstance(keyword.value, ast.Constant)
        }
        self.assertEqual(owners, {"target", "draft"})
        self.assertTrue(
            all(
                any(keyword.arg == "flat_arena_set" for keyword in call.keywords)
                for call in pool_calls
            )
        )

        plan_kwargs_assignment = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "deepseek_v4_flat_plan_kwargs"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "dict"
        )
        draft_graph_rows = next(
            keyword.value
            for keyword in plan_kwargs_assignment.value.keywords
            if keyword.arg == "draft_max_graph_bs"
        )
        target_graph_rows = next(
            keyword.value
            for keyword in plan_kwargs_assignment.value.keywords
            if keyword.arg == "target_max_graph_bs"
        )
        self.assertIsInstance(target_graph_rows, ast.Name)
        self.assertEqual(target_graph_rows.id, "flat_graph_batch_rows")
        draft_graph_source = ast.unparse(draft_graph_rows)
        self.assertIn("flat_graph_batch_rows", draft_graph_source)
        self.assertNotIn("draft_attn_config.max_graph_bs", draft_graph_source)

        graph_rows_resolver = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_resolve_v4_flat_graph_batch_rows"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_batch_sizes_to_capture"
                for node in ast.walk(graph_rows_resolver)
            )
        )

    def test_registry_agrees_final_plan_before_arena_allocation(self):
        tree = ast.parse(_REGISTRY_FILE.read_text())
        create_components = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "create_attn_components"
        )
        final_rebuild_line = max(
            node.lineno
            for node in ast.walk(create_components)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_deepseek_v4_flat_memory_plan"
        )
        agreement_line = next(
            node.lineno
            for node in ast.walk(create_components)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assert_v4_flat_plan_tp_agreement"
        )
        arena_line = next(
            node.lineno
            for node in ast.walk(create_components)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "V4FlatArenaSet"
        )
        self.assertLess(final_rebuild_line, agreement_line)
        self.assertLess(agreement_line, arena_line)

        agreement_helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_assert_v4_flat_plan_tp_agreement"
        )
        helper_source = ast.unparse(agreement_helper)
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_process_group"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "gloo"
                and isinstance(node.args[1], ast.Name)
                and node.args[1].id == "attn_tp_group"
                for node in ast.walk(agreement_helper)
            )
        )
        self.assertIn(
            "dist.all_gather_object(gathered_records, local_record, group=cpu_group)",
            helper_source,
        )
        self.assertIn("if len(attn_tp_group) <= 1", helper_source)


if __name__ == "__main__":
    unittest.main()
