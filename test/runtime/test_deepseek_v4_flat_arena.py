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

import contextlib
import importlib.util
import logging
import math
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

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


with mock.patch.dict(sys.modules):
    _torch = _install_stubs()
    _contract = _load(
        "tokenspeed.runtime.configs.flat_kv_contract",
        _CONFIGS_DIR / "flat_kv_contract.py",
    )
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


def _with_isolated_utils(function):
    def wrapped(*args, **kwargs):
        utils = types.ModuleType("tokenspeed.runtime.utils")
        utils.__path__ = []
        common = types.ModuleType("tokenspeed.runtime.utils.common")
        common.ceil_div = lambda numer, denom: -(-numer // denom)
        with mock.patch.dict(
            sys.modules,
            {
                utils.__name__: utils,
                common.__name__: common,
            },
        ):
            return function(*args, **kwargs)

    return wrapped


for _helper_name in (
    "compute_paged_cache_group_page_counts",
    "compute_flat_capture_cols",
    "compute_flat_export_cols",
):
    setattr(_cache, _helper_name, _with_isolated_utils(getattr(_cache, _helper_name)))


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


if __name__ == "__main__":
    unittest.main()
