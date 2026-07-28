from __future__ import annotations

import importlib
from test.runtime.conftest import KIMI_GROUP_IDS as _KIMI_GROUP_IDS
from test.runtime.conftest import TP8_PHYSICAL_PAGE_BYTES as _KIMI_TP8_PAGE_BYTES
from test.runtime.conftest import kimi_tp8_plan as _kimi_tp8_plan
from test.runtime.conftest import make_synthetic_hybrid_plan as _make_plan
from test.runtime.conftest import reduced_kimi_plan as _reduced_kimi_tp8_plan

import pytest
import torch

import tokenspeed.runtime.configs.hybrid_cache_plan as hybrid_cache_plan


def flat_hybrid_module():
    return importlib.import_module(
        "tokenspeed.runtime.layers.attention.kv_cache.flat_hybrid"
    )


@pytest.fixture
def plan() -> hybrid_cache_plan.FlatHybridCachePlan:
    return _make_plan()


def test_plan_publication(plan: hybrid_cache_plan.FlatHybridCachePlan) -> None:
    specs = flat_hybrid_module().paged_cache_group_specs_from_plan(plan)

    assert tuple(spec.group_id for spec in specs) == tuple(
        group.group_id for group in plan.groups
    )
    assert all(spec.rows_per_page == plan.block_size for spec in specs)
    assert all(spec.entry_stride_tokens == 1 for spec in specs)
    assert all(spec.block_size == plan.block_size for spec in specs)
    assert tuple(spec.family for spec in specs) == tuple(
        group.family for group in plan.groups
    )
    assert tuple(spec.retention for spec in specs) == tuple(
        group.retention for group in plan.groups
    )
    assert tuple(spec.sliding_window_tokens for spec in specs) == tuple(
        group.sliding_window_tokens for group in plan.groups
    )


def test_sliding_history_publication_preserves_positive_window() -> None:
    plan = _make_plan(history_retention="sliding_window", sliding_window_tokens=8)

    specs = flat_hybrid_module().paged_cache_group_specs_from_plan(plan)

    history = next(spec for spec in specs if spec.family == "history")
    assert history.retention == "sliding_window"
    assert history.sliding_window_tokens == 8


def test_raw_slab_owners_match_the_plan(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
) -> None:
    module = flat_hybrid_module()
    pool = module.FlatHybridCachePool(plan=plan, device="cpu")
    slabs = (pool.raw_slab(0), pool.raw_slab(1))

    assert pool.plan is plan
    assert len(slabs) == len(plan.physical_slots) == 2
    assert all(slab.dtype == torch.uint8 for slab in slabs)
    assert all(
        slab.shape == (plan.usable_pages + 1, plan.physical_page_bytes)
        for slab in slabs
    )
    assert all(slab.device == torch.device("cpu") for slab in slabs)
    assert (
        slabs[0].untyped_storage().data_ptr() != slabs[1].untyped_storage().data_ptr()
    )
    assert all(torch.count_nonzero(slab).item() == 0 for slab in slabs)
    assert pool.raw_slab(0) is slabs[0]
    assert pool.raw_slab(1) is slabs[1]

    allocated = sum(slab.nbytes for slab in slabs)
    assert pool.allocated_bytes() == allocated
    assert allocated == plan.diagnostics.total_allocated_bytes


def test_lifecycle_publication_and_layer_lookups(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
) -> None:
    module = flat_hybrid_module()
    pool = module.FlatHybridCachePool(plan=plan, device=torch.device("cpu"))

    assert pool.paged_cache_group_specs == module.paged_cache_group_specs_from_plan(
        plan
    )
    assert pool.num_device_pages_with_null == plan.usable_pages + 1
    assert pool.supports_hierarchical_kv_cache is False
    assert pool.supports_disaggregation is True
    pd_layout, registrations = pool.get_flatkv_pd_contract()
    assert pd_layout.block_size == plan.block_size
    assert pd_layout.num_pages_with_null == plan.usable_pages + 1
    assert pd_layout.physical_page_bytes == plan.physical_page_bytes
    assert tuple(group.group_id for group in pd_layout.groups) == tuple(
        group.group_id for group in plan.groups
    )
    assert tuple(registration.buffer_id for registration in registrations) == (
        pd_layout.physical_buffer_ids
    )
    assert tuple(registration.base_addr for registration in registrations) == tuple(
        pool.raw_slab(slot).data_ptr() for slot in range(len(plan.physical_slots))
    )
    assert all(
        registration.length
        == pd_layout.num_pages_with_null * pd_layout.physical_page_bytes
        for registration in registrations
    )
    for binding in plan.layer_bindings:
        assert pool.group_id_for_layer(binding.layer_id) == binding.group_id
        assert pool.physical_slot_for_layer(binding.layer_id) == binding.physical_slot

    with pytest.raises(KeyError, match="layer_id 999"):
        pool.group_id_for_layer(999)
    with pytest.raises(KeyError, match="layer_id 999"):
        pool.physical_slot_for_layer(999)
    with pytest.raises(TypeError, match="layer_id"):
        pool.group_id_for_layer(True)
    with pytest.raises(TypeError, match="layer_id"):
        pool.physical_slot_for_layer(True)
    with pytest.raises(TypeError, match="physical_slot"):
        pool.raw_slab(True)
    with pytest.raises(IndexError, match="physical_slot"):
        pool.raw_slab(-1)
    with pytest.raises(IndexError, match="physical_slot"):
        pool.raw_slab(len(plan.physical_slots))


@pytest.mark.parametrize("clear_method", ["clear", "clear_kv_buffers"])
def test_clear_zeroes_raw_owners_in_place(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
    clear_method: str,
) -> None:
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cpu")
    slabs = tuple(pool.raw_slab(slot) for slot in range(len(plan.physical_slots)))
    pointers = tuple(slab.data_ptr() for slab in slabs)
    storage_pointers = tuple(slab.untyped_storage().data_ptr() for slab in slabs)
    for value, slab in enumerate(slabs, start=1):
        slab.fill_(value)

    getattr(pool, clear_method)()

    assert tuple(slab.data_ptr() for slab in slabs) == pointers
    assert (
        tuple(slab.untyped_storage().data_ptr() for slab in slabs) == storage_pointers
    )
    assert all(torch.count_nonzero(slab).item() == 0 for slab in slabs)


def test_zero_pages_clears_selected_page_across_every_raw_owner(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
) -> None:
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cpu")
    slabs = tuple(pool.raw_slab(slot) for slot in range(len(plan.physical_slots)))
    for value, slab in enumerate(slabs, start=1):
        slab.fill_(value)

    pool.zero_pages([3, 1, 3])

    for value, slab in enumerate(slabs, start=1):
        assert torch.count_nonzero(slab[1]).item() == 0
        assert torch.count_nonzero(slab[3]).item() == 0
        assert torch.all(slab[0] == value)
        assert torch.all(slab[2] == value)


def test_raw_allocation_occurs_inside_kv_cache_memory_saver_region(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = flat_hybrid_module()
    real_zeros = torch.zeros
    events: list[object] = []
    active = False

    class Region:
        def __enter__(self) -> None:
            nonlocal active
            active = True
            events.append("enter")

        def __exit__(self, *exc_info: object) -> None:
            nonlocal active
            active = False
            events.append("exit")

    class Adapter:
        def region(
            self, tag: str | None = None, enable_cpu_backup: bool = False
        ) -> Region:
            events.append((tag, enable_cpu_backup))
            return Region()

    adapter = Adapter()
    monkeypatch.setattr(
        module.TorchMemorySaverAdapter,
        "create",
        lambda *, enable: adapter,
    )

    def checked_zeros(*args: object, **kwargs: object) -> torch.Tensor:
        assert active
        events.append("zeros")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(module.torch, "zeros", checked_zeros)

    module.FlatHybridCachePool(plan=plan, device="cpu", enable_memory_saver=True)

    assert events == [("kv_cache", False), "enter", "zeros", "zeros", "exit"]


def _component_binding(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
    layer_id: int,
    component_name: str,
) -> hybrid_cache_plan.ComponentBinding:
    binding = next(
        binding for binding in plan.layer_bindings if binding.layer_id == layer_id
    )
    return next(
        component
        for component in binding.components
        if component.name == component_name
    )


def test_component_views_have_exact_shape_stride_and_storage_offsets(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
) -> None:
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cpu")
    history = pool.get_component(0, "kv")
    other_history = pool.get_component(1, "kv")
    conv = pool.get_component(2, "conv_state")
    recurrent = pool.get_component(2, "recurrent_state")

    assert history.dtype == torch.float8_e4m3fn
    assert history.shape == (4, 4, 1, 4)
    assert history.stride() == (16, 4, 4, 1)
    assert conv.dtype == torch.bfloat16
    assert conv.shape == (4, 2)
    assert conv.stride() == (8, 1)
    assert recurrent.dtype == torch.float32
    assert recurrent.shape == (4, 2)
    assert recurrent.stride() == (4, 1)

    for layer_id, component_name, view in (
        (0, "kv", history),
        (1, "kv", other_history),
        (2, "conv_state", conv),
        (2, "recurrent_state", recurrent),
    ):
        binding = _component_binding(plan, layer_id, component_name)
        raw = pool.raw_slab(pool.physical_slot_for_layer(layer_id))
        assert view.data_ptr() == raw.data_ptr() + binding.byte_offset
        assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert view.untyped_storage().nbytes() == raw.untyped_storage().nbytes()

    slot_zero_pointer = pool.raw_slab(0).untyped_storage().data_ptr()
    assert history.untyped_storage().data_ptr() == slot_zero_pointer
    assert conv.untyped_storage().data_ptr() == slot_zero_pointer
    assert recurrent.untyped_storage().data_ptr() == slot_zero_pointer
    assert other_history.untyped_storage().data_ptr() != slot_zero_pointer


def test_typed_component_writes_alias_only_planned_raw_regions(
    plan: hybrid_cache_plan.FlatHybridCachePlan,
) -> None:
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cpu")
    history = pool.get_component(0, "kv")
    other_slot_history = pool.get_component(1, "kv")
    conv = pool.get_component(2, "conv_state")
    recurrent = pool.get_component(2, "recurrent_state")
    raw = pool.raw_slab(0)

    conv[1].copy_(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    conv_bytes = conv[1].view(torch.uint8)
    assert torch.equal(raw[1, :4], conv_bytes)
    assert torch.equal(history[1].view(torch.uint8).reshape(-1)[:4], conv_bytes)
    assert torch.count_nonzero(raw[1, 4:]).item() == 0
    assert torch.count_nonzero(recurrent[1]).item() == 0

    recurrent[1].copy_(torch.tensor([3.5, -4.25], dtype=torch.float32))
    recurrent_bytes = recurrent[1].view(torch.uint8)
    assert torch.equal(raw[1, 4:12], recurrent_bytes)
    assert torch.equal(history[1].view(torch.uint8).reshape(-1)[4:12], recurrent_bytes)
    assert torch.equal(raw[1, :4], conv_bytes)
    assert torch.count_nonzero(raw[1, 12:]).item() == 0
    assert torch.count_nonzero(other_slot_history.view(torch.uint8)).item() == 0


def test_kimi_tp8_reference_geometry_and_publication_are_exact() -> None:
    plan = _kimi_tp8_plan()

    assert plan.block_size == 1_536
    assert plan.physical_page_bytes == _KIMI_TP8_PAGE_BYTES
    assert plan.usable_pages + plan.diagnostics.null_pages == 8
    assert tuple(group.group_id for group in plan.groups) == _KIMI_GROUP_IDS
    specs = flat_hybrid_module().paged_cache_group_specs_from_plan(plan)
    assert tuple(spec.group_id for spec in specs) == _KIMI_GROUP_IDS
    assert all(spec.block_size == 1_536 for spec in specs)
    assert all(spec.rows_per_page == 1_536 for spec in specs)


def test_reduced_kimi_tp8_pool_materializes_complete_topology() -> None:
    plan = _reduced_kimi_tp8_plan()
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cpu")
    slabs = tuple(pool.raw_slab(slot) for slot in range(24))

    assert len(plan.physical_slots) == len(slabs) == 24
    assert len({slab.untyped_storage().data_ptr() for slab in slabs}) == 24
    assert all(
        slab.shape == (2, _KIMI_TP8_PAGE_BYTES) and slab.dtype == torch.uint8
        for slab in slabs
    )
    assert len(plan.layer_bindings) == 93
    assert tuple(group.group_id for group in plan.groups) == _KIMI_GROUP_IDS
    assert plan.diagnostics.padding_binding_count == 3
    assert (
        sum(
            layer_id is None
            for group in plan.groups
            for layer_id in group.slot_layer_ids
        )
        == 3
    )
    assert pool.num_device_pages_with_null == 2
    assert pool.allocated_bytes() == plan.diagnostics.total_allocated_bytes
    assert pool.allocated_bytes() == sum(slab.nbytes for slab in slabs)
    assert tuple(spec.group_id for spec in pool.paged_cache_group_specs) == (
        _KIMI_GROUP_IDS
    )
    with pytest.raises(IndexError, match="physical_slot"):
        pool.raw_slab(24)

    for binding in plan.layer_bindings:
        assert pool.group_id_for_layer(binding.layer_id) == binding.group_id
        assert pool.physical_slot_for_layer(binding.layer_id) == binding.physical_slot
        expected_names = (
            ("latent_kv",)
            if binding.group_id == "full_attention"
            else ("conv_state", "recurrent_state")
        )
        assert (
            tuple(component.name for component in binding.components) == expected_names
        )
        for component in binding.components:
            view = pool.get_component(binding.layer_id, component.name)
            raw = pool.raw_slab(binding.physical_slot)
            assert view.dtype == component.dtype
            assert view.data_ptr() == raw.data_ptr() + component.byte_offset
            assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_component_views_share_planned_storage() -> None:
    plan = _make_plan()
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cuda")
    raw_zero = pool.raw_slab(0)
    raw_one = pool.raw_slab(1)
    history = pool.get_component(0, "kv")
    other_history = pool.get_component(1, "kv")
    conv = pool.get_component(2, "conv_state")
    recurrent = pool.get_component(2, "recurrent_state")

    assert raw_zero.is_cuda and raw_one.is_cuda
    assert raw_zero.device.type == raw_one.device.type == "cuda"
    assert raw_zero.untyped_storage().data_ptr() != raw_one.untyped_storage().data_ptr()
    assert pool.allocated_bytes() == plan.diagnostics.total_allocated_bytes
    assert pool.allocated_bytes() == raw_zero.nbytes + raw_one.nbytes
    expected_owner_bytes = (plan.usable_pages + 1) * plan.physical_page_bytes
    assert raw_zero.nbytes == raw_one.nbytes == expected_owner_bytes

    for layer_id, component_name, view in (
        (0, "kv", history),
        (1, "kv", other_history),
        (2, "conv_state", conv),
        (2, "recurrent_state", recurrent),
    ):
        component = _component_binding(plan, layer_id, component_name)
        raw = pool.raw_slab(pool.physical_slot_for_layer(layer_id))
        assert view.is_cuda
        assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert view.untyped_storage().nbytes() == raw.untyped_storage().nbytes()
        assert view.data_ptr() == raw.data_ptr() + component.byte_offset
        assert view.data_ptr() % component.dtype.itemsize == 0
        assert view.stride(0) * component.dtype.itemsize == plan.physical_page_bytes

    conv_values = torch.tensor([1.5, -2.0], dtype=torch.bfloat16, device="cuda")
    conv[1].copy_(conv_values)
    conv_bytes = conv_values.view(torch.uint8)
    assert torch.equal(raw_zero[1, :4], conv_bytes)
    assert torch.equal(history[1].view(torch.uint8).reshape(-1)[:4], conv_bytes)
    assert torch.count_nonzero(raw_one).item() == 0

    recurrent_values = torch.tensor([3.25, -4.5], dtype=torch.float32, device="cuda")
    raw_zero[2, 4:12].copy_(recurrent_values.view(torch.uint8))
    assert torch.equal(recurrent[2], recurrent_values)
    assert torch.equal(
        history[2].view(torch.uint8).reshape(-1)[4:12],
        recurrent_values.view(torch.uint8),
    )
    assert torch.count_nonzero(other_history.view(torch.uint8)).item() == 0

    history[0].fill_(1)
    raw_one[0].fill_(0xA5)
    assert torch.count_nonzero(raw_zero[0]).item() > 0
    assert torch.count_nonzero(raw_one[0]).item() > 0
    pool.clear()
    assert torch.count_nonzero(raw_zero).item() == 0
    assert torch.count_nonzero(raw_one).item() == 0
    assert torch.count_nonzero(history.view(torch.uint8)).item() == 0
    assert torch.count_nonzero(other_history.view(torch.uint8)).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_zero_pages_clears_all_slabs_without_touching_neighbors() -> None:
    plan = _make_plan()
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cuda")
    slabs = tuple(pool.raw_slab(slot) for slot in range(len(plan.physical_slots)))
    for value, slab in enumerate(slabs, start=1):
        slab.fill_(value)

    pool.zero_pages([3, 1])
    torch.cuda.synchronize()

    for value, slab in enumerate(slabs, start=1):
        assert torch.count_nonzero(slab[1]).item() == 0
        assert torch.count_nonzero(slab[3]).item() == 0
        assert torch.all(slab[0] == value)
        assert torch.all(slab[2] == value)


def test_runtime_contract_is_the_only_scheduler_publication(plan) -> None:
    pool = flat_hybrid_module().FlatHybridCachePool(plan=plan, device="cpu")
    contract = pool.runtime_contract
    group_ids = tuple(group.group_id for group in plan.groups)

    assert contract.block_size == plan.block_size
    assert contract.usable_pages == plan.usable_pages
    assert contract.num_device_pages_with_null == plan.usable_pages + 1
    assert contract.token_capacity == plan.usable_pages * plan.block_size
    assert contract.group_specs == pool.paged_cache_group_specs
    assert tuple(contract.group_page_counts) == group_ids
    assert set(contract.group_page_counts.values()) == {plan.usable_pages + 1}
    assert pool.page_size == plan.block_size
    assert pool.size == contract.token_capacity
    assert pool.num_usable_pages == plan.usable_pages
    assert pool.paged_cache_group_page_counts == contract.group_page_counts
    assert pool.prefix_cache_required_group_ids is None
