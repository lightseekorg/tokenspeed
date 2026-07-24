from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("enable_kvstore", "paged_cache_groups", "draft_pool", "should_raise"),
    [
        (True, [object()], object(), True),
        (False, [object()], object(), False),
        (True, [object()], None, False),
        (True, [], object(), False),
    ],
)
def test_grouped_kvstore_rejects_only_paged_draft_pool(
    enable_kvstore, paged_cache_groups, draft_pool, should_raise
):
    from tokenspeed.runtime.engine.event_loop import (
        _validate_grouped_kvstore_draft_pool,
    )

    if should_raise:
        with pytest.raises(NotImplementedError, match="--disable-kvstore"):
            _validate_grouped_kvstore_draft_pool(
                enable_kvstore, paged_cache_groups, draft_pool
            )
    else:
        _validate_grouped_kvstore_draft_pool(
            enable_kvstore, paged_cache_groups, draft_pool
        )


def test_deepseek_v4_scheduler_host_pages_follow_kvstore_mode():
    from tokenspeed.runtime.engine.event_loop import (
        _paged_cache_host_group_pages_for_scheduler,
    )
    from tokenspeed.runtime.engine.scheduler_utils import make_config

    host_pool = SimpleNamespace(paged_cache_group_page_counts={"v4.swa_kv": 1})

    hidden_host_groups = _paged_cache_host_group_pages_for_scheduler(False, host_pool)
    visible_host_groups = _paged_cache_host_group_pages_for_scheduler(True, host_pool)
    assert hidden_host_groups == {}
    assert visible_host_groups == {"v4.swa_kv": 1}

    common_config = dict(
        num_device_pages=8,
        max_scheduled_tokens=4,
        max_batch_size=2,
        page_size=64,
        enable_l3_storage=False,
        prefetch_threshold=4,
        role="null",
    )
    disabled_config = make_config(
        **common_config,
        num_host_pages=32,
        disable_l2_cache=True,
        paged_cache_host_group_pages=hidden_host_groups,
    )
    enabled_config = make_config(
        **common_config,
        num_host_pages=0,
        disable_l2_cache=False,
        paged_cache_host_group_pages=visible_host_groups,
    )

    assert disabled_config.disable_l2_cache
    assert disabled_config.num_host_pages == 32
    assert disabled_config.paged_cache_host_group_pages == {}
    assert enabled_config.paged_cache_host_group_pages == {"v4.swa_kv": 1}


def _make_v4_pool():
    from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
        DeepseekV4TokenToKVPool,
        deepseek_v4_cache_layout_from_config,
    )

    hf_config = SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )
    layout = deepseek_v4_cache_layout_from_config(
        hf_config,
        page_size=64,
        use_fp4_indexer_cache=True,
    )
    return DeepseekV4TokenToKVPool(
        size=512,
        model_dtype=torch.bfloat16,
        layout=layout,
        layer_num=3,
        device="cpu",
        enable_memory_saver=False,
        max_batch_size=2,
        max_context_len=512,
        page_size=64,
        rank=0,
        hf_config=hf_config,
        max_scheduled_tokens=64,
    )


def _make_host_pool(device_pool, ratio: float = 1.5):
    from tokenspeed.runtime.cache.deepseek_v4_cache_host import (
        DeepseekV4TokenToKVPoolHost,
    )

    return DeepseekV4TokenToKVPoolHost(
        device_pool,
        host_to_device_ratio=ratio,
        host_size_gb=0,
        register_host=False,
    )


def _make_transfer_pool(io_backend: str = "kernel"):
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    return (
        device_pool,
        host_pool,
        DeepseekV4CachePool(device_pool, host_pool, io_backend=io_backend),
    )


def _make_ratio4_paged_transfers():
    return [
        SimpleNamespace(
            group_id="v4.c4a.compressed_kv",
            src_pages=[3, 1, 3],
            dst_pages=[9, 7, 9],
        ),
        SimpleNamespace(
            group_id="v4.c4a.compressed_kv",
            src_pages=[2, 1],
            dst_pages=[8, 7],
        ),
    ]


def _patch_direct_transfer_recorder(monkeypatch):
    from tokenspeed.runtime.cache.transfer import deepseek_v4_pool

    calls = []

    def fake_transfer_kv_direct(
        *,
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        page_size,
    ):
        calls.append(
            (
                src_layers,
                dst_layers,
                src_indices.tolist(),
                dst_indices.tolist(),
                page_size,
            )
        )

    monkeypatch.setattr(
        deepseek_v4_pool,
        "transfer_kv_direct",
        fake_transfer_kv_direct,
    )
    return calls


def _assert_ratio4_direct_call(call, src_tensors, dst_tensors):
    src_layers, dst_layers, src_indices, dst_indices, page_size = call
    assert (src_indices, dst_indices, page_size) == ([1, 2, 3], [7, 8, 9], 1)
    assert all(src_layers[i] is expected for i, expected in enumerate(src_tensors))
    assert all(dst_layers[i] is expected for i, expected in enumerate(dst_tensors))


def test_deepseek_v4_host_pool_shapes_and_group_counts():
    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool, ratio=1.5)

    for group_id, device_pages in device_pool.paged_cache_group_page_counts.items():
        assert host_pool.paged_cache_group_page_counts[group_id] >= device_pages

    assert (
        host_pool.swa_kv_buffer[0].shape[1:] == device_pool.swa_kv_buffer[0].shape[1:]
    )
    assert host_pool.compressed_kv_buffer[1].shape[1:] == (
        device_pool.compressed_kv_buffer[1].shape[1:]
    )
    assert host_pool.indexer_kv_buffer[1].shape[0] == (
        host_pool.compressed_kv_buffer[1].shape[0]
    )
    assert host_pool.indexer_state_buffer[1].shape[1:] == (
        device_pool.indexer_state_buffer[1].shape[1:]
    )
    assert host_pool.page_num > 0
    assert host_pool.total_bytes > 0


def test_deepseek_v4_host_group_page_sizing_keeps_one_usable_page_per_group():
    from tokenspeed.runtime.cache.deepseek_v4_cache_host import (
        _allocate_host_group_pages,
    )

    ratio_counts = _allocate_host_group_pages(
        device_counts={"a": 3, "b": 1},
        page_bytes={"a": 100, "b": 200},
        host_ratio=0.1,
        host_size_gb=0,
    )
    assert ratio_counts == {"a": 2, "b": 2}

    size_counts = _allocate_host_group_pages(
        device_counts={"a": 3, "b": 1},
        page_bytes={"a": 100, "b": 200},
        host_ratio=1.0,
        host_size_gb=1,
    )
    assert size_counts["a"] >= 2
    assert size_counts["b"] >= 2

    capped_counts = _allocate_host_group_pages(
        device_counts={"a": 100, "b": 100},
        page_bytes={"a": 10, "b": 30},
        host_ratio=10.0,
        host_size_gb=0,
        host_budget_bytes=120,
    )
    assert capped_counts["a"] >= 2
    assert capped_counts["b"] >= 2
    assert (
        sum(
            capped_counts[group_id] * {"a": 10, "b": 30}[group_id]
            for group_id in capped_counts
        )
        <= 120
    )

    with pytest.raises(ValueError, match="too small"):
        _allocate_host_group_pages(
            device_counts={"a": 100, "b": 100},
            page_bytes={"a": 10, "b": 30},
            host_ratio=10.0,
            host_size_gb=0,
            host_budget_bytes=79,
        )


def test_deepseek_v4_shadow_capacity_uses_complete_history_limit():
    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    host_pool.paged_cache_group_page_counts.update(
        {
            "v4.c4a.compressed_kv": 2,
            "v4.c128a.compressed_kv": 32,
        }
    )

    expected_usable_pages = min(
        (
            (host_pool.paged_cache_group_page_counts[spec.group_id] - 1)
            * spec.rows_per_page
            * spec.entry_stride_tokens
            + device_pool.page_size
            - 1
        )
        // device_pool.page_size
        for spec in device_pool.paged_cache_group_specs
        if spec.family == "history"
    )

    assert host_pool._compute_shadow_page_num(device_pool) == expected_usable_pages + 1


def test_deepseek_v4_descriptor_expansion_maps_paged_groups():
    device_pool, host_pool, transfer_pool = _make_transfer_pool("direct")

    checks = [
        (
            "v4.swa_kv",
            None,
            [("swa_kv_buffer", 0), ("swa_kv_buffer", 1), ("swa_kv_buffer", 2)],
        ),
        (
            "v4.c4a.compressed_kv",
            1,
            [("compressed_kv_buffer", 1), ("indexer_kv_buffer", 1)],
        ),
        ("v4.c4a.compressor_state", 1, [("compressor_state_buffer", 1)]),
        ("v4.c128a.compressor_state", 2, [("compressor_state_buffer", 2)]),
        ("v4.c4a.indexer_compressor_state", 1, [("indexer_state_buffer", 1)]),
    ]
    for group_id, layer_idx, expected in checks:
        refs = transfer_pool.tensor_refs_for_group(group_id, layer_idx=layer_idx)
        assert len(refs) == len(expected)
        for ref, (buffer_name, layer_id) in zip(refs, expected):
            assert ref.layer_id == layer_id
            assert ref.device_tensor is getattr(device_pool, buffer_name)[layer_id]
            assert ref.host_tensor is getattr(host_pool, buffer_name)[layer_id]
            assert ref.page_bytes == ref.device_tensor[0].nbytes


def test_deepseek_v4_paged_pool_prepares_coalesced_transfers():
    _, _, transfer_pool = _make_transfer_pool()
    transfers = _make_ratio4_paged_transfers()

    prepared = transfer_pool.prepare_paged_transfers(transfers)

    assert len(prepared) == 1
    assert prepared[0].page_count == 3
    assert prepared[0].span_count == 1
    assert prepared[0].src_indices.tolist() == [1, 2, 3]
    assert prepared[0].dst_indices.tolist() == [7, 8, 9]


@pytest.mark.parametrize("host_to_device", [False, True], ids=["writeback", "loadback"])
def test_deepseek_v4_paged_pool_prepared_maps_group_tensors(
    monkeypatch, host_to_device
):
    device_pool, host_pool, transfer_pool = _make_transfer_pool()
    calls = _patch_direct_transfer_recorder(monkeypatch)
    prepared = transfer_pool.prepare_paged_transfers(_make_ratio4_paged_transfers())
    if host_to_device:
        transfer_pool.loadback_prepared_paged(prepared, layer_idx=1)
        src_pool, dst_pool = host_pool, device_pool
    else:
        transfer_pool.writeback_prepared_paged(prepared)
        src_pool, dst_pool = device_pool, host_pool

    assert len(calls) == 1
    _assert_ratio4_direct_call(
        calls[-1],
        (src_pool.compressed_kv_buffer[1], src_pool.indexer_kv_buffer[1]),
        (dst_pool.compressed_kv_buffer[1], dst_pool.indexer_kv_buffer[1]),
    )


def test_deepseek_v4_paged_pool_reuses_and_invalidates_h2d_scatter_plan(
    monkeypatch,
):
    from tokenspeed.runtime.cache.transfer import deepseek_v4_pool

    device_pool, host_pool, transfer_pool = _make_transfer_pool()
    monkeypatch.setattr(deepseek_v4_pool, "_H2D_SCATTER_MIN_COPY_CALLS", 0)
    prepared = transfer_pool.prepare_paged_transfers(_make_ratio4_paged_transfers())
    prepared_plans = []
    launches = []

    def prepare_plan(src_layers, dst_layers, entry_ids):
        plan = object()
        prepared_plans.append((plan, src_layers, dst_layers, entry_ids))
        return plan, ""

    def launch(plan, src_indices, dst_indices, entry_begin, entry_end):
        launches.append((plan, src_indices, dst_indices, entry_begin, entry_end))
        return SimpleNamespace(
            used=True,
            buckets=1,
            kernel_launches=1,
            fallback_reason="",
        )

    monkeypatch.setattr(
        deepseek_v4_pool,
        "prepare_kv_direct_h2d_scatter_plan",
        prepare_plan,
    )
    monkeypatch.setattr(
        deepseek_v4_pool,
        "transfer_kv_direct_h2d_scatter_prepared",
        launch,
    )

    transfer_pool.loadback_prepared_paged(prepared, layer_idx=1)
    transfer_pool.loadback_prepared_paged(prepared, layer_idx=1)

    assert len(prepared_plans) == 1
    assert len(launches) == 2
    assert launches[0][0] is launches[1][0] is prepared_plans[0][0]
    assert launches[0][1] is launches[1][1]
    assert launches[0][2] is launches[1][2]

    host_pool.compressed_kv_buffer[1] = host_pool.compressed_kv_buffer[1].clone()
    transfer_pool.loadback_prepared_paged(prepared, layer_idx=1)

    assert len(prepared_plans) == 2
    assert launches[-1][0] is prepared_plans[-1][0]


def test_deepseek_v4_layer_getters_wait_on_registered_counter():
    device_pool = _make_v4_pool()
    waits = []

    class FakeCounter:
        def wait_until(self, layer_id: int):
            waits.append(layer_id)

    device_pool.register_layer_transfer_counter(FakeCounter())

    device_pool.get_swa_kv_buffer(0)
    device_pool.get_compressed_kv_buffer_2d(1)
    device_pool.get_compressor_state_buffer(1)
    device_pool.get_indexer_kv_buffer_2d(1)
    device_pool.get_indexer_state_buffer(1)
    device_pool.get_kv_buffer(2)

    assert waits == [0, 1, 1, 1, 1, 2]
