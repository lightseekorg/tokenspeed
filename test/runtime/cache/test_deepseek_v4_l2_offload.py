from __future__ import annotations

from types import SimpleNamespace

import torch


def test_deepseek_v4_disable_kvstore_hides_host_groups_from_scheduler():
    from tokenspeed.runtime.engine.event_loop import (
        _paged_cache_host_group_pages_for_scheduler,
    )

    host_pool = SimpleNamespace(paged_cache_group_page_counts={"v4.swa_kv": 1})

    assert _paged_cache_host_group_pages_for_scheduler(False, host_pool) == {}
    assert _paged_cache_host_group_pages_for_scheduler(True, host_pool) == {
        "v4.swa_kv": 1
    }


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
    pool = DeepseekV4TokenToKVPool(
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
    return pool


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


def test_deepseek_v4_host_pool_reserves_dummy_page_zero():
    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    group_id = "v4.swa_kv"
    group_pages = host_pool.paged_cache_group_page_counts[group_id]

    assert host_pool.available_size(group_id) == group_pages - 1
    allocated = host_pool.alloc(group_id, min(2, group_pages - 1))
    assert allocated is not None
    assert 0 not in allocated.tolist()

    assert host_pool.free(group_id, torch.tensor([0], dtype=torch.int64)) == 0
    assert 0 not in host_pool.free_pages_by_group[group_id].tolist()


def test_deepseek_v4_host_pool_free_rejects_duplicates_and_out_of_range_pages():
    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    group_id = "v4.swa_kv"
    group_pages = host_pool.paged_cache_group_page_counts[group_id]

    allocated = host_pool.alloc(group_id, 1)
    assert allocated is not None
    page = int(allocated[0])
    available_after_alloc = host_pool.available_size(group_id)

    assert (
        host_pool.free(
            group_id,
            torch.tensor([page, page, 0, group_pages], dtype=torch.int64),
        )
        == 1
    )
    assert host_pool.available_size(group_id) == available_after_alloc + 1
    assert host_pool.free(group_id, torch.tensor(page, dtype=torch.int64)) == 0
    free_ids = host_pool.free_pages_by_group[group_id].tolist()
    assert free_ids.count(page) == 1
    assert group_pages not in free_ids


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


def test_deepseek_v4_descriptor_expansion_ties_ratio4_indexer_to_compressed_group():
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="direct")

    swa_refs = transfer_pool.tensor_refs_for_group("v4.swa_kv")
    assert [ref.layer_id for ref in swa_refs] == [0, 1, 2]

    c4_refs = transfer_pool.tensor_refs_for_group(
        "v4.c4a.compressed_kv",
        layer_idx=1,
    )
    assert len(c4_refs) == 2
    assert c4_refs[0].device_tensor is device_pool.compressed_kv_buffer[1]
    assert c4_refs[0].host_tensor is host_pool.compressed_kv_buffer[1]
    assert c4_refs[1].device_tensor is device_pool.indexer_kv_buffer[1]
    assert c4_refs[1].host_tensor is host_pool.indexer_kv_buffer[1]
    assert all(ref.page_bytes == ref.device_tensor[0].nbytes for ref in c4_refs)


def test_deepseek_v4_descriptor_expansion_maps_state_groups_to_state_buffers():
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="direct")

    c4_state_refs = transfer_pool.tensor_refs_for_group(
        "v4.c4a.compressor_state",
        layer_idx=1,
    )
    assert len(c4_state_refs) == 1
    assert c4_state_refs[0].device_tensor is device_pool.compressor_state_buffer[1]
    assert c4_state_refs[0].host_tensor is host_pool.compressor_state_buffer[1]

    c128_state_refs = transfer_pool.tensor_refs_for_group(
        "v4.c128a.compressor_state",
        layer_idx=2,
    )
    assert len(c128_state_refs) == 1
    assert c128_state_refs[0].device_tensor is device_pool.compressor_state_buffer[2]
    assert c128_state_refs[0].host_tensor is host_pool.compressor_state_buffer[2]

    indexer_state_refs = transfer_pool.tensor_refs_for_group(
        "v4.c4a.indexer_compressor_state",
        layer_idx=1,
    )
    assert len(indexer_state_refs) == 1
    assert indexer_state_refs[0].device_tensor is device_pool.indexer_state_buffer[1]
    assert indexer_state_refs[0].host_tensor is host_pool.indexer_state_buffer[1]


def test_deepseek_v4_paged_pool_dispatches_direct_page_copies(monkeypatch):
    from tokenspeed.runtime.cache.transfer import deepseek_v4_pool
    from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import DeepseekV4CachePool

    device_pool = _make_v4_pool()
    host_pool = _make_host_pool(device_pool)
    transfer_pool = DeepseekV4CachePool(device_pool, host_pool, io_backend="kernel")

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
    transfer = SimpleNamespace(
        group_id="v4.c4a.compressed_kv",
        src_pages=[3, 1, 3],
        dst_pages=[9, 7, 9],
    )

    transfer_pool.writeback_paged([transfer])

    assert len(calls) == 1
    src_layers, dst_layers, src_indices, dst_indices, page_size = calls[0]
    assert src_indices == [1, 3]
    assert dst_indices == [7, 9]
    assert page_size == 1
    assert src_layers[0] is device_pool.compressed_kv_buffer[1]
    assert src_layers[1] is device_pool.indexer_kv_buffer[1]
    assert dst_layers[0] is host_pool.compressed_kv_buffer[1]
    assert dst_layers[1] is host_pool.indexer_kv_buffer[1]
    c4_bytes = (
        device_pool.compressed_kv_buffer[1][0].nbytes
        + device_pool.indexer_kv_buffer[1][0].nbytes
    )
    assert transfer_pool.get_transfer_stats()["D2H"]["v4.c4a.compressed_kv"] == {
        "calls": 1,
        "pages": 2,
        "bytes": 2 * c4_bytes,
    }

    calls.clear()
    transfer_pool.loadback_paged([transfer], layer_idx=1)

    assert len(calls) == 1
    src_layers, dst_layers, src_indices, dst_indices, page_size = calls[0]
    assert src_layers[0] is host_pool.compressed_kv_buffer[1]
    assert src_layers[1] is host_pool.indexer_kv_buffer[1]
    assert dst_layers[0] is device_pool.compressed_kv_buffer[1]
    assert dst_layers[1] is device_pool.indexer_kv_buffer[1]
    assert src_indices == [1, 3]
    assert dst_indices == [7, 9]
    assert page_size == 1
    assert transfer_pool.get_transfer_stats()["H2D"]["v4.c4a.compressed_kv"] == {
        "calls": 1,
        "pages": 2,
        "bytes": 2 * c4_bytes,
    }
    transfer_pool.reset_transfer_stats()
    assert transfer_pool.get_transfer_stats() == {"D2H": {}, "H2D": {}}

    def failing_transfer_kv_direct(**kwargs):
        del kwargs
        raise RuntimeError("copy submission failed")

    monkeypatch.setattr(
        deepseek_v4_pool,
        "transfer_kv_direct",
        failing_transfer_kv_direct,
    )
    try:
        transfer_pool.writeback_paged([transfer])
    except RuntimeError as exc:
        assert "copy submission failed" in str(exc)
    else:
        raise AssertionError("expected transfer failure")
    assert transfer_pool.get_transfer_stats() == {"D2H": {}, "H2D": {}}


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


def test_scheduler_config_accepts_paged_cache_host_group_pages():
    from tokenspeed.runtime.engine.scheduler_utils import make_config

    cfg = make_config(
        num_device_pages=8,
        max_scheduled_tokens=4,
        max_batch_size=2,
        page_size=64,
        num_host_pages=0,
        disable_l2_cache=False,
        enable_l3_storage=False,
        prefetch_threshold=4,
        role="null",
        paged_cache_host_group_pages={"v4.swa_kv": 17},
    )

    assert cfg.paged_cache_host_group_pages == {"v4.swa_kv": 17}
