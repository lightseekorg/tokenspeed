import pytest
import torch


def test_transfer_kv_direct_prefers_cpp_binding(monkeypatch):
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio

    calls = []

    def fake_direct(
        src_layer_ptrs,
        dst_layer_ptrs,
        src_indices,
        dst_indices,
        item_size,
        page_size,
    ):
        calls.append(
            (
                src_layer_ptrs,
                dst_layer_ptrs,
                src_indices,
                dst_indices,
                item_size,
                page_size,
            )
        )

    monkeypatch.setattr(kvcacheio, "_load_transfer_kv_direct_func", lambda: fake_direct)
    monkeypatch.setattr(kvcacheio, "_has_cuda_layer", lambda *_args: True)

    src = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dst = torch.zeros_like(src)
    indices = torch.tensor([0, 1], dtype=torch.int64)

    kvcacheio.transfer_kv_direct([src], [dst], indices, indices, page_size=1)

    assert len(calls) == 1
    (
        called_src_ptrs,
        called_dst_ptrs,
        called_src_indices,
        called_dst_indices,
        item_size,
        page_size,
    ) = calls[0]
    assert called_src_ptrs.tolist() == [src.data_ptr()]
    assert called_dst_ptrs.tolist() == [dst.data_ptr()]
    assert called_src_indices is indices
    assert called_dst_indices is indices
    assert item_size == src.stride(0) * src.element_size()
    assert page_size == 1
    assert torch.equal(dst, torch.zeros_like(dst))


def test_transfer_kv_direct_python_fallback(monkeypatch):
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio

    monkeypatch.setattr(kvcacheio, "_load_transfer_kv_direct_func", lambda: None)

    src = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    dst = torch.zeros_like(src)
    src_indices = torch.tensor([0, 2], dtype=torch.int64)
    dst_indices = torch.tensor([1, 2], dtype=torch.int64)

    kvcacheio.transfer_kv_direct([src], [dst], src_indices, dst_indices, page_size=1)

    expected = torch.zeros_like(src)
    expected[1] = src[0]
    expected[2] = src[2]
    assert torch.equal(dst, expected)


def test_transfer_kv_direct_h2d_scatter_threshold_fallback(monkeypatch):
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio

    monkeypatch.setattr(
        kvcacheio,
        "_load_transfer_kv_direct_scatter_h2d_func",
        lambda: None,
    )

    src = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dst = torch.zeros_like(src)
    indices = torch.tensor([0, 1], dtype=torch.int64)

    result = kvcacheio.transfer_kv_direct_h2d_scatter(
        [src],
        [dst],
        indices,
        indices,
        page_size=1,
        effective_copy_calls=16,
    )

    assert not result.used
    assert result.fallback_reason == "below_threshold"


def test_transfer_kv_direct_h2d_scatter_symbol_missing(monkeypatch):
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio

    monkeypatch.setattr(
        kvcacheio,
        "_load_transfer_kv_direct_scatter_h2d_func",
        lambda: None,
    )

    src = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dst = torch.zeros_like(src)
    indices = torch.tensor([0, 1], dtype=torch.int64)

    result = kvcacheio.transfer_kv_direct_h2d_scatter(
        [src],
        [dst],
        indices,
        indices,
        page_size=1,
        effective_copy_calls=4096,
    )

    assert not result.used
    assert result.fallback_reason == "symbol_missing"


def test_transfer_kv_direct_h2d_scatter_bucketed_pointer_tables(monkeypatch):
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio

    calls = []

    def fake_scatter(
        src_layer_ptrs,
        dst_layer_ptrs,
        src_indices,
        dst_indices,
        item_size,
        page_size,
    ):
        calls.append(
            (
                src_layer_ptrs,
                dst_layer_ptrs,
                src_indices,
                dst_indices,
                item_size,
                page_size,
            )
        )

    monkeypatch.setattr(
        kvcacheio,
        "_load_transfer_kv_direct_scatter_h2d_func",
        lambda: fake_scatter,
    )
    monkeypatch.setattr(
        kvcacheio,
        "_h2d_scatter_device",
        lambda *_args: (torch.device("cpu"), ""),
    )

    src_a = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    dst_a = torch.zeros_like(src_a)
    src_b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    dst_b = torch.zeros_like(src_b)
    src_indices = torch.tensor([0, 2], dtype=torch.int64)
    dst_indices = torch.tensor([1, 3], dtype=torch.int64)

    result = kvcacheio.transfer_kv_direct_h2d_scatter(
        [src_a, src_b],
        [dst_a, dst_b],
        src_indices,
        dst_indices,
        page_size=1,
        effective_copy_calls=4096,
    )

    assert result.used
    assert result.buckets == 2
    assert result.kernel_launches == 2
    assert [call[4] for call in calls] == [
        src_a.stride(0) * src_a.element_size(),
        src_b.stride(0) * src_b.element_size(),
    ]
    assert calls[0][0].tolist() == [src_a.data_ptr()]
    assert calls[0][1].tolist() == [dst_a.data_ptr()]
    assert calls[1][0].tolist() == [src_b.data_ptr()]
    assert calls[1][1].tolist() == [dst_b.data_ptr()]
    assert calls[0][2] is src_indices
    assert calls[0][3] is dst_indices
    assert calls[0][5] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_transfer_kv_direct_h2d_scatter_cuda_smoke():
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio

    if kvcacheio._load_transfer_kv_direct_scatter_h2d_func() is None:
        pytest.skip("scatter H2D kvcacheio symbol is not built")

    src = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    current_platform().register_host_tensor_for_gpu_access(src)
    dst = torch.zeros_like(src, device="cuda")
    src_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
    dst_indices = torch.tensor([1, 3, 5], dtype=torch.int64)

    result = kvcacheio.transfer_kv_direct_h2d_scatter(
        [src],
        [dst],
        src_indices,
        dst_indices,
        page_size=1,
        effective_copy_calls=4096,
    )

    assert result.used
    torch.cuda.synchronize()
    expected = torch.zeros_like(src)
    expected[1] = src[0]
    expected[3] = src[2]
    expected[5] = src[4]
    assert torch.equal(dst.cpu(), expected)
