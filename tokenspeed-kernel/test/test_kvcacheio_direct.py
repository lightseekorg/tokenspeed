import pytest
import torch


class _FakePointerPlatform:
    def __init__(self, lookups=None):
        self.lookups = lookups

    def device_visible_data_ptr(self, tensor):
        if self.lookups is not None:
            self.lookups.append(tensor)
        return tensor.data_ptr()


@pytest.fixture
def kvcacheio():
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio

    return kvcacheio


def _capture(calls):
    return lambda *args: calls.append(args)


def test_transfer_kv_direct_prefers_cpp_binding(monkeypatch, kvcacheio):
    calls = []
    monkeypatch.setattr(
        kvcacheio, "_load_transfer_kv_direct_func", lambda: _capture(calls)
    )
    monkeypatch.setattr(kvcacheio, "_has_cuda_layer", lambda *_args: True)
    monkeypatch.setattr(kvcacheio, "current_platform", _FakePointerPlatform)

    src = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dst = torch.zeros_like(src)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    kvcacheio.transfer_kv_direct([src], [dst], indices, indices, page_size=1)

    (call,) = calls
    assert [ptrs.tolist() for ptrs in call[:2]] == [
        [src.data_ptr()],
        [dst.data_ptr()],
    ]
    assert call[2] is call[3] is indices
    assert call[4:] == (src.stride(0) * src.element_size(), 1)
    assert torch.equal(dst, torch.zeros_like(dst))


def test_transfer_kv_direct_python_fallback(monkeypatch, kvcacheio):
    monkeypatch.setattr(kvcacheio, "_load_transfer_kv_direct_func", lambda: None)
    src = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    dst = torch.zeros_like(src)
    src_indices = torch.tensor([0, 2], dtype=torch.int64)
    dst_indices = torch.tensor([1, 2], dtype=torch.int64)

    kvcacheio.transfer_kv_direct([src], [dst], src_indices, dst_indices, page_size=1)

    expected = torch.zeros_like(src)
    expected[dst_indices] = src[src_indices]
    assert torch.equal(dst, expected)


def test_prepared_h2d_scatter_reuses_bucketed_pointer_tables(monkeypatch, kvcacheio):
    calls, pointer_lookups = [], []
    monkeypatch.setattr(
        kvcacheio,
        "_load_transfer_kv_direct_scatter_h2d_func",
        lambda: _capture(calls),
    )
    monkeypatch.setattr(
        kvcacheio, "_h2d_scatter_device", lambda *_: (torch.device("cpu"), "")
    )
    monkeypatch.setattr(
        kvcacheio, "current_platform", lambda: _FakePointerPlatform(pointer_lookups)
    )

    src = [
        torch.arange(4 * width, dtype=torch.float32).reshape(4, width)
        for width in (2, 3)
    ]
    src += [tensor.clone() for tensor in src]
    dst = [torch.zeros_like(tensor) for tensor in src]
    src_indices = torch.tensor([0, 2], dtype=torch.int64)
    dst_indices = torch.tensor([1, 3], dtype=torch.int64)

    plan, reason = kvcacheio.prepare_kv_direct_h2d_scatter_plan(src, dst, [0, 0, 1, 1])
    assert plan is not None and reason == ""
    assert len(pointer_lookups) == 8

    def run(entry):
        return kvcacheio.transfer_kv_direct_h2d_scatter_prepared(
            plan,
            src_indices,
            dst_indices,
            entry_begin=entry,
            entry_end=entry + 1,
        )

    results = [run(entry) for entry in range(2)]
    assert all(result.used and result.kernel_launches == 2 for result in results)
    assert len(pointer_lookups) == 8
    assert [call[4] for call in calls] == [
        tensor.stride(0) * tensor.element_size() for tensor in src
    ]
    assert [call[0].item() for call in calls] == [tensor.data_ptr() for tensor in src]
    assert [call[1].item() for call in calls] == [tensor.data_ptr() for tensor in dst]
    assert all(call[2] is src_indices and call[3] is dst_indices for call in calls)

    monkeypatch.setattr(kvcacheio, "_current_stream_id", lambda _device: 1)
    mismatch = run(0)
    assert not mismatch.used
    assert mismatch.fallback_reason == "stream_mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_prepared_h2d_scatter_cuda_smoke(kvcacheio):
    if kvcacheio._load_transfer_kv_direct_scatter_h2d_func() is None:
        pytest.skip("scatter H2D kvcacheio symbol is not built")

    src = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    kvcacheio.current_platform().register_host_tensor_for_gpu_access(src)
    dst = torch.zeros_like(src, device="cuda")
    src_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
    dst_indices = torch.tensor([1, 3, 5], dtype=torch.int64)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        plan, reason = kvcacheio.prepare_kv_direct_h2d_scatter_plan([src], [dst], [0])
        assert plan is not None and reason == ""
        result = kvcacheio.transfer_kv_direct_h2d_scatter_prepared(
            plan,
            src_indices,
            dst_indices,
            entry_begin=0,
            entry_end=1,
        )

    assert result.used
    stream.synchronize()
    expected = torch.zeros_like(src)
    expected[dst_indices] = src[src_indices]
    assert torch.equal(dst.cpu(), expected)
