from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.host_transfer import (
    _triton_is_unavailable,
    transfer_cache_ranges,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_workspace_load_ranges_empty_is_noop():
    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    workspace = HostTransferWorkspace()
    assert workspace.load_ranges(()) == (0, 0)
    assert workspace._range_host is None


def test_workspace_reuses_host_range_storage():
    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    workspace = HostTransferWorkspace()
    first = ((0, 0, 0, 8), (1, 16, 32, 24))
    count, max_bytes = workspace.load_ranges(first)
    assert count == 2
    assert max_bytes == 24
    host_ptr = workspace._range_host.data_ptr()
    count, max_bytes = workspace.load_ranges(((0, 8, 8, 16),))
    assert count == 1
    assert max_bytes == 16
    assert workspace._range_host.data_ptr() == host_ptr


def test_workspace_load_range_batches_flattens_rows_once():
    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    workspace = HostTransferWorkspace()
    host = MagicMock()
    host_rows = MagicMock()
    host.__getitem__.return_value = host_rows
    workspace.ensure_range_host = MagicMock(return_value=host)
    batches = (
        ((0, 0, 32, 8), (1, 16, 64, 24)),
        (),
        ((0, 8, 96, 16),),
    )

    descriptors = workspace.load_range_batches(batches)

    assert descriptors == ((0, 2, 24), (2, 0, 0), (2, 1, 16))
    workspace.ensure_range_host.assert_called_once_with(3)
    host.__getitem__.assert_called_once_with(slice(None, 3))
    copied = host_rows.copy_.call_args.args[0]
    assert torch.equal(
        copied,
        torch.tensor(
            ((0, 0, 32, 8), (1, 16, 64, 24), (0, 8, 96, 16)),
            dtype=torch.int64,
        ),
    )


def test_workspace_device_rows_returns_committed_slice_without_copy():
    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    workspace = HostTransferWorkspace()
    workspace._range_device = torch.arange(24, dtype=torch.int64).reshape(6, 4)
    workspace._num_committed_ranges = 6

    rows = workspace.device_rows(2, 3)

    assert torch.equal(rows, workspace._range_device[2:5])
    assert rows.data_ptr() == workspace._range_device[2:].data_ptr()


def test_committed_range_slice_does_not_upload_metadata_again(monkeypatch):
    import tokenspeed_kernel.ops.kvcache.host_transfer as host_transfer

    workspace = MagicMock()
    address_table = object()
    range_table = object()
    workspace.bind_addresses.return_value = address_table
    workspace.device_rows.return_value = range_table
    device = SimpleNamespace(type="cuda")
    device_buffer = SimpleNamespace(device=device)
    stream = object()
    device_module = MagicMock()
    device_module.stream.return_value = nullcontext()
    triton_transfer = MagicMock()
    monkeypatch.setattr(torch, "get_device_module", lambda _: device_module)
    monkeypatch.setattr(host_transfer, "_transfer_cache_ranges_triton", triton_transfer)
    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)

    host_transfer.transfer_cache_ranges(
        "h2d",
        (device_buffer,),
        object(),
        (),
        stream,
        backend="triton",
        workspace=workspace,
        num_ranges=2,
        max_bytes=64,
        range_offset=7,
        ranges_committed=True,
    )

    workspace.commit_ranges.assert_not_called()
    workspace.device_rows.assert_called_once_with(7, 2)
    triton_transfer.assert_called_once_with(
        address_table,
        range_table,
        1,
        num_ranges=2,
        max_bytes=64,
        num_device_buffers=1,
        grid_cap=None,
    )


def test_unrelated_triton_runtime_error_does_not_fall_back_to_dma():
    assert not _triton_is_unavailable(
        RuntimeError("requested kernel specialization is not available")
    )


@requires_cuda
@pytest.mark.parametrize("backend", ["dma", "auto", "triton"])
def test_cache_ranges_round_trip_across_multiple_device_buffers(backend):
    first = torch.arange(64, dtype=torch.uint8, device="cuda")
    second = torch.arange(48, dtype=torch.bfloat16, device="cuda")
    second_bytes = second.view(torch.uint8)
    host = torch.zeros(96, dtype=torch.uint8, pin_memory=True)
    ranges = ((0, 8, 0, 24), (1, 16, 48, 32))
    stream = torch.cuda.Stream()

    try:
        transfer_cache_ranges(
            "d2h", (first, second), host, ranges, stream, backend=backend
        )
    except RuntimeError as error:
        message = str(error).lower()
        if backend == "triton" and (
            "unavailable" in message or "not device-mapped" in message
        ):
            pytest.skip(str(error))
        raise
    stream.synchronize()
    assert torch.equal(host[0:24], first[8:32].cpu())
    assert torch.equal(host[48:80], second_bytes[16:48].cpu())

    host[0:24].fill_(7)
    host[48:80].fill_(9)
    transfer_cache_ranges("h2d", (first, second), host, ranges, stream, backend=backend)
    stream.synchronize()
    assert torch.equal(first[8:32].cpu(), torch.full((24,), 7, dtype=torch.uint8))
    assert torch.equal(
        second_bytes[16:48].cpu(), torch.full((32,), 9, dtype=torch.uint8)
    )


@requires_cuda
def test_cache_ranges_grid_stride_and_workspace_reuse():
    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    device = torch.arange(256, dtype=torch.uint8, device="cuda")
    host = torch.zeros(256, dtype=torch.uint8, pin_memory=True)
    first = tuple((0, offset, offset, 8) for offset in range(0, 128, 8))
    second = tuple((0, offset, offset, 8) for offset in range(128, 256, 8))
    workspace = HostTransferWorkspace()
    stream = torch.cuda.Stream()

    try:
        transfer_cache_ranges(
            "d2h",
            (device,),
            host,
            first,
            stream,
            backend="triton",
            workspace=workspace,
            grid_cap=3,
        )
    except RuntimeError as error:
        message = str(error).lower()
        if "unavailable" in message or "not device-mapped" in message:
            pytest.skip(str(error))
        raise
    stream.synchronize()
    assert torch.equal(host[0:128], device[0:128].cpu())
    address_ptr = workspace._address_table.data_ptr()
    range_ptr = workspace._range_device.data_ptr()

    transfer_cache_ranges(
        "d2h",
        (device,),
        host,
        second,
        stream,
        backend="triton",
        workspace=workspace,
        grid_cap=3,
    )
    stream.synchronize()
    assert torch.equal(host[128:256], device[128:256].cpu())
    assert workspace._address_table.data_ptr() == address_ptr
    assert workspace._range_device.data_ptr() == range_ptr


@requires_cuda
def test_layerwise_workspace_reuse_preserves_small_h2d_after_large():
    """One immutable table keeps later KDA slices intact after a large MLA batch."""

    from tokenspeed_kernel.ops.kvcache.host_transfer import HostTransferWorkspace

    device = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    host = torch.arange(1 << 20, dtype=torch.uint8, pin_memory=True)
    workspace = HostTransferWorkspace()
    stream = torch.cuda.Stream()
    # One large MLA-like batch, then several small KDA-like (conv, recurrent)
    # slices launched without an intervening synchronize.
    large = tuple((0, i * 64, i * 64, 64) for i in range(400))
    small_batches = [
        ((0, 800 * 64, 800 * 64, 32), (0, 801 * 64, 801 * 64, 48)),
        ((0, 802 * 64, 802 * 64, 32), (0, 803 * 64, 803 * 64, 48)),
        ((0, 804 * 64, 804 * 64, 32), (0, 805 * 64, 805 * 64, 48)),
    ]
    try:
        batches = (large, *small_batches)
        descriptors = workspace.load_range_batches(batches)
        total_ranges = sum(count for _, count, _ in descriptors)
        with torch.cuda.stream(stream):
            workspace.commit_ranges(total_ranges, device.device, non_blocking=True)
        for range_offset, count, max_bytes in descriptors:
            transfer_cache_ranges(
                "h2d",
                (device,),
                host,
                (),
                stream,
                backend="triton",
                workspace=workspace,
                num_ranges=count,
                max_bytes=max_bytes,
                grid_cap=64,
                range_offset=range_offset,
                ranges_committed=True,
            )
    except RuntimeError as error:
        message = str(error).lower()
        if "unavailable" in message or "not device-mapped" in message:
            pytest.skip(str(error))
        raise
    stream.synchronize()
    for batch in small_batches:
        for _, device_offset, host_offset, num_bytes in batch:
            assert torch.equal(
                device[device_offset : device_offset + num_bytes].cpu(),
                host[host_offset : host_offset + num_bytes],
            )
