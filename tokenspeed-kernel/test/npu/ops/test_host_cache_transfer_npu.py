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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend NPU device tests for HBM-staged SDMA Host cache transfers."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.kvcache import host_transfer
from tokenspeed_kernel.ops.kvcache.triton import _staged_copy_geometry_slice_kernel
from utils import assert_no_triton_compile

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires an Ascend NPU device",
)


@requires_npu
@pytest.mark.parametrize("backend", ["auto", "dma"])
def test_npu_host_cache_transfer_multifield_layout_round_trip(backend):
    """Verify 6-layer multi-field (144K/32K/4K/4K) D2H and layerwise H2D."""
    num_layers = 6
    num_blocks_pool = 12
    # Multi-field layout mixing large latent buffers with small 4 KB metadata
    # fields (144 KB, 32 KB, 4 KB, 4 KB per layer).
    field_sizes = (147456, 32768, 4096, 4096)
    buffers = []
    rows = []
    layer_slices = []
    host_block_bytes = num_layers * sum(field_sizes)
    host_cursor = 0
    for _layer in range(num_layers):
        slice_start = len(rows)
        for field_bytes in field_sizes:
            buf_idx = len(buffers)
            buffers.append(
                torch.randint(
                    0,
                    251,
                    ((num_blocks_pool + 1) * field_bytes,),
                    dtype=torch.uint8,
                    device="npu",
                )
            )
            rows.append(
                (
                    0,
                    buf_idx,
                    0,
                    field_bytes,
                    host_block_bytes,
                    host_cursor,
                    1,
                    field_bytes,
                )
            )
            host_cursor += field_bytes
        layer_slices.append((slice_start, len(field_sizes)))

    geometry = host_transfer.build_host_transfer_geometry(
        rows=tuple(rows),
        layer_slices=tuple(layer_slices),
        group_packing=(1,),
        host_group_packing=(1,),
        host_lcm_block_bytes=host_block_bytes,
        num_host_lcm_blocks=num_blocks_pool,
        num_device_lcm_blocks=num_blocks_pool,
        num_device_buffers=len(buffers),
    )
    host = torch.zeros(
        num_blocks_pool * host_block_bytes, dtype=torch.uint8, pin_memory=True
    )
    # Small 2 MiB staging buffer forces multi-chunk staging across 6 blocks (~6.5 MiB).
    workspace = host_transfer.HostTransferWorkspace()
    workspace.bind_staging_buffer(
        host_transfer.StagingBuffer(default_bytes=2 * 1024 * 1024)
    )
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        bound = geometry.bind(buffers[0].device, non_blocking=False)
        mode = workspace.prepare_backend(buffers, host, backend=backend)
        if backend == "auto":
            assert mode.backend == "staged_dma"
            assert mode.uses_device_tables is True
            assert mode.layer_ready is False
        else:
            assert mode.backend == "dma"

    transfers = ((0, 1, 5), (0, 4, 2), (0, 7, 9), (0, 10, 1), (0, 3, 6), (0, 12, 3))
    num_blocks, _ = workspace.load_block_transfers(transfers, geometry=bound)
    with torch.npu.stream(stream):
        if mode.uses_device_tables:
            workspace.commit_block_transfers(
                num_blocks, buffers[0].device, non_blocking=True
            )

    expected_src = [buf.cpu() for buf in buffers]
    # Warm up both directions before checking steady-state compilation count.
    host_transfer.transfer_cache_blocks(
        "d2h",
        buffers,
        host,
        bound,
        workspace,
        stream,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=bound.num_field_rows,
        backend=backend,
        grid_cap=None,
        layer_ready_flags=None,
    )
    stream.synchronize()

    expected_host = torch.zeros_like(host)
    for buf_idx, dev_off, host_off, payload in host_transfer._make_block_ranges(
        bound,
        workspace,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=bound.num_field_rows,
    ):
        expected_host[host_off : host_off + payload] = expected_src[buf_idx][
            dev_off : dev_off + payload
        ]
    assert torch.equal(host, expected_host)

    # Layerwise H2D into zeroed buffers.
    for buf in buffers:
        buf.zero_()
    stream.wait_stream(torch.npu.current_stream())
    for slice_offset, slice_rows in bound.layer_slices:
        host_transfer.transfer_cache_blocks(
            "h2d",
            buffers,
            host,
            bound,
            workspace,
            stream,
            num_blocks=num_blocks,
            geometry_offset=slice_offset,
            num_geometry_rows=slice_rows,
            backend=backend,
            grid_cap=None,
            layer_ready_flags=None,
        )
    stream.synchronize()

    expected_restored = [torch.zeros_like(buf) for buf in expected_src]
    for buf_idx, dev_off, host_off, payload in host_transfer._make_block_ranges(
        bound,
        workspace,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=bound.num_field_rows,
    ):
        expected_restored[buf_idx][dev_off : dev_off + payload] = expected_host[
            host_off : host_off + payload
        ]
    for actual, wanted in zip(buffers, expected_restored, strict=True):
        assert torch.equal(actual.cpu(), wanted)

    if backend == "auto":
        # Varying block counts and layer slices must not recompile the kernel.
        with assert_no_triton_compile(_staged_copy_geometry_slice_kernel):
            for subset in (transfers[:1], transfers[:3], transfers[2:]):
                count, _ = workspace.load_block_transfers(subset, geometry=bound)
                with torch.npu.stream(stream):
                    workspace.commit_block_transfers(
                        count, buffers[0].device, non_blocking=True
                    )
                host_transfer.transfer_cache_blocks(
                    "d2h",
                    buffers,
                    host,
                    bound,
                    workspace,
                    stream,
                    num_blocks=count,
                    geometry_offset=0,
                    num_geometry_rows=bound.num_field_rows,
                    backend="auto",
                    grid_cap=None,
                    layer_ready_flags=None,
                )
                for slice_offset, slice_rows in bound.layer_slices:
                    host_transfer.transfer_cache_blocks(
                        "h2d",
                        buffers,
                        host,
                        bound,
                        workspace,
                        stream,
                        num_blocks=count,
                        geometry_offset=slice_offset,
                        num_geometry_rows=slice_rows,
                        backend="auto",
                        grid_cap=None,
                        layer_ready_flags=None,
                    )
            stream.synchronize()


@requires_npu
def test_npu_staged_transfer_handles_int64_offsets_past_2gb():
    """Guard against int32 offset wrap on large (>2 GiB) NPU KV cache pools."""
    page_stride = 1 << 20  # 1 MiB
    num_pages = 2100  # > 2 GiB
    payload = 4096
    far_page = num_pages  # offset = 2100 MiB > 2^31 bytes
    device_buf = torch.zeros(
        (num_pages + 1) * page_stride, dtype=torch.uint8, device="npu"
    )
    pattern = torch.arange(payload, dtype=torch.int64).remainder(251).to(torch.uint8)
    device_buf[far_page * page_stride : far_page * page_stride + payload].copy_(
        pattern.npu()
    )
    host = torch.zeros(payload * 2, dtype=torch.uint8, pin_memory=True)

    geometry = host_transfer.build_host_transfer_geometry(
        rows=((0, 0, 0, page_stride, payload, 0, 1, payload),),
        layer_slices=((0, 1),),
        group_packing=(1,),
        host_group_packing=(1,),
        host_lcm_block_bytes=payload,
        num_host_lcm_blocks=2,
        num_device_lcm_blocks=num_pages,
        num_device_buffers=1,
    ).bind(device_buf.device, non_blocking=False)

    workspace = host_transfer.HostTransferWorkspace()
    workspace.bind_staging_buffer(host_transfer.StagingBuffer(default_bytes=65536))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    num_blocks, _ = workspace.load_block_transfers(
        ((0, far_page, 2),), geometry=geometry
    )
    with torch.npu.stream(stream):
        workspace.prepare_backend((device_buf,), host, backend="auto")
        workspace.commit_block_transfers(
            num_blocks, device_buf.device, non_blocking=True
        )
    host_transfer.transfer_cache_blocks(
        "d2h",
        (device_buf,),
        host,
        geometry,
        workspace,
        stream,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=1,
        backend="auto",
        grid_cap=None,
        layer_ready_flags=None,
    )
    stream.synchronize()
    assert torch.equal(host[payload : 2 * payload], pattern)

    device_buf[far_page * page_stride : far_page * page_stride + payload].zero_()
    stream.wait_stream(torch.npu.current_stream())
    host_transfer.transfer_cache_blocks(
        "h2d",
        (device_buf,),
        host,
        geometry,
        workspace,
        stream,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=1,
        backend="auto",
        grid_cap=None,
        layer_ready_flags=None,
    )
    stream.synchronize()
    assert torch.equal(
        device_buf[far_page * page_stride : far_page * page_stride + payload].cpu(),
        pattern,
    )


@requires_npu
def test_npu_staged_transfer_unaligned_byte_fallback_is_byte_exact():
    """Verify the USE_WORDS=False byte path preserves neighboring bytes on NPU."""
    geometry = host_transfer.build_host_transfer_geometry(
        rows=(
            (0, 0, 13, 6001, 5014, 0, 2, 5003),
            (0, 1, 5, 41, 5014, 5003, 2, 11),
            (1, 1, 17, 53, 19, 0, 3, 13),
            (1, 0, 29, 47, 19, 13, 3, 5),
        ),
        layer_slices=((0, 4),),
        group_packing=(2, 3),
        host_group_packing=(2, 3),
        host_lcm_block_bytes=10028,
        num_host_lcm_blocks=2,
        num_device_lcm_blocks=4,
        num_device_buffers=2,
    )
    first = torch.full((32768,), 0xA5, dtype=torch.uint8, device="npu")
    second = torch.full((32768,), 0x5A, dtype=torch.uint8, device="npu")
    host = (
        torch.arange(20056, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .pin_memory()
    )
    workspace = host_transfer.HostTransferWorkspace()
    workspace.bind_staging_buffer(host_transfer.StagingBuffer(default_bytes=8192))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    transfers = ((0, 1, 1), (0, 4, 4), (1, 2, 3), (1, 4, 6))
    num_blocks, _ = workspace.load_block_transfers(transfers, geometry=geometry)
    with torch.npu.stream(stream):
        bound = geometry.bind(first.device, non_blocking=False)
        workspace.prepare_backend((first, second), host, backend="auto")
        workspace.commit_block_transfers(num_blocks, first.device, non_blocking=True)

    host_transfer.transfer_cache_blocks(
        "h2d",
        (first, second),
        host,
        bound,
        workspace,
        stream,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=4,
        backend="auto",
        grid_cap=4,
        layer_ready_flags=None,
    )
    stream.synchronize()

    expected_first = torch.full((32768,), 0xA5, dtype=torch.uint8)
    expected_second = torch.full((32768,), 0x5A, dtype=torch.uint8)
    expected_buffers = (expected_first, expected_second)
    for buf_idx, dev_off, host_off, payload in host_transfer._make_block_ranges(
        bound,
        workspace,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=4,
    ):
        expected_buffers[buf_idx][dev_off : dev_off + payload] = host[
            host_off : host_off + payload
        ]
    assert torch.equal(first.cpu(), expected_first)
    assert torch.equal(second.cpu(), expected_second)

    round_trip_host = torch.full_like(host, 0xCC)
    host_transfer.transfer_cache_blocks(
        "d2h",
        (first, second),
        round_trip_host,
        bound,
        workspace,
        stream,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=4,
        backend="auto",
        grid_cap=4,
        layer_ready_flags=None,
    )
    stream.synchronize()
    expected_host = torch.full_like(host, 0xCC)
    for _buf_idx, _dev_off, host_off, payload in host_transfer._make_block_ranges(
        bound,
        workspace,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=4,
    ):
        expected_host[host_off : host_off + payload] = host[
            host_off : host_off + payload
        ]
    assert torch.equal(round_trip_host, expected_host)
