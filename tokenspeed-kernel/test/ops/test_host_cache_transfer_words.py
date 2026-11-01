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

# Compare word and byte Host-cache transfers with mixed field alignment.

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache import host_transfer
from tokenspeed_kernel.ops.kvcache.triton import _transfer_cache_blocks_kernel
from utils import assert_no_triton_compile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "direction,layered", [("d2h", False), ("h2d", False), ("h2d", True)]
)
def test_mixed_field_alignment_preserves_bytes_and_layer_readiness(
    direction, layered, graph
):
    if layered and not host_transfer.layer_ready_ptx_supported():
        pytest.skip("layer-ready flags require NVIDIA PTX")
    # One field has aligned and unaligned compact children. The large field
    # crosses a transfer chunk and ends with a masked group of words.
    rows = (
        (0, 0, 0, 128, 77, 0, 3, 64),
        (0, 0, 64, 128, 77, 64, 3, 13),
        (1, 1, 0, 4160, 4160, 0, 1, 4160),
    )
    geometry = host_transfer.build_host_transfer_geometry(
        rows=rows,
        layer_slices=((0, 2), (2, 0), (2, 1)),
        group_packing=(2, 1),
        host_group_packing=(3, 1),
        host_lcm_block_bytes=8192,
        num_host_lcm_blocks=4,
        num_device_lcm_blocks=2,
        num_device_buffers=2,
    )
    buffers = (
        torch.full((640,), 0xA5, dtype=torch.uint8, device="cuda"),
        torch.full((12480,), 0xA5, dtype=torch.uint8, device="cuda"),
    )
    host = torch.full((32768,), 0xA5, dtype=torch.uint8, pin_memory=True)
    workspace = host_transfer.HostTransferWorkspace()
    stream, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    initialized = torch.cuda.Event()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bound = geometry.bind(buffers[0].device, non_blocking=True)
    initial = ((0, 1, 1), (0, 2, 2), (0, 4, 6), (1, 2, 4))
    changed = ((0, 1, 2), (0, 2, 1), (0, 4, 5), (1, 1, 3))

    def commit(transfers):
        count, _ = workspace.load_block_transfers(transfers, geometry=geometry)
        with torch.cuda.stream(stream):
            workspace.commit_block_transfers(
                count, buffers[0].device, non_blocking=True
            )
        return count

    count = commit(initial)
    snapshots = []

    def submit():
        with torch.cuda.stream(stream):
            flags = (
                workspace.prepare_layer_ready(3, buffers[0].device) if layered else None
            )
            initialized.record(stream)
            host_transfer.transfer_cache_blocks(
                direction,
                buffers,
                host,
                bound,
                workspace,
                stream,
                num_blocks=count,
                geometry_offset=0,
                num_geometry_rows=len(rows),
                backend="triton",
                grid_cap=2,
                layer_ready_flags=flags,
            )
        if layered:
            with torch.cuda.stream(consumer):
                consumer.wait_event(initialized)
                host_transfer.wait_layer_ready(flags, 0)
                snapshots.append(buffers[0].clone())
                # Empty consumers must also publish readiness.
                host_transfer.wait_layer_ready(flags, 1)
                host_transfer.wait_layer_ready(flags, 2)
                snapshots.append(buffers[1].clone())
            stream.wait_stream(consumer)

    # Bind Host pointers and warm both branches before graph capture.
    submit()
    torch.cuda.synchronize()
    captured = None
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            submit()
        torch.cuda.synchronize()

    # Captured kernels must observe new block IDs and payloads while all
    # allocation addresses stay fixed.
    assert commit(changed) == count
    generator = torch.Generator().manual_seed(1927)
    for buffer in buffers:
        buffer.fill_(0xA5)
    host.fill_(0xA5)
    if direction == "d2h":
        for buffer in buffers:
            buffer.copy_(
                torch.randint(
                    0, 256, (buffer.numel(),), dtype=torch.uint8, generator=generator
                )
            )
    else:
        host.copy_(
            torch.randint(
                0, 256, (host.numel(),), dtype=torch.uint8, generator=generator
            )
        )
    torch.cuda.synchronize()
    expected_device = [buffer.cpu() for buffer in buffers]
    expected_host = host.clone()
    for group, buffer, zero, stride, child_bytes, offset, packing, payload in rows:
        for transfer_group, device_id, host_id in changed:
            if group != transfer_group:
                continue
            parent, child = divmod(host_id - 1, packing)
            h = parent * 8192 + child * child_bytes + offset
            d = zero + device_id * stride
            if direction == "d2h":
                expected_host[h : h + payload] = expected_device[buffer][
                    d : d + payload
                ]
            else:
                expected_device[buffer][d : d + payload] = host[h : h + payload]
    with assert_no_triton_compile(_transfer_cache_blocks_kernel):
        if captured is None:
            submit()
        else:
            with torch.cuda.stream(stream):
                captured.replay()
        torch.cuda.synchronize()
    assert torch.equal(host, expected_host)
    for actual, expected in zip(buffers, expected_device, strict=True):
        assert torch.equal(actual.cpu(), expected)
    if layered:
        assert torch.equal(snapshots[-2].cpu(), expected_device[0])
        assert torch.equal(snapshots[-1].cpu(), expected_device[1])
