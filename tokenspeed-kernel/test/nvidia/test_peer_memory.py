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

"""GPU tests for peer-sharded CUDA VMM memory and release/resume lifecycle.

Run single-GPU directly with ``pytest <this file>`` or multi-GPU with
``torchrun --nproc-per-node=4 -m pytest <this file>``.
"""

from __future__ import annotations

import os
import socket
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from utils import is_nvidia

if not (is_nvidia() and torch.cuda.is_available()):
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

pytest.importorskip("cuda.bindings.driver")

from tokenspeed_kernel import peer_memory  # noqa: E402
from tokenspeed_kernel.ops.embedding import host_gather  # noqa: E402


def _default_pool_reserved_bytes() -> int:
    """Reserved bytes in PyTorch's ordinary CUDA caching allocator pool (0, 0)."""
    total = 0
    for segment in torch.cuda.memory_snapshot():
        if segment.get("segment_pool_id") == (0, 0):
            total += int(segment.get("total_size", 0))
    return total


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def peer_env():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    initialized = False
    tmp_ctx = None
    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group("gloo", timeout=timedelta(minutes=5))
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            dist.init_process_group(
                "gloo",
                init_method=f"file://{tmp_ctx.name}/rendezvous",
                rank=0,
                world_size=1,
                timeout=timedelta(minutes=5),
            )
        initialized = True
    try:
        yield rank, world_size, device
    finally:
        if initialized:
            dist.destroy_process_group()
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


@pytest.fixture(scope="module", params=["fabric", "posix_fd"])
def handle_type(request, peer_env):
    _, world_size, device = peer_env
    local_error = peer_memory.probe_peer_device_memory(device.index, request.param)
    if world_size > 1:
        errors = [None] * world_size
        dist.all_gather_object(errors, local_error)
        if any(errors):
            pytest.skip(f"{request.param} handles unavailable: {errors}")
    elif local_error is not None:
        pytest.skip(f"{request.param} handles unavailable: {local_error}")
    return request.param


def test_peer_device_memory_gather_and_release_resume(peer_env, handle_type):
    _, world_size, device = peer_env
    exchange = peer_memory.DistributedPeerExchange(
        dist.group.WORLD,
        namespace=f"test-peermem-{handle_type}",
    )
    # 70,000 rows x 66 bytes (~4.62 MiB) spans multiple 2 MiB VMM granules
    # with boundary-straddling rows across peer shards.
    rows = 70_000
    dim = 64
    row_bytes = dim + dim // 32
    nbytes = rows * row_bytes

    torch.cuda.empty_cache()
    pool_before = _default_pool_reserved_bytes()
    memory = peer_memory.create_peer_device_memory(
        nbytes,
        device_index=device.index,
        handle_type=handle_type,
        exchange=exchange,
    )
    try:
        # PeerDeviceMemory allocates through CUDA VMM, never Torch's (0, 0) pool.
        assert _default_pool_reserved_bytes() == pool_before
        original_ptr = memory.tensor.data_ptr()
        table = memory.tensor.view(rows, row_bytes)
        start, end = peer_memory.plan_peer_row_range(rows, row_bytes, memory.layout)

        generator = torch.Generator().manual_seed(42)
        ref_codes = torch.randint(
            0, 256, (rows, dim), dtype=torch.uint8, generator=generator
        )
        ref_codes[(ref_codes & 0x7F) == 0x7F] = 0
        ref_scales = torch.randint(
            100, 132, (rows, dim // 32), dtype=torch.uint8, generator=generator
        )
        ref_packed = torch.cat([ref_codes, ref_scales], dim=1)

        if end > start:
            table[start:end].copy_(ref_packed[start:end].to(device))
        torch.cuda.synchronize(device)
        memory.barrier("loaded")

        # Include chunk-boundary straddling rows (31775, 63550) and out-of-range ids.
        ids_cpu = torch.tensor(
            [[0, 31774, 31775, 31776, 63550, rows - 1, -1, rows]],
            dtype=torch.int64,
        )
        ref_device = ref_packed.to(device)
        ids_device = ids_cpu.to(device)
        expected_codes, expected_scales = host_gather.mxfp8_row_gather(
            ref_device, ids_device, None, None, scale_layout="packed"
        )
        expected_codes_cpu = expected_codes.view(torch.uint8).cpu()
        expected_scales_cpu = expected_scales.cpu()
        del ref_device, ids_device, expected_codes, expected_scales

        for _cycle in range(2):
            ids_device = ids_cpu.to(device)
            actual_codes, actual_scales = host_gather.mxfp8_row_gather(
                table, ids_device, None, None, scale_layout="packed"
            )
            assert torch.equal(actual_codes.view(torch.uint8).cpu(), expected_codes_cpu)
            assert torch.equal(actual_scales.cpu(), expected_scales_cpu)
            del ids_device, actual_codes, actual_scales

            memory.release_storage(backup=True)
            assert memory.released
            assert memory.local_physical_bytes == 0
            assert memory.tensor.data_ptr() == original_ptr
            torch.cuda.empty_cache()
            # After draining and releasing, ordinary Torch CUDA allocator
            # segments (0, 0) must not be pinned by PeerDeviceMemory.
            assert _default_pool_reserved_bytes() == pool_before

            memory.resume_storage(backup=True)
            assert not memory.released
            assert memory.local_physical_bytes == memory.layout.local_length
            assert memory.tensor.data_ptr() == original_ptr

        # Verify restored data after repeated release/resume cycles.
        ids_device = ids_cpu.to(device)
        actual_codes, actual_scales = host_gather.mxfp8_row_gather(
            table, ids_device, None, None, scale_layout="packed"
        )
        assert torch.equal(actual_codes.view(torch.uint8).cpu(), expected_codes_cpu)
        assert torch.equal(actual_scales.cpu(), expected_scales_cpu)
        if world_size > 1:
            memory.barrier("pre-close")
    finally:
        memory.close()
    assert memory.closed
    assert memory.local_physical_bytes == 0


def test_cross_process_peer_memory_refuses_per_engine_release(peer_env, handle_type):
    rank, world_size, device = peer_env
    if world_size != 1:
        pytest.skip("single-rank cross-process refusal check")
    port = _free_port()
    store = peer_memory.serve_peer_rendezvous(f"127.0.0.1:{port}")
    try:
        exchange = peer_memory.TCPStorePeerExchange.connect(
            f"127.0.0.1:{port}",
            rank=rank,
            world_size=1,
            rendezvous_id="gpu-test-1",
            namespace=f"test-xproc-{handle_type}",
        )
        memory = peer_memory.create_peer_device_memory(
            2 << 20,
            device_index=device.index,
            handle_type=handle_type,
            exchange=exchange,
        )
        try:
            assert memory.is_cross_process
            with pytest.raises(RuntimeError, match="Cross-process peer GPU memory"):
                memory.release_storage(backup=True)
            with pytest.raises(RuntimeError, match="Cross-process peer GPU memory"):
                memory.resume_storage(backup=True)
        finally:
            memory.close()
    finally:
        del store
