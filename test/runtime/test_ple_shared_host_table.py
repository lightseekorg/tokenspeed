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

import ctypes
import ctypes.util
import gc
import importlib.util
import mmap
import os
import sys
import weakref
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.layers.ple_lookup import PLELookup, quantize_ple_embedding_rows
from tokenspeed.runtime.layers.shared_host_table import (
    HostTableGroup,
    SharedHostTable,
    _prefer_local_numa,
)


def _mapping_class():
    path = (
        Path(__file__).parents[2] / "python/tokenspeed/runtime/distributed/mapping.py"
    )
    spec = importlib.util.spec_from_file_location("ple_shared_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Mapping


def _single_group():
    return HostTableGroup(0, (0,), (0,), None)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux NUMA policy")
@pytest.mark.parametrize("interleave", [False, True])
def test_gpu_nodes_are_retained_in_numa_policy(interleave):
    library = ctypes.util.find_library("numa")
    if library is None:
        pytest.skip("libnuma unavailable")
    lib = ctypes.CDLL(library, use_errno=True)
    query = lib.get_mempolicy
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    allowed = ctypes.c_ulong()
    bits = ctypes.sizeof(allowed) * 8
    if query(None, ctypes.byref(allowed), bits + 1, None, 4) != 0:
        pytest.skip(f"NUMA policy unavailable: {os.strerror(ctypes.get_errno())}")
    nodes = tuple(i for i in range(bits) if allowed.value & (1 << i))
    if not nodes:
        pytest.skip("no allowed NUMA node in the queried bitmap")
    if interleave and len(nodes) < 2:
        pytest.skip("requires two allowed NUMA nodes")
    nodes = nodes[:2] if interleave else nodes[-1:]
    with mmap.mmap(
        -1, mmap.PAGESIZE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    ) as mapping:
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
        lib.mbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        if lib.mbind(address, mmap.PAGESIZE, 0, None, 0, 0) != 0:
            pytest.skip(f"mbind unavailable: {os.strerror(ctypes.get_errno())}")
        _prefer_local_numa(address, mmap.PAGESIZE, nodes)
        mode, mask = ctypes.c_int(), ctypes.c_ulong()
        assert query(ctypes.byref(mode), ctypes.byref(mask), bits + 1, address, 2) == 0
        assert mode.value == (3 if interleave else 1)
        assert mask.value == sum(1 << node for node in nodes)


def test_shared_storage_outlives_its_owner(tmp_path):
    table = SharedHostTable.allocate(
        (32, 4),
        torch.float32,
        group=_single_group(),
        backend="shm",
        directory=str(tmp_path),
        register=False,
    )
    table.host.fill_(13)
    view = table.host[3:5]
    owner = weakref.ref(table)
    del table
    gc.collect()
    assert owner() is None
    torch.testing.assert_close(view, torch.full((2, 4), 13.0))
    view.add_(1)
    torch.testing.assert_close(view, torch.full((2, 4), 14.0))
    assert not list(tmp_path.iterdir())


def test_insufficient_capacity_fails_before_mapping(tmp_path):
    with pytest.raises(RuntimeError, match="setup failed"):
        SharedHostTable.allocate(
            (1 << 58, 8),
            torch.float32,
            group=_single_group(),
            backend="shm",
            directory=str(tmp_path),
            register=False,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("shape", [(), (0,), (-1, 3)])
def test_invalid_shared_shape(shape, tmp_path):
    with pytest.raises(ValueError):
        SharedHostTable.allocate(
            shape,
            torch.float32,
            group=_single_group(),
            backend="shm",
            directory=str(tmp_path),
            register=False,
        )


def _worker(rank, rendezvous, directory, tp_size, scale_mode, node_size):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=40),
    )
    try:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager,
        )

        process_group_manager.register_process_group(
            "gloo", (0, 1, 2, 3), dist.group.WORLD
        )
        mapping = _mapping_class()(
            rank=rank,
            world_size=4,
            attn_tp_size=tp_size,
            attn_dp_size=4 // tp_size,
            nprocs_per_node=node_size,
        )
        lookup = PLELookup(
            mapping,
            vocab_size=256,
            ngram_heads=2,
            head_dim=3,
            storage_dtype=None if scale_mode == "none" else torch.float8_e4m3fn,
            output_dtype=torch.float32,
            offload=False,
            prefix="ple",
            host_table_backend="shm",
            shared_directory=directory,
        )
        assert lookup.ngram_embedding.weight.shape == (256, 3)
        assert lookup._shared_table.is_writer == (rank % node_size == 0)
        source = torch.arange(768).reshape(256, 3).float() / 32
        if scale_mode == "offline":
            payload = source.to(torch.float8_e4m3fn)
            expected = payload.float() * 0.5
            lookup.load_scale(torch.tensor(0.5))
        elif scale_mode == "online":
            payload = source
            quantized, scales = quantize_ple_embedding_rows(source)
            expected = quantized.float() * scales[:, None]
        else:
            payload, expected = source, source
        if not lookup._shared_table.is_writer:
            payload = (payload.float() + 10).to(payload.dtype)
        for part in reversed(range(7)):
            lookup.load_shard(payload[part * 37 : (part + 1) * 37], part, 7)
        with pytest.raises(RuntimeError, match="finish_loading"):
            lookup.start(
                torch.zeros((1, 2), dtype=torch.long), lookup.make_layout(None, 1)
            )
        lookup.finish_loading()
        lookup.finish_loading()
        # Unequal forward counts cannot rendezvous in any lookup collective.
        for step in range(rank + 2):
            rows = (rank + step) % 4
            ids = (torch.arange(rows * 2).reshape(rows, 2) + rank * 63 + step) % 256
            actual = lookup.finish(lookup.start(ids, lookup.make_layout(None, rows)))
            torch.testing.assert_close(
                actual, expected[ids].reshape(rows, 6), rtol=0, atol=0
            )
        with pytest.raises(RuntimeError, match="Reloading"):
            lookup.load_shard(payload[:37], 0, 7)
        dist.barrier()
        assert not [
            path
            for path in Path(directory).iterdir()
            if path.name.startswith("ts-ple-")
        ]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("scale_mode", ["none", "offline", "online"])
def test_shared_full_table_local_lookup(tmp_path, tp_size, scale_mode):
    mp.spawn(
        _worker,
        args=(
            (tmp_path / "rendezvous").as_uri(),
            str(tmp_path),
            tp_size,
            scale_mode,
            4,
        ),
        nprocs=4,
        join=True,
    )


def test_shared_full_table_one_copy_per_node(tmp_path):
    mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), 2, "online", 2),
        nprocs=4,
        join=True,
    )


def _failure_worker(rank, rendezvous, directory):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        local = (0, 1) if rank < 2 else (2, 3)
        group = HostTableGroup(rank, (0, 1, 2, 3), local, dist.group.WORLD)
        path = directory if rank < 2 else str(Path(directory) / "missing-directory")
        with pytest.raises(RuntimeError, match="setup failed"):
            SharedHostTable.allocate(
                (32, 8),
                torch.float32,
                group=group,
                backend="shm",
                directory=path,
                register=False,
            )
        dist.barrier()
        assert not [
            path
            for path in Path(directory).iterdir()
            if path.name.startswith("ts-ple-")
        ]
    finally:
        dist.destroy_process_group()


def test_failure_on_one_node_aborts_every_rank(tmp_path):
    mp.spawn(
        _failure_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=4,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA UVA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_shm_cuda_alias_reads_same_physical_bytes(dtype):
    from tokenspeed_kernel.ops.ple import ple_host_gather

    table = SharedHostTable.allocate(
        (256, 64),
        dtype,
        group=_single_group(),
        backend="shm",
        directory="/dev/shm",
        register=True,
    )
    values = (torch.arange(256 * 64).reshape(256, 64).float() % 128).to(dtype)
    table.host.copy_(values)
    ids = torch.tensor([0, 31, 255], dtype=torch.int64, device="cuda")
    out_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    out = torch.empty((3, 64), dtype=out_dtype, device="cuda")
    ple_host_gather(
        table.device,
        ids,
        out,
        0,
        256,
        0.5 if dtype == torch.float8_e4m3fn else None,
        None,
    )
    expected = values.float()[ids.cpu()] * (0.5 if dtype == torch.float8_e4m3fn else 1)
    torch.testing.assert_close(out.cpu(), expected.to(out_dtype), rtol=0, atol=0)
