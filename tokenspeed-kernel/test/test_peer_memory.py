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

"""CPU tests for peer-sharded CUDA VMM layout planning and TCPStore exchange."""

from __future__ import annotations

import socket

import pytest
import torch.multiprocessing as mp
from tokenspeed_kernel import peer_memory


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize(
    ("nbytes", "granularity", "world_size"),
    [
        (4096, 2 << 20, 1),
        (4096, 2 << 20, 4),
        (70_000 * 66, 2 << 20, 4),
        (70_000 * 66, 2 << 20, 16),
        (101_381_000_000, 2 << 20, 16),
        (101_381_000_000, 2 << 20, 14),
    ],
)
def test_plan_peer_device_layout_partitions_and_aligns(nbytes, granularity, world_size):
    layouts = [
        peer_memory.plan_peer_device_layout(nbytes, granularity, rank, world_size)
        for rank in range(world_size)
    ]
    base = layouts[0]
    assert base.nbytes == nbytes
    assert base.padded_bytes >= nbytes
    assert base.padded_bytes % granularity == 0
    assert base.padded_bytes - nbytes < granularity
    assert len(base.chunks) == world_size

    cursor = 0
    total_live = 0
    for rank, layout in enumerate(layouts):
        assert layout.chunks == base.chunks
        offset, length, owner = layout.chunks[rank]
        assert owner == rank
        assert offset == cursor
        assert length % granularity == 0
        assert layout.local_offset == offset
        assert layout.local_length == length
        assert layout.local_live_bytes <= length
        total_live += layout.local_live_bytes
        cursor += length
    assert cursor == base.padded_bytes
    assert total_live == nbytes


@pytest.mark.parametrize(
    ("num_rows", "row_bytes", "granularity", "world_size"),
    [
        (17, 66, 2 << 20, 4),
        (70_000, 66, 2 << 20, 4),
        (70_000, 66, 2 << 20, 16),
        # Realistic > 2^31 byte table (94.42 GiB) across 14 and 16 peers.
        (1_536_000_000, 66, 2 << 20, 16),
        (1_536_000_000, 66, 2 << 20, 14),
    ],
)
def test_plan_peer_row_range_covers_all_rows_without_overlap(
    num_rows, row_bytes, granularity, world_size
):
    nbytes = num_rows * row_bytes
    cursor = 0
    for rank in range(world_size):
        layout = peer_memory.plan_peer_device_layout(
            nbytes, granularity, rank, world_size
        )
        start, end = peer_memory.plan_peer_row_range(num_rows, row_bytes, layout)
        assert 0 <= start <= end <= num_rows
        assert start == cursor
        if start < end:
            # First owned row starts inside or after this peer's chunk.
            assert start * row_bytes >= layout.local_offset
            # Last owned row starts strictly inside this peer's live byte span.
            assert (end - 1) * row_bytes < layout.local_offset + layout.local_live_bytes
        cursor = end
    assert cursor == num_rows


def test_parse_host_port_ipv4_ipv6_and_errors():
    assert peer_memory.parse_host_port("127.0.0.1:29500") == ("127.0.0.1", 29500)
    assert peer_memory.parse_host_port("worker-0.local:1234") == (
        "worker-0.local",
        1234,
    )
    assert peer_memory.parse_host_port("[::1]:29501") == ("::1", 29501)
    assert peer_memory.parse_host_port("[2001:db8::1]:47100") == (
        "2001:db8::1",
        47100,
    )
    for bad in ("", "localhost", ":1234", "[::1]", "[::1]:0", "host:70000", "host:abc"):
        with pytest.raises(ValueError):
            peer_memory.parse_host_port(bad)


def test_resolve_peer_handle_type_rules():
    assert (
        peer_memory.resolve_peer_handle_type("auto", [1, 2], [None, None], [None, None])
        == "fabric"
    )
    assert (
        peer_memory.resolve_peer_handle_type(
            "auto", [1, 1], [None, "no fabric"], [None, None]
        )
        == "posix_fd"
    )
    with pytest.raises(RuntimeError, match="multi-host"):
        peer_memory.resolve_peer_handle_type(
            "auto", [1, 2], [None, "no fabric"], [None, None]
        )
    with pytest.raises(RuntimeError, match="fabric"):
        peer_memory.resolve_peer_handle_type(
            "fabric", [1, 1], [None, "no IMEX"], [None, None]
        )
    with pytest.raises(RuntimeError, match="single host"):
        peer_memory.resolve_peer_handle_type(
            "posix_fd", [1, 2], [None, None], [None, None]
        )
    with pytest.raises(ValueError, match="handle_type"):
        peer_memory.resolve_peer_handle_type("invalid", [1], [None], [None])


def _tcp_exchange_worker(rank: int, port: int) -> None:
    exchange = peer_memory.TCPStorePeerExchange.connect(
        f"127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        namespace="test-peer-mem",
    )
    gathered = exchange.all_gather_object(
        "hello", {"rank": rank, "payload": b"\x01\x02"}
    )
    assert gathered == [
        {"rank": 0, "payload": b"\x01\x02"},
        {"rank": 1, "payload": b"\x01\x02"},
    ]
    scoped = exchange.scoped("L1")
    assert scoped.all_gather_object("round", rank * 10) == [0, 10]
    scoped.barrier("done-0")
    scoped.barrier("done-1")


def test_tcp_store_peer_exchange_all_gather_and_barriers():
    port = _free_port()
    mp.spawn(_tcp_exchange_worker, args=(port,), nprocs=2, join=True)
