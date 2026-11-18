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

from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
import uuid

import pytest
import torch
from tokenspeed_kernel.ops.communication.peer_weights import ExpertPartition
from tokenspeed_kernel.thirdparty.cuda.dwdp import store
from tokenspeed_kernel.thirdparty.cuda.dwdp.layout import WeightLayout


def _fd_count() -> int:
    return len(os.listdir("/dev/fd"))


def _socket_path(label: str) -> str:
    directory = tempfile.gettempdir()
    name = f"tsd-{uuid.uuid4().hex[:8]}-{label}.sock"
    # Keep within the sockaddr_un path limit on hosts with long temp paths.
    if len(os.fsencode(directory)) + len(os.fsencode(name)) + 1 >= 100:
        directory = "/tmp"
    return os.path.join(directory, name)


def _send_one_then_finish(
    address: str,
    fd: int,
    ready: threading.Event,
    *,
    keep_open: threading.Event | None = None,
) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(address)
        server.listen(1)
        ready.set()
        connection, _ = server.accept()
        with connection:
            socket.send_fds(connection, [(1).to_bytes(4, "little")], [fd])
            if keep_open is not None:
                keep_open.wait(timeout=2)


@pytest.mark.parametrize("failure", ["eof", "timeout"])
def test_receive_fds_closes_descriptors_after_partial_receive(monkeypatch, failure):
    address = _socket_path("partial")
    read_fd, write_fd = os.pipe()
    keep_open = threading.Event() if failure == "timeout" else None
    ready = threading.Event()
    monkeypatch.setattr(store, "_FD_EXCHANGE_TIMEOUT_S", 0.1)
    server = threading.Thread(
        target=_send_one_then_finish,
        args=(address, read_fd, ready),
        kwargs={"keep_open": keep_open},
    )
    before = _fd_count()
    server.start()
    assert ready.wait(timeout=2)
    expected_error = RuntimeError if failure == "eof" else TimeoutError
    expected_message = (
        "closed the descriptor exchange early" if failure == "eof" else "timed out"
    )
    try:
        with pytest.raises(expected_error, match=expected_message):
            store._receive_fds(address, 2)
    finally:
        if keep_open is not None:
            keep_open.set()
        server.join(timeout=3)
        os.close(read_fd)
        os.close(write_fd)
        os.unlink(address)
    assert not server.is_alive()
    assert _fd_count() == before - 2


def _rank_exchange(
    rank, addresses, server, keys, exported, results, errors, start_delay=0
):
    time.sleep(start_delay)
    instance = object.__new__(store.CudaPeerWeightStore)
    rank_count = len(addresses)
    instance.partition = ExpertPartition(
        num_experts=2 * rank_count, rank=rank, size=rank_count
    )
    try:
        results[rank] = instance._exchange_descriptors(
            keys, exported, addresses, server
        )
    except Exception as error:
        errors[rank] = error


def _run_rank_exchanges(rank_count, keys, exported_by_rank):
    addresses = [_socket_path(f"rank-{rank}") for rank in range(rank_count)]
    servers = []
    for address in addresses:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(address)
        server.listen(rank_count)
        server.settimeout(2)
        servers.append(server)
    results = [None] * rank_count
    errors = [None] * rank_count
    threads = [
        threading.Thread(
            target=_rank_exchange,
            args=(
                rank,
                addresses,
                servers[rank],
                keys,
                exported_by_rank[rank],
                results,
                errors,
            ),
            kwargs={"start_delay": rank * 0.02},
        )
        for rank in range(rank_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
    assert all(not thread.is_alive() for thread in threads)
    for server in servers:
        server.close()
    for address in addresses:
        os.unlink(address)
    return results, errors


def _close_all(fds):
    for fd in fds:
        os.close(fd)


def _owner_files(count):
    files = []
    for rank in range(count):
        owner_file = tempfile.TemporaryFile()
        owner_file.write(f"owner-{rank}".encode())
        owner_file.flush()
        files.append(owner_file)
    return files


def _exported_by_owner(keys, files):
    return [{key: owner_file.fileno() for key in keys} for owner_file in files]


def test_exchange_descriptors_handles_four_skewed_ranks_and_empty_keys():
    keys = [(index, "weight") for index in range(4)]
    sources = _owner_files(4)
    exported = _exported_by_owner(keys, sources)
    try:
        results, errors = _run_rank_exchanges(4, keys, exported)
        assert errors == [None] * 4
        for rank, rank_results in enumerate(results):
            assert len(rank_results) == 4
            assert rank_results[rank] == {}
            for peer in range(4):
                if peer == rank:
                    continue
                received = rank_results[peer]
                assert list(received) == keys
                assert all(
                    os.pread(fd, 64, 0) == f"owner-{peer}".encode()
                    for fd in received.values()
                )
                assert len(set(received.values())) == len(keys)
                _close_all(received.values())
        empty_results, empty_errors = _run_rank_exchanges(4, [], [{} for _ in range(4)])
        assert empty_errors == [None] * 4
        assert empty_results == [[{} for _ in range(4)] for _ in range(4)]
    finally:
        for owner_file in sources:
            owner_file.close()


def test_exchange_descriptors_sends_more_than_250_fds_in_two_ranks():
    keys = [(index, "weight") for index in range(251)]
    sources = _owner_files(2)
    exported = _exported_by_owner(keys, sources)
    try:
        results, errors = _run_rank_exchanges(2, keys, exported)
        assert errors == [None, None]
        for rank, rank_results in enumerate(results):
            received = rank_results[1 - rank]
            assert list(received) == keys
            assert len(set(received.values())) == len(keys)
            assert all(
                os.pread(fd, 64, 0) == f"owner-{1 - rank}".encode()
                for fd in received.values()
            )
            _close_all(received.values())
    finally:
        for owner_file in sources:
            owner_file.close()


def test_exchange_descriptors_cleans_prior_peer_fds_after_later_peer_failure(
    monkeypatch,
):
    keys = [(index, "weight") for index in range(4)]
    sources = _owner_files(3)
    exported = _exported_by_owner(keys, sources)
    exported[2][keys[-1]] = -1
    before = _fd_count()
    monkeypatch.setattr(store, "_FD_EXCHANGE_TIMEOUT_S", 0.5)
    try:
        _, errors = _run_rank_exchanges(3, keys, exported)
        assert all(error is not None for error in errors)
        assert _fd_count() == before
    finally:
        for owner_file in sources:
            owner_file.close()


def test_finalize_closes_earlier_exports_when_later_export_fails(monkeypatch):
    source = tempfile.TemporaryFile()
    exported = []
    calls = 0

    def export(handle, *, handle_type):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected export failure")
        fd = os.dup(source.fileno())
        exported.append(fd)
        return fd

    class PublishedEvent:
        def synchronize(self):
            pass

    instance = object.__new__(store.CudaPeerWeightStore)
    instance._closed = False
    instance._finalized = False
    instance._published = PublishedEvent()
    instance.partition = ExpertPartition(num_experts=2, rank=0, size=2)
    instance.model_identity = "fd-cleanup-test"
    instance.handle_type = "posix_fd"
    instance.pool_page_bytes = 1
    instance.granularity = 1
    layout = WeightLayout.create(
        expert_bytes=1,
        experts=2,
        first=0,
        count=1,
        granularity=1,
        pool_page_bytes=1,
    )
    instance._layers = {
        0: store._Layer(
            index=0,
            slot=0,
            layout="test",
            fields=tuple(
                store._Field(
                    name=f"weight_{index}",
                    shape=(1,),
                    dtype=torch.uint8,
                    expert_bytes=1,
                    layout=layout,
                    handle=100 + index,
                    address=0,
                    replica=None,
                )
                for index in range(3)
            ),
        )
    }
    monkeypatch.setattr(store.vmm, "export", export)

    try:
        with pytest.raises(RuntimeError, match="injected export failure"):
            instance.finalize()
        assert calls == 3
        assert len(exported) == 2
        for fd in exported:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        source.close()
