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

import multiprocessing
import os
import resource
import socket
import tempfile
import threading
import time
import uuid
from contextlib import closing

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
    expected_error = ConnectionError if failure == "eof" else TimeoutError
    expected_message = (
        "closed the descriptor exchange early" if failure == "eof" else "timed out"
    )
    try:
        with pytest.raises(expected_error, match=expected_message):
            store._receive_fds(address, 2, uid=os.geteuid())
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
        received_values = []
        with closing(
            instance._exchange_descriptors(keys, exported, addresses, server)
        ) as owners:
            for owner, received in enumerate(owners):
                assert list(received) == ([] if owner == rank else keys)
                assert len(set(received.values())) == len(received)
                received_values.append(
                    {key: os.pread(fd, 64, 0) for key, fd in received.items()}
                )
        results[rank] = received_values
    except Exception as error:
        errors[rank] = error


def _run_rank_exchanges(rank_count, keys, exported_by_rank):
    before = _fd_count()
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
    assert _fd_count() == before
    return results, errors


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
                    value == f"owner-{peer}".encode() for value in received.values()
                )

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
            assert all(
                value == f"owner-{1 - rank}".encode() for value in received.values()
            )
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


def _limited_rank_exchange(rank, addresses, servers, keys, results):
    exported = []
    store._FD_EXCHANGE_TIMEOUT_S = 2.0
    try:
        for peer, server in enumerate(servers):
            if peer != rank:
                server.close()
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
        with tempfile.TemporaryFile() as source:
            source.write(f"owner-{rank}".encode())
            source.flush()
            exported = [os.dup(source.fileno()) for _ in keys]
            instance = object.__new__(store.CudaPeerWeightStore)
            instance.partition = ExpertPartition(
                2 * len(addresses), rank, len(addresses)
            )
            for owner, received in enumerate(
                instance._exchange_descriptors(
                    keys,
                    dict(zip(keys, exported, strict=True)),
                    addresses,
                    servers[rank],
                )
            ):
                assert len(received) == (0 if owner == rank else len(keys))
                assert all(
                    os.pread(fd, 64, 0) == f"owner-{owner}".encode()
                    for fd in received.values()
                )
        results.put((rank, None))
    except Exception as error:
        results.put((rank, repr(error)))
    finally:
        for fd in exported:
            os.close(fd)
        servers[rank].close()


def test_exchange_imports_one_owner_at_a_time_under_fd_limit():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1024:
        pytest.skip("The process cannot raise its descriptor limit to 1024")
    ranks = 6
    keys = [(index, "weight") for index in range(200)]
    addresses = [_socket_path(f"limited-{rank}") for rank in range(ranks)]
    servers = []
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    processes = []
    try:
        for address in addresses:
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(address)
            server.listen(ranks)
            server.settimeout(5)
            servers.append(server)
        processes = [
            context.Process(
                target=_limited_rank_exchange,
                args=(rank, addresses, servers, keys, results),
            )
            for rank in range(ranks)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 10
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
        received = [results.get(timeout=2) for _ in processes]
        assert sorted(received) == [(rank, None) for rank in range(ranks)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        results.close()
        results.join_thread()
        for server in servers:
            server.close()
        for address in addresses:
            if os.path.exists(address):
                os.unlink(address)


def test_send_fds_rejects_unexpected_uid_before_sending():
    address = _socket_path("wrong-client-uid")
    received = []
    before = _fd_count()

    def receive():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(address)
            _, fds, _, _ = socket.recv_fds(client, 4, 1)
            received.extend(fds)
            for fd in fds:
                os.close(fd)

    try:
        with tempfile.TemporaryFile() as source, socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM
        ) as server:
            server.bind(address)
            server.listen(1)
            server.settimeout(2)
            client = threading.Thread(target=receive)
            client.start()
            try:
                with pytest.raises(PermissionError, match="descriptor exchange UID"):
                    store._send_fds(server, [source.fileno()], 1, uid=os.geteuid() ^ 1)
            finally:
                client.join(timeout=3)
            assert not client.is_alive()
            assert received == []
    finally:
        os.unlink(address)
    assert _fd_count() == before


def test_receive_fds_rejects_unexpected_server_uid(monkeypatch):
    address = _socket_path("wrong-server-uid")
    closed = []
    monkeypatch.setattr(store, "_FD_EXCHANGE_TIMEOUT_S", 0.1)
    before = _fd_count()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(address)
            server.listen(1)

            def accept():
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(2)
                    closed.append(connection.recv(1) == b"")

            sender = threading.Thread(target=accept)
            sender.start()
            try:
                with pytest.raises(PermissionError, match="descriptor exchange UID"):
                    store._receive_fds(address, 1, uid=os.geteuid() ^ 1)
            finally:
                sender.join(timeout=3)
            assert not sender.is_alive()
            assert closed == [True]
    finally:
        os.unlink(address)
    assert _fd_count() == before


def test_failed_import_iteration_closes_current_owner_fds():
    address = _socket_path("failed-consumer")
    ready = threading.Event()
    received = []
    before = _fd_count()
    try:
        with tempfile.TemporaryFile() as source:
            sender = threading.Thread(
                target=_send_one_then_finish, args=(address, source.fileno(), ready)
            )
            sender.start()
            assert ready.wait(timeout=2)
            instance = object.__new__(store.CudaPeerWeightStore)
            instance.partition = ExpertPartition(4, 1, 2)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                owners = instance._exchange_descriptors(
                    [(0, "weight")], {}, [address, "unused"], server
                )
                try:
                    with pytest.raises(ValueError, match="import failed"):
                        with closing(owners):
                            received.extend(next(owners).values())
                            assert len(received) == 1
                            os.fstat(received[0])
                            raise ValueError("import failed")
                finally:
                    sender.join(timeout=3)
                assert not sender.is_alive()
                for fd in received:
                    with pytest.raises(OSError):
                        os.fstat(fd)
    finally:
        os.unlink(address)
    assert _fd_count() == before
