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

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tokenspeed.runtime.pd.base.mooncake_engine import MooncakeTransferEngine


def test_ascend_initialization_uses_existing_batch_api(monkeypatch):
    class Engine:
        def initialize(self, *args):
            self.args = args
            return 0

        def get_rpc_port(self):
            return 4321

        def batch_transfer_sync_write(self, *args):
            return 0

    monkeypatch.setitem(
        sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=Engine)
    )
    engine = MooncakeTransferEngine("10.0.0.1", 2, transport="ascend_direct")
    assert engine.engine.args == ("10.0.0.1", "P2PHANDSHAKE", "ascend_direct", "")
    assert engine.get_session_id() == "10.0.0.1:4321"


@pytest.mark.parametrize("batch_size", [1, 2, 4096])
def test_ascend_page_addresses_and_bounded_batches(batch_size):
    seen = []
    engine = object.__new__(MooncakeTransferEngine)
    engine.transport = "ascend_direct"
    engine.engine = SimpleNamespace(
        batch_transfer_sync_write=lambda *args: seen.append(args) or 0
    )
    src = np.array([3, 1, 7], dtype=np.int64)
    dst = np.array([8, 2, 5], dtype=np.int64)
    fields = np.array(
        [[2**36, 4096, 2**37, 8192, 2048], [2**38, 128, 2**39, 256, 64]], dtype=np.int64
    )
    assert (
        engine.batch_transfer_sync_pages(
            "peer", src, dst, fields, max_batch_size=batch_size
        )
        == 0
    )
    actual = []
    for session, sources, destinations, lengths in seen:
        assert session == "peer"
        assert 0 < len(sources) <= batch_size
        actual.extend(zip(sources, destinations, lengths))
    expected = [
        (int(sb + s * ss), int(db + d * ds), int(n))
        for sb, ss, db, ds, n in fields
        for s, d in zip(src, dst)
    ]
    assert actual == expected


def test_ascend_failure_stops_remaining_pages():
    calls = []
    engine = object.__new__(MooncakeTransferEngine)
    engine.transport = "ascend_direct"
    engine.engine = SimpleNamespace(
        batch_transfer_sync_write=lambda *a: calls.append(a) or -1
    )
    pages = np.array([1, 2, 3], dtype=np.int64)
    fields = np.array([[100, 16, 200, 16, 8]], dtype=np.int64)
    assert (
        engine.batch_transfer_sync_pages("peer", pages, pages, fields, max_batch_size=1)
        == -1
    )
    assert len(calls) == 1
    calls.clear()
    assert (
        engine.batch_transfer_sync_pages(
            "peer", pages[:0], pages[:0], fields, max_batch_size=1
        )
        == 0
    )
    assert calls == []


@pytest.mark.parametrize("transport", ["rdma", "ascend_direct"])
def test_decode_registration_rebuild_preserves_transport(monkeypatch, transport):
    import threading

    from tokenspeed.runtime.pd.mooncake import decode

    created = []
    old = SimpleNamespace(
        hostname="10.0.0.1",
        gpu_id=2,
        ib_device=None,
        transport=transport,
        get_session_id=lambda: "old",
    )
    replacement = SimpleNamespace(get_session_id=lambda: "new")

    def make_engine(**kwargs):
        created.append(kwargs)
        return replacement

    monkeypatch.setattr(decode, "MooncakeTransferEngine", make_engine)
    monkeypatch.setattr(
        decode.MooncakeKVManagerBase, "restore_buffer_registration", lambda self: None
    )
    manager = object.__new__(decode.MooncakeKVManagerDecode)
    manager.engine = old
    manager.connection_lock = threading.Lock()
    manager.connection_pool = {"peer": object()}
    manager.restore_buffer_registration()
    assert created == [
        dict(hostname="10.0.0.1", gpu_id=2, ib_device=None, transport=transport)
    ]
    assert manager.engine is replacement
    assert manager.connection_pool == {}
