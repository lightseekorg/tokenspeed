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

"""EPD encode->prefill transfer: wire-codec round-trips, row-shard geometry,
and prefill receive-buffer sizing. Pure-logic silent-corruption guards (the
RDMA write itself is unchecked); no GPU, no live Mooncake hop."""

from __future__ import annotations

import threading

import pytest
import torch

from tokenspeed.runtime.epd.entities import (
    REGISTER_ROOM_SENTINEL,
    EmbeddingArgsRegisterInfo,
    EmbeddingChunk,
    EmbeddingTransferError,
    EmbeddingTransferInfo,
)
from tokenspeed.runtime.epd.mooncake.encode import (
    shard_payload,
    validate_fanout_frames,
)
from tokenspeed.runtime.epd.mooncake.sender import (
    MooncakeEmbeddingSender,
)
from tokenspeed.runtime.epd.prefill_admission import (
    DONE,
    receive_encoded_embeddings,
    shard_rows,
    start_embedding_receive,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
)
from tokenspeed.runtime.pd.base.status import TransferPoll

# Each merged concern keeps its own per-test setup (the shard
# pointer-math tests need the TOKENSPEED_EPD_RECV_POOL_SLOTS=0 path, the
# receive tests need a small pool); route by test name.
_RECV_TESTS = {
    "test_receive_sizes_buffers_per_item_no_deepstack",
    "test_receive_allocates_deepstack_columns",
    "test_receive_skips_already_encoded_item_on_recall",
}


@pytest.fixture(autouse=True)
def _epd_transfer_env(request, monkeypatch):
    if request.node.name in _RECV_TESTS:
        _recv_setup(monkeypatch)
    else:
        _shard_setup(monkeypatch)


# A high device VA to catch any 32-bit truncation / wrong struct format.
HIGH_PTR = 0xFFFF_FFFF_FFFF_F000
MID_PTR = 0x7FFF_FFFF_0000


def test_register_info_round_trip():
    info = EmbeddingArgsRegisterInfo(
        room=REGISTER_ROOM_SENTINEL,
        endpoint="10.0.0.7",
        dst_port=5123,
        mooncake_session_id="10.0.0.7:41999",
        dst_embedding_ptr=HIGH_PTR,
        dst_deepstack_ptr=MID_PTR,
    )
    frames = info.to_zmq()
    assert len(frames) == 6  # layout lock
    assert frames[0] == b"None"
    assert EmbeddingArgsRegisterInfo.from_zmq(frames) == info


def test_transfer_info_round_trip_with_deepstack():
    info = EmbeddingTransferInfo(
        room=42,
        endpoint="10.0.0.7",
        dst_port=5123,
        mooncake_session_id="10.0.0.7:41999",
        dst_embedding_ptr=HIGH_PTR,
        dst_deepstack_ptr=MID_PTR,
        n_tokens=1369,
        hidden=3584,
        dtype="torch.bfloat16",
        has_deepstack=True,
        required_dst_info_num=1,
    )
    frames = info.to_zmq()
    assert len(frames) == 13  # layout lock (v2: row_start + span appended)
    rt = EmbeddingTransferInfo.from_zmq(frames)
    assert rt == info
    assert rt.dtype == "torch.bfloat16"
    assert rt.has_deepstack is True


def test_transfer_info_round_trip_no_deepstack():
    info = EmbeddingTransferInfo(
        room=0,
        endpoint="::1",
        dst_port=65535,
        mooncake_session_id="x:0",
        dst_embedding_ptr=0,
        dst_deepstack_ptr=0,
        n_tokens=0,
        hidden=4096,
        dtype="torch.float32",
        has_deepstack=False,
        required_dst_info_num=2,
    )
    rt = EmbeddingTransferInfo.from_zmq(info.to_zmq())
    assert rt == info
    assert rt.has_deepstack is False


def test_pointer_encoding_is_full_64bit():
    # Pack/parse the boundary directly through a frame to ensure no truncation.
    info = EmbeddingArgsRegisterInfo(
        room=REGISTER_ROOM_SENTINEL,
        endpoint="h",
        dst_port=2,
        mooncake_session_id="s",
        dst_embedding_ptr=HIGH_PTR,
        dst_deepstack_ptr=0xDEAD_BEEF_CAFE_0000,
    )
    rt = EmbeddingArgsRegisterInfo.from_zmq(info.to_zmq())
    assert rt.dst_embedding_ptr == HIGH_PTR
    assert rt.dst_deepstack_ptr == 0xDEAD_BEEF_CAFE_0000


def test_transfer_info_integer_room_distinct_from_sentinel():
    # The bootstrap thread routes on frame[0]: "None" => register, else int room.
    reg = EmbeddingArgsRegisterInfo(REGISTER_ROOM_SENTINEL, "h", 1, "s", 1, 0).to_zmq()
    xfer = EmbeddingTransferInfo(
        7, "h", 1, "s", 1, 0, 10, 8, "torch.bfloat16", False, 1
    ).to_zmq()
    assert reg[0] == b"None"
    assert xfer[0] == b"7"
    assert xfer[0] != b"None"


# --------------------------------------------------------------------------- #
# MooncakeEmbeddingSender (driven by a fake manager; no Mooncake engine)
# --------------------------------------------------------------------------- #
class _SenderFakeMgr:
    """Minimal stand-in for MooncakeEmbeddingManagerEncode: just the surface the
    sender touches."""

    def __init__(self):
        self.request_status = {}
        self.failure_records = {}
        self.failure_lock = threading.Lock()
        self.bootstrap_time_out = 300
        self.added = []  # (room, EmbeddingChunk)

    def update_status(self, room, status):
        self.request_status[room] = status

    def check_status(self, room):
        return self.request_status.get(room, TransferPoll.Bootstrapping)

    def record_failure(self, room, reason):
        self.failure_records[room] = reason

    def add_transfer_request(self, room, chunk):
        self.added.append((room, chunk))


def test_sender_failure_exception_raises_and_clears():
    mgr = _SenderFakeMgr()
    s = MooncakeEmbeddingSender(mgr, "h:9", bootstrap_room=3)
    mgr.failure_records[3] = "boom on rank 1"
    with pytest.raises(EmbeddingTransferError) as exc_info:
        s.failure_exception()
    assert exc_info.value.bootstrap_room == 3
    assert exc_info.value.failure_reason == "boom on rank 1"
    assert exc_info.value.remote_endpoint == "h:9"
    assert str(exc_info.value) == (
        "EmbeddingTransferError(bootstrap_room=3, remote_endpoint=h:9): "
        "boom on rank 1"
    )
    assert 3 not in mgr.request_status  # cleared
    assert s.conclude_state == TransferPoll.Failed


# --- shard_rows: the single source of shard geometry for both sides ---


@pytest.mark.parametrize("span", [1, 2, 3, 7, 8, 128, 129])
@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_shard_rows_tiles_span_exactly(span, size):
    cursor = 0
    for rank in range(size):
        start, count = shard_rows(span, rank, size)
        assert start == cursor  # contiguous, in rank order
        assert count >= 0
        cursor += count
    assert cursor == span  # disjoint cover of [0, span)
    counts = [shard_rows(span, r, size)[1] for r in range(size)]
    assert max(counts) - min(counts) <= 1  # balanced


# --- wire frame: round-trip + malformed input ---


def _info(**overrides) -> EmbeddingTransferInfo:
    base = dict(
        room=7,
        endpoint="1.2.3.4",
        dst_port=5555,
        mooncake_session_id="s",
        dst_embedding_ptr=0x1000,
        dst_deepstack_ptr=0,
        n_tokens=10,
        hidden=64,
        dtype="torch.bfloat16",
        has_deepstack=False,
        required_dst_info_num=4,
    )
    base.update(overrides)
    return EmbeddingTransferInfo(**base)


# --- encode-side payload math + fanout validation ---


def _chunk(n_tokens=10, hidden=64, deepstack_width=0) -> EmbeddingChunk:
    itemsize = 2  # bf16
    return EmbeddingChunk(
        room=7,
        src_embedding_ptr=0x10_0000,
        n_tokens=n_tokens,
        hidden=hidden,
        dtype="torch.bfloat16",
        nbytes=n_tokens * hidden * itemsize,
        src_deepstack_ptr=0x20_0000 if deepstack_width else 0,
        deepstack_width=deepstack_width,
        deepstack_nbytes=(
            n_tokens * deepstack_width * itemsize if deepstack_width else 0
        ),
    )


def test_shard_payload_identity_is_whole_chunk():
    chunk = _chunk(n_tokens=10, deepstack_width=128)
    src, nbytes, deep_src, deep_nbytes = shard_payload(chunk, _info(n_tokens=10))
    assert (src, nbytes) == (chunk.src_embedding_ptr, chunk.nbytes)
    assert (deep_src, deep_nbytes) == (chunk.src_deepstack_ptr, chunk.deepstack_nbytes)


def test_shard_payload_offsets_rows():
    chunk = _chunk(n_tokens=10, hidden=64, deepstack_width=128)
    row_bytes = chunk.nbytes // 10
    deep_row_bytes = chunk.deepstack_nbytes // 10
    src, nbytes, deep_src, deep_nbytes = shard_payload(
        chunk, _info(n_tokens=3, row_start=5)
    )
    assert src == chunk.src_embedding_ptr + 5 * row_bytes
    assert nbytes == 3 * row_bytes
    assert deep_src == chunk.src_deepstack_ptr + 5 * deep_row_bytes
    assert deep_nbytes == 3 * deep_row_bytes


def test_validate_rejects_gap_overlap_range_dtype():
    chunk = _chunk(n_tokens=10)
    gap = [
        _info(n_tokens=3, row_start=0, span=10),
        _info(n_tokens=3, row_start=5, span=10),
    ]
    overlap = [
        _info(n_tokens=5, row_start=0, span=10),
        _info(n_tokens=5, row_start=3, span=10),
    ]
    out_of_range = [_info(n_tokens=6, row_start=5, span=10)]
    bad_dtype = [_info(n_tokens=10, dtype="torch.float32", span=10)]
    mixed = [_info(n_tokens=10, span=10), _info(n_tokens=5, row_start=5, span=10)]
    for infos in (gap, overlap, out_of_range, bad_dtype, mixed):
        assert validate_fanout_frames(infos, chunk) is not None


def test_validate_rejects_span_mismatch_even_single_frame():
    # The G2 token-count tripwire: a cross-side row-count divergence (image
    # processor / grid contract / cache-key bug) must fail LOUD even at
    # fanout == 1, where a lone frame has no tiling partner to expose it.
    chunk = _chunk(n_tokens=10)
    # span carries the receiver's full image span -> direct check.
    under = [_info(n_tokens=6, row_start=0, span=6)]
    assert validate_fanout_frames(under, chunk) is not None
    # matching span passes.
    assert validate_fanout_frames([_info(n_tokens=10, span=10)], chunk) is None


def test_validate_rejects_deepstack_presence_mismatch():
    # An encode chunk that lost its deepstack half (e.g. a cache hit cached
    # without it) must fail loud, not push Success while the receiver
    # publishes a never-written deepstack buffer; and vice versa.
    plain_chunk = _chunk(n_tokens=10)
    deep_chunk = _chunk(n_tokens=10, deepstack_width=128)
    wants_deep = [
        _info(n_tokens=10, span=10, has_deepstack=True, dst_deepstack_ptr=0x2000)
    ]
    no_deep = [_info(n_tokens=10, span=10)]
    assert validate_fanout_frames(wants_deep, plain_chunk) is not None
    assert validate_fanout_frames(no_deep, deep_chunk) is not None
    assert validate_fanout_frames(wants_deep, deep_chunk) is None


# --- receiver job: PER-ITEM shard placement + reassembly schedule ---


class _ShardFakeEngine:
    def register(self, *a, **k):
        pass

    def deregister(self, *a, **k):
        pass


class _ShardFakeMgr:
    def __init__(self):
        self.engine = _ShardFakeEngine()


class _ShardReceiver:
    """Fake receiver: Bootstrapped until pre_alloc, Success after (mirrors
    test_encode_receiver's _FakeReceiver)."""

    created: list["_ShardReceiver"] = []

    def __init__(self, manager, addr, room):
        self.addr = addr
        self.room = room
        self._pre_alloced = False
        self.pre_alloc_kwargs = None
        _ShardReceiver.created.append(self)

    def poll(self):
        return TransferPoll.Success if self._pre_alloced else TransferPoll.Bootstrapped

    def pre_alloc(self, **kwargs):
        self._pre_alloced = True
        self.pre_alloc_kwargs = kwargs


def _shard_setup(monkeypatch):
    import tokenspeed.runtime.epd.prefill_admission as er

    _ShardReceiver.created.clear()
    # Pin the LEGACY per-request buffer path: the pointer-math assertions below
    # compare pre_alloc destinations against item.encoded, which on the pooled
    # path is a post-DONE CLONE at a different address. Pool+shard interplay is
    # covered separately by test_pool_shard_reassembles_into_published_clone.
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOTS", "0")
    er._POOLS.clear()


def _shard_item(span, *, room=100, offsets=None):
    # ONE item == one EPD image of `span` concatenated-subgrid tokens, carrying
    # its per-item encode handshake. Multi-subgrid offsets still sum to `span` and
    # are received/sharded as one image over one room (the encode worker
    # concatenates the subgrids and row-splits the item's full embedding).
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets if offsets is not None else [(0, span - 1)],
    )
    item.encode_handshake = {
        "bootstrap_room": room,
        "bootstrap_host": "h",
        "bootstrap_port": 1,
    }
    return item


def _start(items, shard_rank, shard_size, factory=_ShardReceiver, num_deepstack=0):
    return start_embedding_receive(
        items,
        manager=_ShardFakeMgr(),
        hidden=8,
        num_deepstack=num_deepstack,
        dtype=torch.float32,
        device="cpu",
        receiver_factory=factory,
        shard_rank=shard_rank,
        shard_size=shard_size,
    )


def test_packed_to_full_scatters_shard_rows_to_absolute_offsets():
    # The publish scatter is the silent-corruption-critical mapping: packed shard
    # rows -> their absolute offsets in the item's full embedding. rank 1 of 2 on
    # a span-10 item: packed rows [0,5) -> full rows [5,10). Non-owned rows are
    # left for reassemble (asserted only on the owned rows here).
    import types

    job = _start([_shard_item(10)], shard_rank=1, shard_size=2)
    it = types.SimpleNamespace(n_tokens=10, spans=[10], row_starts=[5], row_counts=[5])
    packed = torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8)
    full = job._packed_to_full(it, packed, 8)
    assert full.shape == (10, 8)
    assert torch.equal(full[5:10], packed[0:5])


def _record_broadcasts(monkeypatch):
    calls = []

    def fake_broadcast(tensor, src=None, group=None):
        calls.append((tensor, src))

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    return calls


def test_reassemble_broadcasts_item_subranges(monkeypatch):
    # The reassembly schedule follows the item's shard tiling: each rank
    # broadcasts the contiguous row range it owns, together covering [0, span)
    # exactly once (one item == one image under per-item rooms).
    item = _shard_item(10)
    job = _start([item], shard_rank=0, shard_size=2)
    assert job.poll() == DONE
    calls = _record_broadcasts(monkeypatch)
    job.reassemble(nccl_group="g", group_ranks=(7, 9))

    itemsize = item.encoded.element_size()
    base = item.encoded.data_ptr()
    rows = [
        ((t.data_ptr() - base) // (8 * itemsize), t.shape[0], src) for t, src in calls
    ]
    # span 10, 2-way: rank0 rows [0,5) from global 7, rank1 rows [5,10) from 9.
    assert rows == [(0, 5, 7), (5, 5, 9)]


# --- encode worker: concluded-sender sweep (the O1 hygiene rider) ---


class _FakeReceiver:
    """Drives the poll state machine without any transport: Bootstrapped until
    pre_alloc, Success after. Records the pre_alloc kwargs for assertions."""

    created: list["_FakeReceiver"] = []

    def __init__(self, manager, addr, room):
        self.manager = manager
        self.addr = addr
        self.room = room
        self._pre_alloced = False
        self.pre_alloc_kwargs = None
        _FakeReceiver.created.append(self)

    def poll(self):
        return TransferPoll.Success if self._pre_alloced else TransferPoll.Bootstrapped

    def pre_alloc(self, **kwargs):
        self._pre_alloced = True
        self.pre_alloc_kwargs = kwargs


class _RecvFakeEngine:
    """No-op Mooncake engine: receive_encoded_embeddings register/deregisters each
    receive buffer (a real RDMA NIC needs pre-registered targets); on CPU there is
    nothing to register."""

    def register(self, *args, **kwargs):
        pass

    def deregister(self, *args, **kwargs):
        pass


class _RecvFakeMgr:
    def __init__(self):
        self.engine = _RecvFakeEngine()


def _recv_item(n_tokens: int) -> MultimodalDataItem:
    # _item_token_count sums (end - start + 1) over offsets; one subgrid here.
    return MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, n_tokens - 1)])


def _epd(item: MultimodalDataItem, *, room: int, host: str, port: int):
    """Attach an EPD encode->prefill handshake onto an item (one room per item)."""
    item.encode_handshake = {
        "bootstrap_room": room,
        "bootstrap_host": host,
        "bootstrap_port": port,
    }
    return item


def _recv_setup(monkeypatch):
    import tokenspeed.runtime.epd.prefill_admission as er

    _FakeReceiver.created.clear()
    # Small pool defaults so each test's fresh fake engine doesn't allocate
    # the production 16x256MB region; pools are keyed by engine identity, so
    # clear them (and the lazy-dereg queue) between tests.
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOTS", "4")
    monkeypatch.setenv("TOKENSPEED_EPD_RECV_POOL_SLOT_MB", "1")
    er._POOLS.clear()
    er._pending_dereg.clear()


def test_recv_pool_release_waits_for_clone_event(monkeypatch):
    import tokenspeed.runtime.epd.prefill_admission as er

    class _FakeEvent:
        def __init__(self):
            self.ready = False

        def query(self):
            return self.ready

    event = _FakeEvent()
    monkeypatch.setattr(er, "_record_current_stream_event", lambda _tensor: event)
    pool = er._RecvBufferPool(_RecvFakeEngine(), "cpu", slot_bytes=64, n_slots=1)

    slot = pool.lease(8)
    assert slot == 0
    pool.release_after_copy(slot, torch.empty(1))
    assert pool.lease(8) is None

    event.ready = True
    assert pool.lease(8) == 0


def test_receive_sizes_buffers_per_item_no_deepstack():
    items = [
        _epd(_recv_item(6), room=11, host="h0", port=7001),
        _epd(_recv_item(4), room=22, host="h1", port=7002),
    ]
    receive_encoded_embeddings(
        items,
        manager=_RecvFakeMgr(),
        hidden=2048,
        num_deepstack=0,
        dtype=torch.bfloat16,
        device="cpu",
        receiver_factory=_FakeReceiver,
    )

    assert items[0].encoded.shape == (6, 2048)
    assert items[1].encoded.shape == (4, 2048)
    assert items[0].encoded.dtype == torch.bfloat16
    assert items[0].encoded_deepstack is None
    assert items[1].encoded_deepstack is None

    # Each receiver pre_alloc'd a buffer matching its item's token count, with
    # the dtype string the encode side asserts against (str(torch.bfloat16)).
    r0 = next(r for r in _FakeReceiver.created if r.room == 11)
    assert r0.pre_alloc_kwargs["n_tokens"] == 6
    assert r0.pre_alloc_kwargs["hidden"] == 2048
    assert r0.pre_alloc_kwargs["dtype"] == "torch.bfloat16"
    assert r0.pre_alloc_kwargs["has_deepstack"] is False
    assert r0.pre_alloc_kwargs["dst_deepstack_ptr"] == 0
    assert r0.addr == "h0:7001"


def test_receive_allocates_deepstack_columns():
    items = [_epd(_recv_item(5), room=99, host="h", port=8000)]
    receive_encoded_embeddings(
        items,
        manager=_RecvFakeMgr(),
        hidden=128,
        num_deepstack=3,
        dtype=torch.float32,
        device="cpu",
        receiver_factory=_FakeReceiver,
    )

    assert items[0].encoded.shape == (5, 128)
    assert items[0].encoded_deepstack.shape == (5, 128 * 3)
    r = _FakeReceiver.created[0]
    assert r.pre_alloc_kwargs["has_deepstack"] is True
    assert r.pre_alloc_kwargs["dst_deepstack_ptr"] != 0


def test_receive_skips_already_encoded_item_on_recall():
    # Chunked prefill re-runs receive_encoded_embeddings on the SAME item object
    # across forwards. After the first call sets item.encoded, a second call must
    # skip the item: construct NO new receiver (re-bootstrapping a room already at
    # Success would dead-wait phase-1 for Bootstrapped and time out) and leave the
    # encoded buffer untouched.
    item = _epd(_recv_item(5), room=77, host="h", port=9)
    common = dict(
        manager=_RecvFakeMgr(),
        hidden=16,
        num_deepstack=0,
        dtype=torch.float32,
        device="cpu",
        receiver_factory=_FakeReceiver,
    )

    receive_encoded_embeddings([item], **common)
    assert item.encoded is not None
    assert len(_FakeReceiver.created) == 1  # first forward received it
    first_encoded = item.encoded

    # Second forward (chunked prefill): item.encoded is already set -> skipped.
    receive_encoded_embeddings([item], **common)
    assert len(_FakeReceiver.created) == 1  # no new receiver constructed
    assert item.encoded is first_encoded  # buffer untouched
