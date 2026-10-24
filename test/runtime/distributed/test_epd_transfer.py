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

"""EPD encode->prefill transfer, the encode side: wire-codec round-trips and
the row-shard payload math. Pure-logic silent-corruption guards (the RDMA
write itself is unchecked); no GPU, no live Mooncake hop. The prefill side's
receiver is the host's (``ts-pd``'s ``epd::receiver``)."""

from __future__ import annotations

import threading

import pytest

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
from tokenspeed.runtime.pd.base.status import TransferPoll

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
