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

import dataclasses
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# Sentinel room (on-wire ascii) for the one-shot registration frame; per-request
# pre-alloc frames carry an integer room.
REGISTER_ROOM_SENTINEL = "None"


def _pack_ptr(ptr: int) -> bytes:
    """Encode a device virtual address as 8 little-endian bytes (uint64)."""
    return struct.pack("Q", ptr)


def _unpack_ptr(frame: bytes) -> int:
    return struct.unpack("Q", frame)[0]


def _b(value: object) -> bytes:
    return str(value).encode("ascii")


def _bool_frame(value: bool) -> bytes:
    return b"1" if value else b"0"


def _parse_bool(frame: bytes) -> bool:
    return frame == b"1"


@dataclasses.dataclass
class EmbeddingArgs:
    """Per-rank buffer/engine info for an embedding endpoint.

    On the encode (sender) side the buffers hold the vision tower's output; on
    the prefill (receiver) side they are the pre-registered receive buffers. A
    ``deepstack`` buffer is present only for models that emit deepstack
    embeddings (e.g. Qwen3.5); ``0`` means absent.
    """

    engine_rank: int
    gpu_id: int
    ib_device: str | None
    embedding_data_ptr: int
    embedding_data_len: int
    deepstack_data_ptr: int = 0
    deepstack_data_len: int = 0


class EmbeddingTransferError(Exception):
    def __init__(
        self, bootstrap_room: int, failure_reason: str, remote_endpoint: str = None
    ):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason
        self.remote_endpoint = remote_endpoint

    def __str__(self):
        if self.remote_endpoint:
            return (
                "EmbeddingTransferError("
                f"bootstrap_room={self.bootstrap_room}, "
                f"remote_endpoint={self.remote_endpoint}): {self.failure_reason}"
            )
        else:
            return (
                f"EmbeddingTransferError(bootstrap_room={self.bootstrap_room}): "
                f"{self.failure_reason}"
            )


@dataclasses.dataclass
class EmbeddingChunk:
    """One item's embedding queued for transfer (encode-side, in-process only).

    Not serialized to ZMQ: it lives on the encode manager's transfer queue. The
    pointers reference the contiguous ``item.encoded`` / ``item.encoded_deepstack``
    tensors produced by the vision tower.
    """

    room: int
    src_embedding_ptr: int
    n_tokens: int
    hidden: int
    dtype: str
    nbytes: int
    src_deepstack_ptr: int = 0
    deepstack_width: int = 0  # hidden * num_deepstack; 0 == no deepstack
    deepstack_nbytes: int = 0
    # CUDA event recorded on the encode loop's stream after the device copy that
    # filled this chunk's ring slot. The transfer worker waits it before its
    # one-sided RDMA read so the read never races the copy (ViT->send corruption
    # hazard). None on CPU/no-CUDA.
    copy_event: "torch.cuda.Event | None" = None


@dataclasses.dataclass
class EmbeddingTransferInfo:
    """Receiver(prefill)->sender(encode) per-request pre-allocation frame.

    Tells the encode side where (``dst_embedding_ptr`` already includes this
    request's row offset) and how big a transfer to issue. The RDMA write is
    unchecked, so the encode asserts ``n_tokens``/``hidden``/``dtype`` against
    the tensor it sends, else a mismatch silently writes a truncated/oversized
    region.

    Row sharding: ``row_start`` + ``n_tokens`` (the SHARD's row count) select
    the rows the encode writes at the shard-offset ``dst_embedding_ptr``;
    ``span`` is the image's FULL row count, the cross-side token-count tripwire
    regardless of shard geometry. Identity frames set ``row_start == 0`` and
    ``n_tokens == span``. The encode validates shard geometry per frame: a
    full-span write at a shard-offset pointer would corrupt neighboring image
    rows in the same buffer.
    """

    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_ptr: int
    dst_deepstack_ptr: int
    n_tokens: int
    hidden: int
    dtype: str
    has_deepstack: bool
    required_dst_info_num: int
    row_start: int = 0
    span: int = 0

    def to_zmq(self) -> list[bytes]:
        return [
            _b(self.room),
            _b(self.endpoint),
            _b(self.dst_port),
            _b(self.mooncake_session_id),
            _pack_ptr(self.dst_embedding_ptr),
            _pack_ptr(self.dst_deepstack_ptr),
            _b(self.n_tokens),
            _b(self.hidden),
            _b(self.dtype),
            _bool_frame(self.has_deepstack),
            _b(self.required_dst_info_num),
            _b(self.row_start),
            _b(self.span),
        ]

    @classmethod
    def from_zmq(cls, msg: list[bytes]) -> "EmbeddingTransferInfo":
        # Length-guarded so a truncated frame is logged and dropped by the
        # bootstrap listener instead of mis-parsing.
        if len(msg) < 13:
            raise ValueError(
                f"malformed EmbeddingTransferInfo frame: {len(msg)} parts < 13"
            )
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_ptr=_unpack_ptr(msg[4]),
            dst_deepstack_ptr=_unpack_ptr(msg[5]),
            n_tokens=int(msg[6].decode("ascii")),
            hidden=int(msg[7].decode("ascii")),
            dtype=msg[8].decode("ascii"),
            has_deepstack=_parse_bool(msg[9]),
            required_dst_info_num=int(msg[10].decode("ascii")),
            row_start=int(msg[11].decode("ascii")),
            span=int(msg[12].decode("ascii")),
        )


@dataclasses.dataclass
class EmbeddingArgsRegisterInfo:
    """Receiver(prefill)->sender(encode) one-shot endpoint registration.

    Sent once per receiver rank with room == ``REGISTER_ROOM_SENTINEL`` so the
    encode side records where to PUSH completion status. The buffer pointers
    are the receiver's base receive buffers (per-item row offsets are carried
    later, in :class:`EmbeddingTransferInfo`).
    """

    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_ptr: int
    dst_deepstack_ptr: int

    def to_zmq(self) -> list[bytes]:
        return [
            _b(self.room),
            _b(self.endpoint),
            _b(self.dst_port),
            _b(self.mooncake_session_id),
            _pack_ptr(self.dst_embedding_ptr),
            _pack_ptr(self.dst_deepstack_ptr),
        ]

    @classmethod
    def from_zmq(cls, msg: list[bytes]) -> "EmbeddingArgsRegisterInfo":
        return cls(
            room=msg[0].decode("ascii"),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_ptr=_unpack_ptr(msg[4]),
            dst_deepstack_ptr=_unpack_ptr(msg[5]),
        )


@dataclasses.dataclass
class EmbeddingManagerArgs:
    """Connection config for the embedding Mooncake endpoint.

    Embedding transfer has a single DP group today. ``tp_size`` is the number
    of encode/prefill endpoint ranks that participate in that group; the shared
    bootstrap layer still receives it as ``world_size`` with ``dp_size=1``.
    """

    bootstrap_port: int
    tp_size: int
    bootstrap_host: str | None = None
