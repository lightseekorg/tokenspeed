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

import dataclasses
import json
import struct

import numpy as np
import numpy.typing as npt

from tokenspeed.runtime.pd.transfer_plan import (
    TransferFragment,
    decode_transfer_fragments,
)
from tokenspeed.runtime.pd.utils import PageTransferMetadata

_PAGED_CACHE_PAGES_PREFIX = b"tokenspeed-paged-cache-pages-v1:"


@dataclasses.dataclass(frozen=True)
class PagedCachePages:
    page_ids: npt.NDArray[np.int64]
    base_logical_page: int = 0


def paged_cache_pages_from_forward_op(
    op,
    index: int,
    tokens_per_page: dict[str, int] | None = None,
    token_begin: int | None = None,
    token_end: int | None = None,
    include_partial_page: bool = True,
) -> dict[str, PagedCachePages]:
    tables = dict(getattr(op, "paged_cache_block_tables", {}))
    base_offsets = dict(getattr(op, "paged_cache_block_table_base_offsets", {}))
    pages: dict[str, PagedCachePages] = {}
    for group_id, table in tables.items():
        row = np.asarray(table[index], dtype=np.int64)
        row = row[row >= 0]
        offsets = base_offsets.get(group_id)
        base = int(offsets[index]) if offsets is not None else 0
        group_id = str(group_id)
        page_tokens = (tokens_per_page or {}).get(group_id)
        if (
            page_tokens is not None
            and token_begin is not None
            and token_end is not None
        ):
            logical_begin = token_begin // page_tokens
            if include_partial_page:
                logical_end = (token_end + page_tokens - 1) // page_tokens
            else:
                logical_end = token_end // page_tokens
            selected_begin = max(logical_begin, base)
            selected_end = min(logical_end, base + len(row))
            if selected_begin >= selected_end:
                pages[group_id] = PagedCachePages(
                    np.array([], dtype=np.int64), selected_begin
                )
                continue
            row_begin = selected_begin - base
            row = row[row_begin : row_begin + selected_end - selected_begin]
            base = selected_begin
        pages[group_id] = PagedCachePages(row, base)
    return pages


def encode_paged_cache_pages(pages: dict[str, PagedCachePages]) -> bytes:
    payload = {
        group_id: [entry.base_logical_page, entry.page_ids.tolist()]
        for group_id, entry in pages.items()
    }
    return _PAGED_CACHE_PAGES_PREFIX + json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def decode_paged_cache_pages(frame: bytes) -> dict[str, PagedCachePages]:
    if not frame.startswith(_PAGED_CACHE_PAGES_PREFIX):
        return {}
    payload = json.loads(frame[len(_PAGED_CACHE_PAGES_PREFIX) :].decode("utf-8"))
    return {
        str(group_id): PagedCachePages(
            np.asarray(entry[1], dtype=np.int64), int(entry[0])
        )
        for group_id, entry in payload.items()
    }


@dataclasses.dataclass
class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    offsets: list[tuple[int]]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str
    gpu_id: int
    target_layer_num: int
    draft_layer_num: int
    kv_layer_ids: list[int] = dataclasses.field(default_factory=list)
    kv_unit_lens: list[int] = dataclasses.field(default_factory=list)
    kv_group_ids: list[str | None] = dataclasses.field(default_factory=list)
    paged_cache_group_tokens_per_page: dict[str, int] = dataclasses.field(
        default_factory=dict
    )
    state_data_ptrs: list[int] = dataclasses.field(default_factory=list)
    state_data_lens: list[int] = dataclasses.field(default_factory=list)
    state_item_lens: list[int] = dataclasses.field(default_factory=list)
    state_unit_lens: list[int] = dataclasses.field(default_factory=list)
    state_type: str = "none"
    state_layer_ids: list[int] = dataclasses.field(default_factory=list)
    mamba_offsets: list[int] | None = None


class KVTransferError(Exception):
    def __init__(
        self, bootstrap_room: int, failure_reason: str, remote_endpoint: str = None
    ):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason
        self.remote_endpoint = remote_endpoint

    def __str__(self):
        if self.remote_endpoint:
            return f"KVTransferError(bootstrap_room={self.bootstrap_room}, remote_endpoint={self.remote_endpoint}): {self.failure_reason}"
        else:
            return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int64]
    index_slice: slice
    is_last: bool
    prefill_aux_index: int | None
    mla_l1_5_args: PageTransferMetadata | None
    prefill_mamba_indices: npt.NDArray[np.int64] | None = None
    # First token generated by prefill (delivered via ZMQ, not RDMA). -1 = unavailable.
    bootstrap_token: int = -1
    begin_cache_step: int | None = None
    layerwise_interval: int = 1
    wait_for_bootstrap_token: bool = False
    spec_candidate_ids: list[int] | None = None
    prefill_paged_cache_pages: dict[str, PagedCachePages] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class TransferIndexResolution:
    src_indices: npt.NDArray[np.int64]
    dst_indices: npt.NDArray[np.int64]


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int64]
    dst_aux_index: int
    required_dst_info_num: int
    decode_prefix_len: int
    dst_indices_are_local: bool
    dst_page_transfer_mask: npt.NDArray[np.bool_] | None
    dst_page_local_indices: npt.NDArray[np.int64] | None
    dst_page_indices_mapping: npt.NDArray[np.int64] | None
    dst_mamba_indices: npt.NDArray[np.int64] | None
    is_dummy: bool
    transfer_fragments: tuple[TransferFragment, ...] = ()
    dst_paged_cache_pages: dict[str, PagedCachePages] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        transfer_fragments = ()
        dst_paged_cache_pages = decode_paged_cache_pages(msg[-1]) if msg else {}
        if msg[4] == b"" and msg[5] == b"":
            dst_kv_indices = np.array([], dtype=np.int64)
            dst_aux_index = None
            decode_prefix_len = 0
            dst_indices_are_local = False
            dst_page_transfer_mask = None
            dst_page_local_indices = None
            dst_page_indices_mapping = None
            dst_mamba_indices = None
            is_dummy = True
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int64)
            dst_aux_index = int(msg[5].decode("ascii"))
            # decode_prefix_len is now msg[7] (we added it as additional message part)
            decode_prefix_len = int(msg[7].decode("ascii")) if len(msg) > 7 else 0
            dst_indices_are_local = (
                bool(int(msg[8].decode("ascii"))) if len(msg) > 8 else False
            )
            dst_page_transfer_mask = (
                np.frombuffer(msg[9], dtype=np.bool_) if len(msg) > 9 else None
            )
            dst_page_local_indices = (
                np.frombuffer(msg[10], dtype=np.int64) if len(msg) > 10 else None
            )
            dst_page_indices_mapping = (
                np.cumsum(dst_page_transfer_mask) - 1
                if dst_page_transfer_mask is not None
                else None
            )
            dst_mamba_indices = (
                np.frombuffer(msg[11], dtype=np.int64)
                if len(msg) > 11 and msg[11] != b""
                else None
            )
            transfer_fragments = (
                decode_transfer_fragments(msg[12], msg[13]) if len(msg) > 13 else ()
            )
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=int(msg[6].decode("ascii")),
            decode_prefix_len=decode_prefix_len,
            dst_indices_are_local=dst_indices_are_local,
            dst_page_transfer_mask=dst_page_transfer_mask,
            dst_page_local_indices=dst_page_local_indices,
            dst_page_indices_mapping=dst_page_indices_mapping,
            dst_mamba_indices=dst_mamba_indices,
            is_dummy=is_dummy,
            transfer_fragments=transfer_fragments,
            dst_paged_cache_pages=dst_paged_cache_pages,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: list[int]
    decode_prefix_len: int

    @classmethod
    def from_zmq(cls, msg: list[bytes]):
        # Format: room, endpoint, port, session, kv_ptrs, aux_ptrs, state_ptrs, decode_prefix_len.
        # Older senders used msg[6] for decode_prefix_len and omitted state_ptrs.
        has_state_frame = len(msg) >= 8
        if has_state_frame:
            state_frame = msg[6]
            decode_prefix_len = int(msg[7].decode("ascii")) if msg[7] else 0
        else:
            state_frame = b""
            decode_prefix_len = (
                int(msg[6].decode("ascii")) if len(msg) >= 7 and msg[6] else 0
            )

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_state_data_ptrs=list(
                struct.unpack(f"{len(state_frame) // 8}Q", state_frame)
            ),
            decode_prefix_len=decode_prefix_len,
        )


@dataclasses.dataclass
class KVManagerArgs:
    bootstrap_port: int
    dist_init_addr: str

    world_size: int
    dp_size: int
    attn_tp_rank: int
    attn_dp_rank: int

    is_mla_backend: bool
    draft_is_mla_backend: bool

    enable_metrics: bool
    enable_dp_attention: bool
    enable_mla_l1_5_cache: bool

    served_model_name: str
    app_key: str
    metrics_reporters: str
