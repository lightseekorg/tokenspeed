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

import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import requests
import zmq

from tokenspeed.runtime.pd.base.conn import (
    KVPoll,
)
from tokenspeed.runtime.pd.mooncake.entities import KVTransferError
from tokenspeed.runtime.pd.transfer_plan import (
    BufferKind,
    BufferLayout,
    ParallelLayout,
    PDTransferPlanner,
    RankTransferPlan,
    encode_transfer_fragments,
)
from tokenspeed.runtime.pd.utils import (
    PageTransferMetadata,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.network import get_local_ip_by_remote

logger = get_colorful_logger(__name__)

from tokenspeed.runtime.pd.mooncake.decode import (
    MooncakeKVManagerDecode,
    PrefillParallelInfo,
)


def _get_prefill_parallel_info_from_server(
    bootstrap_addr,
) -> Optional[PrefillParallelInfo]:
    """Fetch the prefill parallel info from the bootstrap server."""
    try:
        url = f"http://{bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
        response = requests.get(url)
        if response.status_code == 200:
            prefill_parallel_info = response.json()
            return PrefillParallelInfo(
                tp_size=int(prefill_parallel_info["prefill_tp_size"]),
                dp_size=int(prefill_parallel_info["prefill_dp_size"]),
                enable_mla_l1_5_cache=bool(
                    prefill_parallel_info["enable_mla_l1_5_cache"]
                ),
                kv_item_lens=tuple(
                    int(x) for x in prefill_parallel_info.get("kv_item_lens", [])
                ),
                kv_unit_lens=tuple(
                    int(x) for x in prefill_parallel_info.get("kv_unit_lens", [])
                ),
                state_item_lens=tuple(
                    int(x) for x in prefill_parallel_info.get("state_item_lens", [])
                ),
                state_unit_lens=tuple(
                    int(x) for x in prefill_parallel_info.get("state_unit_lens", [])
                ),
            )
        else:
            logger.error(
                "Failed to get prefill parallel info: %s, %s",
                response.status_code,
                response.text,
            )
            return None
    except Exception as e:
        logger.error("Error fetching prefill parallel info from bootstrap: %s", e)
        return None


def _get_bootstrap_info_from_server(bootstrap_addr, engine_rank, target_dp_group):
    """Fetch the bootstrap info from the bootstrap server."""
    try:
        url = f"http://{bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            bootstrap_info = response.json()
            return bootstrap_info
        else:
            logger.error(
                "Failed to get prefill server info: %s, %s",
                response.status_code,
                response.text,
            )
            return None
    except Exception as e:
        logger.error("Error fetching prefill info from bootstrap: %s", e)
        return None


@dataclass(frozen=True)
class ReceiverRoutePlan:
    target_tp_rank: int | None
    target_tp_ranks: tuple[int, ...]
    required_prefill_response_num: int
    default_required_dst_info_num: int
    transfer_plan: RankTransferPlan | None = None
    supports_remote_spec_candidates: bool = True

    def required_dst_info_num_for_tp_rank(self, tp_rank: int) -> int:
        if self.transfer_plan is None:
            return self.default_required_dst_info_num
        return self.transfer_plan.required_dst_info_num_for_prefill_rank(tp_rank)

    def fragments_for_tp_rank(self, tp_rank: int):
        if self.transfer_plan is None:
            return ()
        return self.transfer_plan.fragments_by_prefill_rank.get(tp_rank, ())


def _buffer_kind_for_layer_offset(
    kv_args, layer_index: int, offset_index: int
) -> BufferKind:
    is_draft = layer_index >= getattr(kv_args, "target_layer_num", len(kv_args.offsets))
    if is_draft:
        return BufferKind.DRAFT_K if offset_index == 0 else BufferKind.DRAFT_V
    return BufferKind.TARGET_K if offset_index == 0 else BufferKind.TARGET_V


def _unit_lens_or_default(unit_lens, item_lens):
    if unit_lens:
        return tuple(int(x) for x in unit_lens)
    return tuple(1 for _ in item_lens)


def _build_buffer_layout_pair(
    *,
    buffer_index: int,
    buffer_kind: BufferKind,
    sharded_axis: str,
    prefill_item_len: int,
    decode_item_len: int,
    prefill_unit_len: int,
    decode_unit_len: int,
    prefill_tp_size: int,
    decode_tp_size: int,
):
    if prefill_unit_len != decode_unit_len:
        raise ValueError(
            f"prefill/decode unit sizes differ for {buffer_kind.value}: "
            f"prefill={prefill_unit_len}, decode={decode_unit_len}"
        )
    if prefill_item_len % prefill_unit_len != 0:
        raise ValueError(
            f"prefill item length is not unit-aligned for {buffer_kind.value}: "
            f"item={prefill_item_len}, unit={prefill_unit_len}"
        )
    if decode_item_len % decode_unit_len != 0:
        raise ValueError(
            f"decode item length is not unit-aligned for {buffer_kind.value}: "
            f"item={decode_item_len}, unit={decode_unit_len}"
        )

    prefill_local_units = prefill_item_len // prefill_unit_len
    decode_local_units = decode_item_len // decode_unit_len
    prefill_global_units = prefill_local_units * prefill_tp_size
    decode_global_units = decode_local_units * decode_tp_size
    if prefill_global_units == decode_global_units:
        logical_axis = sharded_axis
        logical_size = decode_global_units
    elif prefill_item_len == decode_item_len:
        logical_axis = "replicated"
        logical_size = decode_local_units
    else:
        raise ValueError(
            f"unsupported heterogeneous TP buffer layout for {buffer_kind.value}: "
            f"prefill_item={prefill_item_len}, decode_item={decode_item_len}, "
            f"prefill_tp={prefill_tp_size}, decode_tp={decode_tp_size}, "
            f"unit={decode_unit_len}"
        )

    return (
        BufferLayout(
            buffer_index=buffer_index,
            buffer_kind=buffer_kind,
            logical_axis=logical_axis,
            logical_size=logical_size,
            page_size=1,
            bytes_per_logical_unit=decode_unit_len,
            item_stride_bytes=prefill_item_len,
        ),
        BufferLayout(
            buffer_index=buffer_index,
            buffer_kind=buffer_kind,
            logical_axis=logical_axis,
            logical_size=logical_size,
            page_size=1,
            bytes_per_logical_unit=decode_unit_len,
            item_stride_bytes=decode_item_len,
        ),
    )


def _build_kv_buffer_layouts(
    kv_args, prefill_parallel_info: PrefillParallelInfo, decode_tp_size: int
):
    prefill_tp_size = prefill_parallel_info.prefill_tp_size_per_dp_rank
    prefill_kv_item_lens = tuple(prefill_parallel_info.kv_item_lens)
    if not prefill_kv_item_lens:
        prefill_kv_item_lens = tuple(
            int(x) * decode_tp_size // prefill_tp_size for x in kv_args.kv_item_lens
        )
    prefill_kv_unit_lens = _unit_lens_or_default(
        prefill_parallel_info.kv_unit_lens, prefill_kv_item_lens
    )
    decode_kv_unit_lens = _unit_lens_or_default(
        getattr(kv_args, "kv_unit_lens", []), kv_args.kv_item_lens
    )

    prefill_buffers = []
    decode_buffers = []
    for layer_index, ptr_offsets in enumerate(kv_args.offsets):
        for offset_index, ptr_offset in enumerate(ptr_offsets):
            buffer_kind = _buffer_kind_for_layer_offset(
                kv_args, layer_index, offset_index
            )
            prefill_buffer, decode_buffer = _build_buffer_layout_pair(
                buffer_index=ptr_offset,
                buffer_kind=buffer_kind,
                sharded_axis="kv_head",
                prefill_item_len=int(prefill_kv_item_lens[ptr_offset]),
                decode_item_len=int(kv_args.kv_item_lens[ptr_offset]),
                prefill_unit_len=int(prefill_kv_unit_lens[ptr_offset]),
                decode_unit_len=int(decode_kv_unit_lens[ptr_offset]),
                prefill_tp_size=prefill_tp_size,
                decode_tp_size=decode_tp_size,
            )
            prefill_buffers.append(prefill_buffer)
            decode_buffers.append(decode_buffer)

    decode_state_item_lens = tuple(getattr(kv_args, "state_item_lens", []) or [])
    prefill_state_item_lens = tuple(prefill_parallel_info.state_item_lens)
    if not prefill_state_item_lens:
        prefill_state_item_lens = tuple(
            int(x) * decode_tp_size // prefill_tp_size for x in decode_state_item_lens
        )
    prefill_state_unit_lens = _unit_lens_or_default(
        prefill_parallel_info.state_unit_lens, prefill_state_item_lens
    )
    decode_state_unit_lens = _unit_lens_or_default(
        getattr(kv_args, "state_unit_lens", []), decode_state_item_lens
    )

    for state_index, decode_item_len in enumerate(decode_state_item_lens):
        prefill_buffer, decode_buffer = _build_buffer_layout_pair(
            buffer_index=state_index,
            buffer_kind=BufferKind.MAMBA_STATE,
            sharded_axis="state_channel",
            prefill_item_len=int(prefill_state_item_lens[state_index]),
            decode_item_len=int(decode_item_len),
            prefill_unit_len=int(prefill_state_unit_lens[state_index]),
            decode_unit_len=int(decode_state_unit_lens[state_index]),
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=decode_tp_size,
        )
        prefill_buffers.append(prefill_buffer)
        decode_buffers.append(decode_buffer)
    return tuple(prefill_buffers), tuple(decode_buffers)


def _build_non_mla_route_plan(kv_mgr, prefill_parallel_info: PrefillParallelInfo):
    prefill_tp_size = prefill_parallel_info.prefill_tp_size_per_dp_rank
    decode_tp_size = kv_mgr.world_size // kv_mgr.dp_size
    decode_tp_rank = kv_mgr.kv_args.engine_rank % decode_tp_size
    prefill_buffers, decode_buffers = _build_kv_buffer_layouts(
        kv_mgr.kv_args,
        prefill_parallel_info,
        decode_tp_size,
    )
    planner = PDTransferPlanner(
        prefill_layout=ParallelLayout(
            role="prefill",
            world_size=prefill_tp_size,
            dp_size=1,
        ),
        decode_layout=ParallelLayout(
            role="decode",
            world_size=decode_tp_size,
            dp_size=1,
        ),
        prefill_buffers=prefill_buffers,
        decode_buffers=decode_buffers,
    )
    transfer_plan = planner.plan_for_decode_rank(decode_tp_rank)
    target_tp_ranks = tuple(transfer_plan.target_prefill_ranks)
    target_tp_rank = (
        target_tp_ranks[0] if transfer_plan.plan_kind == "identity" else None
    )
    default_required_dst_info_num = (
        transfer_plan.required_dst_info_num_for_prefill_rank(target_tp_ranks[0])
        if target_tp_ranks
        else 1
    )
    return ReceiverRoutePlan(
        target_tp_rank=target_tp_rank,
        target_tp_ranks=target_tp_ranks,
        required_prefill_response_num=transfer_plan.required_prefill_response_num,
        default_required_dst_info_num=default_required_dst_info_num,
        transfer_plan=transfer_plan,
        supports_remote_spec_candidates=transfer_plan.plan_kind == "identity",
    )


def _legacy_mla_route_plan(
    *,
    target_tp_rank: int | None,
    target_tp_ranks,
    required_dst_info_num: int,
    required_prefill_response_num: int,
) -> ReceiverRoutePlan:
    return ReceiverRoutePlan(
        target_tp_rank=target_tp_rank,
        target_tp_ranks=tuple(target_tp_ranks),
        required_prefill_response_num=required_prefill_response_num,
        default_required_dst_info_num=required_dst_info_num,
    )


def _calc(kv_mgr, prefill_parallel_info: PrefillParallelInfo) -> ReceiverRoutePlan:
    prefill_tp_size_per_dp_rank = prefill_parallel_info.prefill_tp_size_per_dp_rank
    local_tp_size_per_dp_rank = kv_mgr.world_size // kv_mgr.dp_size

    if prefill_parallel_info.enable_mla_l1_5_cache:
        assert kv_mgr.is_mla_backend, "PD with  is not yet supported for non-MLA models"
        return _legacy_mla_route_plan(
            target_tp_rank=None,
            target_tp_ranks=range(prefill_tp_size_per_dp_rank),
            required_dst_info_num=local_tp_size_per_dp_rank,
            required_prefill_response_num=prefill_tp_size_per_dp_rank,
        )

    if not kv_mgr.is_mla_backend:
        return _build_non_mla_route_plan(kv_mgr, prefill_parallel_info)

    if local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
        target_tp_rank = kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
        return _legacy_mla_route_plan(
            target_tp_rank=target_tp_rank,
            target_tp_ranks=(target_tp_rank,),
            required_dst_info_num=1,
            required_prefill_response_num=1,
        )

    if local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
        target_tp_rank = (kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank) // (
            local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
        )
        return _legacy_mla_route_plan(
            target_tp_rank=target_tp_rank,
            target_tp_ranks=(target_tp_rank,),
            required_dst_info_num=local_tp_size_per_dp_rank
            // prefill_tp_size_per_dp_rank,
            required_prefill_response_num=1,
        )

    target_tp_ranks = tuple(
        range(
            (kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank)
            * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
            (kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank + 1)
            * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
        )
    )
    return _legacy_mla_route_plan(
        target_tp_rank=target_tp_ranks[0],
        target_tp_ranks=target_tp_ranks,
        required_dst_info_num=1,
        required_prefill_response_num=1,
    )


class MooncakeKVReceiver:
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self, mgr: MooncakeKVManagerDecode, bootstrap_addr: str, bootstrap_room: int
    ):
        self.kv_mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room

        self.session_id = self.kv_mgr.get_session_id()
        self.conclude_state = None
        self.init_time = None
        self.prefill_enable_mla_l1_5_cache = None
        self.dst_enable_mla_l1_5_cache = False

        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        logger.info(
            "[MooncakeKVReceiver.__init__] bootstrap_addr=%s bootstrap_room=%s session_id=%s",
            bootstrap_addr,
            bootstrap_room,
            self.session_id,
        )

        prefill_parallel_info = self._get_prefill_parallel_info()
        if prefill_parallel_info is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)

        route_plan = _calc(self.kv_mgr, prefill_parallel_info)
        self.route_plan = route_plan
        self.supports_remote_spec_candidates = (
            route_plan.supports_remote_spec_candidates
        )
        self.required_dst_info_num = route_plan.default_required_dst_info_num
        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            route_plan.required_prefill_response_num
        )
        target_dp_group = self.bootstrap_room % prefill_parallel_info.dp_size
        target_tp_key = ",".join(str(rank) for rank in route_plan.target_tp_ranks)
        bootstrap_key = f"{self.bootstrap_addr}_{target_dp_group}_{target_tp_key}"
        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = self._get_bootstrap_infos(target_dp_group, route_plan)
            if bootstrap_infos is None:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {target_dp_group}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            else:
                assert len(bootstrap_infos) > 0
                self.bootstrap_infos = bootstrap_infos
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos
                # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
                self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapped)
        logger.info(
            "[MooncakeKVReceiver.__init__] done, status set to Bootstrapped. "
            "bootstrap_room=%s bootstrap_addr=%s session_id=%s",
            self.bootstrap_room,
            self.bootstrap_addr,
            self.session_id,
        )

    def _get_prefill_parallel_info(self):
        prefill_parallel_info = self.kv_mgr.prefill_parallel_info.get(
            self.bootstrap_addr
        )

        if prefill_parallel_info is not None:
            return prefill_parallel_info
        else:
            prefill_parallel_info = _get_prefill_parallel_info_from_server(
                self.bootstrap_addr
            )

            if prefill_parallel_info is None:
                return None
            else:
                logger.debug(
                    "Fetch prefill parallel info from [%s]: DP size:%s, TP size:%s",
                    self.bootstrap_addr,
                    prefill_parallel_info.dp_size,
                    prefill_parallel_info.tp_size,
                )
                self.kv_mgr.prefill_parallel_info[self.bootstrap_addr] = (
                    prefill_parallel_info
                )
                return prefill_parallel_info

    def _get_bootstrap_infos(self, target_dp_group, route_plan: ReceiverRoutePlan):
        bootstrap_infos = []
        for _target_tp_rank in route_plan.target_tp_ranks:
            bootstrap_info = _get_bootstrap_info_from_server(
                self.bootstrap_addr,
                _target_tp_rank,
                target_dp_group,
            )
            if bootstrap_info is not None:
                #  only support MLA for now: select one prefill rank as real rank
                bootstrap_info["is_dummy"] = not bool(
                    _target_tp_rank == route_plan.target_tp_rank
                    or route_plan.target_tp_rank is None
                )
                bootstrap_info["required_dst_info_num"] = (
                    route_plan.required_dst_info_num_for_tp_rank(_target_tp_rank)
                )
                bootstrap_info["transfer_fragments"] = route_plan.fragments_for_tp_rank(
                    _target_tp_rank
                )
                logger.debug(
                    "Fetched bootstrap info: %s for DP %s TP %s",
                    bootstrap_info,
                    target_dp_group,
                    _target_tp_rank,
                )
                bootstrap_infos.append(bootstrap_info)
            else:
                return None
        return bootstrap_infos

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            logger.info(
                "[MooncakeKVReceiver._register_kv_args] sending kv_args to prefill=%s bootstrap_room=%s session_id=%s",
                self.prefill_server_url,
                self.bootstrap_room,
                self.session_id,
            )
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_state_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.state_data_ptrs
            )

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        b"",  # aux_data_ptrs removed; kept as empty frame for protocol compat
                        packed_state_data_ptrs,
                        # Include decode_prefix_len for kv_args registration
                        str(getattr(self, "decode_prefix_len", 0)).encode("ascii"),
                    ]
                )

    @classmethod
    def _connect(cls, endpoint: str):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = threading.Lock()
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    def prefill(
        self,
        kv_indices: npt.NDArray[np.int64],
        aux_index: Optional[int] = None,
        decode_prefix_len: Optional[int] = 0,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
        mamba_indices: Optional[npt.NDArray[np.int64]] = None,
    ):
        logger.info(
            "[MooncakeKVReceiver.init] bootstrap_room=%s kv_indices_len=%d aux_index=%s decode_prefix_len=%s",
            self.bootstrap_room,
            len(kv_indices),
            aux_index,
            decode_prefix_len,
        )
        # Store decode_prefix_len to be sent back to prefill
        self.decode_prefix_len = decode_prefix_len
        dst_page_transfer_mask = None
        dst_page_local_indices = None
        if mla_l1_5_args is not None:
            dst_page_transfer_mask = mla_l1_5_args.page_transfer_mask
            dst_page_local_indices = mla_l1_5_args.page_local_indices

        for bootstrap_info in self.bootstrap_infos:
            self.prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            is_dummy = bootstrap_info["is_dummy"]

            logger.info(
                "[MooncakeKVReceiver.init] sending pre-alloc multipart to prefill=%s bootstrap_room=%s is_dummy=%s",
                self.prefill_server_url,
                self.bootstrap_room,
                bootstrap_info["is_dummy"],
            )
            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                message_parts = [
                    str(self.bootstrap_room).encode("ascii"),
                    get_local_ip_by_remote().encode("ascii"),
                    str(self.kv_mgr.rank_port).encode("ascii"),
                    self.session_id.encode("ascii"),
                    kv_indices.tobytes() if not is_dummy else b"",
                    str(aux_index).encode("ascii") if not is_dummy else b"",
                    str(
                        bootstrap_info.get(
                            "required_dst_info_num", self.required_dst_info_num
                        )
                    ).encode("ascii"),
                    # Send decode_prefix_len as additional message part
                    (
                        str(self.decode_prefix_len).encode("ascii")
                        if not is_dummy
                        else b""
                    ),
                    (
                        str(int(self.dst_enable_mla_l1_5_cache)).encode("ascii")
                        if not is_dummy
                        else b""
                    ),
                    (
                        dst_page_transfer_mask.tobytes()
                        if (not is_dummy and dst_page_transfer_mask is not None)
                        else b""
                    ),
                    (
                        dst_page_local_indices.tobytes()
                        if (not is_dummy and dst_page_local_indices is not None)
                        else b""
                    ),
                    (
                        mamba_indices.tobytes()
                        if (not is_dummy and mamba_indices is not None)
                        else b""
                    ),
                ]
                transfer_fragments = bootstrap_info.get("transfer_fragments", ())
                if not is_dummy and transfer_fragments:
                    message_parts.extend(encode_transfer_fragments(transfer_fragments))
                sock.send_multipart(message_parts)
            self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed
            elif status == KVPoll.Transferring:
                logger.warning(
                    "Req(room=%s) in Transferring, which is unexpected",
                    self.bootstrap_room,
                )

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
            self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
            self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason, self.bootstrap_addr)
