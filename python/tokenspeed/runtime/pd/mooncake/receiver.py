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
from typing import Optional

import numpy as np
import numpy.typing as npt
import requests
import zmq

from tokenspeed.runtime.pd.base.conn import (
    KVPoll,
)
from tokenspeed.runtime.pd.mooncake.entities import KVTransferError
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


def _calc(kv_mgr, prefill_parallel_info: PrefillParallelInfo):
    # Currently, we don't allow prefill instance and decode instance to
    # have different TP sizes per DP rank, except for models using MLA.
    prefill_tp_size_per_dp_rank = prefill_parallel_info.prefill_tp_size_per_dp_rank
    local_tp_size_per_dp_rank = kv_mgr.world_size // kv_mgr.dp_size

    if prefill_parallel_info.enable_mla_l1_5_cache:
        assert kv_mgr.is_mla_backend, "PD with  is not yet supported for non-MLA models"
        target_tp_rank = None  # make all tp ranks not dummy rank
        target_tp_ranks = [i for i in range(prefill_tp_size_per_dp_rank)]
        required_dst_info_num = local_tp_size_per_dp_rank
        required_prefill_response_num = prefill_tp_size_per_dp_rank
    elif local_tp_size_per_dp_rank == prefill_tp_size_per_dp_rank:
        target_tp_rank = kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank
        target_tp_ranks = [target_tp_rank]
        required_dst_info_num = 1
        required_prefill_response_num = 1
    elif local_tp_size_per_dp_rank > prefill_tp_size_per_dp_rank:
        assert (
            kv_mgr.is_mla_backend
        ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"
        target_tp_rank = (kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank) // (
            local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
        )
        target_tp_ranks = [target_tp_rank]
        required_dst_info_num = local_tp_size_per_dp_rank // prefill_tp_size_per_dp_rank
        required_prefill_response_num = 1
    else:
        assert (
            kv_mgr.is_mla_backend
        ), "PD with different TP sizes per DP rank is not yet supported for non-MLA models"

        # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks for non MLA models;
        target_tp_ranks = [
            rank
            for rank in range(
                (kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank)
                * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
                (kv_mgr.kv_args.engine_rank % local_tp_size_per_dp_rank + 1)
                * (prefill_tp_size_per_dp_rank // local_tp_size_per_dp_rank),
            )
        ]
        # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
        # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
        # or the KVPoll will never be set correctly
        target_tp_rank = target_tp_ranks[0]
        required_dst_info_num = 1
        required_prefill_response_num = 1

    return (
        target_tp_rank,
        target_tp_ranks,
        required_dst_info_num,
        required_prefill_response_num,
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

        (
            target_tp_rank,
            target_tp_ranks,
            required_dst_info_num,
            required_prefill_response_num,
        ) = _calc(self.kv_mgr, prefill_parallel_info)
        self.required_dst_info_num = required_dst_info_num
        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            required_prefill_response_num
        )
        target_dp_group = self.bootstrap_room % prefill_parallel_info.dp_size
        bootstrap_key = f"{self.bootstrap_addr}_{target_dp_group}_{target_tp_rank}"
        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = self._get_bootstrap_infos(
                target_dp_group, target_tp_rank, target_tp_ranks
            )
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

    def _get_bootstrap_infos(self, target_dp_group, target_tp_rank, target_tp_ranks):
        bootstrap_infos = []
        for _target_tp_rank in target_tp_ranks:
            bootstrap_info = _get_bootstrap_info_from_server(
                self.bootstrap_addr,
                _target_tp_rank,
                target_dp_group,
            )
            if bootstrap_info is not None:
                #  only support MLA for now: select one prefill rank as real rank
                bootstrap_info["is_dummy"] = not bool(
                    _target_tp_rank == target_tp_rank or target_tp_rank is None
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
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        str(self.required_dst_info_num).encode("ascii"),
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
                )
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
