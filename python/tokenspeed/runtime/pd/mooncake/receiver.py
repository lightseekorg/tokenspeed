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

import struct
import threading
import time
from dataclasses import dataclass

import requests
import zmq

from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    CachePDBlockManifest,
    CacheTransferContract,
    validate_cache_manifest,
)
from tokenspeed.runtime.pd.mooncake.entities import KVTransferError
from tokenspeed.runtime.pd.transfer_plan import (
    CacheTransferPlanner,
    RankTransferPlan,
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
) -> PrefillParallelInfo | None:
    """Fetch the prefill parallel info from the bootstrap server."""
    try:
        url = f"http://{bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}"
        response = requests.get(url)
        if response.status_code == 200:
            prefill_parallel_info = response.json()
            cache_layout_wire = prefill_parallel_info.get("cache_layout")
            cache_layout = (
                CacheTransferContract.from_wire_bytes(cache_layout_wire.encode("ascii"))
                if cache_layout_wire is not None
                else None
            )
            return PrefillParallelInfo(
                tp_size=int(prefill_parallel_info["prefill_tp_size"]),
                dp_size=int(prefill_parallel_info["prefill_dp_size"]),
                cache_layout=cache_layout,
            )
        else:
            logger.error(
                "Failed to get prefill parallel info: %s, %s",
                response.status_code,
                response.text,
            )
            return None
    except Exception as exc:
        logger.error("Error fetching prefill parallel info from bootstrap: %s", exc)
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
    except Exception as exc:
        logger.error("Error fetching prefill info from bootstrap: %s", exc)
        return None


@dataclass(frozen=True)
class ReceiverRoutePlan:
    transfer_plan: RankTransferPlan
    dummy_tp_ranks: tuple[int, ...] = ()

    @property
    def target_tp_ranks(self) -> tuple[int, ...]:
        return tuple(
            sorted((*self.transfer_plan.target_prefill_ranks, *self.dummy_tp_ranks))
        )

    def is_dummy_tp_rank(self, tp_rank: int) -> bool:
        return tp_rank in self.dummy_tp_ranks


def _calc(kv_mgr, prefill_parallel_info: PrefillParallelInfo) -> ReceiverRoutePlan:
    prefill_tp_size_per_dp_rank = prefill_parallel_info.prefill_tp_size_per_dp_rank
    local_tp_size_per_dp_rank = kv_mgr.topology.tp_size
    local_cache_layout = kv_mgr.kv_args.cache_layout
    prefill_cache_layout = prefill_parallel_info.cache_layout

    if prefill_cache_layout is None:
        raise RuntimeError("Paged cache decode connected to a non-Paged cache prefill")
    decode_tp_rank = kv_mgr.topology.tp_rank
    planner = CacheTransferPlanner(
        prefill_tp_size=prefill_tp_size_per_dp_rank,
        decode_tp_size=local_tp_size_per_dp_rank,
        prefill_layout=prefill_cache_layout,
        decode_layout=local_cache_layout,
    )
    transfer_plan = planner.plan_for_decode_rank(decode_tp_rank)
    dummy_tp_ranks = ()
    if decode_tp_rank == 0:
        dummy_tp_ranks = tuple(
            rank
            for rank, decode_ranks in planner.decode_ranks_by_prefill_rank.items()
            if not decode_ranks
        )
    return ReceiverRoutePlan(
        transfer_plan=transfer_plan,
        dummy_tp_ranks=dummy_tp_ranks,
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

        self.kv_mgr.update_status(self.bootstrap_room, TransferPoll.Bootstrapping)
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
            self.kv_mgr.update_status(self.bootstrap_room, TransferPoll.Failed)
            return

        route_plan = _calc(self.kv_mgr, prefill_parallel_info)
        self.route_plan = route_plan
        self.kv_mgr.expected_prefill_ranks_table[self.bootstrap_room] = frozenset(
            route_plan.transfer_plan.target_prefill_ranks
        )
        target_dp_group = self.bootstrap_room % prefill_parallel_info.dp_size
        target_tp_key = ",".join(str(rank) for rank in route_plan.target_tp_ranks)
        bootstrap_key = f"{self.bootstrap_addr}_{target_dp_group}_{target_tp_key}"
        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = self._get_bootstrap_infos(target_dp_group, route_plan)
            if not bootstrap_infos:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {target_dp_group}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, TransferPoll.Failed)
                return
            else:
                self.bootstrap_infos = bootstrap_infos
                self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos
                # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
                self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, TransferPoll.Bootstrapped)
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
                # Control-only rendezvous ranks participate in Prefill TP
                # status collectives but do not receive cache data.
                bootstrap_info["is_dummy"] = route_plan.is_dummy_tp_rank(
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
            packed_kv_data_ptr = struct.pack("Q", self.kv_mgr.kv_args.kv_data_ptr)
            cache_layout = self.kv_mgr.kv_args.cache_layout
            decode_tp_size = self.kv_mgr.topology.tp_size
            decode_tp_rank = self.kv_mgr.topology.tp_rank

            sock, lock = self._connect("tcp://" + self.prefill_server_url)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        get_local_ip_by_remote().encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptr,
                        cache_layout.to_wire_bytes(),
                        str(decode_tp_size).encode("ascii"),
                        str(decode_tp_rank).encode("ascii"),
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
        block_manifest: CachePDBlockManifest | None = None,
    ):
        logger.info(
            "[MooncakeKVReceiver.init] bootstrap_room=%s",
            self.bootstrap_room,
        )
        cache_layout = self.kv_mgr.kv_args.cache_layout
        if block_manifest is None:
            raise ValueError("Mooncake PD requires a block manifest")
        validate_cache_manifest(
            block_manifest,
            layout=cache_layout,
            peer="destination",
        )
        self.init_time = time.time()
        self.kv_mgr.update_status(self.bootstrap_room, TransferPoll.WaitingForInput)

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
                    self.session_id.encode("ascii"),
                    block_manifest.to_wire_bytes() if not is_dummy else b"",
                ]
                sock.send_multipart(message_parts)

    def poll(self) -> TransferPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (TransferPoll.Success, TransferPoll.Failed):
                self.conclude_state = status
            elif status == TransferPoll.WaitingForInput:
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
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in TransferPoll.WaitingForInput",
                        )
                        self.conclude_state = TransferPoll.Failed
                        return TransferPoll.Failed
            elif status == TransferPoll.Transferring:
                logger.warning(
                    "Req(room=%s) in Transferring, which is unexpected",
                    self.bootstrap_room,
                )

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room, None)
        self.kv_mgr.expected_prefill_ranks_table.pop(self.bootstrap_room, None)
        self.kv_mgr.bootstrap_token_table.pop(self.bootstrap_room, None)
        self.kv_mgr.spec_candidate_ids_table.pop(self.bootstrap_room, None)
        self.kv_mgr._pending_bootstrap_token_table.pop(self.bootstrap_room, None)
        self.kv_mgr._pending_spec_candidate_ids_table.pop(self.bootstrap_room, None)
        with self.kv_mgr.failure_lock:
            self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        with self.kv_mgr.connection_lock:
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = TransferPoll.Failed

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.get(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        self.clear()
        raise KVTransferError(self.bootstrap_room, failure_reason, self.bootstrap_addr)
