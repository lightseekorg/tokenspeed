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

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, Union

import numpy as np
import requests

from tokenspeed.runtime.pd.base.conn import (
    KVArgs,
    KVPoll,
)
from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVManagerBase
from tokenspeed.runtime.pd.mooncake.entities import ManagerArgs
from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.network import get_free_port, get_local_ip_by_remote

logger = get_colorful_logger(__name__)


@dataclass
class PrefillParallelInfo:
    tp_size: int
    dp_size: int
    enable_mla_l1_5_cache: bool
    kv_item_lens: tuple[int, ...] = ()
    kv_unit_lens: tuple[int, ...] = ()
    state_item_lens: tuple[int, ...] = ()
    state_unit_lens: tuple[int, ...] = ()

    @property
    def prefill_tp_size_per_dp_rank(self):
        return self.tp_size // self.dp_size


def parse_prefill_status_message(
    parts: list[bytes],
) -> tuple[int, int, int, int, list[int] | None]:
    bootstrap_room = int(parts[0].decode("ascii"))
    status = int(parts[1].decode("ascii"))
    prefill_rank = int(parts[2].decode("ascii"))
    bootstrap_token = int(parts[3].decode("ascii")) if len(parts) > 3 else -1
    spec_candidate_ids = None
    if len(parts) > 4 and parts[4] != b"":
        spec_candidate_ids = np.frombuffer(parts[4], dtype=np.int32).copy().tolist()
    return bootstrap_room, status, prefill_rank, bootstrap_token, spec_candidate_ids


class MooncakeKVManagerDecode(MooncakeKVManagerBase):
    def __init__(
        self,
        args: ManagerArgs,
        kv_args: KVArgs,
    ):
        super().__init__(args, kv_args, DisaggregationMode.DECODE)

        self.heartbeat_failures = {}
        self.session_pool = defaultdict(requests.Session)
        self.session_pool_lock = threading.Lock()
        self.addr_to_rooms_tracker = defaultdict(set)
        self.connection_lock = threading.Lock()
        # Heartbeat interval should be at least 2 seconds
        self.heartbeat_interval = max(
            envs.TOKENSPEED_DISAGGREGATION_HEARTBEAT_INTERVAL.get(),
            2.0,
        )
        # Heartbeat failure should be at least 1
        self.max_failures = max(
            envs.TOKENSPEED_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
        )
        self.start_decode_thread()
        self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
        self.required_prefill_response_num_table: Dict[int, int] = {}
        self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)

        self.prefill_parallel_info: Dict[str, PrefillParallelInfo] = {}

        # If a timeout happens on the decode side, it means decode instances
        # fail to receive the KV Cache transfer done signal after bootstrapping.
        # These timeout requests should be aborted to release the tree cache.
        self.waiting_timeout = envs.TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT.get()

    def start_decode_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")
        # Maps bootstrap_room -> bootstrap_token (first output token from prefill).
        # Populated by decode_thread when a Success message carries a valid token,
        # consumed by DisaggDecodeExecutor.generate_events() via pop_bootstrap_token().
        self.bootstrap_token_table: Dict[int, int] = {}
        self.spec_candidate_ids_table: Dict[int, list[int]] = {}
        self._pending_bootstrap_token_table: Dict[int, int] = {}
        self._pending_spec_candidate_ids_table: Dict[int, list[int]] = {}

        def decode_thread():
            while True:
                parts = self.server_socket.recv_multipart()
                self._handle_prefill_status(*parse_prefill_status_message(parts))

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_parallel_info.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the map
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                "Attempting to reconnect to %s...", bootstrap_addr
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info("Attempting to reconnect to %s...", bootstrap_addr)
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def _handle_prefill_status(
        self,
        bootstrap_room: int,
        status: int,
        prefill_rank: int,
        bootstrap_token: int,
        spec_candidate_ids: list[int] | None,
    ) -> None:
        if status == KVPoll.Success:
            if bootstrap_room not in self.request_status:
                return

            self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
            if bootstrap_token != -1:
                self._pending_bootstrap_token_table.setdefault(
                    bootstrap_room, bootstrap_token
                )
            if spec_candidate_ids is not None:
                self._pending_spec_candidate_ids_table.setdefault(
                    bootstrap_room, spec_candidate_ids
                )

            expected_response_num = self.required_prefill_response_num_table.get(
                bootstrap_room, 1
            )
            arrived_response_num = len(self.prefill_response_tracker[bootstrap_room])
            if arrived_response_num < expected_response_num:
                return

            # Store metadata before marking Success so generate_events() can read
            # it atomically in the same iteration. In heterogeneous TP, the rank
            # carrying bootstrap metadata is not guaranteed to be the last rank
            # whose transfer completes, so preserve the first valid value seen.
            if bootstrap_room in self._pending_bootstrap_token_table:
                self.bootstrap_token_table[bootstrap_room] = (
                    self._pending_bootstrap_token_table.pop(bootstrap_room)
                )
            if bootstrap_room in self._pending_spec_candidate_ids_table:
                self.spec_candidate_ids_table[bootstrap_room] = (
                    self._pending_spec_candidate_ids_table.pop(bootstrap_room)
                )
            self.update_status(bootstrap_room, KVPoll.Success)
            return

        if status == KVPoll.Failed:
            self.record_failure(
                bootstrap_room,
                "Failed to get kvcache from prefill instance, it might be dead",
            )
            self.update_status(bootstrap_room, KVPoll.Failed)
            return

        self.update_status(bootstrap_room, status)

    def pop_bootstrap_token(self, bootstrap_room: int) -> int:
        """Pop and return the bootstrap_token for the given room, or -1 if absent."""
        return self.bootstrap_token_table.pop(bootstrap_room, -1)

    def pop_prefill_metadata(self, bootstrap_room: int) -> tuple[int, list[int] | None]:
        return (
            self.bootstrap_token_table.pop(bootstrap_room, -1),
            self.spec_candidate_ids_table.pop(bootstrap_room, None),
        )

    def get_session_id(self):
        return self.engine.get_session_id()

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_parallel_info:
                del self.prefill_parallel_info[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            "Losing connection with prefill instance (bootstrap_addr: %s), affected %s requests",
            failed_bootstrap_addr,
            len(affected_rooms),
        )
