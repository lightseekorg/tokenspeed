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

import concurrent.futures
import os
import socket
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import requests

from tokenspeed.runtime.pd.base.conn import (
    KVArgs,
    KVPoll,
)
from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVManagerBase
from tokenspeed.runtime.pd.mooncake.entities import (
    KVArgsRegisterInfo,
    ManagerArgs,
    TransferIndexResolution,
    TransferInfo,
    TransferKVChunk,
)
from tokenspeed.runtime.pd.transfer_plan import BufferKind, TransferFragment
from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
    FastQueue,
    PageTransferMetadata,
    StepCounter,
    group_concurrent_contiguous,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.network import (
    get_free_port,
    get_ip,
    get_local_ip_by_remote,
)

logger = get_colorful_logger(__name__)


class MooncakeKVManagerPrefill(MooncakeKVManagerBase):
    def __init__(
        self,
        args: ManagerArgs,
        kv_args: KVArgs,
    ):
        super().__init__(args, kv_args, DisaggregationMode.PREFILL)

        self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
        self.decode_kv_args_table: Dict[str, KVArgsRegisterInfo] = {}
        self.start_prefill_thread()
        self._register_to_bootstrap()
        self.session_failures = defaultdict(int)
        self.failed_sessions: Dict[str, float] = {}
        self.failed_session_ttl = max(
            envs.TOKENSPEED_DISAGGREGATION_FAILED_SESSION_TTL.get(), 0
        )
        self.session_lock = threading.Lock()
        self.kv_layer_ids = list(
            getattr(self.kv_args, "kv_layer_ids", None)
            or range(len(self.kv_args.offsets))
        )
        self.state_layer_ids = list(getattr(self.kv_args, "state_layer_ids", []) or [])
        layer_ids = self.kv_layer_ids + self.state_layer_ids
        self.layer_num = (
            (max(layer_ids) + 1) if layer_ids else len(self.kv_args.offsets)
        )
        self._kv_layer_to_index = {
            layer_id: i
            for i, layer_id in enumerate(self.kv_layer_ids[: len(self.kv_args.offsets)])
        }
        self.layerwise_interval = 1
        self.layerwise_debug = envs.TOKENSPEED_PD_LAYERWISE_DEBUG.get()
        self.step_counter = None
        self.prefill_metadata: Dict[int, Tuple[int, Optional[list[int]]]] = {}
        self.expired_prefill_metadata_rooms: set[int] = set()
        self.bootstrap_token_cond = threading.Condition()
        # Determine the number of threads to use for kv sender
        cpu_count = os.cpu_count()
        transfer_thread_pool_size = (
            envs.TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE.get_set_value_or(
                min(max(4, int(0.75 * cpu_count) // 8), 12)
            )
        )
        transfer_queue_size = envs.TOKENSPEED_DISAGGREGATION_QUEUE_SIZE.get()
        assert transfer_thread_pool_size >= transfer_queue_size, (
            f"The environment variable TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
            f"greater than or equal to TOKENSPEED_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
        )
        self.start_transfer_thread(transfer_thread_pool_size, transfer_queue_size)
        self.bootstrap_time_out = envs.TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()

    def register_layerwise_step_counter(
        self, step_counter: StepCounter, interval: int
    ) -> None:
        self.step_counter = step_counter
        self.layerwise_interval = max(int(interval), 1)

    def reserve_layerwise_cache_steps(self) -> int:
        if self.step_counter is None:
            return 0
        cache_step, _ = self.step_counter.current_step()
        self.step_counter.advance_step(
            delta_cache_step=self.layer_num,
            delta_aux_step=0,
        )
        return cache_step

    def set_prefill_metadata(
        self, room: int, token: int, spec_candidate_ids: list[int] | None = None
    ) -> None:
        with self.bootstrap_token_cond:
            if room in self.expired_prefill_metadata_rooms:
                self.expired_prefill_metadata_rooms.discard(room)
                logger.warning(
                    "Dropping late prefill metadata for expired bootstrap_room=%s",
                    room,
                )
                return
            self.prefill_metadata[room] = (token, spec_candidate_ids)
            self.bootstrap_token_cond.notify_all()

    def discard_expired_metadata_room(self, room: int) -> None:
        """Best-effort cleanup of an expired-room marker; safe to call
        when the room may or may not have ever been added."""
        with self.bootstrap_token_cond:
            self.expired_prefill_metadata_rooms.discard(room)

    def _wait_prefill_metadata(
        self,
        room: Optional[int],
        fallback_token: int,
        fallback_candidate_ids: Optional[list[int]],
    ) -> tuple[int, Optional[list[int]]]:
        if room is None or fallback_token != -1:
            return fallback_token, fallback_candidate_ids
        wait_log_interval = max(envs.TOKENSPEED_PD_PREFILL_METADATA_TIMEOUT.get(), 0.01)
        start_time = time.monotonic()
        next_log_time = start_time + wait_log_interval
        with self.bootstrap_token_cond:
            while room not in self.prefill_metadata:
                if self.request_status.get(room) == KVPoll.Failed:
                    self.expired_prefill_metadata_rooms.add(room)
                    logger.warning(
                        "Prefill metadata unavailable for failed bootstrap_room=%s; using fallback=%s",
                        room,
                        fallback_token,
                    )
                    return fallback_token, fallback_candidate_ids
                now = time.monotonic()
                if now >= next_log_time:
                    logger.debug(
                        "Still waiting for prefill metadata for bootstrap_room=%s after %.2fs",
                        room,
                        now - start_time,
                    )
                    next_log_time = now + wait_log_interval
                self.bootstrap_token_cond.wait(timeout=0.01)
            return self.prefill_metadata.pop(room)

    def _is_session_failed(self, mooncake_session_id: str) -> bool:
        if self.failed_session_ttl <= 0:
            return False
        failed_at = self.failed_sessions.get(mooncake_session_id)
        if failed_at is None:
            return False
        elapsed = time.monotonic() - failed_at
        logger.info(
            "Session %s failed for %.2fs (TTL=%ds).",
            mooncake_session_id,
            elapsed,
            self.failed_session_ttl,
        )
        if elapsed < self.failed_session_ttl:
            return True
        del self.failed_sessions[mooncake_session_id]
        logger.info(
            "Session %s failed TTL expired (%.2fs >= %ds), reset.",
            mooncake_session_id,
            elapsed,
            self.failed_session_ttl,
        )
        return False

    def _mark_session_failed(
        self, mooncake_session_id: str, reason: str = "transfer_failed"
    ) -> None:
        if self.failed_session_ttl <= 0:
            return
        self.failed_sessions[mooncake_session_id] = time.monotonic()
        logger.warning(
            "Session %s marked failed (reason=%s, ttl=%ds).",
            mooncake_session_id,
            reason,
            self.failed_session_ttl,
        )

    def _clear_failed_session(self, mooncake_session_id: str) -> None:
        if mooncake_session_id in self.failed_sessions:
            del self.failed_sessions[mooncake_session_id]
            logger.info(
                "Session %s failed state cleared due to KVArgs registration.",
                mooncake_session_id,
            )
        if mooncake_session_id in self.session_failures:
            del self.session_failures[mooncake_session_id]

    def resolve_transfer_indices(
        self,
        kv_chunk: TransferKVChunk,
        req: TransferInfo,
    ) -> TransferIndexResolution:
        src_indices = kv_chunk.prefill_kv_indices
        dst_indices = req.dst_kv_indices[kv_chunk.index_slice]

        valid_len = min(len(src_indices), len(dst_indices))
        # Fast path: empty transfer chunk. Avoid MLA assertions/index ops on empty payload.
        if valid_len == 0:
            empty = np.array([], dtype=np.int64)
            return TransferIndexResolution(src_indices=empty, dst_indices=empty)

        if valid_len < len(src_indices) or valid_len < len(dst_indices):
            logger.warning(
                "Mismatched transfer indices, truncating to %s (src=%s, dst=%s)",
                valid_len,
                len(src_indices),
                len(dst_indices),
            )
        src_indices = src_indices[:valid_len]
        dst_indices = dst_indices[:valid_len]

        src_mode = self.src_mode
        dst_mode = "ON" if req.dst_indices_are_local else "OFF"
        src_args = kv_chunk.mla_l1_5_args

        # Prefill OFF/Decode OFF: use original indices.
        if src_mode == "OFF" and dst_mode == "OFF":
            return TransferIndexResolution(src_indices, dst_indices)

        # Prefill ON/Decode OFF: prefill-side only has partial kv cache, only send local part.
        if src_mode == "ON" and dst_mode == "OFF":
            assert (
                src_args is not None
            ), "Prefill MLA L1.5 cache is enabled but no transfer metadata provided"
            src_mask = src_args.page_transfer_mask
            src_local = src_args.page_local_indices
            return TransferIndexResolution(
                src_indices=src_local,
                dst_indices=dst_indices[src_mask],
            )

        # Prefill OFF/Decode ON: decode-side only has partial kv cache, only send requested part.
        if src_mode == "OFF" and dst_mode == "ON":
            assert (
                req.dst_page_transfer_mask is not None
            ), "OFF/ON expects decode ownership mask when destination reports local indices."
            dst_mapping = req.dst_page_indices_mapping[kv_chunk.index_slice]
            dst_mask = req.dst_page_transfer_mask[kv_chunk.index_slice]
            dst_mapping = dst_mapping[dst_mask]
            dst_local = req.dst_page_local_indices[dst_mapping]
            return TransferIndexResolution(
                src_indices=src_indices[dst_mask],
                dst_indices=dst_local,
            )

        # Prefill ON/Decode ON: both sides hold partial kv cache, find the intersection part.
        assert (
            src_args is not None
        ), "ON/ON expects prefill metadata (page_transfer_mask/page_local_indices)."
        assert (
            req.dst_page_transfer_mask is not None
        ), "ON/ON expects decode ownership mask when destination uses local index space."

        # src_args.page_transfer_mask is generated from current chunk page_indices,
        # so it is already in chunk-local coordinates.
        src_mask = src_args.page_transfer_mask[:valid_len]
        src_local = src_args.page_local_indices

        # req.dst_page_transfer_mask is request-global and should be sliced by chunk.
        dst_mask = req.dst_page_transfer_mask[kv_chunk.index_slice][:valid_len]

        # Only positions owned by both sides should be transferred.
        common_mask = src_mask & dst_mask

        dst_mapping = req.dst_page_indices_mapping[kv_chunk.index_slice][:valid_len]
        dst_mapping = dst_mapping[common_mask]
        dst_local = req.dst_page_local_indices[dst_mapping]

        src_mapping = np.cumsum(src_args.page_transfer_mask) - 1
        src_mapping = src_mapping[:valid_len]
        src_mapping = src_mapping[common_mask]
        src_local = src_args.page_local_indices[src_mapping]

        return TransferIndexResolution(
            src_indices=src_local,
            dst_indices=dst_local,
        )

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        executor: concurrent.futures.ThreadPoolExecutor,
        transfer_fragments: tuple[TransferFragment, ...] = (),
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        transfer_blocks = self._layer_transfer_blocks(
            dst_ptrs=dst_kv_ptrs,
            src_blocks=prefill_kv_blocks,
            dst_blocks=dst_kv_blocks,
            begin_layer_id=0,
            end_layer_id=self.layer_num,
            transfer_fragments=transfer_fragments,
        )
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def _layer_transfer_blocks(
        self,
        dst_ptrs: list[int],
        src_blocks,
        dst_blocks,
        begin_layer_id: int,
        end_layer_id: int,
        transfer_fragments: tuple[TransferFragment, ...] = (),
    ) -> List[Tuple[int, int, int]]:
        transfer_blocks = []
        fragments_by_buffer: dict[int, list[TransferFragment]] = defaultdict(list)
        for fragment in transfer_fragments:
            if fragment.buffer_kind != BufferKind.MAMBA_STATE:
                fragments_by_buffer[fragment.buffer_index].append(fragment)
        has_fragment_plan = bool(transfer_fragments)

        for layer_id in range(begin_layer_id, end_layer_id):
            for ptr_offset in self.kv_args.offsets[layer_id]:
                src_ptr = self.kv_args.kv_data_ptrs[ptr_offset]
                dst_ptr = dst_ptrs[ptr_offset]
                item_len = self.kv_args.kv_item_lens[ptr_offset]
                buffer_fragments = fragments_by_buffer.get(ptr_offset, ())
                if has_fragment_plan:
                    for fragment in buffer_fragments:
                        for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                            page_count = min(len(prefill_index), len(decode_index))
                            if fragment.page_count is not None:
                                page_count = min(page_count, fragment.page_count)
                            for page_offset in range(page_count):
                                src_addr = (
                                    src_ptr
                                    + int(prefill_index[page_offset])
                                    * fragment.src_page_stride_bytes
                                    + fragment.src_byte_offset
                                )
                                dst_addr = (
                                    dst_ptr
                                    + int(decode_index[page_offset])
                                    * fragment.dst_page_stride_bytes
                                    + fragment.dst_byte_offset
                                )
                                transfer_blocks.append(
                                    (src_addr, dst_addr, fragment.bytes_per_page)
                                )
                    continue

                for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                    src_addr = src_ptr + int(prefill_index[0]) * item_len
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    length = item_len * len(prefill_index)
                    transfer_blocks.append((src_addr, dst_addr, length))
        return transfer_blocks

    def _wait_until_cache_step(self, target_step: int) -> None:
        if self.step_counter is None:
            return
        while True:
            ready_step = self.step_counter.query_ready_cache_step()
            if StepCounter.is_step_ready(ready_step, target_step):
                return
            time.sleep(1e-4)

    def send_mamba_cache(
        self,
        mooncake_session_id: str,
        prefill_mamba_indices: Optional[npt.NDArray[np.int64]],
        dst_state_data_ptrs: list[int],
        dst_mamba_indices: Optional[npt.NDArray[np.int64]],
        begin_layer_id: Optional[int] = None,
        end_layer_id: Optional[int] = None,
        transfer_fragments: tuple[TransferFragment, ...] = (),
    ) -> int:
        if self.kv_args.state_type != "mamba":
            return 0
        state_ptrs = self.kv_args.state_data_ptrs
        state_item_lens = self.kv_args.state_item_lens
        if (
            not state_ptrs
            or not dst_state_data_ptrs
            or prefill_mamba_indices is None
            or dst_mamba_indices is None
        ):
            return 0
        if len(state_ptrs) != len(dst_state_data_ptrs):
            logger.error(
                "Mamba state tensor count mismatch: prefill=%d decode=%d",
                len(state_ptrs),
                len(dst_state_data_ptrs),
            )
            return -1

        if prefill_mamba_indices.shape != dst_mamba_indices.shape:
            if prefill_mamba_indices.size == 1 and dst_mamba_indices.size > 1:
                prefill_mamba_indices = np.full(
                    dst_mamba_indices.shape,
                    int(prefill_mamba_indices[0]),
                    dtype=np.int64,
                )
            else:
                logger.error(
                    "Mamba state slot count mismatch: prefill=%s decode=%s",
                    prefill_mamba_indices.tolist(),
                    dst_mamba_indices.tolist(),
                )
                return -1

        state_items = list(
            enumerate(zip(state_ptrs, dst_state_data_ptrs, state_item_lens))
        )
        if begin_layer_id is not None or end_layer_id is not None:
            begin = 0 if begin_layer_id is None else begin_layer_id
            end = self.layer_num if end_layer_id is None else end_layer_id
            if len(self.state_layer_ids) != len(state_items):
                logger.error(
                    "Mamba state layer id count mismatch: ids=%d tensors=%d",
                    len(self.state_layer_ids),
                    len(state_items),
                )
                return -1
            state_items = [
                item
                for item, layer_id in zip(state_items, self.state_layer_ids)
                if begin <= layer_id < end
            ]
            if not state_items:
                return 0

        valid = (prefill_mamba_indices >= 0) & (dst_mamba_indices >= 0)
        log_layerwise = getattr(self, "layerwise_debug", False)
        if log_layerwise and begin_layer_id is not None and end_layer_id is not None:
            logger.info(
                "[layerwise_transfer] session=%s layers=[%d,%d) "
                "send mamba tensors=%d bytes=%d",
                mooncake_session_id,
                begin_layer_id,
                end_layer_id,
                len(state_items),
                sum(item_len for _, (_, _, item_len) in state_items) * int(valid.sum()),
            )
        if not valid.any():
            return 0

        src_indices = prefill_mamba_indices[valid]
        dst_indices = dst_mamba_indices[valid]
        src_blocks, dst_blocks = group_concurrent_contiguous(src_indices, dst_indices)
        transfer_blocks = []
        state_fragments_by_buffer: dict[int, list[TransferFragment]] = defaultdict(list)
        for fragment in transfer_fragments:
            if fragment.buffer_kind == BufferKind.MAMBA_STATE:
                state_fragments_by_buffer[fragment.buffer_index].append(fragment)
        has_state_fragment_plan = bool(state_fragments_by_buffer)

        for state_index, (src_ptr, dst_ptr, item_len) in state_items:
            buffer_fragments = state_fragments_by_buffer.get(state_index, ())
            if has_state_fragment_plan:
                for fragment in buffer_fragments:
                    for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                        page_count = min(len(prefill_index), len(decode_index))
                        if fragment.page_count is not None:
                            page_count = min(page_count, fragment.page_count)
                        for page_offset in range(page_count):
                            src_addr = (
                                src_ptr
                                + int(prefill_index[page_offset])
                                * fragment.src_page_stride_bytes
                                + fragment.src_byte_offset
                            )
                            dst_addr = (
                                dst_ptr
                                + int(decode_index[page_offset])
                                * fragment.dst_page_stride_bytes
                                + fragment.dst_byte_offset
                            )
                            transfer_blocks.append(
                                (src_addr, dst_addr, fragment.bytes_per_page)
                            )
                continue

            for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))

        total_bytes = sum(length for _, _, length in transfer_blocks)
        ret = self._transfer_data(mooncake_session_id, transfer_blocks)
        logger.debug(
            "Transferred mamba cache for session=%s slots=%s blocks=%d bytes=%d ret=%s",
            mooncake_session_id,
            src_indices.tolist(),
            len(transfer_blocks),
            total_bytes,
            ret,
        )
        return ret

    def send_kvcache_layerwise(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
        begin_cache_step: int,
        interval: int,
        dst_state_data_ptrs: Optional[list[int]] = None,
        prefill_mamba_indices: Optional[npt.NDArray[np.int64]] = None,
        dst_mamba_indices: Optional[npt.NDArray[np.int64]] = None,
        transfer_fragments: tuple[TransferFragment, ...] = (),
    ) -> int:
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        interval = max(int(interval), 1)
        log_layerwise = getattr(self, "layerwise_debug", False)
        for begin_layer_id in range(0, self.layer_num, interval):
            end_layer_id = min(begin_layer_id + interval, self.layer_num)
            target_step = begin_cache_step + end_layer_id - 1
            if log_layerwise:
                logger.info(
                    "[layerwise_transfer] session=%s layers=[%d,%d) wait_cache_step=%d pages=%d",
                    mooncake_session_id,
                    begin_layer_id,
                    end_layer_id,
                    target_step,
                    len(prefill_kv_indices),
                )
            self._wait_until_cache_step(target_step)

            transfer_blocks = []
            if prefill_kv_blocks:
                for global_layer_id in range(begin_layer_id, end_layer_id):
                    kv_layer_index = self._kv_layer_to_index.get(global_layer_id)
                    if kv_layer_index is None:
                        continue
                    transfer_blocks.extend(
                        self._layer_transfer_blocks(
                            dst_ptrs=dst_kv_ptrs,
                            src_blocks=prefill_kv_blocks,
                            dst_blocks=dst_kv_blocks,
                            begin_layer_id=kv_layer_index,
                            end_layer_id=kv_layer_index + 1,
                            transfer_fragments=transfer_fragments,
                        )
                    )
            if transfer_blocks:
                if log_layerwise:
                    total_bytes = sum(length for _, _, length in transfer_blocks)
                    logger.info(
                        "[layerwise_transfer] session=%s layers=[%d,%d) send kv blocks=%d bytes=%d",
                        mooncake_session_id,
                        begin_layer_id,
                        end_layer_id,
                        len(transfer_blocks),
                        total_bytes,
                    )
                ret = self._transfer_data(mooncake_session_id, transfer_blocks)
                if ret != 0:
                    return ret

            ret = self.send_mamba_cache(
                mooncake_session_id,
                prefill_mamba_indices,
                dst_state_data_ptrs or [],
                dst_mamba_indices,
                begin_layer_id=begin_layer_id,
                end_layer_id=end_layer_id,
                transfer_fragments=transfer_fragments,
            )
            if ret != 0:
                return ret
            if log_layerwise:
                logger.info(
                    "[layerwise_transfer] session=%s layers=[%d,%d) done",
                    mooncake_session_id,
                    begin_layer_id,
                    end_layer_id,
                )
        return 0

    def sync_status_to_decode_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        status: int,
        prefill_rank: int,
        bootstrap_token: int = -1,
        spec_candidate_ids: Optional[list[int]] = None,
    ):
        if ":" in remote:
            remote = remote.split(":")[0]
        spec_candidate_payload = (
            np.asarray(spec_candidate_ids, dtype=np.int32).tobytes()
            if spec_candidate_ids is not None
            else b""
        )
        self._connect("tcp://" + remote + ":" + str(dst_port)).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
                str(bootstrap_token).encode("ascii"),
                spec_candidate_payload,
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                logger.debug(
                    "[TRANSFER_WORKER] Got transfer request for room %s, is_last=%s, kv_indices_len=%s",
                    kv_chunk.room,
                    kv_chunk.is_last,
                    len(kv_chunk.prefill_kv_indices),
                )
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if self._is_session_failed(req.mooncake_session_id):
                                logger.info(
                                    "Blocked transfer due to failed session (room=%s, session=%s).",
                                    kv_chunk.room,
                                    req.mooncake_session_id,
                                )
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    self.attn_tp_rank,
                                )
                                break
                        resolved = self.resolve_transfer_indices(kv_chunk, req)

                        logger.debug(
                            "[TRANSFER_WORKER] Calling send_kvcache for room %s, session %s",
                            kv_chunk.room,
                            req.mooncake_session_id,
                        )
                        tm_start = time.monotonic()
                        dst_kv_ptrs = self.decode_kv_args_table[
                            req.mooncake_session_id
                        ].dst_kv_ptrs
                        if kv_chunk.begin_cache_step is None:
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                resolved.src_indices,
                                dst_kv_ptrs,
                                resolved.dst_indices,
                                executor,
                                req.transfer_fragments,
                            )
                        else:
                            ret = self.send_kvcache_layerwise(
                                req.mooncake_session_id,
                                resolved.src_indices,
                                dst_kv_ptrs,
                                resolved.dst_indices,
                                kv_chunk.begin_cache_step,
                                kv_chunk.layerwise_interval,
                                self.decode_kv_args_table[
                                    req.mooncake_session_id
                                ].dst_state_data_ptrs,
                                kv_chunk.prefill_mamba_indices,
                                req.dst_mamba_indices,
                                req.transfer_fragments,
                            )
                        if (
                            ret == 0
                            and kv_chunk.is_last
                            and kv_chunk.begin_cache_step is None
                        ):
                            if kv_chunk.wait_for_bootstrap_token:
                                # Block until prefill metadata is published, then
                                # discard the returned tuple — the bootstrap token
                                # itself is delivered via the status message that
                                # follows this Mamba send (line ~711). This call
                                # only exists to serialize "Mamba send" after
                                # sampling completes.
                                self._wait_prefill_metadata(
                                    kv_chunk.room,
                                    kv_chunk.bootstrap_token,
                                    kv_chunk.spec_candidate_ids,
                                )
                            ret = self.send_mamba_cache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_mamba_indices,
                                self.decode_kv_args_table[
                                    req.mooncake_session_id
                                ].dst_state_data_ptrs,
                                req.dst_mamba_indices,
                                transfer_fragments=req.transfer_fragments,
                            )
                        logger.debug(
                            "[TRANSFER_WORKER] send_kvcache returned %s for room %s",
                            ret,
                            kv_chunk.room,
                        )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self._mark_session_failed(
                                        req.mooncake_session_id, reason="send_kvcache"
                                    )
                                    logger.error(
                                        "Session %s failed.", req.mooncake_session_id
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                self.attn_tp_rank,
                            )
                            break

                        if kv_chunk.is_last:
                            polls.append(True)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                # bootstrap_token is carried directly in the chunk (set by
                                # DisaggPrefillExecutor._decode after prefill forward).
                                bootstrap_token, spec_candidate_ids = (
                                    self._wait_prefill_metadata(
                                        kv_chunk.room,
                                        kv_chunk.bootstrap_token,
                                        kv_chunk.spec_candidate_ids,
                                    )
                                    if kv_chunk.wait_for_bootstrap_token
                                    else (
                                        kv_chunk.bootstrap_token,
                                        kv_chunk.spec_candidate_ids,
                                    )
                                )
                                if self.check_status(req.room) == KVPoll.Failed:
                                    status = KVPoll.Failed
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint,
                                        dst_port,
                                        room,
                                        status,
                                        self.attn_tp_rank,
                                        bootstrap_token=bootstrap_token,
                                        spec_candidate_ids=spec_candidate_ids,
                                    )
                        elapsed_seconds = time.monotonic() - tm_start
                        if self.kv_transfer_metrics:
                            self.kv_transfer_metrics.observe_kv_transfer_latency(
                                elapsed_seconds
                            )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_prefill_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                logger.info(
                    "[Prefill bootstrap_thread] recv multipart: room=%s session_id=%s",
                    room,
                    mooncake_session_id,
                )
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    with self.session_lock:
                        self._clear_failed_session(mooncake_session_id)
                    logger.info(
                        "[Prefill bootstrap_thread] registered kv_args from decode session=%s",
                        mooncake_session_id,
                    )
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[6].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.info(
                        "[Prefill bootstrap_thread] pre-alloc received: room=%d session=%s got=%d/%d, status -> %s",
                        room,
                        mooncake_session_id,
                        len(self.transfer_infos[room]),
                        required_dst_info_num,
                        (
                            "Bootstrapped"
                            if len(self.transfer_infos[room]) == required_dst_info_num
                            else "waiting more"
                        ),
                    )
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.Bootstrapped)

        threading.Thread(target=bootstrap_thread).start()

    def start_transfer_thread(
        self, transfer_thread_pool_size: int, transfer_queue_size: int
    ):
        self.transfer_queues: List[FastQueue] = [
            FastQueue() for _ in range(transfer_queue_size)
        ]
        self.executors = [
            concurrent.futures.ThreadPoolExecutor(
                transfer_thread_pool_size // transfer_queue_size
            )
            for _ in range(transfer_queue_size)
        ]
        for queue, executor in zip(self.transfer_queues, self.executors):
            threading.Thread(
                target=self.transfer_worker, args=(queue, executor), daemon=True
            ).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int64],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        mla_l1_5_args: Optional[PageTransferMetadata] = None,
        bootstrap_token: int = -1,
        begin_cache_step: Optional[int] = None,
        layerwise_interval: int = 1,
        wait_for_bootstrap_token: bool = False,
        mamba_indices: Optional[npt.NDArray[np.int64]] = None,
        spec_candidate_ids: Optional[list[int]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        #  sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.split(":")[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
                mla_l1_5_args=mla_l1_5_args,
                bootstrap_token=bootstrap_token,
                begin_cache_step=begin_cache_step,
                layerwise_interval=layerwise_interval,
                wait_for_bootstrap_token=wait_for_bootstrap_token,
                prefill_mamba_indices=mamba_indices,
                spec_candidate_ids=spec_candidate_ids,
            )
        )

    def receive_decode_prefix_info(self, bootstrap_room: int) -> int:
        """Receive decode prefix info from decode side"""
        # In mooncake implementation, decode_prefix_len is handled via ZMQ messages
        # Check the stored transfer info for this room
        if bootstrap_room in self.transfer_infos:
            for transfer_info in self.transfer_infos[bootstrap_room].values():
                if (
                    hasattr(transfer_info, "decode_prefix_len")
                    and transfer_info.decode_prefix_len > 0
                ):
                    logger.debug(
                        "Found decode_prefix_len=%s for room %s",
                        transfer_info.decode_prefix_len,
                        bootstrap_room,
                    )
                    return transfer_info.decode_prefix_len
        logger.debug("No decode_prefix_len found for room %s, using 0", bootstrap_room)
        return 0

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        if self.dist_init_addr:
            ip_address = socket.gethostbyname(self.dist_init_addr.split(":")[0])
        else:
            ip_address = get_ip()

        bootstrap_server_url = f"{ip_address}:{self.bootstrap_port}"
        url = f"http://{bootstrap_server_url}/route"
        payload = {
            "role": "Prefill",
            "world_size": self.world_size,
            "dp_size": self.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.kv_args.engine_rank,
            "enable_mla_l1_5_cache": self.args.enable_mla_l1_5_cache,
            "kv_item_lens": self.kv_args.kv_item_lens,
            "kv_unit_lens": getattr(self.kv_args, "kv_unit_lens", []),
            "state_item_lens": self.kv_args.state_item_lens,
            "state_unit_lens": getattr(self.kv_args, "state_unit_lens", []),
        }

        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    "Prefill instance failed to connect to bootstrap server: %s, %s",
                    response.status_code,
                    response.text,
                )
        except Exception as e:
            logger.error(
                "Prefill instance failed to register with bootstrap server: %s", e
            )


from tokenspeed.runtime.pd.mooncake.sender import MooncakeKVSender

__all__ = ["MooncakeKVManagerPrefill", "MooncakeKVSender"]
