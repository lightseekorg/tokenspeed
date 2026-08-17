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
from collections.abc import Iterator
from dataclasses import replace
from itertools import chain, islice

import numpy as np
import requests

from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    CachePDBlockManifest,
    CachePDLayerwiseBlockSelection,
    CacheProducerSchedule,
    CacheTransferContract,
    validate_cache_manifest,
)
from tokenspeed.runtime.pd.mooncake.conn import MooncakeKVManagerBase
from tokenspeed.runtime.pd.mooncake.entities import (
    KVArgs,
    KVArgsRegisterInfo,
    KVManagerArgs,
    TransferInfo,
    TransferKVChunk,
)
from tokenspeed.runtime.pd.transfer_plan import (
    CacheTransferFragment,
    CacheTransferPlanner,
)
from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
    FastQueue,
    StepCounter,
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

# Application-side descriptor batching keeps heterogeneous-TP row fragments
# bounded in Python memory. This is not a Mooncake backend limit.
_TRANSFER_DESCRIPTOR_BATCH_SIZE = 4096


class MooncakeKVManagerPrefill(MooncakeKVManagerBase):
    def __init__(
        self,
        args: KVManagerArgs,
        kv_args: KVArgs,
    ):
        super().__init__(args, kv_args, DisaggregationMode.PREFILL)

        self.transfer_infos: dict[int, dict[str, TransferInfo]] = {}
        self.decode_kv_args_table: dict[str, KVArgsRegisterInfo] = {}
        self.rejected_decode_sessions: dict[str, tuple[str, int]] = {}
        self.session_failures = defaultdict(int)
        self.failed_sessions: dict[str, float] = {}
        self.failed_session_ttl = max(
            envs.TOKENSPEED_DISAGGREGATION_FAILED_SESSION_TTL.get(), 0
        )
        self.session_lock = threading.Lock()
        self.producer_schedule: CacheProducerSchedule | None = None
        self.producer_step_count = 0
        self.layerwise_interval = 1
        self.layerwise_debug = envs.TOKENSPEED_PD_LAYERWISE_DEBUG.get()
        self.step_counter = None
        # room -> (bootstrap_token, spec_candidate_ids). Published after the prefill
        # forward; the transfer thread reads it on the wait_for_bootstrap_token path.
        self.prefill_metadata: dict[int, tuple[int, list[int] | None]] = {}
        self.bootstrap_token_cond = threading.Condition()
        # Determine the number of threads to use for kv sender
        cpu_count = os.cpu_count()
        transfer_thread_pool_size = (
            envs.TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE.get_set_value_or(
                min(max(4, int(0.75 * cpu_count) // 8), 12)
            )
        )
        transfer_queue_size = envs.TOKENSPEED_DISAGGREGATION_QUEUE_SIZE.get()
        if transfer_thread_pool_size < transfer_queue_size:
            raise ValueError(
                "TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE="
                f"{transfer_thread_pool_size} must be greater than or equal to "
                f"TOKENSPEED_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
        self.start_transfer_thread(transfer_thread_pool_size, transfer_queue_size)
        self.bootstrap_time_out = envs.TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        # Publish this manager only after every field used by its bootstrap and
        # transfer threads has been initialized.
        self.start_prefill_thread()
        self._register_to_bootstrap()

    def register_layerwise_step_counter(
        self, step_counter: StepCounter, interval: int
    ) -> None:
        producer_schedule = self.kv_args.cache_producer_schedule
        if producer_schedule is None:
            raise ValueError("layerwise cache transfer requires a producer schedule")
        self.producer_schedule = producer_schedule
        self.producer_step_count = producer_schedule.step_count
        self.step_counter = step_counter
        self.layerwise_interval = max(int(interval), 1)

    def reserve_layerwise_cache_steps(self) -> int:
        if self.step_counter is None:
            return 0
        cache_step, _ = self.step_counter.current_step()
        self.step_counter.advance_step(
            delta_cache_step=self.producer_step_count,
            delta_aux_step=0,
        )
        return cache_step

    def set_prefill_metadata(
        self,
        room: int,
        token: int,
        spec_candidate_ids: list[int] | None = None,
    ) -> None:
        with self.bootstrap_token_cond:
            if self.request_status.get(room) in (None, TransferPoll.Failed):
                logger.warning(
                    "Dropping late prefill metadata for expired bootstrap_room=%s",
                    room,
                )
                return
            self.prefill_metadata[room] = (
                token,
                spec_candidate_ids,
            )
            self.bootstrap_token_cond.notify_all()

    def begin_room(self, room: int) -> None:
        """Reset request metadata before publishing a room."""
        with self.bootstrap_token_cond:
            self.prefill_metadata.pop(room, None)
        self.update_status(room, TransferPoll.Bootstrapping)

    def discard_room(self, room: int) -> None:
        """Drop all Prefill-manager state owned by one terminal request."""
        self.transfer_infos.pop(room, None)
        self.request_status.pop(room, None)
        with self.bootstrap_token_cond:
            self.prefill_metadata.pop(room, None)
            self.bootstrap_token_cond.notify_all()

    def _wait_prefill_metadata(
        self,
        room: int | None,
        fallback_token: int,
        fallback_candidate_ids: list[int] | None,
    ) -> tuple[int, list[int] | None]:
        if room is None or fallback_token != -1:
            return fallback_token, fallback_candidate_ids
        wait_log_interval = max(envs.TOKENSPEED_PD_PREFILL_METADATA_TIMEOUT.get(), 0.01)
        start_time = time.monotonic()
        next_log_time = start_time + wait_log_interval
        with self.bootstrap_token_cond:
            while room not in self.prefill_metadata:
                if self.request_status.get(room) in (None, TransferPoll.Failed):
                    logger.warning(
                        "Prefill metadata unavailable for failed "
                        "bootstrap_room=%s; using fallback=%s",
                        room,
                        fallback_token,
                    )
                    return fallback_token, fallback_candidate_ids
                now = time.monotonic()
                if now >= next_log_time:
                    logger.debug(
                        "Still waiting for prefill metadata for "
                        "bootstrap_room=%s after %.2fs",
                        room,
                        now - start_time,
                    )
                    next_log_time = now + wait_log_interval
                self.bootstrap_token_cond.wait(timeout=0.01)
            return self.prefill_metadata[room]

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

    def get_decode_registration(self, req: TransferInfo) -> KVArgsRegisterInfo:
        registration = self.decode_kv_args_table.get(req.mooncake_session_id)
        if registration is None:
            raise ValueError(
                "CachePD request references an unregistered Decode session"
            )
        return registration

    def _reject_decode_registration(
        self,
        *,
        endpoint: str,
        dst_port: int,
        session_id: str,
        reason: str,
    ) -> None:
        """Invalidate a rejected session and fail rooms that already reference it."""
        affected_rooms = tuple(
            room for room, infos in self.transfer_infos.items() if session_id in infos
        )
        for room in affected_rooms:
            self.abort_room(
                room,
                f"Decode session {session_id} registration rejected: {reason}",
            )
        self.decode_kv_args_table.pop(session_id, None)
        self.rejected_decode_sessions[session_id] = (endpoint, dst_port)

    def _prepare_decode_registration(
        self, registration: KVArgsRegisterInfo
    ) -> KVArgsRegisterInfo:
        """Validate and cache the route owned by this Prefill rank once."""
        layout = self.kv_args.cache_layout
        peer_layout = registration.peer_cache_layout
        prefill_tp_size = self.topology.tp_size
        local_tp_rank = self.topology.tp_rank
        planner = CacheTransferPlanner(
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=registration.decode_tp_size,
            prefill_layout=layout,
            decode_layout=peer_layout,
        )
        route = planner.plan_for_decode_rank(registration.decode_tp_rank)
        expected_decode_ranks = planner.decode_ranks_by_prefill_rank[local_tp_rank]
        if local_tp_rank in route.target_prefill_ranks:
            return replace(
                registration,
                transfer_fragments=route.fragments_by_prefill_rank[local_tp_rank],
                is_dummy=False,
                expected_decode_ranks=expected_decode_ranks,
            )
        if registration.decode_tp_rank == 0 and not expected_decode_ranks:
            return replace(
                registration,
                transfer_fragments=(),
                is_dummy=True,
                expected_decode_ranks=frozenset({0}),
            )
        raise ValueError(
            "CachePD Decode registration targets the wrong Prefill TP rank"
        )

    def _validate_cache_room_fanout(self, reqs: tuple[TransferInfo, ...]) -> None:
        """Require one complete, unique Decode-rank set for a Paged source."""
        registrations = tuple(self.get_decode_registration(req) for req in reqs)
        decode_tp_sizes = {reg.decode_tp_size for reg in registrations}
        expected_rank_sets = {reg.expected_decode_ranks for reg in registrations}
        if len(decode_tp_sizes) != 1 or len(expected_rank_sets) != 1:
            raise ValueError("Paged cache room has inconsistent TP metadata")
        expected_decode_ranks = next(iter(expected_rank_sets))
        actual_decode_ranks = tuple(reg.decode_tp_rank for reg in registrations)
        if len(reqs) != len(expected_decode_ranks) or set(actual_decode_ranks) != set(
            expected_decode_ranks
        ):
            raise ValueError(
                "Paged cache room destination ranks disagree with the typed route plan"
            )

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        block_iter = iter(transfer_blocks)
        while batch := tuple(islice(block_iter, _TRANSFER_DESCRIPTOR_BATCH_SIZE)):
            src_addrs, dst_addrs, lengths = zip(*batch, strict=True)
            ret = self.engine.batch_transfer_sync(
                mooncake_session_id,
                list(src_addrs),
                list(dst_addrs),
                list(lengths),
            )
            if ret != 0:
                return ret
        return 0

    def _cache_transfer_blocks(
        self,
        *,
        dst_ptr: int,
        src_block_manifest: CachePDBlockManifest | None,
        dst_block_manifest: CachePDBlockManifest,
        transfer_fragments: tuple[CacheTransferFragment, ...] = (),
        dst_cache_layout: CacheTransferContract,
        block_selection: CachePDLayerwiseBlockSelection | None = None,
        field_ids: frozenset[str] | None = None,
    ) -> Iterator[tuple[int, int, int]]:
        layout = self.kv_args.cache_layout

        cache_fragments = tuple(transfer_fragments)
        local_segments = {
            (segment.group_id, segment.field_id): segment
            for segment in layout.plan.fields
        }
        peer_segments = {
            (segment.group_id, segment.field_id): segment
            for segment in dst_cache_layout.plan.fields
        }
        fragments_by_group: dict[str, list[CacheTransferFragment]] = defaultdict(list)
        for fragment in cache_fragments:
            fragments_by_group[fragment.group_id].append(fragment)

        # Resolve block ids directly from the validated manifest/selection. This
        # leaves one authoritative representation of the request's block map.
        group_transfers = []
        source_groups = (
            block_selection.groups
            if block_selection is not None
            else src_block_manifest.groups
        )
        for group_spec, src_group, dst_group in zip(
            layout.group_specs, source_groups, dst_block_manifest.groups, strict=True
        ):
            source_block_ids = (
                src_group.source_block_ids
                if block_selection is not None
                else src_group.block_ids
            )
            destination_block_ids = (
                tuple(
                    dst_group.block_ids[position]
                    for position in src_group.destination_positions
                )
                if block_selection is not None
                else dst_group.block_ids
            )
            group_transfers.append(
                (
                    group_spec,
                    source_block_ids,
                    destination_block_ids,
                )
            )

        for group_spec, group_src_indices, group_dst_indices in group_transfers:
            if cache_fragments:
                for fragment in fragments_by_group.get(group_spec.group_id, ()):
                    key = (fragment.group_id, fragment.field_id)
                    src_segment = local_segments[key]
                    dst_segment = peer_segments[key]
                    if field_ids is not None and src_segment.field_id not in field_ids:
                        continue
                    for src_page, dst_page in zip(
                        group_src_indices, group_dst_indices, strict=True
                    ):
                        src_page_addr = (
                            self.kv_args.kv_data_ptr
                            + layout.plan.field_page_byte_offset(
                                src_segment.field_id, 0
                            )
                            + int(src_page) * src_segment.page_stride_bytes
                            + fragment.src_byte_offset
                        )
                        dst_page_addr = (
                            dst_ptr
                            + dst_cache_layout.plan.field_page_byte_offset(
                                dst_segment.field_id, 0
                            )
                            + int(dst_page) * dst_segment.page_stride_bytes
                            + fragment.dst_byte_offset
                        )
                        for row in range(fragment.rows_per_page):
                            yield (
                                src_page_addr + row * fragment.src_row_stride_bytes,
                                dst_page_addr + row * fragment.dst_row_stride_bytes,
                                fragment.bytes_per_row,
                            )
                continue

            group_fields = layout.fields_for_group(group_spec.group_id)
            if group_fields:
                for src_segment in group_fields:
                    if field_ids is not None and src_segment.field_id not in field_ids:
                        continue
                    key = (group_spec.group_id, src_segment.field_id)
                    dst_segment = peer_segments[key]
                    for src_page, dst_page in zip(
                        group_src_indices, group_dst_indices, strict=True
                    ):
                        yield (
                            self.kv_args.kv_data_ptr
                            + layout.plan.field_page_byte_offset(
                                src_segment.field_id, 0
                            )
                            + int(src_page) * src_segment.page_stride_bytes,
                            dst_ptr
                            + dst_cache_layout.plan.field_page_byte_offset(
                                dst_segment.field_id, 0
                            )
                            + int(dst_page) * dst_segment.page_stride_bytes,
                            src_segment.payload_bytes,
                        )
                continue

    def _wait_until_cache_step(
        self,
        target_step: int,
        *,
        room: int | None = None,
        session_ids: tuple[str, ...] = (),
    ) -> None:
        if self.step_counter is None:
            if room is not None:
                raise ValueError(
                    "Paged cache layerwise transfer has no producer step counter"
                )
            return
        next_session_check = time.monotonic()
        while True:
            if room is not None and self.request_status.get(room) in (
                None,
                TransferPoll.Failed,
            ):
                raise ValueError(f"Paged cache layerwise wait aborted for room {room}")
            now = time.monotonic()
            if session_ids and now >= next_session_check:
                with self.session_lock:
                    failed = any(
                        self._is_session_failed(session_id)
                        for session_id in session_ids
                    )
                if failed:
                    raise ValueError(
                        f"Paged cache layerwise peer failed for room {room}"
                    )
                next_session_check = now + 0.1
            ready_step = self.step_counter.query_ready_cache_step()
            if StepCounter.is_step_ready(ready_step, target_step):
                return
            time.sleep(1e-4)

    @staticmethod
    def _prime_transfer_blocks(
        transfer_blocks: Iterator[tuple[int, int, int]],
    ) -> Iterator[tuple[int, int, int]]:
        """Start a lazy descriptor before any destination begins DMA."""
        iterator = iter(transfer_blocks)
        try:
            first = next(iterator)
        except StopIteration:
            return iter(())
        return chain((first,), iterator)

    def _send_cache_layerwise_fanout(
        self,
        kv_chunk: TransferKVChunk,
        reqs: tuple[TransferInfo, ...],
    ) -> bool:
        """Wait one producer interval, then fan it out to every Decode peer."""
        block_selection = kv_chunk.cache_block_selection

        registered_reqs = []
        for req in reqs:
            with self.session_lock:
                session_failed = self._is_session_failed(req.mooncake_session_id)
            if session_failed:
                self.abort_room(
                    kv_chunk.room,
                    "Decode instance could be dead, remote Mooncake session "
                    f"{req.mooncake_session_id} is not alive",
                )
                return False
            registration = self.get_decode_registration(req)
            registered_reqs.append((req, registration))

        producer_schedule = self.producer_schedule
        if producer_schedule is None:
            raise ValueError("layerwise cache transfer requires a producer schedule")
        producer_step_count = producer_schedule.step_count
        interval = max(int(kv_chunk.layerwise_interval), 1)
        for begin_step in range(0, producer_step_count, interval):
            end_step = min(begin_step + interval, producer_step_count)
            target_step = (
                kv_chunk.begin_cache_step + end_step - 1
            ) % StepCounter.COUNT_NUM_MAX
            if self.layerwise_debug:
                logger.info(
                    "[cache_layerwise_transfer] room=%s producer_steps=[%d,%d) "
                    "wait_cache_step=%d peers=%d",
                    kv_chunk.room,
                    begin_step,
                    end_step,
                    target_step,
                    len(reqs),
                )
            self._wait_until_cache_step(
                target_step,
                room=kv_chunk.room,
                session_ids=tuple(req.mooncake_session_id for req in reqs),
            )

            # Start every peer's lazy generator before transferring this interval.
            prepared = []
            for req, registration in registered_reqs:
                assert req.block_manifest is not None
                blocks = self._cache_transfer_blocks(
                    dst_ptr=registration.dst_kv_ptr,
                    src_block_manifest=None,
                    dst_block_manifest=req.block_manifest,
                    transfer_fragments=registration.transfer_fragments,
                    dst_cache_layout=registration.peer_cache_layout,
                    block_selection=block_selection,
                    field_ids=producer_schedule.fields_in_range(begin_step, end_step),
                )
                prepared.append(
                    (
                        req,
                        registration,
                        self._prime_transfer_blocks(blocks),
                        time.monotonic(),
                    )
                )

            for req, registration, blocks, started_at in prepared:
                ret = self._transfer_data(req.mooncake_session_id, blocks)
                if self.kv_transfer_metrics:
                    self.kv_transfer_metrics.observe_kv_transfer_latency(
                        time.monotonic() - started_at
                    )
                if ret != 0:
                    with self.session_lock:
                        self.session_failures[req.mooncake_session_id] += 1
                        self._mark_session_failed(
                            req.mooncake_session_id, reason="send_cache_layerwise"
                        )
                    self.abort_room(
                        kv_chunk.room,
                        f"Failed to send Paged cache layer interval of "
                        f"{kv_chunk.room} to "
                        f"{registration.endpoint}:{registration.dst_port}",
                    )
                    return False

        if not kv_chunk.is_last:
            return True

        bootstrap_token, spec_candidate_ids = self._wait_prefill_metadata(
            kv_chunk.room,
            kv_chunk.bootstrap_token,
            kv_chunk.spec_candidate_ids,
        )
        if self.check_status(kv_chunk.room) == TransferPoll.Failed:
            return False
        self.update_status(kv_chunk.room, TransferPoll.Success)
        for req, registration in registered_reqs:
            self.sync_status_to_decode_endpoint(
                registration.endpoint,
                registration.dst_port,
                req.room,
                TransferPoll.Success,
                self.topology.tp_rank,
                bootstrap_token=bootstrap_token,
                spec_candidate_ids=spec_candidate_ids,
            )
        return True

    def sync_status_to_decode_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        status: int,
        prefill_rank: int,
        bootstrap_token: int = -1,
        spec_candidate_ids: list[int] | None = None,
    ):
        if ":" in remote:
            remote = remote.split(":")[0]
        spec_candidate_payload = (
            np.asarray(spec_candidate_ids, dtype=np.int32).tobytes()
            if spec_candidate_ids is not None
            else b""
        )
        socket, lock = self._connect("tcp://" + remote + ":" + str(dst_port))
        with lock:
            socket.send_multipart(
                [
                    str(room).encode("ascii"),
                    str(status).encode("ascii"),
                    str(prefill_rank).encode("ascii"),
                    str(bootstrap_token).encode("ascii"),
                    spec_candidate_payload,
                ]
            )

    def abort_room(self, room: int, reason: str) -> None:
        """Notify the decode that a room failed before any KV transfer.

        EPD: when the prefill aborts a request on embedding-receive timeout it never
        sends KV, so the decode's dual-dispatched KV receiver would wait
        indefinitely -- its heartbeat only trips if the prefill /health dies, and the
        receiver waiting_timeout only covers the WaitingForInput state (a receiver
        whose prefill never registered a sender is stuck earlier). Push a Failed
        status to every decode endpoint that already pre-allocated for this room
        (mirrors the in-transfer failure path), so the decode raises a FailedEvent and
        the client gets an error instead of hanging. A room whose decode has not
        pre-allocated yet is only marked Failed locally (no endpoint to notify).
        """
        self.record_failure(room, reason)
        self.update_status(room, TransferPoll.Failed)
        for req in list(self.transfer_infos.get(room, {}).values()):
            if not req.is_dummy:
                try:
                    registration = self.decode_kv_args_table.get(
                        req.mooncake_session_id
                    )
                    rejected = self.rejected_decode_sessions.get(
                        req.mooncake_session_id
                    )
                    if registration is None and rejected is None:
                        raise ValueError(
                            "CachePD room references an unregistered Decode session"
                        )
                    endpoint, dst_port = (
                        (registration.endpoint, registration.dst_port)
                        if registration is not None
                        else rejected
                    )
                    self.sync_status_to_decode_endpoint(
                        endpoint,
                        dst_port,
                        req.room,
                        TransferPoll.Failed,
                        self.topology.tp_rank,
                    )
                except Exception:
                    logger.exception(
                        "Failed to notify Decode about room-level transfer "
                        "failure (room=%s session=%s)",
                        room,
                        req.mooncake_session_id,
                    )
        self.transfer_infos.pop(room, None)

    def transfer_worker(
        self,
        queue: FastQueue,
        _executor: concurrent.futures.ThreadPoolExecutor,
    ):
        while True:
            kv_chunk = queue.get()
            try:
                reqs = tuple(self.transfer_infos.get(kv_chunk.room, {}).values())
                if not reqs:
                    raise ValueError("CachePD transfer room has no destinations")
                if any(req.is_dummy for req in reqs):
                    if kv_chunk.is_last and kv_chunk.room in self.request_status:
                        self.update_status(kv_chunk.room, TransferPoll.Success)
                        self.transfer_infos.pop(kv_chunk.room, None)
                    continue

                if kv_chunk.cache_block_selection is not None:
                    self._send_cache_layerwise_fanout(kv_chunk, reqs)
                    if kv_chunk.room not in self.request_status or self.check_status(
                        kv_chunk.room
                    ) in (TransferPoll.Success, TransferPoll.Failed):
                        self.transfer_infos.pop(kv_chunk.room, None)
                    continue
                if kv_chunk.begin_cache_step is not None:
                    raise ValueError(
                        "CachePD producer-step transfer requires a layerwise selection"
                    )

                prepared = []
                for req in reqs:
                    with self.session_lock:
                        if self._is_session_failed(req.mooncake_session_id):
                            raise ValueError(
                                f"Decode session {req.mooncake_session_id} is not alive"
                            )
                    registration = self.get_decode_registration(req)
                    assert req.block_manifest is not None
                    blocks = self._cache_transfer_blocks(
                        dst_ptr=registration.dst_kv_ptr,
                        src_block_manifest=kv_chunk.block_manifest,
                        dst_block_manifest=req.block_manifest,
                        transfer_fragments=registration.transfer_fragments,
                        dst_cache_layout=registration.peer_cache_layout,
                    )
                    prepared.append(
                        (
                            req,
                            registration,
                            self._prime_transfer_blocks(blocks),
                            time.monotonic(),
                        )
                    )

                for req, registration, blocks, started_at in prepared:
                    ret = self._transfer_data(req.mooncake_session_id, blocks)
                    if ret != 0:
                        with self.session_lock:
                            self.session_failures[req.mooncake_session_id] += 1
                            self._mark_session_failed(
                                req.mooncake_session_id, reason="send_kvcache"
                            )
                        raise RuntimeError(
                            "Mooncake transfer failed for "
                            f"{registration.endpoint}:{registration.dst_port}"
                        )
                    if self.kv_transfer_metrics:
                        self.kv_transfer_metrics.observe_kv_transfer_latency(
                            time.monotonic() - started_at
                        )

                if not kv_chunk.is_last:
                    continue
                if kv_chunk.wait_for_bootstrap_token:
                    bootstrap_token, spec_candidate_ids = self._wait_prefill_metadata(
                        kv_chunk.room,
                        kv_chunk.bootstrap_token,
                        kv_chunk.spec_candidate_ids,
                    )
                else:
                    bootstrap_token = kv_chunk.bootstrap_token
                    spec_candidate_ids = kv_chunk.spec_candidate_ids
                if self.check_status(kv_chunk.room) == TransferPoll.Failed:
                    continue
                self.update_status(kv_chunk.room, TransferPoll.Success)
                for req in reqs:
                    registration = self.get_decode_registration(req)
                    self.sync_status_to_decode_endpoint(
                        registration.endpoint,
                        registration.dst_port,
                        req.room,
                        TransferPoll.Success,
                        self.topology.tp_rank,
                        bootstrap_token=bootstrap_token,
                        spec_candidate_ids=spec_candidate_ids,
                    )
                self.transfer_infos.pop(kv_chunk.room, None)
            except Exception as exc:
                logger.exception("CachePD transfer failed for room=%s", kv_chunk.room)
                self.abort_room(kv_chunk.room, f"CachePD transfer failed: {exc}")

    def _handle_bootstrap_message(self, frames: list[bytes]) -> None:
        """Consume one static registration or one manifest-only request."""
        try:
            room_header = frames[0].decode("ascii")
        except (IndexError, UnicodeError):
            logger.exception("Rejecting malformed Mooncake bootstrap message header")
            return

        if room_header == "None":
            try:
                registration = KVArgsRegisterInfo.from_zmq(frames)
                registration = self._prepare_decode_registration(registration)
            except (IndexError, UnicodeError, ValueError) as exc:
                try:
                    endpoint, dst_port, session_id = (
                        KVArgsRegisterInfo.route_header_from_zmq(frames)
                    )
                    self._reject_decode_registration(
                        endpoint=endpoint,
                        dst_port=dst_port,
                        session_id=session_id,
                        reason=str(exc),
                    )
                except (IndexError, UnicodeError, ValueError):
                    pass
                logger.exception("Rejecting malformed CachePD Decode registration")
                return
            session_id = registration.mooncake_session_id
            previous = self.decode_kv_args_table.get(session_id)
            if previous is not None and previous != registration:
                reason = "conflicting registration for an active CachePD session"
                self._reject_decode_registration(
                    endpoint=registration.endpoint,
                    dst_port=registration.dst_port,
                    session_id=session_id,
                    reason=reason,
                )
                logger.error(
                    "Rejecting conflicting CachePD registration for session=%s",
                    session_id,
                )
                return
            self.decode_kv_args_table[session_id] = registration
            self.rejected_decode_sessions.pop(session_id, None)
            with self.session_lock:
                self._clear_failed_session(session_id)
            logger.info(
                "[Prefill bootstrap_thread] registered kv_args from decode session=%s",
                session_id,
            )
            return

        parsed_room = None
        session_id = None
        registration = None
        try:
            parsed_room, session_id = TransferInfo.route_header_from_zmq(frames)
            transfer_info = TransferInfo.from_zmq(frames)
            registration = self.get_decode_registration(transfer_info)
            if transfer_info.is_dummy != registration.is_dummy:
                raise ValueError("CachePD request kind disagrees with its registration")
            if self.request_status.get(parsed_room) == TransferPoll.Failed:
                self.sync_status_to_decode_endpoint(
                    registration.endpoint,
                    registration.dst_port,
                    parsed_room,
                    TransferPoll.Failed,
                    self.topology.tp_rank,
                )
                return
            if transfer_info.block_manifest is not None:
                validate_cache_manifest(
                    transfer_info.block_manifest,
                    layout=registration.peer_cache_layout,
                    peer="destination",
                )
            expected_fanout = len(registration.expected_decode_ranks)
            candidate_infos = dict(self.transfer_infos.get(parsed_room, {}))
            if session_id in candidate_infos:
                raise ValueError("duplicate CachePD pre-allocation for Decode session")
            candidate_infos[session_id] = transfer_info
            if len(candidate_infos) > expected_fanout:
                raise ValueError("pre-allocation exceeds destination fanout")
            if len(candidate_infos) == expected_fanout:
                self._validate_cache_room_fanout(tuple(candidate_infos.values()))
        except (IndexError, UnicodeError, ValueError) as exc:
            logger.exception(
                "Rejecting malformed pre-allocation metadata for room=%s",
                room_header,
            )
            if parsed_room is None:
                return
            try:
                self.abort_room(
                    parsed_room,
                    f"invalid pre-allocation metadata: {exc}",
                )
            except Exception:
                logger.exception(
                    "Could not abort malformed CachePD room=%s", parsed_room
                )
            try:
                if registration is None and session_id is not None:
                    registration = self.decode_kv_args_table.get(session_id)
                rejected = (
                    self.rejected_decode_sessions.get(session_id)
                    if registration is None and session_id is not None
                    else None
                )
                if registration is None and rejected is None:
                    raise ValueError("malformed request has no registered session")
                endpoint, dst_port = (
                    (registration.endpoint, registration.dst_port)
                    if registration is not None
                    else rejected
                )
                self.sync_status_to_decode_endpoint(
                    endpoint,
                    dst_port,
                    parsed_room,
                    TransferPoll.Failed,
                    self.topology.tp_rank,
                )
            except Exception:
                logger.exception(
                    "Could not notify malformed Decode peer for room=%s",
                    parsed_room,
                )
            return

        self.transfer_infos[parsed_room] = candidate_infos
        # A transfer worker can fail and remove the room after the precheck but
        # before this bootstrap-thread commit.  Do not let that interleaving
        # resurrect destination state for a sticky-Failed room.
        if self.request_status.get(parsed_room) == TransferPoll.Failed:
            self.transfer_infos.pop(parsed_room, None)
            self.sync_status_to_decode_endpoint(
                registration.endpoint,
                registration.dst_port,
                parsed_room,
                TransferPoll.Failed,
                self.topology.tp_rank,
            )
            return
        complete = len(candidate_infos) == expected_fanout
        logger.info(
            "[Prefill bootstrap_thread] pre-alloc received: room=%d "
            "session=%s got=%d/%d, status -> %s",
            parsed_room,
            session_id,
            len(candidate_infos),
            expected_fanout,
            "Bootstrapped" if complete else "waiting more",
        )
        if complete:
            self.update_status(parsed_room, TransferPoll.Bootstrapped)

    def start_prefill_thread(self):
        self.rank_port = get_free_port()
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            """Receive registrations and pre-allocation manifests from Decode."""
            while True:
                frames = self.server_socket.recv_multipart()
                try:
                    self._handle_bootstrap_message(frames)
                except Exception:
                    # A malformed peer must never terminate the process-wide
                    # bootstrap worker and strand every later request.
                    logger.exception("Unexpected CachePD bootstrap message failure")

        threading.Thread(target=bootstrap_thread).start()

    def start_transfer_thread(
        self, transfer_thread_pool_size: int, transfer_queue_size: int
    ):
        self.transfer_queues: list[FastQueue] = [
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
        is_last: bool,
        bootstrap_token: int = -1,
        begin_cache_step: int | None = None,
        layerwise_interval: int = 1,
        wait_for_bootstrap_token: bool = False,
        spec_candidate_ids: list[int] | None = None,
        block_manifest: CachePDBlockManifest | None = None,
        cache_block_selection: CachePDLayerwiseBlockSelection | None = None,
    ):
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            raise RuntimeError("Transfer requests can only be added in prefill mode.")
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == TransferPoll.Failed
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
                is_last=is_last,
                bootstrap_token=bootstrap_token,
                begin_cache_step=begin_cache_step,
                layerwise_interval=layerwise_interval,
                wait_for_bootstrap_token=wait_for_bootstrap_token,
                spec_candidate_ids=spec_candidate_ids,
                block_manifest=block_manifest,
                cache_block_selection=cache_block_selection,
            )
        )

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
            "world_size": self.topology.world_size,
            "dp_size": self.topology.dp_size,
            "rank_ip": get_local_ip_by_remote(),
            "rank_port": self.rank_port,
            "engine_rank": self.topology.global_rank,
            "cache_layout": self.kv_args.cache_layout.to_wire_bytes().decode("ascii"),
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
        except Exception as exc:
            logger.error(
                "Prefill instance failed to register with bootstrap server: %s", exc
            )


from tokenspeed.runtime.pd.mooncake.sender import MooncakeKVSender

__all__ = ["MooncakeKVManagerPrefill", "MooncakeKVSender"]
