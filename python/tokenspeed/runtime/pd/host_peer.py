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

"""The data plane's half of PD under the Rust engine host (RFC 0002).

The host owns the Mooncake transfer engine, the P/D handshake, the transfer
threads and the room states. What stays here is what needs this rank's
cache layout: the routes a peer's registration implies, the block manifests
read off the plan's remote streams, and the copies one request's transfer
makes. Nothing here starts a thread, opens a socket or moves a byte.

``DeviceHandle`` drives a peer through the same execution face as the Python
executors (``execute``, ``prepare_prefill``); the host's pump binds requests
to their rooms (``track``, ``release``), and its PD threads call the rest
with the interpreter lock held. Work for the host leaves through ``outbox``,
an object the host provides:

* ``outbox.send(rid, room, bootstrap_token, spec_candidate_ids)``: a Prefill
  request's KV is ready to go to its destinations (``plan_send``).
* ``outbox.send_unplanned(rid, room)``: a Prefill request reached its remote
  decode with no destination on record.
* ``outbox.receive(rid, room, request_pool_index, manifest)``: a Decode
  request's pages are admitted; ``manifest`` goes to its Prefill ranks.

The host starts its peer from ``settings`` and ``role_settings``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
from dataclasses import dataclass

import numpy as np

from tokenspeed.runtime.pd.cache_protocol import (
    CachePDBlockManifest,
    CacheTransferContract,
    build_cache_block_manifest,
    build_cache_layerwise_block_selection,
    validate_cache_manifest,
)
from tokenspeed.runtime.pd.mooncake.decode import PrefillParallelInfo
from tokenspeed.runtime.pd.mooncake.entities import (
    KVArgs,
    KVArgsRegisterInfo,
    TransferInfo,
)
from tokenspeed.runtime.pd.mooncake.pack import (
    PageFieldCopies,
    coalesce_transfer_blocks,
    flatten_transfer_blocks,
)
from tokenspeed.runtime.pd.mooncake.prefill import (
    cache_transfer_blocks,
    prepare_decode_registration,
)
from tokenspeed.runtime.pd.topology import PDParallelTopology
from tokenspeed.runtime.pd.transfer_plan import build_pipeline_transfer_plan
from tokenspeed.runtime.pd.utils import StepCounter
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.network import (
    get_ip,
    get_local_ip_by_remote,
    join_host_port,
)


@dataclass(frozen=True)
class PeerSettings:
    """What the host starts either role's peer from.

    Attributes:
        library: The Mooncake library with the TokenSpeed C entry points.
        local_ip: Where peers reach this rank.
        ib_device: The RDMA devices to use, comma-separated; empty for all.
        engine_rank: This rank's global rank.
        regions: The device ranges the transfer engine registers.
    """

    library: str
    local_ip: str
    ib_device: str
    engine_rank: int
    regions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PrefillSettings:
    """A Prefill rank's peer settings (``MooncakeKVManagerPrefill``).

    Attributes:
        status_rank: This rank in Decode's completion-tracking rank space.
        bootstrap_url: The bootstrap server's registration route.
        bootstrap_payload: The JSON this rank registers there, but its
            control port, which the host adds.
        bootstrap_timeout_s: How long a request may wait for its Decode
            ranks' pre-allocations.
        failed_session_ttl_s: How long a session whose transfer failed
            fails fast; 0 disables.
        transfer_workers: Transfer worker threads.
    """

    status_rank: int
    bootstrap_url: str
    bootstrap_payload: str
    bootstrap_timeout_s: float
    failed_session_ttl_s: float
    transfer_workers: int


@dataclass(frozen=True)
class DecodeSettings:
    """A Decode rank's peer settings (``MooncakeKVManagerDecode``).

    Attributes:
        arena_base: The cache arena's base, registered with Prefill ranks.
        layout: The cache layout's wire bytes.
        tp_size: The attention TP size.
        tp_rank: The attention TP rank.
        waiting_timeout_s: How long an admitted request may wait for its KV.
        heartbeat_interval_s: How often the Prefill nodes' health is checked.
        heartbeat_max_failures: Failed checks after which a node is lost.
    """

    arena_base: int
    layout: bytes
    tp_size: int
    tp_rank: int
    waiting_timeout_s: float
    heartbeat_interval_s: float
    heartbeat_max_failures: int


def mooncake_library() -> str:
    """The Mooncake library the host loads: ``TOKENSPEED_MOONCAKE_LIBRARY``,
    else the one the tokenspeed-mooncake wheel installs."""
    override = envs.TOKENSPEED_MOONCAKE_LIBRARY.get()
    if override:
        return override
    spec = importlib.util.find_spec("mooncake")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("PD on the Rust engine host needs tokenspeed-mooncake")
    path = os.path.join(
        spec.submodule_search_locations[0], "libtokenspeed_transfer_engine.so"
    )
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} is missing: PD on the Rust engine host needs a "
            "tokenspeed-mooncake build with the TokenSpeed C entry points"
        )
    return path


def _as_bytes(array, dtype) -> bytes:
    """An array's values as little-endian ``dtype`` bytes, for the host."""
    return np.ascontiguousarray(
        array, dtype=np.dtype(dtype).newbyteorder("<")
    ).tobytes()


def _copy_items(blocks) -> list[tuple]:
    """One transfer's copies in the host's format: ``("pages", src_pages,
    dst_pages, fields)`` page grids (``fields`` is ``[F, 5]``) as
    little-endian int64 bytes, and ``("spans", sources, destinations,
    lengths)`` as little-endian uint64 bytes, in order."""
    items: list[tuple] = []
    pending: list[object] = []

    def flush() -> None:
        # Strided rows go out one span each: the host packs nothing.
        spans = list(coalesce_transfer_blocks(flatten_transfer_blocks(pending)))
        pending.clear()
        if spans:
            columns = np.asarray(spans, dtype=np.uint64)
            items.append(
                (
                    "spans",
                    _as_bytes(columns[:, 0], np.uint64),
                    _as_bytes(columns[:, 1], np.uint64),
                    _as_bytes(columns[:, 2], np.uint64),
                )
            )

    for item in blocks:
        if isinstance(item, PageFieldCopies):
            flush()
            items.append(
                (
                    "pages",
                    _as_bytes(item.src_pages, np.int64),
                    _as_bytes(item.dst_pages, np.int64),
                    _as_bytes(item.fields, np.int64),
                )
            )
        else:
            pending.append(item)
    flush()
    return items


class _HostPeer:
    """What both roles' peers share: the cache facts the host registers."""

    def __init__(
        self,
        kv_args: KVArgs,
        topology: PDParallelTopology,
        outbox,
        settings: PeerSettings,
        role_settings: PrefillSettings | DecodeSettings,
    ) -> None:
        self.kv_args = kv_args
        self.topology = topology
        self.settings = settings
        self.role_settings = role_settings
        self._outbox = outbox
        # Request id -> bootstrap room, from the host at admission.
        self._rooms: dict[str, int] = {}

    def track(self, rid: str, room: int) -> None:
        """Bind an admitted request to its bootstrap room."""
        self._rooms[rid] = room

    def release(self, rid: str) -> None:
        """Drop an ended request's binding."""
        self._rooms.pop(rid, None)

    def release_cache_registration(self) -> None:
        """A sleep releases the cache: the host deregisters it from its
        transfer engine before the memory saver unmaps it (refused while a
        transfer is in flight)."""
        self._outbox.release_registration()

    def restore_cache_registration(self) -> None:
        """A wake re-mapped the cache: the host registers it again, a Decode
        rank under a fresh transfer session."""
        self._outbox.restore_registration()


class HostPrefillPeer(_HostPeer):
    """A Prefill rank's layout-dependent PD work."""

    mode = "prefill"

    def __init__(
        self,
        kv_args: KVArgs,
        topology: PDParallelTopology,
        outbox,
        settings: PeerSettings,
        role_settings: PrefillSettings,
    ) -> None:
        super().__init__(kv_args, topology, outbox, settings, role_settings)
        # Session id -> the Decode rank's registration, planned.
        self._registrations: dict[str, KVArgsRegisterInfo] = {}
        # Room -> session id -> the Decode rank's request manifest.
        self._destinations: dict[int, dict[str, TransferInfo]] = {}
        # Room -> this rank's manifest of the request's blocks.
        self._sources: dict[int, CachePDBlockManifest] = {}
        # Layerwise streaming (``register_layerwise_step_counter``): the
        # counter the attention backends tick as each producer step's KV
        # lands, the producer schedule, and the steps per transfer interval.
        self._step_counter: StepCounter | None = None
        self._producer_schedule = None
        self._layerwise_interval = 0
        # Rooms whose KV streams layerwise, and the chunks armed and not yet
        # planned by a transfer worker: chunk id -> (room, block selection).
        self._layerwise_rooms: set[int] = set()
        self._layerwise_chunks: dict[int, tuple[int, object]] = {}
        self._next_chunk_id = 0

    def release(self, rid: str) -> None:
        """Drop an ended request's binding and its room's manifests."""
        room = self._rooms.pop(rid, None)
        if room is not None:
            self.drop_room(room)

    def prepare_registration(
        self, frames: list[bytes]
    ) -> tuple[str, int, str, int, int, bool, list[int]]:
        """Plan a Decode rank's registration; the host keeps its route.

        Returns:
            ``(endpoint, port, session_id, decode_tp_size, decode_tp_rank,
            is_dummy, expected_decode_ranks)``.

        Raises:
            ValueError: The registration is malformed, misrouted, or
                conflicts with the session's registration on record.
        """
        registration = prepare_decode_registration(
            self.kv_args, self.topology, KVArgsRegisterInfo.from_zmq(frames)
        )
        session_id = registration.mooncake_session_id
        previous = self._registrations.get(session_id)
        if previous is not None and previous != registration:
            raise ValueError("conflicting registration for an active CachePD session")
        self._registrations[session_id] = registration
        return (
            registration.endpoint,
            registration.dst_port,
            session_id,
            registration.decode_tp_size,
            registration.decode_tp_rank,
            registration.is_dummy,
            sorted(registration.expected_decode_ranks),
        )

    def reject_registration(self, session_id: str) -> None:
        """Forget a session the host rejected."""
        self._registrations.pop(session_id, None)

    def accept_preallocation(self, frames: list[bytes]) -> bool:
        """Validate a Decode rank's request manifest against its registration
        and keep it for the transfer.

        Returns:
            Whether the pre-allocation is a dummy (no manifest).

        Raises:
            ValueError: The message is malformed or disagrees with the
                session's registration.
        """
        info = TransferInfo.from_zmq(frames)
        registration = self._registrations.get(info.mooncake_session_id)
        if registration is None or registration.transfer_owner_filters is None:
            raise ValueError(
                "CachePD request references an unregistered Decode session"
            )
        if info.is_dummy != registration.is_dummy:
            raise ValueError("CachePD request kind disagrees with its registration")
        if info.block_manifest is not None:
            validate_cache_manifest(
                info.block_manifest,
                layout=registration.peer_cache_layout,
                peer="destination",
            )
        destinations = self._destinations.setdefault(info.room, {})
        if info.mooncake_session_id in destinations:
            raise ValueError("duplicate CachePD pre-allocation for Decode session")
        destinations[info.mooncake_session_id] = info
        return info.is_dummy

    def drop_room(self, room: int) -> None:
        """Forget an ended room's manifests."""
        self._destinations.pop(room, None)
        self._sources.pop(room, None)
        self._layerwise_rooms.discard(room)

    def register_layerwise_step_counter(self, step_counter, interval: int) -> None:
        """Stream each prefill chunk's KV layerwise: ``step_counter`` ticks
        as each producer step lands, and the host's transfer workers read
        its pinned host copy (``layerwise_step_address``)."""
        producer_schedule = self.kv_args.cache_producer_schedule
        if producer_schedule is None:
            raise ValueError("layerwise cache transfer requires a producer schedule")
        self._step_counter = step_counter
        self._producer_schedule = producer_schedule
        self._layerwise_interval = max(int(interval), 1)

    @property
    def layerwise_step_address(self) -> int | None:
        """The pinned host word holding the last landed producer step, or
        None without layerwise streaming."""
        if self._step_counter is None:
            return None
        return self._step_counter.h_ready_cache_step.data_ptr()

    def prepare_prefill(self, op) -> None:
        """Arm one layerwise chunk per extend row that has a Decode
        destination (``DisaggPrefillExecutor._prepare_cache_prefill``): its
        block selection stays here for the transfer worker's
        ``plan_layerwise``; the host learns the producer steps to wait for.
        Runs on the forward thread, ahead of the forward it arms."""
        if self._step_counter is None or op.num_extends() == 0:
            return
        pending = []
        for index, rid in enumerate(op.request_ids[: op.num_extends()]):
            room = self._rooms.get(rid)
            if room is None:
                continue
            destinations = [
                info
                for info in self._destinations.get(room, {}).values()
                if not info.is_dummy
            ]
            # A dummy-only room completes after the forward like any other.
            if not destinations:
                continue
            window = destinations[0].block_manifest
            for destination in destinations:
                other = destination.block_manifest
                if (
                    other.prefix_len != window.prefix_len
                    or other.prompt_len != window.prompt_len
                    or other.state_prefix_cached != window.state_prefix_cached
                ):
                    raise ValueError(
                        "Cache-transfer destinations disagree on the prompt window"
                    )
            chunk_begin = int(op.extend_prefix_lens[index])
            chunk_end = chunk_begin + int(op.input_lengths[index])
            if chunk_end > window.prompt_len:
                raise ValueError(
                    "Cache-transfer Prefill chunk extends past Decode's prompt manifest"
                )
            is_last = chunk_end == window.prompt_len
            # The first chunk also carries the blocks this rank found in its
            # cache that Decode still needs; later ones start at their own
            # boundary, so nothing goes twice.
            selection_start = chunk_begin
            if room not in self._layerwise_rooms:
                selection_start = min(selection_start, window.prefix_len)
            selection = build_cache_layerwise_block_selection(
                op,
                layout=self.kv_args.cache_layout,
                request_row=index,
                prefix_len=window.prefix_len,
                prompt_len=window.prompt_len,
                chunk_start=selection_start,
                chunk_end=chunk_end,
                state_prefix_cached=window.state_prefix_cached,
            )
            if not is_last and not any(
                group.source_block_ids for group in selection.groups
            ):
                continue
            pending.append((rid, room, is_last, selection))
        # The attention backends record one producer range per forward, even
        # when its chunk completes no block.
        begin_step, _ = self._step_counter.current_step()
        step_count = self._producer_schedule.step_count
        self._step_counter.advance_step(delta_cache_step=step_count, delta_aux_step=0)
        targets = [
            (begin_step + min(begin + self._layerwise_interval, step_count) - 1)
            % StepCounter.COUNT_NUM_MAX
            for begin in range(0, step_count, self._layerwise_interval)
        ]
        for rid, room, is_last, selection in pending:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            self._layerwise_chunks[chunk_id] = (room, selection)
            self._layerwise_rooms.add(room)
            self._outbox.send_layerwise(rid, room, chunk_id, targets, is_last)

    def plan_layerwise(
        self, chunk_id: int, session_ids: list[str]
    ) -> list[list[list[tuple]]]:
        """The copies of one armed chunk to each of ``session_ids``, per
        transfer interval, in ``plan_send``'s format; forgets the chunk."""
        room, selection = self._layerwise_chunks.pop(chunk_id)
        step_count = self._producer_schedule.step_count
        step_field_ids = [
            self._producer_schedule.fields_in_range(
                begin, min(begin + self._layerwise_interval, step_count)
            )
            for begin in range(0, step_count, self._layerwise_interval)
        ]
        plans = []
        for session_id in session_ids:
            registration = self._registrations[session_id]
            info = self._destinations[room][session_id]
            intervals = cache_transfer_blocks(
                self.kv_args,
                dst_ptr=registration.dst_kv_ptr,
                dst_tp_rank=registration.decode_tp_rank,
                src_block_manifest=None,
                dst_block_manifest=info.block_manifest,
                transfer_fragments=registration.transfer_fragments,
                owner_filters=registration.transfer_owner_filters,
                dst_cache_layout=registration.peer_cache_layout,
                block_selection=selection,
                step_field_ids=step_field_ids,
            )
            plans.append([_copy_items(blocks) for blocks in intervals])
        return plans

    def execute(self, op) -> None:
        """The plan's remote decode: read each completed prompt's blocks and
        hand the request to the host (``DisaggPrefillExecutor._cache_decode``).
        Runs on the forward thread, behind the forwards that wrote the KV."""
        for index, rid in enumerate(op.request_ids):
            room = self._rooms.get(rid)
            if room is None:
                continue
            token = int(op.decode_input_ids[index])
            if token < 0:
                raise RuntimeError("Cache-transfer bootstrap token is unavailable")
            spec_candidate_ids = list(op.spec_candidate_ids[index]) or None
            destinations = [
                info
                for info in self._destinations.get(room, {}).values()
                if not info.is_dummy
            ]
            if not self._destinations.get(room):
                self._outbox.send_unplanned(rid, room)
                continue
            # A layerwise room's KV already streams: the host only needs its
            # bootstrap payload.
            if destinations and room not in self._layerwise_rooms:
                window = destinations[0].block_manifest
                for destination in destinations:
                    other = destination.block_manifest
                    if (
                        other.prefix_len != window.prefix_len
                        or other.prompt_len != window.prompt_len
                        or other.state_prefix_cached != window.state_prefix_cached
                    ):
                        raise ValueError(
                            "Cache-transfer destinations disagree on the prompt window"
                        )
                self._sources[room] = build_cache_block_manifest(
                    op,
                    layout=self.kv_args.cache_layout,
                    request_row=index,
                    prefix_len=window.prefix_len,
                    prompt_len=window.prompt_len,
                    state_prefix_cached=window.state_prefix_cached,
                )
            self._outbox.send(rid, room, token, spec_candidate_ids)

    def plan_send(self, room: int, session_id: str) -> list[tuple]:
        """The copies that move one request's blocks to one Decode rank.

        Returns:
            In order, ``("pages", src_pages, dst_pages, fields)`` page grids
            (``fields`` is ``[F, 5]``) as little-endian int64 bytes, and
            ``("spans", sources, destinations, lengths)`` as little-endian
            uint64 bytes.
        """
        registration = self._registrations[session_id]
        info = self._destinations[room][session_id]
        (blocks,) = cache_transfer_blocks(
            self.kv_args,
            dst_ptr=registration.dst_kv_ptr,
            dst_tp_rank=registration.decode_tp_rank,
            src_block_manifest=self._sources[room],
            dst_block_manifest=info.block_manifest,
            transfer_fragments=registration.transfer_fragments,
            owner_filters=registration.transfer_owner_filters,
            dst_cache_layout=registration.peer_cache_layout,
            step_field_ids=(None,),
        )
        return _copy_items(blocks)


class HostDecodePeer(_HostPeer):
    """A Decode rank's layout-dependent PD work."""

    mode = "decode"

    def plan_route(
        self, prefill_info: str
    ) -> tuple[list[int], list[int], list[int], int]:
        """The Prefill ranks a request of this rank reads from, given the
        bootstrap server's Prefill description (its ``engine_rank=-1`` JSON).

        Returns:
            ``(target_tp_ranks, dummy_tp_ranks, expected_prefill_ranks,
            prefill_dp_size)``: the ranks to register with and send each
            request's manifest to (dummy ones get an empty manifest), and
            the ranks whose completion the request waits for.
        """
        payload = json.loads(prefill_info)
        layout_wire = payload.get("cache_layout")
        if layout_wire is None:
            raise RuntimeError(
                "Cache-transfer decode connected to a non-cache-transfer prefill"
            )
        info = PrefillParallelInfo(
            tp_size=int(payload["prefill_tp_size"]),
            dp_size=int(payload["prefill_dp_size"]),
            cache_fields_by_stage=tuple(
                tuple(stage) for stage in payload["cache_fields_by_stage"]
            ),
            cache_layout=CacheTransferContract.from_wire_bytes(
                layout_wire.encode("ascii")
            ),
            pp_size=int(payload.get("prefill_pp_size", 1)),
        )
        plan, dummy = build_pipeline_transfer_plan(
            prefill_tp_size=info.prefill_tp_size_per_dp_rank,
            decode_tp_size=self.topology.tp_size,
            decode_tp_rank=self.topology.tp_rank,
            prefill_layout=info.cache_layout,
            decode_layout=self.kv_args.cache_layout,
            cache_fields_by_stage=info.cache_fields_by_stage,
        )
        targets = sorted((*plan.target_prefill_ranks, *dummy))
        return targets, list(dummy), sorted(plan.target_prefill_ranks), info.dp_size

    def execute(self, op) -> None:
        """The plan's remote prefill: each admitted prompt's destination
        manifest goes to the host (``DisaggDecodeExecutor._cache_prefill``).
        Every row is validated before any is handed over."""
        num_extends = op.num_extends()
        if num_extends != len(op.request_ids):
            raise RuntimeError(
                "CachePD Decode admission does not support mixed batches"
            )
        pending = []
        for index, rid in enumerate(op.request_ids[:num_extends]):
            room = self._rooms.get(rid)
            if room is None:
                continue
            manifest = build_cache_block_manifest(
                op,
                layout=self.kv_args.cache_layout,
                request_row=index,
                prefix_len=int(op.extend_prefix_lens[index]),
                prompt_len=int(op.prefill_lengths[index]),
                state_prefix_cached=bool(op.state_prefix_cached[index]),
            )
            validate_cache_manifest(
                manifest, layout=self.kv_args.cache_layout, peer="destination"
            )
            pending.append(
                (
                    rid,
                    room,
                    int(op.request_pool_indices[index]),
                    manifest.to_wire_bytes(),
                )
            )
        for rid, room, request_pool_index, manifest in pending:
            self._outbox.receive(rid, room, request_pool_index, manifest)


def build_host_peer(
    server_args,
    executor,
    *,
    cache_fields_by_stage: tuple[tuple[str, ...], ...],
    producer_fields_by_step: tuple[tuple[str, ...], ...],
    logical_plan,
    model_config,
    draft_model_config,
    gpu_id: int,
    global_rank: int,
    outbox,
) -> HostPrefillPeer | HostDecodePeer | None:
    """The Rust host's ``kv_transfer_builder`` (see ``build_device_side``):
    the peer of this rank's role over the rank's KV arena (``get_kv_args``),
    or None outside disaggregation.

    Args:
        outbox: Where the peer hands the host its work.
    """
    if server_args.disaggregation_mode == "null":
        return None
    from tokenspeed.runtime.pd.factory import get_kv_args

    kv_args = get_kv_args(
        global_rank,
        gpu_id,
        server_args.disaggregation_ib_device,
        executor.token_to_kv_pool,
        model_config=model_config,
        draft_model_config=draft_model_config,
        cache_fields_by_stage=cache_fields_by_stage,
        producer_fields_by_step=producer_fields_by_step,
        logical_plan=logical_plan,
    )
    topology = PDParallelTopology.from_mapping(server_args.mapping)
    local_ip = get_local_ip_by_remote()
    settings = PeerSettings(
        library=mooncake_library(),
        local_ip=local_ip,
        ib_device=kv_args.ib_device or "",
        engine_rank=topology.global_rank,
        regions=tuple(
            kv_args.registration_regions
            or ((kv_args.kv_data_ptr, kv_args.cache_layout.plan.arena_bytes),)
        ),
    )
    if server_args.disaggregation_mode == "prefill":
        return HostPrefillPeer(
            kv_args,
            topology,
            outbox,
            settings,
            _prefill_settings(server_args, kv_args, topology, local_ip),
        )
    if server_args.disaggregation_mode == "decode":
        return HostDecodePeer(
            kv_args,
            topology,
            outbox,
            settings,
            DecodeSettings(
                arena_base=kv_args.kv_data_ptr,
                layout=kv_args.cache_layout.to_wire_bytes(),
                tp_size=topology.tp_size,
                tp_rank=topology.tp_rank,
                waiting_timeout_s=float(
                    envs.TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT.get()
                ),
                heartbeat_interval_s=max(
                    envs.TOKENSPEED_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
                ),
                heartbeat_max_failures=max(
                    envs.TOKENSPEED_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
                ),
            ),
        )
    raise NotImplementedError(
        f"the Rust engine host has no {server_args.disaggregation_mode} peer"
    )


def _prefill_settings(
    server_args, kv_args: KVArgs, topology: PDParallelTopology, local_ip: str
) -> PrefillSettings:
    """``MooncakeKVManagerPrefill``'s settings and bootstrap registration."""
    if server_args.dist_init_addr:
        bootstrap_ip = socket.gethostbyname(server_args.dist_init_addr.split(":")[0])
    else:
        bootstrap_ip = get_ip()
    payload = {
        "role": "Prefill",
        "world_size": topology.world_size,
        "dp_size": topology.dp_size,
        "pp_size": topology.pp_size,
        "cache_fields_by_stage": [
            list(stage) for stage in kv_args.cache_fields_by_stage
        ],
        "rank_ip": local_ip,
        "engine_rank": topology.global_rank,
        "cache_layout": kv_args.wire_layout.to_wire_bytes().decode("ascii"),
    }
    # Stage-major, so Decode counts pp*tp distinct completions with PP.
    status_rank = topology.pp_rank * topology.tp_size + topology.tp_rank
    return PrefillSettings(
        status_rank=status_rank,
        bootstrap_url=(
            f"http://{join_host_port(bootstrap_ip, server_args.disaggregation_bootstrap_port)}"
            "/route"
        ),
        bootstrap_payload=json.dumps(payload),
        bootstrap_timeout_s=float(
            envs.TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
        ),
        failed_session_ttl_s=float(
            max(envs.TOKENSPEED_DISAGGREGATION_FAILED_SESSION_TTL.get(), 0)
        ),
        transfer_workers=envs.TOKENSPEED_DISAGGREGATION_QUEUE_SIZE.get(),
    )
