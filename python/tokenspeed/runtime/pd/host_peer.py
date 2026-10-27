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

The host runs the whole transfer (``ts-pd``): the Mooncake engine, the P/D
handshake, the routes between unequal rank sets, the block manifests read
off each plan's block tables, and the copies. What stays here is what only
the data plane knows: the rank's cache transfer contract and arena, built
once into the settings the host starts its peer from, and the device fences
a plan's remote stream waits on.

``DeviceHandle`` calls ``execute(op)`` with the plan's remote stream once its
device prerequisites are met -- on a Prefill rank behind the drafter's
pending round, whose KV goes out too; on a Decode rank behind the plan's page
zeroing, as the Prefill ranks' writes are not ordered with the device's --
and the host then plans and sends the stream (``outbox.remote_stream_ready``).
The outbox is an object the host provides; a sleep releases the cache's
transfer registration through it and a wake restores it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.pd.topology import PDParallelTopology
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.network import (
    get_ip,
    get_local_ip_by_remote,
    is_valid_ipv6_address,
    join_host_port,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.pd.factory import KVArgs


@dataclass(frozen=True)
class PeerSettings:
    """What the host starts either role's peer from.

    Attributes:
        library: The Mooncake library with the TokenSpeed C entry points.
        local_ip: Where peers reach this rank.
        protocol: The Mooncake transport selected for this rank's device.
        ib_device: The RDMA devices to use, comma-separated; empty for all.
        engine_rank: This rank's global rank.
        regions: The device ranges the transfer engine registers.
    """

    library: str
    local_ip: str
    protocol: str
    ib_device: str
    engine_rank: int
    regions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PrefillSettings:
    """A Prefill rank's peer settings.

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
        layout: This rank's cache transfer contract's wire bytes, with its
            fields' landing addresses: the stage's resident fields only
            under a pipeline.
        wire_layout: The logical contract every rank registers: the complete
            model's, without addresses.
        arena_base: The cache arena's base.
        tp_size: The attention TP size within a stage.
        tp_rank: This rank's attention TP rank.
        pp_rank: This rank's pipeline stage.
        fields_by_stage: Every stage's resident cache fields.
        fields_by_step: The fields each producer step makes ready, for
            layerwise streaming (``CacheProducerSchedule``).
    """

    status_rank: int
    bootstrap_url: str
    bootstrap_payload: str
    bootstrap_timeout_s: float
    failed_session_ttl_s: float
    transfer_workers: int
    layout: bytes
    wire_layout: bytes
    arena_base: int
    tp_size: int
    tp_rank: int
    pp_rank: int
    fields_by_stage: tuple[tuple[str, ...], ...]
    fields_by_step: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DecodeSettings:
    """A Decode rank's peer settings.

    Attributes:
        arena_base: The cache arena's base, registered with Prefill ranks.
        layout: The cache transfer contract's wire bytes, with its fields'
            landing addresses.
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


def enable_mooncake_ipv6_for_host(hostname: str) -> None:
    """Make Mooncake's handshake use IPv6 when ``hostname``, the address the
    host's transfer engine starts on, is an IPv6 literal.

    Mooncake opens its P2P handshake sockets as IPv4 unless ``MC_USE_IPV6``
    is set, and reads it when its first engine starts, so the bootstrap sets
    it before the host starts one; an explicit setting is kept.
    """
    if is_valid_ipv6_address(hostname):
        os.environ.setdefault("MC_USE_IPV6", "1")


def mooncake_protocol() -> str:
    """The native transport for this rank's device, resolved at startup."""
    return "ascend_direct" if current_platform().is_npu else "rdma"


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


class _HostPeer:
    """What both roles' peers share: the settings the host starts from, and
    the outbox the device fences reach it through."""

    def __init__(
        self,
        outbox,
        settings: PeerSettings,
        role_settings: PrefillSettings | DecodeSettings,
    ) -> None:
        self.settings = settings
        self.role_settings = role_settings
        self._outbox = outbox

    def execute(self, op) -> None:
        """The plan's remote stream ``op`` may go out: the host plans and
        sends it from the plan it staged. Runs on the forward thread, inside
        the plan's execution."""
        del op  # The host reads the stream off the plan itself.
        self._outbox.remote_stream_ready()

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
    """A Prefill rank's peer: its settings and the layerwise step counter."""

    mode = "prefill"

    def __init__(
        self, outbox, settings: PeerSettings, role_settings: PrefillSettings
    ) -> None:
        super().__init__(outbox, settings, role_settings)
        # Layerwise streaming (``register_layerwise_step_counter``): the
        # counter the attention backends tick as each producer step's KV
        # lands, and the steps per transfer interval.
        self._step_counter = None
        self._layerwise_interval = 0

    def register_layerwise_step_counter(self, step_counter, interval: int) -> None:
        """Stream each prefill chunk's KV layerwise: ``step_counter`` ticks
        as each producer step lands, and the host's transfer workers read
        its pinned host copy (``layerwise_step_address``)."""
        if not self.role_settings.fields_by_step:
            raise ValueError("layerwise cache transfer requires a producer schedule")
        self._step_counter = step_counter
        self._layerwise_interval = max(int(interval), 1)

    @property
    def layerwise_step_address(self) -> int | None:
        """The pinned host word holding the last landed producer step, or
        None without layerwise streaming."""
        if self._step_counter is None:
            return None
        return self._step_counter.h_ready_cache_step.data_ptr()

    @property
    def layerwise_interval(self) -> int:
        """Producer steps per layerwise transfer interval; 0 without
        layerwise streaming."""
        return self._layerwise_interval


class HostDecodePeer(_HostPeer):
    """A Decode rank's peer: its settings."""

    mode = "decode"


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
        outbox: Where the peer reaches the host.
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
    enable_mooncake_ipv6_for_host(local_ip)
    settings = PeerSettings(
        library=mooncake_library(),
        local_ip=local_ip,
        protocol=mooncake_protocol(),
        ib_device=kv_args.ib_device or "",
        engine_rank=topology.global_rank,
        regions=tuple(
            kv_args.registration_regions
            or ((kv_args.kv_data_ptr, kv_args.cache_layout.plan.arena_bytes),)
        ),
    )
    if server_args.disaggregation_mode == "prefill":
        return HostPrefillPeer(
            outbox,
            settings,
            _prefill_settings(server_args, kv_args, topology, local_ip),
        )
    if server_args.disaggregation_mode == "decode":
        return HostDecodePeer(
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
    """A Prefill rank's settings and bootstrap registration."""
    if server_args.dist_init_addr:
        bootstrap_ip = socket.gethostbyname(server_args.dist_init_addr.split(":")[0])
    else:
        bootstrap_ip = get_ip()
    wire_layout = kv_args.wire_layout.to_wire_bytes()
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
        "cache_layout": wire_layout.decode("ascii"),
    }
    # Stage-major, so Decode counts pp*tp distinct completions with PP.
    status_rank = topology.pp_rank * topology.tp_size + topology.tp_rank
    schedule = kv_args.cache_producer_schedule
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
        layout=kv_args.cache_layout.to_wire_bytes(),
        wire_layout=wire_layout,
        arena_base=kv_args.kv_data_ptr,
        tp_size=topology.tp_size,
        tp_rank=topology.tp_rank,
        pp_rank=topology.pp_rank,
        fields_by_stage=tuple(tuple(stage) for stage in kv_args.cache_fields_by_stage),
        fields_by_step=schedule.fields_by_step if schedule is not None else (),
    )
