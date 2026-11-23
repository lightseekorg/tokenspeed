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

"""A rank joins the shared L2 pool at device construction: its transfer engine
and its client in the engine host, started through the host's shared L2
handle around the rank's hello to the owner, and its load executor over the
rank's cache pools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.cache.l3.factory import cache_layout_deployment_id
from tokenspeed.runtime.cache.shared_l2.client import (
    SharedL2ClientConfig,
    SharedL2Provider,
    wait_until_ready,
)
from tokenspeed.runtime.cache.shared_l2.executor import (
    SharedL2LoadExecutor,
    build_shared_l2_load_executor,
)
from tokenspeed.runtime.cache.shared_l2.layout import (
    SharedL2PageField,
    build_shared_l2_layout,
)
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection
from tokenspeed.runtime.pd.host_peer import (
    enable_mooncake_ipv6_for_host,
    mooncake_library,
    mooncake_protocol,
)
from tokenspeed.runtime.utils.network import get_local_ip_by_remote

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(frozen=True, slots=True)
class SharedL2PageGroup:
    """A cache group's pool pages.

    Attributes:
        packing: Pool pages per LCM block.
        page_bytes: Bytes of one pool page. A pool page is a Host block of
            the same size: this is the layout's ``block_bytes`` of the group.
        fields: The group's fields, as ``SharedL2Layout.page_fields`` lists
            them.
    """

    packing: int
    page_bytes: int
    fields: tuple[SharedL2PageField, ...]


@dataclass(frozen=True, slots=True)
class SharedL2ClientSettings:
    """What the host starts the rank's pool client from, once the rank has
    joined (``SharedL2Handle.start_client``).

    Attributes:
        owner_address: The owner's ZMQ address.
        client_id: The rank's id at the owner, from its hello.
        request_timeout_ms: How long a request waits for the owner's answer.
        read_workers: Threads reading one load's providers in parallel.
        write_workers: Threads writing to the pool.
        lcm_block_bytes: Bytes of one pool LCM block.
        providers: Where the pool's bytes are.
        groups: The cache groups' pages, in the scheduler's order.
        num_layers: The rank's layers, the target's and then the draft's.
        signal_words_address: Where the loads' signal words, their start and
            ready words, are in host memory.
        num_signal_words: How many words there are.
    """

    owner_address: str
    client_id: int
    request_timeout_ms: int
    read_workers: int
    write_workers: int
    lcm_block_bytes: int
    providers: tuple[SharedL2Provider, ...]
    groups: tuple[SharedL2PageGroup, ...]
    num_layers: int
    signal_words_address: int
    num_signal_words: int


def join_shared_l2_pool(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
    *,
    device_pool: Any,
    draft_pool: Any | None,
    cache_groups: Sequence[Any],
    prefix_granularity: int,
    checkpoint_granularities: tuple[tuple[str, int], ...],
    attention_backend_name: str,
    draft_attention_backend_name: str,
    world_cpu_group: ProcessGroup | None,
    gpu_id: int,
    shared_l2_handle: Any,
) -> tuple[SharedL2ClientConfig, SharedL2LoadExecutor]:
    """Join the pool ``--shared-l2-config`` names, waiting for it to be ready.

    Args:
        server_args: Resolved launch settings.
        model_config: Target checkpoint configuration.
        draft_model_config: Draft checkpoint configuration, or None.
        device_pool: The target cache pool.
        draft_pool: The draft pool sharing the arena, or None.
        cache_groups: The scheduler's ``CacheGroupConfig`` of each cache
            group.
        prefix_granularity: The scheduler's prefix page, in tokens.
        checkpoint_granularities: The groups whose checkpoints are coarser
            than a page, with their granularity.
        attention_backend_name: The resolved target attention backend.
        draft_attention_backend_name: The resolved draft attention backend.
        world_cpu_group: The world's Gloo group, for the ranks to agree on
            the checkpoint of the deployment identity; None in a
            single-process deployment.
        gpu_id: Local device index.
        shared_l2_handle: The host's shared L2 handle (``SharedL2Handle``):
            the rank's transfer engine and pool client start there, and its
            loads run there.

    Returns:
        The rank's client config, and the executor that takes the Host cache
        executor's place.
    """
    config = SharedL2ClientConfig.from_json(server_args.shared_l2_config)
    if not 0 <= gpu_id < len(config.device_names):
        raise ValueError(
            f"--shared-l2-config device_names has no entry for GPU {gpu_id}"
        )
    layout = build_shared_l2_layout(
        device_pool,
        draft_pool,
        cache_groups=cache_groups,
        prefix_granularity=prefix_granularity,
    )
    # The identity L3 names its objects under: ranks whose bytes differ never
    # share a pool.
    deployment_id = cache_layout_deployment_id(
        server_args,
        model_config,
        draft_model_config,
        cache_layout=layout.transfer_layout,
        checkpoint_granularities=checkpoint_granularities,
        group_shard_counts=tuple(
            spec.shard_count for spec in device_pool.arena.runtime_contract.group_specs
        ),
        attention_backend_name=attention_backend_name,
        draft_attention_backend_name=draft_attention_backend_name,
        world_cpu_group=world_cpu_group,
    )
    local_ip = get_local_ip_by_remote()
    enable_mooncake_ipv6_for_host(local_ip)
    # Before the hello: registering the cache takes time, and the owner
    # drops a client it does not hear from.
    shared_l2_handle.start_transfer_engine(
        library=mooncake_library(),
        hostname=local_ip,
        protocol=mooncake_protocol(),
        device_filter=config.device_names[gpu_id],
        regions=layout.registration_regions(),
    )
    connection = OwnerConnection(
        config.owner_address, timeout_ms=config.request_timeout_ms
    )
    try:
        joined_pool = wait_until_ready(
            config,
            geometry=layout.geometry,
            deployment_id=deployment_id,
            connection=connection,
        )
    finally:
        connection.close()
    executor = build_shared_l2_load_executor(
        layout,
        shared_l2_handle,
        joined_pool=joined_pool,
        device_pool=device_pool,
        draft_pool=draft_pool,
        gpu_id=gpu_id,
    )
    signal_words_address, num_signal_words = executor.signal_words
    # The client's keepalive starts here, right after the hello.
    shared_l2_handle.start_client(
        SharedL2ClientSettings(
            owner_address=config.owner_address,
            client_id=joined_pool.client_id,
            request_timeout_ms=config.request_timeout_ms,
            read_workers=config.read_workers,
            write_workers=config.write_workers,
            lcm_block_bytes=joined_pool.lcm_block_bytes,
            providers=joined_pool.providers,
            groups=tuple(
                SharedL2PageGroup(
                    packing=layout.host_group_packing[group],
                    page_bytes=layout.block_bytes[group],
                    fields=layout.page_fields(group),
                )
                for group in range(len(layout.transfer_layout.groups))
            ),
            num_layers=len(layout.transfer_layout.consumers),
            signal_words_address=signal_words_address,
            num_signal_words=num_signal_words,
        )
    )
    return config, executor
