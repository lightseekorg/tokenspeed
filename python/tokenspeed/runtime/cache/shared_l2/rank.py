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

"""A rank joins the shared L2 pool: its control client and its load executor,
built over the rank's cache pools at device construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.cache.l3.factory import resolve_deployment_id
from tokenspeed.runtime.cache.shared_l2.client import (
    SharedL2Client,
    SharedL2ClientConfig,
)
from tokenspeed.runtime.cache.shared_l2.executor import (
    SharedL2LoadExecutor,
    build_shared_l2_load_executor,
)
from tokenspeed.runtime.cache.shared_l2.layout import build_shared_l2_layout
from tokenspeed.runtime.cache.shared_l2.transfer import RdmaEngine, SharedL2Transfer
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.utils.server_args import ServerArgs


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
) -> tuple[SharedL2Client, SharedL2LoadExecutor]:
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

    Returns:
        The rank's pool client, for the event loop, and the executor that
        takes the Host cache executor's place.
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
    deployment_id = resolve_deployment_id(
        server_args,
        model_config,
        draft_model_config,
        cache_layout=layout.transfer,
        checkpoint_granularities=checkpoint_granularities,
        group_shard_counts=tuple(
            spec.shard_count for spec in device_pool.arena.runtime_contract.group_specs
        ),
        attention_backend_name=attention_backend_name,
        draft_attention_backend_name=draft_attention_backend_name,
        world_cpu_group=world_cpu_group,
    )
    engine = RdmaEngine(
        device_name=config.device_names[gpu_id],
        regions=layout.registration_regions(),
    )
    client = SharedL2Client(
        config,
        geometry=layout.geometry,
        deployment_id=deployment_id,
        transfer=SharedL2Transfer(layout, engine),
        connection=OwnerConnection(config.owner, timeout_ms=config.request_timeout_ms),
    )
    executor = build_shared_l2_load_executor(
        client,
        layout,
        engine,
        device_pool=device_pool,
        draft_pool=draft_pool,
        gpu_id=gpu_id,
    )
    return client, executor
