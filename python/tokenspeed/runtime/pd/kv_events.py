# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

"""KV cache events: their wire schema and the publisher configuration.

Every rank's Rust control plane drains its scheduler's events and the
attention-TP group's first rank publishes them over ZMQ (``ts-control``'s
``kv_events``). These structs are the payload's schema as a consumer decodes
it with msgspec; the publisher's encoding matches them byte for byte.
"""

from typing import Any

import msgspec
from pydantic import BaseModel


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    attn_dp_rank: int | None = None


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events."""


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]


class KVEventsConfig(BaseModel):
    """Configuration for KV event publishing."""

    publisher: str | None = None
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    Defaults to "zmq" when events are enabled and "null" otherwise.
    """

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    replay_endpoint: str | None = None
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    enable_kv_cache_events: bool = False
    """Whether to publish scheduler KV cache mutation events."""

    @classmethod
    def from_cli(cls, cli_value: str) -> "KVEventsConfig":
        """Parse the CLI value for the event publisher config."""
        return KVEventsConfig.model_validate_json(cli_value)


def kv_events_enabled(config: str | None) -> bool:
    """Whether ``--kv-events-config`` turns the scheduler's KV events on."""
    if not config:
        return False
    return KVEventsConfig.from_cli(config).enable_kv_cache_events
