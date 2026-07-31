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

"""Mooncake master KV-events config + optional relay scaffold (RFC #1527).

When Mooncake master publishes KV events (PR #2214), TokenSpeed can subscribe
and relay instead of (or in addition to) engine-side L3 disk publish. Until the
master publisher SDK/API is available in the environment, the subscriber stays
idle and fail-open: missing endpoint or missing event API only logs a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from tokenspeed.runtime.pd.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)

logger = logging.getLogger(__name__)

MooncakeKvEventsSource = Literal["engine", "master", "both"]

_VALID_SOURCES: frozenset[str] = frozenset({"engine", "master", "both"})

# Map RFC #1527 / legacy master event type strings → wire kind.
_STORED_TYPES = frozenset({"stored", "blockstored", "BlockStored"})
_REMOVED_TYPES = frozenset({"removed", "blockremoved", "BlockRemoved"})
_CLEARED_TYPES = frozenset({"cleared", "allblockscleared", "AllBlocksCleared"})


@dataclass(frozen=True)
class MooncakeKvEventsConfig:
    """Parsed ``extra_config["kv_events"]`` for Mooncake master relay.

    Attributes:
        source: Who emits L3/disk KV events. ``engine`` (default) keeps
            engine-side L3 publish; ``master`` relies on master publisher;
            ``both`` keeps engine L3 and also attempts master subscribe.
        master_subscribe_endpoint: ZMQ SUB endpoint for master PUB (e.g.
            ``tcp://mooncake-master:6000``). Required for master/both to
            become active; missing endpoint → fail-open idle.
        backend_id: Optional cache-owner identity when normalizing/relaying
            master events (DaemonSet topology).
    """

    source: MooncakeKvEventsSource = "engine"
    master_subscribe_endpoint: str | None = None
    backend_id: str | None = None


def parse_mooncake_kv_events_config(
    extra_config: dict | None,
) -> MooncakeKvEventsConfig:
    """Parse nested ``kv_events`` from Mooncake ``extra_config``.

    Missing or empty ``kv_events`` defaults to ``source=engine`` so existing
    deployments are unchanged.

    Args:
        extra_config: Dict from ``kvstore_storage_backend_extra_config`` JSON,
            or ``None``.

    Returns:
        Normalized :class:`MooncakeKvEventsConfig`.

    Raises:
        ValueError: If ``source`` is present but not one of
            ``engine`` / ``master`` / ``both``.
    """
    if not extra_config:
        return MooncakeKvEventsConfig()

    section = extra_config.get("kv_events")
    if not section:
        return MooncakeKvEventsConfig()
    if not isinstance(section, dict):
        raise ValueError("kv_events must be a JSON object")

    source_raw = section.get("source", "engine")
    if source_raw not in _VALID_SOURCES:
        raise ValueError(
            f"kv_events.source must be one of {sorted(_VALID_SOURCES)}, "
            f"got {source_raw!r}"
        )
    source: MooncakeKvEventsSource = source_raw  # type: ignore[assignment]

    endpoint = section.get("master_subscribe_endpoint")
    if endpoint is not None and not isinstance(endpoint, str):
        raise ValueError("kv_events.master_subscribe_endpoint must be a string")
    if isinstance(endpoint, str) and not endpoint.strip():
        endpoint = None

    backend_id = section.get("backend_id")
    if backend_id is not None and not isinstance(backend_id, str):
        raise ValueError("kv_events.backend_id must be a string")

    return MooncakeKvEventsConfig(
        source=source,
        master_subscribe_endpoint=endpoint,
        backend_id=backend_id,
    )


def engine_publishes_l3_disk(config: MooncakeKvEventsConfig) -> bool:
    """Whether the engine should publish Mooncake L3 (disk) KV events.

    ``source=master`` disables engine-side L3 callbacks to avoid duplicates
    when the master publisher is the authority. ``engine`` and ``both`` keep
    engine L3 publish.
    """
    return config.source != "master"


def normalize_master_event(
    raw: dict,
) -> BlockStored | BlockRemoved | AllBlocksCleared:
    """Map RFC #1527 / legacy master JSON fields onto TokenSpeed wire structs.

    Accepts synthetic dicts for unit tests. Recognizes both RFC names
    (``event_type``, ``seq_hashes``, ``parent_hash``) and legacy aliases
    (``type``, ``block_hashes``, ``parent_block_hash``).

    Args:
        raw: One master event as a plain dict.

    Returns:
        A ``BlockStored``, ``BlockRemoved``, or ``AllBlocksCleared`` instance.

    Raises:
        ValueError: If ``event_type`` / ``type`` is missing or unknown.
    """
    type_raw = raw.get("event_type", raw.get("type"))
    if type_raw is None:
        raise ValueError("master event missing event_type/type")
    type_key = str(type_raw)
    type_cmp = type_key.lower().replace("_", "")

    hashes = raw.get("seq_hashes", raw.get("block_hashes"))
    parent = raw.get("parent_hash", raw.get("parent_block_hash"))
    token_ids = raw.get("token_ids") or []
    block_size = raw.get("block_size")
    backend_id = raw.get("backend_id")
    medium = raw.get("medium")
    dp_rank = raw.get("dp_rank")
    model_name = raw.get("model_name")
    tenant_id = raw.get("tenant_id")
    event_id = raw.get("event_id")

    def _matches(candidates: frozenset[str]) -> bool:
        return type_key in candidates or type_cmp in {
            c.lower().replace("_", "") for c in candidates
        }

    if _matches(_STORED_TYPES):
        if hashes is None:
            raise ValueError("stored master event requires seq_hashes/block_hashes")
        if block_size is None:
            block_size = len(token_ids) if token_ids else 0
        return BlockStored(
            block_hashes=[int(h) for h in hashes],
            parent_block_hash=None if parent is None else int(parent),
            token_ids=[int(t) for t in token_ids],
            block_size=int(block_size),
            backend_id=backend_id,
            medium=medium,
            dp_rank=dp_rank,
            model_name=model_name,
            tenant_id=tenant_id,
            event_id=event_id,
        )

    if _matches(_REMOVED_TYPES):
        if hashes is None:
            raise ValueError("removed master event requires seq_hashes/block_hashes")
        return BlockRemoved(
            block_hashes=[int(h) for h in hashes],
            backend_id=backend_id,
            medium=medium,
            dp_rank=dp_rank,
            model_name=model_name,
            tenant_id=tenant_id,
            event_id=event_id,
        )

    if _matches(_CLEARED_TYPES):
        return AllBlocksCleared(
            backend_id=backend_id,
            medium=medium,
            dp_rank=dp_rank,
            model_name=model_name,
            tenant_id=tenant_id,
            event_id=event_id,
        )

    raise ValueError(f"unknown master event_type/type: {type_raw!r}")


class MooncakeMasterEventSubscriber:
    """Optional SUB client for Mooncake master KV events (fail-open scaffold).

    When ``source`` is ``engine``, :meth:`start` is a no-op. When ``source`` is
    ``master`` or ``both``, start attempts to attach to
    ``master_subscribe_endpoint``. If the endpoint is missing or the Mooncake
    SDK lacks a KV-events API, logs a warning and stays idle — never raises
    into the engine path.
    """

    def __init__(self, config: MooncakeKvEventsConfig) -> None:
        self._config = config
        self._active = False

    @property
    def is_active(self) -> bool:
        """True only when a live master subscription is established."""
        return self._active

    def start(self) -> None:
        """Begin subscribe attempt (or no-op). Never blocks or crashes."""
        if self._config.source == "engine":
            return

        endpoint = self._config.master_subscribe_endpoint
        if not endpoint:
            logger.warning(
                "Mooncake KV events source=%s but master_subscribe_endpoint "
                "is missing; master relay stays idle (fail-open).",
                self._config.source,
            )
            self._active = False
            return

        # Scaffold only: do not open a production ZMQ poll loop until the
        # master publisher / SDK event API is available in this environment.
        if not self._try_attach_master_event_api(endpoint):
            logger.warning(
                "Mooncake master KV-events API unavailable or attach failed "
                "for endpoint %s; master relay stays idle (fail-open). "
                "Scaffold pending Mooncake PR #2214 publisher availability.",
                endpoint,
            )
            self._active = False
            return

        self._active = True

    def stop(self) -> None:
        """Release any subscription resources (no-op when idle)."""
        self._active = False

    def _try_attach_master_event_api(self, endpoint: str) -> bool:
        """Attempt to attach to a master event publisher. Returns True if live.

        Intentionally conservative: returns False unless a known SDK entry
        point exists. Avoids hanging on ZMQ connect to a dead endpoint.
        """
        del endpoint  # reserved for future SUB connect
        try:
            import mooncake  # type: ignore[import-untyped]
        except ImportError:
            return False

        # Probe for an event-API surface that does not exist in current
        # releases; when PR #2214 lands, replace this with a real attach.
        for attr in ("KvEventSubscriber", "kv_events", "subscribe_kv_events"):
            if hasattr(mooncake, attr):
                # Real attach would go here; scaffold treats discovery-only
                # as not yet sufficient for an active poll loop.
                return False
        return False
