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

"""A prefill rank's settings for the shared L2 pool, which is its Host L2, and
its join of the pool.

``--shared-l2-config`` is a ``SharedL2ClientConfig``. ``join_pool`` says the
rank's hello to the pool's owner at device construction and returns the pool
as the owner described it, a ``SharedL2PoolView``.

From then on the rank's client of the owner runs in the engine host
(``shared_l2/client.rs`` of ``ts-control``): the scheduler keeps a proxy Host
tier that mirrors the pool, and the client carries its lookups, writes and
notices to the owner under the id the hello chose. The host starts it from
``rank.join_shared_l2_pool``.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection

logger = logging.getLogger(__name__)

# Between hellos while the pool is not ready.
_HELLO_RETRY_S = 1.0


@dataclass(frozen=True, slots=True)
class SharedL2ClientConfig:
    """``--shared-l2-config``: every field is required."""

    owner: str
    # RDMA devices for each local GPU index, as Mooncake names them.
    device_names: tuple[str, ...]
    # How long a request waits for the owner's answer. One not answered by
    # then has failed.
    request_timeout_ms: int
    connect_timeout_s: float
    # Threads reading one load's providers in parallel.
    load_workers: int
    write_workers: int
    # The scheduler's limits: writes in flight, and keys in one write.
    max_writes: int
    max_write_keys: int

    @classmethod
    def from_json(cls, raw: str) -> SharedL2ClientConfig:
        values = json.loads(raw)
        fields = {
            "owner",
            "device_names",
            "request_timeout_ms",
            "connect_timeout_s",
            "load_workers",
            "write_workers",
            "scheduler",
        }
        if not isinstance(values, dict) or values.keys() != fields:
            raise ValueError(f"--shared-l2-config needs exactly {sorted(fields)}")
        limits = values["scheduler"]
        limit_fields = {"max_writes", "max_write_keys"}
        if not isinstance(limits, dict) or limits.keys() != limit_fields:
            raise ValueError(
                f"--shared-l2-config scheduler needs exactly {sorted(limit_fields)}"
            )
        device_names = values["device_names"]
        if not isinstance(device_names, list) or not all(
            isinstance(name, str) for name in device_names
        ):
            raise ValueError(
                "--shared-l2-config device_names must be a list of strings"
            )
        config = cls(
            owner=str(values["owner"]),
            device_names=tuple(device_names),
            request_timeout_ms=int(values["request_timeout_ms"]),
            connect_timeout_s=float(values["connect_timeout_s"]),
            load_workers=int(values["load_workers"]),
            write_workers=int(values["write_workers"]),
            max_writes=int(limits["max_writes"]),
            max_write_keys=int(limits["max_write_keys"]),
        )
        if min(config.max_writes, config.max_write_keys) < 1:
            raise ValueError("--shared-l2-config scheduler needs positive write limits")
        if min(config.load_workers, config.write_workers) < 1:
            raise ValueError(
                "--shared-l2-config needs at least one load and one write worker"
            )
        if min(config.request_timeout_ms, config.connect_timeout_s) <= 0:
            raise ValueError("--shared-l2-config needs positive timeouts")
        return config


@dataclass(frozen=True, slots=True)
class SharedL2Provider:
    """One node's share of the pool: ``lcm_blocks`` LCM blocks at ``base`` of
    the RDMA session, the pool's LCM blocks from ``start`` on."""

    session: str
    base: int
    lcm_blocks: int
    start: int


@dataclass(frozen=True, slots=True)
class SharedL2PoolView:
    """The pool as the rank's hello described it: the rank's client id there
    and where the bytes are."""

    client: int
    providers: tuple[SharedL2Provider, ...]
    lcm_bytes: int

    @property
    def num_lcm_blocks(self) -> int:
        return sum(provider.lcm_blocks for provider in self.providers)


def join_pool(
    config: SharedL2ClientConfig,
    *,
    geometry: dict[str, Any],
    deployment_id: str,
    connection: OwnerConnection,
) -> SharedL2PoolView:
    """Join the pool as a new client, waiting up to ``connect_timeout_s`` for
    it.

    Args:
        config: ``--shared-l2-config``.
        geometry: The rank's cache geometry, as ``owner`` describes it.
        deployment_id: This rank's ``storage_deployment_id``; the owner
            serves one.
        connection: To the owner, with the config's request timeout.

    Returns:
        The pool, with the id the rank's client has at the owner.

    Raises:
        TimeoutError: The pool did not become ready.
        OwnerError: The owner refused the geometry or the deployment.
    """
    client = secrets.randbits(63)
    deadline = time.monotonic() + config.connect_timeout_s
    while True:
        try:
            reply = connection.request(
                "hello",
                client=client,
                geometry=geometry,
                deployment_id=deployment_id,
            )
        except TimeoutError:
            # The owner is not there yet.
            reply = {"ready": False}
        # Not ready while the owner still waits for providers.
        if reply["ready"]:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("the shared L2 pool did not become ready")
        time.sleep(_HELLO_RETRY_S)
    pool = SharedL2PoolView(
        client=client,
        providers=tuple(
            SharedL2Provider(
                session=provider["session"],
                base=int(provider["base"]),
                lcm_blocks=int(provider["lcm_blocks"]),
                start=int(provider["start"]),
            )
            for provider in reply["providers"]
        ),
        lcm_bytes=int(reply["lcm_bytes"]),
    )
    logger.info(
        f"shared L2 client {client} joined: {len(pool.providers)} providers, "
        f"{pool.num_lcm_blocks} LCM blocks of {pool.lcm_bytes} bytes"
    )
    return pool
