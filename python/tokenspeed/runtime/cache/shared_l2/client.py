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

"""A prefill or fused rank's config for the shared L2 pool, which is its Host L2,
and its join of the pool.

``--shared-l2-config`` is a ``SharedL2ClientConfig``. ``wait_until_ready`` says the
rank's hello to the pool's owner at device construction and returns the pool
as the owner described it, a ``JoinedSharedL2Pool``.

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
    """``--shared-l2-config``: every field is required.

    In the JSON, ``scheduler`` is an object of the two write limits,
    ``max_writes_in_flight`` and ``max_keys_per_write``, which are fields of
    their own here, and ``device_names`` is a list of strings; the other
    fields are plain values.
    """

    owner_address: str
    # RDMA devices for each local GPU index, as Mooncake names them.
    device_names: tuple[str, ...]
    # How long a request waits for the owner's answer. One not answered by
    # then has failed.
    request_timeout_ms: int
    join_timeout_s: float
    # Threads reading one load's providers in parallel.
    read_workers: int
    write_workers: int
    # The scheduler's limits: writes in flight, and keys in one write.
    max_writes_in_flight: int
    max_keys_per_write: int

    @classmethod
    def from_json(cls, raw: str) -> SharedL2ClientConfig:
        values = json.loads(raw)
        fields = {
            "owner_address",
            "device_names",
            "request_timeout_ms",
            "join_timeout_s",
            "read_workers",
            "write_workers",
            "scheduler",
        }
        if not isinstance(values, dict) or values.keys() != fields:
            raise ValueError(f"--shared-l2-config needs exactly {sorted(fields)}")
        limits = values["scheduler"]
        limit_fields = {"max_writes_in_flight", "max_keys_per_write"}
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
            owner_address=str(values["owner_address"]),
            device_names=tuple(device_names),
            request_timeout_ms=int(values["request_timeout_ms"]),
            join_timeout_s=float(values["join_timeout_s"]),
            read_workers=int(values["read_workers"]),
            write_workers=int(values["write_workers"]),
            max_writes_in_flight=int(limits["max_writes_in_flight"]),
            max_keys_per_write=int(limits["max_keys_per_write"]),
        )
        if min(config.max_writes_in_flight, config.max_keys_per_write) < 1:
            raise ValueError("--shared-l2-config scheduler needs positive write limits")
        if min(config.read_workers, config.write_workers) < 1:
            raise ValueError(
                "--shared-l2-config read_workers and write_workers must be at least 1"
            )
        if min(config.request_timeout_ms, config.join_timeout_s) <= 0:
            raise ValueError("--shared-l2-config needs positive timeouts")
        return config


@dataclass(frozen=True, slots=True)
class SharedL2Provider:
    """One node's share of the pool: ``num_lcm_blocks`` LCM blocks at
    ``base_address`` of the RDMA session, the pool's LCM blocks from
    ``first_lcm_block`` on."""

    session: str
    base_address: int
    num_lcm_blocks: int
    first_lcm_block: int


@dataclass(frozen=True, slots=True)
class JoinedSharedL2Pool:
    """The pool as the rank's hello described it: the rank's client id there
    and where the bytes are."""

    client_id: int
    providers: tuple[SharedL2Provider, ...]
    lcm_block_bytes: int

    @property
    def num_lcm_blocks(self) -> int:
        return sum(provider.num_lcm_blocks for provider in self.providers)


def wait_until_ready(
    config: SharedL2ClientConfig,
    *,
    geometry: dict[str, Any],
    deployment_id: str,
    connection: OwnerConnection,
) -> JoinedSharedL2Pool:
    """Join the pool as a new client, waiting up to ``join_timeout_s`` for
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
        TimeoutError: The pool did not become ready: the last attempt got no
            reply, or the owner still waited for providers.
        OwnerError: The owner refused the geometry or the deployment.
    """
    client = secrets.randbits(63)
    deadline = time.monotonic() + config.join_timeout_s
    while True:
        try:
            reply = connection.request(
                "hello",
                client=client,
                geometry=geometry,
                deployment_id=deployment_id,
            )
            if reply["ready"]:
                break
            not_ready_reason = (
                f"the owner at {config.owner_address} replied that the pool is not "
                "ready: it still waits for providers"
            )
        except TimeoutError:
            # The owner is not there yet.
            not_ready_reason = f"no reply from the owner at {config.owner_address}"
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"the shared L2 pool did not become ready: {not_ready_reason}"
            )
        time.sleep(_HELLO_RETRY_S)
    joined_pool = JoinedSharedL2Pool(
        client_id=client,
        providers=tuple(
            SharedL2Provider(
                session=provider["session"],
                base_address=int(provider["base_address"]),
                num_lcm_blocks=int(provider["num_lcm_blocks"]),
                first_lcm_block=int(provider["first_lcm_block"]),
            )
            for provider in reply["providers"]
        ),
        lcm_block_bytes=int(reply["lcm_block_bytes"]),
    )
    logger.info(
        f"shared L2 client {client} joined: {len(joined_pool.providers)} providers, "
        f"{joined_pool.num_lcm_blocks} LCM blocks of {joined_pool.lcm_block_bytes} bytes"
    )
    return joined_pool
