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

"""A rank's settings for the shared L2 pool and its join of the pool, against
the real owner service, run from its command line.

What the rank does once it has joined (lookups, writes, notices) is the engine
host's: its tests are ``shared_l2`` of ``ts-control``."""

import copy
import json
import os
import sys
import threading

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from test.runtime.shared_l2_test_utils import (  # noqa: E402
    GEOMETRY,
    LCM_BYTES,
    ServedOwner,
    deployment_id,
)

from tokenspeed.runtime.cache.shared_l2.client import (  # noqa: E402
    JoinedSharedL2Pool,
    SharedL2ClientConfig,
    SharedL2Provider,
    wait_until_ready,
)
from tokenspeed.runtime.cache.shared_l2.wire import (  # noqa: E402
    OwnerConnection,
    OwnerError,
)


def config_fields(address: str) -> dict:
    return {
        "owner_address": address,
        "device_names": [""],
        "request_timeout_ms": 1000,
        "join_timeout_s": 5,
        "read_workers": 1,
        "write_workers": 1,
        "scheduler": {"max_writes_in_flight": 2, "max_keys_per_write": 64},
    }


@pytest.fixture
def serve():
    """Starts owners, each with its provider joined unless told otherwise."""
    owners = []

    def start(**settings) -> ServedOwner:
        owners.append(ServedOwner(**settings))
        return owners[-1]

    yield start
    for owner in owners:
        owner.close()


def join(owner: ServedOwner, *, geometry=GEOMETRY, **settings) -> JoinedSharedL2Pool:
    """Join a rank to an owner's pool, as the device build does."""
    fields = {**config_fields(owner.address), **settings}
    config = SharedL2ClientConfig.from_json(json.dumps(fields))
    connection = OwnerConnection(owner.address, timeout_ms=config.request_timeout_ms)
    try:
        return wait_until_ready(
            config,
            geometry=geometry,
            deployment_id=deployment_id(),
            connection=connection,
        )
    finally:
        connection.close()


def test_a_rank_joins_once_the_pool_is_ready(serve):
    owner = serve(joined_providers=0)
    late = threading.Timer(0.15, owner.add_provider, args=("node0", 8))
    late.start()

    pool = join(owner)

    late.join()
    assert pool.providers == (
        SharedL2Provider(
            session="node0", base_address=1 << 40, num_lcm_blocks=8, first_lcm_block=0
        ),
    )
    assert (pool.lcm_block_bytes, pool.num_lcm_blocks) == (LCM_BYTES, 8)
    assert owner.admin.request("stats")["num_clients"] == 1
    # The owner knows the rank by the id the join chose, which the rank's
    # client in the host goes on with.
    assert 0 <= pool.client_id < 1 << 63
    assert owner.admin.request("ping", client=pool.client_id)["ok"]
    # Every join is a client of its own.
    assert join(owner).client_id != pool.client_id
    assert owner.admin.request("stats")["num_clients"] == 2


def test_a_rank_does_not_start_without_a_pool_it_can_use(serve):
    # The error says what the last attempt met: an owner that still waits
    # for providers, or no owner answering.
    waiting = serve(joined_providers=0)
    with pytest.raises(
        TimeoutError,
        match=f"the owner at {waiting.address} replied that the pool is not ready",
    ):
        join(waiting, join_timeout_s=0.2)
    silent = serve()
    silent.stall()
    with pytest.raises(
        TimeoutError, match=f"no reply from the owner at {silent.address}"
    ):
        join(silent, request_timeout_ms=100, join_timeout_s=0.2)

    owner = serve()
    join(owner)
    # The pool caches for another layout than this rank's.
    other = copy.deepcopy(GEOMETRY)
    other["prefix_granularity"] = 4
    with pytest.raises(OwnerError, match="geometry differs"):
        join(owner, geometry=other)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda fields: fields.pop("request_timeout_ms"), "needs exactly"),
        (lambda fields: fields.update(namespace="model"), "needs exactly"),
        (
            lambda fields: fields["scheduler"].pop("max_writes_in_flight"),
            "scheduler needs exactly",
        ),
        (
            lambda fields: fields["scheduler"].update(max_keys_per_write=0),
            "positive write limits",
        ),
        (lambda fields: fields.update(write_workers=0), "must be at least 1"),
        (lambda fields: fields.update(request_timeout_ms=0), "positive timeouts"),
        (lambda fields: fields.update(join_timeout_s=-1), "positive timeouts"),
        (lambda fields: fields.update(device_names="mlx5_0"), "list of strings"),
        (lambda fields: fields.update(device_names=[0]), "list of strings"),
    ],
)
def test_a_client_config_names_every_field_and_usable_values(change, message):
    fields = config_fields("tcp://127.0.0.1:1")
    change(fields)
    with pytest.raises(ValueError, match=message):
        SharedL2ClientConfig.from_json(json.dumps(fields))


def test_a_client_config_keeps_every_field():
    config = SharedL2ClientConfig.from_json(
        json.dumps(config_fields("tcp://127.0.0.1:1"))
    )
    assert config == SharedL2ClientConfig(
        owner_address="tcp://127.0.0.1:1",
        device_names=("",),
        request_timeout_ms=1000,
        join_timeout_s=5.0,
        read_workers=1,
        write_workers=1,
        max_writes_in_flight=2,
        max_keys_per_write=64,
    )
