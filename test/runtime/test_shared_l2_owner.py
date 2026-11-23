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

"""The shared L2 pool's owner service over its real wire: a ZMQ ROUTER
served from a thread, asked through ``OwnerConnection``. The pool behind it
keeps the books of the scheduler's ``SharedL2Pool`` in Python."""

import copy
import functools
import os
import subprocess
import sys
import threading
import time
import types

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from test.runtime.shared_l2_test_utils import (  # noqa: E402
    GEOMETRY,
    LCM_BYTES,
    ServedOwner,
    deployment_id,
    lookup_fields,
    prompt_keys,
    wait_until,
    write_fields,
)

from tokenspeed.runtime.cache.shared_l2.layout import SharedL2Layout  # noqa: E402
from tokenspeed.runtime.cache.shared_l2.owner import (  # noqa: E402
    SharedL2Owner,
    make_pool,
)
from tokenspeed.runtime.cache.shared_l2.provider import run_provider  # noqa: E402
from tokenspeed.runtime.cache.shared_l2.wire import (  # noqa: E402
    OwnerConnection,
    OwnerError,
)
from tokenspeed.runtime.cache.transfer.layout import (  # noqa: E402
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
)
from tokenspeed.runtime.utils.network import get_free_port  # noqa: E402

PROMPT = [1, 2, 3, 4, 5]


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


@pytest.fixture
def connect():
    """Opens connections that say hello as the client they are asked to be."""
    connections = []

    def join(owner: ServedOwner, client: int) -> OwnerConnection:
        connections.append(OwnerConnection(owner.address, timeout_ms=5000))
        connections[-1].request(
            "hello", client=client, geometry=GEOMETRY, deployment_id=deployment_id()
        )
        return connections[-1]

    yield join
    for connection in connections:
        connection.close()


def store(connection: OwnerConnection, client: int, prompt) -> list:
    """Reserve and commit ``prompt``'s blocks and checkpoint; its keys."""
    keys, buckets, _, publications = write_fields(prompt)
    ticket = connection.request("reserve", client=client, keys=keys, buckets=buckets)[
        "ticket"
    ]
    assert connection.request(
        "commit", client=client, ticket=ticket, publications=publications
    )["committed"]
    return keys


def probe(connection: OwnerConnection, client: int, prompt) -> list:
    (hit,) = connection.request(
        "probe_many", client=client, lookups=[lookup_fields(prompt)]
    )["hits"]
    return hit


def test_a_hello_is_not_ready_until_every_provider_joined(serve):
    owner = serve(num_providers=2, joined_providers=1, provider_blocks=8)
    hello = dict(geometry=GEOMETRY, deployment_id=deployment_id())

    assert owner.admin.request("hello", client=1, **hello) == {
        "ok": True,
        "ready": False,
    }
    # It has not joined either: nothing is kept for it.
    with pytest.raises(OwnerError, match="has not joined the pool"):
        owner.admin.request("ping", client=1)

    owner.add_provider("node1", 4)
    reply = owner.admin.request("hello", client=1, **hello)
    assert reply["ready"] and reply["lcm_bytes"] == LCM_BYTES
    # A provider's LCM blocks follow those of the providers before it.
    assert reply["providers"] == [
        {"session": "node0", "base": 1 << 40, "lcm_blocks": 8, "start": 0},
        {"session": "node1", "base": 1 << 40, "lcm_blocks": 4, "start": 8},
    ]
    assert owner.pool.num_lcm_blocks == 12 and owner.pool.geometry == GEOMETRY
    assert owner.admin.request("ping", client=1) == {"ok": True}


def test_the_first_client_sets_the_geometry_and_deployment_later_ones_match(
    serve, connect
):
    owner = serve()
    connect(owner, 1)
    connect(owner, 2)
    assert len(owner.pools) == 1

    other_layout = copy.deepcopy(GEOMETRY)
    other_layout["groups"][0]["host_packing"] = 4
    with pytest.raises(OwnerError, match="geometry differs"):
        owner.admin.request(
            "hello", client=3, geometry=other_layout, deployment_id=deployment_id()
        )
    # The same layout holding another checkpoint's KV.
    with pytest.raises(OwnerError, match="deployment differs"):
        owner.admin.request(
            "hello", client=3, geometry=GEOMETRY, deployment_id=deployment_id("def")
        )
    with pytest.raises(OwnerError, match="geometry differs"):
        owner.admin.request(
            "hello",
            client=3,
            geometry={**GEOMETRY, "lcm_bytes": 128},
            deployment_id=deployment_id(),
        )
    with pytest.raises(OwnerError, match="has not joined the pool"):
        owner.admin.request("ping", client=3)
    assert owner.admin.request("stats")["clients"] == 2


def test_providers_join_before_the_pool_runs(serve, connect):
    owner = serve(num_providers=2, joined_providers=1)
    owner.add_provider("node1", 8)
    with pytest.raises(OwnerError, match="every expected provider"):
        owner.add_provider("node2", 8)
    connect(owner, 1)
    with pytest.raises(OwnerError, match="cannot join a running pool"):
        owner.add_provider("node2", 8)


def test_the_first_client_sets_the_providers_shares(serve):
    owner = serve(num_providers=2, joined_providers=0)
    # Providers join with their bytes, whatever a block will hold.
    owner.admin.request("provider", session="node0", base=4096, bytes=200)
    owner.admin.request("provider", session="node1", base=8192, bytes=LCM_BYTES)
    hello = dict(geometry=GEOMETRY, deployment_id=deployment_id())

    reply = owner.admin.request("hello", client=1, **hello)

    # The pool is cut into the first client's LCM blocks; a provider's bytes
    # short of a whole block are left out.
    assert reply["ready"] and reply["lcm_bytes"] == LCM_BYTES
    assert reply["providers"] == [
        {"session": "node0", "base": 4096, "lcm_blocks": 3, "start": 0},
        {"session": "node1", "base": 8192, "lcm_blocks": 1, "start": 3},
    ]
    assert owner.pool.num_lcm_blocks == 4
    assert owner.admin.request("stats")["providers"] == reply["providers"]
    # From then on that block size is the pool's.
    with pytest.raises(OwnerError, match="geometry differs"):
        owner.admin.request(
            "hello",
            client=2,
            geometry={**GEOMETRY, "lcm_bytes": 128},
            deployment_id=deployment_id(),
        )
    assert owner.admin.request("hello", client=2, **hello)["providers"] == (
        reply["providers"]
    )


def test_a_provider_smaller_than_a_block_stops_the_pool_from_being_built(serve):
    owner = serve(num_providers=2, joined_providers=0)
    owner.admin.request("provider", session="node0", base=4096, bytes=10 * LCM_BYTES)
    owner.admin.request("provider", session="small", base=8192, bytes=LCM_BYTES - 1)
    hello = dict(client=1, geometry=GEOMETRY, deployment_id=deployment_id())

    # Every hello says which provider is too small, and no pool is built.
    for _ in range(2):
        with pytest.raises(
            OwnerError,
            match="provider small holds 63 bytes, less than one LCM block of 64 bytes",
        ):
            owner.admin.request("hello", **hello)
    assert owner.pools == []
    with pytest.raises(OwnerError, match="pool is not ready"):
        owner.admin.request("stats")
    with pytest.raises(OwnerError, match="has not joined the pool"):
        owner.admin.request("ping", client=1)


def test_probe_many_answers_every_lookup_in_order_and_pins_each_hit(serve, connect):
    owner = serve()
    writer, reader = connect(owner, 1), connect(owner, 2)
    keys = store(writer, 1, PROMPT)

    hits = reader.request(
        "probe_many",
        client=2,
        lookups=[lookup_fields(PROMPT + [6, 7]), lookup_fields([9, 9, 9])],
    )["hits"]

    ticket, num_tokens, kv_tokens, tail, slots = hits[0]
    assert ticket != 0 and (num_tokens, kv_tokens) == (5, 5)
    assert tail == list(prompt_keys(PROMPT)[1])
    # Every slot is a key with its bucket and its pool page.
    assert [(tuple(key), bucket) for key, bucket, _ in slots] == [
        (key, 0) for key in keys
    ]
    assert all(page > 0 for _, _, page in slots)
    assert hits[1] == [0, 0, 0, None, []]
    assert owner.stats()["pinned_blocks"] == len(keys)
    # The prompt's tokens reached the pool as the int32 bytes they left as.
    hashes, tokens, max_tokens = lookup_fields([9, 9, 9])
    assert owner.pool.received["probe"] == (hashes, tokens, max_tokens)


def test_a_write_is_reserved_then_committed(serve, connect):
    owner = serve(provider_blocks=4)
    first, second = connect(owner, 1), connect(owner, 2)
    keys, buckets, _, publications = write_fields(PROMPT)
    reserve = dict(keys=keys, buckets=buckets)

    reply = first.request("reserve", client=1, **reserve)
    assert reply["ticket"] != 0 and all(page > 0 for page in reply["pages"])
    # The keys reached the pool as the tuples they were, each with its bucket.
    assert owner.pool.received["reserve"] == (keys, buckets)
    # Another client gets no page for a block that is being written.
    assert second.request("reserve", client=2, **reserve) == {
        "ok": True,
        "ticket": 0,
        "pages": [-1] * len(keys),
    }
    assert owner.stats()["reserved_blocks"] == len(keys)

    assert first.request(
        "commit", client=1, ticket=reply["ticket"], publications=publications
    )["committed"]
    assert owner.pool.received["commit"] == publications
    stats = owner.stats()
    assert (stats["reserved_blocks"], stats["cached_blocks"]) == (0, len(keys))
    # A ticket that reserves nothing commits nothing.
    assert not first.request(
        "commit", client=1, ticket=reply["ticket"], publications=publications
    )["committed"]

    # One free page is left of four: the second new block has no room.
    more = [(0, 0, "x", 0), (0, 0, "y", 0)]
    assert (
        first.request("reserve", client=1, keys=more, buckets=[0, 0])["pages"][1] == -2
    )


def test_a_release_unpins_what_its_clients_lookup_hit(serve, connect):
    owner = serve()
    writer, reader = connect(owner, 1), connect(owner, 2)
    keys = store(writer, 1, PROMPT)

    first = probe(reader, 2, PROMPT)[0]
    second = probe(reader, 2, PROMPT)[0]
    assert first != second
    assert owner.stats()["pinned_blocks"] == 2 * len(keys)

    assert reader.request("release", client=2, ticket=first) == {"ok": True}
    assert owner.stats()["pinned_blocks"] == len(keys)
    # A ticket is its client's: another client's release does not touch it.
    writer.request("release", client=1, ticket=second)
    assert owner.stats()["pinned_blocks"] == len(keys)


def test_a_silent_client_loses_what_it_held_and_is_told_so(serve, connect):
    owner = serve(client_timeout_s=0.5)
    silent, alive = connect(owner, 1), connect(owner, 2)
    store(alive, 2, PROMPT)
    probe(silent, 1, PROMPT)
    silent.request("reserve", client=1, keys=[(0, 0, "new", 0)], buckets=[0])
    assert owner.stats()["pinned_blocks"] > 0 and owner.stats()["reserved_blocks"] == 1

    deadline = time.monotonic() + 5
    while not owner.pool.disconnected:
        assert time.monotonic() < deadline
        alive.request("ping", client=2)
        time.sleep(0.02)

    assert owner.pool.disconnected == [1]
    stats = owner.stats()
    assert (stats["pinned_blocks"], stats["reserved_blocks"]) == (0, 0)
    # Its next request tells it; the client that kept talking is still in.
    with pytest.raises(OwnerError, match="has not joined the pool"):
        silent.request("release", client=1, ticket=1)
    assert alive.request("ping", client=2) == {"ok": True}
    assert owner.admin.request("stats")["clients"] == 1


def test_touch_clear_and_stats(serve, connect):
    owner = serve()
    client = connect(owner, 1)
    keys = store(client, 1, PROMPT)

    client.request("touch", client=1, keys=keys[:2], buckets=[0, 0])
    assert owner.pool.touched == keys[:2]
    # The pool holds one bucket per group so far.
    with pytest.raises(OwnerError, match="bucket 0 only"):
        client.request("touch", client=1, keys=keys[:1], buckets=[1])

    ticket = probe(client, 1, PROMPT)[0]
    assert owner.admin.request("clear") == {"ok": True, "cleared": False}
    client.request("release", client=1, ticket=ticket)
    assert owner.admin.request("clear") == {"ok": True, "cleared": True}

    reply = owner.admin.request("stats")
    assert reply["stats"]["cached_blocks"] == 0
    assert reply["clients"] == 1 and len(reply["providers"]) == 1


def test_a_failed_request_gets_an_error_and_the_owner_keeps_serving(serve, connect):
    owner = serve()
    with pytest.raises(OwnerError, match="pool is not ready"):
        owner.admin.request("stats")
    client = connect(owner, 1)
    with pytest.raises(OwnerError, match="unknown shared L2 request 'steal'"):
        client.request("steal", client=1)
    # A pool error fails the request, not the owner.
    owner.pool.failing.add("reserve")
    with pytest.raises(OwnerError, match="reserve failed"):
        client.request("reserve", client=1, keys=[], buckets=[])
    assert client.request("ping", client=1) == {"ok": True}


def test_a_stalled_owner_times_a_request_out_and_answers_the_next(serve, connect):
    owner = serve()
    connect(owner, 1)
    client = OwnerConnection(owner.address, timeout_ms=200)
    try:
        owner.stall()
        with pytest.raises(TimeoutError):
            client.request("ping", client=1)
        owner.resume()
        # The late answer to the first ping is not taken for this request's.
        assert client.request("stats")["clients"] == 1
        wait_until(lambda: client.request("ping", client=1) == {"ok": True})
    finally:
        client.close()


def test_a_provider_registers_its_memory_and_joins_the_pool(serve, monkeypatch):
    class Engine:
        """Mooncake's engine, which needs an RDMA device, as the provider
        calls it."""

        registered = {}

        def initialize(self, hostname, metadata, protocol, device_name):
            assert (hostname, protocol, device_name) == ("127.0.0.1", "rdma", "mlx5_0")
            return 0

        def register_memory(self, base, size):
            self.registered[base] = size
            return 0

        def get_rpc_port(self):
            return 12345

        def unregister_memory(self, base):
            del self.registered[base]
            return 0

    engine = types.ModuleType("mooncake.engine")
    engine.TransferEngine = Engine
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine)
    owner = serve(joined_providers=0)
    stop = threading.Event()
    provider = threading.Thread(
        target=run_provider,
        kwargs=dict(
            owner=owner.address,
            hostname="127.0.0.1",
            device_name="mlx5_0",
            size_bytes=8 * LCM_BYTES + 1,
            stop=stop,
        ),
    )
    provider.start()
    try:
        hello = dict(client=1, geometry=GEOMETRY, deployment_id=deployment_id())
        wait_until(lambda: owner.admin.request("hello", **hello)["ready"])
        # The pool is the provider's registered memory, in whole LCM blocks.
        ((base, size),) = Engine.registered.items()
        assert size == 8 * LCM_BYTES + 1
        assert owner.admin.request("hello", **hello)["providers"] == [
            {"session": "127.0.0.1:12345", "base": base, "lcm_blocks": 8, "start": 0}
        ]
    finally:
        stop.set()
        provider.join()
    assert Engine.registered == {}


def test_the_owner_service_runs_from_the_command_line():
    address = f"tcp://127.0.0.1:{get_free_port()}"
    service = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tokenspeed.runtime.cache.shared_l2",
            "owner",
            f"--address={address}",
            "--providers=1",
            "--headroom-lcm-blocks=0",
            "--client-timeout-s=60",
            "--checkpoint-retention-requests=0",
            "--checkpoint-retention-leaf-distance=0",
        ]
    )
    connection = OwnerConnection(address, timeout_ms=30000)
    try:
        # It serves, and waits for its provider.
        assert connection.request(
            "hello", client=1, geometry=GEOMETRY, deployment_id=deployment_id()
        ) == {"ok": True, "ready": False}
        service.terminate()
        assert service.wait(timeout=30) == 0
    finally:
        connection.close()
        service.kill()


def test_the_owner_serves_the_schedulers_pool_over_the_groups_a_rank_reports():
    pytest.importorskip("tokenspeed_engine.scheduler")
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
        cache_group_config,
    )

    # A rank's groups: full KV, State, and a window of three tokens that
    # also keeps the prompt's first two (an attention sink).
    specs = (
        CacheGroupSpec(
            "kv",
            "full_history",
            rows_per_page=1,
            entry_stride_tokens=2,
            replayable=False,
        ),
        CacheGroupSpec("state", "state", replayable=False),
        CacheGroupSpec(
            "window",
            "sliding_window",
            rows_per_page=1,
            entry_stride_tokens=2,
            sliding_window_tokens=3,
            sink_tokens=2,
            replayable=False,
        ),
    )
    geometry = SharedL2Layout.build(
        CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                CacheGroupLayout("kv", 4, (CacheField("k", 0, 0, 32, 24),)),
                CacheGroupLayout("state", 8, (CacheField("s", 0, 0, 16, 8),)),
                CacheGroupLayout("window", 8, (CacheField("w", 0, 0, 16, 12),)),
            ),
            buffers=(bytearray(1),),
            consumers=(("k", "s", "w"),),
        ),
        view_layers=(1,),
        group_specs=specs,
        cache_groups=[
            cache_group_config(spec, total_pages=0, cache_blocks_per_lcm_block=packing)
            for spec, packing in zip(specs, (4, 8, 8))
        ],
        prefix_granularity=2,
    ).geometry
    # The owner over the scheduler's pool, of 16 LCM blocks; its requests are
    # handled here as its service hands them over.
    owner = SharedL2Owner(
        address=f"tcp://127.0.0.1:{get_free_port()}",
        num_providers=1,
        client_timeout_s=60.0,
        make_pool=functools.partial(
            make_pool,
            headroom_lcm_blocks=0,
            checkpoint_retention_requests=0,
            checkpoint_retention_leaf_distance=0,
        ),
    )
    try:
        owner.handle(
            {
                "op": "provider",
                "session": "node0",
                "base": 0,
                "bytes": 16 * geometry["lcm_bytes"],
            }
        )
        for client in (1, 2):
            assert owner.handle(
                {
                    "op": "hello",
                    "client": client,
                    "geometry": geometry,
                    "deployment_id": deployment_id(),
                }
            )["ready"]

        # Six pages of every history group and the checkpoint at their end.
        prompt = list(range(1, 13))
        hashes, _, keys = prompt_keys(prompt)
        window = [(0, 2, page, 0) for page in hashes]
        reserved = owner.handle(
            {"op": "reserve", "client": 1, "keys": keys + window, "buckets": [0] * 13}
        )
        # An LCM block holds one KV page, three States or two window pages, as
        # the rank's Host tier packs them; each group numbers its own pages.
        assert reserved["pages"] == [1, 2, 3, 4, 5, 6, 19, 15, 16, 17, 18, 19, 20]
        assert owner.handle(
            {
                "op": "commit",
                "client": 1,
                "ticket": reserved["ticket"],
                "publications": write_fields(prompt)[3],
            }
        )["committed"]

        # The owner matches as the rank's scheduler does. Of the window group
        # a hit needs, and pins, the sink's page and the last page.
        ((_, num_tokens, _, _, slots),) = owner.handle(
            {"op": "probe_many", "client": 2, "lookups": [lookup_fields(prompt)]}
        )["hits"]
        assert num_tokens == len(prompt)
        assert [key for key, _, _ in slots] == keys + [window[0], window[-1]]
        assert owner.handle({"op": "stats"})["stats"]["pinned_blocks"] == len(keys) + 2
    finally:
        owner.close()
