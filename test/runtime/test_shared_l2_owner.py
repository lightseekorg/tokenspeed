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

"""The shared L2 pool's owner service from its command line, asked through
``OwnerConnection`` as a rank's hello and a provider ask it, and the provider.

What the owner does with each request is the engine host's: its tests are
``shared_l2`` of ``ts-control`` (``tests/unit/shared_l2_owner.rs``)."""

import os
import signal
import sys
import threading

import pytest
import tokenspeed_engine

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

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
from tokenspeed.runtime.cache.shared_l2.provider import run_provider  # noqa: E402
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection  # noqa: E402
from tokenspeed.runtime.cache.transfer.layout import (  # noqa: E402
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
)


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
        assert client.request("stats")["num_clients"] == 1
        wait_until(lambda: client.request("ping", client=1) == {"ok": True})
    finally:
        client.close()


@pytest.mark.parametrize("named_in_the_environment", [True, False])
def test_a_provider_registers_its_memory_and_joins_the_pool(
    serve, monkeypatch, tmp_path, named_in_the_environment
):
    # The Mooncake library the engine hosts load: the one the environment
    # names, else the installed tokenspeed-mooncake package's.
    if named_in_the_environment:
        found_library = "/opt/libtokenspeed_transfer_engine.so"
        monkeypatch.setenv("TOKENSPEED_MOONCAKE_LIBRARY", found_library)
    else:
        package = tmp_path / "mooncake"
        package.mkdir()
        (package / "__init__.py").touch()
        (package / "libtokenspeed_transfer_engine.so").touch()
        found_library = str(package / "libtokenspeed_transfer_engine.so")
        monkeypatch.delenv("TOKENSPEED_MOONCAKE_LIBRARY", raising=False)
        monkeypatch.delitem(sys.modules, "mooncake", raising=False)
        monkeypatch.syspath_prepend(tmp_path)

    class TransferEngine:
        """The engine host's transfer engine, which needs an RDMA device, as
        the provider calls it."""

        registered = {}

        def __init__(self, *, library, hostname, protocol, device_filter):
            assert (library, hostname, protocol, device_filter) == (
                found_library,
                "127.0.0.1",
                "rdma",
                "mlx5_0",
            )
            self.session_id = "127.0.0.1:12345"

        def register(self, address, length):
            self.registered[address] = length

        def unregister(self, address):
            del self.registered[address]

    monkeypatch.setattr(tokenspeed_engine, "TransferEngine", TransferEngine)
    owner = serve(joined_providers=0)
    stop = threading.Event()
    provider = threading.Thread(
        target=run_provider,
        kwargs=dict(
            owner_address=owner.address,
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
        ((base, size),) = TransferEngine.registered.items()
        assert size == 8 * LCM_BYTES + 1
        assert owner.admin.request("hello", **hello)["providers"] == [
            {
                "session": "127.0.0.1:12345",
                "base_address": base,
                "num_lcm_blocks": 8,
                "first_lcm_block": 0,
            }
        ]
    finally:
        stop.set()
        provider.join()
    assert TransferEngine.registered == {}


def test_a_provider_that_finds_no_mooncake_library_does_not_start(
    monkeypatch, tmp_path
):
    provider = dict(
        owner_address="tcp://127.0.0.1:1",
        hostname="127.0.0.1",
        device_name="mlx5_0",
        size_bytes=LCM_BYTES,
        stop=threading.Event(),
    )
    monkeypatch.delenv("TOKENSPEED_MOONCAKE_LIBRARY", raising=False)
    # No tokenspeed-mooncake is installed: the import system finds no package.
    monkeypatch.setitem(sys.modules, "mooncake", None)
    with pytest.raises(RuntimeError, match="provider needs tokenspeed-mooncake"):
        run_provider(**provider)
    # A Mooncake package without the library.
    (tmp_path / "mooncake").mkdir()
    (tmp_path / "mooncake" / "__init__.py").touch()
    monkeypatch.delitem(sys.modules, "mooncake")
    monkeypatch.syspath_prepend(tmp_path)
    with pytest.raises(
        RuntimeError, match="libtokenspeed_transfer_engine.so is missing"
    ):
        run_provider(**provider)


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_the_owner_service_runs_from_the_command_line(serve, stop_signal):
    owner = serve(joined_providers=0)
    hello = dict(client=1, geometry=GEOMETRY, deployment_id=deployment_id())

    # It serves, and waits for its provider.
    assert owner.admin.request("hello", **hello) == {"ok": True, "ready": False}
    owner.add_provider("node0", 8)
    # The pool it then builds is the scheduler's, over the provider's blocks.
    assert owner.admin.request("hello", **hello) == {
        "ok": True,
        "ready": True,
        "providers": [
            {
                "session": "node0",
                "base_address": 1 << 40,
                "num_lcm_blocks": 8,
                "first_lcm_block": 0,
            }
        ],
        "lcm_block_bytes": LCM_BYTES,
    }
    assert owner.admin.request("ping", client=1) == {"ok": True}
    assert owner.stats()["num_lcm_blocks"] == 8

    # Either signal stops it in order.
    owner.process.send_signal(stop_signal)
    assert owner.process.wait(timeout=30) == 0


def test_the_owner_serves_the_schedulers_pool_over_the_groups_a_rank_reports(serve):
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
        view_num_layers=(1,),
        group_specs=specs,
        cache_groups=[
            cache_group_config(spec, total_pages=0, cache_blocks_per_lcm_block=packing)
            for spec, packing in zip(specs, (4, 8, 8))
        ],
        prefix_granularity=2,
    ).geometry
    # The owner over the scheduler's pool, of 16 LCM blocks, which it builds
    # from the geometry the first rank's hello carries.
    owner = serve(joined_providers=0)
    owner.admin.request(
        "add_provider",
        session="node0",
        base_address=0,
        bytes=16 * geometry["lcm_block_bytes"],
    )
    for client in (1, 2):
        assert owner.admin.request(
            "hello", client=client, geometry=geometry, deployment_id=deployment_id()
        )["ready"]

    # Six pages of every history group and the checkpoint at their end.
    prompt = list(range(1, 13))
    hashes, _, keys = prompt_keys(prompt)
    window = [(0, 2, page, 0) for page in hashes]
    reserved = owner.admin.request(
        "reserve", client=1, keys=keys + window, buckets=[0] * 13
    )
    # An LCM block holds one KV page, three States or two window pages, as
    # the rank's Host tier packs them; each group numbers its own pages.
    assert reserved["pages"] == [1, 2, 3, 4, 5, 6, 19, 15, 16, 17, 18, 19, 20]
    owner.admin.request(
        "commit",
        client=1,
        ticket=reserved["ticket"],
        checkpoints=write_fields(prompt)[3],
    )

    # The owner matches as the rank's scheduler does. Of the window group
    # a hit needs, and pins, the sink's page and the last page.
    ((_, num_tokens, _, _, pages),) = owner.admin.request(
        "lookup", client=2, lookups=[lookup_fields(prompt)]
    )["hits"]
    assert num_tokens == len(prompt)
    assert [tuple(key) for key, _, _ in pages] == keys + [window[0], window[-1]]
    assert owner.stats()["pinned_blocks"] == len(keys) + 2
