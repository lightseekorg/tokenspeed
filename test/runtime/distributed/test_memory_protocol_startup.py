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

"""Exercise the memory protocol handshake with real multi-process Gloo."""

from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.engine.memory_protocol import gather_collective_addresses

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo"
)


def _rendezvous(rank, rendezvous, protocols, addresses, legacy_ranks):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=len(protocols),
        timeout=timedelta(seconds=15),
    )
    try:
        if rank in legacy_ranks:
            # The old Python code passes the gathered objects to Rust as-is.
            # The old host can only extract Vec<String> from that result.
            gathered = [None] * len(protocols)
            dist.all_gather_object(gathered, addresses[rank], group=dist.group.WORLD)
            if 3 in protocols:
                assert (3, addresses[protocols.index(3)]) in gathered
            else:
                assert gathered == list(addresses)
        elif len(set(protocols)) > 1:
            with pytest.raises(
                RuntimeError,
                match="Memory protocol mismatch before serving",
            ) as refused:
                gather_collective_addresses(
                    dist.group.WORLD,
                    addresses[rank],
                    replica_memory_transitions=protocols[rank] == 3,
                )
            assert str(list(protocols)) in str(refused.value)
        else:
            assert gather_collective_addresses(
                dist.group.WORLD,
                addresses[rank],
                replica_memory_transitions=protocols[rank] == 3,
            ) == list(addresses)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("protocols", "hosts", "legacy_ranks"),
    [
        pytest.param((2, 3), (1, 2), (), id="pp2-mixed-v2-ingress"),
        pytest.param((3, 2), (1, 2), (), id="pp2-mixed-v3-ingress"),
        pytest.param((2, 3), (1, 2), (0,), id="pp2-mixed-legacy-ingress"),
        pytest.param((2, 2, 3, 3), (1, 1, 2, 2), (0, 1), id="ep4-mixed-legacy"),
        pytest.param((2, 2), (1, 2), (0,), id="pp2-legacy-compatible-v2"),
        pytest.param((3, 3), (1, 2), (), id="pp2-all-v3"),
    ],
)
def test_memory_protocol_startup_rendezvous(tmp_path, protocols, hosts, legacy_ranks):
    # Distinct advertised hosts model one PP stage or one EP ingress per
    # machine; only the existing Gloo rendezvous is needed for these tests.
    addresses = tuple(
        f"192.0.2.{host}:{7800 + rank}" for rank, host in enumerate(hosts)
    )
    mp.spawn(
        _rendezvous,
        args=((tmp_path / "rendezvous").as_uri(), protocols, addresses, legacy_ranks),
        nprocs=len(protocols),
        join=True,
    )
