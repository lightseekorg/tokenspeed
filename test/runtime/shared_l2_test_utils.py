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

"""Shared pieces of the shared L2 service tests: a real owner served from a
thread over real ZMQ, and a pool that keeps the books of the scheduler's
``SharedL2Pool`` in Python, where every test can read them."""

from __future__ import annotations

import itertools
import threading
import time
from array import array

import zmq

from tokenspeed.runtime.cache.l3.identity import (
    L3_RUNTIME_COMPAT,
    storage_deployment_id,
)
from tokenspeed.runtime.cache.shared_l2.owner import SharedL2Owner
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection
from tokenspeed.runtime.utils.network import get_free_port

LCM_BYTES = 64
GEOMETRY = {
    "prefix_granularity": 2,
    "lcm_bytes": LCM_BYTES,
    "groups": [
        {
            "group_id": "kv",
            "attn_kind": "Full",
            "block_granularity": 2,
            "cache_blocks_per_lcm_block": 2,
            "host_packing": 2,
            "shard_count": 1,
            "sliding_window_tokens": None,
            "sink_tokens": 0,
            "replayable": False,
            "checkpoint_granularity": 1,
        },
        {
            "group_id": "state",
            "attn_kind": "State",
            "block_granularity": 0,
            "cache_blocks_per_lcm_block": 1,
            "host_packing": 1,
            "shard_count": 1,
            "sliding_window_tokens": None,
            "sink_tokens": 0,
            "replayable": False,
            "checkpoint_granularity": 1,
        },
    ],
}


def deployment_id(revision: str = "abc") -> str:
    """The deployment identity of a rank that loaded ``revision``."""
    return storage_deployment_id(
        "org/model",
        revision=revision,
        model_overrides={},
        cache_signature="layout",
        pipeline_rank=0,
        attn_tp_size=1,
        group_shard_counts=(1, 1),
        draft_model="",
        draft_revision="",
        cache_quantization="",
        runtime_compat=L3_RUNTIME_COMPAT,
        attention_backend="mha",
        draft_attention_backend="",
        skip_softmax_threshold=0.0,
        eagle3_layers_to_capture=[],
        kv_producer={"numerics": "resolved", "attn_kvp_size": 1},
    )


def token_bytes(prompt: list[int]) -> bytes:
    return array("i", prompt).tobytes()


def prompt_keys(prompt: list[int]) -> tuple[list[str], tuple | None, list[tuple]]:
    """A prompt's page hashes (two tokens a page), its tail, and the keys of
    its blocks: a KV block per page and the State block at its end."""
    hashes = [
        "-".join(str(token) for token in prompt[:end])
        for end in range(2, len(prompt) + 1, 2)
    ]
    tail = (1, "-".join(str(token) for token in prompt)) if len(prompt) % 2 else None
    end = hashes[-1] if tail is None else tail[1]
    return hashes, tail, [(0, 0, page, 0) for page in hashes] + [(0, 1, end, 0)]


def write_fields(prompt: list[int]) -> tuple[list, list, list, list]:
    """The keys, buckets, Device blocks and publications of the write that
    stores ``prompt`` with the checkpoint at its end."""
    hashes, tail, keys = prompt_keys(prompt)
    return (
        keys,
        [0] * len(keys),
        list(range(1, len(keys) + 1)),
        [(token_bytes(prompt), hashes, tail, 7)],
    )


def lookup_fields(prompt: list[int]) -> tuple[list[str], bytes, int]:
    return prompt_keys(prompt)[0], token_bytes(prompt), len(prompt)


def wait_until(predicate, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "the awaited state never came"
        time.sleep(0.002)


class FakePool:
    """The scheduler's ``SharedL2Pool``'s interface over plain dicts.

    A committed publication is a checkpoint: a probe whose tokens start with
    its tokens hits it and pins the blocks its write asked for.
    """

    def __init__(self, geometry, num_lcm_blocks):
        self.geometry = geometry
        self.num_lcm_blocks = num_lcm_blocks
        self._free = list(range(num_lcm_blocks, 0, -1))
        # (key, bucket) -> page.
        self.cached = {}
        # (tokens, tail, [(key, bucket)]).
        self.checkpoints = []
        # (client, ticket) -> slots.
        self.pins = {}
        # (client, ticket) -> ({(key, bucket): page}, every (key, bucket)).
        self.reserved = {}
        self._tickets = itertools.count(1)
        # The arguments of the last call of a method, as the owner passed them.
        self.received = {}
        # The methods that raise.
        self.failing = set()
        self.touched = []
        self.disconnected = []

    def _called(self, method, received=None):
        self.received[method] = received
        if method in self.failing:
            raise RuntimeError(f"{method} failed")

    def probe(self, client, prefix_hashes, tokens, max_tokens):
        self._called("probe", received=(prefix_hashes, tokens, max_tokens))
        hit = max(
            (found for found in self.checkpoints if tokens.startswith(found[0])),
            key=lambda found: len(found[0]),
            default=None,
        )
        if hit is None:
            return (0, 0, 0, None, [])
        hit_tokens, tail, blocks = hit
        ticket = next(self._tickets)
        slots = [(key, bucket, self.cached[key, bucket]) for key, bucket in blocks]
        self.pins[client, ticket] = slots
        return (ticket, len(hit_tokens) // 4, len(hit_tokens) // 4, tail, slots)

    def release(self, client, ticket):
        self._called("release")
        self.pins.pop((client, ticket), None)

    def reserve(self, client, keys, buckets):
        self._called("reserve", received=(keys, buckets))
        writing = {block for placed, _ in self.reserved.values() for block in placed}
        placed, pages = {}, []
        for block in zip(keys, buckets, strict=True):
            if block in self.cached or block in writing or block in placed:
                pages.append(-1)
            elif not self._free:
                pages.append(-2)
            else:
                placed[block] = self._free.pop()
                pages.append(placed[block])
        if not placed:
            return (0, pages)
        ticket = next(self._tickets)
        self.reserved[client, ticket] = (placed, list(zip(keys, buckets)))
        return (ticket, pages)

    def commit(self, client, ticket, publications):
        self._called("commit", received=publications)
        placed, blocks = self.reserved.pop((client, ticket), (None, None))
        if placed is None:
            return False
        self.cached.update(placed)
        for tokens, _, tail, _ in publications:
            self.checkpoints.append((tokens, tail, blocks))
        return True

    def touch(self, keys):
        self.touched.extend(keys)

    def disconnect(self, client):
        self.disconnected.append(client)
        for held in (self.pins, self.reserved):
            for owner, ticket in list(held):
                if owner == client:
                    del held[owner, ticket]

    def clear(self):
        if self.pins or self.reserved:
            return False
        self._free.extend(self.cached.values())
        self.cached.clear()
        self.checkpoints.clear()
        return True

    def stats(self):
        return {
            "total_lcm_blocks": self.num_lcm_blocks,
            "free_lcm_blocks": len(self._free),
            "cached_blocks": len(self.cached),
            "pinned_blocks": sum(len(slots) for slots in self.pins.values()),
            "reserved_blocks": sum(len(placed) for placed, _ in self.reserved.values()),
            "lookups": len(self.pins),
            "evicted_blocks": 0,
            "checkpoints": len(self.checkpoints),
        }


class ServedOwner:
    """A ``SharedL2Owner`` over ``FakePool``, serving from a thread."""

    def __init__(
        self,
        address: str | None = None,
        *,
        num_providers: int = 1,
        joined_providers: int = 1,
        provider_blocks: int = 8,
        client_timeout_s: float = 60.0,
    ):
        """
        Args:
            address: Where to bind; None picks a free local port.
            num_providers: Providers the owner waits for.
            joined_providers: Providers that join right away.
            provider_blocks: LCM blocks of each.
            client_timeout_s: The owner's client timeout.
        """
        self.address = address or f"tcp://127.0.0.1:{get_free_port()}"
        self.pools: list[FakePool] = []
        # The address may still belong to an owner that was just closed.
        deadline = time.monotonic() + 5
        while True:
            try:
                self.owner = SharedL2Owner(
                    address=self.address,
                    num_providers=num_providers,
                    client_timeout_s=client_timeout_s,
                    make_pool=self._make_pool,
                )
                break
            except zmq.ZMQError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
        self.admin = OwnerConnection(self.address, timeout_ms=5000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.resume()
        for index in range(joined_providers):
            self.add_provider(f"node{index}", provider_blocks)

    def _make_pool(self, geometry, num_lcm_blocks):
        self.pools.append(FakePool(geometry, num_lcm_blocks))
        return self.pools[-1]

    @property
    def pool(self) -> FakePool:
        (pool,) = self.pools
        return pool

    def add_provider(self, session: str, blocks: int) -> None:
        self.admin.request(
            "provider", session=session, base=1 << 40, bytes=blocks * LCM_BYTES
        )

    def stats(self) -> dict:
        return self.admin.request("stats")["stats"]

    def stall(self) -> None:
        """Stop answering. Requests wait in the socket for ``resume``."""
        self._stop.set()
        self._thread.join()
        self._thread = None

    def resume(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self.owner.serve, args=(self._stop,), daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self.stall()
        self.admin.close()
        self.owner.close()
