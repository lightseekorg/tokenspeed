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

"""A rank's control side of the shared L2 pool against a real owner service
over real ZMQ. The pool behind the owner keeps the books of the scheduler's
``SharedL2Pool`` in Python, and the transfer records the copies it is asked
for instead of making them over RDMA."""

import copy
import json
import os
import sys
import threading
import time

import msgspec
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
    lookup_fields,
    prompt_keys,
    wait_until,
    write_fields,
)

from tokenspeed.runtime.cache.shared_l2 import client as client_module  # noqa: E402
from tokenspeed.runtime.cache.shared_l2.client import (  # noqa: E402
    SharedL2Client,
    SharedL2ClientConfig,
    SharedL2Provider,
)
from tokenspeed.runtime.cache.shared_l2.wire import (  # noqa: E402
    OwnerConnection,
    OwnerError,
    UnknownClientError,
)
from tokenspeed.runtime.engine.rust_host import HostSharedL2  # noqa: E402

PROMPT = [1, 2, 3, 4, 5]
BLOCKS = len(prompt_keys(PROMPT)[2])


def config_fields(address: str) -> dict:
    return {
        "owner": address,
        "device_names": [""],
        "request_timeout_ms": 1000,
        "connect_timeout_s": 5,
        "load_workers": 1,
        "write_workers": 1,
        "scheduler": {"max_writes": 2, "max_write_keys": 64},
    }


class RecordingTransfer:
    """Stands in for the RDMA copies: records each write it is asked for."""

    def __init__(self):
        self.writes = []
        # Called with every write before it is recorded; may raise.
        self.before_write = None
        self.closed = False

    def write(self, pool, placed):
        if self.before_write is not None:
            self.before_write(pool, placed)
        self.writes.append((pool, placed))

    def close(self):
        self.closed = True


class RecordingConnection(OwnerConnection):
    """The real connection, keeping the operation of every request it sent."""

    def __init__(self, address, *, timeout_ms):
        super().__init__(address, timeout_ms=timeout_ms)
        self.sent = []

    def request(self, op, **fields):
        self.sent.append(op)
        return super().request(op, **fields)


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
def join(serve):
    """Joins ranks to an owner's pool; closed before the owners."""
    connections, clients = [], []

    def start(owner: ServedOwner, *, connection=None, **settings) -> SharedL2Client:
        fields = {**config_fields(owner.address), **settings}
        config = SharedL2ClientConfig.from_json(json.dumps(fields))
        if connection is None:
            connection = OwnerConnection(
                owner.address, timeout_ms=config.request_timeout_ms
            )
        connections.append(connection)
        clients.append(
            SharedL2Client(
                config,
                geometry=GEOMETRY,
                deployment_id=deployment_id(),
                transfer=RecordingTransfer(),
                connection=connection,
            )
        )
        return clients[-1]

    yield start
    for client in clients:
        client.close()
    # A client that failed to start left its connection open.
    for connection in connections:
        connection.close()


def finished(client: SharedL2Client, count: int) -> list:
    got = []
    wait_until(lambda: got.extend(client.poll_writes()) or len(got) >= count)
    return got


def failure(client: SharedL2Client) -> BaseException:
    """What failed, once the client's polls raise."""
    raised = []

    def polled() -> bool:
        try:
            client.poll_writes()
        except RuntimeError as exc:
            raised.append(exc)
        return bool(raised)

    wait_until(polled)
    assert str(raised[0]) == "shared L2 operation failed"
    return raised[0].__cause__


def store(client: SharedL2Client, write_id: int, prompt) -> None:
    client.submit_write(write_id, *write_fields(prompt))
    assert finished(client, 1) == [write_id]


def look_up(client: SharedL2Client, lookup_id: int, prompt) -> tuple:
    (answer,) = client.lookup(
        [(lookup_id, f"request-{lookup_id}", *lookup_fields(prompt))]
    )
    return answer


def miss(lookup_id: int) -> tuple:
    return (lookup_id, 0, 0, 0, None, [])


def test_a_rank_joins_once_the_pool_is_ready(serve, join):
    owner = serve(joined_providers=0)
    late = threading.Timer(0.15, owner.add_provider, args=("node0", 8))
    late.start()

    rank = join(owner)

    late.join()
    assert rank.pool.providers == (
        SharedL2Provider(session="node0", base=1 << 40, lcm_blocks=8, start=0),
    )
    assert (rank.pool.lcm_bytes, rank.pool.num_lcm_blocks) == (LCM_BYTES, 8)
    assert owner.admin.request("stats")["clients"] == 1


def test_a_rank_does_not_start_without_a_pool_it_can_use(serve, join):
    with pytest.raises(TimeoutError, match="did not become ready"):
        join(serve(joined_providers=0), connect_timeout_s=0.2)

    owner = serve()
    join(owner)
    # The pool caches for another layout than this rank's.
    other = copy.deepcopy(GEOMETRY)
    other["prefix_granularity"] = 4
    config = SharedL2ClientConfig.from_json(json.dumps(config_fields(owner.address)))
    connection = OwnerConnection(owner.address, timeout_ms=1000)
    try:
        with pytest.raises(OwnerError, match="geometry differs"):
            SharedL2Client(
                config,
                geometry=other,
                deployment_id=deployment_id(),
                transfer=RecordingTransfer(),
                connection=connection,
            )
    finally:
        connection.close()


def test_one_request_looks_up_every_prompt_and_answers_them_in_order(serve, join):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    writer, reader = join(owner), join(owner, connection=connection)
    store(writer, 1, PROMPT)
    _, tail, keys = prompt_keys(PROMPT)

    hit, cold, same = reader.lookup(
        [
            (10, "hit", *lookup_fields(PROMPT + [6, 7])),
            (11, "cold", *lookup_fields([9, 9, 9, 9])),
            (12, "same", *lookup_fields(PROMPT)),
        ]
    )

    assert connection.sent.count("probe_many") == 1
    # In the shape the event loop takes: tuples for the tail and the keys, a
    # list of slots.
    lookup_id, ticket, num_tokens, kv_tokens, hit_tail, slots = hit
    assert (lookup_id, num_tokens, kv_tokens, hit_tail) == (10, 5, 5, tail)
    assert type(ticket) is int and ticket > 0
    assert isinstance(slots, list)
    assert [(key, bucket) for key, bucket, _ in slots] == [(key, 0) for key in keys]
    # A prompt the pool holds no prefix of is answered like the others: a miss.
    assert cold == miss(11)
    assert same[0] == 12 and same[1] not in (0, ticket)
    assert owner.stats()["pinned_blocks"] == 2 * BLOCKS
    # Without a lookup there is nothing to ask.
    assert reader.lookup([]) == []
    assert connection.sent.count("probe_many") == 1

    reader.release(ticket)
    reader.release(same[1])
    wait_until(lambda: owner.stats()["pinned_blocks"] == 0)
    assert reader.poll_writes() == []


def test_a_lookup_the_owner_does_not_answer_in_time_raises(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner, request_timeout_ms=100)

    owner.stall()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not answer 'probe_many'"):
        look_up(reader, 10, PROMPT)
    assert time.monotonic() - started >= 0.1
    owner.resume()


def test_a_lookup_the_owner_answers_with_an_error_raises(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner)
    owner.pool.failing.add("probe")

    with pytest.raises(OwnerError, match="probe failed"):
        look_up(reader, 10, PROMPT)


def test_a_lookup_by_a_rank_the_owner_does_not_know_raises(serve, join):
    owner = serve(client_timeout_s=0.2)
    rank = join(owner)
    # The rank has nothing to send, and its next ping is seconds away: the
    # owner drops it.
    wait_until(lambda: owner.pool.disconnected)

    with pytest.raises(UnknownClientError, match="has not joined the pool"):
        look_up(rank, 10, PROMPT)


# Hits that are not of the wire's shape, and the error each is refused with.
BAD_HITS = [
    (spoil, msgspec.ValidationError, "")
    for spoil in (
        # One element short.
        lambda hit: hit[:4],
        lambda hit: (hit[0], str(hit[1]), *hit[2:]),
        lambda hit: (True, *hit[1:]),
        lambda hit: (hit[0], hit[1], float(hit[2]), *hit[3:]),
        lambda hit: (*hit[:3], (1, b"tail"), hit[4]),
        # A key without its page offset, a page past int32, a negative group.
        lambda hit: (
            *hit[:4],
            [(key[:3], bucket, page) for key, bucket, page in hit[4]],
        ),
        lambda hit: (*hit[:4], [(key, bucket, 1 << 31) for key, bucket, _ in hit[4]]),
        lambda hit: (
            *hit[:4],
            [((key[0], -1, *key[2:]), bucket, page) for key, bucket, page in hit[4]],
        ),
        lambda hit: None,
    )
] + [
    (spoil, OwnerError, "which the pool does not have")
    for spoil in (
        # Pages the pool does not have: its 8 LCM blocks hold KV pages 1 to 16
        # and States 1 to 8.
        lambda hit: (*hit[:4], [(key, bucket, 0) for key, bucket, _ in hit[4]]),
        lambda hit: (*hit[:4], [(key, bucket, -page) for key, bucket, page in hit[4]]),
        lambda hit: (
            *hit[:4],
            [(key, bucket, 16 + (key[1] == 0)) for key, bucket, _ in hit[4]],
        ),
        lambda hit: (
            *hit[:4],
            [(key, bucket, 8 + (key[1] == 1)) for key, bucket, _ in hit[4]],
        ),
    )
]


@pytest.mark.parametrize("spoil, error, message", BAD_HITS)
def test_an_answer_with_a_malformed_hit_raises(serve, join, spoil, error, message):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner)
    probe = owner.pool.probe
    # The second of the two hits is spoiled.
    second = iter([False, True])
    owner.pool.probe = lambda *lookup: (
        spoil(probe(*lookup)) if next(second) else probe(*lookup)
    )

    with pytest.raises(error, match=message):
        reader.lookup(
            [
                (10, "hit", *lookup_fields(PROMPT)),
                (11, "spoiled", *lookup_fields(PROMPT)),
            ]
        )


@pytest.mark.parametrize(
    "spoil, message",
    [
        # No reply of the wire: no ``ok`` field, and no map.
        (lambda reply: {"hits": reply["hits"]}, "is not a reply"),
        (lambda reply: ["ok"], "is not a reply"),
        (lambda reply: {**reply, "hits": reply["hits"][:1]}, "1 hits for 2 lookups"),
    ],
)
def test_an_answer_that_is_not_one_hit_for_each_lookup_raises(
    serve, join, spoil, message
):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner)
    handle = owner.owner.handle
    owner.owner.handle = lambda request: (
        spoil(handle(request)) if request["op"] == "probe_many" else handle(request)
    )

    with pytest.raises(OwnerError, match=message):
        reader.lookup(
            [
                (10, "hit", *lookup_fields(PROMPT)),
                (11, "cold", *lookup_fields([9, 9])),
            ]
        )


def test_a_hit_may_name_the_last_page_of_each_group(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner)
    probe = owner.pool.probe

    def at_the_end(*lookup):
        # The pool's 8 LCM blocks end at KV page 16 and State 8.
        *hit, slots = probe(*lookup)
        return (*hit, [(key, bucket, 16 >> key[1]) for key, bucket, _ in slots])

    owner.pool.probe = at_the_end

    assert [page for _, _, page in look_up(reader, 10, PROMPT)[5]] == [16, 16, 8]


def test_a_write_copies_only_the_blocks_the_owner_placed(serve, join):
    owner = serve()
    first, second = join(owner), join(owner)
    keys, buckets, blocks, _ = write_fields(PROMPT)

    store(first, 1, PROMPT)
    ((pool, placed),) = first.transfer.writes
    assert pool is first.pool
    assert [(key, bucket, block) for key, bucket, _, block in placed] == list(
        zip(keys, buckets, blocks)
    )
    assert sorted(page for _, _, page, _ in placed) == sorted(
        owner.pool.cached.values()
    )

    # The pool holds every block of the same prompt: nothing to copy.
    store(second, 2, PROMPT)
    assert second.transfer.writes == []
    # A longer prompt shares its first pages: only the new blocks are copied.
    longer = PROMPT[:4] + [8, 9]
    store(second, 3, longer)
    ((_, placed),) = second.transfer.writes
    assert [key for key, _, _, _ in placed] == prompt_keys(longer)[2][2:]
    assert owner.stats()["checkpoints"] == 2


def test_a_key_the_owner_gives_no_page_is_skipped_and_the_write_completes(serve, join):
    # The pool has room for two of the prompt's three blocks.
    owner = serve(provider_blocks=2)
    rank = join(owner)
    keys, _, blocks, _ = write_fields(PROMPT)

    store(rank, 1, PROMPT)

    ((_, placed),) = rank.transfer.writes
    assert [(key, block) for key, _, _, block in placed] == list(zip(keys, blocks))[:2]
    assert all(page > 0 for _, _, page, _ in placed)
    assert owner.stats()["cached_blocks"] == 2
    # Nothing failed.
    assert rank.poll_writes() == []


def fail_reserve(owner, rank):
    owner.pool.failing.add("reserve")


def fail_transfer(owner, rank):
    def refuse(pool, placed):
        raise RuntimeError("the copy failed")

    rank.transfer.before_write = refuse


def fail_commit(owner, rank):
    owner.pool.failing.add("commit")


def drop_reservation(owner, rank):
    # The owner no longer holds the reservation when the copy is done.
    rank.transfer.before_write = lambda pool, placed: owner.pool.reserved.clear()


@pytest.mark.parametrize(
    "fail, message",
    [
        (fail_reserve, "reserve failed"),
        (fail_transfer, "the copy failed"),
        (fail_commit, "commit failed"),
        (drop_reservation, "no longer holds the reservation"),
    ],
)
def test_a_write_that_fails_at_any_step_fails_the_polls_from_then_on(
    serve, join, fail, message
):
    owner = serve()
    rank = join(owner)
    store(rank, 1, PROMPT)
    fail(owner, rank)

    rank.submit_write(5, *write_fields(PROMPT + [6]))

    assert message in str(failure(rank))
    # The write is not reported as finished, and the failure stays.
    with pytest.raises(RuntimeError, match="shared L2 operation failed"):
        rank.poll_writes()


def test_a_notice_the_owner_does_not_answer_fails_the_polls_and_is_sent_once(
    serve, join
):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=100)
    rank = join(owner, connection=connection, request_timeout_ms=100)
    store(rank, 1, PROMPT)
    ticket = look_up(rank, 10, PROMPT)[1]

    owner.stall()
    rank.release(ticket)

    assert isinstance(failure(rank), TimeoutError)
    # Well past the request timeout: the release was not sent again.
    time.sleep(0.3)
    assert connection.sent.count("release") == 1
    owner.resume()


def test_an_idle_rank_pings_the_owner_which_keeps_its_pins(serve, join, monkeypatch):
    monkeypatch.setattr(client_module, "_KEEPALIVE_S", 0.1)
    owner = serve(client_timeout_s=1.0)
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    rank = join(owner, connection=connection)
    store(rank, 1, PROMPT)
    look_up(rank, 10, PROMPT)

    # More than twice the owner's client timeout without a lookup, a write
    # or a notice.
    time.sleep(2.2)

    assert connection.sent.count("ping") >= 10
    assert owner.pool.disconnected == []
    assert owner.stats()["pinned_blocks"] == BLOCKS
    assert rank.poll_writes() == []


def test_closing_a_rank_leaves_what_a_thread_that_still_runs_uses(
    serve, join, monkeypatch
):
    monkeypatch.setattr(client_module, "_CLOSE_JOIN_S", 0.2)
    rank = join(serve())
    copying, resume = threading.Event(), threading.Event()
    rank.transfer.before_write = lambda pool, placed: (
        copying.set(),
        resume.wait(timeout=30),
    )
    rank.submit_write(5, *write_fields(PROMPT))
    assert copying.wait(timeout=5)

    # The copy outlasts the close: the memory it writes from stays
    # registered, and the socket its commit goes out on stays open.
    rank.close()
    assert not rank.transfer.closed
    resume.set()
    assert finished(rank, 1) == [5]


def test_closing_a_rank_stops_its_threads_and_closes_its_transfer(serve, join):
    rank = join(serve())
    rank.close()
    assert rank.transfer.closed
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("shared-l2-")
    ]


def test_the_rank_keeps_the_pool_page_behind_each_proxy_block(serve, join):
    rank = join(serve())
    rank.map_proxy_pages([(0, 11, 0, 3), (1, 12, 0, 9)])
    assert rank.proxy_page(0, 11) == (0, 3)
    assert rank.proxy_page(1, 12) == (0, 9)
    # A proxy block the scheduler uses again stands for its new page.
    rank.map_proxy_pages([(0, 11, 0, 4)])
    assert rank.proxy_page(0, 11) == (0, 4)
    with pytest.raises(KeyError):
        rank.proxy_page(0, 99)


def test_tokens_leave_the_rank_as_int32_bytes(serve, join):
    rank = join(serve())
    hashes, tokens, max_tokens = lookup_fields(PROMPT)
    with pytest.raises(TypeError, match="int32 bytes"):
        rank.lookup([(1, "request", hashes, PROMPT, max_tokens)])
    keys, buckets, blocks, _ = write_fields(PROMPT)
    with pytest.raises(TypeError, match="int32 bytes"):
        rank.submit_write(1, keys, buckets, blocks, [(PROMPT, hashes, None, 7)])


def test_the_host_adapter_carries_the_loops_calls_to_the_client(serve, join):
    owner = serve()
    rank = join(owner)
    pool = HostSharedL2(rank)
    keys = prompt_keys(PROMPT)[2]

    # A plan's write, and the poll that finds it finished.
    pool.submit([(31, *write_fields(PROMPT))], [], [])
    polled = []
    wait_until(lambda: polled.extend(pool.poll()) or polled)
    assert polled == [31] and pool.poll() == []
    # A round's lookups: one answer each, in order.
    hit, cold = pool.lookup(
        [(4, "r1", *lookup_fields(PROMPT)), (5, "r2", *lookup_fields([9, 9]))]
    )
    assert (hit[0], hit[2], cold) == (4, len(PROMPT), miss(5))
    ticket, page = hit[1], hit[5][0][2]
    # The pool page behind a proxy block the scheduler made for the hit.
    pool.map_pages([(0, 7, 0, page)])
    assert rank.proxy_page(0, 7) == (0, page)
    # A later plan releases the hit and touches two blocks.
    pool.submit([], [ticket], [keys[:2]])
    wait_until(lambda: owner.pool.touched == keys[:2])
    assert owner.stats()["pinned_blocks"] == 0

    # What the pool fails, the adapter raises.
    owner.pool.failing.update({"probe", "reserve"})
    with pytest.raises(OwnerError, match="probe failed"):
        pool.lookup([(6, "r3", *lookup_fields(PROMPT))])
    pool.submit([(32, *write_fields(PROMPT + [6]))], [], [])
    assert "reserve failed" in str(failure(rank))
    with pytest.raises(RuntimeError, match="shared L2 operation failed"):
        pool.poll()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda fields: fields.pop("request_timeout_ms"), "needs exactly"),
        (lambda fields: fields.update(namespace="model"), "needs exactly"),
        (
            lambda fields: fields["scheduler"].pop("max_writes"),
            "scheduler needs exactly",
        ),
        (
            lambda fields: fields["scheduler"].update(max_write_keys=0),
            "positive write limits",
        ),
        (lambda fields: fields.update(write_workers=0), "at least one"),
        (lambda fields: fields.update(request_timeout_ms=0), "positive timeouts"),
        (lambda fields: fields.update(connect_timeout_s=-1), "positive timeouts"),
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
        owner="tcp://127.0.0.1:1",
        device_names=("",),
        request_timeout_ms=1000,
        connect_timeout_s=5.0,
        load_workers=1,
        write_workers=1,
        max_writes=2,
        max_write_keys=64,
    )
