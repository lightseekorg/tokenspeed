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
over real ZMQ. The pool behind the owner keeps the books of
``tokenspeed_scheduler.SharedL2Pool`` in Python, and the transfer records the
copies it is asked for instead of making them over RDMA."""

import copy
import itertools
import json
import os
import sys
import threading
import time

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
)

PROMPT = [1, 2, 3, 4, 5]
BLOCKS = len(prompt_keys(PROMPT)[2])


def config_fields(address: str) -> dict:
    return {
        "owner": address,
        "device_names": [""],
        "request_timeout_ms": 1000,
        "connect_timeout_s": 5,
        "lookup_deadline_ms": 3000,
        "no_pool_after_s": 30,
        "lease_s": 0.3,
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
    """The real connection, keeping every request it sent and the thread
    that sent it: each thread has its own socket."""

    def __init__(self, address, *, timeout_ms):
        super().__init__(address, timeout_ms=timeout_ms)
        self.sent = []

    def request(self, op, **fields):
        self.sent.append((op, threading.current_thread().name))
        return super().request(op, **fields)

    def count(self, op):
        return sum(sent == op for sent, _ in self.sent)

    def first_sender(self, op):
        return next(thread for sent, thread in self.sent if sent == op)


class LateAnswers(RecordingConnection):
    """Hands an answer over ``delay_s`` after the owner gave it: the answers
    to ``op``, or to every request when ``op`` is None."""

    def __init__(self, address, *, timeout_ms, delay_s, op):
        super().__init__(address, timeout_ms=timeout_ms)
        self.delay_s = delay_s
        self.op = op

    def request(self, op, **fields):
        reply = super().request(op, **fields)
        if self.op in (None, op):
            time.sleep(self.delay_s)
        return reply


class BreaksOnce(RecordingConnection):
    """Fails the first request of each of ``ops`` as a broken socket does:
    with an error that is no timeout and no refusal."""

    def __init__(self, address, *, timeout_ms, ops):
        super().__init__(address, timeout_ms=timeout_ms)
        self.ops = set(ops)

    def request(self, op, **fields):
        if op in self.ops:
            self.ops.remove(op)
            self.sent.append((op, threading.current_thread().name))
            raise ConnectionError(f"{op} broke")
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


def answers(client: SharedL2Client, count: int) -> list:
    """The next ``count`` lookup answers, by lookup id."""
    got = []
    wait_until(lambda: got.extend(client.poll_lookups()) or len(got) >= count)
    assert len(got) == count
    return sorted(got)


def finished(client: SharedL2Client, count: int) -> list:
    got = []
    wait_until(lambda: got.extend(client.poll_writes()) or len(got) >= count)
    return got


def store(client: SharedL2Client, write_id: int, prompt) -> None:
    client.submit_write(write_id, *write_fields(prompt))
    assert finished(client, 1) == [(write_id, True)]


def look_up(client: SharedL2Client, lookup_id: int, prompt) -> tuple:
    client.submit_lookups([(lookup_id, f"request-{lookup_id}", *lookup_fields(prompt))])
    (answer,) = answers(client, 1)
    return answer


def miss(lookup_id: int) -> tuple:
    return (lookup_id, 0, 0, 0, None, [])


def test_a_rank_joins_once_the_pool_is_ready(serve, join):
    owner = serve(joined_providers=0)
    late = threading.Timer(0.15, owner.add_provider, args=("node0", 8))
    late.start()

    rank = join(owner)

    late.join()
    assert not rank.no_pool
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
    # A lease that outlasts the owner's client timeout protects nothing, and
    # the margin a planned load has to start counts toward it.
    with pytest.raises(ValueError, match="below the owner's client timeout"):
        join(owner, lease_s=60)
    with pytest.raises(ValueError, match="below the owner's client timeout"):
        join(owner, lease_s=58)
    assert not join(owner, lease_s=58, load_start_margin_s=1).no_pool


def test_one_request_looks_up_every_prompt_and_poll_returns_the_answers(serve, join):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    writer, reader = join(owner), join(owner, connection=connection)
    store(writer, 1, PROMPT)
    _, tail, keys = prompt_keys(PROMPT)

    reader.submit_lookups(
        [
            (10, "hit", *lookup_fields(PROMPT + [6, 7])),
            (11, "cold", *lookup_fields([9, 9, 9, 9])),
            (12, "same", *lookup_fields(PROMPT)),
        ]
    )
    hit, cold, same = answers(reader, 3)

    assert connection.count("probe_many") == 1
    lookup_id, ticket, num_tokens, kv_tokens, hit_tail, slots = hit
    assert (lookup_id, num_tokens, kv_tokens, hit_tail) == (10, 5, 5, tail)
    assert [(key, bucket) for key, bucket, _ in slots] == [(key, 0) for key in keys]
    assert cold == miss(11)
    assert same[1] not in (0, ticket)
    wait_until(lambda: reader.loads_allowed(ticket) and reader.loads_allowed(same[1]))
    assert reader.load_pool(ticket) is reader.pool
    assert owner.stats()["pinned_blocks"] == 2 * BLOCKS

    reader.release(ticket)
    # A hit the scheduler refused: its lookup is abandoned and its ticket
    # released.
    reader.abandon_lookup(12)
    reader.release(same[1])
    wait_until(
        lambda: [call[0] for call in owner.pool.calls].count("release") == 2
        and owner.stats()["pinned_blocks"] == 0
    )
    for released in (ticket, same[1]):
        assert not reader.loads_allowed(released)
        assert reader.load_pool(released) is None
    assert (reader.lookup_timeouts, reader.dropped_write_blocks) == (0, 0)


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
    assert first.dropped_write_blocks == second.dropped_write_blocks == 0


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
    assert finished(rank, 1) == [(5, True)]


def test_closing_a_rank_stops_its_threads_and_closes_its_transfer(serve, join):
    rank = join(serve())
    rank.close()
    assert rank.transfer.closed
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("shared-l2-")
    ]


def test_a_wait_for_lookups_ends_when_their_answers_are_there(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    # A wait can end for three reasons. Each part keeps the other two a
    # minute away, so no part depends on how fast this machine is.
    reader = join(owner, lookup_deadline_ms=60_000)
    brief = join(owner, lookup_deadline_ms=200)

    # Answered: the wait ends long before its limit, with every answer queued.
    reader.submit_lookups(
        [(10, "hit", *lookup_fields(PROMPT)), (11, "cold", *lookup_fields([9, 9]))]
    )
    started = time.monotonic()
    reader.wait_lookups([10, 11], 60.0)
    assert time.monotonic() - started < 30
    assert sorted(answer[0] for answer in reader.poll_lookups()) == [10, 11]
    # Nothing waits: lookups already answered, or never made.
    reader.wait_lookups([10, 11, 99], 60.0)
    assert time.monotonic() - started < 30

    # Unanswered: the wait ends at its own limit, with the lookup still open...
    owner.stall()
    reader.submit_lookups([(12, "late", *lookup_fields(PROMPT))])
    started = time.monotonic()
    reader.wait_lookups([12], 0.05)
    assert 0.05 <= time.monotonic() - started < 30
    assert reader.poll_lookups() == []
    # ... or at the lookup's deadline, when the next poll answers a miss.
    started = time.monotonic()
    brief.submit_lookups([(13, "late", *lookup_fields(PROMPT))])
    brief.wait_lookups([13], 60.0)
    assert 0.2 <= time.monotonic() - started < 30
    assert brief.poll_lookups() == [miss(13)]
    owner.resume()


def test_the_rank_keeps_the_pool_page_behind_each_proxy_block(serve, join):
    rank = join(serve())
    rank.map_proxy_pages(5, [(0, 11, 0, 3), (1, 12, 0, 9)])
    assert rank.proxy_page(0, 11) == (5, 0, 3)
    assert rank.proxy_page(1, 12) == (5, 0, 9)
    # A proxy block the scheduler uses again stands for its new page.
    rank.map_proxy_pages(6, [(0, 11, 0, 4)])
    assert rank.proxy_page(0, 11) == (6, 0, 4)
    with pytest.raises(KeyError):
        rank.proxy_page(0, 99)


def test_tokens_leave_the_rank_as_int32_bytes(serve, join):
    rank = join(serve())
    hashes, tokens, max_tokens = lookup_fields(PROMPT)
    with pytest.raises(TypeError, match="int32 bytes"):
        rank.submit_lookups([(1, "request", hashes, PROMPT, max_tokens)])
    keys, buckets, blocks, _ = write_fields(PROMPT)
    with pytest.raises(TypeError, match="int32 bytes"):
        rank.submit_write(1, keys, buckets, blocks, [(PROMPT, hashes, None, 7)])


def test_a_lookup_past_its_deadline_is_a_miss_the_owner_then_unpins(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    reader = join(owner, connection=connection, lookup_deadline_ms=100)
    client = reader.pool.client

    owner.stall()
    started = time.monotonic()
    assert look_up(reader, 10, PROMPT) == miss(10)
    assert time.monotonic() - started >= 0.1
    assert reader.lookup_timeouts == 1

    # The owner gets to the lookup late and pins its hit; the abandon that
    # follows it unpins.
    owner.resume()
    wait_until(lambda: ("abandon_lookup", client, 10) in owner.pool.calls)
    calls = owner.pool.calls
    assert calls.index(("probe", client, 10)) < calls.index(
        ("abandon_lookup", client, 10)
    )
    assert owner.pool.hits == 1 and owner.stats()["pinned_blocks"] == 0
    # The socket that carried the lookup carried the abandon, so the owner
    # could not read them the other way round.
    assert connection.first_sender("abandon_lookup") == "shared-l2-lookup"
    # The late answer is not the rank's any more.
    assert reader.poll_lookups() == []


def test_an_answer_that_comes_after_the_deadline_is_not_applied(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(
        owner,
        lookup_deadline_ms=100,
        connection=LateAnswers(
            owner.address, timeout_ms=1000, delay_s=0.3, op="probe_many"
        ),
    )
    client = reader.pool.client

    assert look_up(reader, 10, PROMPT) == miss(10)
    assert reader.lookup_timeouts == 1

    # The owner had pinned the hit; the rank gives it back unused.
    wait_until(lambda: ("abandon_lookup", client, 10) in owner.pool.calls)
    assert owner.pool.hits == 1
    assert owner.stats()["pinned_blocks"] == 0
    assert reader.poll_lookups() == []


def test_every_lookup_is_answered_once_and_only_a_used_hit_stays_pinned(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner, lookup_deadline_ms=20)
    # Every third lookup reaches the pool's answer after its deadline.
    delays = itertools.cycle([0, 0, 0.05])
    owner.pool.before = lambda method: method == "probe" and time.sleep(next(delays))

    lookup_ids = list(range(100, 140))
    answered = []
    for lookup_id in lookup_ids:
        reader.submit_lookups([(lookup_id, "request", *lookup_fields(PROMPT))])
        time.sleep(0.004)
        answered += reader.poll_lookups()
    wait_until(
        lambda: answered.extend(reader.poll_lookups())
        or len(answered) >= len(lookup_ids)
    )

    assert sorted(answer[0] for answer in answered) == lookup_ids
    tickets = [answer[1] for answer in answered if answer[1]]
    assert len(tickets) < len(lookup_ids)
    assert reader.lookup_timeouts == len(lookup_ids) - len(tickets)
    # Whatever the owner pinned for a lookup that missed here is unpinned.
    wait_until(lambda: owner.stats()["pinned_blocks"] == BLOCKS * len(tickets))
    for ticket in tickets:
        reader.release(ticket)
    wait_until(lambda: owner.stats()["pinned_blocks"] == 0)


def test_a_lookup_the_owner_fails_is_a_miss_and_unpins_its_request(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner)
    client = reader.pool.client
    owner.pool.failing.add(("probe", 11))

    # The owner pins the first lookup's hit, then fails the request.
    reader.submit_lookups(
        [
            (10, "hit", *lookup_fields(PROMPT)),
            (11, "fails", *lookup_fields(PROMPT)),
        ]
    )

    # Both miss at once, long before the deadline, and nothing stays pinned.
    assert answers(reader, 2) == [miss(10), miss(11)]
    assert reader.lookup_timeouts == 0
    wait_until(lambda: ("abandon_lookup", client, 11) in owner.pool.calls)
    assert owner.pool.hits == 1 and owner.stats()["pinned_blocks"] == 0
    # The next lookup is answered as usual.
    assert look_up(reader, 12, PROMPT)[2] == len(PROMPT)


def fail_reserve(owner, rank, write_id):
    owner.pool.failing.add("reserve")


def fail_transfer(owner, rank, write_id):
    def refuse(pool, placed):
        raise RuntimeError("the copy failed")

    rank.transfer.before_write = refuse


def fail_commit(owner, rank, write_id):
    owner.pool.failing.add("commit")


def drop_reservation(owner, rank, write_id):
    # The owner gives the reservation up while the copy runs.
    rank.transfer.before_write = lambda pool, placed: owner.admin.request(
        "abandon_write", client=pool.client, write_id=write_id
    )


@pytest.mark.parametrize(
    "fail", [fail_reserve, fail_transfer, fail_commit, drop_reservation]
)
def test_a_write_that_fails_at_any_step_is_abandoned_and_reported_failed(
    serve, join, fail, monkeypatch
):
    monkeypatch.setattr(client_module, "_WRITE_QUARANTINE_S", 0.3)
    owner = serve()
    rank = join(owner)
    client = rank.pool.client
    fail(owner, rank, 5)
    abandoned_at = []
    owner.pool.before = lambda method: method == "abandon_write" and (
        abandoned_at.append(time.monotonic())
    )

    submitted_at = time.monotonic()
    rank.submit_write(5, *write_fields(PROMPT))

    assert finished(rank, 1) == [(5, False)]
    assert rank.dropped_write_blocks == BLOCKS
    if fail is fail_transfer:
        # Slices of the failed copy may still land: the write is reported
        # with the owner still holding its reservation, and so its pages,
        # which the abandon frees only after the quarantine.
        assert owner.stats()["reserved_blocks"] == BLOCKS or abandoned_at
    wait_until(lambda: ("abandon_write", client, 5) in owner.pool.calls)
    if fail is fail_transfer:
        assert abandoned_at[0] - submitted_at >= 0.3
    stats = owner.stats()
    assert (stats["reserved_blocks"], stats["cached_blocks"]) == (0, 0)

    # Nothing is left behind: the same blocks can be written.
    owner.pool.failing.clear()
    rank.transfer.before_write = None
    store(rank, 6, PROMPT)
    assert owner.stats()["cached_blocks"] == BLOCKS
    assert rank.dropped_write_blocks == BLOCKS


def test_a_write_the_owner_does_not_answer_fails_and_frees_its_pages_later(serve, join):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=150)
    rank = join(owner, connection=connection, request_timeout_ms=150)
    client = rank.pool.client

    owner.stall()
    rank.submit_write(5, *write_fields(PROMPT))
    assert finished(rank, 1) == [(5, False)]
    assert rank.transfer.writes == [] and rank.dropped_write_blocks == BLOCKS

    # The owner reserves the pages late; the abandon behind it frees them.
    owner.resume()
    wait_until(lambda: ("abandon_write", client, 5) in owner.pool.calls)
    calls = owner.pool.calls
    assert calls.index(("reserve", client, 5)) < calls.index(
        ("abandon_write", client, 5)
    )
    wait_until(lambda: owner.stats()["reserved_blocks"] == 0)
    assert owner.stats()["cached_blocks"] == 0
    assert connection.first_sender("abandon_write") == "shared-l2-write-0"


def test_a_notice_is_sent_again_until_the_owner_answers(serve, join):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    rank = join(owner, connection=connection)
    store(rank, 1, PROMPT)
    ticket = look_up(rank, 10, PROMPT)[1]
    keys = prompt_keys(PROMPT)[2]

    owner.stall()
    rank.release(ticket)
    rank.touch(keys[:1], [0])
    wait_until(lambda: connection.count("release") >= 3)
    assert owner.pool.pins and not owner.pool.touched

    # The owner reads every copy of the release; the repeats change nothing.
    owner.resume()
    wait_until(lambda: owner.pool.touched)
    assert owner.stats()["pinned_blocks"] == 0
    assert set(owner.pool.touched) == set(keys[:1])
    assert not rank.no_pool


def test_a_silent_owner_ends_the_lease_and_its_answer_renews_it(serve, join):
    owner = serve()
    rank = join(owner, lease_s=0.2, load_start_margin_s=1.0)
    store(rank, 1, PROMPT)
    ticket = look_up(rank, 10, PROMPT)[1]
    wait_until(lambda: rank.loads_allowed(ticket))

    owner.stall()
    wait_until(lambda: not rank.loads_allowed(ticket))
    # Only new plans stop: a load planned before may still start, for the
    # margin, and the rank is still the pool's client.
    assert rank.load_pool(ticket) is rank.pool
    assert not rank.no_pool
    wait_until(lambda: rank.load_pool(ticket) is None)
    assert not rank.no_pool

    owner.resume()
    wait_until(lambda: rank.loads_allowed(ticket))
    assert rank.load_pool(ticket) is rank.pool
    assert owner.stats()["pinned_blocks"] == BLOCKS


def test_an_owner_that_does_not_know_the_rank_renews_no_lease(serve, join):
    owner = serve()
    rank = join(owner, lease_s=0.2, load_start_margin_s=0.2)
    store(rank, 1, PROMPT)
    old_pool, ticket = rank.pool, look_up(rank, 10, PROMPT)[1]

    # The owner goes, and the ticket's start bound passes.
    owner.close()
    wait_until(lambda: rank.load_pool(ticket) is None)
    assert rank.pool is old_pool

    # A new owner's answer, that it does not know the rank, ends the
    # membership. It is no contact with the pool that granted the ticket.
    serve(address=owner.address)
    wait_until(lambda: rank.pool is not None and rank.pool is not old_pool)
    assert rank.load_pool(ticket) is None
    assert not rank.loads_allowed(ticket)


def test_an_answer_slower_than_the_lease_does_not_renew_it(serve, join):
    owner = serve()
    store(join(owner), 1, PROMPT)
    # Every answer reaches the rank 0.3 s after its request left, which is
    # when the owner last heard the rank: longer ago than the lease.
    rank = join(
        owner,
        lease_s=0.2,
        connection=LateAnswers(owner.address, timeout_ms=1000, delay_s=0.3, op=None),
    )

    ticket = look_up(rank, 10, PROMPT)[1]

    assert ticket != 0 and not rank.loads_allowed(ticket)


def test_heartbeats_keep_an_idle_rank_and_its_pins_at_the_owner(serve, join):
    owner = serve(client_timeout_s=1.0)
    rank = join(owner, lease_s=0.15, load_start_margin_s=0.15)
    store(rank, 1, PROMPT)
    ticket = look_up(rank, 10, PROMPT)[1]

    # More than twice the owner's client timeout without a request.
    time.sleep(2.2)

    assert owner.pool.disconnected == []
    assert owner.stats()["pinned_blocks"] == BLOCKS
    assert not rank.no_pool
    wait_until(lambda: rank.loads_allowed(ticket))


def test_a_rank_serves_without_the_pool_while_the_owner_is_gone(serve, join):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=100)
    rank = join(
        owner,
        connection=connection,
        request_timeout_ms=100,
        no_pool_after_s=0.5,
        lease_s=0.2,
    )
    store(rank, 1, PROMPT)
    old_pool, old_ticket = rank.pool, look_up(rank, 10, PROMPT)[1]
    old_client = old_pool.client

    owner.close()
    wait_until(lambda: rank.no_pool)

    # A lookup misses at once, without waiting for its deadline.
    rank.submit_lookups([(11, "request", *lookup_fields(PROMPT))])
    assert rank.poll_lookups() == [miss(11)]
    assert rank.lookup_timeouts == 0
    # A write is dropped, and reported so its Device blocks are let go.
    copies = len(rank.transfer.writes)
    rank.submit_write(2, *write_fields(PROMPT + [6]))
    assert finished(rank, 1) == [(2, False)]
    assert len(rank.transfer.writes) == copies
    assert rank.dropped_write_blocks == len(prompt_keys(PROMPT + [6])[2])
    # No load is planned from the earlier hit; one planned before the rank
    # left may still start, from the pool the hit is of.
    assert not rank.loads_allowed(old_ticket)
    assert rank.load_pool(old_ticket) is old_pool

    # A new owner on the address: the rank joins as a new client.
    restarted = serve(address=owner.address)
    wait_until(lambda: not rank.no_pool)
    assert rank.pool.client != old_client
    store(rank, 3, PROMPT)
    new_ticket = look_up(rank, 12, PROMPT)[1]
    wait_until(lambda: rank.loads_allowed(new_ticket))
    assert rank.load_pool(new_ticket) is rank.pool
    assert not rank.loads_allowed(old_ticket)
    assert rank.load_pool(old_ticket) is old_pool
    # The earlier client's ticket is of no pool the rank is in: its release
    # goes nowhere, though the new owner has a ticket of that number.
    releases = connection.count("release")
    rank.release(old_ticket)
    assert rank.load_pool(old_ticket) is None
    rank.touch(prompt_keys(PROMPT)[2][:1], [0])
    wait_until(lambda: restarted.pool.touched)
    assert connection.count("release") == releases
    assert restarted.stats()["pinned_blocks"] == BLOCKS


def test_a_restarted_owner_does_not_know_the_rank_which_joins_again(serve, join):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    rank = join(owner, connection=connection, lookup_deadline_ms=1000)
    store(rank, 1, PROMPT)
    old_pool, old_ticket = rank.pool, look_up(rank, 10, PROMPT)[1]
    old_client, held_ticket = old_pool.client, look_up(rank, 9, PROMPT)[1]

    # A release and a lookup are still unanswered when the owner goes.
    owner.stall()
    rank.release(old_ticket)
    rank.submit_lookups([(11, "request", *lookup_fields(PROMPT))])
    wait_until(lambda: connection.count("release") >= 1)
    owner.close()
    restarted = serve(address=owner.address)

    # The new owner's first answer ends the old membership, long before the
    # no-pool threshold.
    wait_until(lambda: rank.pool is not None and rank.pool.client != old_client)
    assert not rank.loads_allowed(old_ticket)
    # A load planned from a hit the rank still held may start: the new owner
    # has none of the old pool's providers, so those pages are as they were.
    assert not rank.loads_allowed(held_ticket)
    assert rank.load_pool(held_ticket) is old_pool
    assert rank.load_pool(old_ticket) is None
    # The lookup misses with the membership, without waiting for its deadline.
    assert rank.poll_lookups() == [miss(11)]
    assert restarted.admin.request("stats")["clients"] == 1
    assert restarted.stats()["pinned_blocks"] == 0
    store(rank, 2, PROMPT)
    assert restarted.stats()["cached_blocks"] == BLOCKS


def test_lookups_queued_when_the_rank_leaves_the_pool_are_not_sent(serve, join):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=100)
    rank = join(
        owner,
        connection=connection,
        request_timeout_ms=100,
        lookup_deadline_ms=1000,
        no_pool_after_s=0.6,
    )

    # The owner hears nothing. The first lookup holds the lookup thread until
    # its deadline; the second waits behind it, with a later deadline.
    owner.stall()
    started = time.monotonic()
    rank.submit_lookups([(10, "first", *lookup_fields(PROMPT))])
    time.sleep(0.15)
    rank.submit_lookups([(11, "queued", *lookup_fields(PROMPT))])
    wait_until(lambda: rank.no_pool)
    assert answers(rank, 2) == [miss(10), miss(11)]

    # The second was answered when the rank left: once the thread is free,
    # it is not sent to the owner as a client the rank no longer is.
    time.sleep(max(0.0, started + 1.4 - time.monotonic()))
    assert connection.count("probe_many") == 1


BAD_HITS = [
    # One element short.
    lambda hit: hit[:4],
    lambda hit: (hit[0], str(hit[1]), *hit[2:]),
    lambda hit: (True, *hit[1:]),
    lambda hit: (hit[0], hit[1], float(hit[2]), *hit[3:]),
    lambda hit: (*hit[:3], (1, b"tail"), hit[4]),
    # A key without its page offset, a page past int32, a negative group.
    lambda hit: (*hit[:4], [(key[:3], bucket, page) for key, bucket, page in hit[4]]),
    lambda hit: (*hit[:4], [(key, bucket, 1 << 31) for key, bucket, _ in hit[4]]),
    lambda hit: (
        *hit[:4],
        [((key[0], -1, *key[2:]), bucket, page) for key, bucket, page in hit[4]],
    ),
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
    lambda hit: None,
]


@pytest.mark.parametrize("spoil", BAD_HITS)
def test_an_answer_of_the_wrong_shape_is_a_miss_the_owner_then_unpins(
    serve, join, spoil
):
    owner = serve()
    store(join(owner), 1, PROMPT)
    reader = join(owner)
    client = reader.pool.client
    probe = owner.pool.probe
    owner.pool.probe = lambda client, lookup_id, *lookup: (
        spoil(probe(client, lookup_id, *lookup))
        if lookup_id == 11
        else probe(client, lookup_id, *lookup)
    )

    reader.submit_lookups(
        [
            (10, "hit", *lookup_fields(PROMPT)),
            (11, "spoiled", *lookup_fields(PROMPT)),
        ]
    )

    # The whole answer is refused, at once: both miss, and what the owner
    # pinned for either is unpinned.
    assert answers(reader, 2) == [miss(10), miss(11)]
    assert reader.lookup_timeouts == 0
    wait_until(lambda: ("abandon_lookup", client, 11) in owner.pool.calls)
    assert owner.pool.hits == 2 and owner.stats()["pinned_blocks"] == 0
    # The next lookup is answered as usual, in the shape the event loop
    # takes: tuples for the tail and the keys, a list of slots.
    owner.pool.probe = probe
    _, tail, keys = prompt_keys(PROMPT)
    lookup_id, ticket, num_tokens, kv_tokens, hit_tail, slots = look_up(
        reader, 12, PROMPT
    )
    assert (lookup_id, num_tokens, kv_tokens, hit_tail) == (12, 5, 5, tail)
    assert type(ticket) is int and ticket > 0
    assert isinstance(slots, list)
    assert [key for key, _, _ in slots] == keys


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


def test_a_notice_that_breaks_is_sent_again_unless_it_cannot_be_sent(serve, join):
    owner = serve()
    connection = BreaksOnce(
        owner.address, timeout_ms=1000, ops={"release", "abandon_lookup"}
    )
    rank = join(owner, connection=connection)
    client = rank.pool.client
    store(rank, 1, PROMPT)
    ticket = look_up(rank, 10, PROMPT)[1]
    owner.pool.failing.add(("probe", 12))

    # A notice the wire cannot carry is dropped; the release behind it
    # breaks on its first send and is sent again.
    rank.touch([object()], [0])
    rank.release(ticket)
    wait_until(lambda: owner.stats()["pinned_blocks"] == 0)
    assert connection.count("release") == 2

    # So is the abandon of a lookup the owner failed after it pinned the
    # first hit of the request.
    rank.submit_lookups(
        [
            (11, "hit", *lookup_fields(PROMPT)),
            (12, "fails", *lookup_fields(PROMPT)),
        ]
    )
    assert answers(rank, 2) == [miss(11), miss(12)]
    wait_until(lambda: ("abandon_lookup", client, 12) in owner.pool.calls)
    assert owner.pool.hits == 2 and owner.stats()["pinned_blocks"] == 0
    assert connection.count("abandon_lookup") == 2
    assert not rank.no_pool


# Answers that are not a reply of the wire: no ``ok`` field, and no map.
WRONG_FORMS = [{"hits": []}, ["ok"]]


@pytest.mark.parametrize("reply", WRONG_FORMS)
def test_a_notice_answered_in_the_wrong_form_is_dropped(serve, join, reply):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    rank = join(owner, connection=connection)
    store(rank, 1, PROMPT)
    ticket = look_up(rank, 10, PROMPT)[1]
    handle = owner.owner.handle
    owner.owner.handle = lambda request: (
        reply if request["op"] == "release" else handle(request)
    )

    # The release is not sent again: the touch behind it reaches the owner.
    rank.release(ticket)
    rank.touch(prompt_keys(PROMPT)[2][:1], [0])
    wait_until(lambda: owner.pool.touched)
    assert connection.count("release") == 1
    assert not rank.no_pool


@pytest.mark.parametrize("reply", WRONG_FORMS)
def test_an_owner_answering_in_the_wrong_form_is_an_owner_not_heard(serve, join, reply):
    owner = serve()
    connection = RecordingConnection(owner.address, timeout_ms=1000)
    rank = join(
        owner,
        connection=connection,
        no_pool_after_s=1.5,
        lease_s=0.2,
        load_start_margin_s=0.2,
    )
    store(rank, 1, PROMPT)
    old_client, ticket = rank.pool.client, look_up(rank, 10, PROMPT)[1]
    wait_until(lambda: rank.loads_allowed(ticket))
    handle = owner.owner.handle
    owner.owner.handle = lambda request: reply

    # Such an answer renews no lease.
    wait_until(lambda: not rank.loads_allowed(ticket))
    # A lookup misses and a write fails at once. The owner refused their
    # abandons too, which are therefore not kept to be sent again.
    assert look_up(rank, 11, PROMPT) == miss(11)
    rank.submit_write(2, *write_fields(PROMPT + [6]))
    assert finished(rank, 1) == [(2, False)]
    wait_until(lambda: connection.count("abandon_write") == 1)
    # After no_pool_after_s of it the rank serves without the pool ...
    wait_until(lambda: rank.no_pool)
    assert connection.count("abandon_lookup") == 1
    assert connection.count("abandon_write") == 1
    # ... having pinged at the pace of its retries, however fast the owner
    # answered.
    assert connection.count("ping") < 100

    # ... and joins again when the owner answers as the wire has it.
    owner.owner.handle = handle
    wait_until(lambda: not rank.no_pool)
    assert rank.pool.client != old_client


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda fields: fields.pop("lookup_deadline_ms"), "needs exactly"),
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
        (lambda fields: fields.update(lease_s=0), "positive timeouts"),
        (lambda fields: fields.update(no_pool_after_s=-1), "positive timeouts"),
        (lambda fields: fields.update(load_start_margin_s=0), "positive timeouts"),
        (lambda fields: fields.update(device_names="mlx5_0"), "list of strings"),
        (lambda fields: fields.update(device_names=[0]), "list of strings"),
        (
            lambda fields: fields.update(request_timeout_ms=30_001),
            "must not exceed no_pool_after_s",
        ),
    ],
)
def test_a_client_config_names_every_field_and_usable_values(change, message):
    fields = config_fields("tcp://127.0.0.1:1")
    change(fields)
    with pytest.raises(ValueError, match=message):
        SharedL2ClientConfig.from_json(json.dumps(fields))


def test_a_client_config_keeps_every_field():
    fields = config_fields("tcp://127.0.0.1:1")
    config = SharedL2ClientConfig.from_json(json.dumps(fields))
    named = SharedL2ClientConfig.from_json(
        json.dumps({**fields, "load_start_margin_s": 2})
    )
    assert named.load_start_margin_s == 2.0
    assert config == SharedL2ClientConfig(
        owner="tcp://127.0.0.1:1",
        device_names=("",),
        request_timeout_ms=1000,
        connect_timeout_s=5.0,
        lookup_deadline_ms=3000,
        no_pool_after_s=30.0,
        lease_s=0.3,
        # Not named: the default.
        load_start_margin_s=5.0,
        load_workers=1,
        write_workers=1,
        max_writes=2,
        max_write_keys=64,
    )
