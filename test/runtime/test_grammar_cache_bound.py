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

"""The grammar backend keeps a bounded amount of compiled grammars.

A compiled grammar is kept only once a request is admitted with it, for the
most recently used schemas or tool sets within RETAINED_GRAMMAR_LIMIT_BYTES
(size estimate plus key size); the others are dropped, and so is a grammar
no admitted request used, once no queued request waits for it. Grammars are
kept and dropped where grammar admission is agreed, so every attention TP
rank keeps the same ones. xgrammar's own compiler cache is off.

Pure CPU; xgrammar runs on a small synthetic vocabulary.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.engine.generation_output_processor import (  # noqa: E402
    RequestState,
)
from tokenspeed.runtime.grammar import base_grammar_backend  # noqa: E402
from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    RETAINED_GRAMMAR_LIMIT_BYTES,
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from tokenspeed.runtime.grammar.grammar_manager import GrammarManager  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import (  # noqa: E402
    SamplingParams,
)

SCHEMAS = [
    json.dumps({"type": "object", "properties": {f"p{i}": {"type": "integer"}}})
    for i in range(24)
]
BAD = '{"type": "object", "bad": 0}'  # fails to compile on every rank
LOCAL_BAD = '{"type": "object", "bad": 1}'  # fails to compile on rank 1 only
SLOW = '{"type": "object", "slow": 1}'  # compiles too slowly on rank 1 only


def _key(schema: str) -> tuple[str, str]:
    return ("json", schema)


class _Grammar(BaseGrammarObject):
    def __init__(self, size: int = 0):
        self.size = size

    @property
    def memory_size_bytes(self) -> int:
        return self.size

    def copy(self):
        return _Grammar(self.size)


class _Backend(BaseGrammarBackend):
    """Compiles after ``delays[schema]`` seconds into a grammar of
    ``sizes[schema]`` bytes (``size`` by default); fails ``failing``."""

    def __init__(self, delays=None, sizes=None, size=0, failing=()):
        super().__init__()
        self.delays = delays or {}
        self.sizes = sizes if sizes is not None else {}
        self.size = size
        self.failing = set(failing)
        self.compiles = []

    def init_value_impl(self, key, require_reasoning: bool = False):
        time.sleep(self.delays.get(key[1], 0.0))
        self.compiles.append(key)
        if key[1] in self.failing:
            return InvalidGrammarObject("bad schema")
        return _Grammar(self.sizes.get(key[1], self.size))


class _Peer:
    """The other attention TP rank: its ready and failed index sets."""

    def __init__(self):
        self.ready: set[int] = set()
        self.failed: set[int] = set()

    def all_gather_object(self, gathered, obj, group=None):
        gathered[0] = obj
        gathered[1] = (set(self.ready), set(self.failed), {})


class _Group:
    """all_gather_object across rank threads, one barrier per call; objects
    are pickled as gloo does."""

    def __init__(self, size: int):
        self.slots = [None] * size
        self.barrier = threading.Barrier(size, timeout=20)
        self.local = threading.local()

    def all_gather_object(self, gathered, obj, group=None):
        self.slots[self.local.rank] = pickle.dumps(obj)
        self.barrier.wait()
        gathered[:] = [pickle.loads(slot) for slot in self.slots]
        self.barrier.wait()


def _manager(
    backend: BaseGrammarBackend, sync_size: int = 1, timeout: float = 30.0
) -> GrammarManager:
    args = SimpleNamespace(
        grammar_backend="none",
        grammar_compile_timeout_secs=timeout,
        grammar_compile_max_retries=1,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=[0])),
    )
    manager = GrammarManager(args, tokenizer=None, vocab_size=32)
    manager.grammar_backend = backend
    manager.grammar_sync_size = sync_size
    manager.grammar_sync_group = object() if sync_size > 1 else None
    return manager


def _state(schema: str) -> RequestState:
    return RequestState(
        prompt_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=8, json_schema=schema),
        stream=False,
        tokenizer=None,
        computes_prompt_logprobs=False,
    )


def _admit(manager: GrammarManager, schema: str) -> tuple[RequestState, bool]:
    """Admits one request on a single rank as the event loop does; returns
    its state and whether it was admitted at once (a cache hit)."""
    state = _state(schema)
    if manager.process_req_with_grammar(state):
        return state, True
    manager.add_to_queue(SimpleNamespace(request_id="r"), state, None)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if any(s is state for _, s, _ in manager.get_ready_grammar_requests()):
            break
        time.sleep(0.001)
    return state, False


def _key_bytes(schema: str) -> int:
    return len(schema) * (1 if schema.isascii() else 4)


def _retained_bytes(backend: BaseGrammarBackend) -> int:
    return sum(
        _key_bytes(key[1]) + g.memory_size_bytes
        for key, (g, _) in backend.retained.items()
    )


class TestGrammarCacheBound:
    def test_keeps_the_most_recently_admitted_grammars_within_the_bound(self):
        backend = _Backend(size=100 << 20)  # ten fit in 1 GiB
        manager = _manager(backend)
        for i in range(12):
            state, hit = _admit(manager, SCHEMAS[i])
            assert not hit
            assert not state.finished

        assert list(backend.retained) == [_key(SCHEMAS[i]) for i in range(2, 12)]
        assert backend.retained_bytes == _retained_bytes(backend)
        assert backend.retained_bytes <= RETAINED_GRAMMAR_LIMIT_BYTES
        assert backend.cache == {}  # no compiled grammar outside them

        assert _admit(manager, SCHEMAS[11])[1]
        assert not _admit(manager, SCHEMAS[0])[1]
        assert len(backend.compiles) == 13
        assert len(backend.retained) == 10

    def test_the_bound_counts_size_and_key(self):
        long_schema = json.dumps({"type": "object", "description": "x" * 4000})
        wide_schema = json.dumps(
            {"type": "object", "description": "\u00e9" * 1000}, ensure_ascii=False
        )
        sizes = {SCHEMAS[0]: 3000, SCHEMAS[1]: 3000, SCHEMAS[2]: 9000}
        backend = _Backend(sizes=sizes)
        manager = _manager(backend)
        with mock.patch.object(
            base_grammar_backend, "RETAINED_GRAMMAR_LIMIT_BYTES", 10_000
        ):
            _admit(manager, SCHEMAS[0])
            _admit(manager, SCHEMAS[1])
            assert len(backend.retained) == 2
            assert backend.retained_bytes == 6000 + 2 * len(SCHEMAS[0])

            # A grammar of size 0 still counts its key: the oldest goes.
            _admit(manager, long_schema)
            assert list(backend.retained) == [_key(SCHEMAS[1]), _key(long_schema)]

            # A non-ASCII key counts 4 bytes per character.
            _admit(manager, wide_schema)
            assert list(backend.retained) == [
                _key(long_schema),
                _key(wide_schema),
            ]
            assert backend.retained_bytes == (len(long_schema) + 4 * len(wide_schema))

            # A larger grammar drops every older one; one larger than the
            # bound by itself is not kept and leaves the others.
            _admit(manager, SCHEMAS[2])
            assert list(backend.retained) == [_key(SCHEMAS[2])]
            sizes[SCHEMAS[3]] = 20_000
            state, _ = _admit(manager, SCHEMAS[3])
            assert not state.finished
            assert list(backend.retained) == [_key(SCHEMAS[2])]
            assert backend.retained_bytes == _retained_bytes(backend)
            assert backend.cache == {}

    def test_a_flood_of_distinct_grammars_keeps_the_bound_and_the_hot_ones(self):
        """1,100 distinct schemas sized like tool-calling grammars (most under
        a MiB, one in ten 24 MiB), with three hot ones of 50 MiB requested in
        between: the retained grammars stay within the bound, no other
        compiled grammar is kept, and the hot ones stay cache hits."""
        flood = [json.dumps({"type": "object", "title": f"t{i}"}) for i in range(1100)]
        sizes = {
            s: (24 << 20) if i % 10 == 9 else (700 << 10) for i, s in enumerate(flood)
        }
        hot = SCHEMAS[:3]
        sizes.update({s: 50 << 20 for s in hot})
        backend = _Backend(sizes=sizes)
        manager = _manager(backend)
        for i, schema in enumerate(flood):
            assert not _admit(manager, schema)[1]
            if i % 10 == 0:
                state, hit = _admit(manager, hot[i // 10 % 3])
                assert not state.finished
                assert hit == (i >= 30)
            assert backend.retained_bytes <= RETAINED_GRAMMAR_LIMIT_BYTES
            assert backend.cache == {}

        assert backend.retained_bytes == _retained_bytes(backend)
        assert backend.retained_bytes > RETAINED_GRAMMAR_LIMIT_BYTES * 0.9
        assert set(map(_key, hot)) <= set(backend.retained)
        assert len(backend.compiles) == len(flood) + len(hot)

    def test_a_cache_hit_keeps_a_grammar_recent(self):
        backend = _Backend(size=100 << 20)
        manager = _manager(backend)
        for i in range(10):
            _admit(manager, SCHEMAS[i])
        assert _admit(manager, SCHEMAS[0])[1]

        _admit(manager, SCHEMAS[10])
        assert _key(SCHEMAS[0]) in backend.retained
        assert _key(SCHEMAS[1]) not in backend.retained

    def test_a_timeout_does_not_mark_a_retained_grammar(self):
        backend = _Backend()
        manager = _manager(backend)
        _admit(manager, SCHEMAS[0])
        backend.record_compile_timeout(
            _key(SCHEMAS[0]), "timed out", 30.0, max_retries=0
        )
        assert _key(SCHEMAS[0]) not in backend.cache
        assert _key(SCHEMAS[0]) not in backend.timeout_history
        assert _admit(manager, SCHEMAS[0])[1]

    def test_a_grammar_no_request_was_admitted_with_is_dropped(self):
        backend = _Backend()
        manager = _manager(backend)
        # A compile whose request was aborted before admission.
        backend.get_cached_or_future_value(_key(SCHEMAS[0]))[0].result(timeout=5)
        assert _key(SCHEMAS[0]) in backend.cache

        _admit(manager, SCHEMAS[1])
        assert list(backend.retained) == [_key(SCHEMAS[1])]
        assert _key(SCHEMAS[0]) not in backend.cache

    @pytest.mark.parametrize("other_waiting", [False, True])
    def test_a_compile_aborted_while_queued_is_dropped_once_it_finishes(
        self, other_waiting
    ):
        """The compiles finish after the pass that dropped their requests;
        the next pass drops them, whether the queue is then empty or holds
        a request still compiling."""
        gate, other_gate = threading.Event(), threading.Event()

        class _Gated(_Backend):
            def init_value_impl(self, key, require_reasoning: bool = False):
                other = key[1] == SCHEMAS[3]
                (other_gate if other else gate).wait(timeout=10)
                return super().init_value_impl(key, require_reasoning)

        backend = _Gated()
        manager = _manager(backend)
        states = [_state(SCHEMAS[i]) for i in range(3)]
        for i, state in enumerate(states):
            assert not manager.process_req_with_grammar(state)
            manager.add_to_queue(SimpleNamespace(request_id=str(i)), state, None)
            manager.mark_abort(str(i))
        assert len(manager.get_ready_grammar_requests()) == 3
        assert len(manager) == 0
        other = _state(SCHEMAS[3])
        if other_waiting:
            assert not manager.process_req_with_grammar(other)
            manager.add_to_queue(SimpleNamespace(request_id="o"), other, None)

        gate.set()
        for state in states:
            state.grammar.result(timeout=5)
        assert len(backend.compiled_keys) == 3
        assert manager.get_ready_grammar_requests() == []
        assert list(backend.cache) == ([_key(SCHEMAS[3])] if other_waiting else [])
        assert backend.compiled_keys == set()
        assert len(backend.retained) == 0
        other_gate.set()

    def test_a_compile_is_kept_while_its_request_waits(self):
        """The worker publishes a compiled grammar before its future resolves,
        and another request is admitted in that window. Dropping the entry
        then would leave a request that waited long in the executor queue
        measured from its queue time, so it would time out although its
        compile succeeded."""
        gate = threading.Event()

        class _SlowCopy(_Grammar):
            def copy(self):
                gate.wait(timeout=10)  # the worker's copy for the future
                return _Grammar()

        class _Gated(_Backend):
            def init_value_impl(self, key, require_reasoning: bool = False):
                if key[1] == SCHEMAS[0]:
                    return _SlowCopy()
                return super().init_value_impl(key, require_reasoning)

        backend = _Gated()
        manager = _manager(backend)
        state = _state(SCHEMAS[0])
        assert not manager.process_req_with_grammar(state)
        manager.add_to_queue(SimpleNamespace(request_id="a"), state, None)
        state.grammar_queued_ts -= 1000  # waited long for an executor worker
        deadline = time.monotonic() + 5
        while _key(SCHEMAS[0]) not in backend.compiled_keys:
            assert time.monotonic() < deadline
            time.sleep(0.001)

        _admit(manager, SCHEMAS[1])  # runs retain() while the copy blocks
        manager.get_ready_grammar_requests()
        assert backend.compile_started_at(_key(SCHEMAS[0])) is not None
        assert not state.finished
        assert len(manager) == 1

        gate.set()
        deadline = time.monotonic() + 5
        while len(manager) and time.monotonic() < deadline:
            manager.get_ready_grammar_requests()
            time.sleep(0.001)
        assert not state.finished
        assert isinstance(state.grammar, _Grammar)
        assert _key(SCHEMAS[0]) not in backend.cache
        assert _key(SCHEMAS[0]) in backend.retained


class TestGrammarCacheBoundPeer:
    """This rank with attention TP 2; the other rank is a stub."""

    @pytest.fixture(autouse=True)
    def _patch_peer(self, request):
        self.peer = _Peer()
        patcher = mock.patch(
            "torch.distributed.all_gather_object", self.peer.all_gather_object
        )
        patcher.start()
        request.addfinalizer(patcher.stop)

    def test_a_group_timeout_does_not_mark_a_compile_done_here(self):
        """This rank compiled the first request's grammar while the other
        rank still compiles it, and a second request is admitted in between.
        The compiled grammar stays while its request waits, and when the
        group times the first request out, this rank records no timeout for
        the key: its compile finished."""
        backend = _Backend()
        manager = _manager(backend, sync_size=2)
        states = []
        for i in range(2):
            state = _state(SCHEMAS[i])
            assert not manager.process_req_with_grammar(state)
            manager.add_to_queue(SimpleNamespace(request_id=str(i)), state, None)
            state.grammar.result(timeout=5)
            states.append(state)

        self.peer.ready = {1}  # the other rank still compiles the first
        promoted = manager.get_ready_grammar_requests()
        assert [entry[1] for entry in promoted] == [states[1]]
        assert list(backend.retained) == [_key(SCHEMAS[1])]
        entry = backend.cache[_key(SCHEMAS[0])]
        assert not entry.value.is_invalid

        self.peer.ready, self.peer.failed = set(), {0}  # its compile timed out
        promoted = manager.get_ready_grammar_requests()
        assert [entry[1] for entry in promoted] == [states[0]]
        assert "timed out" in states[0].finished_reason.to_json()["message"]
        assert _key(SCHEMAS[0]) not in backend.timeout_history
        # No request was admitted with it, so it is not kept either.
        assert _key(SCHEMAS[0]) not in backend.cache
        assert _key(SCHEMAS[0]) not in backend.retained

    def test_a_queued_hit_keeps_its_grammar_when_its_key_is_dropped(self):
        """A hit is queued as a resolved future with attention TP > 1. While
        the other rank holds it back, newer grammars push its key out of the
        retained ones; it is still admitted with its grammar, and its key is
        retained again."""
        backend = _Backend(size=100 << 20)  # ten fit in 1 GiB
        manager = _manager(backend, sync_size=2)

        def queue(schema):
            state = _state(schema)
            assert not manager.process_req_with_grammar(state)
            manager.add_to_queue(SimpleNamespace(request_id=schema), state, None)
            state.grammar.result(timeout=5)
            return state

        queue(SCHEMAS[0])
        self.peer.ready = {0}
        manager.get_ready_grammar_requests()
        held = queue(SCHEMAS[0])  # a hit: queued, ready here
        self.peer.ready = {1}  # the other rank holds it back
        for i in range(1, 11):
            state = queue(SCHEMAS[i])
            promoted = manager.get_ready_grammar_requests()
            assert [entry[1] for entry in promoted] == [state]
        assert _key(SCHEMAS[0]) not in backend.retained

        self.peer.ready = {0}
        promoted = manager.get_ready_grammar_requests()
        assert [entry[1] for entry in promoted] == [held]
        assert not held.finished
        assert isinstance(held.grammar, _Grammar)
        assert next(reversed(backend.retained)) == _key(SCHEMAS[0])
        assert len(backend.compiles) == 11

    def test_a_group_timeout_does_not_count_against_a_marker_here(self):
        """This rank's own timeout marker for a key is still live when the
        next request for it arrives, so this rank answers from the marker
        while the other rank compiles it again and times out. Only the ranks
        that were still compiling record that timeout: this rank's marker
        keeps its TTL and count, as when the other rank kept its grammar."""
        backend = _Backend()
        manager = _manager(backend, sync_size=2)
        key = _key(SCHEMAS[0])
        backend.record_compile_timeout(key, "timed out", ttl_secs=30.0, max_retries=1)

        state = _state(SCHEMAS[0])
        assert not manager.process_req_with_grammar(state)
        manager.add_to_queue(SimpleNamespace(request_id="0"), state, None)

        self.peer.ready, self.peer.failed = set(), {0}  # its compile timed out
        promoted = manager.get_ready_grammar_requests()
        assert [entry[1] for entry in promoted] == [state]
        assert "timed out" in state.finished_reason.to_json()["message"]
        assert backend.timeout_history[key].count == 1
        assert backend.cache[key].value.expires_at is not None


class TestGrammarCacheBoundTwoRanks:
    """Ranks 0 and 1 in threads, with a bound that keeps three grammars, run
    the event loop's admission step on a stream of requests: distinct schemas,
    repeated ones (retained, dropped, or still compiling on one rank), a
    schema that fails everywhere, one that fails on rank 1 only, and one
    whose compile times out on rank 1 only. Rank 1 also compiles some
    schemas more slowly. Both ranks must admit or abort every request in the
    same iteration with the same result, and keep the same grammars."""

    TIMEOUT = 0.5

    def test_ranks_admit_alike_under_eviction(self):
        size = 1000
        limit = 3 * size + 3 * max(map(len, SCHEMAS + [SLOW]))
        slow = {SCHEMAS[i]: 0.03 for i in range(0, 12, 3)}
        backends = [
            _Backend(size=size, failing=[BAD]),
            _Backend(
                delays={**slow, SLOW: 3 * self.TIMEOUT},
                size=size,
                failing=[BAD, LOCAL_BAD],
            ),
        ]
        managers = [_manager(b, sync_size=2, timeout=self.TIMEOUT) for b in backends]
        idle = [None] * 30
        # One request (or none) per iteration, in arrival order.
        script = (
            [SCHEMAS[i] for i in range(6)]
            + [SCHEMAS[3], SCHEMAS[3]]  # repeated while rank 1 still compiles
            + idle
            + [SCHEMAS[5]]  # retained on both: admitted on arrival
            + [BAD, SCHEMAS[0], LOCAL_BAD, SLOW, SCHEMAS[4]]
            + idle
            + [SCHEMAS[i] for i in range(6, 12)]
            + [SLOW, BAD, LOCAL_BAD, SCHEMAS[1], SCHEMAS[11]]
            + idle
            + [SCHEMAS[11], SCHEMAS[6]]
        )
        arrivals = [i for i, s in enumerate(script) if s is not None]
        on_arrival = script.index(SCHEMAS[5], len(idle))

        group = _Group(2)
        step = threading.Barrier(2, timeout=20)
        outcomes = [None, None]
        retained = [[], []]
        errors = []

        def rank(r):
            try:
                group.local.rank = r
                manager, backend = managers[r], backends[r]
                states, done_at = {}, {}
                for it in range(len(script) + 2000):
                    step.wait()  # the request broadcast
                    if it < len(script) and script[it] is not None:
                        state = _state(script[it])
                        states[it] = state
                        if manager.process_req_with_grammar(state):
                            done_at[id(state)] = it
                        else:
                            manager.add_to_queue(
                                SimpleNamespace(request_id=str(it)), state, None
                            )
                    for _, state, _ in manager.get_ready_grammar_requests():
                        done_at[id(state)] = it
                    retained[r].append(list(backend.retained))
                    assert backend.retained_bytes == _retained_bytes(backend)
                    assert backend.retained_bytes <= limit
                    if it >= len(script) and not len(manager):
                        break
                    time.sleep(0.005)
                outcomes[r] = [
                    (
                        done_at.get(id(s)),
                        s.finished,
                        s.finished_reason and s.finished_reason.to_json()["message"],
                        s.grammar is not None,
                    )
                    for _, s in sorted(states.items())
                ]
            except BaseException as exc:  # surface failures from the threads
                errors.append(exc)
                group.barrier.abort()
                step.abort()

        with (
            mock.patch("torch.distributed.all_gather_object", group.all_gather_object),
            mock.patch.object(
                base_grammar_backend, "RETAINED_GRAMMAR_LIMIT_BYTES", limit
            ),
        ):
            threads = [threading.Thread(target=rank, args=(r,)) for r in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)

        assert errors == []
        assert outcomes[0] is not None
        assert outcomes[0] == outcomes[1]
        assert retained[0] == retained[1]

        for it, (done, finished, message, has_grammar) in zip(arrivals, outcomes[0]):
            assert done is not None
            if script[it] in (BAD, LOCAL_BAD):
                assert finished
                assert "bad schema" in message
            elif script[it] == SLOW:
                assert finished
                assert "timed out" in message
            else:
                assert (finished, has_grammar) == (False, True)
        assert outcomes[0][arrivals.index(on_arrival)][0] == on_arrival
        # Grammars were dropped and compiled again.
        assert len(backends[0].compiles) > 15


class TestXGrammarCompilerCache:
    VOCAB = ["<s>", "</s>"] + [chr(c) for c in range(32, 127)]

    def _backend(self):
        import xgrammar

        from tokenspeed.runtime.grammar import xgrammar_backend

        info = xgrammar.TokenizerInfo(self.VOCAB, stop_token_ids=[1])
        with mock.patch.object(
            xgrammar_backend.TokenizerInfo, "from_huggingface", return_value=info
        ):
            return xgrammar_backend.XGrammarGrammarBackend(
                tokenizer=None, vocab_size=len(self.VOCAB)
            )

    def test_compiler_keeps_nothing_and_reset_clears_retained(self):
        backend = self._backend()
        compiler = backend.grammar_compiler

        manager = _manager(backend)
        state, _ = _admit(manager, SCHEMAS[0])
        assert not state.finished
        assert list(backend.retained) == [_key(SCHEMAS[0])]
        grammar = backend.retained[_key(SCHEMAS[0])][0]
        assert grammar.memory_size_bytes > 0
        assert backend.retained_bytes == len(SCHEMAS[0]) + grammar.memory_size_bytes
        # The only copy is the retained one.
        assert compiler.get_cache_size_bytes() == 0

        backend.reset()
        assert len(backend.retained) == 0
        assert backend.retained_bytes == 0

    def test_a_running_request_keeps_its_grammar_after_eviction(self):
        """Dropping a retained grammar drops the backend's reference only: a
        request admitted with it masks and accepts tokens to the end of its
        output."""
        backend = self._backend()
        manager = _manager(backend)
        running, _ = _admit(manager, SCHEMAS[0])
        limit = backend.retained_bytes * 3 // 2  # one of these grammars fits
        with mock.patch.object(
            base_grammar_backend, "RETAINED_GRAMMAR_LIMIT_BYTES", limit
        ):
            _admit(manager, SCHEMAS[1])
        assert list(backend.retained) == [_key(SCHEMAS[1])]
        assert backend.cache == {}

        grammar = running.grammar
        mask = grammar.allocate_vocab_mask(len(self.VOCAB), 1, "cpu")
        for ch in '{"p0": 7}':
            grammar.fill_vocab_mask(mask, 0)
            grammar.accept_token(self.VOCAB.index(ch))
        grammar.accept_token(1)  # </s>
        assert grammar.is_terminated()
