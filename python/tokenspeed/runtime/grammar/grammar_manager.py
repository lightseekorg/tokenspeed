# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.
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

from __future__ import annotations

import time
from concurrent import futures
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.grammar.base_grammar_backend import (
    InvalidGrammarObject,
    create_grammar_backend,
)
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.generation_output_processor import RequestState
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


# A pending grammar request: the spec must be submitted to the C++ scheduler and
# the state registered with the executor *only* after its grammar future resolves.
QueueEntry = tuple["object", "RequestState", "object"]  # (spec, state, bootstrap_info)


class GrammarManager:
    def __init__(
        self,
        server_args: ServerArgs,
        tokenizer,
        vocab_size: int,
    ) -> None:

        self.server_args = server_args
        self.grammar_queue: list[QueueEntry] = []

        self.compile_timeout_secs = float(server_args.grammar_compile_timeout_secs)
        self.compile_max_retries = int(server_args.grammar_compile_max_retries)

        # Backend is None when (a) the user disabled grammar via
        # --grammar-backend none or (b) tokenizer init is skipped (no tokenizer
        # → can't build a backend). Either way, requests with grammar fields
        # get rejected at admission time in process_req_with_grammar.
        if server_args.skip_tokenizer_init:
            self.grammar_backend = None

        else:
            self.grammar_backend = create_grammar_backend(
                server_args, tokenizer, vocab_size
            )

        # Grammar admission must be coherent across attention TP ranks: every rank
        # in the group sees the same recv_reqs (broadcast in RequestHandler.recv_reqs),
        # so they must agree on which requests have a ready grammar before admitting
        # any of them. Different DP ranks see different requests and don't sync.
        attn_group = server_args.mapping.attn.tp_group

        if len(attn_group) > 1 and pg_manager.has_process_group("gloo", attn_group):
            self.grammar_sync_group = pg_manager.get_process_group("gloo", attn_group)
            self.grammar_sync_size = len(attn_group)

        else:
            self.grammar_sync_group = None
            self.grammar_sync_size = 1

    def __len__(self):

        return len(self.grammar_queue)

    def clear(self):

        if self.grammar_backend:
            self.grammar_backend.reset()

    def mark_abort(self, rid: str) -> None:
        """Cancel a queued request whose grammar is still compiling.

        Paired with ``OutputProcesser.mark_abort`` in the event loop:
        together they cover both already-registered requests and those
        still blocked on a grammar future. Without this, an aborted
        request would finish compiling, get admitted, and only then be
        noticed as aborted — burning capacity in the meantime.
        """
        for spec, state, _ in self.grammar_queue:
            if spec.request_id == rid:
                logger.debug("Abort grammar queue request. rid=%s", rid)

                # Don't cancel the compile future: it's shared across
                # every concurrent request for the same grammar key
                # (see ``get_cached_or_future_value``), so cancelling
                # would raise CancelledError on still-valid waiters.
                # The compile runs to completion in the background and
                # its result lands in the cache for future reuse.
                state.set_finish_with_abort("Aborted by AbortReq.")

                # The queue entry is cleaned up by the next
                # get_ready_grammar_requests pass (which treats
                # state.finished as "ready, but don't admit").
                return

    def process_req_with_grammar(self, state: RequestState) -> bool:
        """Attach grammar (or future) to ``state``.

        Returns True if the request is admittable now (no grammar, cache hit, or
        already aborted), False if it must be queued until its future resolves.
        """
        sp = state.sampling_params

        if (
            sp.json_schema is None
            and sp.regex is None
            and sp.ebnf is None
            and sp.structural_tag is None
        ):
            return True

        if self.grammar_backend is None:
            state.set_finish_with_abort(
                "Grammar-based generation (json_schema, regex, ebnf, structural_tag) "
                "is not supported when the server is launched with --grammar-backend none"
            )

            return True

        if sp.json_schema is not None:
            key = ("json", sp.json_schema)

        elif sp.regex is not None:
            key = ("regex", sp.regex)

        elif sp.ebnf is not None:
            key = ("ebnf", sp.ebnf)

        else:
            key = ("structural_tag", sp.structural_tag)

        value, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
        state.grammar_key = key

        if cache_hit:
            if value.is_invalid:
                state.set_finish_with_abort(
                    f"Failed to compile {key[0]} grammar: {value.error_message}"
                )

                state.grammar = None

            else:
                state.grammar = value

            return True

        # Compile is in flight; caller should add to queue via add_to_queue.
        state.grammar = value  # Future
        return False

    def add_to_queue(self, spec, state: RequestState, bootstrap_info) -> None:
        # Per-state queue timestamp bounds the time a request can spend in
        # the executor's pending queue before it's picked up by a worker.
        # The compile-only budget (after the worker starts) is measured
        # against backend.compile_started_at(key) instead.
        state.grammar_queued_ts = time.monotonic()
        self.grammar_queue.append((spec, state, bootstrap_info))

    def get_ready_grammar_requests(self) -> list[QueueEntry]:
        """Promote queued requests whose grammar is ready (or has timed out).

        Per-rank: scan futures, mark ready/failed locally.
        Cross-rank (attn TP > 1): all_gather indices and admit only the
        intersection of ready sets and the union of failed sets, so every rank
        admits the same requests in the same iteration.

        Caller is responsible for invoking this every loop iteration when
        ``grammar_sync_size > 1`` so the collective stays in sync; with size 1
        it's a no-op when the queue is empty.
        """
        # Queue length is identical on every attn-TP rank: recv_reqs
        # broadcasts the new/abort sets, so add_to_queue and mark_abort
        # run in lockstep, and pops below only happen after the cross-
        # rank consensus. If our queue is empty, every rank's is — no
        # indices to negotiate, so we can safely skip the collective.
        if not self.grammar_queue:
            return []

        now = time.monotonic()
        ready_idxs: set[int] = set()
        failed_idxs: set[int] = set()

        for i, (_, state, _) in enumerate(self.grammar_queue):
            if state.finished or state.grammar is None:
                # Aborted while queued.
                ready_idxs.add(i)
                continue

            if not isinstance(state.grammar, futures.Future):
                raise TypeError(
                    f"Queued grammar state must hold a Future, got "
                    f"{type(state.grammar).__name__}: {state=}"
                )

            if state.grammar.done():
                ready_idxs.add(i)
                continue

            # Two-phase timeout:
            #  - while still queued in the executor, bound the queue-wait by
            #    compile_timeout_secs measured from queued_ts;
            #  - once the worker actually starts running init_value_impl,
            #    switch to a fresh compile_timeout_secs budget measured
            #    from compile_started_at (so executor queueing doesn't eat
            #    into the per-compile budget).
            # Worst-case total wait is ~2 * compile_timeout_secs, but a
            # request can never wait forever even if the executor is wedged.
            started_at = self.grammar_backend.compile_started_at(state.grammar_key)

            if started_at is None:
                elapsed = now - state.grammar_queued_ts

            else:
                elapsed = now - started_at

            if elapsed >= self.compile_timeout_secs:
                # Closes the race where the worker finishes between the
                # state.grammar.done() check above and the elapsed check
                # here: prefer admitting a successfully-completed compile
                # over aborting it for timeout.
                if state.grammar.done():
                    ready_idxs.add(i)

                else:
                    failed_idxs.add(i)

        if self.grammar_sync_size > 1:
            gathered: list[tuple[set, set]] = [None] * self.grammar_sync_size

            torch.distributed.all_gather_object(
                gathered,
                (ready_idxs, failed_idxs),
                group=self.grammar_sync_group,
            )

            ready_idxs = set.intersection(*[g[0] for g in gathered])
            failed_idxs = set.union(*[g[1] for g in gathered])

        if not ready_idxs and not failed_idxs:
            return []

        promoted: list[QueueEntry] = []

        for i in sorted(ready_idxs):
            spec, state, bootstrap = self.grammar_queue[i]
            promoted.append((spec, state, bootstrap))

            if state.finished or state.grammar is None:
                continue

            # init_value_impl is supposed to fold compile errors into an
            # InvalidGrammarObject, but it can leak (e.g. KeyError from the
            # structural_tag legacy dict-walk in xgrammar_backend). Catching
            # here keeps the leak from re-raising out of the event loop.
            try:
                value = state.grammar.result()
            except Exception as e:
                value = InvalidGrammarObject(f"{type(e).__name__}: {e}")

            if value.is_invalid:
                state.grammar = None

                state.set_finish_with_abort(
                    f"Failed to compile {state.grammar_key[0]} grammar: "
                    f"{value.error_message}"
                )

            else:
                # Future.result() returns the same object to every waiter on a
                # shared compile; copy so each request has its own matcher
                # state.
                state.grammar = value.copy()

        # Dedupe record_compile_timeout by key: multiple concurrent
        # requests share a single compile, so one timeout wave would
        # otherwise increment the per-key retry counter N times and
        # blow past compile_max_retries in a single pass.
        timed_out_keys: set = set()

        for i in sorted(failed_idxs):
            spec, state, bootstrap = self.grammar_queue[i]
            promoted.append((spec, state, bootstrap))

            # Race: the compile future may have completed successfully
            # between the classification loop and this block. Re-check
            # done() so we don't falsely time out a compile that just
            # finished — the cost of one extra ``result()`` here is
            # negligible compared to the hours a user could spend
            # debugging a phantom timeout.
            if isinstance(state.grammar, futures.Future) and state.grammar.done():
                try:
                    value = state.grammar.result()
                except BaseException:
                    value = None

                if value is not None and not value.is_invalid:
                    # See above: copy per-request so concurrent waiters on a
                    # shared compile future don't share matcher state.
                    state.grammar = value.copy()
                    continue

            # Don't cancel the future: it's shared across all
            # concurrent requests for this key. The compile runs to
            # completion in the background; its result (or the timeout
            # marker we post below) lands in the cache for future
            # requests to consume.
            state.grammar = None

            timeout_msg = (
                f"Grammar compilation timed out after {self.compile_timeout_secs:.1f}s"
            )

            # Cache the timeout so concurrent retries short-circuit
            # instead of all timing out one by one. The backend tracks
            # attempt count per key: each timeout caches a TTL'd marker,
            # and after compile_max_retries timeouts the marker is
            # escalated to permanent (the compiler is consistently
            # broken for this key).
            if state.grammar_key not in timed_out_keys:
                timed_out_keys.add(state.grammar_key)
                self.grammar_backend.record_compile_timeout(
                    state.grammar_key,
                    timeout_msg,
                    ttl_secs=self.compile_timeout_secs,
                    max_retries=self.compile_max_retries,
                )

            state.set_finish_with_abort(f"{timeout_msg}: key={state.grammar_key}")

        drop = ready_idxs | failed_idxs

        self.grammar_queue = [
            entry for i, entry in enumerate(self.grammar_queue) if i not in drop
        ]

        return promoted
