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

"""Base classes for grammar-guided constrained decoding backends."""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass
class CacheEntry:

    value: Any
    event: Event

    # Set non-None when a worker begins compiling for this key. Doubles as
    # an ownership marker so a second ``init_value`` call for the same key
    # waits on ``event`` instead of starting a redundant compile.
    started_at: float | None = None

    # Shared future for the in-flight compile, set by
    # ``get_cached_or_future_value`` under the cache lock. Concurrent
    # callers for the same key reuse this instead of submitting duplicate
    # tasks that would saturate the executor on repeated same-schema traffic.
    future: Future | None = None


class BaseGrammarObject:
    """Base for compiled grammar wrappers. Concrete backends subclass this."""

    is_invalid: bool = False
    expires_at: float | None = None

    @property
    def expired(self) -> bool:

        return False


class InvalidGrammarObject(BaseGrammarObject):
    """Sentinel cached when a grammar fails to compile (or compile times out).

    Carries the underlying error string so the request abort message can
    surface the actual reason — bad regex, malformed schema, timeout, etc.

    ``expires_at`` (monotonic seconds) lets transient failures decay so a
    one-off slow compile doesn't poison a valid key forever. ``None`` means
    the failure is permanent (e.g. malformed schema — recompiling won't help).
    """

    is_invalid = True

    def __init__(
        self,
        error_message: str = "Unknown grammar error",
        expires_at: float | None = None,
    ) -> None:

        super().__init__()

        self.error_message = error_message
        self.expires_at = expires_at

    @property
    def expired(self) -> bool:

        return self.expires_at is not None and time.monotonic() >= self.expires_at

    def __repr__(self) -> str:

        return (
            f"InvalidGrammarObject(error_message={self.error_message!r}, "
            f"expires_at={self.expires_at!r})"
        )


@dataclass
class TimeoutHistory:
    """Per-key bookkeeping so the cache can escalate to a permanent failure
    after enough transient timeouts (a single slow compile shouldn't poison
    the key, but a key that *consistently* times out is broken).
    """

    count: int = 0


class BaseGrammarBackend:
    """Base backend with shared cache management for grammar objects."""

    def __init__(self):

        self.executor = ThreadPoolExecutor()

        self.cache: dict[tuple[str, str], CacheEntry] = {}
        self.cache_lock = Lock()

        # Tracks transient-timeout attempts per key. Cleared on a successful
        # compile (in init_value) or on reset().
        self.timeout_history: dict[tuple[str, str], TimeoutHistory] = {}

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def init_value_impl(
        self, key: tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # Cache API
    # ------------------------------------------------------------------

    def init_value(self, key: tuple[str, str]) -> BaseGrammarObject:

        with self.cache_lock:

            entry = self.cache.get(key)

            if entry is None:

                entry = CacheEntry(None, Event(), started_at=time.monotonic())
                self.cache[key] = entry

                cache_hit = False

            elif entry.event.is_set() or entry.started_at is not None:

                # Either already compiled, or another worker claimed the
                # compile (``started_at`` set). Wait on the event below.
                cache_hit = True

            else:

                # Entry was pre-created by ``get_cached_or_future_value``
                # but no worker has claimed it yet — take ownership.
                entry.started_at = time.monotonic()
                cache_hit = False

        if cache_hit:

            entry.event.wait()

        else:

            # Backends should return InvalidGrammarObject(message); accept
            # legacy None as "unknown error" for safety.
            if (value := self.init_value_impl(key, False)) is None:

                value = InvalidGrammarObject()

            with self.cache_lock:

                is_active = self.cache.get(key) is entry

                if is_active:

                    # If a parallel caller (e.g. GrammarManager on
                    # compile timeout) wrote an InvalidGrammarObject
                    # while we were compiling, keep that marker as long
                    # as it hasn't expired — otherwise a slow compile
                    # finishing a moment after timeout would silently
                    # un-invalidate the cache. Once the marker expires
                    # we let the freshly-compiled value take over so a
                    # transient slow compile doesn't poison the key
                    # forever.
                    cached = entry.value

                    if not (
                        cached is not None and cached.is_invalid and not cached.expired
                    ):
                        entry.value = value

                        if not value.is_invalid:
                            # Successful compile — clear timeout bookkeeping.
                            self.timeout_history.pop(key, None)

                else:

                    # Orphan: our entry was evicted (e.g. its timeout
                    # marker expired and ``get_cached_or_future_value``
                    # replaced it). Don't touch shared state —
                    # ``timeout_history`` now belongs to the new attempt
                    # — but still publish our compiled value on the old
                    # entry so any future/wait still holding a reference
                    # to it resolves with a usable result instead of
                    # deadlocking or returning None.
                    entry.value = value

                entry.event.set()

        if entry.value.is_invalid:

            return entry.value

        return entry.value.copy()

    def get_cached_or_future_value(
        self, key: tuple[str, str]
    ) -> tuple[BaseGrammarObject | Future, bool]:
        """Return (value, cache_hit).

        On cache hit: value is either a fresh grammar copy or an
        InvalidGrammarObject carrying the original compile error.
        On miss: value is a Future that resolves to the same.

        Expired InvalidGrammarObject markers are evicted on lookup so a
        transient timeout decays into a retry instead of poisoning the key.
        """
        with self.cache_lock:

            entry = self.cache.get(key)

            if (
                entry is not None
                and entry.event.is_set()
                and entry.value.is_invalid
                and entry.value.expired
            ):
                # Drop the stale marker.
                del self.cache[key]
                entry = None

            if entry is not None and entry.event.is_set():

                if entry.value.is_invalid:

                    return entry.value, True

                return entry.value.copy(), True

            # In-flight compile — share its future so concurrent callers
            # for the same key don't each submit duplicate work that
            # would park executor workers on the same event.
            if entry is not None and entry.future is not None:

                return entry.future, False

            # Pre-create the entry under the lock so any caller that
            # races in after us finds our shared future instead of
            # submitting its own.
            if entry is None:

                entry = CacheEntry(value=None, event=Event())
                self.cache[key] = entry

            entry.future = self.executor.submit(self.init_value, key)

            return entry.future, False

    def compile_started_at(self, key: tuple[str, str]) -> float | None:
        """Monotonic timestamp at which the worker thread began running
        ``init_value_impl`` for ``key``, or None if no compile has ever
        started for this key (no cache entry, or the entry was created
        directly via ``cache_invalid`` / ``record_compile_timeout`` without
        a backing compile).

        Returned even after the compile has finished — callers want
        "compile-only elapsed", which is ``now - started_at`` regardless of
        completion. Otherwise a worker that finishes between a caller's
        ``future.done()`` check and the elapsed-time check would silently
        flip the elapsed calculation back to wall-clock-from-submit and
        could trigger a spurious timeout against a request that actually
        succeeded.
        """
        with self.cache_lock:

            entry = self.cache.get(key)

            return entry.started_at if entry is not None else None

    def record_compile_timeout(
        self,
        key: tuple[str, str],
        error_message: str,
        ttl_secs: float,
        max_retries: int,
    ) -> None:
        """Cache a compile-timeout marker for ``key``.

        Each call increments a per-key attempt counter. While the count is at
        or below ``max_retries`` the marker has a finite TTL so the next
        request (after the marker expires) gets a fresh compile attempt.
        Once the count crosses the threshold the marker is escalated to
        permanent (no TTL) — at that point the compiler is consistently
        broken for this key and there's nothing to gain by retrying.

        Spurious-timeout safety: if the cache already holds a valid grammar
        (the worker just finished compiling, racing this call), this is a
        no-op — we don't penalize a working key, and we don't clobber a
        freshly-committed result with a stale timeout.

        Counter reset semantics:
          - cleared on a successful compile (see ``init_value``)
          - cleared on ``reset()``
        """
        with self.cache_lock:

            entry = self.cache.get(key)

            if (
                entry is not None
                and entry.event.is_set()
                and not entry.value.is_invalid
            ):
                # A valid grammar landed in the cache while we were about to
                # declare timeout. Drop the timeout — the grammar works.
                return

            history = self.timeout_history.setdefault(key, TimeoutHistory())
            history.count += 1

            if history.count > max_retries:

                expires_at = None  # permanent

                error_message = (
                    f"{error_message} (gave up after {history.count - 1} retries)"
                )

            else:

                expires_at = time.monotonic() + ttl_secs

            invalid = InvalidGrammarObject(error_message, expires_at=expires_at)

            if entry is None:

                event = Event()
                event.set()

                self.cache[key] = CacheEntry(invalid, event)

            elif not entry.event.is_set():

                entry.value = invalid
                entry.event.set()

            else:

                # entry.value.is_invalid (valid was filtered above) — refresh.
                entry.value = invalid

    def cache_invalid(self, key: tuple[str, str], error_message: str) -> None:
        """Cache a permanent compile failure (e.g. bad syntax — retrying
        won't help). Use ``record_compile_timeout`` for transient failures."""
        invalid = InvalidGrammarObject(error_message, expires_at=None)

        with self.cache_lock:

            entry = self.cache.get(key)

            if entry is None:

                event = Event()
                event.set()

                self.cache[key] = CacheEntry(invalid, event)

            elif not entry.event.is_set():

                entry.value = invalid
                entry.event.set()

            elif entry.value.is_invalid:

                entry.value = invalid

    def reset(self):

        with self.cache_lock:

            self.cache.clear()
            self.timeout_history.clear()


def create_grammar_backend(server_args: ServerArgs, tokenizer, vocab_size):

    if server_args.grammar_backend == "none":

        return None

    elif server_args.grammar_backend == "xgrammar":

        from tokenspeed.runtime.grammar.xgrammar_backend import (
            XGrammarGrammarBackend,
        )

        grammar_backend = XGrammarGrammarBackend(
            tokenizer,
            vocab_size=vocab_size,
            disable_any_whitespace=server_args.disable_any_whitespace,
        )

    else:

        raise ValueError(f"Invalid grammar backend: {server_args.grammar_backend}")

    # Reasoning + grammar deferral lives in the OpenAI serving layer now:
    # response_format is rewritten into an xgrammar structural_tag whose
    # trigger covers the post-reasoning preamble. gpt-oss is wired today
    # (``<|start|>assistant<|channel|>final<|message|>``); other reasoning
    # parsers (qwen3-thinking, deepseek-r1, ...) can be added the same
    # way as needed. The previous token-id-based ``ReasonerGrammarBackend``
    # wrapper has been removed: it couldn't handle multi-token channel
    # preambles and carried known P1 bugs around state cloning + reasoning
    # initialization.
    return grammar_backend


def get_apply_vocab_mask_func(grammar_backend: str):
    """Return the backend-specific in-place vocab-mask-apply function.

    The function's signature is ``(logits, vocab_mask) -> None``. It is
    stored on ``SamplingBatchInfo.apply_vocab_mask`` so the captured
    sampler can call it without branching on backend. Only xgrammar is
    wired up today; add branches here when new backends are added.
    """
    if grammar_backend == "xgrammar":

        from xgrammar import apply_token_bitmask_inplace

        # xgrammar's native signature is (logits, bitmask, *, ...);
        # adapt to the canonical (logits, vocab_mask) kwargs the
        # sampler uses.
        def _apply(logits, vocab_mask):
            apply_token_bitmask_inplace(logits, vocab_mask)

        return _apply

    raise ValueError(f"Unsupported grammar backend: {grammar_backend}")
