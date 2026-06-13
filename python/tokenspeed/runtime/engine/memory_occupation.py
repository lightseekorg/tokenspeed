# SPDX-License-Identifier: Apache-2.0
"""GPU-memory data plane for sleep/wake (release/resume_memory_occupation).

Pairs with the control-plane :class:`PauseController`. A *release* pauses and
drains the scheduler (delegated to the pause controller) and only then frees GPU
memory via the torch_memory_saver adapter; a *resume* re-maps memory and, when
the KV region returns, repairs the KV cache. Tags (``weights``, ``kv_cache``)
are freed/restored independently so the RL loop can resume weights, push fresh
weights, then resume the KV cache.

This module knows nothing about ZMQ beyond a ``send_func`` reply socket, and
nothing about scheduling beyond delegating drain to the pause controller.
"""

from __future__ import annotations

from collections.abc import Callable

from tokenspeed.runtime.engine.io_struct import (
    IsSleepingReqInput,
    IsSleepingReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
)
from tokenspeed.runtime.engine.pause import PauseController

VALID_TAGS = ("weights", "kv_cache")


def _normalize_tags(tags: list[str] | None) -> tuple[list[str] | None, str | None]:
    """Return ``(ordered_tags, error)``. ``None`` ⇒ all tags. The order is fixed
    (``weights`` before ``kv_cache``) for determinism; an unknown tag yields an
    error and ``None`` tags."""
    if tags is None:
        return list(VALID_TAGS), None
    bad = [t for t in tags if t not in VALID_TAGS]
    if bad:
        return None, f"invalid tags: {bad!r}; valid: {list(VALID_TAGS)}"
    return [t for t in VALID_TAGS if t in tags], None


class MemoryOccupationController:
    """Owns the data-plane release/resume for one scheduler event loop."""

    def __init__(
        self,
        *,
        send_func,
        pause_controller: PauseController,
        adapter,
        enabled: bool,
        reset_caches_fn: Callable[[], None],
        kv_repair_fn: Callable[[], None],
        kv_cache_release_allowed: bool = True,
    ) -> None:
        self._send = send_func
        self._pause = pause_controller
        self._adapter = adapter
        self._enabled = enabled
        self._reset_caches = reset_caches_fn
        self._kv_repair = kv_repair_fn
        # Whether releasing the ``kv_cache`` region is safe in this engine. False
        # when prefix caching is on but the scheduler exposes no prefix-cache
        # reset: discarded KV would leave stale cache entries that survive wake
        # and produce silently wrong hits, so we reject the release up front
        # rather than free KV we cannot invalidate. A declared capability (set
        # at construction), not a runtime duck-typed probe.
        self._kv_cache_release_allowed = kv_cache_release_allowed
        self.released_tags: set[str] = set()

    @property
    def is_sleeping(self) -> bool:
        """True while any GPU-memory region is released (data-plane sleep)."""
        return bool(self.released_tags)

    # -- control-request handlers (driven by the request handler) -------------

    def handle_release(self, req: ReleaseMemoryOccupationReqInput) -> None:
        if not self._enabled:
            self._fail_release("memory saver not enabled (--enable-memory-saver)")
            return
        tags, err = _normalize_tags(req.tags)
        if err is not None:
            self._fail_release(err)
            return
        if "kv_cache" in tags and not self._kv_cache_release_allowed:
            self._fail_release(
                "cannot release kv_cache: prefix caching is enabled but the "
                "scheduler has no prefix-cache reset, so discarded KV would "
                "leave stale cache entries after wake"
            )
            return
        already = [t for t in tags if t in self.released_tags]
        if already:
            self._fail_release(f"tags already released: {already!r}")
            return
        if self._pause.is_drain_pending:
            self._fail_release("a pause or release is already in progress")
            return
        # Defer the actual free until the scheduler drains. wait-mode: let
        # in-flight requests finish (the RL rollout is idle between steps).
        self._pause.request_drain(
            abort_inflight=False,
            on_drained=lambda: self._finish_release(tags),
            on_cancelled=lambda: self._fail_release("resumed before release drained"),
        )

    def _finish_release(self, tags: list[str]) -> None:
        if "kv_cache" in tags:
            # KV is discarded; any prefix-cache entry pointing at it is stale.
            self._reset_caches()
        for tag in tags:
            self._adapter.pause(tag=tag)
            self.released_tags.add(tag)
        self._pause.set_released(True)
        self._send.send_pyobj(ReleaseMemoryOccupationReqOutput(success=True))

    def _fail_release(self, message: str) -> None:
        self._send.send_pyobj(
            ReleaseMemoryOccupationReqOutput(success=False, message=message)
        )

    def handle_resume(self, req: ResumeMemoryOccupationReqInput) -> None:
        # ``None`` on resume means "wake exactly what is asleep" — NOT all valid
        # tags. Expanding to all (as on release) would make a bare resume after a
        # partial release (e.g. weights-only) fail on the never-released tag,
        # stranding the released region. Resolve against ``released_tags`` so a
        # partial release round-trips with a default resume_memory_occupation().
        if req.tags is None:
            tags = [t for t in VALID_TAGS if t in self.released_tags]
        else:
            tags, err = _normalize_tags(req.tags)
            if err is not None:
                self._send.send_pyobj(
                    ResumeMemoryOccupationReqOutput(success=False, message=err)
                )
                return
        not_released = [t for t in tags if t not in self.released_tags]
        if not_released:
            self._send.send_pyobj(
                ResumeMemoryOccupationReqOutput(
                    success=False, message=f"tags not released: {not_released!r}"
                )
            )
            return
        for tag in tags:
            self._adapter.resume(tag=tag)
            self.released_tags.discard(tag)
        if "kv_cache" in tags:
            self._kv_repair()
        if not self.released_tags:
            # Fully awake → resume scheduling.
            self._pause.set_released(False)
        self._send.send_pyobj(ResumeMemoryOccupationReqOutput(success=True))

    def handle_is_sleeping(self, req: IsSleepingReqInput) -> None:
        self._send.send_pyobj(IsSleepingReqOutput(is_sleeping=self.is_sleeping))
