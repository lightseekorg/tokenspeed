# SPDX-License-Identifier: Apache-2.0
#
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

"""Pause / resume control state for a scheduler event loop.

The pause gate lives in Python: requests are admitted to the scheduler from the
event loop (``scheduler.submit_requests``), so withholding new work while paused
is handled here rather than inside the scheduler itself.

Two classes split the work: :class:`PauseController` owns the state machine and
is driven by the request handler (control plane); :class:`PauseHooks` is the
EventLoop-side integration — everything the pause/resume API needs from the
loop lives there, so the loop's normal scheduling paths carry single-line
hooks only.

Modes (how a pause treats in-flight requests):

- ``abort``: cancel in-flight requests, then drain and reply.
- ``wait`` : let in-flight requests finish naturally, then drain and reply.
- ``keep`` : freeze everything in place; reply immediately; resume later.

``abort`` and ``wait`` both leave the scheduler in ``PAUSED_NEW`` (no new
requests admitted, running requests keep stepping) and defer their reply until
the scheduler has drained. ``keep`` moves to ``PAUSED_ALL`` (nothing scheduled)
and replies immediately.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from tokenspeed.runtime.engine.io_struct import (
    IsSchedulerPausedReqInput,
    IsSchedulerPausedReqOutput,
    PauseSchedulerReqInput,
    PauseSchedulerReqOutput,
    ResumeSchedulerReqInput,
    ResumeSchedulerReqOutput,
)

logger = logging.getLogger(__name__)

# Sleep between iterations while frozen (PAUSED_ALL) so the keep-mode pause does
# not busy-spin a CPU core waiting for /resume.
_PAUSED_IDLE_SLEEP_S = 0.001


@dataclass
class _PendingDrain:
    """A deferred action resolved when the scheduler drains.

    ``on_drained`` runs once ``scheduler_drained`` is true and ``ready`` (when
    provided) succeeds: it sends the success reply and, for a memory release,
    frees GPU memory. ``on_cancelled`` runs if a resume arrives before the drain
    completes: it sends the failure reply to the correct communicator (pause vs
    release use different ZMQ channels, so the action carries its own reply
    rather than the controller hard-coding one).
    """

    on_drained: Callable[[], None]
    on_cancelled: Callable[[], None]
    ready: Callable[[], bool] | None = None


class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - ``UNPAUSED``: normal operation.
    - ``PAUSED_NEW``: no new requests admitted; running requests keep stepping.
    - ``PAUSED_ALL``: nothing scheduled; everything frozen in place.
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


def scheduler_drained(scheduler) -> bool:
    """True when the scheduler holds no requests that need a forward pass.

    Covers every state that still needs a forward pass. Submitted and Retracted
    requests are both included in waiting_size.
    Post-finish writeback states are async teardown and do not block a drain.
    """
    return (
        scheduler.waiting_size() == 0
        and scheduler.decoding_size() == 0
        and scheduler.prefilling_size() == 0
    )


class PauseController:
    """Owns pause/resume state for one scheduler event loop.

    Split of responsibilities: the ``handle_*`` methods are driven by the
    request handler (they only touch local state + the reply socket); the event
    loop queries :pyattr:`admit_blocked` / :pyattr:`forward_blocked`, drains
    buffered specs on resume, and calls :pymeth:`maybe_finish_drain` once per
    iteration to resolve a deferred abort/wait reply.

    ``send_func`` is the scheduler→tokenizer reply socket (a no-op
    ``NullSender`` on non-rank-0 TP ranks, matching the existing control-reply
    pattern), so the ``handle_*`` methods are safe to call on every rank.
    """

    def __init__(self, send_func) -> None:
        self._send = send_func
        self.state = PauseState.UNPAUSED
        # RequestSpecs withheld from the scheduler while paused; flushed on resume.
        self.buffered_specs: list = []
        # Deferred post-drain action for abort/wait pause OR memory release; held
        # until the scheduler drains. Single-consumer: only one may be armed.
        self._pending_drain: _PendingDrain | None = None
        # True once GPU memory has actually been released (data plane). Distinct
        # from forward_blocked: PAUSED_ALL alone still permits DP idle forwards,
        # which run the model (touch weights) and must be suppressed while the
        # weights region is unmapped.
        self.released: bool = False
        # Set by a pause(mode="abort"); consumed once by the event loop to
        # cancel in-flight requests already in the scheduler.
        self._abort_all_pending = False
        # Set by abort/wait; consumed once by the event loop to cancel requests
        # still compiling in the grammar queue (not yet in the scheduler).
        self._cancel_grammar_pending = False

    # -- state queried by the event loop --------------------------------------

    @property
    def is_paused(self) -> bool:
        return self.state != PauseState.UNPAUSED

    @property
    def admit_blocked(self) -> bool:
        """Whether new requests should be withheld from the scheduler."""
        return self.state != PauseState.UNPAUSED

    @property
    def forward_blocked(self) -> bool:
        """Whether the loop should run no forward work this iteration."""
        return self.state == PauseState.PAUSED_ALL

    def consume_abort_all(self) -> bool:
        """Return (once) whether the event loop should cancel all in-flight reqs."""
        if self._abort_all_pending:
            self._abort_all_pending = False
            return True
        return False

    def consume_cancel_grammar(self) -> bool:
        """Return (once) whether the event loop should cancel grammar-queued reqs.

        Set for abort/wait: requests still compiling a grammar are not yet in
        the scheduler or ``rid_to_state``, so the abort sweep and the drain
        check both miss them. Left compiling, they would be promoted and either
        run after a weight swap (abort) or be buffered past a wait drain. They
        have produced no output, so cancelling them is safe.
        """
        if self._cancel_grammar_pending:
            self._cancel_grammar_pending = False
            return True
        return False

    def buffer_specs(self, specs: list) -> None:
        self.buffered_specs.extend(specs)

    def take_buffered_specs(self) -> list:
        specs, self.buffered_specs = self.buffered_specs, []
        return specs

    # -- generic drain machinery (shared by pause and memory release) ----------

    @property
    def is_drain_pending(self) -> bool:
        return self._pending_drain is not None

    def request_drain(
        self,
        *,
        abort_inflight: bool,
        on_drained: Callable[[], None],
        on_cancelled: Callable[[], None],
        ready: Callable[[], bool] | None = None,
    ) -> bool:
        """Start a wait-style drain (PAUSED_NEW, cancel grammar-queued) and arm a
        post-drain action. Returns False if a drain is already pending (the
        caller should send its own busy reply). ``abort_inflight=True`` also
        cancels in-flight requests (abort mode); False lets them finish (wait
        mode / memory release). An optional ``ready`` gate can keep a drained
        action pending while an asynchronous prerequisite completes."""
        if self._pending_drain is not None:
            return False
        self.state = PauseState.PAUSED_NEW
        self._pending_drain = _PendingDrain(
            on_drained=on_drained,
            on_cancelled=on_cancelled,
            ready=ready,
        )
        self._cancel_grammar_pending = True
        if abort_inflight:
            self._abort_all_pending = True
        return True

    def set_released(self, released: bool) -> None:
        """Mark GPU memory released (freeze fully) or restored (unpause)."""
        self.released = released
        self.state = PauseState.PAUSED_ALL if released else PauseState.UNPAUSED

    # -- control-request handlers (driven by the request handler) -------------

    def handle_pause(self, req: PauseSchedulerReqInput) -> None:
        if req.mode not in ("abort", "wait", "keep"):
            self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message=f"invalid pause mode: {req.mode!r}"
                )
            )
            return

        # Reject any new pause while an abort/wait pause or memory release is
        # still draining: the post-drain action is a single-consumer promise
        # (``_pending_drain``), so a second drain would overwrite it and strand
        # the first caller forever on its ZMQ await. ``keep`` never arms a drain,
        # so it can't be the *first* pause here, but it must not clobber a
        # draining one either.
        if self._pending_drain is not None:
            self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message="a pause is already in progress"
                )
            )
            return

        if req.mode == "keep":
            # Freeze in place and reply now — nothing to drain.
            self.state = PauseState.PAUSED_ALL
            self._send.send_pyobj(PauseSchedulerReqOutput(success=True))
            return

        # abort / wait: stop admitting new requests, keep stepping so in-flight
        # requests drain, and reply once the scheduler is empty. Both also
        # cancel grammar-queued (still-compiling) pre-pause requests.
        self.request_drain(
            abort_inflight=(req.mode == "abort"),
            on_drained=lambda: self._send.send_pyobj(
                PauseSchedulerReqOutput(success=True)
            ),
            on_cancelled=lambda: self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message="resumed before pause drained"
                )
            ),
        )

    def handle_resume(self, req: ResumeSchedulerReqInput) -> None:
        # Reject a scheduler-level resume while GPU memory is still released.
        # ``released`` is owned by the memory controller (its ``set_released``
        # is the sole writer); clearing it here would flip the state to
        # UNPAUSED without remapping the weights/KV regions, so the next admit
        # or DP idle forward would touch unmapped memory. The caller must wake
        # via ``resume_memory_occupation`` instead.
        if self.released:
            self._send.send_pyobj(
                ResumeSchedulerReqOutput(
                    success=False,
                    message=(
                        "memory is released; call resume_memory_occupation to "
                        "wake before resuming the scheduler"
                    ),
                )
            )
            return
        # If a wait/abort pause is still awaiting its drain reply, it has NOT
        # drained — ``maybe_finish_drain`` clears ``_pending_reply`` the instant
        # it does. We must still reply (resume uses a separate communicator and
        # cannot otherwise wake the pause caller, who would block forever), but
        # the reply must be a failure: acking success here would tell a
        # weight-swapping caller it is safe to proceed while pre-pause requests
        # are still in flight under the old weights.
        if self._pending_drain is not None:
            action = self._pending_drain
            self._pending_drain = None
            action.on_cancelled()
        # Buffered specs are flushed by the event loop on its next admission
        # pass (state is already UNPAUSED by then). ``released`` is intentionally
        # NOT touched here — see the guard above; only set_released() writes it.
        self.state = PauseState.UNPAUSED
        self._abort_all_pending = False
        self._cancel_grammar_pending = False
        self._send.send_pyobj(ResumeSchedulerReqOutput(success=True))

    def handle_is_paused(self, req: IsSchedulerPausedReqInput) -> None:
        self._send.send_pyobj(IsSchedulerPausedReqOutput(is_paused=self.is_paused))

    # -- per-iteration drain check (driven by the event loop) -----------------

    def maybe_finish_drain(self, scheduler) -> None:
        """Resolve a deferred pause/release action once the scheduler has drained.

        The action is cleared *before* it runs so a release's ``on_drained`` can
        re-arm controller state (``set_released``) without tripping the
        single-consumer guard.
        """
        if self._pending_drain is None:
            return
        if not scheduler_drained(scheduler):
            return
        action = self._pending_drain
        if action.ready is not None and not action.ready():
            return
        self._pending_drain = None
        action.on_drained()


class PauseHooks:
    """EventLoop-side pause/resume integration.

    Stateless glue: all pause state stays in :class:`PauseController` and the
    loop's collaborators; this class only holds a loop back-reference and acts
    on them, so the loop's normal scheduling paths carry single-line hooks
    only. Attribute access on ``loop`` is lazy, so it may be constructed
    before the loop's collaborators exist.
    """

    def __init__(self, loop, controller: PauseController, device) -> None:
        self._loop = loop
        self._pause = controller
        # Injected, not reached through the loop: a hook that has to walk
        # ``loop.<something>.<something>`` to get at the GPU is one traversal
        # away from walking one step further.
        self._device = device

    # -- admission-path hooks (called from _process_new_requests) -------------

    def apply_transitions(self, grammar_manager) -> None:
        """Apply one-shot pause/resume edges to in-flight and queued requests.

        Called once per iteration from ``_process_new_requests``, right after
        control messages were processed (so a fresh edge is acted on in the
        same iteration) and before the early ``if not ready: return`` (which
        would otherwise strand buffered specs until the next inbound request).

        - pause(mode="abort"): cancel every in-flight request through the same
          marker path as a client abort; they finish on their next scheduled
          step, then the drain check resolves the pause reply. notify_client:
          a pause aborts a passive client's request, so it must receive a
          terminating finish (unlike a client abort).
        - abort/wait: cancel requests still compiling in the grammar queue —
          they are not yet in ``rid_to_state`` or the scheduler, so the abort
          sweep and the drain check both miss them. A finished state makes the
          next ``get_ready_grammar_requests`` pass publish them instead of
          admitting, so they never run under post-resume weights or strand the
          drain.
        - resume: flush specs buffered while paused, even when no new request
          arrives this iteration. Specs aborted while paused are reaped in
          place rather than admitted, so they don't burn a scheduler slot or
          leak their rid — see ``_reap_or_keep_buffered_spec``.
        """
        loop = self._loop
        if self._pause.consume_abort_all():
            for rid in list(loop.output_processor.rid_to_state.keys()):
                loop._request_abort_or_mark(
                    rid, "request aborted by pause", notify_client=True
                )
                grammar_manager.mark_abort(rid)

        if self._pause.consume_cancel_grammar():
            for _, state, _ in grammar_manager.grammar_queue:
                state.set_finish_with_abort("Aborted by pause", notify_client=True)

        if not self._pause.admit_blocked and self._pause.buffered_specs:
            specs = [
                spec
                for spec in self._pause.take_buffered_specs()
                if self._reap_or_keep_buffered_spec(spec)
            ]
            if specs:
                loop._submit_scheduler_requests(specs)

    def _reap_or_keep_buffered_spec(self, spec) -> bool:
        """Resolve a buffered spec on resume; return True if it should be admitted.

        A buffered spec was already registered in ``rid_to_state`` before it was
        withheld, so if it was aborted while paused it never reached the
        scheduler and the forward path can never reap it. Handle that here:

        - state missing  -> already published and reaped; drop silently.
        - state finished -> aborted in place. Stream a terminating finish for
          pause-initiated aborts (the passive client is still waiting) and drop
          the registered state so the rid does not leak; client-initiated aborts
          already tore down their own state, so just reap.
        - otherwise      -> still live; admit it.
        """
        output_processor = self._loop.output_processor
        state = output_processor.rid_to_state.get(spec.request_id)
        if state is None:
            return False
        if state.finished:
            output_processor.reap_finished_orphan(spec.request_id, state)
            return False
        return True

    def withhold_admissions(
        self, admitted_specs: list, pause_blocked_before: bool
    ) -> bool:
        """Pause admission gate: while paused, buffer ``admitted_specs`` instead
        of submitting them (running requests keep stepping) and return True so
        the caller skips submission. Buffered specs are flushed on resume by
        ``apply_transitions``, ahead of any newly-admitted ones, preserving
        FIFO order.

        TODO(pause-fifo): recv_reqs() drains the socket non-blocking, so a
        generate request that arrived *before* a pause control message can be
        coalesced into the same batch and reach here after the pause flipped
        admit_blocked. Such a pre-pause request is buffered as post-pause work
        instead of running (wait) / being aborted (abort). Correct handling
        needs the batch processed as an ordered stream that respects the
        control request's FIFO position. Tracked as a follow-up; until then we
        warn when the coalescing condition is observed so it is not silent.
        """
        if not self._pause.admit_blocked:
            return False
        if admitted_specs and not pause_blocked_before:
            logger.warning(
                "Pause engaged in the same recv batch as %d generate "
                "request(s) (rids=%s); their FIFO order relative to the "
                "pause is not preserved, so a pre-pause request may be "
                "buffered as post-pause work and run only after resume. "
                "See TODO(pause-fifo).",
                len(admitted_specs),
                [spec.request_id for spec in admitted_specs],
            )
        self._pause.buffer_specs(admitted_specs)
        return True

    # -- freeze-loop hook (called from event_loop) -----------------------------

    def paused_idle_step(self) -> None:
        """Run one iteration under ``PAUSED_ALL`` (keep mode): no new forward
        work, but keep DP ranks in lockstep and yield the CPU so the freeze
        does not busy-spin a core. The drain check runs at the event loop's
        tail (after the round's results advance the scheduler), not here."""
        loop = self._loop
        if loop.has_dp:
            dp_metadata = loop._dp_sync_and_check(None)
            # While memory is released the weights region is unmapped; an idle
            # forward runs the model and would read freed memory. All DP ranks
            # release together, so skipping the idle forward stays consistent
            # across ranks (the small DP sync above still runs to keep lockstep).
            if dp_metadata.need_idle_forward and not self._pause.released:
                self._device.run_idle_forward(dp_metadata)

        time.sleep(_PAUSED_IDLE_SLEEP_S)

    # -- memory release/wake data plane (wired into MemoryOccupationController) -

    def reset_caches_for_release(self) -> bool:
        """Invalidate the prefix/single-table cache before KV is discarded on release.

        KV pages are re-mapped + zeroed on wake, so any retained prefix entry
        would be stale. The unsafe case (prefix caching on with no reset) is
        rejected up front in ``MemoryOccupationController.handle_release`` via
        ``kv_cache_release_allowed``, so by the time we get here either a clear
        exists or prefix caching is off (nothing to invalidate). Returns False
        while an asynchronous cache transfer still pins L1 so the release can
        remain pending and retry on the next event-loop iteration.
        """
        clear = getattr(self._loop.scheduler, "clear_l1_cache", None)
        return not callable(clear) or clear()

    def kv_repair_after_wake(self) -> None:
        """Zero re-mapped KV buffers (garbage after re-map) for every KV pool.

        The pools live behind the data plane and the zeroing is device work,
        so the handle owns which pools are walked and does the walking; this
        side only says when.
        """
        self._device.run_kv_repair()
