# SPDX-License-Identifier: Apache-2.0
"""Pause / resume control state for a scheduler event loop.

The pause gate lives in Python: requests are admitted to the scheduler from the
event loop (``scheduler.submit_requests``), so withholding new work while paused
is handled here rather than inside the scheduler itself.

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

from tokenspeed.runtime.engine.io_struct import (
    IsSchedulerPausedReqInput,
    IsSchedulerPausedReqOutput,
    PauseSchedulerReqInput,
    PauseSchedulerReqOutput,
    ResumeSchedulerReqInput,
    ResumeSchedulerReqOutput,
)


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

    Covers every active lifecycle state (waiting/submitted, prefilling,
    decoding, retracted). Post-finish writeback states are async teardown and
    do not run forward work, so they do not block a drain.
    """
    return (
        scheduler.waiting_size() == 0
        and scheduler.decoding_size() == 0
        and scheduler.prefilling_size() == 0
        and scheduler.retract_count() == 0
    )


class PauseController:
    """Owns pause/resume state for one scheduler event loop.

    Split of responsibilities: the ``handle_*`` methods are driven by the
    request handler (they only touch local state + the reply socket); the event
    loop queries :pyattr:`admit_blocked` / :pyattr:`forward_blocked`, drains
    buffered specs on resume, and calls :pymeth:`maybe_finish_drain` once per
    iteration to resolve a deferred abort/wait reply.

    ``send_func`` is the scheduler→tokenizer reply socket (a no-op
    ``_NullSender`` on non-rank-0 TP ranks, matching the existing control-reply
    pattern), so the ``handle_*`` methods are safe to call on every rank.
    """

    def __init__(self, send_func) -> None:
        self._send = send_func
        self.state = PauseState.UNPAUSED
        # RequestSpecs withheld from the scheduler while paused; flushed on resume.
        self.buffered_specs: list = []
        # Deferred reply for abort/wait; held until the scheduler drains.
        self._pending_reply: PauseSchedulerReqOutput | None = None
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

    # -- control-request handlers (driven by the request handler) -------------

    def handle_pause(self, req: PauseSchedulerReqInput) -> None:
        if req.mode not in ("abort", "wait", "keep"):
            self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message=f"invalid pause mode: {req.mode!r}"
                )
            )
            return

        # Reject any new pause while an abort/wait pause is still draining: its
        # reply is a single-consumer promise (``_pending_reply``), so a second
        # pause would overwrite it and strand the first caller forever on its ZMQ
        # await — only one reply ever fires per pending pause. ``keep`` never sets
        # ``_pending_reply``, so it can't be the *first* pause here, but it must
        # not clobber a draining one either.
        if self._pending_reply is not None:
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
        self.state = PauseState.PAUSED_NEW
        self._pending_reply = PauseSchedulerReqOutput(success=True)
        self._cancel_grammar_pending = True
        if req.mode == "abort":
            self._abort_all_pending = True

    def handle_resume(self, req: ResumeSchedulerReqInput) -> None:
        # If a wait/abort pause is still awaiting its drain reply, it has NOT
        # drained — ``maybe_finish_drain`` clears ``_pending_reply`` the instant
        # it does. We must still reply (resume uses a separate communicator and
        # cannot otherwise wake the pause caller, who would block forever), but
        # the reply must be a failure: acking success here would tell a
        # weight-swapping caller it is safe to proceed while pre-pause requests
        # are still in flight under the old weights.
        if self._pending_reply is not None:
            self._send.send_pyobj(
                PauseSchedulerReqOutput(
                    success=False, message="resumed before pause drained"
                )
            )
            self._pending_reply = None
        # Buffered specs are flushed by the event loop on its next admission
        # pass (state is already UNPAUSED by then).
        self.state = PauseState.UNPAUSED
        self._abort_all_pending = False
        self._cancel_grammar_pending = False
        self._send.send_pyobj(ResumeSchedulerReqOutput(success=True))

    def handle_is_paused(self, req: IsSchedulerPausedReqInput) -> None:
        self._send.send_pyobj(IsSchedulerPausedReqOutput(is_paused=self.is_paused))

    # -- per-iteration drain check (driven by the event loop) -----------------

    def maybe_finish_drain(self, scheduler) -> None:
        """Resolve a deferred abort/wait reply once the scheduler has drained."""
        if self._pending_reply is None:
            return
        if not scheduler_drained(scheduler):
            return
        self._send.send_pyobj(self._pending_reply)
        self._pending_reply = None
