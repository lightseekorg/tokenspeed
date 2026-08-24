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

"""EventLoop-side EPD prefill integration.

Split of responsibilities, mirroring pause.py's PauseController/PauseHooks:
``EpdPrefillAdmission`` (prefill_admission.py) DECIDES — it owns the receive
jobs, the rank-synced admission drain, and the optional NCCL row-shard
reassembly. ``EpdPrefillHooks`` here ACTS on those decisions with the event
loop's collaborators, so the loop's normal scheduling paths carry single-line
hooks only.
"""

from __future__ import annotations

import logging
import time

from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor

logger = logging.getLogger(__name__)


def _is_epd_request(state) -> bool:
    """True iff this request's images are encode-routed (smg injected per-image
    encode handshakes) -- it must wait for its embeddings (staged via the EPD
    admission controller, polled in drain_ready_embeddings) before being
    scheduled. Everything else admits immediately.
    """
    mm = getattr(state, "multimodal_inputs", None)
    return mm is not None and any(
        getattr(it, "encode_handshake", None) for it in mm.mm_items
    )


class EpdPrefillHooks:
    """EventLoop-side EPD prefill hooks.

    Holds a loop back-reference and the admission controller; the only state
    of its own is the staged request payloads awaiting their embeddings.
    Constructed with ``admission=None`` on non-EPD nodes (decode / encode /
    text-only), where every hook is a cheap no-op.
    """

    def __init__(self, loop, admission) -> None:
        self._loop = loop
        self._admission = admission
        # Staged EPD request payloads (request_id -> (spec, state, bootstrap)),
        # held here while the controller (rid-keyed, like kv_transfer) runs the
        # async receive; popped in drain_ready_embeddings on admit/abort.
        self._staged: dict = {}

    def try_stage(self, spec, state, bootstrap) -> bool:
        """Stage an encode-routed request OUT of the scheduler until its
        per-image embeddings have been received; return False for everything
        else (the caller registers the P->D sender and admits immediately).

        The P->D sender registration is DEFERRED to admission (in
        ``drain_ready_embeddings``, just before submit_requests). Registering
        it now -- while the request is staged and NOT yet in the C++ scheduler
        -- would let DisaggPrefillExecutor.generate_events poll the sender and
        emit a BootstrappedEvent that the scheduler's requests_.at(rid) THROWS
        on (no such request yet). Rank-identical because the caller's ready
        list is rank-synced (recv_reqs broadcast + grammar gather).
        """
        if self._admission is None or not _is_epd_request(state):
            return False
        self._admission.stage(spec.request_id, state.multimodal_inputs.mm_items)
        self._staged[spec.request_id] = (spec, state, bootstrap)
        return True

    def drain_ready_embeddings(self) -> None:
        """Admit EPD requests whose async embedding receives completed this cycle.

        The EpdPrefillAdmission controller DECIDES (poll + rank-lockstep MIN
        all-reduce + reassemble) and returns (admitted, failed); here we ACT on
        those decisions with the EventLoop's collaborators -- register/abort the
        P->D sender, submit admitted requests, finish failed ones. No-op (and no
        collective) on non-EPD nodes.
        """
        if self._admission is None:
            return
        loop = self._loop
        # Pause gate: withhold EPD admission while paused, mirroring the non-EPD
        # admit_blocked gate -- else the drain below would submit and RUN reassembled
        # specs during the pause. Staged receives wait in _pending until resume.
        # Rank-safe: admit_blocked is rank-identical, so all ranks skip together.
        if loop._pause.admit_blocked:
            return
        admitted_ids, failed_ids = self._admission.drain()
        for rid in failed_ids:
            spec, state, bootstrap = self._staged.pop(rid)
            # Signal the dual-dispatched decode that this request failed so its KV
            # receiver fails (FailedEvent -> PdTransferHooks abort)
            # instead of waiting forever for KV the prefill will never send. The
            # prefill never registered a P->D sender (deferred to admission), so the
            # decode has no other reliable way to learn (heartbeat only trips on a
            # dead prefill /health). Best-effort: only reaches decodes that already
            # pre-allocated.
            if (
                isinstance(loop.kv_transfer, DisaggPrefillExecutor)
                and bootstrap is not None
            ):
                try:
                    loop.kv_transfer.abort(rid, bootstrap)
                except Exception as exc:  # never let it wedge the loop
                    logger.warning(
                        "EPD abort->decode signal failed for rid=%s: %s",
                        rid,
                        exc,
                    )
            state.set_finish_with_abort("EPD embedding receive failed or timed out")
            loop.output_processor.publish_finished_at_admission(rid, state)
        admitted_specs = []
        for rid in admitted_ids:
            spec, state, bootstrap = self._staged.pop(rid)
            # Aborted mid-receive (no abort path, so drain still returns it admitted):
            # don't register the P->D sender or submit -- that runs a wasted forward
            # and leaks the sender. Stream its finish instead.
            if state.finished:
                loop.output_processor.publish_finished_at_admission(rid, state)
                continue
            # Register the P->D sender now (deferred from admission) -- the request
            # is about to enter the scheduler.
            if loop.kv_transfer is not None:
                loop.kv_transfer.register(rid, bootstrap)
            admitted_specs.append(spec)
        if admitted_specs:
            loop.scheduler.submit_requests(admitted_specs)
        elif self._admission.has_pending():
            # Nothing advanced this cycle but requests are still receiving; yield the
            # GIL so the Python daemon transfer/recv threads run (rank-consistent:
            # admitted/leftover are rank-identical here).
            time.sleep(0.0005)

    def assert_embeddings_received(self, multimodal_context) -> None:
        """EPD invariant: every handshaked item is filled with its embedding by the
        async EPD admission drain (EpdPrefillAdmission.drain) BEFORE admission, so by
        it is already encoded. This is a defensive check, not a receive: a handshaked
        item that reached the forward un-received leaked past async admission (the
        only EPD admission path) -- fail loud instead of running the tower or
        publishing shard-only rows. No-op for non-EPD / text-only requests.
        """
        if (
            self._admission is None
            or multimodal_context is None
            or not multimodal_context.has_extend_inputs()
        ):
            return
        for mm in multimodal_context.mm_inputs:
            if mm is None:
                continue
            missing = [
                i
                for i, item in enumerate(mm.mm_items)
                if getattr(item, "encode_handshake", None) is not None
                and item.encoded is None
            ]
            if missing:
                raise RuntimeError(
                    f"EPD: handshaked items {missing} reached the prefill forward "
                    "un-received; they must be admitted via the EPD admission drain"
                )
