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

"""Turning one planned round into submitted GPU work, per engine role.

What a dispatch does depends only on the role this engine was started in,
which never changes at runtime: a plain engine forwards; a PD prefill node
also hands the KV off; a PD decode node first pulls the remote KV in. The
loop therefore picks its dispatcher once, at startup, instead of asking
``isinstance(kv_transfer, ...)`` on every round — and each role's rules
live in one contiguous place rather than interleaved in one long branch.

Every dispatcher returns the same pair:

- ``pending``: the handle for the submitted forward, or None for rounds
  that produce no model output (a KV handoff, an RDMA receive trigger).
- ``on_first_token``: a callback the commit path runs for the round's
  first sampled token, or None when the role does not need one.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from tokenspeed.runtime.execution.types import DpForwardMetadata, PendingExecution


@dataclass(frozen=True)
class PlannedForward:
    """One round's planned work, as the dispatcher needs to see it.

    Attributes:
        forward_op: The scheduler's op for this round.
        sampling_params_list: Per-slot sampling params, gathered by the loop.
        dp_metadata: CPU-gathered DP metadata, None outside DP.
        grammar_inputs: Per-batch grammar state, None when no request in the
            batch is constrained.
        multimodal_context: Per-batch multimodal state, None for text-only.
        cache_zero_future: The round's page-zeroing submission, None when
            no page needed sanitizing.
    """

    forward_op: Any
    sampling_params_list: list
    dp_metadata: DpForwardMetadata | None
    grammar_inputs: Any
    multimodal_context: Any
    cache_zero_future: Future | None


DispatchResult = tuple[PendingExecution | None, Callable | None]


class ForwardDispatcher:
    """Submit forwards for a plain (non-disaggregated) engine.

    The class attributes and predicates below are what the loop asks instead
    of re-deriving the role from ``kv_transfer``: the role decides them and
    knows them exactly once.
    """

    #: This engine hands its KV to a decode node after prefill.
    is_prefill_role = False
    #: This engine decodes requests whose prompt ran on another node.
    is_decode_role = False

    def __init__(self, executor) -> None:
        self._executor = executor

    def dispatch(self, planned: PlannedForward) -> DispatchResult:
        return self._submit(planned, grammar_inputs=planned.grammar_inputs), None

    def produces_model_output(self, forward_op) -> bool:
        """Whether dispatching this op enters the model forward path.

        DP ranks size their collectives from this, so it must answer for the
        same op the role would dispatch — hence the role answers, rather than
        the loop keeping a second copy of the rule.
        """
        return True

    def needs_pending_commit(self, forward_op) -> bool:
        """Whether this dispatch reads state only a pending commit produces.

        The loop drains its in-flight queue first when so; see
        ``EventLoop._dispatch_depends_on_pending_commit``, which folds this
        in with the role-independent rules.
        """
        return False

    def _submit(
        self,
        planned: PlannedForward,
        *,
        grammar_inputs,
        capture_next_input_ids: bool = False,
    ) -> PendingExecution:
        """Queue the forward on the data plane; never blocks."""
        executor = self._executor

        def _run_forward():
            return executor.execute_forward_op(
                planned.forward_op,
                planned.sampling_params_list,
                dp_metadata=planned.dp_metadata,
                grammar_inputs=grammar_inputs,
                multimodal_context=planned.multimodal_context,
                capture_next_input_ids=capture_next_input_ids,
            )

        return PendingExecution(executor.forward_thread.submit(_run_forward))


class DecodeDispatcher(ForwardDispatcher):
    """Submit forwards for a PD decode node.

    An extend op here is not prefill work: it means the request's KV still
    lives on the prefill node, so the round triggers an RDMA receive and
    produces no model output. The forward runs in a later round, once the
    transfer completes and the scheduler advances the request into decode.
    """

    is_decode_role = True

    def __init__(self, executor, kv_transfer, *, pd_cache_enabled: bool) -> None:
        super().__init__(executor)
        self._kv_transfer = kv_transfer
        self._pd_cache_enabled = pd_cache_enabled

    def produces_model_output(self, forward_op) -> bool:
        return not self._receives_remote_kv(forward_op)

    @staticmethod
    def _receives_remote_kv(forward_op) -> bool:
        """An extend op whose prompt ran on the prefill node."""
        return forward_op.num_extends() > 0 and not forward_op.is_local_prefill()

    def dispatch(self, planned: PlannedForward) -> DispatchResult:
        forward_op = planned.forward_op
        if self._receives_remote_kv(forward_op):
            self._trigger_remote_receive(planned)
            return None, None
        # Decode and local recovery-prefill batches execute normally. The
        # matcher of a constrained request was advanced past the prefill
        # node's token when its RemotePrefillDoneEvent landed, so masking
        # here continues from the right state.
        return self._submit(planned, grammar_inputs=planned.grammar_inputs), None

    def _trigger_remote_receive(self, planned: PlannedForward) -> None:
        """Pull the remote KV in for requests admitted this round.

        The slot preparation and cache-length reset touch the execution
        stream, so they run on the forward thread; the RDMA trigger itself
        is CPU-side but must follow them (and the zeroing barrier), so the
        whole path is submitted as one unit.
        """
        executor = self._executor
        kv_transfer = self._kv_transfer
        forward_op = planned.forward_op
        cache_zero_future = planned.cache_zero_future
        pd_cache_enabled = self._pd_cache_enabled

        def _trigger_receive():
            executor.prepare_remote_cache_slots(
                list(forward_op.request_pool_indices[: forward_op.num_extends()])
            )
            executor.reset_remote_prefill_cache_lengths(forward_op)
            if pd_cache_enabled and cache_zero_future is not None:
                # Page zeroing runs asynchronously on a CUDA stream, while
                # Mooncake/GPUDirect writes are not ordered by that stream.
                # Do not publish the destination manifest until the newly
                # assigned pages are fully sanitized.
                cache_zero_event = cache_zero_future.result()
                if cache_zero_event is not None:
                    cache_zero_event.synchronize()
            kv_transfer.execute(forward_op)

        executor.forward_thread.run(_trigger_receive)


class PrefillDispatcher(ForwardDispatcher):
    """Submit forwards for a PD prefill node.

    The overlap schedule is disabled for this role; under PP the in-flight
    depth is the chunk-pipeline depth instead. A round with no extend work
    means every chunk is done, so the KV goes to the decode side.
    """

    is_prefill_role = True

    def __init__(self, executor, kv_transfer, *, epd_hooks) -> None:
        super().__init__(executor)
        self._kv_transfer = kv_transfer
        self._epd_hooks = epd_hooks

    def needs_pending_commit(self, forward_op) -> bool:
        # The handoff batch needs the final chunk's bootstrap token, which
        # only lands at commit.
        return forward_op.num_extends() == 0

    def dispatch(self, planned: PlannedForward) -> DispatchResult:
        kv_transfer = self._kv_transfer
        if planned.forward_op.num_extends() == 0:
            kv_transfer.execute(planned.forward_op)
            return None, None

        # prepare_prefill is CPU-side transfer bookkeeping and must complete
        # before the forward's layerwise events are armed; the EPD assertion
        # is a pure-CPU invariant check. Both stay on the control plane; the
        # GPU work is submitted after.
        kv_transfer.prepare_prefill(planned.forward_op)
        # EPD invariant: handshaked items are filled by the async EPD
        # admission drain before admission; assert none reached the forward
        # un-received (no-op for non-EPD / text-only requests).
        self._epd_hooks.assert_embeddings_received(planned.multimodal_context)
        pending = self._submit(
            planned,
            grammar_inputs=planned.grammar_inputs,
            capture_next_input_ids=True,
        )
        return pending, kv_transfer.store_prefill_token
