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

"""The control plane's entire handle on the GPU.

``ModelExecutor`` owns the device: the execution stream, the runtime states,
the input buffers, the CUDA graphs, the attention backends and their forward
metadata. All of it must be touched by one thread — the forward thread — or
two threads race on one stream and a control-plane write lands in the middle
of a forward's own writes. Neither failure crashes; both surface much later
as a wrong token.

So the event loop does not hold a ``ModelExecutor``. It holds one of these,
and the operations below are the complete list of what it can do to the GPU.
(The PD transfer peer is the one object it also holds directly, for the
control-plane half of the transport — bootstrap, event polling; its
execution half is reached only through this handle.)
Each one packages its arguments into a closure and hands that closure to the
forward thread; there is no accessor here for the executor, a backend, a
buffer, or a stream. The rule "the control plane issues no CUDA work, and
what it hands over it does not touch again" therefore does not need to be
remembered or asserted — the loop cannot reach anything it could violate it
with. What it cannot see, it cannot change.

``build_device_side`` is why that holds at construction too. The model
runners, the attention backends and the KV pools are locals of that function
and the loop never names them, so it cannot keep one by accident. Startup
itself runs on the building thread, before the forward thread has any work,
which is why weight loading, autotuning and CUDA graph capture can touch the
device directly in there.

What comes back is one ``DeviceBuild``, split by how long the caller may
hold each piece:

- ``DeviceSpecs`` — plain values the control plane plans with (cache
  geometry, speculation widths, capability flags). No device object, safe to
  keep forever, and reading one never goes through the handle.
- ``transfer`` — the PD transfer peer's CONTROL face (bootstrap
  register/abort, event polling), or None outside PD. Built here too: its
  execution face lives inside the handle, and everything its construction
  needs is ``server_args`` or an object this builder already owns.
- ``encoder_model_facts`` — a startup-only callable for EPD admission,
  resolved past its gate because a text-only model has no vision tower.
- ``DeviceHandle`` — the running handle, and the only piece the loop stores.

Startup steps that need a real device object (the layerwise step counter,
the peer's KV description) happen INSIDE the builder, so no such object ever
comes back out. A new device interaction belongs in here if it runs once at
startup, and on ``DeviceHandle`` only if the running loop genuinely needs it
— the second widens what the loop can do to the GPU mid-flight.

See ``forward_thread.py`` for the capture contract each closure must satisfy.
"""

from __future__ import annotations

import contextlib
import enum
import os
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.platform import current_platform
from torch.utils._python_dispatch import TorchDispatchMode

from tokenspeed.runtime.cache.l2.sizing import (
    NO_POOL,
    RetractionPoolRequest,
    tail_lcm_blocks_per_request,
)
from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan
from tokenspeed.runtime.epd.recv_pool import recv_pool_bytes
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    PendingExecution,
    PlannedForward,
)
from tokenspeed.runtime.moe.expert_location import (
    ExpertLoadSnapshot,
    get_global_expert_location_metadata,
)
from tokenspeed.runtime.moe.expert_rebalance import ExpertRebalanceSpecs, SlotMoves
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.host_sync import allow_host_sync
from tokenspeed.runtime.utils.startup_timing import startup_phase

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.engine.scheduler_utils import SchedulerCacheGeometry
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheMemoryPlan,
    )
    from tokenspeed.runtime.layers.attention.registry import AttentionBuild
    from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(frozen=True)
class EncoderModelFacts:
    """The narrow model facts EPD admission needs, without the model.

    Attributes:
        device: The engine's device.
        hidden: Model hidden width.
        num_deepstack: Number of deepstack embeddings, 0 when unsupported.
        dtype: The vision tower's dtype.
    """

    device: Any
    hidden: int
    num_deepstack: int
    dtype: Any


@dataclass(frozen=True)
class DeviceSpecs:
    """What building the device side determined, as plain values.

    Facts, not operations. The control plane plans with these — scheduler
    capacity, cache groups, speculation widths, capability flags — and none
    of them is a handle to anything a forward touches, so they are safe to
    copy around and keep. They are returned ALONGSIDE the ``DeviceHandle``
    rather than read off it, so that the handle is only ever used to ask for
    work; a caller that just needs to know something never has to hold the
    thing that can do something.

    Attributes:
        cache_geometry: Page/token capacity and prefix granularity the C++
            scheduler is configured from.
        cache_groups: Per-group cache descriptors for the scheduler.
        cache_storage: Allocated-bytes report, republished in engine info.
        multimodal_encoder_dtype: The vision tower's dtype, or None.
        spec_num_steps: Draft steps per verify, 0 without speculation.
        spec_num_tokens: Verify width, 0 without speculation.
        uses_eager_grammar: Grammar masks are filled inline by the forward
            (rather than by the capturable side-stream executor), which is
            what makes a grammar batch depend on the pending commit.
        supports_disaggregation: The KV arena can hand pages to a peer node.
        supports_pd_layerwise_finalization: This rank can finalize layerwise
            KV writes with speculation on, required for PD layerwise
            transfer: the draft-field writer (producer or drafter) declares
            it, and a pipeline stage owning no draft fields has nothing to
            finalize.
        supports_prompt_logprobs: This engine can serve a request that asks
            for prompt (input) logprobs: either it never computes the prompt
            rows (the PD decode role, whose prefill node returns them) or its
            LM head scores every prompt row on this rank (no row narrowing,
            no pipeline split). The ingress refuses such requests otherwise.
        cache_state_group_ids: Group ids of the state-family cache groups,
            for the per-group page-usage debug line. Empty for pools with no
            recurrent/conv state.
        num_host_pages: The L2 host tier's page count (incl. the null page),
            sized here because it depends on the pools' transfer layout; 0
            without ``--enable-kvstore``. The scheduler is configured from it.
        num_snapshot_pages: The retraction snapshot pool's page count (incl.
            the null page), resolved like the L2 tier's size
            (``cache/l2/sizing.py``); 1 (the null page alone) without a
            pool, when nothing can be imaged and a capacity-blocked round
            aborts a resident instead.
        max_retracted_requests: Slot-state image rows of the snapshot pool,
            i.e. how many requests may be retracted at once; 0 without a
            pool.
        expert_rebalance: The expert placement's geometry and this rank's
            position in its EP group, for the online rebalance controller;
            None unless the server started with ``--enable-eplb``.
    """

    cache_geometry: Any
    cache_groups: Any
    cache_storage: Any
    multimodal_encoder_dtype: Any
    spec_num_steps: int
    spec_num_tokens: int
    uses_eager_grammar: bool
    supports_disaggregation: bool
    supports_pd_layerwise_finalization: bool
    supports_prompt_logprobs: bool
    cache_state_group_ids: tuple[str, ...]
    num_host_pages: int
    num_snapshot_pages: int
    max_retracted_requests: int
    expert_rebalance: ExpertRebalanceSpecs | None


def speculative_widths(
    spec_algo: str | None, spec_num_steps: int | None, spec_num_tokens: int | None
) -> tuple[int, int]:
    """The ``DeviceSpecs`` speculation widths: (draft steps, verify width).

    ``ServerArgs`` keeps its default widths with speculation off, so the
    algorithm, not the widths, says whether speculation is on. Both are 0
    without it: the accept-length log views verify rows by them.
    """
    if spec_algo is None:
        return 0, 0
    return spec_num_steps or 0, spec_num_tokens or 0


@dataclass(frozen=True)
class DeviceBuild:
    """What constructing the device side produces, split by how long the
    caller may hold each piece.

    Attributes:
        specs: Plain values the control plane plans with; safe to keep.
        transfer: The PD transfer peer's control face — bootstrap
            register/abort, event polling — or None outside PD.
        handle: The running handle; the loop's only reach into the device.
        encoder_model_facts: Zero-argument callable extracting the facts EPD
            admission needs (``EncoderModelFacts``). A callable, not a value:
            reading the vision tower's dtype raises on a text-only model, so
            it must only run after the EPD admission gate has decided the
            node is a multimodal prefill node. Startup-only, like the rest of
            this object.
    """

    specs: DeviceSpecs
    transfer: Any
    handle: "DeviceHandle"
    encoder_model_facts: Any


class DeviceRole(enum.Enum):
    """How this engine divides a scheduler plan between its two executors.

    Fixed at startup by the deployment mode: the values ARE
    ``server_args.disaggregation_mode``, and the split mirrors the C++
    scheduler's own (``config_.role``). A value rather than a class hierarchy: the
    three roles differ by about a dozen lines, and giving each a subclass
    forced the handle to publish its own internals so those subclasses could
    call back into it — a reference cycle for no gain.
    """

    #: No disaggregation: the model runs every batch, and the plan carries
    #: no remote streams.
    PLAIN = "null"
    #: Prefills prompts; a completed prompt's KV goes out on
    #: ``plan.remote_decode`` for the node that will decode it.
    PD_PREFILL = "prefill"
    #: Decodes prompts another node prefilled; ``plan.remote_prefill`` pulls
    #: an admitted prompt's KV in.
    PD_DECODE = "decode"


def _settle(submissions: deque, failure: str) -> None:
    """Drain finished data-plane submissions, re-raising the first failure.

    A submission's semantic completion arrives elsewhere (a transfer event,
    a cache-op ack), so a submission that RAISED produces no completion at
    all: swallowing it would strand its request silently. Only finished
    futures are inspected, so this never blocks.

    Args:
        submissions: The pending submission futures, oldest first.
        failure: What the caller loses if the failure is swallowed.

    Raises:
        RuntimeError: A settled submission raised on the data plane.
    """
    while submissions and submissions[0].done():
        exc = submissions.popleft().exception()
        if exc is not None:
            raise RuntimeError(failure) from exc


class DeviceHandle:
    """What the running control plane may ask of the GPU, and nothing else.

    Every method packages its arguments into a closure and hands that closure
    to the forward thread, and none of them returns anything the caller could
    then use to bypass the rest.
    """

    def __init__(
        self,
        executor,
        *,
        host_cache_executor=None,
        kv_transfer=None,
    ) -> None:
        # Private by convention AND by absence: nothing below returns it.
        self._executor = executor
        self._thread = executor.forward_thread
        # The compact Host cache -- the L2 prefix tier, the retraction
        # snapshot pool, or both -- or None with neither. Behind the handle
        # for the same reason the pools are: its submit path launches
        # transfers and records events. Snapshot stores ride ``execute``
        # with the write-backs and restores with the load-backs; all four
        # ACK kinds ride the cache poll.
        self._l2 = host_cache_executor
        # The PD transfer peer's execution face: its transfers move KV-pool
        # device memory over RDMA, so they need the same ordering against
        # forwards and page zeroing as everything else behind this handle.
        self._kv_transfer = kv_transfer
        self._role = _resolve_role(kv_transfer)
        # Cache-plan submission futures, drained by poll_cache_results: a
        # submission that raised must surface, or its ops stay counted
        # in flight forever and the cache-gated requests hang silently.
        # Appended and drained on the control-plane thread only.
        self._l2_submissions: deque = deque()
        # The transfer peer's submissions, settled at the next round's
        # execute (see ``_settle``).
        self._transfer_submissions: deque = deque()

    # ------------------------------------------------------------------
    # Per-round work
    # ------------------------------------------------------------------

    def execute(
        self,
        execution_plan,
        planned: "PlannedForward | None",
    ) -> PendingExecution | None:
        """Execute one scheduler plan; never blocks on the per-round path.

        The whole plan on the FIFO -- one thread, explicitly ordered streams -- in an order
        that IS the correctness argument for same-round page reuse: the
        Host-bound stores first (``WriteBackOp`` and ``SnapshotOp``: a
        retraction image's two legs -- the stream-ordered L2 write-back of
        the victim's hash-complete pages and the snapshot store of its tail
        pages and slot state -- read pages this plan re-grants, so they ride
        the write stream consecutively and ONE completion event fences the
        forward thread's default stream before anything else of the plan is
        enqueued; a pinned write-back -- an ordinary publication, whose
        sources the scheduler holds until the ACK -- rides the write stream
        and fences nothing), then page zeroing (the new owner's
        sanitization), then the L3 prefetches (``PrefetchOp``: a waiting
        request's Host pages filled from L3 on a CPU lane -- no stream, no
        Device page, acknowledged after the replica converges on the landed
        prefix), then the Device-bound copies (``LoadBackOp`` and
        ``RestoreOp``, ordered behind the zeroing: a load-back lands on
        zeroed pages, a restore overwrites its destinations whole), the
        transfer peer's remote streams, and finally the ``ForwardBatch``. The
        loop hands the round over and does not branch on it.

        A store's ``request_pool_index`` names the victim's slot, which the
        scheduler frees in the same plan build: the slot's next owner first
        writes it in a forward, and every forward of this plan is enqueued
        behind the store fence, so the slot-state export reads the victim's
        bytes. A restore's names the resumed request's new slot.

        Args:
            execution_plan: The round's plan, a per-round value copy out of
                C++. Its cache ops are adapted here, once, on the control
                plane (``cache_ops_from_plan``: the binding's batched snapshot
                ops become per-request ops) and read on the data plane.
            planned: The control plane's half of the round — the gathered
                per-batch state only it can produce (sampling params, live
                grammar matchers, the multimodal snapshot, DP metadata).
                None when no batch runs this round, which the plan alone
                cannot always say: an empty ``ForwardBatch`` could be read
                off it, but a DP-idle rank is the outcome of the loop's gloo
                collective. Either way the plan's page zeroing and cache
                transfers — retraction writebacks, load-back destinations —
                still run; they do not depend on a forward.

        Returns:
            The submitted forward's ``PendingExecution``, or None on rounds
            that run no batch — DP-idle, an empty plan, or a D-role round
            whose only work was the peer's.
        """
        # See ``_settle``: a submission that raised produces no transfer event.
        _settle(
            self._transfer_submissions,
            "PD transfer submission failed on the data plane; it produces no "
            "transfer event, so its request would hang silently",
        )

        executor = self._executor
        cache_ops = cache_ops_from_plan(execution_plan)
        l2 = self._l2 if cache_ops else None
        if l2 is not None:
            # Ahead of the zeroing: a stream-ordered store's sources may be
            # this very plan's pages_to_zero, and its fence lands on the
            # default stream the zeroing runs on, before the zeroing is
            # enqueued. The copies themselves order behind the execution
            # stream, where the forwards wrote the pages.
            self._l2_submissions.append(
                self._thread.submit(
                    lambda: l2.submit_write_backs(
                        cache_ops,
                        prerequisite_stream=executor.execution_stream,
                        fence_stream=executor.default_stream,
                    )
                )
            )
        pages = execution_plan.pages_to_zero
        zero_future = (
            self._thread.submit(lambda: executor.zero_cache_pages(pages))
            if pages
            else None
        )
        if l2 is not None:
            # The L3 prefetches: CPU work on the executor's own lane, started
            # here on the control plane -- they touch no stream and no Device
            # page, so they neither wait on the zeroing nor hold up the FIFO.
            l2.submit_prefetches(cache_ops)
            # Behind the zeroing: the loads' destinations were zeroed on the
            # default stream, so that is the prerequisite they order after.
            self._l2_submissions.append(
                self._thread.submit(
                    lambda: l2.submit_load_backs(
                        cache_ops, prerequisite_stream=executor.default_stream
                    )
                )
            )

        # The transfer peer's streams: prefills or decodes the peer NODE runs,
        # submitted asynchronously exactly like the model's ForwardBatch —
        # same FIFO, fire-and-forget, completion via the transfer events. They
        # are the plan's own work, so they go out even on a DP-idle round
        # that runs no batch.
        peer = self._kv_transfer
        remote_decode = execution_plan.remote_decode
        if remote_decode is not None:
            # P role: the prompt's decode happens on the peer, so its KV goes
            # out. The scheduler emits this only after the final chunk's
            # result landed, so the forwards that wrote the KV are already
            # ahead of it on the FIFO.
            self._transfer_submissions.append(
                self._thread.submit(lambda: peer.execute(remote_decode))
            )
        remote_prefill = execution_plan.remote_prefill
        if remote_prefill is not None:
            # D role: the prompt prefills on the peer; pull its KV into the
            # pages this plan admitted (and may be zeroing).
            self._transfer_submissions.append(
                self._submit_remote_prefill(remote_prefill, zero_future)
            )

        if planned is None:
            return None

        if self._role is DeviceRole.PD_PREFILL:
            # A prefill chunk. Arming its layerwise KV streaming must be
            # enqueued ahead of the forward it arms — same FIFO, first, and
            # submitted like everything else on the per-round path (waiting
            # here would drain the queue, collapsing the PP chunk pipeline
            # to lockstep on every chunk).
            forward_op = planned.forward_op
            self._transfer_submissions.append(
                self._thread.submit(lambda: peer.prepare_prefill(forward_op))
            )
            return self._submit_forward(planned, capture_next_input_ids=True)

        # Plain engine, or a D-role decode (the D role runs no local prefill:
        # a retracted request resumes by restore, not by recompute). A
        # constrained request's matcher was advanced past the prefill node's
        # token when its RemotePrefillDoneEvent landed, so masking continues
        # from the right state.
        return self._submit_forward(planned)

    @property
    def role(self) -> DeviceRole:
        """How this engine divides a plan; see ``DeviceRole``."""
        return self._role

    def _submit_forward(
        self,
        planned,
        *,
        capture_next_input_ids: bool = False,
    ) -> PendingExecution:
        """Queue this round's model forward; never blocks.

        ``planned`` is the single source of everything the round hands over
        (see ``PlannedForward``'s field-by-field capture notes).

        Args:
            planned: The round's ``PlannedForward``.
            capture_next_input_ids: Whether to keep the round's sampled rows
                (P role: the commit path folds them into the final chunk's
                ExtendResult as the bootstrap payload).

        Returns:
            A ``PendingExecution`` the loop resolves at commit.
        """
        executor = self._executor

        def _forward():
            return executor.execute_forward_op(
                planned.forward_op,
                planned.sampling_params_list,
                dp_metadata=planned.dp_metadata,
                grammar_inputs=planned.grammar_inputs,
                multimodal_context=planned.multimodal_context,
                capture_next_input_ids=capture_next_input_ids,
                ngram_inputs=planned.ngram_inputs,
                request_history_seeds=planned.request_history_seeds,
                input_logprob_plan=planned.input_logprob_plan,
            )

        return PendingExecution(self._thread.submit(_forward))

    def poll_cache_results(self) -> list:
        """Collect completed Host cache ops (both tiers); never blocks.

        Yields the scheduler's ACK kinds -- ``WriteBackDoneEvent``,
        ``LoadBackDoneEvent``, ``PrefetchDoneEvent`` (once the hooks
        converged the op), ``SnapshotDoneEvent``, ``RestoreDoneEvent`` --
        for ``CacheOpHooks`` to replica-intersect and the loop to advance.

        Stays on the control plane deliberately: completion is CUDA event
        queries plus queue drains (serialized against the data-plane submit
        by the executor's own lock). Routing it through the FIFO would park
        the round head behind every queued forward.

        Also settles finished submission futures first: a submission that
        raised produces no completion acks, so swallowing it would leave its
        ops counted in flight forever — the failure re-raises here, at the
        round head, with the data-plane traceback chained.

        Returns:
            The completed ops' scheduler events; empty when nothing finished.

        Raises:
            RuntimeError: A queued cache submission raised on the data plane.
        """
        l2 = self._l2
        if l2 is None:
            raise RuntimeError(
                "cache results polled without --enable-kvstore or a retraction "
                "snapshot pool"
            )
        _settle(
            self._l2_submissions,
            "L2 cache-plan submission failed on the data plane; its ops have "
            "no completion events and would hang their cache-gated requests",
        )
        return l2.poll_results()

    def consume_l3_backup_poll_failure(self) -> bool:
        """Whether an L3 backup future failed since the last consume.

        ``poll_cache_results`` must not raise that failure:
        ``CacheOpHooks.poll_ready_events`` still has to enter replica
        collectives. A rank-local raise hangs peers in those waits.
        """

        l2 = self._l2
        if l2 is None:
            return False
        consume = getattr(l2, "consume_backup_poll_failure", None)
        if consume is None:
            return False
        return bool(consume())

    def query_l3_storage(self, pages) -> list[bool] | None:
        """Probe immutable L3 objects without exposing the Host-cache tier."""

        l2 = self._l2
        if l2 is None:
            return None
        return l2.l3_exists(pages)

    def l3_prefetch_progress(self) -> dict[int, tuple[bool, int]]:
        """This rank's in-flight L3 prefetches: op id to ``(done, landed_pages)``.

        Read on the control plane each round by ``L3CacheHooks``, which
        MIN-reduces the vector across the replica; the op-id set is mirrored.
        """

        l2 = self._l2
        if l2 is None:
            return {}
        return l2.l3_prefetch_progress()

    def complete_l3_prefetch(self, op_id: int, landed_pages: int) -> None:
        """Record a prefetch op's replica-converged landed prefix.

        The next ``poll_cache_results`` yields its one ``PrefetchDoneEvent``.
        """

        l2 = self._l2
        if l2 is None:
            raise RuntimeError("L3 prefetch completed without a Host cache executor")
        l2.complete_l3_prefetch(op_id, landed_pages)

    def delete_l3_namespace(self) -> bool:
        """Delete L3 objects under the current prefix. Device/Host stay intact.

        Returns True when L2/L3 is unset or the store reports the prefix
        is gone. Call this before ``ClearCache``; a False must leave
        every rank's Device/Host indexes untouched. In-flight prefetches
        are drained first: the lane may still be writing their Host pages.
        """

        if self._l2 is None:
            return True
        return self._l2.delete_l3_namespace()

    def set_l3_weight_version(self, weight_version: str) -> None:
        """Publish subsequent Host pages under the new checkpoint identity."""

        if self._l2 is not None:
            self._l2.set_l3_weight_version(weight_version)

    def shutdown_cache(self) -> None:
        """Join queued cache submissions, then close the Host cache (L2/L3, pool)."""

        if self._l2 is None:
            return
        errors = []
        while self._l2_submissions:
            future = self._l2_submissions.popleft()
            try:
                future.result()
            except BaseException as exc:  # noqa: BLE001 — surface after close
                errors.append(exc)
        self._thread.run(self._l2.shutdown)
        if errors:
            raise errors[0]

    def run_idle_forward(self, dp_metadata: DpForwardMetadata) -> None:
        """Run a zero-token forward so this DP rank joins the round's collectives.

        Args:
            dp_metadata: The round's CPU-gathered DP metadata.
        """
        executor = self._executor
        self._thread.run(lambda: executor.execute_idle_forward(dp_metadata))

    def _submit_remote_prefill(self, remote_prefill, zero_future):
        """Queue a remote-prefill stream: pull its rows' KV from the peer.

        Slot preparation and the cache-length reset touch the execution
        stream, each behind the default stream (the store fence: the slot
        may be a victim's this plan imaged, and its slot-state export reads
        it on the write stream); the RDMA trigger is CPU-issued but writes
        the same device pages, so it must follow them and the zeroing fence
        (Mooncake and GPUDirect writes are not ordered by the zeroing
        stream, so the destination pages must be published from sanitized
        memory). The same fence covers a retraction's stream-ordered
        write-back reading pages this admission was granted: the zero event
        is recorded on the forward thread's stream AFTER that stream waited
        on the write-back's completion, so waiting on it waits on the copy
        too. One ordered
        unit, so one submission — asynchronous like every other: completion
        arrives through the transfer events, and a submission failure
        surfaces from the settle at the next round's execute.

        Args:
            remote_prefill: The plan's remote-prefill stream; supplies the
                admitted rows.
            zero_future: This plan's page-zeroing submission, or None when
                the plan zeroes nothing (an admission always zeroes its fresh
                pages, so a granted request's receive never sees None).

        Returns:
            The submission future, for the settle queue.
        """
        executor = self._executor
        kv_transfer = self._kv_transfer
        num_extends = remote_prefill.num_extends()

        def _receive():
            executor.prepare_remote_cache_slots(
                list(remote_prefill.request_pool_indices[:num_extends])
            )
            executor.reset_remote_prefill_cache_lengths(remote_prefill)
            if zero_future is not None:
                zero_event = zero_future.result()
                if zero_event is not None:
                    zero_event.synchronize()
            kv_transfer.execute(remote_prefill)

        return self._thread.submit(_receive)

    def run_multimodal_work(self, work: Callable[[], Any], *, wait: bool = True) -> Any:
        """Run multimodal feature-lifecycle work on the data plane, in order.

        The one generic slot on this handle: the multimodal feature lifecycle
        is a state machine reached from several control-plane points (EPD
        admission's device half, the commit-side SHM release), and naming
        each step would pull those state machines' internals in here. A
        second KIND of user gets its own named method instead.

        Args:
            work: Zero-argument callable performing the device step.
            wait: Block for the result (EPD admission) or fire-and-forget
                (the commit-side release).

        Returns:
            ``work``'s return value when waiting, else None.
        """
        if wait:
            return self._thread.run(work)

        def _logged():
            try:
                work()
            except Exception:
                # Nobody observes this future; swallowing would turn an SHM
                # release failure into a silent leak. Log and keep the data
                # plane running.
                logger.exception("fire-and-forget multimodal work failed")

        self._thread.submit(_logged)
        return None

    def run_kv_repair(self) -> None:
        """Zero the KV arena after a memory-saver wake re-maps its storage.

        Re-mapped memory holds garbage. The draft pool names the same arena,
        so clearing the target's clears both — walking two pools only cleared
        it twice. FP8 KV scales ride with the weights region and need no
        reset.
        """
        pool = self._executor.token_to_kv_pool
        self._thread.run(pool.clear_kv_buffers)

    def run_remote_prefill_landing(
        self,
        candidate_info: tuple[int, list[int]] | None,
        remote_cache_slot: int | None,
    ) -> None:
        """Apply a completed remote prefill's device-side effects, in order.

        The candidate ids must precede the readiness arm: hydration reads the
        row the candidates were just written into.

        Args:
            candidate_info: ``(req_pool_idx, candidate_ids)`` when the prefill
                node shipped speculative candidates, else None.
            remote_cache_slot: The slot to arm for first-decode hydration, or
                None when the request no longer needs one.
        """
        if candidate_info is None and remote_cache_slot is None:
            return
        executor = self._executor

        def _land():
            if candidate_info is not None:
                req_pool_idx, candidate_ids = candidate_info
                executor.write_remote_spec_candidate_ids(req_pool_idx, candidate_ids)
            if remote_cache_slot is not None:
                executor.mark_remote_cache_ready(remote_cache_slot)

        # ``run``, not ``submit``: a failure here corrupts the request's first
        # decode, and PD completions are rare enough to afford the wait.
        self._thread.run(_land)

    def reset_expert_load(self) -> None:
        """Zero the router's expert load counters, ordered against forwards.

        The counters are bumped on the execution stream by every forward's
        routing, so the reset rides that stream too: it lands after the
        forwards already issued and before the next one.
        """
        placement = _recording_expert_placement()
        executor = self._executor

        def _reset():
            with executor.device_module.stream(executor.execution_stream):
                placement.reset_load()

        self._thread.run(_reset)

    def dump_expert_load(self, path: str) -> dict[str, torch.Tensor | int]:
        """Write this rank's expert load counted since the last reset to ``path``.

        The counters are read back on the data plane (the one deliberate host
        wait) and saved unreduced, as this rank's record: its physical and
        logical counts, the placement that produced them and its EP rank.
        No collective runs here -- a profile stop reaches attention-DP
        workers independently, so a rank reducing inside the request would
        wait on a peer still synchronizing its round. The ranks' records are
        summed where they are consumed (``merge_expert_load_records``, which
        ``--init-expert-location <dir>`` applies).

        Args:
            path: Destination ``.pt`` file; parent directories are created.

        Returns:
            The saved record.
        """
        placement = _recording_expert_placement()
        executor = self._executor

        def _dump():
            with executor.device_module.stream(executor.execution_stream):
                with allow_host_sync("expert load dump"):
                    physical = placement.physical_load.cpu()
            record = placement.load_record(physical)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(record, path)
            return record

        return self._thread.run(_dump)

    def snapshot_expert_load(self) -> ExpertLoadSnapshot:
        """Read this rank's expert load since the previous snapshot and start a new window.

        The online rebalance's recording window: the counters are copied to
        the host (the one deliberate host wait) and zeroed on the execution
        stream, behind the in-flight forward and ahead of the next one, so the
        boundary is exact at forward granularity. Rank-local like
        ``dump_expert_load``; the caller reduces the result over the EP group
        where that is needed (inside the same-round gate, never here).

        Returns:
            The host counters with the placement they were counted under.
        """
        placement = _recording_expert_placement()
        executor = self._executor

        def _snapshot():
            with executor.device_module.stream(executor.execution_stream):
                with allow_host_sync("expert load snapshot"):
                    return placement.snapshot_load()

        return self._thread.run(_snapshot)

    def apply_expert_placement(
        self,
        layer_ids: Sequence[int],
        new_rows: torch.Tensor,
        moves_by_layer: Mapping[int, SlotMoves],
    ) -> None:
        """Move one chunk of layers' expert weights and switch their routing tables.

        Runs on the forward thread inside the execution stream: the P2P over
        the EP group and the slot copies land behind the in-flight forward,
        each layer's tables switch right after its slots, and the next forward
        is enqueued after the whole chunk. The stream is synchronized at the
        end -- a deliberate, low-rate host wait like ``update_weights`` -- so a
        P2P failure surfaces in this call rather than in a later forward.

        Args:
            layer_ids: The chunk's layers.
            new_rows: ``[len(layer_ids), num_physical]`` host rows of the
                committed placement.
            moves_by_layer: This rank's ``SlotMoves`` per layer.

        Raises:
            RuntimeError: The server did not start with ``--enable-eplb``.
        """
        updater = self._executor.model_runner.expert_location_updater
        if updater is None:
            raise RuntimeError(
                "expert placement updates need the server to start with --enable-eplb"
            )
        executor = self._executor

        def _apply():
            with executor.device_module.stream(executor.execution_stream):
                updater.apply(layer_ids, new_rows, moves_by_layer)
                with allow_host_sync("expert placement chunk"):
                    executor.execution_stream.synchronize()

        self._thread.run(_apply)

    def update_weights(self, req) -> tuple[bool, str]:
        """Apply one in-place RL weight-sync request, ordered against forwards.

        Type-dispatched on the request — join the trainer's NCCL group,
        receive and apply one broadcast, read one committed version from the
        Mooncake weight store, or tear the group down. One entry point because
        it is one capability: rewriting model parameters in place, which must
        be ordered against forwards rather than raced with them.

        Args:
            req: An ``io_struct`` weight-update request.

        Returns:
            ``(ok, message)`` from the model runner.

        Raises:
            TypeError: Not a weight-update request.
        """
        from tokenspeed.runtime.engine.io_struct import (
            DestroyWeightsUpdateGroupReqInput,
            InitWeightsUpdateGroupReqInput,
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromMooncakeReqInput,
        )

        runner = self._executor.model_runner
        handlers = {
            InitWeightsUpdateGroupReqInput: runner.init_weights_update_group,
            UpdateWeightsFromDistributedReqInput: (
                runner.update_weights_from_distributed
            ),
            UpdateWeightsFromMooncakeReqInput: (
                lambda req: runner.update_weights_from_mooncake(
                    req.version, self._mooncake_update_models()
                )
            ),
            DestroyWeightsUpdateGroupReqInput: runner.destroy_weights_update_group,
        }
        handler = handlers.get(type(req))
        if handler is None:
            raise TypeError(f"unsupported weight-update request {type(req).__name__}")
        loads_weights = type(req) in (
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromMooncakeReqInput,
        )

        def _apply_update():
            result = handler(req)
            if loads_weights and result[0] and self._executor.drafter is not None:
                self._executor.drafter.on_target_weights_updated()
            return result

        return self._thread.run(_apply_update)

    def _mooncake_update_models(self) -> list:
        """The modules a Mooncake update streams into, target first.

        ``--model-update-draft-weights refresh`` adds the speculative draft
        model when one is loaded; ``retain`` keeps the draft's weights.
        """
        runner = self._executor.model_runner
        models = [runner.model]
        draft_runner = self._executor.draft_model_runner
        if (
            runner.server_args.model_update_draft_weights == "refresh"
            and draft_runner is not None
        ):
            models.append(draft_runner.model)
        return models


def _recording_expert_placement():
    placement = get_global_expert_location_metadata()
    if placement is None or placement.physical_load is None:
        raise RuntimeError(
            "expert load is not being recorded; start the server with "
            "--expert-distribution-recorder-mode stat"
        )
    return placement


def start_expert_load_window(executor) -> None:
    """Zero the expert load counters once the startup forwards are done.

    Autotune, warm-up and graph capture route tokens through the MoE layers,
    and none of that is traffic: the first window a rebalance
    (``--enable-eplb``) or an ``EXPERT_LOAD`` profile sees must start at the
    first served forward. The device is drained first so no startup kernel
    is still bumping the counters, then the reset rides the execution stream
    like the serving-time resets. A no-op without load recording.
    """
    placement = get_global_expert_location_metadata()
    if placement is None or placement.physical_load is None:
        return
    executor.device_module.synchronize()
    with executor.device_module.stream(executor.execution_stream):
        placement.reset_load()


def arm_data_plane_sync_debug(device: str) -> None:
    """Arm torch's sync-debug mode for the serving phase when asked to.

    Startup synchronizes on purpose -- weight loading, tuning, graph capture,
    the PD transfer and L2 executor builders, the EPD admission's NCCL
    warm-up in ``EventLoop.__init__`` -- so the loop arms the mode as its
    very last step before entering the round loop. It is process-wide:
    the control plane's ``copy_event.synchronize()`` and non-blocking D2H
    copies are not flagged, so what it reports is exactly the host
    synchronization that serializes the data plane against the step in
    flight -- ``.cpu()``, ``.item()``, ``.tolist()``, ``bool(tensor)``,
    ``nonzero``, pageable host<->device copies, ``stream.synchronize()``.
    """
    mode = envs.TOKENSPEED_DATA_PLANE_SYNC_DEBUG.get()
    if mode == "default":
        return
    if mode not in ("warn", "error"):
        raise ValueError(
            f"TOKENSPEED_DATA_PLANE_SYNC_DEBUG must be default, warn or error, got {mode!r}"
        )
    if torch.device(device).type != "cuda":
        logger.warning(
            f"TOKENSPEED_DATA_PLANE_SYNC_DEBUG={mode!s} ignored on {device!s}: "
            "sync-debug mode is a CUDA facility"
        )
        return
    torch.cuda.set_sync_debug_mode(mode)
    logger.info(
        f"Data-plane sync debug armed ({mode!s}): host synchronizations on the "
        "serving path are reported"
    )


def _cudagraph_probe_refusal(server_args: ServerArgs) -> str | None:
    """Why this boot cannot use a probe, or None when it can.

    ``enforce_eager`` captures nothing to measure.
    """
    if server_args.disable_cudagraph_memory_reserve:
        return "--disable-cudagraph-memory-reserve is set"
    if server_args.enforce_eager:
        return "--enforce-eager captures no graphs"
    return None


@dataclass(frozen=True)
class PoolViews:
    """What the boot derives from one bound pool."""

    token_to_kv_pool: CachePool
    draft_token_to_kv_pool: CachePool | None
    cache_geometry: SchedulerCacheGeometry
    cache_groups: list


def pool_views(attention: AttentionBuild) -> PoolViews:
    """Derive all four together, from the pool the caller was handed.

    A rebind that refreshes the pools but leaves the scheduler geometry on the
    probe arena admits against a handful of blocks while the pool holds
    thousands, and fails at serving rather than at boot. Named fields rather
    than a tuple: positional unpacking makes transposing target and draft, or
    binding a geometry to the wrong name, invisible at the call site.
    """
    from tokenspeed.runtime.engine.scheduler_utils import (
        pool_to_cache_groups,
        scheduler_cache_geometry_from_pool,
    )

    pool = attention.token_to_kv_pool
    return PoolViews(
        token_to_kv_pool=pool,
        draft_token_to_kv_pool=attention.draft_token_to_kv_pool,
        cache_geometry=scheduler_cache_geometry_from_pool(pool),
        cache_groups=pool_to_cache_groups(pool),
    )


def _rebind_under_reserve(
    executor: ModelExecutor,
    build_components: Callable[..., AttentionBuild],
    server_args: ServerArgs,
    gpu_id: int,
    probe: AttentionBuild,
    requested_backends: tuple[str | None, str | None],
    *,
    startup_resident_bytes: int,
) -> tuple[AttentionBuild, PoolViews]:
    """Rebuild the real pool under the probe's reserve, and its views with it.

    Returned together so no caller can pair the rebuilt pool with views still
    derived from the probe arena. The probe build wrote its backend resolution
    into ``server_args``; the rebuild resolves again from the operator's
    ``requested_backends``, so both builds pick the same backend and geometry.
    """
    from tokenspeed.runtime.execution.cudagraph_memory import reserve_and_rebind

    server_args.attention_backend, server_args.drafter_attention_backend = (
        requested_backends
    )
    attention = reserve_and_rebind(
        executor,
        build_components,
        server_args,
        gpu_id,
        profiled_cache_bytes=probe.profiled_cache_bytes,
        startup_resident_bytes=startup_resident_bytes,
    )
    return attention, pool_views(attention)


def probe_arena_floor(
    server_args: ServerArgs,
    model_config: ModelConfig,
    max_forward_tokens: int,
    *,
    decoder_rows_per_request: int | None,
) -> int:
    """Parent blocks a probe arena needs for the widest row the boot fabricates.

    Both terms track a knob: the ladder is built from the resolved prefill-graph
    ceiling, which is a default rather than zero when unset, and a configured
    capture batch size fabricates that many rows whatever the bucket. Returned
    rather than inlined so a test can assert the number instead of the source.
    A narrowing model's decoder ladder (``decoder_rows_per_request`` is its
    ``max_decoder_rows_per_request``, None for other models) fabricates one
    request per that many rows of a bucket no wider than the token ladder's,
    and never more than ``max_num_seqs``.
    """
    from tokenspeed.runtime.execution.cudagraph_memory import probe_arena_parent_blocks
    from tokenspeed.runtime.execution.model_executor import (
        _resolve_prefill_graph_max_tokens,
    )

    tokens = max(max_forward_tokens, _resolve_prefill_graph_max_tokens(server_args))
    return probe_arena_parent_blocks(
        max_forward_tokens=tokens,
        context_len=model_config.context_len,
        capture_batch_sizes=server_args.prefill_graph_capture_batch_sizes,
        decoder_requests=(
            0
            if decoder_rows_per_request is None
            else min(-(-tokens // decoder_rows_per_request), server_args.max_num_seqs)
        ),
    )


def _retraction_pool_request(
    server_args,
    contract,
    *,
    max_batch_size: int,
    max_context_len: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
) -> RetractionPoolRequest:
    """The snapshot pool's knobs plus the tail unit a derived size counts in.

    The tail unit comes from the scheduler's own ``CapacityModel`` over the
    bound contract's groups, under the same limits the pool was sized for, so
    an image's tail is counted the way the scheduler will later claim it.
    Skipped (0) when the knobs settle the size without it.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
        SchedulerLimits,
        capacity_model,
        scheduler_role,
    )

    tail_lcm_blocks = 0
    needs_tail = (
        server_args.enable_kvstore
        and server_args.retraction_snapshot_host_gb == 0
        and server_args.retraction_snapshot_ratio is None
    )
    if needs_tail:
        model = capacity_model(
            contract.group_specs,
            prefix_granularity=contract.prefix_granularity,
            virtual_packing=contract.virtual_packing,
            limits=SchedulerLimits(
                role=scheduler_role(server_args.disaggregation_mode),
                max_live_requests=max_batch_size,
                max_scheduled_tokens=int(server_args.chunked_prefill_size),
                max_context_len=max_context_len,
                decode_input_tokens=decode_input_tokens,
                overlap_schedule_depth=overlap_schedule_depth,
                disable_prefix_cache=not server_args.enable_prefix_caching,
            ),
        )
        tail_lcm_blocks = tail_lcm_blocks_per_request(
            model, contract.group_specs, max_context_len
        )
    return RetractionPoolRequest(
        host_gb=server_args.retraction_snapshot_host_gb,
        ratio=server_args.retraction_snapshot_ratio,
        max_retracted_requests=server_args.retraction_snapshot_max_requests,
        tail_lcm_blocks_per_request=tail_lcm_blocks,
    )


def build_device_side(
    *,
    server_args,
    model_config,
    draft_model_config,
    gpu_id: int,
    global_rank: int,
    attn_tp_rank: int,
    min_per_gpu_mem,
    overlap_schedule_depth: int,
    decode_input_tokens: int,
    max_batch_size: int,
) -> DeviceBuild:
    """Construct the whole device side and return the three views of it.

    The model runners, attention backends, KV pools and executor are locals
    of this function and never leave it. What comes back is split by how long
    the caller may hold it: specs it may keep forever, the transfer peer's
    control face and the encoder-facts callable it consumes at startup, and
    the handle it runs with.

    The chain is linear and the order is load-bearing: the multimodal runtime
    and persistent communication buffers must be prepared after weights are
    loaded and before ``create_attn_components`` profiles memory for the KV
    budget (the EPD receive pool, which the admission allocates later, is left
    out of that budget instead), and the chunked-prefill limit must be aligned
    to the cache groups before ``ModelExecutorConfig`` sizes the input buffers
    from it.

    Args:
        server_args: Parsed server arguments. ``chunked_prefill_size`` may
            be lowered here to the cache-group checkpoint grain.
        model_config: The target model's config.
        draft_model_config: The draft model's config, or None.
        gpu_id: Local device index.
        global_rank: This worker's global rank.
        attn_tp_rank: Attention-TP rank; only rank 0 logs the memory
            breakdown.
        min_per_gpu_mem: Free-memory floor from distributed init, used to
            size the KV budget.
        overlap_schedule_depth: Decode KV reservation depth.
        decode_input_tokens: Tokens each decode step feeds per request.
        max_batch_size: Rank-local scheduler batch bound; the executor
            sizes its request pool one row past it for the graph-padding
            sink row.

    Returns:
        A ``DeviceBuild``. No device object escapes except inside the
        handle; the specs hold none at all.
    """
    # Imported here: these pull in the model/backend registries, which
    # import execution types — a module-level import would cycle.
    from tokenspeed.runtime.engine.scheduler_utils import (
        aligned_max_scheduled_tokens,
        log_gpu_memory_summary,
    )
    from tokenspeed.runtime.execution.factory import (
        ModelExecutorConfig,
        _eagle_aux_layer_ids,
        create_model_executor,
        create_model_runner,
    )
    from tokenspeed.runtime.execution.memory_delta import (
        NULL_MEMORY_DELTA_OBSERVER,
        DriverMemoryDeltaObserver,
    )
    from tokenspeed.runtime.execution.prefill_graph import narrowing_prefill_model
    from tokenspeed.runtime.layers.attention.registry import (
        create_attn_components,
    )
    from tokenspeed.runtime.utils import set_random_seed

    target, draft = create_model_runner(
        server_args, model_config, draft_model_config, gpu_id, global_rank
    )
    with startup_phase("multimodal.init"):
        if server_args.disaggregation_mode in ("null", "prefill"):
            target.prepare_multimodal_runtime()
    max_forward_tokens = (
        server_args.chunked_prefill_size
        if server_args.chunked_prefill_size > 0
        else server_args.max_prefill_tokens + server_args.max_model_len
    )
    max_forward_tokens = max(
        max_forward_tokens,
        max_batch_size * decode_input_tokens,
    )
    with startup_phase("communication.prepare"):
        target.prepare_communication_runtime(max_forward_tokens)
        if draft is not None:
            draft.prepare_communication_runtime(max_forward_tokens)

    post_profile_bytes = recv_pool_bytes(server_args, model_config.is_multimodal_active)
    if post_profile_bytes:
        logger.info(
            f"KV cache budget leaves out the {post_profile_bytes / (1 << 30):.2f} GiB "
            "EPD receive pool (TOKENSPEED_EPD_RECV_POOL_SLOTS x "
            "TOKENSPEED_EPD_RECV_POOL_SLOT_MB)"
        )

    @startup_phase("kv.build")
    def build_components(
        *,
        graph_reserve_bytes: int,
        probe_batch_rows: int | None,
        profiled_cache_bytes: int | None,
        reuse_target_backend: AttentionBackend | None,
        reuse_draft_backend: AttentionBackend | None,
    ) -> AttentionBuild:
        return create_attn_components(
            server_args,
            model_config,
            gpu_id,
            global_rank,
            min_per_gpu_mem,
            server_args.enable_memory_saver,
            draft_model_config,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
            graph_reserve_bytes=graph_reserve_bytes,
            post_profile_bytes=post_profile_bytes,
            probe_batch_rows=probe_batch_rows,
            profiled_cache_bytes=profiled_cache_bytes,
            reuse_target_backend=reuse_target_backend,
            reuse_draft_backend=reuse_draft_backend,
        )

    requested_backends = (
        server_args.attention_backend,
        server_args.drafter_attention_backend,
    )
    refusal = _cudagraph_probe_refusal(server_args)
    if refusal is not None:
        logger.info(
            f"CUDA-graph memory reserve off, startup residue unreserved: {refusal}"
        )
    probing = refusal is None
    narrowing = narrowing_prefill_model(target.model)
    attention = build_components(
        graph_reserve_bytes=0,
        probe_batch_rows=(
            probe_arena_floor(
                server_args,
                model_config,
                max_forward_tokens,
                decoder_rows_per_request=(
                    None
                    if narrowing is None
                    else narrowing.max_decoder_rows_per_request
                ),
            )
            if probing
            else None
        ),
        profiled_cache_bytes=None,
        reuse_target_backend=None,
        reuse_draft_backend=None,
    )
    views = pool_views(attention)
    # Lowering the limit is safe; a configured chunk smaller than one
    # state checkpoint block is rejected by aligned_max_scheduled_tokens
    # instead of silently increasing a frozen buffer limit.
    if server_args.enable_prefix_caching:
        aligned = aligned_max_scheduled_tokens(
            server_args.chunked_prefill_size, views.cache_groups
        )
        if aligned != server_args.chunked_prefill_size:
            logger.warning(
                f"chunked_prefill_size={server_args.chunked_prefill_size!s} is not a "
                "multiple of the "
                f"state-snapshot checkpoint grain; using {aligned!s} so recurrent-state"
                " "
                "pages can register for prefix-cache reuse.",
            )
            server_args.chunked_prefill_size = aligned

    # On CUDA, what startup keeps resident after the probe build joins the reserve.
    startup_memory = (
        DriverMemoryDeltaObserver(torch.get_device_module(server_args.device), gpu_id)
        if probing and current_platform().is_nvidia
        else NULL_MEMORY_DELTA_OBSERVER
    )
    with startup_memory.measure("startup"), startup_phase("executor.init"):
        executor = create_model_executor(
            server_args=server_args,
            config=ModelExecutorConfig.from_server_args(
                server_args=server_args,
                model_config=model_config,
                max_req_pool_size=max_batch_size + 1,
                gpu_id=gpu_id,
                global_rank=global_rank,
                prefix_granularity=views.cache_geometry.prefix_granularity,
                overlap_schedule_depth=overlap_schedule_depth,
            ),
            model_runner=target,
            draft_model_runner=draft,
            attn_backend=attention.attn_backend,
            token_to_kv_pool=views.token_to_kv_pool,
            draft_attn_backend=attention.draft_attn_backend,
            draft_token_to_kv_pool=views.draft_token_to_kv_pool,
        )
    # Once per process, before the probe: a graph keeps its capture-time tactic.
    with (
        startup_memory.measure("startup"),
        startup_phase("kernels.autotune"),
        current_platform().restore_stack_limit(),
    ):
        executor.autotune()
    if probing:
        # Consumers above keep the probe's: they read only block-count-invariant fields.
        with startup_phase("graph.probe_rebind"):
            attention, views = _rebind_under_reserve(
                executor,
                build_components,
                server_args,
                gpu_id,
                attention,
                requested_backends,
                startup_resident_bytes=sum(startup_memory.samples.get("startup", ())),
            )

    with startup_phase("graph.capture"):
        executor.capture_graphs(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    # Tuning and capture draw from the generator; this is the state startup leaves.
    set_random_seed(48)
    start_expert_load_window(executor)

    # Per-rank GPU memory breakdown (weights by group, KV/graph/non-torch).
    if attn_tp_rank == 0:
        log_gpu_memory_summary(
            target.model,
            gpu_id,
            global_rank,
            logger,
            device=server_args.device,
            draft_model=draft.model if draft is not None else None,
            kv_pool=views.token_to_kv_pool,
            draft_kv_pool=views.draft_token_to_kv_pool,
        )

    # The compact Host cache: the L2 prefix tier (--enable-kvstore), the
    # retraction snapshot pool (on by default on the roles that retract), or
    # both in one executor over the same field geometry. The executor is the
    # model executor's peer: it lays out the slot-state exporters the model
    # executor lists, so it is built after it.
    host_cache_executor = None
    snapshot_pool = _retraction_pool_request(
        server_args,
        views.token_to_kv_pool.arena.runtime_contract,
        max_batch_size=max_batch_size,
        max_context_len=model_config.context_len + server_args.spec_context_pad,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    if server_args.enable_kvstore or not snapshot_pool.disabled:
        from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor

        host_cache_executor = HostCacheExecutor(
            views.token_to_kv_pool,
            draft_pool=views.draft_token_to_kv_pool,
            l2_tier=server_args.enable_kvstore,
            host_ratio=server_args.kvstore_ratio,
            host_size_gb=server_args.kvstore_size,
            snapshot_pool=snapshot_pool,
            slot_state_exporters=executor.slot_state_exporters(),
            io_backend=server_args.kvstore_io_backend,
            attn_tp_rank=attn_tp_rank,
            kvp_rank=server_args.mapping.attn.kvp_rank,
        )
    elif attn_tp_rank == 0:
        # With an executor the line comes from its resolved sizing.
        logger.info(
            NO_POOL.describe(host_lcm_block_bytes=0, blob_bytes=0, l2_tier=False)
        )
    if server_args.enable_kvstore and server_args.kvstore_storage_backend is not None:
        # L3 under the L2 tier.
        from tokenspeed.runtime.cache.l3.backend import (
            L3_RUNTIME_COMPAT,
            cache_layout_signature,
            l3_cache_quantization_id,
            l3_checkpoint_id,
            share_l3_checkpoint_ids,
            storage_key_prefix,
        )
        from tokenspeed.runtime.cache.l3.factory import (
            create_kvstore_storage_backend,
        )

        storage_backend = create_kvstore_storage_backend(
            server_args.kvstore_storage_backend,
            server_args.kvstore_storage_backend_extra_config,
            host_buffer=host_cache_executor.host_storage.host_buffer,
            tp_size=server_args.mapping.attn.tp_size,
            pp_size=(server_args.mapping.pp_size if server_args.mapping.has_pp else 1),
        )
        cache_signature = cache_layout_signature(
            host_cache_executor.layout,
            cache_dtype=f"{server_args.kv_cache_dtype}:{model_config.dtype}",
        )
        import torch.distributed as dist

        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        rank = dist.get_rank() if world_size > 1 else 0
        checkpoint_id = l3_checkpoint_id(
            model_config.model_path,
            hf_config=model_config.hf_config,
            revision=str(model_config.revision or ""),
            load_format=str(server_args.load_format),
            # LoadConfig currently gets the same empty extra-config;
            # both must stay aligned if a shard pattern is wired through.
            model_loader_extra_config={},
            ext_yaml=str(server_args.ext_yaml or ""),
        )
        if draft_model_config is not None:
            draft_revision = l3_checkpoint_id(
                draft_model_config.model_path,
                hf_config=draft_model_config.hf_config,
                revision=str(draft_model_config.revision or ""),
                load_format=str(server_args.load_format),
                model_loader_extra_config={},
                ext_yaml=str(server_args.ext_yaml or ""),
            )
        else:
            draft_revision = ""

        def gather_checkpoint_ids(payload: list) -> list:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, payload)
            return gathered

        checkpoint_id, draft_revision = share_l3_checkpoint_ids(
            [checkpoint_id, draft_revision],
            rank=rank,
            world_size=world_size,
            gather=gather_checkpoint_ids,
        )
        pipeline_rank = server_args.mapping.pp_rank if server_args.mapping.has_pp else 0
        if draft_model_config is not None:
            draft_model = str(draft_model_config.model_path)
            draft_quantization = str(draft_model_config.quantization or "")
        else:
            draft_model = ""
            draft_quantization = ""
        cache_quantization = l3_cache_quantization_id(
            quantization=str(model_config.quantization or ""),
            quantization_param_path=str(server_args.quantization_param_path or ""),
            draft_quantization=draft_quantization,
        )
        attn_tp_size = int(server_args.mapping.attn.tp_size)
        eagle3_layers_to_capture: list[int] = []
        if server_args.speculative_algorithm == "EAGLE3":
            configured_layers = server_args.eagle3_layers_to_capture
            if configured_layers:
                eagle3_layers_to_capture = [int(layer) for layer in configured_layers]
            elif draft_model_config is not None:
                draft_layers = _eagle_aux_layer_ids(draft_model_config.hf_config)
                if draft_layers:
                    eagle3_layers_to_capture = [int(layer) for layer in draft_layers]

        attention_backend_name = attention.attention_backend_name
        draft_attention_backend_name = attention.draft_attention_backend_name

        def prefix_for_weight_version(weight_version: str) -> str:
            return storage_key_prefix(
                server_args.model,
                revision=checkpoint_id,
                weight_version=weight_version,
                model_overrides=dict(model_config.model_override_args),
                cache_signature=cache_signature,
                pipeline_rank=pipeline_rank,
                attn_tp_size=attn_tp_size,
                draft_model=draft_model,
                draft_revision=draft_revision,
                draft_weight_version=weight_version if draft_model else "",
                cache_quantization=cache_quantization,
                runtime_compat=L3_RUNTIME_COMPAT,
                attention_backend=attention_backend_name,
                draft_attention_backend=draft_attention_backend_name,
                skip_softmax_threshold=float(server_args.skip_softmax_threshold),
                eagle3_layers_to_capture=eagle3_layers_to_capture,
            )

        host_cache_executor.attach_l3_storage(
            storage_backend,
            key_prefix=prefix_for_weight_version(server_args.weight_version),
            rank=attn_tp_rank,
            prefix_for_weight_version=prefix_for_weight_version,
            prefetch_timeout_base_s=server_args.kvstore_prefetch_timeout_base_s,
            prefetch_timeout_per_page_s=server_args.kvstore_prefetch_timeout_per_page_s,
            prefetch_batch_pages=server_args.kvstore_prefetch_batch_pages,
        )

    kv_transfer = _build_kv_transfer(
        server_args,
        executor,
        cache_fields_by_stage=attention.cache_fields_by_stage,
        producer_fields_by_step=attention.producer_fields_by_step,
        logical_plan=attention.logical_plan,
        model_config=model_config,
        draft_model_config=draft_model_config,
        gpu_id=gpu_id,
        global_rank=global_rank,
    )

    spec_num_steps, spec_num_tokens = speculative_widths(
        executor.config.spec_algo,
        executor.config.spec_num_steps,
        executor.config.spec_num_tokens,
    )
    specs = DeviceSpecs(
        cache_geometry=views.cache_geometry,
        cache_groups=views.cache_groups,
        cache_storage=attention.cache_storage,
        multimodal_encoder_dtype=target.multimodal_encoder_dtype,
        spec_num_steps=spec_num_steps,
        spec_num_tokens=spec_num_tokens,
        uses_eager_grammar=executor.eager_grammar_buffers is not None,
        supports_disaggregation=views.token_to_kv_pool.arena.supports_disaggregation,
        supports_pd_layerwise_finalization=_supports_pd_layerwise_finalization(
            executor, server_args.mapping
        ),
        supports_prompt_logprobs=(
            server_args.disaggregation_mode == "decode"
            or executor.supports_prompt_logprobs
        ),
        cache_state_group_ids=tuple(
            str(spec.group_id)
            for spec in views.token_to_kv_pool.arena.cache_group_specs
            if spec.family == "state"
        ),
        num_host_pages=(
            host_cache_executor.num_host_pages if host_cache_executor is not None else 0
        ),
        num_snapshot_pages=(
            host_cache_executor.num_snapshot_pages
            if host_cache_executor is not None
            else 1
        ),
        max_retracted_requests=(
            host_cache_executor.max_retracted_requests
            if host_cache_executor is not None
            else 0
        ),
        expert_rebalance=(
            target.expert_location_updater.specs
            if target.expert_location_updater is not None
            else None
        ),
    )

    def encoder_model_facts() -> EncoderModelFacts:
        model = target.model
        vision = next(
            module
            for module in (
                getattr(model, name, None)
                for name in ("visual", "vision_tower", "vision")
            )
            if module is not None
        )
        dtype = getattr(vision, "dtype", None)
        if dtype is None:
            dtype = next(vision.parameters()).dtype
        return EncoderModelFacts(
            device=executor.device,
            hidden=model.config.hidden_size,
            num_deepstack=getattr(model, "num_deepstack_embeddings", 0),
            dtype=dtype,
        )

    return DeviceBuild(
        specs=specs,
        transfer=kv_transfer,
        encoder_model_facts=encoder_model_facts,
        handle=DeviceHandle(
            executor,
            host_cache_executor=host_cache_executor,
            kv_transfer=kv_transfer,
        ),
    )


def _only_aliases_inputs(func) -> bool:
    """Whether ``func`` is a metadata-only op: every output aliases an input.

    View/slice/alias ops carry non-writing alias annotations in their schema
    (``Tensor(a) -> Tensor(a)``) and issue no device work at all — no kernel,
    no CUDA API call — so the guard lets them touch CUDA tensors. Anything
    that materializes data dispatches a separate non-aliasing op (a lazy
    ``reshape`` copy arrives as ``aten.clone``) and is still caught. In-place
    ops alias too but with ``is_write`` set; they launch kernels, so they
    stay banned.
    """
    try:
        returns = func._schema.returns
    except AttributeError:
        return False
    if not returns:
        return False
    return all(r.alias_info is not None and not r.alias_info.is_write for r in returns)


class _NoDeviceWork(TorchDispatchMode):
    """Raise on any torch op that puts device work on this thread.

    Thread-local by construction (dispatch modes are per-thread stacks), so
    pushing it on the control-plane thread bans device work THERE while the
    forward thread runs free — the per-thread "no CUDA context" that CUDA's
    process-wide primary context cannot give us.

    Deliberately porous in exactly the shape of the contract:

    - ``cuda.Event`` create/query/synchronize are not dispatch ops, so the
      inbound channel (``PendingExecution.result``, cache-result polling)
      passes untouched. The flip side: raw stream/event API misuse is not
      caught — this guard covers tensor ops (launches, copies, allocations),
      the dominant accident class.
    - Metadata-only ops (views, slices) on CUDA tensors pass: they issue no
      device work (see ``_only_aliases_inputs``). The control plane holds
      views legitimately — EPD receive-slot leases, for one.
    - Classic ``torch.distributed`` collectives bypass ``__torch_dispatch__``;
      a CUDA collective on the control plane would only be caught via the
      tensor ops that prepare it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._alias_only: dict = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        stack = list(args) + list(kwargs.values())
        while stack:
            value = stack.pop()
            if isinstance(value, torch.Tensor):
                if value.device.type == "cuda":
                    allowed = self._alias_only.get(func)
                    if allowed is None:
                        allowed = self._alias_only[func] = _only_aliases_inputs(func)
                    if allowed:
                        break
                    raise RuntimeError(
                        f"control-plane thread ran {func} on a CUDA tensor; "
                        "device work crosses only through DeviceHandle — see "
                        "docs/design/event-loop.md Principle 1"
                    )
            elif isinstance(value, torch.device):
                if value.type == "cuda":
                    raise RuntimeError(
                        f"control-plane thread ran CUDA factory {func}; "
                        "device work crosses only through DeviceHandle — see "
                        "docs/design/event-loop.md Principle 1"
                    )
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        return func(*args, **kwargs)


def maybe_control_plane_guard():
    """The event loop's enforcement of Principle 1, on by default.

    ``TOKENSPEED_GUARD_CONTROL_PLANE=0`` disables it — the escape hatch for a
    deployment that trips on a control-plane device op we have not routed
    yet; please report such a trip rather than living with the flag. The
    cost while enabled is the dispatch-mode hook (~10us) on each of the
    control plane's few CPU tensor ops per round.

    Returns:
        A context manager: the guard mode unless disabled, else a null
        context.
    """
    if os.environ.get("TOKENSPEED_GUARD_CONTROL_PLANE", "1") == "0":
        return contextlib.nullcontext()
    return _NoDeviceWork()


def _resolve_role(kv_transfer) -> DeviceRole:
    """Read the engine's role off its transfer peer. Fixed for the process."""
    if kv_transfer is None:
        return DeviceRole.PLAIN
    from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
    from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor

    if isinstance(kv_transfer, DisaggDecodeExecutor):
        return DeviceRole.PD_DECODE
    if isinstance(kv_transfer, DisaggPrefillExecutor):
        return DeviceRole.PD_PREFILL
    raise TypeError("kv_transfer must be a Disagg{Prefill,Decode}Executor.")


def _supports_pd_layerwise_finalization(executor, mapping) -> bool:
    """Whether this rank can finalize layerwise CachePD writes with speculation on.

    The draft cache fields are the last pipeline stage's trailing producer
    step (``CacheLayerOwnership``), so a stage before it owns none: its
    readiness is the target layers' alone and nothing remains to finalize.
    The owning stage, like a non-PP engine, answers for the executor's
    ``draft_field_writer``. Without speculation the answer is unused.
    """
    if mapping.has_pp and not mapping.is_last_pp_rank:
        return True
    writer = executor.draft_field_writer
    return writer is not None and writer.supports_pd_layerwise_finalization


def _build_kv_transfer(
    server_args,
    executor,
    *,
    cache_fields_by_stage: tuple[tuple[str, ...], ...],
    producer_fields_by_step: tuple[tuple[str, ...], ...],
    logical_plan: CacheMemoryPlan | None,
    model_config,
    draft_model_config,
    gpu_id: int,
    global_rank: int,
):
    """Build the PD transfer peer, or None outside disaggregation.

    Here rather than in the event loop because everything it needs is either
    ``server_args`` or the completed attention build: the topology comes from
    the mapping, the sync group from the process-group manager, and the
    peer-facing KV description from the pool and explicit cache placement.
    """
    if server_args.disaggregation_mode == "null":
        return None

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.pd.factory import create_kv_transfer, get_kv_args
    from tokenspeed.runtime.pd.mooncake.entities import KVManagerArgs
    from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
    from tokenspeed.runtime.pd.topology import PDParallelTopology

    mapping = server_args.mapping
    topology = PDParallelTopology.from_mapping(mapping)

    # PP: transfer-status consensus must span every stage — all ranks run the
    # same deterministic scheduler and must agree on Bootstrapped/Succeeded
    # events, and the KV for one request is produced by pp*tp ranks together.
    sync_group = pg_manager.get_process_group(
        "gloo", mapping.world_group if mapping.has_pp else mapping.attn.tp_group
    )
    kv_transfer = create_kv_transfer(
        mode=server_args.disaggregation_mode,
        backend=server_args.disaggregation_transfer_backend,
        args=KVManagerArgs(
            bootstrap_port=server_args.disaggregation_bootstrap_port,
            dist_init_addr=server_args.dist_init_addr,
            topology=topology,
            enable_metrics=False,
            served_model_name=server_args.served_model_name,
            app_key=server_args.app_key,
            metrics_reporters=server_args.metrics_reporters,
            enable_dp_attention=mapping.has_attn_dp,
        ),
        kv_args=get_kv_args(
            global_rank,
            gpu_id,  # local CUDA index; new threads do not inherit current_device
            server_args.disaggregation_ib_device,
            executor.token_to_kv_pool,
            model_config=model_config,
            draft_model_config=draft_model_config,
            cache_fields_by_stage=cache_fields_by_stage,
            producer_fields_by_step=producer_fields_by_step,
            logical_plan=logical_plan,
        ),
        gloo_group=sync_group,
    )
    interval = server_args.disaggregation_layerwise_interval
    if isinstance(kv_transfer, DisaggPrefillExecutor) and interval > 0:
        # P-side layerwise KV streaming: one counter, ticked by the attention
        # backends as each layer's KV lands and read by the sender.
        from tokenspeed.runtime.pd.utils import StepCounter

        step_counter = StepCounter(executor.device, gpu_id)
        executor.attn_backend.register_step_counter(step_counter)
        if executor.draft_attn_backend is not None:
            executor.register_draft_final_step_counter(step_counter)
        kv_transfer.register_layerwise_step_counter(step_counter, interval)
    return kv_transfer
