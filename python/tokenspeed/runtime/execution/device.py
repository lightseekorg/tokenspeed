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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    PendingExecution,
    PlannedForward,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


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
        supports_pd_layerwise_finalization: The drafter can finalize
            layerwise KV writes, required for PD layerwise transfer.
        cache_state_group_ids: Group ids of the state-family cache groups,
            for the per-group page-usage debug line. Empty for pools with no
            recurrent/conv state.
        num_host_pages: The L2 host tier's page count (incl. the null page),
            sized here because it depends on the pools' transfer layout; 0
            without ``--enable-kvstore``. The scheduler is configured from it.
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
    cache_state_group_ids: tuple[str, ...]
    num_host_pages: int


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
        l2_cache_executor=None,
        kv_transfer=None,
    ) -> None:
        # Private by convention AND by absence: nothing below returns it.
        self._executor = executor
        self._thread = executor.forward_thread
        # The host cache tier, or None without --enable-kvstore. Behind the
        # handle for the same reason the pools are: its submit path launches
        # transfers and records events.
        self._l2 = l2_cache_executor
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
        self, execution_plan, planned: "PlannedForward | None"
    ) -> PendingExecution | None:
        """Execute one scheduler plan; never blocks on the per-round path.

        The whole plan on the FIFO -- one thread, one stream -- in an order
        that IS the correctness argument for same-round page reuse:
        retraction write-backs first (they must read the reused pages' old
        bytes), then page zeroing (the new owner's sanitization), then
        load-backs (they target zeroed pages), the transfer peer's remote
        streams, and finally the ``ForwardBatch``. The loop hands the round
        over and does not branch on it.

        Args:
            execution_plan: The round's plan, a per-round value copy out of
                C++; its cache ops are read on the data plane.
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
        l2 = self._l2 if execution_plan.cache else None
        if l2 is not None:
            # Ahead of the zeroing: a retraction's snapshot sources may be
            # this very plan's pages_to_zero.
            self._l2_submissions.append(
                self._thread.submit(lambda: l2.submit_write_backs(execution_plan))
            )
        pages = execution_plan.pages_to_zero
        zero_future = (
            self._thread.submit(lambda: executor.zero_cache_pages(pages))
            if pages
            else None
        )
        if l2 is not None:
            self._l2_submissions.append(
                self._thread.submit(lambda: l2.submit_load_backs(execution_plan))
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

        # Plain engine, or a D-role decode / local recovery prefill. A
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
            )

        return PendingExecution(self._thread.submit(_forward))

    def poll_cache_results(self) -> list:
        """Collect completed L2 cache ops; never blocks.

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
            raise RuntimeError("cache results polled without --enable-kvstore")
        _settle(
            self._l2_submissions,
            "L2 cache-plan submission failed on the data plane; its ops have "
            "no completion events and would hang their cache-gated requests",
        )
        return l2.poll_results()

    def query_l3_storage(self, pages) -> list[bool] | None:
        """Probe immutable L3 objects without exposing the Host-cache tier."""

        l2 = self._l2
        if l2 is None:
            return None
        return l2.l3_exists(pages)

    def rotate_l3_namespace(self) -> None:
        """Invalidate this process's view of objects published before clear."""

        if self._l2 is not None:
            self._l2.rotate_l3_namespace()

    def shutdown_cache(self) -> None:
        """Drain and close the Host-cache tier, when configured."""

        if self._l2 is not None:
            self._l2.shutdown()

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
        stream; the RDMA trigger is CPU-issued but writes the same device
        pages, so it must follow them and the zeroing fence (Mooncake and
        GPUDirect writes are not ordered by the zeroing stream, so the
        destination pages must be published from sanitized memory). The same
        fence covers a retraction write-back reading pages this admission was
        granted: the zero event is recorded on the forward thread's stream
        AFTER the write-back copies, so waiting on it waits on them too. One
        ordered
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

    def update_weights(self, req) -> tuple[bool, str]:
        """Apply one in-place RL weight-sync request, ordered against forwards.

        Type-dispatched on the request — join the trainer's NCCL group,
        receive and apply one broadcast, or tear the group down. One entry
        point because it is one capability: rewriting model parameters in
        place, which must be ordered against forwards rather than raced with
        them.

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
        )

        runner = self._executor.model_runner
        handlers = {
            InitWeightsUpdateGroupReqInput: runner.init_weights_update_group,
            UpdateWeightsFromDistributedReqInput: (
                runner.update_weights_from_distributed
            ),
            DestroyWeightsUpdateGroupReqInput: runner.destroy_weights_update_group,
        }
        handler = handlers.get(type(req))
        if handler is None:
            raise TypeError(f"unsupported weight-update request {type(req).__name__}")
        return self._thread.run(lambda: handler(req))


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

    The chain is linear and the order is load-bearing: the multimodal
    runtime must be prepared after weights are loaded and before
    ``create_attn_components`` profiles memory for the KV budget, and the
    chunked-prefill limit must be aligned to the cache groups before
    ``ModelExecutorConfig`` sizes the input buffers from it.

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
        pool_to_cache_groups,
        scheduler_cache_geometry_from_pool,
    )
    from tokenspeed.runtime.execution.factory import (
        ModelExecutorConfig,
        create_model_executor,
        create_model_runner,
    )
    from tokenspeed.runtime.layers.attention.registry import (
        create_attn_components,
    )
    from tokenspeed.runtime.utils import get_colorful_logger

    logger = get_colorful_logger(__name__)

    target, draft = create_model_runner(
        server_args, model_config, draft_model_config, gpu_id, global_rank
    )
    if server_args.disaggregation_mode in ("null", "prefill"):
        target.prepare_multimodal_runtime()

    (
        attn_backend,
        token_to_kv_pool,
        draft_attn_backend,
        draft_token_to_kv_pool,
        cache_storage,
    ) = create_attn_components(
        server_args,
        model_config,
        gpu_id,
        global_rank,
        min_per_gpu_mem,
        server_args.enable_memory_saver,
        draft_model_config,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )

    cache_geometry = scheduler_cache_geometry_from_pool(token_to_kv_pool)
    cache_groups = pool_to_cache_groups(token_to_kv_pool)
    # Lowering the limit is safe; a configured chunk smaller than one
    # state checkpoint block is rejected by aligned_max_scheduled_tokens
    # instead of silently increasing a frozen buffer limit.
    if server_args.enable_prefix_caching:
        aligned = aligned_max_scheduled_tokens(
            server_args.chunked_prefill_size, cache_groups
        )
        if aligned != server_args.chunked_prefill_size:
            logger.warning(
                "chunked_prefill_size=%s is not a multiple of the "
                "state-snapshot checkpoint grain; using %s so recurrent-state "
                "pages can register for prefix-cache reuse.",
                server_args.chunked_prefill_size,
                aligned,
            )
            server_args.chunked_prefill_size = aligned

    executor = create_model_executor(
        server_args=server_args,
        config=ModelExecutorConfig.from_server_args(
            server_args=server_args,
            model_config=model_config,
            max_req_pool_size=max_batch_size + 1,
            gpu_id=gpu_id,
            global_rank=global_rank,
            prefix_granularity=cache_geometry.prefix_granularity,
            overlap_schedule_depth=overlap_schedule_depth,
        ),
        model_runner=target,
        draft_model_runner=draft,
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        draft_attn_backend=draft_attn_backend,
        draft_token_to_kv_pool=draft_token_to_kv_pool,
    )

    # Per-rank GPU memory breakdown (weights by group, KV/graph/non-torch).
    if attn_tp_rank == 0:
        log_gpu_memory_summary(
            target.model,
            gpu_id,
            global_rank,
            logger,
            device=server_args.device,
            draft_model=draft.model if draft is not None else None,
            kv_pool=token_to_kv_pool,
            draft_kv_pool=draft_token_to_kv_pool,
        )

    l2_cache_executor = None
    if server_args.enable_kvstore:
        from tokenspeed.runtime.cache.l2.executor import L2CacheExecutor

        l2_cache_executor = L2CacheExecutor(
            token_to_kv_pool,
            draft_pool=draft_token_to_kv_pool,
            host_ratio=server_args.kvstore_ratio,
            host_size_gb=server_args.kvstore_size,
            io_backend=server_args.kvstore_io_backend,
            attn_tp_rank=attn_tp_rank,
        )
        if server_args.kvstore_storage_backend is not None:
            from tokenspeed.runtime.cache.l3.backend import (
                cache_layout_signature,
                storage_key_prefix,
            )
            from tokenspeed.runtime.cache.l3.factory import (
                create_kvstore_storage_backend,
            )

            storage_backend = create_kvstore_storage_backend(
                server_args.kvstore_storage_backend,
                server_args.kvstore_storage_backend_extra_config,
                host_buffer=l2_cache_executor.host_storage.host_buffer,
                tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            )
            l2_cache_executor.attach_l3_storage(
                storage_backend,
                key_prefix=storage_key_prefix(
                    server_args.model,
                    weight_version=server_args.weight_version,
                    cache_signature=cache_layout_signature(
                        l2_cache_executor.layout,
                        cache_dtype=(
                            f"{server_args.kv_cache_dtype}:{model_config.dtype}"
                        ),
                    ),
                    pipeline_rank=(
                        server_args.mapping.pp_rank if server_args.mapping.has_pp else 0
                    ),
                ),
                rank=attn_tp_rank,
            )

    kv_transfer = _build_kv_transfer(
        server_args,
        executor,
        model_config=model_config,
        draft_model_config=draft_model_config,
        gpu_id=gpu_id,
        global_rank=global_rank,
    )

    specs = DeviceSpecs(
        cache_geometry=cache_geometry,
        cache_groups=cache_groups,
        cache_storage=cache_storage,
        multimodal_encoder_dtype=target.multimodal_encoder_dtype,
        spec_num_steps=executor.config.spec_num_steps or 0,
        spec_num_tokens=executor.config.spec_num_tokens or 0,
        uses_eager_grammar=executor.eager_grammar_buffers is not None,
        supports_disaggregation=token_to_kv_pool.arena.supports_disaggregation,
        supports_pd_layerwise_finalization=bool(
            getattr(executor.drafter, "supports_pd_layerwise_finalization", False)
        ),
        cache_state_group_ids=tuple(
            str(spec.group_id)
            for spec in token_to_kv_pool.arena.cache_group_specs
            if spec.family == "state"
        ),
        num_host_pages=(
            l2_cache_executor.num_host_pages if l2_cache_executor is not None else 0
        ),
    )

    def encoder_model_facts() -> EncoderModelFacts:
        model = target.model
        return EncoderModelFacts(
            device=executor.device,
            hidden=model.config.hidden_size,
            num_deepstack=getattr(model, "num_deepstack_embeddings", 0),
            dtype=(getattr(model, "visual", None) or model.vision_tower).dtype,
        )

    return DeviceBuild(
        specs=specs,
        transfer=kv_transfer,
        encoder_model_facts=encoder_model_facts,
        handle=DeviceHandle(
            executor,
            l2_cache_executor=l2_cache_executor,
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


def _build_kv_transfer(
    server_args,
    executor,
    *,
    model_config,
    draft_model_config,
    gpu_id: int,
    global_rank: int,
):
    """Build the PD transfer peer, or None outside disaggregation.

    Here rather than in the event loop because everything it needs is either
    ``server_args`` or the KV pool this function already owns: the topology
    comes from the mapping, the sync group from the process-group manager,
    and the peer-facing KV description from the pool's transfer layout.
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
    topology.require_cache_pd_supported()

    pp_layer_window = None
    if mapping.has_pp:
        from tokenspeed.runtime.distributed.pp_stage import (
            pp_layer_window as resolve_pp_layer_window,
        )

        pp_layer_window = resolve_pp_layer_window(
            model_config.num_attention_layers, mapping
        )

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
            global_rank,
            server_args.disaggregation_ib_device,
            executor.token_to_kv_pool,
            model_config=model_config,
            draft_model_config=draft_model_config,
            pp_layer_window=pp_layer_window,
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
