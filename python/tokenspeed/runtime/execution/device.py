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
The host's forward thread (``engine/rust_host.py``) runs them, one at a time
in the order the control plane queued them. The control thread itself calls
only the completion query ``poll_cache_results``, which reads what landed
and touches no stream. There is no accessor here for the executor, a
backend, a buffer, or a stream. The rule "the control plane issues no CUDA
work, and what it hands over it does not touch again" therefore does not
need to be remembered or asserted — the loop cannot reach anything it could
violate it with. What it cannot see, it cannot change.

``build_device_side`` is why that holds at construction too. The model
runners, the attention backends and the KV pools are locals of that function
and the loop never names them, so it cannot keep one by accident. Startup
itself runs on the building thread, before the forward thread starts, which
is why weight loading, autotuning and CUDA graph capture can touch the
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
- ``encoder_model_facts`` — a startup-only callable for the EPD receive,
  resolved past its gate because a text-only model has no vision tower.
- ``host_cache`` — startup-only Host storage and immutable payload facts,
  consumed by Host-only storage construction outside the device builder.
- ``DeviceHandle`` — the running handle, and the only piece the loop stores.

Startup steps that need a real device object (the layerwise step counter,
the peer's KV description) happen INSIDE the builder, so no such object ever
comes back out. A new device interaction belongs in here if it runs once at
startup, and on ``DeviceHandle`` only if the running loop genuinely needs it
— the second widens what the loop can do to the GPU mid-flight.

One thread, one ordering: NCCL collectives on a shared communicator are
issued in the same order on every rank because every rank's forward thread
runs the same operation sequence (the redundant schedulers plan
identically), and a stage's launch-queue backpressure or a blocking stage
recv stalls only the forward thread. Depth 0/1/pp_size only changes how
long a launched forward may stay uncommitted, not where work runs. A
drafter may leave its round pending after a forward and finish it inside
the next one (``BaseDrafter.finish_round``); every other operation here
finishes it first (``settle``).

The capture contract
--------------------

The two planes exchange information ONLY through the two channels below.

1. Outbound is a queued operation's arguments, and only those: plain values
   (lists, bytes, the scheduler's plan moved into a Python object), from
   which the data plane builds what a forward reads (``PlannedForward``).
   Passing something by parking it on a shared object — a field on the
   executor, on an attention backend, on a request's state — is not a
   channel; it is a race waiting for a deep enough queue.

2. Launched means frozen. What a launched forward reads is not edited while
   it may still run: no attribute rebinding, no in-place edit, no releasing
   a resource it holds. Build plain values or a snapshot, and release a
   request's resources through the queue, which runs the release behind the
   forwards that read them.

3. Inbound is the launched forward's result, and only that: its host copies
   land behind its copy event, which the commit waits on before it reads
   them (``HostDataPlane.complete``). Nothing else crosses back.

4. Device-side state is not reachable from the control plane at all. The
   execution stream, ``runtime_states``, ``input_buffers``, and every
   attention backend's forward metadata live behind ``DeviceHandle``, which
   hands back no device object at all. Rules 1 and 2 are enforced by that
   shape rather than by anyone remembering them: what the loop cannot see,
   it cannot pass implicitly or mutate afterwards.

5. Two registered exceptions — and new ones belong in this list, not in a
   comment at the call site:

   - Grammar matchers. ``GrammarStepInputs.grammars`` holds handles to the
     control plane's live matchers (``HostGrammar``), and ownership is split
     by path. Under capturable grammar the side-stream host node advances
     them and the commit path deliberately does not; under eager grammar the
     data plane reads them during the fill and the commit path advances
     them, so the control plane commits its in-flight forwards before
     dispatching a grammar batch. Copying a matcher per round is not free,
     which is why this stays an exception rather than becoming rule 2.

   - Multimodal items. The gather's snapshot is shallow on purpose: the
     ``MultimodalDataItem`` objects stay shared, and the data plane WRITES
     through them mid-forward — the embedder fills ``item.encoded`` and
     drops the raw ``feature``/``feature_shm`` it no longer needs, and EPD
     publishes ``encoded`` into them. The discipline is ordering, not
     freezing: the forward thread makes every such write and runs the
     release of the shared SHM handles, behind any forward still reading
     them. What the shallow copy does freeze is the OUTER struct —
     ``mrope_positions`` and its siblings, which the next gather rebinds.
"""

from __future__ import annotations

import contextlib
import enum
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from tokenspeed_kernel.platform import current_platform
from torch.utils._python_dispatch import TorchDispatchMode

from tokenspeed.runtime.cache.l2.sizing import (
    NO_POOL,
    RetractionPoolRequest,
    tail_lcm_blocks_per_request,
)
from tokenspeed.runtime.distributed.nccl_suspend import (
    resume_nccl_communicators,
    suspend_nccl_communicators,
)
from tokenspeed.runtime.engine.scheduler_utils import is_device_cache_op
from tokenspeed.runtime.epd.recv_pool import recv_pool_bytes
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    ModelExecutionResult,
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
    from tokenspeed.runtime.cache.host_storage import HostCacheResources
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.engine.scheduler_utils import SchedulerCacheGeometry
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.attention.registry import AttentionBuild
    from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass(frozen=True)
class EncoderModelFacts:
    """The narrow model facts the EPD receive needs, without the model.

    Attributes:
        device: The engine's device.
        hidden: Encoder output width, independent of the language hidden width.
        num_deepstack: Number of deepstack embeddings, 0 when unsupported.
        dtype: Encoder output dtype, which may be integer for discrete tokens.
    """

    device: Any
    hidden: int
    num_deepstack: int
    dtype: Any

    @classmethod
    def from_model(cls, model: Any, *, device: Any) -> EncoderModelFacts:
        output_spec = getattr(model, "get_encoder_output_spec", None)
        if output_spec is not None:
            hidden, dtype = output_spec()
        else:
            hidden = model.config.hidden_size
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
        if hidden <= 0:
            raise ValueError("Encoder output width must be positive")
        return cls(
            device=device,
            hidden=hidden,
            num_deepstack=getattr(model, "num_deepstack_embeddings", 0),
            dtype=dtype,
        )


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
        cache_state_group_ids: Group ids of the state cache groups,
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
        host_cache: Startup-only Host resources and immutable payload facts,
            or None without an L2 Host allocation. Contains no Device buffers.
        handle: The running handle; the loop's only reach into the device.
        encoder_model_facts: Zero-argument callable extracting the facts the
            EPD receive needs (``EncoderModelFacts``). A callable, not a
            value: reading the vision tower's dtype raises on a text-only
            model, so it must only run after the EPD receive's gate has
            decided the node is a multimodal prefill node. Startup-only, like
            the rest of this object.
    """

    specs: DeviceSpecs
    transfer: Any
    handle: "DeviceHandle"
    host_cache: HostCacheResources | None
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


def _thread_device(device) -> torch.device:
    """``device`` with the index the calling thread uses: an index-less GPU
    device means the thread's current one."""
    resolved = torch.device(device)
    if resolved.type in ("cuda", "npu") and resolved.index is None:
        index = torch.get_device_module(resolved.type).current_device()
        resolved = torch.device(resolved.type, index)
    return resolved


class DeviceHandle:
    """What the running control plane may ask of the GPU, and nothing else.

    The forward thread calls every method but the completion queries, which
    the control thread calls (see the module docstring); none of them returns
    anything the caller could then use to bypass the rest.
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
        # Resolved here, on the building thread, whose current device the
        # distributed init set: torch starts every new thread on device 0.
        self._forward_device = _thread_device(executor.device)
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

    def bind_forward_thread(self) -> None:
        """Make the calling thread the forward thread: give it the rank's
        device. Called once, by that thread, before any other operation."""
        device = self._forward_device
        if device.type in ("cuda", "npu"):
            torch.get_device_module(device.type).set_device(device.index)

    def settle(self) -> None:
        """Finish a draft round the last forward left pending
        (``complete_pending_drafting``). Every forward-thread task but a
        forward runs behind it; a forward finishes the round itself, at the
        point where it overlaps the most (``docs/design/event-loop.md``)."""
        self._executor.complete_pending_drafting()

    # ------------------------------------------------------------------
    # Per-round work
    # ------------------------------------------------------------------

    def execute(
        self,
        execution_plan,
        planned: "PlannedForward | None",
        *,
        cache_ops: Sequence,
    ) -> ModelExecutionResult | None:
        """Execute one scheduler plan; never waits for the device on the
        per-round path.

        The whole plan, explicitly ordered streams, in an order
        that IS the correctness argument for same-round page reuse: the
        Host-bound stores first (``WriteBackOp`` and ``SnapshotOp``: a
        retraction image's two legs -- the stream-ordered L2 write-back of
        the victim's hash-complete pages and the snapshot store of its tail
        pages and slot state -- read pages this plan re-grants, so they ride
        the write stream consecutively and ONE completion event fences the
        forward thread's default stream before anything else of the plan is
        enqueued; a pinned write-back -- an ordinary publication, whose
        sources the scheduler holds until the ACK -- rides the write stream
        and fences nothing), then the Device-to-Device block copies
        (``pages_to_copy``: a prefix hit's private copy of the block holding
        its boundary, an owner's move off the block it just published; the
        source may be re-granted by this very plan, so the copy precedes the
        zeroing), then page zeroing (the new owner's
        sanitization), then the Device-bound copies (``LoadBackOp`` and
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
                C++. Its cache operations were adapted once by the caller
                (``cache_ops_from_plan``), which passes the Device-copy subset
                below.
            planned: The round's forward, as the data plane built it from the
                control plane's values (sampling params, live grammar
                matchers, the multimodal snapshot, DP metadata). None when no
                batch runs this round, which the plan alone cannot always
                say: an empty ``ForwardBatch`` could be read off it, but a
                DP-idle rank is the outcome of the control plane's agreement.
                Either way the plan's page zeroing and cache transfers —
                retraction writebacks, load-back destinations — still run;
                they do not depend on a forward.

            cache_ops: The plan's Device-copy operations, the ones the control
                plane counts; its storage operations go to the L3 executor.

        Returns:
            The launched forward's result, its host copies landing behind its
            copy event, or None on rounds that run no batch — DP-idle, an
            empty plan, or a D-role round whose only work was the peer's.
        """
        cache_ops = tuple(cache_ops)
        for operation in cache_ops:
            if not is_device_cache_op(operation):
                raise TypeError(
                    f"unsupported Device cache op: {type(operation).__name__}"
                )

        executor = self._executor
        copies = {
            group_id: pairs
            for group_id, pairs in execution_plan.pages_to_copy_arrays().items()
            if len(pairs)
        }
        # The plan lists every group, empty or not; only groups with pages
        # are work, so an idle round launches no zeroing.
        pages = {
            group_id: page_ids
            for group_id, page_ids in execution_plan.pages_to_zero.items()
            if len(page_ids)
        }
        peer = self._kv_transfer
        remote_decode = execution_plan.remote_decode
        remote_prefill = execution_plan.remote_prefill
        if (
            cache_ops
            or copies
            or pages
            or remote_decode is not None
            or remote_prefill is not None
            or (planned is not None and self._role is DeviceRole.PD_PREFILL)
        ):
            # The round's tasks besides its forward run behind a pending
            # draft round.
            self.settle()
        l2 = self._l2 if cache_ops else None
        if l2 is not None:
            # Ahead of the zeroing: a stream-ordered store's sources may be
            # this very plan's pages_to_zero, and its fence lands on the
            # default stream the zeroing runs on, before the zeroing is
            # enqueued. The copies themselves order behind the execution
            # stream, where the forwards wrote the pages.
            l2.submit_write_backs(
                cache_ops,
                prerequisite_stream=executor.execution_stream,
                fence_stream=executor.default_stream,
            )
        # Between the stores and the zeroing: a copy's source may be a block
        # this plan re-grants (and zeroes); its destination is already in the
        # hitter's table and is never zeroed. Stream order on the default
        # stream, behind the execution stream, is the whole argument.
        if copies:
            executor.copy_cache_pages(copies)
        zero_event = executor.zero_cache_pages(pages) if pages else None
        if l2 is not None:
            # Behind the zeroing: the loads' destinations were zeroed on the
            # default stream, so that is the prerequisite they order after.
            l2.submit_load_backs(cache_ops, prerequisite_stream=executor.default_stream)

        # The transfer peer's streams: prefills or decodes the peer NODE runs,
        # handed over asynchronously exactly like the model's ForwardBatch,
        # completion via the transfer events. They are the plan's own work,
        # so they go out even on a DP-idle round that runs no batch.
        if remote_decode is not None:
            # P role: the prompt's decode happens on the peer, so its KV goes
            # out, behind the drafter's pending round (settled above), whose
            # KV goes too. The scheduler emits this only after the final
            # chunk's result landed, so the forwards that wrote the KV were
            # launched before it.
            peer.execute(remote_decode)
        if remote_prefill is not None:
            # D role: the prompt prefills on the peer; pull its KV into the
            # pages this plan admitted (and may be zeroing).
            self._receive_remote_prefill(remote_prefill, zero_event)

        if planned is None:
            return None

        if self._role is DeviceRole.PD_PREFILL:
            # A prefill chunk; the host armed its layerwise KV streaming, if
            # on, when it handed this plan over.
            return self._launch_forward(planned, capture_next_input_ids=True)

        # Plain engine, or a D-role decode (the D role runs no local prefill:
        # a retracted request resumes by restore, not by recompute). A
        # constrained request's matcher was advanced past the prefill node's
        # token when its RemotePrefillDoneEvent landed, so masking continues
        # from the right state.
        return self._launch_forward(planned)

    @property
    def role(self) -> DeviceRole:
        """How this engine divides a plan; see ``DeviceRole``."""
        return self._role

    def _launch_forward(
        self,
        planned,
        *,
        capture_next_input_ids: bool = False,
    ) -> ModelExecutionResult:
        """Launch this round's model forward; never waits for it.

        ``planned`` is the single source of everything the round hands over
        (see ``PlannedForward``'s field-by-field capture notes).

        Args:
            planned: The round's ``PlannedForward``.
            capture_next_input_ids: Whether to keep the round's sampled rows
                (P role: the commit path folds them into the final chunk's
                ExtendResult as the bootstrap payload).

        Returns:
            The forward's result, whose host copies land behind its copy
            event; the commit waits on it.
        """
        phase_range = contextlib.nullcontext()
        # The C++ profiler-enabled query is thread-local and stays false on
        # pre-existing workers under all-thread capture. This installed
        # PyTorch process-global flag is validated by profile startup.
        if getattr(torch.autograd.profiler, "_is_profiler_enabled", False):
            batch = planned.forward_op
            num_extends = batch.num_extends()
            metadata = {
                "mode": ForwardMode.from_num_extends(
                    num_extends, len(batch.request_ids)
                ).name,
                "num_extends": num_extends,
                "request_ids": batch.request_ids,
                "input_lengths": batch.input_lengths,
                "prefill_lengths": batch.prefill_lengths,
                "extend_prefix_lens": batch.extend_prefix_lens,
            }
            phase_range = torch.profiler.record_function(
                f"tokenspeed.forward.v1:{json.dumps(metadata, separators=(',', ':'))}"
            )
        with phase_range:
            return self._executor.execute_forward_op(
                planned.forward_op,
                planned.sampling_params_list,
                dp_metadata=planned.dp_metadata,
                grammar_inputs=planned.grammar_inputs,
                multimodal_context=planned.multimodal_context,
                capture_next_input_ids=capture_next_input_ids,
                ngram_inputs=planned.ngram_inputs,
                request_history_seeds=planned.request_history_seeds,
                input_logprob_plan=planned.input_logprob_plan,
                sampled_top_k=planned.sampled_top_k,
                stop_criteria=planned.stop_criteria,
            )

    def poll_cache_results(self) -> list:
        """Collect completed Host cache ops (both tiers); never blocks.

        Yields the scheduler's ACK kinds -- ``WriteBackDoneEvent``,
        ``LoadBackDoneEvent``, ``SnapshotDoneEvent``, ``RestoreDoneEvent`` --
        for the control plane to replica-intersect and advance.

        The control thread calls it, deliberately: completion is CUDA event
        queries plus queue drains (serialized against the forward thread's
        submissions by the executor's own lock). Queued behind the forward
        thread's work, the round head would wait for every forward still
        being launched.

        Returns:
            The completed ops' scheduler events; empty when nothing finished.
        """
        l2 = self._l2
        if l2 is None:
            raise RuntimeError(
                "cache results polled without --enable-kvstore or a retraction "
                "snapshot pool"
            )
        return l2.poll_results()

    def shutdown_cache(self) -> None:
        """Close the Host cache (the L2 tier and the snapshot pool), once the
        forward thread has stopped, having finished a drafter's pending
        round; a no-op without one."""
        if self._l2 is not None:
            self._l2.shutdown()

    def run_idle_forward(self, dp_metadata: DpForwardMetadata) -> None:
        """Run a zero-token forward so this DP rank joins the round's collectives.

        Args:
            dp_metadata: The round's CPU-gathered DP metadata.
        """
        self.settle()
        self._executor.execute_idle_forward(dp_metadata)

    def _receive_remote_prefill(self, remote_prefill, zero_event) -> None:
        """Pull a remote-prefill stream's rows' KV from the peer.

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
        too. Completion arrives through the transfer events.

        Args:
            remote_prefill: The plan's remote-prefill stream; supplies the
                admitted rows.
            zero_event: This plan's page-zeroing completion, or None when the
                plan zeroes nothing (an admission always zeroes its fresh
                pages, so a granted request's receive never sees None).
        """
        executor = self._executor
        num_extends = remote_prefill.num_extends()
        executor.prepare_remote_cache_slots(
            list(remote_prefill.request_pool_indices[:num_extends])
        )
        executor.reset_remote_prefill_cache_lengths(remote_prefill)
        if zero_event is not None:
            zero_event.synchronize()
        self._kv_transfer.execute(remote_prefill)

    # ------------------------------------------------------------------
    # Memory-saver release/wake (see engine/memory_occupation.py)
    # ------------------------------------------------------------------

    def synchronize(self) -> None:
        """Wait for all device work before a memory region is unmapped."""
        self.settle()
        self._executor.device_module.synchronize()

    def prepare_kv_release(self) -> None:
        """Quiesce everything that can touch the KV arena before it is unmapped.

        The arena's own copy streams (sparse KV offload) and the device are
        synchronized, then the arena is deregistered from the PD transfer
        engine: a registration pins the physical pages the release frees.
        """
        self.settle()
        self._executor.token_to_kv_pool.arena.synchronize()
        self._executor.device_module.synchronize()
        if self._kv_transfer is not None:
            self._kv_transfer.release_cache_registration()

    def finish_kv_resume(self) -> None:
        """Repair the KV arena after a memory-saver wake re-maps its storage.

        Re-mapped memory holds garbage, so the arena is zeroed first. The
        draft pool names the same arena, so clearing the target's clears both.
        FP8 KV scales ride with the weights region and need no reset. The
        arena then goes back to the PD transfer engine, which registers the
        new physical pages.
        """
        self.settle()
        self._executor.token_to_kv_pool.clear_kv_buffers()
        if self._kv_transfer is not None:
            self._kv_transfer.restore_cache_registration()

    def suspend_nccl(self) -> None:
        """Free the NCCL communicators' buffers; they stay usable after resume."""
        self.settle()
        suspend_nccl_communicators()

    def resume_nccl(self) -> None:
        """Restore the NCCL communicators suspended by :meth:`suspend_nccl`."""
        self.settle()
        resume_nccl_communicators()

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
        self.settle()
        executor = self._executor
        if candidate_info is not None:
            req_pool_idx, candidate_ids = candidate_info
            executor.write_remote_spec_candidate_ids(req_pool_idx, candidate_ids)
        if remote_cache_slot is not None:
            executor.mark_remote_cache_ready(remote_cache_slot)

    def reset_expert_load(self) -> None:
        """Zero the router's expert load counters, ordered against forwards.

        The counters are bumped on the execution stream by every forward's
        routing, so the reset rides that stream too: it lands after the
        forwards already issued and before the next one.
        """
        self.settle()
        placement = _recording_expert_placement()
        executor = self._executor
        with executor.device_module.stream(executor.execution_stream):
            placement.reset_load()

    def dump_expert_load(self, path: str) -> dict[str, torch.Tensor | int]:
        """Write this rank's expert load counted since the last reset to ``path``.

        The counters are read back on the forward thread (the one deliberate
        host wait) and saved unreduced, as this rank's record: its physical and
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
        self.settle()
        placement = _recording_expert_placement()
        executor = self._executor
        with executor.device_module.stream(executor.execution_stream):
            with allow_host_sync("expert load dump"):
                physical = placement.physical_load.cpu()
        record = placement.load_record(physical)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, path)
        return record

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
        self.settle()
        placement = _recording_expert_placement()
        executor = self._executor
        with executor.device_module.stream(executor.execution_stream):
            with allow_host_sync("expert load snapshot"):
                return placement.snapshot_load()

    def apply_expert_placement(
        self,
        layer_ids: Sequence[int],
        new_rows: torch.Tensor,
        moves_by_layer: Mapping[int, SlotMoves],
    ) -> None:
        """Move one chunk of layers' expert weights and switch their routing tables.

        Runs inside the execution stream: the P2P over the EP group and the
        slot copies land behind the in-flight forward, each layer's tables
        switch right after its slots, and the next forward is enqueued after
        the whole chunk. The stream is synchronized at the
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
        self.settle()
        updater = self._executor.model_runner.expert_location_updater
        if updater is None:
            raise RuntimeError(
                "expert placement updates need the server to start with --enable-eplb"
            )
        executor = self._executor
        with executor.device_module.stream(executor.execution_stream):
            updater.apply(layer_ids, new_rows, moves_by_layer)
            with allow_host_sync("expert placement chunk"):
                executor.execution_stream.synchronize()

    def update_weights(self, req) -> tuple[bool, str]:
        """Apply one in-place RL weight-sync request, ordered against forwards.

        Type-dispatched on the request — join the trainer's NCCL group,
        receive and apply one broadcast, read one committed version from the
        Mooncake weight store, reload a checkpoint from disk, or tear the group
        down. One entry point because
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
            UpdateWeightFromDiskReqInput,
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromMooncakeReqInput,
        )

        self.settle()
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
            UpdateWeightFromDiskReqInput: runner.update_weights_from_disk,
        }
        handler = handlers.get(type(req))
        if handler is None:
            raise TypeError(f"unsupported weight-update request {type(req).__name__}")
        loads_weights = type(req) in (
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromMooncakeReqInput,
            UpdateWeightFromDiskReqInput,
        )

        result = handler(req)
        if loads_weights and result[0] and self._executor.drafter is not None:
            self._executor.drafter.on_target_weights_updated()
        return result

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
    the PD transfer and L2 executor builders, the EPD receive's NCCL
    warm-up in the rank's bootstrap -- so the rank arms the mode as its
    very last step before serving. It is process-wide:
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
    if server_args.enable_memory_saver:
        # torch_memory_saver 0.0.9 cannot dispose a tagged graph pool and then
        # capture again: empty_cache in the next capture fails with an invalid
        # CUDA free. Keep the serving capture, without a throwaway probe.
        return "memory saver cannot discard a tagged CUDA-graph probe pool"
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
    kv_transfer_builder,
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
    budget (the EPD receive pool, which the rank's bootstrap allocates later,
    is left out of that budget instead).

    Args:
        server_args: Parsed server arguments.
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
        kv_transfer_builder: Builds the PD transfer peer from the finished
            attention build, or returns None outside disaggregation: the
            host's ``build_host_peer`` (``pd/host_peer.py``), bound to the
            outbox its Rust peer reads.

    Returns:
        A ``DeviceBuild``. No device object escapes except inside the
        handle; the specs hold none at all.
    """
    # Imported here: these pull in the model/backend registries, which
    # import execution types — a module-level import would cycle.
    from tokenspeed.runtime.engine.scheduler_utils import log_gpu_memory_summary
    from tokenspeed.runtime.execution.factory import (
        ModelExecutorConfig,
        create_model_executor,
        create_model_runner,
    )
    from tokenspeed.runtime.execution.memory_delta import (
        NULL_MEMORY_DELTA_OBSERVER,
        DriverMemoryDeltaObserver,
    )
    from tokenspeed.runtime.execution.prefill_graph import narrowing_prefill_model
    from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
    from tokenspeed.runtime.layers.attention.registry import (
        create_attn_components,
    )
    from tokenspeed.runtime.utils import set_random_seed

    target, draft = create_model_runner(
        server_args, model_config, draft_model_config, gpu_id, global_rank
    )

    def prepare_multimodal():
        with startup_phase("multimodal.init"):
            if server_args.disaggregation_mode in ("null", "prefill"):
                target.prepare_multimodal_runtime()

    # CDNA4 encoders share the fixed Iris heap with the language model.
    communication_first = current_platform().is_cdna4
    if not communication_first:
        prepare_multimodal()
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
    if communication_first:
        prepare_multimodal()

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
                model=target.model,
                max_req_pool_size=RequestSlotLayout(max_batch_size).padding,
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
        if (
            draft is not None
            and server_args.speculative_algorithm == "DFLASH"
            and not executor.config.prefill_only
        ):
            # Rebinding recreates the draft's auxiliary stream. Graph-phase
            # warmups bypass its overlap path, so initialize its BLAS handle
            # eagerly again before the final capture uses that fresh stream.
            with torch.no_grad():
                executor.forward_step.warmup_decode_path(
                    batch_sizes=(1,), graph_phase=False
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
    host_cache = None
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
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

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
            nprocs_per_node=server_args.mapping.nprocs_per_node,
            world_cpu_group=(
                pg_manager.get_process_group("gloo", server_args.mapping.world_group)
                if server_args.mapping.world_size > 1
                else None
            ),
        )
    elif attn_tp_rank == 0:
        # With an executor the line comes from its resolved sizing.
        logger.info(
            NO_POOL.describe(host_lcm_block_bytes=0, blob_bytes=0, l2_tier=False)
        )
    if host_cache_executor is not None and host_cache_executor.host_storage is not None:
        from tokenspeed.runtime.cache.host_storage import HostCacheResources

        assert host_cache_executor.block_owners is not None
        host_cache = HostCacheResources(
            storage=host_cache_executor.host_storage,
            ownership=host_cache_executor.block_owners,
            groups=host_cache_executor.layout.groups,
            memory_plan=views.token_to_kv_pool.arena.plan,
            attention_backend_name=attention.attention_backend_name,
            draft_attention_backend_name=attention.draft_attention_backend_name,
        )

    kv_transfer = kv_transfer_builder(
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
    _arm_layerwise_streaming(server_args, executor, kv_transfer, gpu_id)

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
            if spec.attn_kind == "state"
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
        return EncoderModelFacts.from_model(target.model, device=executor.device)

    return DeviceBuild(
        host_cache=host_cache,
        specs=specs,
        transfer=kv_transfer,
        encoder_model_facts=encoder_model_facts,
        handle=DeviceHandle(
            executor,
            host_cache_executor=host_cache_executor,
            kv_transfer=kv_transfer,
        ),
    )


@runtime_checkable
class TransferPeer(Protocol):
    """What ``DeviceHandle`` asks of a PD transfer peer.

    ``mode`` is the peer's role, a ``DeviceRole`` value (``"prefill"`` or
    ``"decode"``); ``execute`` takes the plan's remote stream of that role
    once the stream's device prerequisites are met, and sends it.
    """

    mode: str

    def execute(self, op) -> None: ...


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
    pushing it on the Rust host's control thread -- for the completion
    queries it runs in Python -- bans device work THERE while the forward
    thread runs free: the per-thread "no CUDA context" that CUDA's
    process-wide primary context cannot give us.

    Deliberately porous in exactly the shape of the contract:

    - ``cuda.Event`` create/query/synchronize are not dispatch ops, so the
      inbound channel (a result's copy-event wait, cache-result polling)
      passes untouched. The flip side: raw stream/event API misuse is not
      caught — this guard covers tensor ops (launches, copies, allocations),
      the dominant accident class.
    - Metadata-only ops (views, slices) on CUDA tensors pass: they issue no
      device work (see ``_only_aliases_inputs``).
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
    """The control thread's enforcement of Principle 1, on by default.

    ``TOKENSPEED_GUARD_CONTROL_PLANE=0`` disables it — the escape hatch for a
    deployment that trips on a control-plane device op we have not routed
    yet; please report such a trip rather than living with the flag. The
    cost while enabled is the dispatch-mode hook (~10us) on each of the
    control thread's few CPU tensor ops per round.

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
    if not isinstance(kv_transfer, TransferPeer):
        raise TypeError("kv_transfer must be a prefill or decode transfer peer")
    role = DeviceRole(kv_transfer.mode)
    if role is DeviceRole.PLAIN:
        raise TypeError("kv_transfer must be a prefill or decode transfer peer")
    return role


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


def _arm_layerwise_streaming(server_args, executor, kv_transfer, gpu_id: int) -> None:
    """P-side layerwise KV streaming: one counter, ticked by the attention
    backends as each layer's KV lands and read by the transfer peer, whichever
    builder made it."""
    interval = server_args.disaggregation_layerwise_interval
    if kv_transfer is None or kv_transfer.mode != "prefill" or interval <= 0:
        return
    from tokenspeed.runtime.pd.utils import StepCounter

    step_counter = StepCounter(executor.device, gpu_id)
    executor.attn_backend.register_step_counter(step_counter)
    if executor.draft_attn_backend is not None:
        executor.register_draft_final_step_counter(step_counter)
    kv_transfer.register_layerwise_step_counter(step_counter, interval)
