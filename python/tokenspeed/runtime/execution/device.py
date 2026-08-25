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

What comes back is split three ways, by how long the caller may hold it:

- ``DeviceSpecs`` — plain values the control plane plans with (cache
  geometry, speculation widths, capability flags). No device object, safe to
  keep forever, and reading one never goes through the handle.
- ``DeviceWiring`` — the startup steps that need a real device object:
  describing the KV to a PD peer, installing the layerwise step counter,
  reading the encoder's model facts. A local of ``EventLoop.__init__``,
  dropped when it returns.
- ``DeviceHandle`` — the running handle, and the only one the loop keeps.

The split exists because the wiring list is the one that grows: every new
cache tier, transport or accelerator integration wants a hook that needs a
pool or a backend. Sending that growth to an object the running loop does
not hold means it cannot widen what the loop can do to the GPU mid-flight.
Adding a method to ``DeviceHandle`` is then a deliberate change to the running
contract rather than somewhere a startup convenience quietly lands.

See ``forward_thread.py`` for the capture contract each closure must satisfy.
"""

from __future__ import annotations

import contextlib
import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tokenspeed.runtime.execution.types import DpForwardMetadata, PendingExecution


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
    """The three things constructing the device side produces.

    Attributes:
        specs: Plain values the control plane plans with; safe to keep.
        wiring: Startup-only hooks; a local of ``__init__``, dropped after.
        handle: The running handle; the only one the loop keeps.
    """

    specs: DeviceSpecs
    wiring: "DeviceWiring"
    handle: "DeviceHandle"


class DeviceHandle:
    """What the running control plane may ask of the GPU, and nothing else.

    Every method packages its arguments into a closure and hands that closure
    to the forward thread, and none of them returns anything the caller could
    then use to bypass the rest.
    """

    def __init__(self, executor, *, l2_cache_executor=None) -> None:
        # Private by convention AND by absence: nothing below returns it.
        self._executor = executor
        self._thread = executor.forward_thread
        # The host cache tier, or None without --enable-kvstore. Behind the
        # handle for the same reason the pools are: its submit path launches
        # transfers and records events.
        self._l2 = l2_cache_executor
        # Cache-plan submission futures, drained by poll_cache_results: a
        # submission that raised must surface, or its ops stay counted
        # in flight forever and the cache-gated requests hang silently.
        # Appended and drained on the control-plane thread only.
        self._l2_submissions: deque = deque()

    # ------------------------------------------------------------------
    # Per-round work
    # ------------------------------------------------------------------

    def submit_forward(
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
                for a PD prefill handoff.

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

    def submit_page_zeroing(self, pages) -> Future | None:
        """Queue sanitization of newly owned KV pages; never blocks.

        Not awaited by the loop: the forward thread's FIFO already orders the
        zeroing before this round's forward. Only the PD-decode RDMA barrier
        needs the completion event, and it resolves the future from inside
        the thread (see ``run_remote_receive``).

        Args:
            pages: The scheduler's page-reuse list for this round.

        Returns:
            A future resolving to the completion event, or None when no page
            needed sanitizing.
        """
        if not pages:
            return None
        executor = self._executor
        # Bound now, not read at execution time: by then a later round may
        # have rebound the caller's variable.
        return self._thread.submit(lambda pages=pages: executor.zero_cache_pages(pages))

    def submit_cache_plan(self, execution_plan) -> None:
        """Queue the plan's L2 host-cache transfers; never blocks.

        The FIFO position is load-bearing twice over. Launching here (not on
        the control plane) keeps a full CUDA launch queue from stalling the
        scheduler round at its cross-rank collectives. And running AFTER this
        round's page-zeroing closure on the same thread and stream is what
        actually orders "zero, then load" for a page in both sets — the
        load's start event can only capture zeroing that is already enqueued.

        Args:
            execution_plan: The round's plan (a per-round value copy out of
                C++); its cache ops are read on the data plane.
        """
        l2 = self._l2
        if l2 is None:
            raise RuntimeError("cache plan submitted without --enable-kvstore")
        self._l2_submissions.append(
            self._thread.submit(lambda: l2.submit_plan(execution_plan))
        )

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
            RuntimeError: A queued ``submit_plan`` raised on the data plane.
        """
        l2 = self._l2
        if l2 is None:
            raise RuntimeError("cache results polled without --enable-kvstore")
        while self._l2_submissions and self._l2_submissions[0].done():
            exc = self._l2_submissions.popleft().exception()
            if exc is not None:
                raise RuntimeError(
                    "L2 cache-plan submission failed on the data plane; its "
                    "ops have no completion events and would hang their "
                    "cache-gated requests"
                ) from exc
        return l2.poll_results()

    def run_idle_forward(self, dp_metadata: DpForwardMetadata) -> None:
        """Run a zero-token forward so this DP rank joins the round's collectives.

        Args:
            dp_metadata: The round's CPU-gathered DP metadata.
        """
        executor = self._executor
        self._thread.run(lambda: executor.execute_idle_forward(dp_metadata))

    def run_remote_receive(
        self,
        forward_op,
        *,
        cache_zero_future: Future | None,
        trigger: Callable[[Any], None],
    ) -> None:
        """Pull a PD decode request's KV in from the prefill node.

        Slot preparation and the cache-length reset touch the execution
        stream; the RDMA trigger is CPU-side but must follow them and the
        zeroing barrier. One ordered unit, so one submission.

        Args:
            forward_op: The round's op; supplies the admitted rows.
            cache_zero_future: This round's page-zeroing submission, or None.
                Page zeroing runs on a CUDA stream while Mooncake/GPUDirect
                writes are not ordered by it, so the destination manifest is
                published only after the new pages are fully sanitized.
            trigger: The KV transfer executor's ``execute``, run last.
        """
        executor = self._executor
        num_extends = forward_op.num_extends()

        def _receive():
            executor.prepare_remote_cache_slots(
                list(forward_op.request_pool_indices[:num_extends])
            )
            executor.reset_remote_prefill_cache_lengths(forward_op)
            if cache_zero_future is not None:
                cache_zero_event = cache_zero_future.result()
                if cache_zero_event is not None:
                    cache_zero_event.synchronize()
            trigger(forward_op)

        self._thread.run(_receive)

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

    def submit_release(self, release: Callable[[], None]) -> None:
        """Queue a resource release behind every forward already in the queue.

        For resources a queued forward may still read — a request's shared
        multimodal features, freed when it finishes while an earlier round is
        still in flight. The FIFO is the ordering; nothing waits on it.

        Args:
            release: Zero-argument callable performing the release.
        """
        self._thread.submit(release)

    # ------------------------------------------------------------------
    # Memory occupation (pause / release / wake)
    # ------------------------------------------------------------------

    def run_embedding_work(self, work: Callable[[], Any]) -> Any:
        """Run one EPD embedding device step on the data plane; block for it.

        EPD admission's device half — receive-buffer allocation at stage, the
        publish clone/scatter and the NCCL shard reassembly at drain. Named
        for EPD rather than generic on purpose: like ``submit_release`` this
        is a deliberately narrow callable slot, and a second user should add
        its own named method, not widen this one.

        Blocking is the point twice over. Admission needs the results (buffer
        addresses to register, ``item.encoded`` set) before it returns, and
        the P role's round head finds the FIFO empty, so the wait is a thread
        hop. And issuing the reassembly's NCCL broadcasts from the forward
        thread — instead of racing them against forward collectives from a
        second thread — is what keeps the cross-rank launch order identical
        on every rank, which NCCL requires across communicators.

        Args:
            work: Zero-argument callable performing the device step.

        Returns:
            ``work``'s return value; its exception is re-raised here.
        """
        return self._thread.run(work)

    def run_kv_repair(self) -> None:
        """Zero every KV pool's buffers after a wake re-maps them.

        Re-mapped memory holds garbage, and the draft pool is tagged
        ``kv_cache`` too, so a wake that skipped it would feed the draft model
        stale KV. FP8 KV scales ride with the weights region and need no reset.
        """
        executor = self._executor
        self._thread.run(lambda: _clear_kv_buffers(executor))

    # ------------------------------------------------------------------
    # Online weight update (RL trainer sync)
    # ------------------------------------------------------------------
    #
    # These rewrite model parameters in place, so they must be ordered
    # against forwards rather than raced with them — hence the FIFO, and
    # hence the request handler holding this instead of the model runner.

    def init_weights_update_group(self, req) -> tuple[bool, str]:
        """Join the trainer's NCCL weight-update group."""
        runner = self._executor.model_runner
        return self._thread.run(lambda: runner.init_weights_update_group(req))

    def update_weights_from_distributed(self, req) -> tuple[bool, str]:
        """Receive and apply one broadcast of weights from the trainer."""
        runner = self._executor.model_runner
        return self._thread.run(lambda: runner.update_weights_from_distributed(req))

    def destroy_weights_update_group(self, req) -> tuple[bool, str]:
        """Leave the trainer's weight-update group."""
        runner = self._executor.model_runner
        return self._thread.run(lambda: runner.destroy_weights_update_group(req))


class DeviceWiring:
    """Startup-only capabilities: hand the device side to its collaborators.

    Every method here needs a real device object — the KV pool, an attention
    backend, the model — and each exists because one startup step cannot be
    expressed as plain data. That is a list that WILL grow: the next cache
    tier, the next transport, the next accelerator integration all want a
    hook like these.

    So it grows here and not on ``DeviceHandle``. This object lives for the
    length of ``EventLoop.__init__`` and is dropped; the running loop never
    holds one, so nothing added here can widen what the loop can do to the
    GPU mid-flight. If a new hook belongs on ``DeviceHandle`` instead, that is a
    deliberate decision about the RUNNING contract, not a place a startup
    convenience can quietly land.
    """

    def __init__(self, executor) -> None:
        self._executor = executor

    def encoder_model_facts(self) -> EncoderModelFacts:
        """Extract the four model facts EPD admission needs, as plain values.

        Not a ``DeviceSpecs`` field, and handed to the EPD path as a BOUND
        METHOD rather than a value: reading the vision tower's dtype raises
        on a text-only model, so this must only run after the EPD admission
        gate has decided the node is a multimodal prefill node.
        """
        model = self._executor.model_runner.model
        return EncoderModelFacts(
            device=self._executor.device,
            hidden=model.config.hidden_size,
            num_deepstack=getattr(model, "num_deepstack_embeddings", 0),
            dtype=(getattr(model, "visual", None) or model.vision_tower).dtype,
        )

    def pd_kv_args(
        self,
        *,
        global_rank: int,
        ib_device,
        model_config,
        draft_model_config,
        pp_layer_window,
    ):
        """Describe this engine's KV to a PD peer.

        Args:
            global_rank: This worker's global rank, used for both the engine
                and the KV-manager rank fields.
            ib_device: The disaggregation InfiniBand device.
            model_config: The target model's config.
            draft_model_config: The draft model's config, or None.
            pp_layer_window: This stage's layer range under PP, else None.

        Returns:
            The peer-facing KV argument struct.
        """
        from tokenspeed.runtime.pd.factory import get_kv_args

        return get_kv_args(
            global_rank,
            global_rank,
            ib_device,
            self._executor.token_to_kv_pool,
            model_config=model_config,
            draft_model_config=draft_model_config,
            pp_layer_window=pp_layer_window,
        )

    def install_pd_step_counter(self, gpu_id: int):
        """Arm layerwise KV streaming and return the counter the sender reads.

        The counter is what the attention backends tick as each layer's KV
        lands. Registering it is backend surgery, so it happens here; the PD
        sender gets the counter, not the backends.

        Args:
            gpu_id: Local device index the counter lives on.

        Returns:
            The registered ``StepCounter``.
        """
        from tokenspeed.runtime.pd.utils import StepCounter

        executor = self._executor
        step_counter = StepCounter(executor.device, gpu_id)
        executor.attn_backend.register_step_counter(step_counter)
        if executor.draft_attn_backend is not None:
            executor.register_draft_final_step_counter(step_counter)
        return step_counter


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
    the caller may hold it: specs it may keep forever, wiring it must drop
    after construction, and the handle it runs with.

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
        A ``DeviceBuild``. No device object escapes except inside the wiring
        and the handle; the specs hold none at all.
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
    # rank0 only; best-effort, never fails startup.
    if attn_tp_rank == 0:
        log_gpu_memory_summary(
            target.model,
            gpu_id,
            global_rank,
            logger,
            draft_model=draft.model if draft is not None else None,
            kv_pool=token_to_kv_pool,
            draft_kv_pool=draft_token_to_kv_pool,
        )

    l2_cache_executor = None
    if server_args.enable_kvstore:
        if server_args.kvstore_storage_backend is not None:
            raise NotImplementedError(
                "the cache-group scheduler has no L3 storage tier; unset "
                "--kvstore-storage-backend"
            )
        from tokenspeed.runtime.cache.l2.executor import L2CacheExecutor

        l2_cache_executor = L2CacheExecutor(
            token_to_kv_pool,
            draft_pool=draft_token_to_kv_pool,
            host_ratio=server_args.kvstore_ratio,
            host_size_gb=server_args.kvstore_size,
            io_backend=server_args.kvstore_io_backend,
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
    return DeviceBuild(
        specs=specs,
        wiring=DeviceWiring(executor),
        handle=DeviceHandle(executor, l2_cache_executor=l2_cache_executor),
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


def _clear_kv_buffers(executor) -> None:
    """Clear the target and draft KV pools. Runs on the forward thread."""
    for attr in ("token_to_kv_pool", "draft_token_to_kv_pool"):
        pool = getattr(executor, attr, None)
        if pool is not None and hasattr(pool, "clear_kv_buffers"):
            pool.clear_kv_buffers()
