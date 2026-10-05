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

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
import zmq
from tokenspeed_kernel.profiling import (
    ProfilingState,
    profile_config_from_env,
    proton_available,
    start_profiling,
    stop_profiling,
)
from viztracer import VizTracer

from tokenspeed.runtime.cache.l3.backend import (
    L3_FLUSH_REQUIRES_WEIGHT_VERSION,
    resolve_l3_weight_version,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.generation_output_processor import RequestState
from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    IsSchedulerPausedReqInput,
    IsSleepingReqInput,
    PauseSchedulerReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    RebalanceExpertsReqInput,
    RebalanceExpertsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    ResumeSchedulerReqInput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromMooncakeReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    mooncake_load_weight_version,
)
from tokenspeed.runtime.engine.request_types import FINISH_ABORT
from tokenspeed.runtime.engine.scheduler_utils import (
    RETRACTION_SAFE_STEPS,
    UNBOUNDED_CACHED_PREFIX_TOKENS,
    make_spec,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.grammar.grammar_manager import GrammarManager
from tokenspeed.runtime.moe.expert_location import (
    EXPERT_LOAD_RECORD_SUFFIX,
    expert_load_recording_enabled,
)
from tokenspeed.runtime.moe.expert_rebalance import (
    EplbApplyChunk,
    EplbCommit,
    EplbSnapshot,
)
from tokenspeed.runtime.multimodal.shm_transport import prepare_shm_features
from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo
from tokenspeed.runtime.utils import PipelinedPyobjBroadcaster
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


# Frontend ops the scheduler completes through the attention-DP same-round
# gate (``_rendezvous_replica_flush``), with the reply type for each: the
# weight ops, and the manual expert-rebalance trigger, which starts the same
# internal op sequence the periodic trigger does. The position is the op's
# type code on the wire of that gate; append only.
_WEIGHT_OPS: tuple[tuple[type, type], ...] = (
    (InitWeightsUpdateGroupReqInput, InitWeightsUpdateGroupReqOutput),
    (UpdateWeightsFromDistributedReqInput, UpdateWeightsFromDistributedReqOutput),
    (DestroyWeightsUpdateGroupReqInput, DestroyWeightsUpdateGroupReqOutput),
    (UpdateWeightsFromMooncakeReqInput, UpdateWeightsFromMooncakeReqOutput),
    (RebalanceExpertsReqInput, RebalanceExpertsReqOutput),
)
_WEIGHT_OP_CODES: dict[type, int] = {
    req_type: code for code, (req_type, _) in enumerate(_WEIGHT_OPS, start=1)
}
_WEIGHT_OP_OUTPUTS: dict[type, type] = dict(_WEIGHT_OPS)
# Ops that rewrite model parameters: they may ask for a cache flush and they
# publish the L3 weight version; group init/teardown do neither.
_WEIGHT_LOAD_OPS = (
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromMooncakeReqInput,
)
# Internal control ops: the online expert rebalance's steps, enqueued by the
# engine itself at rank-identical rounds (``enqueue_internal_op``) and
# completed through the same gate from their own FIFO. Frontend ops arrive on
# different DP workers in different rounds, so sharing one FIFO would make the
# heads differ in kind across ranks. Position is the wire code; append only.
_INTERNAL_OPS: tuple[type, ...] = (EplbSnapshot, EplbCommit, EplbApplyChunk)
_INTERNAL_OP_CODES: dict[type, int] = {
    kind: code for code, kind in enumerate(_INTERNAL_OPS, start=1)
}
_WEIGHT_GATE_WIDTH = 7


class InternalOpCompleter(Protocol):
    """Completes a ready internal op through named ``DeviceHandle`` operations.

    Installed by the engine's rebalance hooks (``engine/eplb_hooks.py``); the
    handler owns the FIFO and the gate, the completer owns the op semantics.
    """

    def complete(self, op) -> None: ...

    def begin_manual_rebalance(self) -> tuple[bool, str]: ...


def _weight_op_wants_flush(recv_req) -> bool:
    return isinstance(recv_req, _WEIGHT_LOAD_OPS) and bool(recv_req.flush_cache)


def _requested_weight_version(recv_req) -> str | None:
    """The L3 namespace a load op asks to publish, read off the request.

    A flushed Mooncake load without an explicit ``weight_version`` publishes
    the committed version's own identity (``mooncake_load_weight_version``);
    the request object is left as it arrived.
    """
    if isinstance(recv_req, UpdateWeightsFromMooncakeReqInput):
        return mooncake_load_weight_version(
            version=recv_req.version,
            flush_cache=recv_req.flush_cache,
            weight_version=recv_req.weight_version,
        )
    return recv_req.weight_version


def _profile_rank_tag(attn_mapping) -> str:
    """File-name tag identifying this scheduler process's profile outputs."""
    parts = []
    if attn_mapping.has_dp:
        parts.append(f"DP{attn_mapping.dp_rank}")
    parts.append(f"TP{attn_mapping.tp_rank}")
    return "-".join(parts)


class RequestHandler:
    """
    1. Recv Reqs from ZMQ
    2. manage sessions
    """

    def __init__(
        self,
        server_args: ServerArgs,
        hf_eos_token_id,
        max_req_len: int,
        vocab_size: int,
        recv_func,
        send_func,
        can_clear_cache_fn,
        clear_cache_fn=None,
        architectures: list[str] | None = None,
        *,
        tokenizer_kwargs: Mapping[str, object],
        pause_controller=None,
        memory_controller=None,
        device=None,
    ) -> None:

        self.forward_ct = 0
        self.server_args = server_args
        # Owns pause/resume state; shared with the event loop. See pause.py.
        self.pause_controller = pause_controller
        # Owns release/resume_memory_occupation (data plane). See
        # memory_occupation.py. Shares the pause controller's drain machinery.
        self.memory_controller = memory_controller
        # In-place RL weight sync (NCCL group init + receive) goes over the
        # data plane so it is ordered against forwards. The scheduler worker
        # passes the handle in; None elsewhere (e.g. unit tests).
        self._device = device

        mapping = server_args.mapping
        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_rank = mapping.attn.tp_rank
        self.attn_global_rank = mapping.attn.rank
        if mapping.has_pp:
            # Chunk-pipeline: every stage's scheduler runs the same
            # deterministic plan, so every rank in the WORLD must see the
            # same request stream. Only global rank 0 owns the ZMQ input;
            # the broadcast fans out across stages, not just one TP group.
            self.attn_tp_size = mapping.world_size
            self.attn_tp_rank = mapping.rank
            self.attn_tp_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.world_group
            )
            self.attn_tp_src_rank = mapping.world_group[0]
        else:
            self.attn_tp_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.attn.tp_group
            )
            self.attn_tp_src_rank = mapping.attn.tp_group[0]
        # Cache-owning ranks in this DP replica (attention TP × PP).
        # Distinct from attn_tp_* above: with PP those become WORLD so the
        # request stream is identical across stages, which would also pull
        # DP ranks into the TP MIN. Exists uses these replica groups;
        # flush appends attention DP as its own group afterwards.
        self._replica_tp_size = mapping.attn.tp_size
        self._replica_tp_cpu_group = pg_manager.get_process_group(
            "gloo", mapping.attn.tp_group
        )
        self.pp_size = mapping.pp_size
        self.pp_cpu_group = (
            pg_manager.get_process_group("gloo", mapping.pp_group)
            if mapping.has_pp
            else None
        )
        # Flush MIN includes attention DP after TP/PP: object keys omit
        # DP rank, so DP replicas share the Mooncake namespace. Exists,
        # prefetch, and WriteBackDone stay TP/PP only (EventLoop /
        # L2CacheHooks); those ranks hold different sequences.
        self.attn_dp_size = mapping.attn.dp_size
        self.attn_dp_cpu_group = (
            pg_manager.get_process_group("gloo", mapping.attn.dp_group)
            if mapping.has_attn_dp
            else None
        )
        self.req_broadcaster = (
            PipelinedPyobjBroadcaster(
                self.attn_global_rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_src_rank,
            )
            if self.attn_tp_size != 1
            else None
        )
        self.profile_rank_tag = _profile_rank_tag(mapping.attn)
        # Rendezvous buffer for _profile_sync. Preallocated because the stage
        # transitions run on the control-plane thread, where allocating is what
        # Principle 1 forbids -- see _profile_sync.
        self._profile_sync_buf = torch.zeros(1, dtype=torch.int32, device="cpu")
        # Same constraint as _profile_sync: gloo barrier would CUDA-allocate.
        self._replica_decision_buf = torch.zeros(1, dtype=torch.int32, device="cpu")
        # Per-round attention-DP MAX all-reduce: standalone flush intent plus
        # the weight-op gate (see _rendezvous_replica_flush for the layout).
        self._replica_flush_want_buf = torch.zeros(
            _WEIGHT_GATE_WIDTH, dtype=torch.int32, device="cpu"
        )
        # Weight ops (NCCL group init/teardown, distributed and Mooncake
        # loads) in arrival order. The head completes only in a round where
        # every attention-DP rank holds the same op at its head: the device
        # call blocks this control thread in a collective or an SDK read, and
        # a peer that had not dequeued its copy would wait for this rank in
        # the per-round DP all-reduce -- a deadlock. One op per round.
        self._pending_weight_ops: deque = deque()
        # Internal control ops (the expert rebalance's steps) in enqueue
        # order, completed through the same gate; see _INTERNAL_OPS. The
        # completer is installed by the rebalance hooks when --enable-eplb.
        self._pending_internal_ops: deque = deque()
        self._internal_op_completer: InternalOpCompleter | None = None

        self.hf_eos_token_id = hf_eos_token_id
        self.max_req_len = max_req_len
        # Head TP over attention-DP ranks serves decode rows only, so that
        # engine cannot run the local recovery prefill a capacity retraction
        # would need; it admits only requests the scheduler never retracts
        # (generation budget within one safe-step window,
        # docs/design/scheduler.md section 4). Head TP over the query shards
        # of a prefill engine serves its extend rows and keeps no budget.
        self.max_new_tokens_budget: int | None = (
            RETRACTION_SAFE_STEPS if mapping.attn.head_tp_serves_decode_only else None
        )
        # LM-head TP under attention DP exchanges the logits rows with the
        # group once per forward (LogitsProcessor._lm_head_tp_row_counts); the
        # prompt-logprob chunk loop would run that exchange a per-rank number
        # of times, so this engine refuses requests asking for prompt logprobs.
        self.supports_input_logprobs: bool = not (
            mapping.attn.has_dp and mapping.lm_head.has_tp
        )
        self.vocab_size = vocab_size
        self.clear_cache_fn = clear_cache_fn
        self.can_clear_cache_fn = can_clear_cache_fn

        self.tokenizer = get_tokenizer(
            server_args.tokenizer,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            architectures=architectures,
            **tokenizer_kwargs,
        )

        self.recv_func = recv_func
        self.send_func = send_func

        self.control_request_dispatcher = TypeBasedDispatcher(
            [(ProfileReq, self.profile)]
        )

        self.grammar_manager = GrammarManager(
            self.server_args, self.tokenizer, self.vocab_size
        )

        self.init_profiler()

    def _drain_reqs(self) -> list | None:
        if self.attn_tp_rank == 0:
            recv_reqs = []

            while True:
                try:
                    recv_req = self.recv_func.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        return recv_reqs

    def recv_reqs(self) -> list:
        if self.attn_tp_size == 1:
            recv_reqs = self._drain_reqs()
        else:
            if not self.req_broadcaster.in_flight:
                self.req_broadcaster.start(self._drain_reqs())
            recv_reqs = self.req_broadcaster.finish()

        if recv_reqs:
            prepare_shm_features(recv_reqs, self.attn_tp_cpu_group)

        if self.attn_tp_size != 1:
            self.req_broadcaster.start(self._drain_reqs())

        return recv_reqs

    def process_requests(self, recv_reqs: list):
        """Dispatch control requests and return new generate request specs and states."""
        new_req_specs, req_states, bootstrap_infos, abort_rids = [], [], [], []
        pending_flush_outputs = 0
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                req_spec, req_state, bootstrap_info = self.handle_generate_request(
                    recv_req
                )

                new_req_specs.append(req_spec)
                req_states.append(req_state)
                bootstrap_infos.append(bootstrap_info)
            elif isinstance(recv_req, ProfileReq):
                output = self.control_request_dispatcher(recv_req)
                if output is not None:
                    self.send_func.send_pyobj(output)
            elif isinstance(recv_req, AbortReq):
                logger.debug(f"AbortReq for rid={recv_req.rid!s}")
                abort_rids.append(recv_req.rid)
            elif isinstance(recv_req, FlushCacheReqInput):
                pending_flush_outputs += 1
            elif isinstance(recv_req, PauseSchedulerReqInput):
                # State change + reply (abort/wait replies are deferred by the
                # controller until the event loop observes a drained scheduler).
                self.pause_controller.handle_pause(recv_req)
            elif isinstance(recv_req, ResumeSchedulerReqInput):
                self.pause_controller.handle_resume(recv_req)
            elif isinstance(recv_req, IsSchedulerPausedReqInput):
                self.pause_controller.handle_is_paused(recv_req)
            elif isinstance(recv_req, ReleaseMemoryOccupationReqInput):
                # Deferred: pauses + drains, then frees GPU memory and replies.
                self.memory_controller.handle_release(recv_req)
            elif isinstance(recv_req, ResumeMemoryOccupationReqInput):
                self.memory_controller.handle_resume(recv_req)
            elif isinstance(recv_req, IsSleepingReqInput):
                self.memory_controller.handle_is_sleeping(recv_req)
            elif isinstance(recv_req, GetInternalStateReq):
                self.send_func.send_pyobj(GetInternalStateReqOutput(internal_state={}))
            elif isinstance(recv_req, SetInternalStateReq):
                self.send_func.send_pyobj(
                    SetInternalStateReqOutput(updated=False, server_args={})
                )
            elif isinstance(
                recv_req,
                (
                    InitWeightsUpdateGroupReqInput,
                    DestroyWeightsUpdateGroupReqInput,
                    RebalanceExpertsReqInput,
                ),
            ):
                # RL weight sync: join / leave the trainer's NCCL group, and
                # the manual rebalance trigger. Gated like the loads: the
                # rendezvous blocks this thread too.
                self._pending_weight_ops.append(recv_req)
            elif isinstance(recv_req, _WEIGHT_LOAD_OPS):
                ok, msg = self._require_weight_version_for_l3_flush(recv_req)
                if ok:
                    ok, msg = self._require_flush_for_l3_version_switch(recv_req)
                if not ok:
                    self.send_func.send_pyobj(
                        _WEIGHT_OP_OUTPUTS[type(recv_req)](success=ok, message=msg)
                    )
                else:
                    self._pending_weight_ops.append(recv_req)
            # The in-engine RL control app refuses these two sources up front
            # (SUPPORTED_WEIGHT_UPDATE_SOURCES in io_struct, also advertised to
            # gateways as rl.update_from); keep that set in step with the load
            # branches above.
            elif isinstance(recv_req, UpdateWeightsFromTensorReqInput):
                self.send_func.send_pyobj(
                    UpdateWeightsFromTensorReqOutput(
                        success=False,
                        message="update_weights_from_tensor is not supported on "
                        "this engine",
                    )
                )
            elif isinstance(recv_req, UpdateWeightFromDiskReqInput):
                self.send_func.send_pyobj(
                    UpdateWeightFromDiskReqOutput(
                        success=False,
                        message="update_weights_from_disk is not supported on "
                        "this engine",
                        num_paused_requests=0,
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported request type: {type(recv_req)}")
        flush_success, ready_op, ready_internal = self._rendezvous_replica_flush(
            pending_flush_outputs=pending_flush_outputs
        )
        if ready_op is not None:
            self._complete_weight_update(ready_op, flush_success=flush_success)
        elif ready_internal is not None:
            self._complete_internal_op(ready_internal)
        return new_req_specs, req_states, bootstrap_infos, abort_rids

    # ------------------------------------------------------------------
    # Internal control ops (online expert rebalance)
    # ------------------------------------------------------------------

    def set_internal_op_completer(self, completer: InternalOpCompleter) -> None:
        """Install the owner of the internal ops (once, at startup)."""
        if self._internal_op_completer is not None:
            raise RuntimeError("an internal op completer is already installed")
        self._internal_op_completer = completer

    def enqueue_internal_op(self, op) -> None:
        """Queue an internal control op for the same-round gate.

        Called at a rank-identical round (the rebalance controller's forward
        count), so every attention-DP rank queues the same op in the same
        round and the gate completes it as soon as every rank holds it.
        """
        if type(op) not in _INTERNAL_OP_CODES:
            raise TypeError(f"{type(op).__name__} is not an internal control op")
        if self._internal_op_completer is None:
            raise RuntimeError("no internal op completer is installed")
        self._pending_internal_ops.append(op)

    def _complete_internal_op(self, op) -> None:
        """Run a gate-agreed internal op through the completer's named device ops."""
        if self._internal_op_completer is None:
            raise RuntimeError("no internal op completer is installed")
        self._internal_op_completer.complete(op)

    def _require_weight_version_for_l3_flush(self, recv_req) -> tuple[bool, str]:
        """Reject a flushed L3 update that has no checkpoint identity.

        Minting ``{current}-uN`` from the old label is not checkpoint
        specific: two servers that start at ``default`` and load different
        weights would both publish under ``default-u1``. The second flush
        only deletes the old ``default`` prefix, so the first server's
        objects remain and can be restored for incompatible weights.
        """

        storage_backend = getattr(self.server_args, "kvstore_storage_backend", None)
        if (
            storage_backend is None
            or not recv_req.flush_cache
            or _requested_weight_version(recv_req) is not None
        ):
            return True, ""
        return False, L3_FLUSH_REQUIRES_WEIGHT_VERSION

    def _require_flush_for_l3_version_switch(self, recv_req) -> tuple[bool, str]:
        """Reject an L3 namespace change that would skip cache invalidation.

        Device/Host still hold KV from the previous checkpoint until a
        successful ``flush_cache``. Switching the Mooncake prefix first
        lets later D2H copies (not yet in ``_backup_futures``) land under
        the new namespace, and lets Admit reuse the stale local indexes
        as if they belonged to the new weights.
        """

        storage_backend = getattr(self.server_args, "kvstore_storage_backend", None)
        if storage_backend is None or recv_req.flush_cache:
            return True, ""
        version = resolve_l3_weight_version(
            self.server_args.weight_version,
            _requested_weight_version(recv_req),
            flush_cache=False,
            storage_backend=storage_backend,
        )
        if version is None or str(version) == str(self.server_args.weight_version):
            return True, ""
        return (
            False,
            "L3 weight_version cannot change without flush_cache; "
            "retry the update with flush_cache=True so Device/Host KV "
            "and in-flight writebacks cannot land in the new namespace",
        )

    def _rendezvous_replica_flush(self, *, pending_flush_outputs: int):
        """Enter flush collectives from every rank on every round; gate weight ops.

        ``FlushCacheReqInput`` and the weight ops are sent separately to each
        attention-DP worker. If only the worker that dequeued one entered
        ``_try_clear_replica_cache`` or blocked in ``update_weights``, that
        rank would DP all-reduce (or sit in the trainer's NCCL broadcast)
        while a lagging peer continued to ``EventLoop._dp_sync_and_check``
        and world-gathered. One MAX all-reduce on the DP group settles both
        on every rank identically, plus the internal ops' FIFO::

            [0] standalone flush intent (any rank)
            [1] -1 if this rank has a queued weight op, else 0
                -> MAX is -1 only when every rank has one
            [2] +type code of the head op (0 without one)
            [3] -type code of the head op
                -> [2] == -[3] means every head is the same kind of op
            [4] -1 if this rank has a queued internal op, else 0
            [5] +type code of the internal head (0 without one)
            [6] -type code of the internal head

        A head is popped only when every rank has one; the flush intent of
        a load op then counts because every rank reads it off its own copy
        of the same op. Standalone flushes reply here. When both FIFOs are
        ready, the frontend op completes this round and the internal op
        waits for the next: one blocking device call per round.

        Returns:
            ``(flush_success, ready_op, ready_internal)`` -- the frontend op
            to complete this round, or the internal op to complete, or
            neither (never both).

        Raises:
            RuntimeError: Attention-DP ranks hold different kinds of ops at
                their queue heads. The frontend sends every op to every
                worker in one order and internal ops are enqueued at
                rank-identical rounds, so this is a transport bug, and
                completing mismatched ops would hang the collectives.
        """

        head = self._pending_weight_ops[0] if self._pending_weight_ops else None
        head_code = 0 if head is None else _WEIGHT_OP_CODES[type(head)]
        internal = self._pending_internal_ops[0] if self._pending_internal_ops else None
        internal_code = 0 if internal is None else _INTERNAL_OP_CODES[type(internal)]
        want_standalone_flush = pending_flush_outputs > 0
        all_have_head = head is not None
        all_have_internal = internal is not None
        if self.attn_dp_size > 1 and self.attn_dp_cpu_group is not None:
            buf = self._replica_flush_want_buf
            buf[0] = 1 if want_standalone_flush else 0
            buf[1] = -1 if head is not None else 0
            buf[2] = head_code
            buf[3] = -head_code
            buf[4] = -1 if internal is not None else 0
            buf[5] = internal_code
            buf[6] = -internal_code
            torch.distributed.all_reduce(
                buf, op=torch.distributed.ReduceOp.MAX, group=self.attn_dp_cpu_group
            )
            reduced = buf.tolist()
            want_standalone_flush = reduced[0] > 0
            all_have_head = reduced[1] == -1
            if all_have_head and reduced[2] != -reduced[3]:
                raise RuntimeError(
                    "attention-DP ranks hold different weight-update operations "
                    f"at their queue heads (local {type(head).__name__}); the "
                    "frontend must send every weight op to every DP worker in "
                    "the same order"
                )
            all_have_internal = reduced[4] == -1
            if all_have_internal and reduced[5] != -reduced[6]:
                raise RuntimeError(
                    "attention-DP ranks hold different internal control "
                    f"operations at their queue heads (local "
                    f"{type(internal).__name__}); the rebalance controller "
                    "must enqueue the same ops in the same rounds on every rank"
                )
        ready_op = self._pending_weight_ops.popleft() if all_have_head else None
        ready_internal = (
            self._pending_internal_ops.popleft()
            if all_have_internal and ready_op is None
            else None
        )
        want_flush = want_standalone_flush or (
            ready_op is not None and _weight_op_wants_flush(ready_op)
        )
        flush_success = True
        if want_flush:
            flush_success = self._try_clear_replica_cache()
        for _ in range(pending_flush_outputs):
            self.send_func.send_pyobj(FlushCacheReqOutput(success=flush_success))
        return flush_success, ready_op, ready_internal

    def _complete_weight_update(self, recv_req, *, flush_success: bool) -> None:
        """Finish a gated weight op after the rank-identical flush.

        The device result is MIN-reduced across the replica (attention TP,
        CP, PP, then DP) before the L3 weight version is published and the
        reply is sent, so one rank's failed load fails the update everywhere
        and no rank serves the new namespace against old weights.
        """

        if isinstance(recv_req, RebalanceExpertsReqInput):
            # Not a weight op: the frontend trigger of the internal rebalance
            # sequence. The decision is rank-identical (the controller's phase
            # is), and the snapshot it takes runs its collectives inside this
            # gated round like the weight ops do.
            if self._internal_op_completer is None:
                local_ok, msg = False, (
                    "expert rebalancing needs the server to start with --enable-eplb"
                )
            else:
                local_ok, msg = self._internal_op_completer.begin_manual_rebalance()
            ok = self._converge_replica_decision(local_ok)
            if local_ok and not ok:
                msg = f"rebalance refused on another rank in the replica ({msg})"
            self.send_func.send_pyobj(
                RebalanceExpertsReqOutput(success=ok, message=msg)
            )
            return
        if _weight_op_wants_flush(recv_req) and not flush_success:
            ok = False
            msg = (
                "cache flush failed; retry the update after in-flight "
                "Host writebacks drain"
            )
        else:
            local_ok, msg = self._device.update_weights(recv_req)
            ok = self._converge_replica_decision(local_ok)
            if local_ok and not ok:
                msg = f"weight update failed on another rank in the replica ({msg})"
            if ok and isinstance(recv_req, _WEIGHT_LOAD_OPS):
                ok, msg = self._commit_l3_weight_version(recv_req, msg)
        self.send_func.send_pyobj(
            _WEIGHT_OP_OUTPUTS[type(recv_req)](success=ok, message=msg)
        )

    def _try_clear_replica_cache(self) -> bool:
        """MIN-reduce clearability, delete L3, then mutate Device/Host.

        Mirrored schedulers must keep the same prefix indexes. A rank whose
        Host writebacks have drained must not ``ClearCache`` while a TP, CP,
        PP, or DP peer still rejects. Remote L3 deletion is its own
        error-returning phase: it runs after the probe agrees and before
        the irreversible local clear, then MIN-reduces so a Mooncake
        failure cannot leave one rank cleared and another in NCCL.
        Exists uses attention TP, then CP, then PP. Flush appends DP
        because DP replicas share Mooncake objects.
        """

        if not self._converge_replica_decision(self.can_clear_cache_fn()):
            return False
        if not self._converge_replica_decision(self._delete_l3_namespace()):
            return False
        return self.clear_cache_fn is not None and self.clear_cache_fn()

    def _delete_l3_namespace(self) -> bool:
        device = self._device
        if device is None:
            return True
        return bool(device.delete_l3_namespace())

    def _converge_replica_decision(self, local_ok: bool) -> bool:
        """MIN-reduce a yes/no across every rank that shares this flush.

        Attention TP, then PP (same order as
        ``EventLoop._converge_l3_exists``), then attention DP. Exists and
        WriteBackDone omit DP because those ranks hold different sequences.
        Flush includes DP: ``storage_object_key`` has no DP rank, so a
        replica that is clearable must not ``remove_by_prefix`` while
        another DP replica still has in-flight Host-to-store backups.
        CPU-tensor gloo all_reduce, not a barrier: see ``_profile_sync``.
        """

        groups = []
        if self._replica_tp_size > 1 and self._replica_tp_cpu_group is not None:
            groups.append(self._replica_tp_cpu_group)
        if self.pp_size > 1 and self.pp_cpu_group is not None:
            groups.append(self.pp_cpu_group)
        if self.attn_dp_size > 1 and self.attn_dp_cpu_group is not None:
            groups.append(self.attn_dp_cpu_group)
        if not groups:
            return local_ok
        buf = self._replica_decision_buf
        buf[0] = 1 if local_ok else 0
        for group in groups:
            torch.distributed.all_reduce(
                buf, op=torch.distributed.ReduceOp.MIN, group=group
            )
        return bool(buf.item())

    def converge_replica_decision(self, local_ok: bool) -> bool:
        """Public replica MIN-reduce for control-plane yes/no decisions."""

        return self._converge_replica_decision(local_ok)

    def _commit_l3_weight_version(self, recv_req, msg: str) -> tuple[bool, str]:
        """Publish under the new checkpoint after a successful GPU load.

        Device/Host have already been flushed when ``flush_cache`` was
        requested. The prefix is rebuilt here so newly computed KV cannot
        land in a peer still serving the previous ``weight_version``.
        Flushed L3 updates require an explicit ``weight_version``. An
        explicit new version with ``flush_cache=False`` is rejected
        before the GPU load.
        """

        ok, err = self._require_weight_version_for_l3_flush(recv_req)
        if not ok:
            return False, err
        ok, err = self._require_flush_for_l3_version_switch(recv_req)
        if not ok:
            return False, err
        version = resolve_l3_weight_version(
            self.server_args.weight_version,
            _requested_weight_version(recv_req),
            flush_cache=recv_req.flush_cache,
            storage_backend=getattr(self.server_args, "kvstore_storage_backend", None),
        )
        if version is None:
            return True, msg
        self.server_args.weight_version = str(version)
        self._device.set_l3_weight_version(str(version))
        return True, msg

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        if recv_req.bootstrap_port is None:
            recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

        # The decode role never computes prompt rows (the prefill node returns
        # the prompt logprobs), so it neither accumulates them nor caps its
        # admission probe for them.
        computes_prompt_logprobs = self.server_args.disaggregation_mode != "decode"
        req_state = RequestState.from_recv_req(
            recv_req,
            tokenizer=self.tokenizer,
            eos_token_ids=self.hf_eos_token_id,
            computes_prompt_logprobs=computes_prompt_logprobs,
        )
        # Prompt logprobs need logits for every position from the start on, so
        # the admission probe may not match those positions from the prefix
        # cache.
        max_cached_prefix_tokens = (
            req_state.logprob_start_len
            if req_state.wants_input_logprobs
            else UNBOUNDED_CACHED_PREFIX_TOKENS
        )
        req_spec = make_spec(
            rid=recv_req.rid,
            tokens=recv_req.input_ids,
            max_cached_prefix_tokens=max_cached_prefix_tokens,
        )

        # A transport that validates requests itself (msgpack ZMQ) marks
        # rejected ones instead of dropping them; admit pre-finished so the
        # client gets a terminal abort rather than a hung stream.
        if getattr(recv_req, "validation_error", None):
            req_state.finished_reason = FINISH_ABORT(
                f"Invalid request: {recv_req.validation_error}"
            )
            return (
                req_spec,
                req_state,
                BootstrapInfo(
                    recv_req.bootstrap_host,
                    recv_req.bootstrap_port,
                    recv_req.bootstrap_room,
                ),
            )

        if (
            recv_req.session_params is not None
            and recv_req.session_params.id is not None
        ):
            req_state.finished_reason = FINISH_ABORT(
                f"Invalid request: session id {recv_req.session_params.id} does not exist"
            )
            return (
                req_spec,
                req_state,
                BootstrapInfo(
                    recv_req.bootstrap_host,
                    recv_req.bootstrap_port,
                    recv_req.bootstrap_room,
                ),
            )

        self._refuse_unsupported_input_logprobs(req_state)
        self._apply_generation_budget(req_spec, req_state)
        return (
            req_spec,
            req_state,
            BootstrapInfo(
                recv_req.bootstrap_host,
                recv_req.bootstrap_port,
                recv_req.bootstrap_room,
            ),
        )

    def _refuse_unsupported_input_logprobs(self, req_state) -> None:
        """Finish a request asking for prompt logprobs with an abort when this
        engine's LM-head layout cannot compute them (``supports_input_logprobs``)."""
        if req_state.wants_input_logprobs and not self.supports_input_logprobs:
            req_state.finished_reason = FINISH_ABORT(
                "Invalid request: prompt logprobs (logprob_start_len) are not "
                "available with --lm-head-tp-size > 1 under attention DP"
            )

    def _apply_generation_budget(self, req_spec, req_state) -> None:
        """Clamp ``max_new_tokens`` to the context; refuse what exceeds this
        engine's per-request budget (``max_new_tokens_budget``), finishing the
        request with an abort instead of admitting work the engine cannot
        complete."""
        req_state.sampling_params.max_new_tokens = min(
            (
                req_state.sampling_params.max_new_tokens
                if req_state.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req_state.prompt_input_ids) - 1,
        )
        req_spec.max_new_tokens = req_state.sampling_params.max_new_tokens
        if (
            self.max_new_tokens_budget is not None
            and req_spec.max_new_tokens > self.max_new_tokens_budget
        ):
            req_state.finished_reason = FINISH_ABORT(
                "Invalid request: this decode engine (--attn-head-tp-size) serves "
                f"at most {self.max_new_tokens_budget} new tokens per request; "
                f"got max_new_tokens={req_spec.max_new_tokens}"
            )

    # ------------------------------------------------------------------
    # Profiling: torch / cuda / viztracer / mem-snapshot / proton, driven
    # by /start_profile and /stop_profile control requests. Proton must be
    # driven from this process (not the frontend): its GPU hooks are
    # per-process and the scheduler subprocess is torn down with SIGKILL,
    # so an atexit-based finalize would never write the profile.
    # ------------------------------------------------------------------

    def init_profiler(self):
        self.torch_profiler = None
        self.profiler_output_dir: str | None = None
        self.profiler_activities: list[str] | None = None
        self.profile_id: str | None = None
        self.profiler_start_forward_ct: int | None = None
        self.profiler_target_forward_ct: int | None = None
        self.profiler_target_prefill_ct: int | None = None
        self.profiler_target_decode_ct: int | None = None
        self.profiler_prefill_ct: int | None = None
        self.profiler_decode_ct: int | None = None
        self.profile_by_stage: bool = False
        self.profile_in_progress: bool = False
        self.viztracer = None

    def init_profile(
        self,
        output_dir: str | None,
        start_step: int | None,
        num_steps: int | None,
        activities: list[str] | None,
        with_stack: bool | None,
        record_shapes: bool | None,
        profile_by_stage: bool,
        profile_id: str,
    ) -> ProfileReqOutput:
        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        if output_dir is None:
            output_dir = envs.TOKENSPEED_PROFILER_DIR.get()
        if activities is None:
            activities = ["CPU", "GPU"]

        # All validation must precede any state mutation: the event loop runs
        # _profile_batch_predicate on every batch, so a rejected request that
        # left partial profiler state behind would crash the scheduler.
        if "PROTON" in activities:
            conflicting = sorted({"GPU", "CUDA_PROFILER"} & set(activities))
            if conflicting:
                return ProfileReqOutput(
                    success=False,
                    message="PROTON cannot be combined with "
                    f"{', '.join(conflicting)}: CUPTI/roctracer supports only "
                    "one GPU profiling client per process.",
                )
            if not proton_available():
                return ProfileReqOutput(
                    success=False,
                    message="Proton is not available: the installed "
                    "tokenspeed-triton does not provide a profiler.",
                )
            if torch.version.hip and "HIP_VISIBLE_DEVICES" in os.environ:
                return ProfileReqOutput(
                    success=False,
                    message="Proton on AMD requires ROCR_VISIBLE_DEVICES; "
                    "unset HIP_VISIBLE_DEVICES before calling /start_profile.",
                )
            if ProfilingState.get().active:
                return ProfileReqOutput(
                    success=False,
                    message="A Proton session is already active in this "
                    "process (e.g. via TOKENSPEED_KERNEL_PROFILE); it cannot "
                    "be controlled through /start_profile.",
                )
        if "EXPERT_LOAD" in activities:
            if self._device is None:
                return ProfileReqOutput(
                    success=False,
                    message="EXPERT_LOAD needs the device handle.",
                )
            if self.server_args.enable_eplb:
                # The rebalance owns the counters: its snapshot zeroes them,
                # so a profile window would be cut at every snapshot.
                return ProfileReqOutput(
                    success=False,
                    message="EXPERT_LOAD cannot run under --enable-eplb: the "
                    "online rebalance snapshots and resets the same load "
                    "counters (its log reports the balancedness it saw).",
                )
            if not expert_load_recording_enabled():
                return ProfileReqOutput(
                    success=False,
                    message="EXPERT_LOAD needs the routing load counters; start "
                    "the server with --expert-distribution-recorder-mode stat.",
                )

        self.profile_by_stage = profile_by_stage
        self.profiler_output_dir = output_dir
        self.torch_profiler_with_stack = with_stack
        self.torch_profiler_record_shapes = record_shapes
        self.profiler_activities = activities
        self.profile_id = profile_id

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.forward_ct + 1)

        if num_steps:
            if self.profile_by_stage:
                self.profiler_target_prefill_ct = num_steps
                self.profiler_target_decode_ct = num_steps
                self.profiler_prefill_ct = 0
                self.profiler_decode_ct = 0
            elif start_step:
                self.profiler_target_forward_ct = (
                    self.profiler_start_forward_ct + num_steps
                )
            else:
                self.profiler_target_forward_ct = self.forward_ct + num_steps
            # The caller will be notified when reaching profiler_target_forward_ct
        else:
            self.profiler_target_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded")

    def start_profile(
        self, stage: ForwardMode | None = None
    ) -> ProfileReqOutput | None:
        stage_str = f" for {stage.name}" if stage else ""
        stage_suffix = f"-{stage.name}" if stage else ""

        activities = self.profiler_activities
        with_stack = self.torch_profiler_with_stack
        record_shapes = self.torch_profiler_record_shapes

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if torchprof_activities:
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
            )
            self.torch_profiler.start()

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(max_entries=100000)

        if "CUDA_PROFILER" in activities:
            torch.cuda.cudart().cudaProfilerStart()

        if "EXPERT_LOAD" in activities:
            # Routing counts every route; the window starts from zero.
            self._device.reset_expert_load()

        if "PROTON" in activities:
            Path(self.profiler_output_dir).mkdir(parents=True, exist_ok=True)
            # Proton appends the output format extension (e.g. ".hatchet").
            proton_output = os.path.join(
                self.profiler_output_dir,
                f"{self.profile_id}-{self.profile_rank_tag}{stage_suffix}.proton",
            )
            try:
                start_profiling(profile_config_from_env(output=proton_output))
            except Exception as exc:
                logger.exception("Failed to start Proton profiling")
                if self.torch_profiler is not None:
                    self.torch_profiler.stop()
                    self.torch_profiler = None
                if "MEM" in activities:
                    torch.cuda.memory._record_memory_history(enabled=None)
                return ProfileReqOutput(
                    success=False,
                    message=f"Failed to start Proton profiling: {exc}",
                )

        if "VIZTRACER" in activities:
            Path(self.profiler_output_dir).mkdir(parents=True, exist_ok=True)
            self.viztracer = VizTracer(
                output_file=os.path.join(
                    self.profiler_output_dir,
                    f"{self.profile_id}-{self.profile_rank_tag}{stage_suffix}.viztracer.json",
                ),
                min_duration=int(
                    os.environ.get("TOKENSPEED_VIZTRACER_MIN_DURATION_US", "100")
                ),
                log_async=True,
            )
            self.viztracer.start()

        if activities:
            if activities != ["CUDA_PROFILER"]:
                logger.info(
                    f"Profiling starts{stage_str!s}. Traces will be saved to: "
                    f"{self.profiler_output_dir!s} (with profile id: "
                    f"{self.profile_id!s})",
                )
            self.profile_in_progress = True

        return ProfileReqOutput(success=True, message="Succeeded")

    def _profile_sync(self) -> None:
        """Rendezvous the attention TP peers without touching the device.

        ``torch.distributed.barrier`` cannot be used here. It prefers
        ``group.bound_device_id`` over its CPU branch, and
        ``DistributedInitializer`` binds ``cuda:N`` to every group -- including
        the gloo ones -- to force eager NCCL init. A gloo barrier therefore
        allocates ``aten.empty`` on CUDA, and these calls run on the
        control-plane thread (stage transitions arrive through
        ``_profile_batch_predicate``), where ``_NoDeviceWork`` rejects exactly
        that. The result was every rank dying mid-profile with "control-plane
        thread ran CUDA factory".

        An all-reduce over a preallocated CPU tensor rendezvouses identically
        and allocates nothing, matching how the loop's other in-round
        collectives are written.
        """
        if self.attn_tp_size == 1:
            return
        torch.distributed.all_reduce(
            self._profile_sync_buf, group=self.attn_tp_cpu_group
        )

    def stop_profile(self, stage: ForwardMode | None = None) -> ProfileReqOutput | None:
        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        Path(self.profiler_output_dir).mkdir(parents=True, exist_ok=True)

        stage_suffix = f"-{stage.name}" if stage else ""
        logger.info(f"Stop profiling{stage_suffix!s}...")

        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            self.torch_profiler.export_chrome_trace(
                os.path.join(
                    self.profiler_output_dir,
                    f"{self.profile_id}-{self.profile_rank_tag}{stage_suffix}.trace.json.gz",
                )
            )
            self._profile_sync()

        if self.profiler_activities is not None and "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.profiler_output_dir,
                f"{self.profile_id}-{self.profile_rank_tag}-memory{stage_suffix}.pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        if "CUDA_PROFILER" in self.profiler_activities:
            torch.cuda.cudart().cudaProfilerStop()

        if "EXPERT_LOAD" in self.profiler_activities:
            # Per-rank, unreduced (a collective here would wait on DP peers);
            # the ranks' records are summed when the directory is consumed.
            record_path = os.path.join(
                self.profiler_output_dir,
                f"{self.profile_id}-{self.profile_rank_tag}{stage_suffix}"
                f"{EXPERT_LOAD_RECORD_SUFFIX}",
            )
            record = self._device.dump_expert_load(record_path)
            logger.info(
                f"Expert load: {int(record['physical_count'].sum())} routes counted "
                f"on EP rank {record['ep_rank']} written to {record_path}; pass the "
                "directory to --init-expert-location to merge every rank's record"
            )

        proton_error: Exception | None = None
        if "PROTON" in self.profiler_activities:
            # Finalizes the session and writes the profile now, while this
            # process is still alive (shutdown is SIGKILL; no atexit).
            try:
                stop_profiling()
            except Exception as exc:
                logger.exception("Failed to finalize Proton profiling")
                proton_error = exc
            finally:
                # Do not reply until every TP peer has finished writing.
                self._profile_sync()

        if "VIZTRACER" in self.profiler_activities and self.viztracer is not None:
            self.viztracer.stop()
            self.viztracer.save()
            self.viztracer = None

        if self.profiler_activities and self.profiler_activities != ["CUDA_PROFILER"]:
            logger.info(
                f"Profiling done. Traces are saved to: {self.profiler_output_dir!s}",
            )

        self.torch_profiler = None
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

        if proton_error is not None:
            return ProfileReqOutput(
                success=False,
                message=f"Failed to finalize Proton profiling: {proton_error}",
            )
        return ProfileReqOutput(success=True, message="Succeeded.")

    def _profile_batch_predicate(self, forward_mode=None):
        """Check and toggle profiling based on forward step count.

        Args:
            forward_mode: Optional ForwardMode for stage-based profiling.
                Not needed for step-count-based profiling.
        """
        if self.profile_by_stage and forward_mode is not None:
            if forward_mode.is_extend_or_mixed():
                if self.profiler_prefill_ct == 0:
                    self.start_profile(forward_mode)
                self.profiler_prefill_ct += 1
                if self.profiler_prefill_ct > self.profiler_target_prefill_ct:
                    if self.profile_in_progress:
                        self.stop_profile(stage=ForwardMode.EXTEND)
            elif forward_mode.is_decode():
                if self.profiler_decode_ct == 0:
                    if self.profile_in_progress:
                        self.stop_profile(ForwardMode.EXTEND)
                    self.start_profile(forward_mode)
                self.profiler_decode_ct += 1
                if self.profiler_decode_ct > self.profiler_target_decode_ct:
                    if self.profile_in_progress:
                        self.stop_profile(stage=ForwardMode.DECODE)
            elif forward_mode.is_idle():
                pass
        else:
            if (
                self.profiler_target_forward_ct
                and self.profiler_target_forward_ct <= self.forward_ct
            ):
                self.stop_profile()
            if (
                self.profiler_start_forward_ct
                and self.profiler_start_forward_ct == self.forward_ct
            ):
                self.start_profile()

    def profile(self, recv_req: ProfileReq):
        if recv_req.type == ProfileReqType.START_PROFILE:
            res = self.init_profile(
                recv_req.output_dir,
                recv_req.start_step,
                recv_req.num_steps,
                recv_req.activities,
                recv_req.with_stack,
                recv_req.record_shapes,
                recv_req.profile_by_stage,
                recv_req.profile_id,
            )
            if not res.success or recv_req.profile_by_stage or recv_req.start_step:
                return res
            return self.start_profile()
        else:
            return self.stop_profile()
