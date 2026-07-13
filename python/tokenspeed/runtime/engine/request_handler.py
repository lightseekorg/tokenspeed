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
from pathlib import Path
from typing import TYPE_CHECKING

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
    GetLoadReqInput,
    GetLoadReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    IsSchedulerPausedReqInput,
    IsSleepingReqInput,
    PauseSchedulerReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    ResumeSchedulerReqInput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedGenerateReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
)
from tokenspeed.runtime.engine.request_types import FINISH_ABORT
from tokenspeed.runtime.engine.scheduler_utils import make_spec
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.grammar.grammar_manager import GrammarManager
from tokenspeed.runtime.multimodal.shm_transport import sync_shm_features
from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo
from tokenspeed.runtime.utils import broadcast_pyobj
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _profile_rank_tag(attn_mapping) -> str:
    """File-name tag identifying this scheduler process's profile outputs."""
    parts = []
    if attn_mapping.has_dp:
        parts.append(f"DP{attn_mapping.dp_rank}")
    if attn_mapping.has_cp:
        parts.append(f"CP{attn_mapping.cp_rank}")
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
        get_load_fn=None,
        architectures: list[str] | None = None,
        pause_controller=None,
        memory_controller=None,
        model_runner=None,
    ) -> None:

        self.forward_ct = 0
        self.server_args = server_args
        # Owns pause/resume state; shared with the event loop. See pause.py.
        self.pause_controller = pause_controller
        # Owns release/resume_memory_occupation (data plane). See
        # memory_occupation.py. Shares the pause controller's drain machinery.
        self.memory_controller = memory_controller
        # ModelRunner for in-place RL weight sync (NCCL group init + receive).
        # The scheduler worker passes it in; None elsewhere (e.g. unit tests).
        self.model_runner = model_runner

        mapping = server_args.mapping
        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_rank = mapping.attn.tp_rank
        self.attn_tp_cpu_group = pg_manager.get_process_group(
            "gloo", mapping.attn.tp_group
        )
        self.attn_tp_src_rank = mapping.attn.tp_group[0]
        self.profile_rank_tag = _profile_rank_tag(mapping.attn)

        self.hf_eos_token_id = hf_eos_token_id
        self.max_req_len = max_req_len
        self.vocab_size = vocab_size
        self.get_load_fn = get_load_fn

        self.tokenizer = get_tokenizer(
            server_args.tokenizer,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            architectures=architectures,
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

    def recv_reqs(self) -> list:
        """Receive results at attn_tp_rank = 0 and broadcast it to all other TP ranks."""
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

        if self.attn_tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.attn_tp_rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_src_rank,
            )

        if recv_reqs:
            sync_shm_features(recv_reqs, self.attn_tp_cpu_group, self.attn_tp_size)

        return recv_reqs

    def process_requests(self, recv_reqs: list):
        """Dispatch control requests and return new generate request specs and states."""
        new_req_specs, req_states, bootstrap_infos, abort_rids = [], [], [], []
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
                logger.debug("AbortReq for rid=%s", recv_req.rid)
                abort_rids.append(recv_req.rid)
            elif isinstance(recv_req, FlushCacheReqInput):
                # Prefix cache is owned by the scheduler path; acknowledge the
                # control request here so API callers still get a typed reply.
                self.send_func.send_pyobj(FlushCacheReqOutput(success=True))
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
            elif isinstance(recv_req, GetLoadReqInput):
                if self.get_load_fn is not None:
                    self.send_func.send_pyobj(self.get_load_fn())
                else:
                    self.send_func.send_pyobj(GetLoadReqOutput())
            elif isinstance(recv_req, InitWeightsUpdateGroupReqInput):
                # RL weight sync: join the trainer's NCCL group on this worker.
                ok, msg = self.model_runner.init_weights_update_group(recv_req)
                self.send_func.send_pyobj(
                    InitWeightsUpdateGroupReqOutput(success=ok, message=msg)
                )
            elif isinstance(recv_req, UpdateWeightsFromDistributedReqInput):
                # RL weight sync: receive broadcast weights + load into the model.
                ok, msg = self.model_runner.update_weights_from_distributed(recv_req)
                self.send_func.send_pyobj(
                    UpdateWeightsFromDistributedReqOutput(success=ok, message=msg)
                )
            elif isinstance(recv_req, DestroyWeightsUpdateGroupReqInput):
                # RL weight sync: tear down the trainer's NCCL group on this worker.
                ok, msg = self.model_runner.destroy_weights_update_group(recv_req)
                self.send_func.send_pyobj(
                    DestroyWeightsUpdateGroupReqOutput(success=ok, message=msg)
                )
            else:
                raise NotImplementedError(f"Unsupported request type: {type(recv_req)}")
        return new_req_specs, req_states, bootstrap_infos, abort_rids

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        if recv_req.bootstrap_port is None:
            recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

        req_spec = make_spec(
            rid=recv_req.rid,
            tokens=recv_req.input_ids,
        )
        req_state = RequestState.from_recv_req(
            recv_req,
            tokenizer=self.tokenizer,
            eos_token_ids=self.hf_eos_token_id,
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

        req_state.sampling_params.max_new_tokens = min(
            (
                req_state.sampling_params.max_new_tokens
                if req_state.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req_state.prompt_input_ids) - 1,
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
            if ProfilingState.get().active:
                return ProfileReqOutput(
                    success=False,
                    message="A Proton session is already active in this "
                    "process (e.g. via TOKENSPEED_KERNEL_PROFILE); it cannot "
                    "be controlled through /start_profile.",
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

        if "PROTON" in activities:
            Path(self.profiler_output_dir).mkdir(parents=True, exist_ok=True)
            # Proton appends the output format extension (e.g. ".hatchet").
            proton_output = os.path.join(
                self.profiler_output_dir,
                f"{self.profile_id}-{self.profile_rank_tag}{stage_suffix}.proton",
            )
            start_profiling(profile_config_from_env(output=proton_output))

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
                    "Profiling starts%s. Traces will be saved to: %s (with profile id: %s)",
                    stage_str,
                    self.profiler_output_dir,
                    self.profile_id,
                )
            self.profile_in_progress = True

        return ProfileReqOutput(success=True, message="Succeeded")

    def stop_profile(self, stage: ForwardMode | None = None) -> ProfileReqOutput | None:
        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        Path(self.profiler_output_dir).mkdir(parents=True, exist_ok=True)

        stage_suffix = f"-{stage.name}" if stage else ""
        logger.info("Stop profiling%s...", stage_suffix)

        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            self.torch_profiler.export_chrome_trace(
                os.path.join(
                    self.profiler_output_dir,
                    f"{self.profile_id}-{self.profile_rank_tag}{stage_suffix}.trace.json.gz",
                )
            )
            torch.distributed.barrier(self.attn_tp_cpu_group)

        if self.profiler_activities is not None and "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.profiler_output_dir,
                f"{self.profile_id}-{self.profile_rank_tag}-memory{stage_suffix}.pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        if "CUDA_PROFILER" in self.profiler_activities:
            torch.cuda.cudart().cudaProfilerStop()

        if "PROTON" in self.profiler_activities:
            # Finalizes the session and writes the profile now, while this
            # process is still alive (shutdown is SIGKILL; no atexit).
            stop_profiling()
            torch.distributed.barrier(self.attn_tp_cpu_group)

        if "VIZTRACER" in self.profiler_activities and self.viztracer is not None:
            self.viztracer.stop()
            self.viztracer.save()
            self.viztracer = None

        if self.profiler_activities and self.profiler_activities != ["CUDA_PROFILER"]:
            logger.info(
                "Profiling done. Traces are saved to: %s", self.profiler_output_dir
            )

        self.torch_profiler = None
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

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
