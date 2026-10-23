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

"""The profiler a scheduler process runs on request (``ProfileReq``).

The Rust engine host's forward thread drives it (``engine/rust_host.py``):
the control plane's profile requests and the note of each committed forward
(``note_forward``) reach it in the control plane's order, so ``start_step``
and ``num_steps`` count forwards, and the attention-TP peers rendezvous over
a CPU group around each trace they write.

Profilers: torch (CPU, GPU), CUDA_PROFILER, MEM snapshots, EXPERT_LOAD
counters, viztracer and Proton. Proton must be driven from the scheduler
process, not the frontend: its GPU hooks are per process, and the process
is torn down with SIGKILL, so an atexit-based finalize would never write the
profile.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from tokenspeed_kernel.profiling import (
    ProfilingState,
    profile_config_from_env,
    proton_available,
    start_profiling,
    stop_profiling,
)
from viztracer import VizTracer

from tokenspeed.runtime.engine.io_struct import (
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.moe.expert_location import (
    EXPERT_LOAD_RECORD_SUFFIX,
    expert_load_recording_enabled,
)
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)


def _profile_rank_tag(attn_mapping) -> str:
    """File-name tag identifying this scheduler process's profile outputs."""
    parts = []
    if attn_mapping.has_dp:
        parts.append(f"DP{attn_mapping.dp_rank}")
    parts.append(f"TP{attn_mapping.tp_rank}")
    return "-".join(parts)


def _all_thread_profile_config():
    # The process-global flag also covers workers created before profiling.
    if not isinstance(
        getattr(torch.autograd.profiler, "_is_profiler_enabled", None), bool
    ):
        raise RuntimeError("missing process-global profiler state")
    try:
        return torch.profiler._ExperimentalConfig(profile_all_threads=True)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(str(exc)) from exc


class Profiler:
    """Profiling state and its start/stop, for one scheduler process.

    Args:
        server_args: The server arguments (``enable_eplb`` gates
            ``EXPERT_LOAD``).
        device: The device handle (expert-load counters), or None.
        attn_mapping: The attention mapping, for the trace file names.
        sync_size: Ranks that rendezvous around a trace: attention TP, or
            the world with pipeline parallelism.
        sync_group: Their CPU group (unused with one rank).
    """

    def __init__(
        self, *, server_args, device, attn_mapping, sync_size: int, sync_group
    ) -> None:
        self.server_args = server_args
        self._device = device
        self.forward_ct = 0
        self.attn_tp_size = sync_size
        self.attn_tp_cpu_group = sync_group
        self.profile_rank_tag = _profile_rank_tag(attn_mapping)
        # Rendezvous buffer for _profile_sync, allocated once -- see
        # _profile_sync.
        self._profile_sync_buf = torch.zeros(1, dtype=torch.int32, device="cpu")
        self.init_profiler()

    def note_forward(self, forward_mode: ForwardMode) -> None:
        """Count one committed forward and start or stop the profiler at the
        step a request asked for."""
        self.forward_ct += 1
        self._profile_batch_predicate(forward_mode)

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

        if {"CPU", "GPU"} & set(activities):
            try:
                _all_thread_profile_config()
            except RuntimeError as exc:
                return ProfileReqOutput(
                    success=False,
                    message=f"This PyTorch runtime cannot provide all-thread phase profiling: {exc}",
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
            try:
                experimental_config = _all_thread_profile_config()
            except RuntimeError as exc:
                return ProfileReqOutput(
                    success=False,
                    message=f"This PyTorch runtime cannot provide all-thread phase profiling: {exc}",
                )
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
                experimental_config=experimental_config,
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

        Not ``torch.distributed.barrier``: it prefers
        ``group.bound_device_id`` over its CPU branch, and
        ``DistributedInitializer`` binds ``cuda:N`` to every group -- including
        the gloo ones -- to force eager NCCL init, so a gloo barrier allocates
        ``aten.empty`` on CUDA. An all-reduce over a preallocated CPU tensor
        rendezvouses identically and allocates nothing.
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
