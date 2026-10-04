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
import time
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
)

from tokenspeed.runtime.engine.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    IsSchedulerPausedReqInput,
    IsSchedulerPausedReqOutput,
    IsSleepingReqInput,
    IsSleepingReqOutput,
    PauseMode,
    PauseSchedulerReqInput,
    PauseSchedulerReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    RebalanceExpertsReqInput,
    RebalanceExpertsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ResumeSchedulerReqInput,
    ResumeSchedulerReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromMooncakeReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from tokenspeed.runtime.engine.scheduler_communicator import _Communicator
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.server_args import ServerArgs

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)


def combined_flush_cache_output(
    results: list[FlushCacheReqOutput],
) -> FlushCacheReqOutput:
    """AND every DP replica's flush reply into one frontend result.

    ``/flush_cache`` fans out to ``attn.dp_size`` workers. Object keys omit
    DP rank, so those replicas share the Mooncake namespace. Returning only
    replica 0 would report success after a peer rejected, or hide a peer
    that already deleted the namespace. Worker-side DP MIN still has to
    agree before any rank deletes; this AND is the frontend's matching
    report.
    """

    return FlushCacheReqOutput(
        success=bool(results) and all(result.success for result in results),
    )


def combined_weight_update_output(
    results: Sequence[
        InitWeightsUpdateGroupReqOutput
        | DestroyWeightsUpdateGroupReqOutput
        | UpdateWeightsFromDistributedReqOutput
        | UpdateWeightsFromMooncakeReqOutput
        | RebalanceExpertsReqOutput
    ],
) -> tuple[bool, str]:
    """AND every DP replica's weight-op reply into one frontend result.

    Weight ops fan out to ``attn.dp_size`` workers, each of which replies
    once its own GPU load (or group init/teardown) finished. The scheduler
    already MIN-reduces the outcome across the replica before replying, so
    the replies normally agree; the AND is the frontend's matching report,
    and the message keeps each distinct reply once, in first-seen order, so a
    single failing worker's error is not drowned by the others' success.
    """

    success = bool(results) and all(result.success for result in results)
    messages = list(dict.fromkeys(result.message for result in results))
    return success, " | ".join(messages)


class SchedulerControlClient:
    """Scheduler control-plane client methods for AsyncLLM."""

    def init_communicators(self: AsyncLLM, server_args: ServerArgs):
        # Communicators
        self.init_weights_update_group_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.destroy_weights_update_group_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.update_weights_from_distributed_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.update_weights_from_mooncake_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.rebalance_experts_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.update_weights_from_tensor_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.get_weights_by_name_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.release_memory_occupation_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.flush_cache_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.pause_scheduler_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.resume_scheduler_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.is_scheduler_paused_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.is_sleeping_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.profile_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.get_internal_state_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.set_internal_state_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )

        self._result_dispatcher += self._get_communicator_dispatcher()

    def _get_communicator_dispatcher(self: AsyncLLM):
        return TypeBasedDispatcher(
            [
                (
                    InitWeightsUpdateGroupReqOutput,
                    self.init_weights_update_group_communicator.handle_recv,
                ),
                (
                    DestroyWeightsUpdateGroupReqOutput,
                    self.destroy_weights_update_group_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromDistributedReqOutput,
                    self.update_weights_from_distributed_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromMooncakeReqOutput,
                    self.update_weights_from_mooncake_communicator.handle_recv,
                ),
                (
                    RebalanceExpertsReqOutput,
                    self.rebalance_experts_communicator.handle_recv,
                ),
                (
                    UpdateWeightsFromTensorReqOutput,
                    self.update_weights_from_tensor_communicator.handle_recv,
                ),
                (
                    GetWeightsByNameReqOutput,
                    self.get_weights_by_name_communicator.handle_recv,
                ),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
                (
                    FlushCacheReqOutput,
                    self.flush_cache_communicator.handle_recv,
                ),
                (
                    PauseSchedulerReqOutput,
                    self.pause_scheduler_communicator.handle_recv,
                ),
                (
                    ResumeSchedulerReqOutput,
                    self.resume_scheduler_communicator.handle_recv,
                ),
                (
                    IsSchedulerPausedReqOutput,
                    self.is_scheduler_paused_communicator.handle_recv,
                ),
                (
                    IsSleepingReqOutput,
                    self.is_sleeping_communicator.handle_recv,
                ),
                (
                    ProfileReqOutput,
                    self.profile_communicator.handle_recv,
                ),
                (
                    GetInternalStateReqOutput,
                    self.get_internal_state_communicator.handle_recv,
                ),
                (
                    SetInternalStateReqOutput,
                    self.set_internal_state_communicator.handle_recv,
                ),
            ]
        )

    async def flush_cache(self: AsyncLLM) -> FlushCacheReqOutput:
        results = await self.flush_cache_communicator(FlushCacheReqInput())
        return combined_flush_cache_output(results)

    async def pause_scheduler(self: AsyncLLM, *, mode: PauseMode = "abort") -> bool:
        """Pause generation to allow model weight updates.

        ``mode`` controls in-flight requests: ``"abort"`` cancels them,
        ``"wait"`` lets them finish, ``"keep"`` freezes them for ``/resume``.
        For ``abort``/``wait`` the reply only returns once the scheduler has
        drained, so on return no forward work is in flight.

        Cache invalidation after a weight swap is the weight-update op's
        responsibility (``update_weights_*(flush_cache=...)``), not pause's.
        """
        # Pause may be the very first call (e.g. weight swap before serving),
        # so ensure the output-dispatch loop is running to receive the reply.
        self.auto_create_handle_loop()
        result = (
            await self.pause_scheduler_communicator(PauseSchedulerReqInput(mode=mode))
        )[0]
        return result.success

    async def resume_scheduler(self: AsyncLLM) -> bool:
        """Resume generation after :meth:`pause_scheduler`."""
        self.auto_create_handle_loop()
        result = (await self.resume_scheduler_communicator(ResumeSchedulerReqInput()))[
            0
        ]
        return result.success

    async def is_scheduler_paused(self: AsyncLLM) -> bool:
        """Return whether the scheduler is currently paused."""
        self.auto_create_handle_loop()
        result = (
            await self.is_scheduler_paused_communicator(IsSchedulerPausedReqInput())
        )[0]
        return result.is_paused

    async def start_profile(
        self: AsyncLLM,
        output_dir: str | None = None,
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
        profile_by_stage: bool = False,
        profile_id: str | None = None,
    ):
        self.auto_create_handle_loop()
        env_with_stack = envs.TOKENSPEED_PROFILE_WITH_STACK.get()
        with_stack = not (with_stack is False or env_with_stack is False)
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_by_stage=profile_by_stage,
            profile_id=profile_id or time.strftime("%Y%m%d-%H%M%S"),
        )
        return await self._execute_profile(req)

    async def stop_profile(self: AsyncLLM):
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self: AsyncLLM, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    # Weight ops fan out to every attention-DP worker (the DP controller
    # broadcasts control requests) and the scheduler completes each one only
    # once every DP rank holds it at the head of its queue, so the replies
    # arrive together and are ANDed here. The writer lock keeps generation
    # out while the model is rewritten; group init/teardown rewrite nothing
    # and take no lock, so a trainer's rendezvous does not wait for every
    # in-flight generation to finish.

    async def init_weights_update_group(
        self: AsyncLLM,
        obj: InitWeightsUpdateGroupReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        results = await self.init_weights_update_group_communicator(obj)
        return combined_weight_update_output(results)

    async def destroy_weights_update_group(
        self: AsyncLLM,
        obj: DestroyWeightsUpdateGroupReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        results = await self.destroy_weights_update_group_communicator(obj)
        return combined_weight_update_output(results)

    async def update_weights_from_distributed(
        self: AsyncLLM,
        obj: UpdateWeightsFromDistributedReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        async with self.model_update_lock.writer_lock:
            results = await self.update_weights_from_distributed_communicator(obj)
        return combined_weight_update_output(results)

    async def update_weights_from_mooncake(
        self: AsyncLLM,
        obj: UpdateWeightsFromMooncakeReqInput,
    ) -> tuple[bool, str]:
        """Load one committed Mooncake weight-store version on every worker."""
        self.auto_create_handle_loop()
        async with self.model_update_lock.writer_lock:
            results = await self.update_weights_from_mooncake_communicator(obj)
        return combined_weight_update_output(results)

    async def rebalance_experts(
        self: AsyncLLM,
        obj: RebalanceExpertsReqInput,
    ) -> tuple[bool, str]:
        """Start one online expert rebalance on every worker (``--enable-eplb``).

        Fans out like the weight ops: every attention-DP worker queues the
        request on its same-round gate and replies once the load snapshot was
        taken; the weight moves follow in later rounds. The replies are ANDed.
        """
        self.auto_create_handle_loop()
        results = await self.rebalance_experts_communicator(obj)
        return combined_weight_update_output(results)

    async def update_weights_from_tensor(
        self: AsyncLLM,
        obj: UpdateWeightsFromTensorReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        if self.server_args.mapping.attn.has_dp:
            raise RuntimeError("dp_size must be 1 for update weights from tensor")

        # This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            result = (await self.update_weights_from_tensor_communicator(obj))[0]
            return result.success, result.message

    async def get_weights_by_name(
        self: AsyncLLM,
        obj: GetWeightsByNameReqInput,
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if not self.server_args.mapping.attn.has_dp:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self: AsyncLLM,
        obj: ReleaseMemoryOccupationReqInput,
    ) -> ReleaseMemoryOccupationReqOutput:
        self.auto_create_handle_loop()
        return (await self.release_memory_occupation_communicator(obj))[0]

    async def resume_memory_occupation(
        self: AsyncLLM,
        obj: ResumeMemoryOccupationReqInput,
    ) -> ResumeMemoryOccupationReqOutput:
        self.auto_create_handle_loop()
        return (await self.resume_memory_occupation_communicator(obj))[0]

    async def is_sleeping(self: AsyncLLM) -> bool:
        self.auto_create_handle_loop()
        result = (await self.is_sleeping_communicator(IsSleepingReqInput()))[0]
        return result.is_sleeping

    async def get_internal_state(self: AsyncLLM) -> list[dict[Any, Any]]:
        req = GetInternalStateReq()
        responses: list[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)
        )
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: AsyncLLM, obj: SetInternalStateReq
    ) -> list[bool]:
        responses: list[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)
        )
        return [res.updated for res in responses]

    async def get_load(self: AsyncLLM) -> list[GetLoadReqOutput]:
        self.auto_create_handle_loop()
        return self.load_snapshot_store.project_loads()

    async def load_snapshot_loop(self: AsyncLLM) -> None:
        """Cache scheduler-published snapshots and relay accepted updates to DP."""
        while True:
            snapshot = await self.engine_core_client.recv_load_snapshot.recv_pyobj()
            if not self.load_snapshot_store.accept(snapshot):
                continue
            if (
                self.server_args.mapping.attn.has_dp
                and self.server_args.load_balance_method != "round_robin"
            ):
                await self.engine_core_client.send_to_scheduler.send_pyobj(snapshot)
