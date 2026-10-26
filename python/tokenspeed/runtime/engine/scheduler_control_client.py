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
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    IsSchedulerPausedReqInput,
    IsSleepingReqInput,
    PauseMode,
    PauseSchedulerReqInput,
    ProfileReq,
    ProfileReqType,
    RebalanceExpertsReqInput,
    RebalanceExpertsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ResumeSchedulerReqInput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromMooncakeReqOutput,
    UpdateWeightsFromTensorReqInput,
)
from tokenspeed.runtime.utils.env import envs

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
    """Control operations for AsyncLLM: each runs on every engine (one per
    attention-DP rank) through ``AsyncLLM._control``, which returns every
    engine's reply."""

    async def flush_cache(self: AsyncLLM) -> FlushCacheReqOutput:
        results = await self._control(FlushCacheReqInput())
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
        result = (await self._control(PauseSchedulerReqInput(mode=mode)))[0]
        return result.success

    async def resume_scheduler(self: AsyncLLM) -> bool:
        """Resume generation after :meth:`pause_scheduler`."""
        self.auto_create_handle_loop()
        result = (await self._control(ResumeSchedulerReqInput()))[0]
        return result.success

    async def is_scheduler_paused(self: AsyncLLM) -> bool:
        """Return whether the scheduler is currently paused."""
        self.auto_create_handle_loop()
        result = (await self._control(IsSchedulerPausedReqInput()))[0]
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
        result = (await self._control(req))[0]
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
        results = await self._control(obj)
        return combined_weight_update_output(results)

    async def destroy_weights_update_group(
        self: AsyncLLM,
        obj: DestroyWeightsUpdateGroupReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        results = await self._control(obj)
        return combined_weight_update_output(results)

    async def update_weights_from_distributed(
        self: AsyncLLM,
        obj: UpdateWeightsFromDistributedReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        async with self.model_update_lock.writer_lock:
            results = await self._control(obj)
        return combined_weight_update_output(results)

    async def update_weights_from_mooncake(
        self: AsyncLLM,
        obj: UpdateWeightsFromMooncakeReqInput,
    ) -> tuple[bool, str]:
        """Load one committed Mooncake weight-store version on every worker."""
        self.auto_create_handle_loop()
        async with self.model_update_lock.writer_lock:
            results = await self._control(obj)
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
        results = await self._control(obj)
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
            result = (await self._control(obj))[0]
            return result.success, result.message

    async def get_weights_by_name(
        self: AsyncLLM,
        obj: GetWeightsByNameReqInput,
    ):
        raise NotImplementedError(
            "get_weights_by_name is not served: no engine handles it"
        )

    async def release_memory_occupation(
        self: AsyncLLM,
        obj: ReleaseMemoryOccupationReqInput,
    ) -> ReleaseMemoryOccupationReqOutput:
        self.auto_create_handle_loop()
        return (await self._control(obj))[0]

    async def resume_memory_occupation(
        self: AsyncLLM,
        obj: ResumeMemoryOccupationReqInput,
    ) -> ResumeMemoryOccupationReqOutput:
        self.auto_create_handle_loop()
        return (await self._control(obj))[0]

    async def is_sleeping(self: AsyncLLM) -> bool:
        self.auto_create_handle_loop()
        result = (await self._control(IsSleepingReqInput()))[0]
        return result.is_sleeping

    async def get_internal_state(self: AsyncLLM) -> list[dict[Any, Any]]:
        req = GetInternalStateReq()
        responses: list[GetInternalStateReqOutput] = await self._control(req)
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: AsyncLLM, obj: SetInternalStateReq
    ) -> list[bool]:
        responses: list[SetInternalStateReqOutput] = await self._control(obj)
        return [res.updated for res in responses]

    async def get_load(self: AsyncLLM) -> list[GetLoadReqOutput]:
        """Each engine's latest load, as it reported it with its outputs."""
        self.auto_create_handle_loop()
        return [
            GetLoadReqOutput(
                dp_rank=dp_rank,
                num_reqs=num_running + num_waiting,
                num_waiting_reqs=num_waiting,
                num_pages=kv_active_pages,
                num_total_pages=kv_total_pages,
            )
            for dp_rank, (
                num_running,
                num_waiting,
                kv_active_pages,
                kv_total_pages,
            ) in enumerate(self.frontend.loads())
        ]
