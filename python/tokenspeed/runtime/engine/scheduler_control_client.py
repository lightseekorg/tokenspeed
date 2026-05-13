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

import asyncio
import copy
import logging
import time
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

import zmq

from tokenspeed.runtime.engine.io_struct import (
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    LoadLoraReqInput,
    LoadLoraReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    UnloadLoraReqInput,
    UnloadLoraReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.server_args import ServerArgs

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _Communicator(Generic[T]):
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: asyncio.Event | None = None
        self._result_values: list[T] | None = None
        self._ready_queue: deque[asyncio.Future] = deque()

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj:
            self._sender.send_pyobj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj:
                self._sender.send_pyobj(obj)

        await self._result_event.wait()
        result_values = copy.deepcopy(self._result_values)
        self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj):
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()


class SchedulerControlClient:
    """Scheduler control-plane client methods for AsyncLLM."""

    def init_communicators(self: AsyncLLM, server_args: ServerArgs):
        # Communicators
        self.init_weights_update_group_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.update_weights_from_distributed_communicator = _Communicator(
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
        self.profile_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.get_internal_state_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.set_internal_state_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )

        self.expert_distribution_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.get_load_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler,
            server_args.mapping.attn.dp_size,
            mode="watching",
        )
        self.load_lora_communicator = _Communicator(
            self.engine_core_client.send_to_scheduler, server_args.mapping.attn.dp_size
        )
        self.unload_lora_communicator = _Communicator(
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
                    UpdateWeightsFromDistributedReqOutput,
                    self.update_weights_from_distributed_communicator.handle_recv,
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
                (
                    ExpertDistributionReqOutput,
                    self.expert_distribution_communicator.handle_recv,
                ),
                (
                    GetLoadReqOutput,
                    self.get_load_communicator.handle_recv,
                ),
                (
                    LoadLoraReqOutput,
                    self.load_lora_communicator.handle_recv,
                ),
                (
                    UnloadLoraReqOutput,
                    self.unload_lora_communicator.handle_recv,
                ),
            ]
        )

    async def load_lora_adapter(
        self: "AsyncLLM",
        lora_name: str,
        lora_path: str,
        pinned: bool = False,
    ) -> tuple[bool, int, str]:
        """Send a LoadLoraReqInput to the scheduler subprocess."""
        result = (
            await self.load_lora_communicator(
                LoadLoraReqInput(
                    lora_name=lora_name, lora_path=lora_path, pinned=pinned
                )
            )
        )[0]
        return result.success, result.lora_id, result.message

    async def unload_lora_adapter(self: "AsyncLLM", lora_name: str) -> tuple[bool, str]:
        """Send an UnloadLoraReqInput to the scheduler subprocess."""
        result = (
            await self.unload_lora_communicator(UnloadLoraReqInput(lora_name=lora_name))
        )[0]
        return result.success, result.message

    async def flush_cache(self: AsyncLLM) -> FlushCacheReqOutput:
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

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
        with_stack = False if with_stack is False or env_with_stack is False else True
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

    async def start_expert_distribution_record(self: AsyncLLM):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.START_RECORD)

    async def stop_expert_distribution_record(self: AsyncLLM):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.STOP_RECORD)

    async def dump_expert_distribution_record(self: AsyncLLM):
        self.auto_create_handle_loop()
        await self.expert_distribution_communicator(ExpertDistributionReq.DUMP_RECORD)

    async def init_weights_update_group(
        self: AsyncLLM,
        obj: InitWeightsUpdateGroupReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            not self.server_args.mapping.attn.has_dp
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.init_weights_update_group_communicator(obj))[0]
        return result.success, result.message

    async def update_weights_from_distributed(
        self: AsyncLLM,
        obj: UpdateWeightsFromDistributedReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            not self.server_args.mapping.attn.has_dp
        ), "dp_size must be for update weights from distributed"

        # This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            result = (await self.update_weights_from_distributed_communicator(obj))[0]
            return result.success, result.message

    async def update_weights_from_tensor(
        self: AsyncLLM,
        obj: UpdateWeightsFromTensorReqInput,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            not self.server_args.mapping.attn.has_dp
        ), "dp_size must be for update weights from distributed"

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
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self: AsyncLLM,
        obj: ResumeMemoryOccupationReqInput,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

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
        req = GetLoadReqInput()
        return await self.get_load_communicator(req)
