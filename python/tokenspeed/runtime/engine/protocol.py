# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

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

"""Serving-facing engine protocol.

``EngineClient`` is the narrow async-engine surface that serving code can type
against instead of the concrete ``AsyncLLM``, which inherits it explicitly
(guarded by ``test/runtime/test_async_llm_protocol.py``). Internal state such
as ``rid_to_state``, ``server_status``, and the ``*_communicator`` attributes
stays off the protocol; callers that need it must hold an ``AsyncLLM``.
"""

from collections.abc import AsyncGenerator
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.engine.io_struct import (
    CloseSessionReqInput,
    ConfigureLoggingReq,
    DestroyWeightsUpdateGroupReqInput,
    EmbeddingReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    RebalanceExpertsReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SetInternalStateReq,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromTensorReqInput,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@runtime_checkable
class EngineClient(Protocol):
    """Serving-facing async engine surface."""

    # Mutable because ``update_weights_from_disk`` reassigns
    # ``served_model_name`` and ``model_path`` on successful reloads.
    server_args: ServerArgs
    model_config: ModelConfig
    tokenizer: Any
    model_path: str
    served_model_name: str
    is_generation: bool
    is_image_gen: bool

    # ``time.time()`` of the last output received from the scheduler.
    last_receive_tstamp: float
    gracefully_exit: bool

    async def generate_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    def abort_request(self, rid: str) -> None: ...

    async def open_session(
        self,
        obj: OpenSessionReqInput,
    ) -> str | None: ...

    async def close_session(
        self,
        obj: CloseSessionReqInput,
    ) -> None: ...

    async def flush_cache(self) -> FlushCacheReqOutput: ...

    def configure_logging(self, obj: ConfigureLoggingReq) -> None: ...

    async def pause_scheduler(self, *, mode: str = "abort") -> bool: ...

    async def resume_scheduler(self) -> bool: ...

    async def is_scheduler_paused(self) -> bool: ...

    def is_server_starting(self) -> bool: ...

    def mark_server_up(self) -> None: ...

    def mark_server_unhealthy(self) -> None: ...

    def drop_request_state(self, rid: str) -> None: ...

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
    ) -> tuple[bool, str, Any]: ...

    async def init_weights_update_group(
        self,
        obj: InitWeightsUpdateGroupReqInput,
    ) -> tuple[bool, str]: ...

    async def destroy_weights_update_group(
        self,
        obj: DestroyWeightsUpdateGroupReqInput,
    ) -> tuple[bool, str]: ...

    async def update_weights_from_distributed(
        self,
        obj: UpdateWeightsFromDistributedReqInput,
    ) -> tuple[bool, str]: ...

    async def update_weights_from_mooncake(
        self,
        obj: UpdateWeightsFromMooncakeReqInput,
    ) -> tuple[bool, str]: ...

    async def rebalance_experts(
        self,
        obj: RebalanceExpertsReqInput,
    ) -> tuple[bool, str]: ...

    async def update_weights_from_tensor(
        self,
        obj: UpdateWeightsFromTensorReqInput,
    ) -> tuple[bool, str]: ...

    async def get_weights_by_name(
        self,
        obj: GetWeightsByNameReqInput,
    ) -> Any: ...

    async def release_memory_occupation(
        self,
        obj: ReleaseMemoryOccupationReqInput,
    ) -> None: ...

    async def resume_memory_occupation(
        self,
        obj: ResumeMemoryOccupationReqInput,
    ) -> None: ...

    async def is_sleeping(self) -> bool: ...

    async def start_profile(
        self,
        output_dir: str | None = None,
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
        profile_by_stage: bool = False,
    ) -> Any: ...

    async def stop_profile(self) -> Any: ...

    async def get_internal_state(self) -> list[dict[Any, Any]]: ...

    async def set_internal_state(self, obj: SetInternalStateReq) -> list[bool]: ...

    async def get_load(self) -> list[GetLoadReqOutput]: ...
