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

``EngineClient`` is the narrow surface that the OpenAI serving layer
and ``http_server.py`` are allowed to depend on, letting callers stop
typing against the concrete ``AsyncLLM`` class.
``AsyncLLM(SchedulerControlClient, EngineClient)`` inherits the protocol so the
conformance is a class-definition-time invariant rather than duck
typing at every call site. The protocol stays ``@runtime_checkable`` so
``isinstance(engine, EngineClient)`` remains a lightweight check
(exercised by ``test/runtime/test_async_llm_protocol.py``).

What's on the protocol
----------------------
The surface covers the calls that the serving layer actually makes:
the generate/embed/abort/attribute path plus the administrative
RPCs (weights, cache, session, profile, expert-distribution,
load-query, internal-state, logging config).

What's intentionally off the protocol
-------------------------------------
Two categories stay concrete and are accessed via a narrower type
(or via ``isinstance`` casts in the caller):

1. Attribute escape hatches — ``rid_to_state``, ``server_status``.
   ``http_server.py`` reads them directly for liveness / health
   reasons that are out of scope for the serving-facing protocol.

2. Purely internal coordination state — ``model_update_lock``,
   ``session_futures``, ``flush_cache_communicator`` and the other
   ``SchedulerControlClient`` ``*_communicator`` attributes.

If a caller needs any of the above, it must hold a concrete
``AsyncLLM`` reference, not an ``EngineClient``-typed one. This is
deliberate.
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
    EmbeddingReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    OpenSessionReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    SetInternalStateReq,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@runtime_checkable
class EngineClient(Protocol):
    """Serving-facing async engine surface.

    ``AsyncLLM(SchedulerControlClient, EngineClient)`` inherits
    this protocol explicitly, so conformance is structurally
    guaranteed at class-definition time rather than relying on
    duck typing at every call site.
    """

    # ---- Configuration / identity ---------------------------------
    # Mutable because ``update_weights_from_disk`` reassigns
    # ``served_model_name`` and ``model_path`` on successful reloads
    # (see ``_wait_for_model_update_from_disk``). Typed as their
    # current runtime shape.
    server_args: ServerArgs
    model_config: ModelConfig
    tokenizer: Any
    model_path: str
    served_model_name: str
    is_generation: bool
    is_image_gen: bool

    # ---- Liveness state -------------------------------------------
    # Monotonic epoch-second timestamp of the last message received
    # from the scheduler's shared output socket. ``http_server.py``
    # reads this for health / idle-timeout logic.
    last_receive_tstamp: float
    gracefully_exit: bool

    # ---- Generate / embed path ------------------------------------

    async def generate_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ) -> AsyncGenerator[dict[str, Any], None]: ...

    def abort_request(self, rid: str) -> None: ...

    # ---- Session management --------------------------------------

    async def open_session(
        self,
        obj: OpenSessionReqInput,
    ) -> str | None: ...

    async def close_session(
        self,
        obj: CloseSessionReqInput,
    ) -> None: ...

    # ---- Cache / logging config ----------------------------------

    async def flush_cache(self) -> FlushCacheReqOutput: ...

    def configure_logging(self, obj: ConfigureLoggingReq) -> None: ...

    # ---- Pause / resume (RLHF weight-update control) --------------

    async def pause_scheduler(self, *, mode: str = "abort") -> bool: ...

    async def resume_scheduler(self) -> bool: ...

    async def is_scheduler_paused(self) -> bool: ...

    # ---- Server lifecycle / health -------------------------------

    def is_server_starting(self) -> bool: ...

    def mark_server_up(self) -> None: ...

    def mark_server_unhealthy(self) -> None: ...

    def drop_request_state(self, rid: str) -> None: ...

    # ---- Weight-update RPCs --------------------------------------

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
    ) -> tuple[bool, str, Any]: ...

    async def init_weights_update_group(
        self,
        obj: InitWeightsUpdateGroupReqInput,
    ) -> tuple[bool, str]: ...

    async def update_weights_from_distributed(
        self,
        obj: UpdateWeightsFromDistributedReqInput,
    ) -> tuple[bool, str]: ...

    async def update_weights_from_tensor(
        self,
        obj: UpdateWeightsFromTensorReqInput,
    ) -> tuple[bool, str]: ...

    async def get_weights_by_name(
        self,
        obj: GetWeightsByNameReqInput,
    ) -> Any: ...

    # ---- Memory occupation RPCs ----------------------------------

    async def release_memory_occupation(
        self,
        obj: ReleaseMemoryOccupationReqInput,
    ) -> None: ...

    async def resume_memory_occupation(
        self,
        obj: ResumeMemoryOccupationReqInput,
    ) -> None: ...

    async def is_sleeping(self) -> bool: ...

    # ---- Profiling / expert distribution -------------------------

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

    async def start_expert_distribution_record(self) -> None: ...

    async def stop_expert_distribution_record(self) -> None: ...

    async def dump_expert_distribution_record(self) -> None: ...

    # ---- Engine internal state -----------------------------------

    async def get_internal_state(self) -> list[dict[Any, Any]]: ...

    async def set_internal_state(self, obj: SetInternalStateReq) -> list[bool]: ...

    async def get_load(self) -> list[GetLoadReqOutput]: ...
