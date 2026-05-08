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

"""AsyncLLM is the main-process async frontend.

Owns request intake, per-request state, scheduler IPC, and the
output-dispatch loop. Inherits from ``EngineClient`` (explicit
structural conformance) and ``SchedulerControlClient`` (scheduler
control-plane helpers).
"""

import asyncio
import copy
import logging
import os
import signal
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable
from enum import Enum
from http import HTTPStatus
from typing import (
    Any,
    Generic,
    TypeVar,
)

import uvloop

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.engine.aio_rwlock import RWLock
from tokenspeed.runtime.engine.collector import RequestOutputCollector
from tokenspeed.runtime.engine.core_client import EngineCoreClient
from tokenspeed.runtime.engine.exceptions import EngineGenerateError
from tokenspeed.runtime.engine.input_processor import InputProcessor
from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    GetLoadReqInput,
    HealthCheckOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    WatchLoadUpdateReq,
)
from tokenspeed.runtime.engine.output_processor import OutputProcessor, ReqState
from tokenspeed.runtime.engine.parallel_sampling import (
    prepare_parallel_sampling_replica,
    prepare_prefix_warmup,
)
from tokenspeed.runtime.engine.protocol import EngineClient
from tokenspeed.runtime.engine.scheduler_control_client import (
    SchedulerControlClient,
)
from tokenspeed.runtime.metrics.collector import RequestMetrics
from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from tokenspeed.runtime.utils import (
    dataclass_to_string_truncated,
    get_colorful_logger,
)
from tokenspeed.runtime.utils.dispatch import TypeBasedDispatcher
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.hf_transformers_utils import get_processor, get_tokenizer
from tokenspeed.runtime.utils.process import kill_process_tree
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = get_colorful_logger(__name__)


def _ignore_health_check_output(_: HealthCheckOutput) -> None:
    return None


class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"
    Crashed = "Crashed"


class AsyncLLM(SchedulerControlClient, EngineClient):
    """Main-process async frontend for the tokenspeed runtime.

    Owns request intake, per-request state, scheduler IPC, and the
    output-dispatch loop. Structurally satisfies :class:`EngineClient`
    via the explicit inheritance declaration above.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.log_requests = server_args.enable_log_requests
        self.log_requests_level = server_args.log_requests_level
        self.logger = logger

        # Init inter-process communication (scheduler IPC owned by EngineCoreClient).
        self.engine_core_client = EngineCoreClient(port_args)

        # Read model args
        self.model_path = server_args.model
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig(
            server_args.model,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.max_model_len,
            model_override_args=server_args.hf_overrides,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            server_args=server_args,
        )

        self.is_generation = self.model_config.is_generation
        self.is_image_gen = self.model_config.is_image_gen
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id
        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    architectures=self.model_config.hf_config.architectures,
                )

        # Store states
        self.no_create_loop = False
        self.rid_to_state: dict[str, ReqState] = {}
        self.gracefully_exit = False
        self.last_receive_tstamp = 0
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: list[tuple] = []
        self.log_request_metadata = self.get_log_request_metadata()
        self.server_status = ServerStatus.Starting

        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.model_update_result: Awaitable[UpdateWeightFromDiskReqOutput] | None = None
        self.asyncio_tasks = set()

        # For session info
        self.session_futures = {}  # session_id -> asyncio event

        # Set after scheduler is initialized
        self.max_req_input_len = None

        self.metrics = RequestMetrics(
            labels={
                "model_name": self.server_args.served_model_name,
                "app_key": self.server_args.app_key,
            },
            enabled=(
                self.enable_metrics
                and "prometheus" in (server_args.metrics_reporters or [])
            ),
        )

        self.output_processor = OutputProcessor(self)

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (
                        BatchStrOut,
                        BatchEmbeddingOut,
                        BatchTokenIDOut,
                        BatchMultimodalOut,
                    ),
                    self.output_processor.handle_batch_output,
                ),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (HealthCheckOutput, _ignore_health_check_output),
            ]
        )

        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        # for disaggregation, start kv bootstrap server on prefill
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # only start bootstrap server on prefill tm
            kv_bootstrap_server_class = get_kv_class(
                self.transfer_backend, KVClassType.BOOTSTRAP_SERVER
            )
            self.bootstrap_server = kv_bootstrap_server_class(
                self.server_args.disaggregation_bootstrap_port
            )

        self.init_communicators(server_args)

        # Tokenization lives in :class:`InputProcessor`; see
        # :meth:`_tokenize_one_request` for the delegation.
        self.input_processor = InputProcessor(self)

    async def generate_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ):
        created_time = time.time()

        self.auto_create_handle_loop()

        self.input_processor.validate_request(obj)

        obj.normalize_batch_and_arguments()

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj=%s",
                dataclass_to_string_truncated(obj, max_length, skip_names=skip_names),
            )

        async with self.model_update_lock.reader_lock:
            is_single = obj.is_single
            if is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(obj, tokenized_obj, created_time)
                async for response in self._wait_one_response(obj):
                    yield response
            else:
                async for response in self._handle_batch_request(obj, created_time):
                    yield response

    async def _tokenize_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ) -> TokenizedGenerateReqInput | TokenizedEmbeddingReqInput:
        """Delegate to :class:`InputProcessor`.

        The tokenization body lives in
        ``InputProcessor.tokenize_one_request``. If the input-side
        surface ever grows (e.g. multimodal routing), it happens in
        that module — not here.
        """
        return await self.input_processor.tokenize_one_request(obj)

    def _send_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        tokenized_obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
        created_time: float | None = None,
    ):
        state = ReqState(
            RequestOutputCollector(),
            False,
            asyncio.Event(),
            obj,
            created_time=created_time,
            tokenized_time=tokenized_obj.created_time,
        )
        self.rid_to_state[obj.rid] = state
        self.engine_core_client.send_to_scheduler.send_pyobj(tokenized_obj)

    async def _wait_one_response(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ):
        """Wait for the response of one request.

        Cancellation contract: callers (FastAPI route handlers, the sync
        ``LLM`` bridge, RL-trainer drivers, etc.) signal client disconnect
        via ``asyncio.CancelledError`` — not via a polled
        ``request.is_disconnected()`` check. If the task driving this
        generator is cancelled mid-wait, the ``finally`` below drops the
        rid from ``rid_to_state`` and fires an ``AbortReq`` at the
        scheduler so no per-request state leaks.
        """
        state = self.rid_to_state[obj.rid]

        try:
            while True:
                await state.event.wait()

                out = state.collector.take()
                if out is None:
                    state.event.clear()
                    continue
                if state.finished:
                    if self.log_requests:
                        max_length, skip_names, out_skip_names = (
                            self.log_request_metadata
                        )
                        if self.model_config.is_multimodal_gen:
                            msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
                        else:
                            if (
                                isinstance(obj, GenerateReqInput)
                                and obj.input_ids is not None
                                and obj.text is None
                            ):
                                if self.tokenizer is not None:
                                    obj.text = self.tokenizer.decode(
                                        obj.input_ids,
                                        skip_special_tokens=getattr(
                                            obj.sampling_params,
                                            "skip_special_tokens",
                                            False,
                                        ),
                                    )
                                else:
                                    obj.text = ""
                            msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                        logger.info(msg)
                    del self.rid_to_state[obj.rid]

                    # Check if this was an abort/error created by scheduler
                    if isinstance(out["meta_info"].get("finish_reason"), dict):
                        finish_reason = out["meta_info"]["finish_reason"]
                        if finish_reason.get("type") == "abort":
                            if (
                                finish_reason.get("status_code")
                                == HTTPStatus.BAD_REQUEST
                            ):
                                raise EngineGenerateError(finish_reason["message"])

                    yield out
                    break

                state.event.clear()
                if state.collector.has_pending():
                    state.event.set()

                if obj.stream:
                    yield out
                # else: non-stream path falls through and waits for the
                # final chunk; external cancellation wakes us via
                # ``asyncio.CancelledError`` from ``state.event.wait()``.
        finally:
            # Idempotent cleanup split on ``state.finished``:
            #
            # * Normal-finish path: the yield loop above already did
            #   ``del self.rid_to_state[obj.rid]`` at ``state.finished``.
            #   We defensively ``pop`` with a default so a second exit
            #   through ``finally`` (e.g. when the yield itself raised
            #   after the del) is a no-op.
            #
            # * Abandoned path (CancelledError / unexpected exception):
            #   the rid is still in ``rid_to_state``. Call
            #   ``abort_request`` which **both** removes it from the
            #   state map **and** sends ``AbortReq`` to the scheduler.
            #   Ordering matters: ``abort_request`` early-returns if
            #   the rid is already gone, so we must not pop first.
            if state.finished:
                self.rid_to_state.pop(obj.rid, None)
            else:
                self.abort_request(obj.rid)

    async def _handle_batch_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        created_time: float | None = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            # Send all requests
            for i in range(batch_size):
                tmp_obj = obj[i]
                tokenized_obj = await self._tokenize_one_request(tmp_obj)
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                generators.append(self._wait_one_response(tmp_obj))
                rids.append(tmp_obj.rid)
        else:
            # Batched parallel sampling still follows a conservative path and
            # can be slower than duplicating requests explicitly.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await self.input_processor.tokenize_batch(objs)

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                warmup_obj = prepare_prefix_warmup(tmp_obj, tokenized_objs[i])
                self._send_one_request(tmp_obj, warmup_obj, created_time)
                await self._wait_one_response(tmp_obj).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    replica_obj = prepare_parallel_sampling_replica(
                        tmp_obj, tokenized_objs[i]
                    )
                    self._send_one_request(tmp_obj, replica_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    async def flush_cache(self) -> FlushCacheReqOutput:
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        self.engine_core_client.send_to_scheduler.send_pyobj(req)

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
    ) -> tuple[bool, str, Any]:
        self.auto_create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        # Hold the lock if it is not async. This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            return await self._wait_for_model_update_from_disk(obj)

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> tuple[bool, str]:
        self.engine_core_client.send_to_scheduler.send_pyobj(obj)
        self.model_update_result = asyncio.Future()
        if not self.server_args.mapping.attn.has_dp:
            result = await self.model_update_result
            if result.success:
                self.served_model_name = obj.model_path
                self.server_args.model = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            return result.success, result.message, result.num_paused_requests
        else:  # self.server_args.mapping.has_attn_dp
            self.model_update_tmp = []
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self.server_args.model = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            all_paused_requests = [r.num_paused_requests for r in result]
            return all_success, all_message, all_paused_requests

    async def open_session(self, obj: OpenSessionReqInput) -> str | None:
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        self.engine_core_client.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(self, obj: CloseSessionReqInput) -> None:
        await self.engine_core_client.send_to_scheduler.send_pyobj(obj)

    async def watch_load_thread(self):
        # Only for dp_controller when dp_size > 1
        if (
            not self.server_args.mapping.attn.has_dp
            or self.server_args.load_balance_method == "round_robin"
        ):
            return

        while True:
            await asyncio.sleep(self.server_args.load_watch_interval)
            loads = await self.get_load_communicator(GetLoadReqInput())
            load_udpate_req = WatchLoadUpdateReq(loads=loads)
            self.engine_core_client.send_to_scheduler.send_pyobj(load_udpate_req)

    def get_log_request_metadata(self):
        max_length = None
        skip_names = None
        out_skip_names = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                    ]
                )
            elif self.log_requests_level == 1:
                max_length = 2048
            elif self.log_requests_level == 2:
                max_length = 1 << 30
            else:
                raise ValueError(
                    f"Invalid --log-requests-level: {self.log_requests_level=}"
                )
        return max_length, skip_names, out_skip_names

    def configure_logging(self, obj: ConfigureLoggingReq):
        if obj.log_requests is not None:
            self.log_requests = obj.log_requests
        if obj.log_requests_level is not None:
            self.log_requests_level = obj.log_requests_level
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        logging.info("Config logging: obj=%r", obj)
        self.log_request_metadata = self.get_log_request_metadata()

    # ---- Server lifecycle / health -------------------------------
    # Intent-revealing wrappers around the private ``server_status``
    # field. Callers (notably ``http_server.py``) drive transitions
    # through these methods so the ``ServerStatus`` enum and the
    # attribute name stay implementation-private.

    def is_server_starting(self) -> bool:
        return self.server_status == ServerStatus.Starting

    def mark_server_up(self) -> None:
        self.server_status = ServerStatus.Up

    def mark_server_unhealthy(self) -> None:
        self.server_status = ServerStatus.UnHealthy

    def drop_request_state(self, rid: str) -> None:
        """Discard the per-request state for ``rid`` if present.

        Used by health probes that synthesize a request, await one
        token through ``generate_request``, and then need to clean
        up the state slot regardless of whether the probe succeeded
        or timed out.
        """
        self.rid_to_state.pop(rid, None)

    def auto_create_handle_loop(self):
        if self.no_create_loop:
            return

        self.no_create_loop = True
        loop = asyncio.get_event_loop()
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )

        # We cannot add signal handler when the tokenizer manager is not in
        # the main thread due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = SignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.signal_handler)
        else:
            logger.warning(
                "Signal handler is not added because the tokenizer manager is "
                "not in the main thread. This disables graceful shutdown of the "
                "tokenizer manager when SIGTERM is received."
            )
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.watch_load_thread))
        )

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            logger.info(
                "Gracefully exiting... remaining number of requests %s", remain_num_req
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj = await self.engine_core_client.recv_from_detokenizer.recv_pyobj()
            self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = time.time()

    def _handle_open_session_req_output(self, recv_obj):
        self.session_futures[recv_obj.session_id].set_result(
            recv_obj.session_id if recv_obj.success else None
        )

    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        if not self.server_args.mapping.attn.has_dp:
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.mapping.has_attn_dp
            self.model_update_tmp.append(recv_obj)
            # set future if the all results are received
            if len(self.model_update_tmp) == self.server_args.mapping.attn.dp_size:
                self.model_update_result.set_result(self.model_update_tmp)


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("AsyncLLM hit an exception: %s", traceback)
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


class SignalHandler:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def signal_handler(self, signum=None, frame=None):
        logger.warning(
            "SIGTERM received. signum=%r frame=%r. Draining requests and shutting down...",
            signum,
            frame,
        )
        self.tokenizer_manager.gracefully_exit = True


T = TypeVar("T")


class _Communicator(Generic[T]):
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._result_event: asyncio.Event | None = None
        self._result_values: list[T] | None = None
        self._ready_queue: deque[asyncio.Future] = deque()

    async def __call__(self, obj):
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

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()
