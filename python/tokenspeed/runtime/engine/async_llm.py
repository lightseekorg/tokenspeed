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

Owns request intake (tokenization and validation), per-request state and
the output-dispatch loop; the engine ranks, Rust hosts, are reached through
``tokenspeed_engine.Frontend`` over the msgpack wire. The frontend renders
their token outputs (detokenization, stop strings, logprob dialects).
Inherits from ``EngineClient`` (explicit structural conformance) and
``SchedulerControlClient`` (control operations).
"""

import asyncio
import copy
import logging
import os
import signal
import sys
import threading
import time
from enum import Enum
from http import HTTPStatus
from typing import (
    Any,
)

import uvloop

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.aio_rwlock import RWLock
from tokenspeed.runtime.engine.collector import RequestOutputCollector
from tokenspeed.runtime.engine.exceptions import EngineGenerateError
from tokenspeed.runtime.engine.input_processor import InputProcessor
from tokenspeed.runtime.engine.io_struct import (
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    GenerateReqInput,
    IpcSender,
    MsgpackEncoder,
    OpenSessionReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
)
from tokenspeed.runtime.engine.output_processor import (
    EngineStream,
    OutputProcessor,
    ReqState,
)
from tokenspeed.runtime.engine.parallel_sampling import (
    prepare_parallel_sampling_replica,
    prepare_prefix_warmup,
)
from tokenspeed.runtime.engine.protocol import EngineClient
from tokenspeed.runtime.engine.scheduler_control_client import (
    SchedulerControlClient,
)
from tokenspeed.runtime.entrypoints import rl_control
from tokenspeed.runtime.metrics.collector import RequestMetrics
from tokenspeed.runtime.metrics.relay import MetricsRelay
from tokenspeed.runtime.metrics.service import metrics_service
from tokenspeed.runtime.utils import (
    dataclass_to_string_truncated,
    get_colorful_logger,
)
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer
from tokenspeed.runtime.utils.process import kill_process_tree
from tokenspeed.runtime.utils.server_args import ServerArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = get_colorful_logger(__name__)


class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"
    Crashed = "Crashed"


class AsyncLLM(SchedulerControlClient, EngineClient):
    """Main-process async frontend for the tokenspeed runtime.

    Owns request intake, per-request state and the output-dispatch loop.
    Structurally satisfies :class:`EngineClient` via the explicit
    inheritance declaration above.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        frontend,
        encode_sender: IpcSender | None,
    ):
        """
        Args:
            server_args: The engine's server arguments.
            frontend: The connected engine ranks
                (``tokenspeed_engine.Frontend``); None on an EPD encode
                server, which runs no engine ranks.
            encode_sender: On an EPD encode server, the channel to its
                encode ranks (``submit_encode``); None elsewhere.
        """
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.log_requests = server_args.enable_log_requests
        self.log_requests_level = server_args.log_requests_level
        self.logger = logger

        self.frontend = frontend
        self._encode_sender = encode_sender
        self._wire_encoder = MsgpackEncoder()
        # The prefill role's KV bootstrap server, which the launcher starts;
        # closed with the engine.
        self.bootstrap_server = None
        # The node's metrics relay, which runs the output plugins for the
        # engine hosts; the launcher starts it, and it closes with the engine.
        self.metrics_relay: MetricsRelay | None = None

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
        # Create tokenizer. The engine never preprocesses images -- the SMG
        # gateway ships precomputed multimodal inputs -- so even multimodal
        # models only need the tokenizer, not the full HF AutoProcessor.
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                architectures=self.model_config.hf_config.architectures,
                **self.model_config.tokenizer_kwargs,
            )
            if self.model_config.is_multimodal:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

        # Held for writing while a weight update rewrites the model.
        self.model_update_lock = RWLock()
        self.asyncio_tasks = set()
        # In-flight aborts sent to the engines (``_abort_on_engine``).
        self._abort_tasks: set[asyncio.Task] = set()

        # Frontend admission barrier paired with the scheduler's native pause.
        # It prevents requests buffered by a paused scheduler from holding the
        # model-update reader lock and deadlocking a weight-update writer.
        self._generation_admit = asyncio.Event()
        self._generation_admit.set()

        # The SGLang-compatible RL control app is launched lazily on this
        # object's event loop (see auto_create_handle_loop).
        self._rl_control_task = None

        # Set after scheduler is initialized
        self.max_req_input_len = None
        self.max_single_request_tokens = None
        # Whether the engine can serve prompt (input) logprobs; None until the
        # scheduler reported it, which the ingress treats as "cannot".
        self.supports_prompt_logprobs: bool | None = None

        self.metrics = RequestMetrics(
            labels={
                "model_name": self.server_args.served_model_name,
                "app_key": self.server_args.app_key,
            },
            enabled=self.enable_metrics,
            # Caps the request-token histogram grids (the 1-2-5 series).
            max_model_len=self.context_len,
            reporters=server_args.metrics_reporters,
            event_sink=metrics_service.get_event_sink(),
        )

        self.output_processor = OutputProcessor(self)

        # Tokenization lives in :class:`InputProcessor`; see
        # :meth:`_tokenize_one_request` for the delegation.
        self.input_processor = InputProcessor(self)

    async def generate_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ):
        created_time = time.time()

        self.auto_create_handle_loop()

        await self._generation_admit.wait()
        self.input_processor.validate_request(obj)

        obj.normalize_batch_and_arguments()

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj="
                f"{dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)!s}",
            )

        async with self.model_update_lock.reader_lock:
            is_single = obj.is_single
            if is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                await self._send_one_request(obj, tokenized_obj, created_time)
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

    async def _send_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        tokenized_obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
        created_time: float | None = None,
    ):
        if not isinstance(tokenized_obj, TokenizedGenerateReqInput):
            raise ValueError("this engine serves generate requests only")
        if self.frontend is None:
            raise ValueError("an EPD encode server serves no generate requests")
        sampling_params = tokenized_obj.sampling_params
        stop_strs = tuple(sampling_params.stop_strs or ())
        if stop_strs and self.tokenizer is None:
            raise ValueError(
                "stop strings are matched on detokenized text, and this engine "
                "runs without a tokenizer (--skip-tokenizer-init); send "
                "stop_token_ids"
            )
        state = ReqState(
            RequestOutputCollector(),
            False,
            asyncio.Event(),
            obj,
            created_time=created_time,
            tokenized_time=tokenized_obj.created_time,
            stream=EngineStream(
                prompt_ids=list(
                    tokenized_obj.input_ids_unpadded
                    if tokenized_obj.input_ids_unpadded is not None
                    else tokenized_obj.input_ids
                ),
                stop_strs=stop_strs,
                stop_str_max_len=sampling_params.stop_str_max_len,
                skip_special_tokens=sampling_params.skip_special_tokens,
                spaces_between_special_tokens=(
                    sampling_params.spaces_between_special_tokens
                ),
                no_stop_trim=sampling_params.no_stop_trim,
            ),
        )
        self.rid_to_state[obj.rid] = state
        mm_inputs = tokenized_obj.multimodal_inputs
        if mm_inputs is not None:
            mm_inputs.publish_shm_features()
        try:
            await self.frontend.generate(self._wire_frames(tokenized_obj))
        except BaseException:
            self.rid_to_state.pop(obj.rid, None)
            raise

    def _wire_frames(self, tokenized_obj: TokenizedGenerateReqInput) -> list[bytes]:
        """The request as the engines take it: one sample (the frontend fans
        out ``n``) and no stop strings, which the frontend matches, so it
        streams every token when it has any."""
        params = copy.copy(tokenized_obj.sampling_params)
        stop_strs = params.stop_strs
        params.n = 1
        params.stop = None
        params.stop_strs = []
        params.stop_str_max_len = 0
        wire = copy.copy(tokenized_obj)
        wire.sampling_params = params
        wire.stream = tokenized_obj.stream or bool(stop_strs)
        return [bytes(frame) for frame in self._wire_encoder.encode(wire)]

    def submit_encode(self, encode_request) -> None:
        """Send an EPD encode request to this encode server's ranks.

        Fire-and-forget over the request channel the first encode rank binds
        (``ts-host`` in encode mode, ``epd/encode_host.py``), which relays it
        to the others. The ranks run the vision tower and ship the embedding
        to prefill over Mooncake; the gateway gets no streamed response, only
        the gRPC ack. ``encode_request`` is an ``encode_worker.EncodeRequest``
        whose pixels ride the channel as tensors or shared-memory handles.
        """
        if self._encode_sender is None:
            raise RuntimeError("only an EPD encode server runs encode ranks")
        self._encode_sender.send_pyobj(encode_request)

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
                                and (skip_names is None or "text" not in skip_names)
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
                await self._send_one_request(tmp_obj, tokenized_obj, created_time)
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
                await self._send_one_request(tmp_obj, warmup_obj, created_time)
                await self._wait_one_response(tmp_obj).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for replica_index in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    replica_obj = prepare_parallel_sampling_replica(
                        tmp_obj, tokenized_objs[i], replica_index
                    )
                    await self._send_one_request(tmp_obj, replica_obj, created_time)
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

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        self._abort_on_engine(rid)

    def _abort_on_engine(self, rid: str) -> None:
        """Ask the engine to abort ``rid``, without waiting for it. The task
        is held until it finishes, then dropped."""

        async def abort() -> None:
            await self.frontend.abort(rid)

        task = asyncio.get_running_loop().create_task(print_exception_wrapper(abort))
        self._abort_tasks.add(task)
        task.add_done_callback(self._abort_tasks.discard)

    def block_generation_admission(self) -> None:
        """Stop requests before they acquire the model-update reader lock."""
        self._generation_admit.clear()

    def allow_generation_admission(self) -> None:
        """Allow requests to enter after scheduler resume or pause failure."""
        self._generation_admit.set()

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
    ) -> tuple[bool, str, Any]:
        self.auto_create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format
        logger.info(f"Start update_weights. Load format={obj.load_format!s}")

        # Hold the lock if it is not async. This means that weight sync
        # cannot run while requests are in progress.
        async with self.model_update_lock.writer_lock:
            return await self._wait_for_model_update_from_disk(obj)

    async def _wait_for_model_update_from_disk(
        self, obj: UpdateWeightFromDiskReqInput
    ) -> tuple[bool, str]:
        results = await self._control(obj)
        if not self.server_args.mapping.attn.has_dp:
            (result,) = results
            if result.success:
                self.served_model_name = obj.model_path
                self.server_args.model = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            return result.success, result.message, result.num_paused_requests
        all_success = all(r.success for r in results)
        if all_success:
            self.server_args.model = obj.model_path
            self.server_args.load_format = obj.load_format
            self.model_path = obj.model_path
        all_message = " | ".join(r.message for r in results)
        all_paused_requests = [r.num_paused_requests for r in results]
        return all_success, all_message, all_paused_requests

    async def open_session(self, obj: OpenSessionReqInput) -> str | None:
        """Sessions are not served: the engines refuse a request that names
        one, so no session ever opens."""
        return None

    async def close_session(self, obj: CloseSessionReqInput) -> None:
        return None

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
                        "precomputed_multimodal_inputs",
                        "input_multi_ids",
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
        logging.info(f"Config logging: obj={obj!r}")
        self.log_request_metadata = self.get_log_request_metadata()

    # ---- Server lifecycle / health -------------------------------
    # Intent-revealing wrappers around the private ``server_status``
    # field. Callers (notably ``control_server.py``) drive transitions
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
        self._maybe_launch_rl_control_plane(loop)

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

    def _maybe_launch_rl_control_plane(self, loop):
        """Launch the SGLang-compatible RL control app on this event loop."""
        if self._rl_control_task is not None:
            return
        port = self.server_args.rl_control_port
        if not port:
            return
        self._rl_control_task = loop.create_task(self._serve_rl_control_plane(port))

    async def _serve_rl_control_plane(self, port: int):
        # A control-plane crash must not take down the engine, so this does not
        # use print_exception_wrapper (which kills the process tree).
        try:
            import uvicorn

            from tokenspeed.runtime.entrypoints.sglang_compat_http import (
                build_sglang_compat_app,
            )

            app = build_sglang_compat_app(self)

            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=rl_control.control_bind_host(self.server_args),
                    port=port,
                    log_level="warning",
                )
            )
            await server.serve()
        except Exception as e:  # noqa: BLE001
            logger.error(f"RL control plane stopped: {e!s}")

    def rl_control_url(self) -> str | None:
        """Base URL of the in-engine RL control app, for gateway discovery."""
        return rl_control.control_url(self.server_args)

    def rl_advertisement(self) -> dict[str, str]:
        """Control URL plus capability labels, for ``GetServerInfo.server_args``."""
        return rl_control.advertisement(self.server_args)

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            logger.info(
                f"Gracefully exiting... remaining number of requests {remain_num_req!s}",
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        # Close local metric resources before this path kills itself;
        # atexit does not run after SIGKILL.
        metrics_service.close(timeout=2.0)
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    async def handle_loop(self):
        """Hand the engines' outputs to their requests, a batch per wakeup."""
        if self.frontend is None:
            return
        while True:
            outputs = await self.frontend.next_outputs()
            for rid in self.output_processor.handle_engine_outputs(outputs):
                # A stop string ended it here; the engine is still running it.
                self._abort_on_engine(rid)
            self.last_receive_tstamp = time.time()

    async def _control(self, obj) -> list:
        """Run a control operation on every engine; each one's reply, in
        arrival order. Control messages carry no tensors, so each is one
        frame."""
        replies = await self.frontend.control([zmq_wire.encode_control(obj)])
        return [zmq_wire.decode_control_output(payload) for (payload,) in replies]


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"AsyncLLM hit an exception: {traceback!s}")
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


class SignalHandler:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def signal_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. signum={signum!r} frame={frame!r}. Draining requests "
            "and shutting down...",
        )
        self.tokenizer_manager.gracefully_exit = True
