# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
#
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

"""
The entry point of inference server.

This file implements python APIs for the inference engine.
"""

# ruff: noqa: E402
import asyncio
import atexit
import copy
import dataclasses
import multiprocessing as mp
import os
import signal
import sys
import threading
from collections.abc import AsyncIterator, Iterator

import zmq
import zmq.asyncio

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.llm import LLM


def _ignore_threading_atexit(*args, **kwargs) -> None:
    return None


# Fix a bug of Python threading
setattr(threading, "_register_atexit", _ignore_threading_atexit)

import torch
import uvloop

from tokenspeed.runtime.engine.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from tokenspeed.runtime.engine.event_loop import run_event_loop
from tokenspeed.runtime.engine.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from tokenspeed.runtime.entrypoints.engine_base import EngineBase
from tokenspeed.runtime.utils import (
    MultiprocessingSerializer,
    configure_logger,
    get_colorful_logger,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.launcher import interface_for_host
from tokenspeed.runtime.utils.process import kill_process_tree
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from tokenspeed.version import __version__

logger = get_colorful_logger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `tokenspeed/runtime/utils/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        self.port_args = PortArgs.init_new(server_args)
        logger.info("server_args=%r", server_args)

        # Launch subprocesses
        tokenizer_manager, _, scheduler_info = _launch_subprocesses(
            server_args=server_args,
            port_args=self.port_args,
        )
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.scheduler_info = scheduler_info

        # Sync facade for blocking callers. Owns its own bg event-loop thread; see runtime/engine/llm.py
        # for the queue-bridge semantics.
        self.llm = LLM(self.tokenizer_manager)

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: list[list[int]] | list[int] | None = None,
        # SGLang-compatible logprob controls; vLLM-compatible requests use
        # sampling_params["logprobs"].
        return_logprob: list[bool] | bool | None = None,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        return_text_in_logprobs: bool = False,
        logprob_format: list[str | None] | str | None = None,
        custom_logit_processor: list[str] | str | None = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: list[str] | str | None = None,
        bootstrap_port: list[int] | int | None = None,
        bootstrap_room: list[int] | int | None = None,
        data_parallel_rank: int | None = None,
    ) -> dict | Iterator[dict]:
        """
        The arguments of this function match
        ``tokenspeed.runtime.engine.io_struct.GenerateReqInput``.
        Please refer to ``GenerateReqInput`` for the documentation.
        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            return_text_in_logprobs=return_text_in_logprobs,
            logprob_format=logprob_format,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
        )
        if stream:
            return self.llm.generate_stream(obj)
        else:
            return self.llm.generate(obj)

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: list[list[int]] | list[int] | None = None,
        input_embeds: torch.Tensor = None,
        input_multi_ids: list[list[int]] = None,
        input_extra_infos: list[dict] = None,
        # Same legacy logprob controls as generate().
        return_logprob: list[bool] | bool | None = None,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        return_text_in_logprobs: bool = False,
        logprob_format: list[str | None] | str | None = None,
        custom_logit_processor: list[str] | str | None = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: list[str] | str | None = None,
        bootstrap_port: list[int] | int | None = None,
        bootstrap_room: list[int] | int | None = None,
        user_rid: list[str] | str | None = None,
        data_parallel_rank: list[int] | int | None = None,
    ) -> dict | AsyncIterator[dict]:
        """
        The arguments of this function match
        ``tokenspeed.runtime.engine.io_struct.GenerateReqInput``.
        Please refer to ``GenerateReqInput`` for the documentation.
        """

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            input_embeds=input_embeds,
            input_multi_ids=input_multi_ids,
            input_extra_infos=input_extra_infos,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            return_text_in_logprobs=return_text_in_logprobs,
            logprob_format=logprob_format,
            return_hidden_states=return_hidden_states,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            user_rid=user_rid,
            data_parallel_rank=data_parallel_rank,
        )
        generator = self.tokenizer_manager.generate_request(obj)

        async def wrapped_output_generator(original_async_gen):
            async for item in original_async_gen:
                yield item

            await asyncio.sleep(1)
            self.tokenizer_manager.abort_request(obj.rid[0])

        if stream is True:
            return wrapped_output_generator(generator)
        else:
            return await generator.__anext__()

    def shutdown(self):
        """Shutdown the engine"""
        # Stop the sync-facade event loop before subprocess teardown so any
        # in-flight blocking callers see a clean loop close instead of a
        # stale-reference error.
        if getattr(self, "llm", None) is not None:
            self.llm.shutdown()
        bootstrap_server = getattr(
            getattr(self, "tokenizer_manager", None),
            "bootstrap_server",
            None,
        )
        close_bootstrap = getattr(bootstrap_server, "close", None)
        try:
            if callable(close_bootstrap):
                close_bootstrap()
        except Exception:
            logger.exception("Failed to close the disaggregation bootstrap server")
        finally:
            kill_process_tree(os.getpid(), include_parent=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def flush_cache(self):
        return self.llm.run(self.tokenizer_manager.flush_cache())

    def pause_scheduler(self, mode: str = "abort"):
        """Pause generation (e.g. to swap weights). See AsyncLLM.pause_scheduler."""
        return self.llm.run(self.tokenizer_manager.pause_scheduler(mode=mode))

    def resume_scheduler(self):
        """Resume generation after :meth:`pause_scheduler`."""
        return self.llm.run(self.tokenizer_manager.resume_scheduler())

    def is_scheduler_paused(self):
        """Return whether the scheduler is currently paused."""
        return self.llm.run(self.tokenizer_manager.is_scheduler_paused())

    def start_profile(self):
        self.llm.run(self.tokenizer_manager.start_profile())

    def stop_profile(self):
        self.llm.run(self.tokenizer_manager.stop_profile())

    def start_expert_distribution_record(self):
        self.llm.run(self.tokenizer_manager.start_expert_distribution_record())

    def stop_expert_distribution_record(self):
        self.llm.run(self.tokenizer_manager.stop_expert_distribution_record())

    def dump_expert_distribution_record(self):
        self.llm.run(self.tokenizer_manager.dump_expert_distribution_record())

    def get_server_info(self):
        internal_states = self.llm.run(self.tokenizer_manager.get_internal_state())
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
        }

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ):
        """Initialize parameter update group."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        return self.llm.run(self.tokenizer_manager.init_weights_update_group(obj))

    def destroy_weights_update_group(
        self,
        group_name: str = "weight_update_group",
    ):
        """Destroy the parameter update group."""
        obj = DestroyWeightsUpdateGroupReqInput(group_name=group_name)
        return self.llm.run(self.tokenizer_manager.destroy_weights_update_group(obj))

    def update_weights_from_distributed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_update_group",
        flush_cache: bool = True,
    ):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtype_names=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
        )
        return self.llm.run(self.tokenizer_manager.update_weights_from_distributed(obj))

    def update_weights_from_tensor(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        load_format: str | None = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.mapping.world_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        return self.llm.run(self.tokenizer_manager.update_weights_from_tensor(obj))

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
    ):
        """Update the weights from disk inplace without re-launching the engine.

        This method allows updating the model weights from disk without restarting
        the engine. It can be used to load a different model or update weights with
        new training.
        """
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
        )

        return self.llm.run(self.tokenizer_manager.update_weights_from_disk(obj))

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        return self.llm.run(self.tokenizer_manager.get_weights_by_name(obj))

    def release_memory_occupation(self, tags: list[str] | None = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return self.llm.run(self.tokenizer_manager.release_memory_occupation(obj))

    def resume_memory_occupation(self, tags: list[str] | None = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        return self.llm.run(self.tokenizer_manager.resume_memory_occupation(obj))

    def is_sleeping(self) -> bool:
        """Return whether any GPU memory is currently released (data-plane sleep)."""
        return self.llm.run(self.tokenizer_manager.is_sleeping())

    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        if not isinstance(recv_req, RpcReqOutput):
            raise TypeError(f"Expected RpcReqOutput, got {type(recv_req).__name__}.")
        if not recv_req.success:
            raise RuntimeError(recv_req.message)

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)


def _set_socket_interface(server_args: ServerArgs):
    """Point gloo and NCCL at the interface that reaches the head node.

    Gloo has no peer-address heuristic: left alone it binds whatever the local
    hostname resolves to, which is a loopback entry on many hosts, and every
    cross-node gloo collective then fails to connect.
    """
    if server_args.mapping.nnodes <= 1 or not server_args.dist_init_addr:
        return

    head = server_args.dist_init_addr.rsplit(":", 1)[0]
    interface = interface_for_host(head)
    if interface is None:
        logger.warning(
            f"cannot tell which interface reaches the head node {head}; set "
            "GLOO_SOCKET_IFNAME and NCCL_SOCKET_IFNAME explicitly if cross-node setup fails"
        )
        return

    for name in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"):
        os.environ.setdefault(name, interface)
    logger.info(f"socket interface reaching {head}: {interface}")


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if server_args.disable_symm_mem:
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
    if server_args.disable_nccl_nvls:
        os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    if not server_args.disable_tf32:
        # Force TF32 on for cuBLAS/cuDNN matmuls. setdefault so a user's
        # explicit env wins; --disable-tf32 is the documented opt-out.
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
        os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")

    _set_socket_interface(server_args)

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Install a launch-phase SIGQUIT handler so a failing child tears down the
    # whole local process tree instead of leaving orphaned workers behind.
    # TokenizerManager may replace this handler later during steady-state
    # serving.
    def launch_phase_sigquit_handler(signum, frame):
        logger.error(
            "Received sigquit from a child process. It usually means the child failed."
        )
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _launch_subprocesses(
    server_args: ServerArgs, port_args: PortArgs | None = None
) -> tuple[AsyncLLM, None, dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info("server_args=%r", server_args)

    # If using model from www.modelscope.cn, first download the model.
    server_args.model, server_args.tokenizer = prepare_model_and_tokenizer(
        server_args.model, server_args.tokenizer
    )

    scheduler_procs = []
    if not server_args.mapping.attn.has_dp:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []
        rank_start = server_args.mapping.nprocs_per_node * server_args.node_rank
        rank_end = rank_start + server_args.mapping.nprocs_per_node
        for rank in range(rank_start, rank_end):
            # Create per-rank server_args with rank-initialized mapping
            rank_server_args = copy.copy(server_args)
            rank_server_args.mapping = copy.deepcopy(server_args.mapping)
            rank_server_args.mapping.rank = rank

            reader, writer = mp.Pipe(duplex=False)

            proc = mp.Process(
                target=run_event_loop,
                args=(
                    rank_server_args,
                    port_args,
                    writer,
                ),
            )
            with memory_saver_adapter.configure_subprocess():
                proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            if data.get("status") != "ready":
                raise RuntimeError(
                    "Initialization failed. Please see the error messages above."
                )

        if not envs.TOKENSPEED_BLOCK_NONZERO_RANK_CHILDREN.get():
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(
            server_args.host, server_args.port, server_args.enable_metrics
        )

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                "Scheduler or DataParallelController %s terminated with %s",
                proc.pid,
                proc.exitcode,
            )
        return None, None, None

    # Launch the main-process async frontend. The detokenizer runs
    # inline inside AsyncLLM — no separate subprocess.
    tokenizer_manager = AsyncLLM(server_args, port_args)

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                "Rank %s scheduler is dead. Please check if there are relevant logs.", i
            )
            scheduler_procs[i].join()
            logger.error("Exit code: %s", scheduler_procs[i].exitcode)
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    tokenizer_manager.max_single_request_tokens = scheduler_info[
        "max_single_request_tokens"
    ]
    tokenizer_manager.context_len = scheduler_info["max_model_len"]
    return tokenizer_manager, None, scheduler_info


def launch_scheduler_headless(server_args: ServerArgs) -> None:
    """Launch scheduler process(es) driven directly by SMG over msgpack ZMQ.

    Headless: no in-process tokenizer_manager and no AsyncLLM detokenizer (SMG
    owns both). This process only spawns and
    supervises the scheduler workers, which connect out to the SMG-bound
    handshake/input/output sockets (see zmq_msgpack.connect_msgpack_engine_for_loop),
    then blocks until they exit.

    Forces ``--zmq-msgpack`` + ``--skip-tokenizer-init`` since SMG passes
    token ids. The workers dial ``tcp://{--data-parallel-address}:
    {--data-parallel-rpc-port}`` (defaults ``127.0.0.1:30500``; the frontend
    binds it). Both flags always carry values — non-emptiness and the u16
    range are enforced at parse time.
    """
    server_args.zmq_msgpack = True
    server_args.skip_tokenizer_init = True
    # DP > 1: each rank dials the frontend with its own engine identity
    # (zmq_engine_index + dp_rank) in zmq_msgpack.connect_msgpack_engine_for_loop, the
    # choke point shared with the non-headless --zmq-msgpack launch path.

    configure_logger(server_args)
    _set_envs_and_config(server_args)

    # PortArgs is still derived (nccl_port drives torch.distributed); the pickle
    # scheduler_input/tokenizer IPC names it carries are unused in msgpack mode.
    port_args = PortArgs.init_new(server_args)
    logger.info("headless server_args=%r", server_args)

    server_args.model, server_args.tokenizer = prepare_model_and_tokenizer(
        server_args.model, server_args.tokenizer
    )

    scheduler_procs: list[mp.Process] = []
    scheduler_pipe_readers = []

    # SIGUSR1 is what a scheduler sends its parent on an internal exception
    # (see run_event_loop) — the child itself then exits 0, so record the
    # failure here or the supervisor would report success.
    child_failed = threading.Event()

    def _terminate_schedulers(signum=None, _frame=None):
        """Forward a shutdown (or child-failure SIGUSR1) to every scheduler."""
        if signum is not None:
            logger.info("received signal %s; terminating scheduler(s)", signum)
        if signum == signal.SIGUSR1:
            child_failed.set()
        for proc in scheduler_procs:
            if proc.is_alive():
                proc.terminate()

    # Install before spawning so a signal in the launch window is not lost;
    # an unhandled SIGUSR1 would kill this supervisor without cleanup.
    signal.signal(signal.SIGTERM, _terminate_schedulers)
    signal.signal(signal.SIGINT, _terminate_schedulers)
    signal.signal(signal.SIGUSR1, _terminate_schedulers)

    if not server_args.mapping.attn.has_dp:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        rank_start = server_args.mapping.nprocs_per_node * server_args.node_rank
        rank_end = rank_start + server_args.mapping.nprocs_per_node
        for rank in range(rank_start, rank_end):
            rank_server_args = copy.copy(server_args)
            rank_server_args.mapping = copy.deepcopy(server_args.mapping)
            rank_server_args.mapping.rank = rank

            reader, writer = mp.Pipe(duplex=False)
            proc = mp.Process(
                target=run_event_loop,
                args=(rank_server_args, port_args, writer),
            )
            with memory_saver_adapter.configure_subprocess():
                proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)
    else:
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers.append(reader)
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    try:
        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(
                    "Rank %s scheduler is dead. Please check if there are "
                    "relevant logs.",
                    i,
                )
                scheduler_procs[i].join()
                logger.error("Exit code: %s", scheduler_procs[i].exitcode)
                raise
            if data.get("status") != "ready":
                raise RuntimeError(
                    "Scheduler initialization failed. See the error messages above."
                )
        logger.info(
            "headless scheduler(s) ready; SMG handshake endpoint=%s",
            server_args.zmq_handshake_endpoint(),
        )

        # Supervise until every scheduler exits. If any rank dies (e.g. OOM
        # SIGKILL) take the survivors down and exit nonzero instead of
        # reporting success.
        failed = False
        alive = list(scheduler_procs)
        while alive:
            for proc in list(alive):
                proc.join(timeout=1)
                if proc.exitcode is None:
                    continue
                alive.remove(proc)
                if proc.exitcode == 0:
                    logger.info(
                        "scheduler %s exited with code %s", proc.pid, proc.exitcode
                    )
                else:
                    logger.error(
                        "scheduler %s exited with code %s; terminating the "
                        "remaining scheduler(s)",
                        proc.pid,
                        proc.exitcode,
                    )
                    failed = True
                    _terminate_schedulers()
        if failed or child_failed.is_set():
            sys.exit(1)
    finally:
        # Backstop: never leak GPU-holding scheduler trees, whichever path
        # (signal, dead rank, readiness failure) unwound this launcher.
        _terminate_schedulers()
        for proc in scheduler_procs:
            proc.join(timeout=5)
        kill_process_tree(os.getpid(), include_parent=False)


def run_scheduler_headless_from_cli(argv: list[str] | None = None) -> None:
    """CLI shim: ``python -m tokenspeed.runtime.entrypoints.engine <ServerArgs>``."""
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    server_args = prepare_server_args(sys.argv[1:] if argv is None else argv)
    launch_scheduler_headless(server_args)


if __name__ == "__main__":
    run_scheduler_headless_from_cli()
