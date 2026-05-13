# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.
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
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    MultimodalDataInputFormat,
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
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/tokenspeed/runtime/utils/common.py:load_image for more details.
        image_data: MultimodalDataInputFormat | None = None,
        audio_data: MultimodalDataInputFormat | None = None,
        video_data: MultimodalDataInputFormat | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        custom_logit_processor: list[str] | str | None = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: list[str] | str | None = None,
        bootstrap_port: list[int] | int | None = None,
        bootstrap_room: list[int] | int | None = None,
        data_parallel_rank: int | None = None,
        lora_path: list[str | None] | str | None = None,
    ) -> dict | Iterator[dict]:
        """
        The arguments of this function match
        ``tokenspeed.runtime.engine.io_struct.GenerateReqInput``.
        Please refer to ``GenerateReqInput`` for the documentation.
        """
        if self.server_args.mapping.has_attn_dp:
            if data_parallel_rank is None:
                logger.debug("data_parallel_rank not provided, using default dispatch")
            elif data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            elif data_parallel_rank >= self.server_args.mapping.attn.dp_size:
                raise ValueError(
                    f"data_parallel_rank must be less than dp_size: {self.server_args.mapping.attn.dp_size}"
                )

        if self.tokenizer_manager.server_args.speculative_algorithm is not None:
            return_logprob = False
            logprob_start_len = None
            top_logprobs_num = None
            token_ids_logprob = None

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            lora_path=lora_path,
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
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/tokenspeed/runtime/utils/common.py:load_image for more details.
        image_data: MultimodalDataInputFormat | None = None,
        audio_data: MultimodalDataInputFormat | None = None,
        video_data: MultimodalDataInputFormat | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        custom_logit_processor: list[str] | str | None = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: list[str] | str | None = None,
        bootstrap_port: list[int] | int | None = None,
        bootstrap_room: list[int] | int | None = None,
        user_rid: list[str] | str | None = None,
    ) -> dict | AsyncIterator[dict]:
        """
        The arguments of this function match
        ``tokenspeed.runtime.engine.io_struct.GenerateReqInput``.
        Please refer to ``GenerateReqInput`` for the documentation.
        """

        if self.tokenizer_manager.server_args.speculative_algorithm is not None:
            return_logprob = False
            logprob_start_len = None
            top_logprobs_num = None
            token_ids_logprob = None

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            input_embeds=input_embeds,
            input_multi_ids=input_multi_ids,
            input_extra_infos=input_extra_infos,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            return_hidden_states=return_hidden_states,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            user_rid=user_rid,
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
        kill_process_tree(os.getpid(), include_parent=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def flush_cache(self):
        return self.llm.run(self.tokenizer_manager.flush_cache())

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
            dtypes=dtypes,
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

    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
        pinned: bool = False,
    ) -> int:
        """Load a PEFT LoRA adapter. Returns the integer lora_id."""
        success, lora_id, message = self.llm.run(
            self.tokenizer_manager.load_lora_adapter(lora_name, lora_path, pinned)
        )
        if not success:
            raise RuntimeError(f"Failed to load LoRA adapter '{lora_name}': {message}")
        # Update the local path→id registry so future requests resolve correctly
        self.tokenizer_manager._lora_path_to_id[lora_name] = lora_id
        return lora_id

    def unload_lora_adapter(self, lora_name: str) -> None:
        """Unload a previously loaded LoRA adapter."""
        success, message = self.llm.run(
            self.tokenizer_manager.unload_lora_adapter(lora_name)
        )
        if not success:
            raise RuntimeError(
                f"Failed to unload LoRA adapter '{lora_name}': {message}"
            )
        self.tokenizer_manager._lora_path_to_id.pop(lora_name, None)

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    if not server_args.enable_symm_mem:
        os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    if not server_args.disable_tf32:
        # Force TF32 on for cuBLAS/cuDNN matmuls. setdefault so a user's
        # explicit env wins; --disable-tf32 is the documented opt-out.
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
        os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")

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
            assert data["status"] == "ready"

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
    return tokenizer_manager, None, scheduler_info
