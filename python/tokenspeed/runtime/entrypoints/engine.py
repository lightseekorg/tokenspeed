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
import contextlib
import copy
import dataclasses
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator

from tokenspeed.runtime.engine.async_llm import AsyncLLM
from tokenspeed.runtime.engine.llm import LLM
from tokenspeed.runtime.metrics.config import metrics_enabled
from tokenspeed.runtime.metrics.relay import MetricsRelay
from tokenspeed.runtime.metrics.service import metrics_service


def _ignore_threading_atexit(*args, **kwargs) -> None:
    return None


# Fix a bug of Python threading
setattr(threading, "_register_atexit", _ignore_threading_atexit)

import torch
import uvloop
import zmq
from tokenspeed_engine import BootstrapServer, FrontendListener

from tokenspeed.runtime.cache.l3.identity import (
    L3_FLUSH_REQUIRES_WEIGHT_VERSION,
    resolve_l3_weight_version,
)
from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.engine.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    IpcSender,
    RebalanceExpertsReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromTensorReqInput,
    mooncake_load_weight_version,
)
from tokenspeed.runtime.engine.rust_host import (
    host_binary,
    rust_host_refusals,
    start_rust_host,
)
from tokenspeed.runtime.entrypoints.engine_base import EngineBase
from tokenspeed.runtime.pd.utils import DisaggregationMode, TransferBackend
from tokenspeed.runtime.utils import (
    MultiprocessingSerializer,
    configure_logger,
    get_colorful_logger,
    get_zmq_socket,
    launch_dummy_health_check_server,
    launch_metrics_server,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.launcher import interface_for_host
from tokenspeed.runtime.utils.network import join_host_port
from tokenspeed.runtime.utils.process import (
    begin_process_shutdown,
    kill_process_tree,
    receive_ready_messages,
    register_child_failure_signals,
    stop_processes,
)
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

        self.server_args = server_args
        # Set once the subprocesses run; `shutdown` may run before that, on a
        # failed launch or at interpreter exit.
        self.tokenizer_manager: AsyncLLM | None = None
        self.llm: LLM | None = None
        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        self.port_args = PortArgs.init_new(server_args)
        logger.info(f"server_args={server_args!r}")

        # Launch subprocesses
        try:
            tokenizer_manager, _, scheduler_info = _launch_subprocesses(
                server_args=server_args,
                port_args=self.port_args,
            )
        except BaseException:
            try:
                begin_process_shutdown()
            finally:
                self.shutdown()
            raise
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
        if self.llm is not None:
            self.llm.shutdown()
        tokenizer_manager = self.tokenizer_manager
        try:
            if (
                tokenizer_manager is not None
                and tokenizer_manager.bootstrap_server is not None
            ):
                tokenizer_manager.bootstrap_server.close()
        except Exception:
            logger.exception("Failed to close the disaggregation bootstrap server")
        finally:
            # Before the process tree goes down: the relay flushes its sinks
            # on a stop, and would be killed with the tree.
            if (
                tokenizer_manager is not None
                and tokenizer_manager.metrics_relay is not None
            ):
                tokenizer_manager.metrics_relay.close(timeout=2.0)
            kill_process_tree(os.getpid(), include_parent=False)
            metrics_service.close(timeout=2.0)

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

    def get_server_info(self):
        internal_states = self.llm.run(self.tokenizer_manager.get_internal_state())
        server_args = dataclasses.asdict(self.tokenizer_manager.server_args)
        server_args.pop("rl_control_api_key", None)
        return {
            **server_args,
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
        *,
        weight_version: str | None,
    ):
        """Update weights from distributed source.

        ``weight_version`` is required. Pass ``None`` to keep the current
        namespace on an intermediate update. Flushed L3 updates must pass
        a caller-supplied identity so independent checkpoints cannot share
        a minted successor.
        """
        if (
            flush_cache
            and weight_version is None
            and self.server_args.kvstore_storage_backend is not None
        ):
            return False, L3_FLUSH_REQUIRES_WEIGHT_VERSION
        weight_version = resolve_l3_weight_version(weight_version)
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtype_names=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            weight_version=weight_version,
        )
        result = self.llm.run(
            self.tokenizer_manager.update_weights_from_distributed(obj)
        )
        success = result[0] if isinstance(result, tuple) else bool(result)
        if success and weight_version is not None:
            self.server_args.weight_version = str(weight_version)
        return result

    def update_weights_from_mooncake(
        self,
        version: int,
        *,
        flush_cache: bool,
        weight_version: str | None,
    ):
        """Load one committed Model Updater SDK version on every worker.

        ``flush_cache`` selects whether the KV caches are dropped and a new
        weight namespace published, so it is explicit here; only the HTTP
        route defaults it, for the reference engine's trainer clients.
        ``weight_version`` is required. Pass ``None`` to publish
        ``str(version)`` on a flushed load, or to keep the current namespace
        on an intermediate (unflushed) one (``mooncake_load_weight_version``).
        """
        weight_version = resolve_l3_weight_version(
            mooncake_load_weight_version(
                version=version,
                flush_cache=flush_cache,
                weight_version=weight_version,
            ),
        )
        obj = UpdateWeightsFromMooncakeReqInput(
            version=version,
            flush_cache=flush_cache,
            weight_version=weight_version,
        )
        result = self.llm.run(self.tokenizer_manager.update_weights_from_mooncake(obj))
        success = result[0] if isinstance(result, tuple) else bool(result)
        if success and weight_version is not None:
            self.server_args.weight_version = str(weight_version)
        return result

    def rebalance_experts(self) -> tuple[bool, str]:
        """Start one online expert rebalance now (``--enable-eplb``)."""
        return self.llm.run(
            self.tokenizer_manager.rebalance_experts(RebalanceExpertsReqInput())
        )

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
        flush_cache: bool = True,
        weight_version: str | None = None,
    ):
        """Update the weights from disk inplace without re-launching the engine.

        This method allows updating the model weights from disk without restarting
        the engine. It can be used to load a different model or update weights with
        new training. With L3 enabled, a flush requires an explicit
        `weight_version` so the new weights publish under a new namespace.
        """
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
            flush_cache=flush_cache,
            weight_version=weight_version,
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


def _force_env(owner: str, key: str, value: str) -> None:
    """Set ``key`` to ``value``: ``owner``'s promise beats the ambient
    environment, so a conflicting value is replaced, loudly, instead of
    silently voiding it."""
    prior = os.environ.get(key)
    if prior is not None and prior != value:
        logger.warning(f"{owner} replaces {key}={prior} with {value}")
    os.environ[key] = value


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if server_args.disable_symm_mem:
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
    if server_args.disable_nccl_nvls:
        os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    if getattr(server_args, "_beam_checkpoint", False) is None:
        # A hub id the local cache does not hold yet (a first launch): fetch
        # its config.json now, before any process starts.
        server_args.resolve_model_defaults(download=True)
        if server_args._beam_checkpoint:
            # Beam's all-reduce fusion default across nodes reads the
            # checkpoint as well (ServerArgs.resolve_communication): resolve
            # it again from the downloaded file, as from a directory, so that
            # every node of a launch decides it from the same config.json
            # whatever its cache held when its arguments resolved.
            server_args.resolve_communication()
            server_args.resolve_numerics()
        if server_args._beam_checkpoint is None and not server_args.disable_tf32:
            logger.warning(
                f"Cannot read the config.json of {server_args.model} before "
                "launch, so TF32 stays at its default. For a Beam checkpoint, "
                "pass its local directory as --model or pass --disable-tf32 so "
                "that its FP32 router runs without TF32."
            )
    if not server_args.disable_tf32:
        # Force TF32 on for cuBLAS/cuDNN matmuls. setdefault so a user's
        # explicit env wins; --disable-tf32 is the documented opt-out.
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
        os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    if getattr(server_args, "_beam_checkpoint", False):
        # Beam serves without TF32 (ServerArgs.resolve_model_defaults): both
        # switches off for the processes started below, whatever the caller
        # exported.
        for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
            _force_env("Beam (no TF32)", key, "0")
    if server_args.numerics in BITWISE_ENVELOPES:
        # Bitwise envelopes: no TF32 anywhere, and pin NCCL to one
        # algorithm/protocol so the reduction association order cannot switch
        # with message size. The envelope's promise beats ambient
        # environment (_force_env).
        for key, value in (
            ("NVIDIA_TF32_OVERRIDE", "0"),
            ("NCCL_ALGO", "Ring"),
            ("NCCL_PROTO", "Simple"),
        ):
            _force_env(f"--numerics {server_args.numerics}", key, value)

    _set_socket_interface(server_args)

    # Set prometheus env vars
    if metrics_enabled(server_args, "prometheus"):
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Let Engine.__init__ or the headless supervisor own process-tree cleanup.
    # A signal callback must not log or acquire locks during an interrupted call.
    register_child_failure_signals()

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _scheduler_ranks(server_args: ServerArgs) -> range:
    """Global ranks of the scheduler processes this node launches."""
    if server_args.emulate_rank_zero:
        return range(1)
    rank_start = server_args.mapping.nprocs_per_node * server_args.node_rank
    return range(rank_start, rank_start + server_args.mapping.nprocs_per_node)


def _stop_schedulers_on_signal(scheduler_procs: list[mp.Process]) -> None:
    """Make a node of rank >= 1 stop its schedulers before a signal ends it.

    Nothing else stops them when the node exits, so they would keep their
    GPUs. On SIGTERM, SIGINT or SIGUSR1 (sent by a failing scheduler) the
    schedulers get SIGTERM and TOKENSPEED_NONZERO_RANK_SHUTDOWN_TIMEOUT
    seconds to exit before they are killed; the node then still ends by
    the signal it received.
    """
    if threading.current_thread() is not threading.main_thread():
        return  # Signal handlers can only be set from the main thread.
    timeout = envs.TOKENSPEED_NONZERO_RANK_SHUTDOWN_TIMEOUT.get()
    stopping = False

    def stop_schedulers_and_exit(signum, _frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        logger.info(f"Received signal {signum!s}; stopping scheduler processes")
        try:
            stop_processes(scheduler_procs, timeout)
        finally:
            # Whatever is left still gets the parent-death signal.
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, stop_schedulers_and_exit)


def _frontend_addresses(server_args: ServerArgs) -> tuple[str, int]:
    """Where an in-process frontend binds its sockets: loopback on one node,
    with a free port; across nodes, the first node's address
    (``--dist-init-addr``) and ``--data-parallel-rpc-port``, which every
    node's ranks can compute."""
    if server_args.mapping.nnodes == 1:
        return "127.0.0.1", 0
    host = server_args.dist_init_addr.rsplit(":", 1)[0].strip("[]")
    return host, server_args.data_parallel_rpc_port


def _frontend_routing(server_args: ServerArgs) -> str:
    """How the frontend places requests on the attention-DP ranks
    (``--load-balance-method``); the disaggregated roles place by bootstrap
    room, the rule both roles apply, except a decode request with a
    ``data_parallel_rank`` pin."""
    if server_args.disaggregation_mode != "null" and server_args.mapping.attn.has_dp:
        return "bootstrap_room"
    return {
        "round_robin": "round_robin",
        "shortest_queue": "least_in_flight",
        "minimum_cache_usage": "minimum_cache_usage",
    }[server_args.load_balance_method]


# How long the frontend waits for every rank to finish loading and register.
# Model loading (and kernel autotuning) bounds it, not the handshake itself;
# a rank that dies first fails the launch through its ready pipe.
_FRONTEND_HANDSHAKE_TIMEOUT_S = 24 * 3600.0
# How long a request may wait to enter a rank's input socket.
_FRONTEND_SEND_TIMEOUT_S = 60.0


def _start_ranks(
    server_args: ServerArgs,
    port_args: PortArgs,
    metrics_relay: MetricsRelay | None,
    parent_death_signal: bool,
) -> tuple[list, list]:
    """Start this node's engine ranks, one ``ts-host`` process each, and
    return them with their ready pipes. ``parent_death_signal`` asks the
    kernel to kill a rank when this node dies (see
    ``utils.process.set_parent_death_signal``)."""
    binary = host_binary()
    memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=server_args.enable_memory_saver
    )
    procs, readers = [], []
    for rank in _scheduler_ranks(server_args):
        rank_server_args = copy.copy(server_args)
        rank_server_args.mapping = copy.deepcopy(server_args.mapping)
        rank_server_args.mapping.rank = rank
        proc, reader = start_rust_host(
            binary,
            rank_server_args,
            port_args,
            memory_saver_adapter.configure_subprocess(),
            metrics_relay.endpoint if metrics_relay is not None else None,
            parent_death_signal,
        )
        procs.append(proc)
        readers.append(reader)
    return procs, readers


def _launch_subprocesses(
    server_args: ServerArgs, port_args: PortArgs | None = None
) -> tuple[AsyncLLM, None, dict]:
    """Launch this node's engine ranks and, on the first node, the
    in-process frontend (``AsyncLLM``) they register with.

    Every rank is a ``ts-host`` process; its attention-TP group's first
    rank dials the frontend over the msgpack wire, so the frontend tokenizes
    and detokenizes and the ranks see token ids only. An EPD encode server
    runs encode ranks instead (``_launch_encode_server``).
    """
    # Configure global environment
    configure_logger(server_args)
    _set_envs_and_config(server_args)
    metrics_service.prepare(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"server_args={server_args!r}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model, server_args.tokenizer = prepare_model_and_tokenizer(
        server_args.model, server_args.tokenizer
    )

    metrics_service.start(server_args, port_args)
    if server_args.disaggregation_mode == "encode":
        return _launch_encode_server(server_args, port_args)
    # Fail before any rank loads the model.
    refusals = rust_host_refusals(server_args)
    if refusals:
        raise NotImplementedError(
            "The engine host does not support " + ", ".join(refusals) + " yet"
        )
    pending_frontend = None
    if server_args.node_rank == 0:
        host, port = _frontend_addresses(server_args)
        listener = FrontendListener.bind(
            handshake_address=f"tcp://{join_host_port(host, port)}",
            input_address=f"tcp://{join_host_port(host, 0)}",
            output_address=f"tcp://{join_host_port(host, 0)}",
            first_engine_index=server_args.zmq_engine_index,
            num_engines=server_args.mapping.attn.dp_size,
            routing=_frontend_routing(server_args),
            handshake_timeout_s=_FRONTEND_HANDSHAKE_TIMEOUT_S,
            send_timeout_s=_FRONTEND_SEND_TIMEOUT_S,
        )
        bound_host, bound_port = listener.handshake_address[len("tcp://") :].rsplit(
            ":", 1
        )
        server_args.data_parallel_address = bound_host.strip("[]")
        server_args.data_parallel_rpc_port = int(bound_port)
        pending_frontend = listener.accept()
    else:
        host, port = _frontend_addresses(server_args)
        server_args.data_parallel_address = host
        server_args.data_parallel_rpc_port = port

    # A host's control thread runs no Python: the selected output plugins
    # run in the node's relay, which the ranks feed.
    metrics_relay = (
        MetricsRelay(port_args.metrics_producer_configs)
        if port_args.metrics_producer_configs
        else None
    )
    kv_bootstrap_server = _start_kv_bootstrap_server(server_args)

    # A node of rank >= 1 that blocks below is the only owner of its
    # schedulers: they must not outlive it, even when it is SIGKILLed.
    owns_schedulers = (
        server_args.node_rank >= 1 and envs.TOKENSPEED_BLOCK_NONZERO_RANK_CHILDREN.get()
    )
    scheduler_procs, scheduler_pipe_readers = _start_ranks(
        server_args, port_args, metrics_relay, owns_schedulers
    )

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes run no frontend, so they
        # can just wait here.
        if owns_schedulers:
            _stop_schedulers_on_signal(scheduler_procs)

        for data in receive_ready_messages(scheduler_pipe_readers, scheduler_procs):
            if data.get("status") != "ready":
                raise RuntimeError(
                    "Initialization failed. Please see the error messages above."
                )

        if not owns_schedulers:
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(
            server_args.host,
            server_args.port,
            metrics_enabled(server_args, "prometheus"),
        )

        for proc in scheduler_procs:
            proc.join()
            logger.error(f"Engine rank {proc.pid!s} terminated with {proc.exitcode!s}")
        return None, None, None

    # Follower nodes (node_rank >= 1) serve /metrics from the dummy
    # health-check server above; the main node gets the dedicated endpoint.
    launch_metrics_server(server_args)

    # Wait for the model to finish loading: a rank reports ready once it
    # registered with the frontend.
    scheduler_infos = []
    for data in receive_ready_messages(scheduler_pipe_readers, scheduler_procs):
        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)
    frontend = pending_frontend.wait()

    tokenizer_manager = AsyncLLM(server_args, frontend, None)
    tokenizer_manager.bootstrap_server = kv_bootstrap_server
    tokenizer_manager.metrics_relay = metrics_relay
    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    _apply_scheduler_info(tokenizer_manager, scheduler_info)
    return tokenizer_manager, None, scheduler_info


def _apply_scheduler_info(tokenizer_manager: AsyncLLM, scheduler_info: dict) -> None:
    """What the frontend learns from the ranks' ready messages."""
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    tokenizer_manager.max_single_request_tokens = scheduler_info[
        "max_single_request_tokens"
    ]
    tokenizer_manager.context_len = scheduler_info["max_model_len"]
    tokenizer_manager.supports_prompt_logprobs = scheduler_info[
        "supports_prompt_logprobs"
    ]


def _start_encode_ranks(
    server_args: ServerArgs, port_args: PortArgs
) -> tuple[list, list]:
    """Start this node's EPD encode ranks, one ``ts-host`` process each in
    encode mode (``epd/encode_host.py``), and return them with their ready
    pipes. The first rank binds the request channel
    (``port_args.scheduler_input_ipc_name``)."""
    binary = host_binary()
    procs, readers = [], []
    for rank in _scheduler_ranks(server_args):
        rank_server_args = copy.copy(server_args)
        rank_server_args.mapping = copy.deepcopy(server_args.mapping)
        rank_server_args.mapping.rank = rank
        proc, reader = start_rust_host(
            binary, rank_server_args, port_args, contextlib.nullcontext(), None, False
        )
        procs.append(proc)
        readers.append(reader)
    return procs, readers


def headless_encode_channel(server_args: ServerArgs) -> str:
    """The request channel of a headless EPD encode server: the first encode
    rank binds ``tcp://{--data-parallel-address}:{--data-parallel-rpc-port}``
    and the external frontend pushes its ``EncodeRequest``s there."""
    address = join_host_port(
        server_args.data_parallel_address, server_args.data_parallel_rpc_port
    )
    return f"tcp://{address}"


def _launch_encode_server(
    server_args: ServerArgs, port_args: PortArgs
) -> tuple[AsyncLLM, None, dict]:
    """An EPD encode server: its encode ranks (``ts-host`` in encode mode,
    ``epd/encode_host.py``) and, on the first node, the frontend that pushes
    them encode requests. The ranks run the vision tower and ship embeddings
    to the prefill servers; no engine rank runs."""
    procs, readers = _start_encode_ranks(server_args, port_args)
    infos = []
    for data in receive_ready_messages(readers, procs):
        if data.get("status") != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        infos.append(data)
    if server_args.node_rank >= 1:
        if not envs.TOKENSPEED_BLOCK_NONZERO_RANK_CHILDREN.get():
            return None, None, None
        for proc in procs:
            proc.join()
        return None, None, None
    launch_metrics_server(server_args)
    # The first encode rank binds the request channel.
    encode_sender = IpcSender(
        get_zmq_socket(
            zmq.Context(1), zmq.PUSH, port_args.scheduler_input_ipc_name, False
        )
    )
    tokenizer_manager = AsyncLLM(server_args, None, encode_sender)
    _apply_scheduler_info(tokenizer_manager, infos[0])
    return tokenizer_manager, None, infos[0]


# How long a headless launcher waits for its schedulers to exit after asking
# them to, before killing them.
_SCHEDULER_EXIT_GRACE_S = 30.0


def _start_kv_bootstrap_server(server_args: ServerArgs):
    """Start a prefill engine's KV bootstrap server, or return None.

    Decode ranks look up the prefill ranks holding a request's KV through
    this server; the launcher hosts it, one per prefill deployment, on node
    0.
    """
    mode = DisaggregationMode(server_args.disaggregation_mode)
    if mode != DisaggregationMode.PREFILL or server_args.node_rank != 0:
        return None
    # Mooncake is the only transfer backend.
    TransferBackend(server_args.disaggregation_transfer_backend)
    return BootstrapServer(
        server_args.disaggregation_bootstrap_port, cache_layouts=True
    )


def _supervise_schedulers(
    procs: list[mp.Process],
    terminate: Callable[[], None],
    terminate_requested_at: list[float],
    *,
    exit_grace_s: float,
) -> bool:
    """Wait until every scheduler exits; return whether any failed.

    A rank that dies with a nonzero code (e.g. an OOM SIGKILL) takes the
    others down through ``terminate``, which records its first call in
    ``terminate_requested_at``. Schedulers still running ``exit_grace_s``
    after that count as failed and are left for the caller to kill: one
    blocked in a collective or a transfer never acts on SIGTERM.
    """
    failed = False
    alive = list(procs)
    while alive:
        if (
            terminate_requested_at
            and time.monotonic() - terminate_requested_at[0] > exit_grace_s
        ):
            logger.error(
                f"{len(alive):d} scheduler(s) still running {exit_grace_s:.0f}s "
                "after termination; killing them",
            )
            return True
        for proc in list(alive):
            proc.join(timeout=1)
            if proc.exitcode is None:
                continue
            alive.remove(proc)
            if proc.exitcode == 0:
                logger.info(
                    f"scheduler {proc.pid!s} exited with code {proc.exitcode!s}"
                )
            else:
                logger.error(
                    f"scheduler {proc.pid!s} exited with code {proc.exitcode!s}; "
                    "terminating the remaining scheduler(s)",
                )
                failed = True
                terminate()
    return failed


def launch_scheduler_headless(server_args: ServerArgs) -> None:
    """Launch this node's engine ranks for an external frontend (SMG,
    Dynamo) that drives them over msgpack ZMQ.

    Headless: no in-process frontend (the external one tokenizes and
    detokenizes). This process only spawns and supervises the ranks, one
    ``ts-host`` process each, which connect out to the frontend-bound
    handshake/input/output sockets, then blocks until they exit.

    Forces ``--skip-tokenizer-init`` since the frontend passes token ids.
    The ranks dial ``tcp://{--data-parallel-address}:
    {--data-parallel-rpc-port}`` (defaults ``127.0.0.1:30500``; the frontend
    binds it). Both flags always carry values — non-emptiness and the u16
    range are enforced at parse time. With attention DP each rank dials with
    its own engine identity, ``--zmq-engine-index`` plus its DP rank.

    An EPD encode server (``--disaggregation-mode encode``) starts its encode
    ranks instead, and its first rank binds that address as its request
    channel (``headless_encode_channel``): the frontend pushes
    ``EncodeRequest``s to it.
    """
    server_args.skip_tokenizer_init = True

    configure_logger(server_args)
    _set_envs_and_config(server_args)
    scheduler_procs: list[mp.Process] = []
    scheduler_pipe_readers = []
    kv_bootstrap_server = None
    metrics_relay = None

    # SIGUSR1 reports a child's internal failure (a rank's host sends it as
    # it fails), so record the failure here: an exit status alone may not
    # tell the supervisor.
    child_failed = threading.Event()
    # When the first termination was requested; schedulers still running a
    # grace period later are killed with the rest of the tree.
    terminate_requested_at: list[float] = []

    def _terminate_schedulers(signum=None, _frame=None):
        """Forward a shutdown (or child-failure SIGUSR1) to every scheduler."""
        if signum is not None:
            logger.info(f"received signal {signum!s}; terminating scheduler(s)")
        if signum == signal.SIGUSR1:
            child_failed.set()
        if not terminate_requested_at:
            terminate_requested_at.append(time.monotonic())
        for proc in scheduler_procs:
            if proc.is_alive():
                proc.terminate()

    try:
        metrics_service.prepare(server_args)
        # Headless runs no dummy health-check server on any node, so every node
        # serves its own /metrics here.
        launch_metrics_server(server_args)

        # PortArgs is still derived (nccl_port drives torch.distributed); the
        # pickle scheduler_input/tokenizer IPC names are unused in msgpack mode.
        port_args = PortArgs.init_new(server_args)
        logger.info(f"headless server_args={server_args!r}")
        server_args.model, server_args.tokenizer = prepare_model_and_tokenizer(
            server_args.model, server_args.tokenizer
        )

        # Install before spawning so a signal in the launch window is not lost;
        # an unhandled SIGUSR1 would kill this supervisor without cleanup.
        signal.signal(signal.SIGTERM, _terminate_schedulers)
        signal.signal(signal.SIGINT, _terminate_schedulers)
        signal.signal(signal.SIGUSR1, _terminate_schedulers)

        metrics_service.start(server_args, port_args)
        kv_bootstrap_server = _start_kv_bootstrap_server(server_args)
        if server_args.disaggregation_mode == "encode":
            # The encode ranks host the node's EPD bootstrap server themselves.
            port_args.scheduler_input_ipc_name = headless_encode_channel(server_args)
            procs, readers = _start_encode_ranks(server_args, port_args)
        else:
            # Fail before any rank loads the model.
            refusals = rust_host_refusals(server_args)
            if refusals:
                raise NotImplementedError(
                    "The engine host does not support " + ", ".join(refusals) + " yet"
                )
            # A host's control thread runs no Python: the selected output
            # plugins run in the node's relay, which the ranks feed.
            if port_args.metrics_producer_configs:
                metrics_relay = MetricsRelay(port_args.metrics_producer_configs)
            procs, readers = _start_ranks(server_args, port_args, metrics_relay, False)
        scheduler_procs.extend(procs)
        scheduler_pipe_readers.extend(readers)
        for data in receive_ready_messages(scheduler_pipe_readers, scheduler_procs):
            if data.get("status") != "ready":
                raise RuntimeError(
                    "Scheduler initialization failed. See the error messages above."
                )
        if server_args.disaggregation_mode == "encode":
            logger.info(
                "headless encode rank(s) ready; request channel="
                f"{port_args.scheduler_input_ipc_name}",
            )
        else:
            logger.info(
                "headless scheduler(s) ready; SMG handshake endpoint="
                f"{server_args.zmq_handshake_endpoint()!s}",
            )

        failed = _supervise_schedulers(
            scheduler_procs,
            _terminate_schedulers,
            terminate_requested_at,
            exit_grace_s=_SCHEDULER_EXIT_GRACE_S,
        )
        if failed or child_failed.is_set():
            sys.exit(1)
    finally:
        # Backstop: never leak GPU-holding scheduler trees, whichever path
        # (signal, dead rank, readiness failure) unwound this launcher.
        _terminate_schedulers()
        for proc in scheduler_procs:
            proc.join(timeout=5)
        # The ranks sent their last events: the relay flushes its sinks
        # before the process tree, which it belongs to, goes down.
        if metrics_relay is not None:
            metrics_relay.close(timeout=2.0)
        if scheduler_procs:
            kill_process_tree(os.getpid(), include_parent=False)
        if kv_bootstrap_server is not None:
            kv_bootstrap_server.close()
        metrics_service.close(timeout=2.0)


def run_scheduler_headless_from_cli(argv: list[str] | None = None) -> None:
    """CLI shim: ``python -m tokenspeed.runtime.entrypoints.engine <ServerArgs>``."""
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    server_args = prepare_server_args(sys.argv[1:] if argv is None else argv)
    launch_scheduler_headless(server_args)


if __name__ == "__main__":
    run_scheduler_headless_from_cli()
