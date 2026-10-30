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

"""The engine launcher: starts a node's engine ranks, one ``ts-host`` process
each, for an external frontend (SMG, Dynamo) that drives them over the
msgpack ZMQ wire, and supervises them.

``python -m tokenspeed.runtime.entrypoints.engine <ServerArgs>`` and ``ts
serve --headless`` run it.
"""

import contextlib
import copy
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections.abc import Callable

import psutil
from tokenspeed_engine import BootstrapServer

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.engine.rust_host import (
    host_binary,
    rust_host_refusals,
    start_rust_host,
)
from tokenspeed.runtime.metrics.config import metrics_enabled
from tokenspeed.runtime.metrics.relay import MetricsRelay
from tokenspeed.runtime.metrics.service import metrics_service
from tokenspeed.runtime.pd.utils import DisaggregationMode, TransferBackend
from tokenspeed.runtime.utils import (
    configure_logger,
    get_colorful_logger,
    launch_metrics_server,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tokenspeed.runtime.utils.launcher import (
    interface_for_host,
    set_pinned_allocator_default,
)
from tokenspeed.runtime.utils.process import (
    kill_process_tree,
    receive_ready_messages,
    register_child_failure_signals,
)
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


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
    if server_args.device == "cuda":
        set_pinned_allocator_default()
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

    # Let the launcher own process-tree cleanup. A signal callback must not
    # log or acquire locks during an interrupted call.
    register_child_failure_signals()

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _scheduler_ranks(server_args: ServerArgs) -> range:
    """Global ranks of the scheduler processes this node launches."""
    if server_args.emulate_rank_zero:
        return range(1)
    rank_start = server_args.mapping.nprocs_per_node * server_args.node_rank
    return range(rank_start, rank_start + server_args.mapping.nprocs_per_node)


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


def _start_encode_ranks(
    server_args: ServerArgs, port_args: PortArgs, parent_death_signal: bool
) -> tuple[list, list]:
    """Start this node's EPD encode ranks, one ``ts-host`` process each in
    encode mode (``epd/encode_host.py``), and return them with their ready
    pipes. The first rank dials the frontend's handshake
    (``server_args.zmq_handshake_endpoint()``), which then sends it the
    encode requests. ``parent_death_signal`` as for ``_start_ranks``."""
    binary = host_binary()
    procs, readers = [], []
    for rank in _scheduler_ranks(server_args):
        rank_server_args = copy.copy(server_args)
        rank_server_args.mapping = copy.deepcopy(server_args.mapping)
        rank_server_args.mapping.rank = rank
        proc, reader = start_rust_host(
            binary,
            rank_server_args,
            port_args,
            contextlib.nullcontext(),
            None,
            parent_death_signal,
        )
        procs.append(proc)
        readers.append(reader)
    return procs, readers


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

    The frontend tokenizes and detokenizes. This process only spawns and
    supervises the ranks, one ``ts-host`` process each, which connect out to
    the frontend-bound handshake/input/output sockets, then blocks until
    they exit.

    The ranks dial ``tcp://{--data-parallel-address}:
    {--data-parallel-rpc-port}`` (defaults ``127.0.0.1:30500``; the frontend
    binds it). Both flags always carry values — non-emptiness and the u16
    range are enforced at parse time. With attention DP each rank dials with
    its own engine identity, ``--zmq-engine-index`` plus its DP rank.

    An EPD encode server (``--disaggregation-mode encode``) starts its encode
    ranks instead; their first rank dials the same address and takes encode
    requests.
    """
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
    # The processes the schedulers started, listed before they are asked to
    # stop: one whose scheduler exits first is no longer below this node, and
    # is killed by this list at the end.
    scheduler_descendants: list[psutil.Process] = []

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
                with contextlib.suppress(psutil.NoSuchProcess):
                    scheduler_descendants.extend(
                        psutil.Process(proc.pid).children(recursive=True)
                    )
                proc.terminate()

    try:
        metrics_service.prepare(server_args)
        # Every node serves its own node's /metrics.
        launch_metrics_server(server_args)

        # PortArgs is still derived (nccl_port drives torch.distributed); its
        # IPC names are unused in msgpack mode.
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
        # Nothing else stops a follower node's ranks, so they must not outlive
        # it, even when it is SIGKILLed; the first node's frontend notices.
        parent_death_signal = server_args.node_rank >= 1
        if server_args.disaggregation_mode == "encode":
            # The encode ranks host the node's EPD bootstrap server themselves.
            procs, readers = _start_encode_ranks(
                server_args, port_args, parent_death_signal
            )
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
            procs, readers = _start_ranks(
                server_args, port_args, metrics_relay, parent_death_signal
            )
        scheduler_procs.extend(procs)
        scheduler_pipe_readers.extend(readers)
        for data in receive_ready_messages(scheduler_pipe_readers, scheduler_procs):
            if data.get("status") != "ready":
                raise RuntimeError(
                    "Scheduler initialization failed. See the error messages above."
                )
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
        for descendant in scheduler_descendants:
            with contextlib.suppress(psutil.NoSuchProcess):
                descendant.kill()
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
