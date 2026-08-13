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

"""Weight Cache Daemon — a persistent process that holds post-quantized,
TP-sharded model weights in GPU memory and serves them via CUDA IPC handles.

Each GPU runs one daemon process for its global rank. The daemon:

1. Loads model weights from disk (full pipeline: disk -> TP shard -> quantize).
2. Exports every parameter/buffer as a CUDA IPC handle.
3. Serves handles over a Unix socket to requesting engine processes.
4. Validates :class:`CacheConfig` compatibility before serving.

Usage::

    # Single-node: launch all ranks with one command.
    python -m tokenspeed.runtime.weight_cache.daemon \\
        --model-path /path/to/model --attn-tp-size 4 \\
        --load-format auto --dtype auto --quantization fp8

    # Multi-node: run on each node with --nnodes / --node-rank.
    python -m tokenspeed.runtime.weight_cache.daemon \\
        --model-path /path/to/model --attn-tp-size 16 \\
        --nnodes 2 --node-rank 0 --dist-init-addr node0-ip:29500

    # Or launch a single daemon for a specific rank:
    python -m tokenspeed.runtime.weight_cache.daemon \\
        --model-path /path/to/model --gpu-id 0 --attn-tp-size 4 --rank 0 \\
        --dist-init-addr 127.0.0.1:29500
"""

import copy
import os
import signal
import socket
import time
from typing import Any

import torch
import torch.distributed as dist

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import MultiprocessingSerializer
from tokenspeed.runtime.weight_cache.protocol import (
    CacheConfig,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    compute_env_stamp,
    get_quant_method_name,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    recv_msg,
    send_msg,
)

logger = get_colorful_logger(__name__)

# Per-connection timeout for the serial serve loop. A client exchange is tiny
# (a config dict + IPC handle metadata), so this generous bound never trips a
# healthy client, yet guarantees one hung/dead peer can't stall the other ranks
# indefinitely.
CLIENT_CONNECTION_TIMEOUT = 30.0


def _kill_itself_when_parent_died() -> None:
    """Best-effort: on Linux, ask the kernel to SIGKILL us if our parent dies.

    Without this an orphaned daemon keeps a full weight copy pinned in GPU
    memory and its live-PID .ready file blocks the next launch — the opposite of
    fast recovery. No-op on non-Linux platforms.
    """
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        logger.warning(
            "[WeightCacheDaemon] Could not install parent-death signal; an "
            "orphaned daemon may need manual cleanup."
        )


class WeightCacheDaemon:
    """Persistent GPU weight cache for a single global rank.

    Holds the complete post-quantization ``state_dict`` in GPU memory and serves
    CUDA IPC handles to engine processes via a Unix socket.
    """

    def __init__(
        self,
        model_path: str,
        rank: int,
        attn_tp_size: int = 1,
        dense_tp_size: int | None = None,
        moe_tp_size: int | None = None,
        ep_size: int = 1,
        dp_size: int = 1,
        nnodes: int = 1,
        node_rank: int = 0,
        base_gpu_id: int = 0,
        gpu_id_step: int = 1,
        load_format: str = "auto",
        dtype: str = "auto",
        quantization: str | None = None,
        trust_remote_code: bool = True,
        revision: str | None = None,
        download_dir: str | None = None,
        device: str = "cuda",
        nccl_port: int | None = None,
        dist_init_addr: str | None = None,
        distributed_timeout_seconds: int = 1800,
    ) -> None:
        self.model_path = model_path
        self.rank = rank
        self.attn_tp_size = attn_tp_size
        self.dense_tp_size = dense_tp_size
        self.moe_tp_size = moe_tp_size
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.nnodes = nnodes
        self.node_rank = node_rank
        self.base_gpu_id = base_gpu_id
        self.gpu_id_step = gpu_id_step
        self.load_format = load_format
        self.dtype = dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.download_dir = download_dir
        self.device = device
        self.nccl_port = nccl_port
        self.dist_init_addr = dist_init_addr
        self.distributed_timeout_seconds = distributed_timeout_seconds

        self.socket_path = get_socket_path(rank)
        self.ready_path = get_ready_path(rank)

        self.model = None
        self.gpu_id = 0
        self.config: CacheConfig | None = None
        # name -> {"handle": base64_str, "shape": list, "dtype": str, "is_param": bool}
        self.state_entries: dict[str, dict[str, Any]] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _build_server_args(self):
        """Build a ServerArgs whose ``__post_init__`` computes the Mapping.

        The daemon reuses the exact same parallelism resolution the engine runs,
        so the shard it loads matches what the engine expects byte-for-byte.
        """
        from tokenspeed.runtime.utils.server_args import ServerArgs

        server_args = ServerArgs(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            dtype=self.dtype,
            quantization=self.quantization,
            load_format=self.load_format,
            download_dir=self.download_dir,
            device=self.device,
            attn_tp_size=self.attn_tp_size,
            dense_tp_size=self.dense_tp_size,
            moe_tp_size=self.moe_tp_size,
            ep_size=self.ep_size,
            data_parallel_size=self.dp_size,
            nnodes=self.nnodes,
            node_rank=self.node_rank,
            base_gpu_id=self.base_gpu_id,
            gpu_id_step=self.gpu_id_step,
            dist_init_addr=self.dist_init_addr,
            enable_memory_saver=False,
        )
        # Pin this daemon to its global rank so the mapping resolves the same
        # per-layer ranks the corresponding engine rank will use.
        server_args.mapping = copy.deepcopy(server_args.mapping)
        server_args.mapping.rank = self.rank
        return server_args

    def _init_distributed(self, server_args, hidden_size: int) -> None:
        """Initialize the distributed backend required for model loading.

        Mirrors the engine worker's bring-up
        (:class:`~tokenspeed.runtime.execution.distributed_initializer.DistributedInitializer`)
        so TP-sharded weight loading forms the same NCCL world the engine will.
        ``hidden_size`` is passed as 0 to skip arming the one-shot all-reduce
        workspaces — a weights-only daemon never runs a forward pass.
        """
        from tokenspeed.runtime.execution.distributed_initializer import (
            DistributedConfig,
            DistributedInitializer,
        )

        mapping = server_args.mapping
        config = DistributedConfig(
            device=self.device,
            gpu_id=self.gpu_id,
            world_size=mapping.world_size,
            global_rank=self.rank,
            local_rank=self.rank % mapping.nprocs_per_node,
            attn_tp_rank=mapping.attn.tp_rank,
            attn_tp_size=mapping.attn.tp_size,
            dp_size=mapping.attn.dp_size,
            dense_tp_size=mapping.dense.tp_size,
            moe_ep_size=mapping.moe.ep_size,
            moe_ep_rank=mapping.moe.ep_rank,
            nccl_port=self.nccl_port or 29500,
            dist_init_addr=self.dist_init_addr,
            distributed_timeout_seconds=self.distributed_timeout_seconds,
            nnodes=mapping.nnodes,
            nprocs_per_node=mapping.nprocs_per_node,
            # 0 -> skip one-shot all-reduce workspace registration (no forward).
            hidden_size=0,
            max_num_tokens=0,
            mapping=mapping,
        )
        DistributedInitializer.initialize(config)

    def load(self) -> None:
        """Full loading pipeline: disk -> TP shard -> quantize -> export IPC."""
        # CUDA IPC weight sharing relies on torch's _share_cuda_ handle export,
        # which only exists on CUDA-alike platforms (CUDA / ROCm). Fail loud
        # here instead of dying deep inside the export with an opaque error.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[WeightCacheDaemon] the weight cache daemon requires a "
                "CUDA-alike platform (CUDA or ROCm) for CUDA IPC weight "
                "sharing, but torch.cuda is not available. Disable the weight "
                "cache (--weight-cache-mode off)."
            )
        # expandable_segments makes torch's caching allocator hand out memory
        # that cannot be exported via _share_cuda_, so the IPC export below would
        # die mid-way with an opaque CUDA error. Fail fast with an actionable
        # message before touching the device.
        self._assert_ipc_compatible_allocator()

        from tokenspeed.runtime.configs.device_config import DeviceConfig
        from tokenspeed.runtime.configs.load_config import LoadConfig
        from tokenspeed.runtime.configs.model_config import ModelConfig
        from tokenspeed.runtime.layers.moe.utils import initialize_moe_config
        from tokenspeed.runtime.model_loader import get_model
        from tokenspeed.runtime.utils import set_cuda_arch
        from tokenspeed.runtime.utils.env import global_server_args_dict_update

        # Reduce thread contention during multi-process loading.
        torch.set_num_threads(1)
        set_cuda_arch()

        server_args = self._build_server_args()
        self.gpu_id = server_args.mapping.gpu_id
        torch.cuda.set_device(self.gpu_id)

        # Mirror the engine's ModelRunner setup so weight loading sees the same
        # global flags / MoE config it would inside a real worker.
        global_server_args_dict_update(server_args)
        initialize_moe_config(server_args)

        model_config = ModelConfig(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            dtype=self.dtype,
            quantization=self.quantization,
            server_args=server_args,
        )

        # Build the cache-config fingerprint BEFORE loading the model. Loading
        # may mutate hf_config.quantization_config (e.g. via
        # process_weights_after_loading), which would produce a different hash
        # than what the engine computes from the original config.
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(
            self.quantization or model_config.quantization
        )
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)

        mapping = server_args.mapping
        self.config = CacheConfig(
            model_path=self.model_path,
            model_arch=(
                model_config.hf_config.architectures[0]
                if getattr(model_config.hf_config, "architectures", None)
                else ""
            ),
            world_size=mapping.world_size,
            rank=mapping.rank,
            attn_tp_size=mapping.attn.tp_size,
            attn_tp_rank=mapping.attn.tp_rank,
            dense_tp_size=mapping.dense.tp_size,
            moe_ep_size=mapping.moe.ep_size,
            moe_ep_rank=mapping.moe.ep_rank,
            dp_size=mapping.attn.dp_size,
            quant_method=quant_method,
            quant_config_hash=hash_quant_config(quant_config),
            dtype=str(model_config.dtype),
            revision=self.revision or "",
            **compute_env_stamp(),
        )

        # Refuse to serve quant methods not verified to round-trip through pure
        # IPC tensor export. Checked before loading so an unsupported model fails
        # fast instead of after minutes of disk I/O.
        check_ipc_quant_support(quant_method, quant_config, where="daemon")

        # Initialize the distributed backend (requires the resolved mapping).
        self._init_distributed(server_args, hidden_size=model_config.hidden_size)

        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
        )

        logger.info(
            "[WeightCacheDaemon gpu=%s rank=%s] Loading model from disk: %s",
            self.gpu_id,
            self.rank,
            self.model_path,
        )
        tic = time.perf_counter()
        self.model = get_model(
            model_config=model_config,
            load_config=load_config,
            device_config=DeviceConfig(self.device, self.gpu_id),
        )
        logger.info(
            "[WeightCacheDaemon gpu=%s rank=%s] Model loaded from disk in %.2fs",
            self.gpu_id,
            self.rank,
            time.perf_counter() - tic,
        )

        # Ensure every post-processing kernel has retired before we export the
        # memory: clients map these tensors read-only via IPC and would
        # otherwise risk observing half-written weights.
        torch.cuda.synchronize()

        self._export_state()
        logger.info(
            "[WeightCacheDaemon gpu=%s rank=%s] Exported %d tensors as IPC "
            "handles. Ready to serve.",
            self.gpu_id,
            self.rank,
            len(self.state_entries),
        )

    @staticmethod
    def _assert_ipc_compatible_allocator() -> None:
        """Reject allocator configs incompatible with CUDA IPC export.

        The expandable-segments allocator returns memory that cannot be shared
        through torch's ``_share_cuda_`` handle, which would make the export
        fail partway with an opaque error. Detect it up front and fail loud.
        """
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            conf = os.environ.get(var, "")
            for field in conf.split(","):
                key, _, value = field.partition(":")
                if (
                    key.strip() == "expandable_segments"
                    and value.strip().lower() == "true"
                ):
                    raise RuntimeError(
                        f"[WeightCacheDaemon] {var} sets expandable_segments:True, "
                        f"which is incompatible with CUDA IPC weight sharing: the "
                        f"expandable-segments allocator hands out memory that "
                        f"cannot be exported via _share_cuda_, so the IPC handle "
                        f"export would fail mid-way. Unset expandable_segments "
                        f"for the weight cache daemon process (it can stay "
                        f"enabled for the engine itself)."
                    )

    def _export_state(self) -> None:
        """Export model parameters and buffers as CUDA IPC handles.

        This includes both persistent buffers (in ``state_dict``) and
        non-persistent buffers (e.g. rotary embedding ``cos_sin_cache``) so the
        engine can fully reconstruct the model state via zero-copy IPC.
        """
        self.state_entries.clear()

        # remove_duplicate=False so tied weights are recognized as parameters
        # under every name. state_dict() below emits both tied keys, and with a
        # deduped set the duplicate would be mis-registered as a buffer, not a
        # parameter, on the client.
        param_names = set(
            name for name, _ in self.model.named_parameters(remove_duplicate=False)
        )
        state_dict_names = set(self.model.state_dict().keys())

        # Export all items from state_dict (parameters + persistent buffers).
        for name, tensor in self.model.state_dict().items():
            ipc_handle = MultiprocessingSerializer.serialize(
                tensor.data, output_str=True
            )
            self.state_entries[name] = {
                "handle": ipc_handle,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": name in param_names,
            }

        # Also export non-persistent buffers (not in state_dict but needed for
        # inference, e.g. rotary embedding cos_sin_cache).
        non_persistent_count = 0
        for name, buf in self.model.named_buffers():
            if name not in state_dict_names:
                ipc_handle = MultiprocessingSerializer.serialize(
                    buf.data, output_str=True
                )
                self.state_entries[name] = {
                    "handle": ipc_handle,
                    "shape": list(buf.shape),
                    "dtype": str(buf.dtype).replace("torch.", ""),
                    "is_param": False,
                }
                non_persistent_count += 1

        total_bytes = sum(
            len(entry["handle"]) if hasattr(entry["handle"], "__len__") else 0
            for entry in self.state_entries.values()
        )
        logger.info(
            "[WeightCacheDaemon gpu=%s] Exported %d tensors (%d non-persistent "
            "buffers), serialized handle size ~%.1f MB",
            self.gpu_id,
            len(self.state_entries),
            non_persistent_count,
            total_bytes / 1024 / 1024,
        )

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------
    def serve(self) -> None:
        """Block and serve IPC handles over a Unix socket."""
        # Do NOT unlink an existing socket here: stale-file cleanup is the launch
        # path's job (cleanup_stale_daemon_files refuses to remove a socket whose
        # .ready still points at a live PID). A leftover live socket makes bind()
        # fail loudly instead of silently stealing another daemon's socket.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            sock.bind(self.socket_path)
        finally:
            os.umask(old_umask)
        sock.listen(8)
        sock.settimeout(1.0)  # Allow periodic shutdown check.

        with open(self.ready_path, "w") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"config={self.config.to_dict()}\n")

        logger.info(
            "[WeightCacheDaemon gpu=%s] Listening on %s",
            self.gpu_id,
            self.socket_path,
        )

        self._running = True

        def _signal_handler(signum, frame):
            logger.info(
                "[WeightCacheDaemon gpu=%s] Received signal %s, shutting down",
                self.gpu_id,
                signum,
            )
            self._running = False

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            while self._running:
                try:
                    conn, _ = sock.accept()
                    # The listen-socket timeout above only bounds accept(); the
                    # accepted connection is blocking by default. Since we serve
                    # connections serially, a client that connects but never
                    # sends (or dies mid-send) would block recv_msg forever and
                    # stall every other engine rank. Bound each exchange instead.
                    conn.settimeout(CLIENT_CONNECTION_TIMEOUT)
                    try:
                        self._handle_connection(conn)
                    except Exception as e:
                        logger.error(
                            "[WeightCacheDaemon gpu=%s] Error handling "
                            "connection: %s",
                            self.gpu_id,
                            e,
                            exc_info=True,
                        )
                    finally:
                        conn.close()
                except socket.timeout:
                    continue
        finally:
            sock.close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(self.ready_path):
                os.unlink(self.ready_path)
            logger.info("[WeightCacheDaemon gpu=%s] Shutdown complete", self.gpu_id)

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single client connection."""
        req = recv_msg(conn)

        if req.get("type") == "query_config":
            send_msg(conn, {"status": "ok", "config": self.config.to_dict()})

        elif req.get("type") == "fetch_state":
            engine_config = CacheConfig.from_dict(req["config"])
            if not self.config.matches(engine_config):
                daemon_dict = self.config.to_dict()
                engine_dict = engine_config.to_dict()
                mismatches = {
                    k: (daemon_dict.get(k), engine_dict.get(k))
                    for k in daemon_dict
                    if daemon_dict.get(k) != engine_dict.get(k)
                }
                logger.warning(
                    "[WeightCacheDaemon gpu=%s] Config mismatch: %s",
                    self.gpu_id,
                    mismatches,
                )
                send_msg(
                    conn, {"status": "mismatch", "daemon_config": self.config.to_dict()}
                )
                return

            logger.info(
                "[WeightCacheDaemon gpu=%s] Serving %d IPC handles to engine",
                self.gpu_id,
                len(self.state_entries),
            )
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": self.config.to_dict(),
                    "entries": self.state_entries,
                    # PID so the client can watch daemon liveness: if this
                    # process dies while clients hold IPC mappings, their
                    # param.data (and any CUDA-graph-captured addresses) dangle.
                    "pid": os.getpid(),
                },
            )

        elif req.get("type") == "ping":
            send_msg(conn, {"status": "ok"})

        else:
            send_msg(
                conn,
                {
                    "status": "error",
                    "message": f"Unknown request type: {req.get('type')}",
                },
            )

    def shutdown(self) -> None:
        """Release GPU memory and clean up."""
        if dist.is_initialized():
            dist.destroy_process_group()
        if self.model is not None:
            del self.model
            self.model = None
        self.state_entries.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._running = False


def run_weight_cache_daemon(
    model_path: str,
    rank: int,
    attn_tp_size: int = 1,
    dense_tp_size: int | None = None,
    moe_tp_size: int | None = None,
    ep_size: int = 1,
    dp_size: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    base_gpu_id: int = 0,
    gpu_id_step: int = 1,
    load_format: str = "auto",
    dtype: str = "auto",
    quantization: str | None = None,
    trust_remote_code: bool = True,
    revision: str | None = None,
    download_dir: str | None = None,
    device: str = "cuda",
    nccl_port: int | None = None,
    dist_init_addr: str | None = None,
) -> None:
    """Entry point for running a single weight cache daemon process."""
    _kill_itself_when_parent_died()

    daemon = WeightCacheDaemon(
        model_path=model_path,
        rank=rank,
        attn_tp_size=attn_tp_size,
        dense_tp_size=dense_tp_size,
        moe_tp_size=moe_tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        nnodes=nnodes,
        node_rank=node_rank,
        base_gpu_id=base_gpu_id,
        gpu_id_step=gpu_id_step,
        load_format=load_format,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
        revision=revision,
        download_dir=download_dir,
        device=device,
        nccl_port=nccl_port,
        dist_init_addr=dist_init_addr,
    )
    daemon.load()
    daemon.serve()


def launch_weight_cache_daemons(
    model_path: str,
    attn_tp_size: int = 1,
    dense_tp_size: int | None = None,
    moe_tp_size: int | None = None,
    ep_size: int = 1,
    dp_size: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    base_gpu_id: int = 0,
    gpu_id_step: int = 1,
    load_format: str = "auto",
    dtype: str = "auto",
    quantization: str | None = None,
    trust_remote_code: bool = True,
    revision: str | None = None,
    download_dir: str | None = None,
    device: str = "cuda",
    dist_init_addr: str | None = None,
    nccl_port: int | None = None,
    timeout: int = 1800,
    force: bool = False,
    wait: bool = True,
) -> list:
    """Launch weight cache daemon processes for this node's ranks.

    Spawns one daemon per local rank via ``subprocess.Popen`` (rather than
    ``multiprocessing.Process``) to avoid initializing CUDA in the parent
    process, which can degrade CUDA IPC performance in child processes.

    Returns the list of :class:`subprocess.Popen` handles. When ``wait`` is
    True this blocks until every local daemon reports ready (or raises on
    timeout / premature exit); the returned processes keep running afterwards.
    """
    import socket as sock_mod
    import subprocess
    import sys

    # Resolve the parallelism topology the same way ServerArgs would so we can
    # enumerate this node's ranks and their GPU ids.
    from tokenspeed.runtime.utils.server_args import ServerArgs

    probe = ServerArgs(
        model=model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        quantization=quantization,
        load_format=load_format,
        attn_tp_size=attn_tp_size,
        dense_tp_size=dense_tp_size,
        moe_tp_size=moe_tp_size,
        ep_size=ep_size,
        data_parallel_size=dp_size,
        nnodes=nnodes,
        node_rank=node_rank,
        base_gpu_id=base_gpu_id,
        gpu_id_step=gpu_id_step,
        dist_init_addr=dist_init_addr,
        enable_memory_saver=False,
    )
    mapping = probe.mapping
    nprocs_per_node = mapping.nprocs_per_node
    rank_start = nprocs_per_node * node_rank
    rank_end = rank_start + nprocs_per_node
    local_ranks = list(range(rank_start, rank_end))

    if nnodes > 1 and dist_init_addr is None:
        raise ValueError(
            "dist_init_addr is required for multi-node weight cache daemons. "
            "Use --dist-init-addr <node0-ip>:<port> to specify the rendezvous "
            "address accessible from all nodes."
        )

    # Auto-allocate a free NCCL port for single-node rendezvous.
    if nccl_port is None and dist_init_addr is None:
        with sock_mod.socket(sock_mod.AF_INET, sock_mod.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            nccl_port = s.getsockname()[1]

    python_path = sys.executable
    daemon_module = "tokenspeed.runtime.weight_cache.daemon"

    # Validate and clean up stale .ready/.sock files from prior runs.
    for rank in local_ranks:
        cleanup_stale_daemon_files(rank, force=force)

    procs = []
    for rank in local_ranks:
        cmd = [
            python_path,
            "-m",
            daemon_module,
            "--model-path",
            model_path,
            "--rank",
            str(rank),
            "--attn-tp-size",
            str(attn_tp_size),
            "--ep-size",
            str(ep_size),
            "--dp-size",
            str(dp_size),
            "--nnodes",
            str(nnodes),
            "--node-rank",
            str(node_rank),
            "--base-gpu-id",
            str(base_gpu_id),
            "--gpu-id-step",
            str(gpu_id_step),
            "--load-format",
            load_format,
            "--dtype",
            dtype,
            "--device",
            device,
        ]
        if dense_tp_size is not None:
            cmd += ["--dense-tp-size", str(dense_tp_size)]
        if moe_tp_size is not None:
            cmd += ["--moe-tp-size", str(moe_tp_size)]
        if quantization:
            cmd += ["--quantization", quantization]
        if revision:
            cmd += ["--revision", revision]
        if download_dir:
            cmd += ["--download-dir", download_dir]
        if trust_remote_code:
            cmd += ["--trust-remote-code"]
        if dist_init_addr:
            cmd += ["--dist-init-addr", dist_init_addr]
        if nccl_port is not None:
            cmd += ["--nccl-port", str(nccl_port)]

        proc = subprocess.Popen(cmd)
        procs.append(proc)
        logger.info("Launched weight cache daemon rank=%s pid=%s", rank, proc.pid)

    if wait:
        _wait_for_ready(procs, local_ranks, timeout=timeout)

    return procs


def _wait_for_ready(procs: list, local_ranks: list, *, timeout: int) -> None:
    """Block until every local daemon writes its .ready file or fail loudly."""
    start_time = time.time()
    for rank in local_ranks:
        ready_path = get_ready_path(rank)
        while not os.path.exists(ready_path):
            time.sleep(2)
            if time.time() - start_time > timeout:
                for p in procs:
                    p.terminate()
                raise TimeoutError(
                    f"Weight cache daemon rank={rank} did not become ready "
                    f"within {timeout}s"
                )
            for p in procs:
                retcode = p.poll()
                if retcode is not None:
                    for other in procs:
                        if other.poll() is None:
                            other.terminate()
                    raise RuntimeError(
                        f"Weight cache daemon exited prematurely with code {retcode}"
                    )
        logger.info("Weight cache daemon rank=%s is ready", rank)

    logger.info("All %d weight cache daemons on this node are ready", len(procs))


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="TokenSpeed Weight Cache Daemon")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--attn-tp-size", type=int, default=1, help="Attention TP size")
    parser.add_argument("--dense-tp-size", type=int, default=None, help="Dense TP size")
    parser.add_argument("--moe-tp-size", type=int, default=None, help="MoE TP size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--nnodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of this node (0-indexed)"
    )
    parser.add_argument(
        "--base-gpu-id",
        type=int,
        default=0,
        help="GPU id of this node's first rank (mirrors the engine's "
        "--base-gpu-id).",
    )
    parser.add_argument(
        "--gpu-id-step",
        type=int,
        default=1,
        help="Stride between consecutive ranks' GPU ids (mirrors the engine's "
        "--gpu-id-step).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank for a single daemon. If omitted, launches daemons "
        "for all local ranks.",
    )
    parser.add_argument("--load-format", default="auto", help="Weight load format")
    parser.add_argument("--dtype", default="auto", help="Model dtype")
    parser.add_argument("--quantization", default=None, help="Quantization method")
    parser.add_argument("--revision", default=None, help="Model revision")
    parser.add_argument("--download-dir", default=None, help="Weight download dir")
    parser.add_argument("--device", default="cuda", help="Device type")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help="Distributed init address (e.g. node0-ip:29500). Required for "
        "multi-node (nnodes > 1) and must be reachable from all nodes.",
    )
    parser.add_argument(
        "--nccl-port",
        type=int,
        default=None,
        help="NCCL rendezvous port for single-node runs. Auto-assigned by the "
        "multi-rank launcher.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Seconds to wait for all daemons to become ready (default: 1800)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Take over a rank whose .ready file still points at a live PID by "
        "killing that daemon (use to reclaim a wedged/orphaned daemon).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.rank is not None:
        # Single-rank mode: launch one daemon for the specified global rank.
        cleanup_stale_daemon_files(args.rank, force=args.force)
        run_weight_cache_daemon(
            model_path=args.model_path,
            rank=args.rank,
            attn_tp_size=args.attn_tp_size,
            dense_tp_size=args.dense_tp_size,
            moe_tp_size=args.moe_tp_size,
            ep_size=args.ep_size,
            dp_size=args.dp_size,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            base_gpu_id=args.base_gpu_id,
            gpu_id_step=args.gpu_id_step,
            load_format=args.load_format,
            dtype=args.dtype,
            quantization=args.quantization,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            download_dir=args.download_dir,
            device=args.device,
            nccl_port=args.nccl_port,
            dist_init_addr=args.dist_init_addr,
        )
    else:
        # Multi-rank mode: launch daemons for all of this node's ranks and block.
        procs = launch_weight_cache_daemons(
            model_path=args.model_path,
            attn_tp_size=args.attn_tp_size,
            dense_tp_size=args.dense_tp_size,
            moe_tp_size=args.moe_tp_size,
            ep_size=args.ep_size,
            dp_size=args.dp_size,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            base_gpu_id=args.base_gpu_id,
            gpu_id_step=args.gpu_id_step,
            load_format=args.load_format,
            dtype=args.dtype,
            quantization=args.quantization,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            download_dir=args.download_dir,
            device=args.device,
            dist_init_addr=args.dist_init_addr,
            nccl_port=args.nccl_port,
            timeout=args.timeout,
            force=args.force,
            wait=True,
        )
        # Keep the launcher alive so the daemons (and their GPU memory) persist,
        # and so parent-death cleanup propagates to the children.
        _monitor_until_exit(procs)


def _monitor_until_exit(procs: list) -> None:
    import subprocess

    exited = None
    try:
        while exited is None:
            for proc in procs:
                if proc.poll() is not None:
                    exited = proc
                    break
            else:
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down daemons")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        logger.info("All weight cache daemons have been terminated")

    if exited is not None:
        raise RuntimeError(
            f"Weight cache daemon (pid={exited.pid}) exited with code "
            f"{exited.returncode}; terminated the remaining daemons."
        )


if __name__ == "__main__":
    main()
