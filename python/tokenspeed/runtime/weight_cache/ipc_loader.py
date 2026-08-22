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

"""IPC Model Loader — loads model weights from a weight cache daemon via CUDA IPC.

Zero-copy mode: ``param.data`` points directly into IPC-mapped GPU memory. Only
1x GPU memory is needed — the engine and daemon share the same physical GPU
memory via CUDA IPC. The engine therefore depends on the daemon staying alive.
"""

import os
import signal
import stat
import threading
import time

import torch
import torch.nn as nn

from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat
from tokenspeed.runtime.model_loader.loader import (
    BaseModelLoader,
    _initialize_model,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import MultiprocessingSerializer
from tokenspeed.runtime.weight_cache.protocol import (
    CacheConfig,
    check_ipc_quant_support,
    compute_env_stamp,
    get_quant_method_name,
    get_socket_path,
    hash_quant_config,
    recv_msg,
    send_msg,
)

logger = get_colorful_logger(__name__)

# How often the client polls the serving daemon's PID for liveness.
_DAEMON_LIVENESS_POLL_INTERVAL = 5.0


class IpcModelLoader(BaseModelLoader):
    """Load model weights from a weight cache daemon via CUDA IPC handles.

    In daemon mode (``weight_cache_mode="daemon"``), the engine and daemon share
    the same GPU. Falling back to disk loading would cause OOM because both
    processes would hold weights on the same GPU. Therefore daemon mode raises
    an error if the daemon is unavailable instead of falling back.

    In client mode, disk fallback is allowed ONLY when the daemon is genuinely
    absent (its Unix socket file does not exist). Every other failure is a hard
    error rather than a silent fallback, so a broken IPC path never masquerades
    as a healthy (but slow, disk-loaded) server:

    - socket file missing            -> fall back to disk load
    - connection refused             -> raise (daemon crashed after binding)
    - CacheConfig mismatch           -> raise (do NOT disk-load on a shared GPU
                                        holding a different config's weights;
                                        also surfaces fingerprint drift bugs)
    - any protocol / transfer error  -> raise

    See :meth:`_fetch_from_cache` for the authoritative fallback-vs-raise contract.
    """

    def __init__(
        self,
        load_config: LoadConfig,
        socket_path: str | None = None,
        fallback_loader_cls=None,
        weight_cache_mode: str = "client",
        fallback_load_format: str | LoadFormat = LoadFormat.AUTO,
    ) -> None:
        super().__init__(load_config)
        self.socket_path = socket_path
        self.weight_cache_mode = weight_cache_mode
        self._fallback_loader_cls = fallback_loader_cls
        self._fallback_load_format = fallback_load_format

    def download_model(self, model_config) -> None:
        """No-op: the daemon handles its own model downloading."""

    def load_model(self, *, model_config, device_config) -> nn.Module:
        """Load model weights from the weight cache daemon.

        In daemon mode, raises :class:`RuntimeError` if the daemon is
        unavailable (fallback to disk loading would cause OOM on shared GPUs).
        In client mode, falls back to :class:`DefaultModelLoader`.
        """
        tic = time.perf_counter()

        # Derive the per-rank Unix socket path from the engine's global rank
        # when the caller didn't pin one explicitly.
        if self.socket_path is None:
            self.socket_path = get_socket_path(model_config.mapping.rank)

        # Hard-gate unsupported quant methods before touching the daemon, so an
        # unsupported model fails explicitly instead of silently disk-loading
        # (client mode) or serving wrong-numerics IPC weights. Checked here so
        # it applies regardless of whether the daemon is reachable.
        quant_method, engine_quant_config = self._resolve_engine_quant(model_config)
        check_ipc_quant_support(quant_method, engine_quant_config, where="client")

        cache_data = self._fetch_from_cache(model_config)

        if cache_data is None:
            if self.weight_cache_mode == "daemon":
                raise RuntimeError(
                    f"[IpcModelLoader] Weight cache daemon not available at "
                    f"{self.socket_path}. In daemon mode, fallback to disk "
                    f"loading is disabled because the daemon process already "
                    f"holds weights on the same GPU — loading from disk would "
                    f"cause OOM. Please ensure the weight cache daemon is "
                    f"running and the config matches."
                )
            logger.warning(
                "[IpcModelLoader] Weight cache not available or config "
                "mismatch, falling back to disk load"
            )
            return self._fallback_load(model_config, device_config)

        entries = cache_data["entries"]
        logger.info(
            "[IpcModelLoader] Fetched %d IPC handles from daemon in %.2fs",
            len(entries),
            time.perf_counter() - tic,
        )

        model = self._load_zero_copy_mode(model_config, device_config, entries)

        # Skip post-load weight processing: the daemon already ran
        # process_weights_after_loading on the weights before exporting IPC
        # handles. Running it again would double-process (e.g. re-quantize
        # already-quantized weights), corrupting tensor data.

        # Rebuild stale tensor views. Some modules store tensor views as plain
        # attributes (not parameters/buffers) during __init__. When the model
        # is initialized on meta device and then weights are replaced via IPC
        # mapping, those views still point at the old meta storage.
        self._rebuild_stale_views(model)

        # The model now points into the daemon's GPU memory via CUDA IPC. If the
        # daemon dies, those pointers dangle, so watch it and fail loud.
        self._start_daemon_liveness_watchdog(cache_data.get("pid"))

        logger.info(
            "[IpcModelLoader] Loaded model via IPC (mode=%s), total=%.2fs",
            self.weight_cache_mode,
            time.perf_counter() - tic,
        )
        return model.eval()

    def _start_daemon_liveness_watchdog(self, daemon_pid: int | None) -> None:
        """Fail loud if the serving daemon dies while we hold its weights.

        In both client and (engine-spawned) daemon mode, the model's
        ``param.data`` points into the daemon's GPU memory via CUDA IPC, and
        CUDA graphs may capture those addresses. If the daemon exits, the
        pointers dangle: forward passes would read freed GPU memory -> illegal
        address crashes or silent garbage. There is no safe in-place recovery,
        so a background thread polls the daemon PID and, on death, SIGKILLs this
        process with a clear message instead of serving corrupt results.
        """
        if not daemon_pid or daemon_pid <= 0:
            logger.warning(
                "[IpcModelLoader] Daemon did not report a PID; skipping the "
                "daemon-liveness watchdog. A daemon crash will not be detected."
            )
            return

        def _daemon_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True  # exists but owned by another user
            return True

        def _watch() -> None:
            while True:
                time.sleep(_DAEMON_LIVENESS_POLL_INTERVAL)
                if not _daemon_alive(daemon_pid):
                    logger.critical(
                        "[IpcModelLoader] Weight cache daemon (pid=%s) died "
                        "while this engine holds its weights via CUDA IPC. The "
                        "mapped weight pointers are now dangling; continuing "
                        "would read freed GPU memory. Terminating this process.",
                        daemon_pid,
                    )
                    os.kill(os.getpid(), signal.SIGKILL)
                    return

        threading.Thread(
            target=_watch, name="weight-cache-daemon-watchdog", daemon=True
        ).start()
        logger.info(
            "[IpcModelLoader] Started daemon-liveness watchdog for pid=%s",
            daemon_pid,
        )

    @staticmethod
    def _resolve_engine_quant(model_config):
        """Return ``(quant_method, quant_config)`` matching the daemon fingerprint.

        Shared by the IPC allowlist gate and the :class:`CacheConfig`
        fingerprint so the two can never drift apart. ``ModelConfig`` always
        exposes ``hf_config``/``quantization`` directly; ``quantization_config``
        is the only genuinely-optional attribute.
        """
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        quant_method = get_quant_method_name(model_config.quantization)
        if not quant_method and quant_config is not None:
            quant_method = get_quant_method_name(quant_config)
        return quant_method, quant_config

    @staticmethod
    def _rebuild_stale_views(model) -> None:
        """Rebuild tensor views that went stale after IPC weight replacement.

        Extension point: some modules store a ``torch.Tensor`` *view* of a
        parameter as a plain attribute during ``__init__``. After IPC mapping
        replaces the underlying parameter with a new tensor, the old view still
        points at meta-device storage and must be recreated from the now-valid
        parameter. tokenspeed currently registers no such views, so this is a
        no-op; add model-specific rewiring here when a model needs it.
        """
        return

    @staticmethod
    def _set_module_tensor(model, name, tensor, is_param=True) -> None:
        """Replace or register a parameter/buffer in the model by dotted name.

        This is necessary because setting ``param.data`` on a meta-device tensor
        raises a type mismatch error (meta and CUDA tensors have incompatible
        dispatch keys). Instead, we walk the module tree and use ``setattr`` to
        replace the entire parameter/buffer object.

        If the attribute already exists as a parameter/buffer, it is replaced.
        If it doesn't exist (e.g. post-quantization params like ``weight_scale``),
        it is registered as a new parameter or buffer.
        """
        parts = name.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf_name = parts[-1]
        if is_param:
            # requires_grad=False: the IPC memory is shared/read-only and
            # tokenspeed is inference-only, so autograd must never write into it.
            new_param = nn.Parameter(tensor, requires_grad=False)
            setattr(obj, leaf_name, new_param)
        else:
            # register_buffer raises KeyError if the name already exists as a
            # parameter or plain attribute (not a buffer). This happens when
            # process_weights_after_loading converts a parameter to a buffer.
            # Remove the old attribute first.
            if leaf_name in obj._parameters:
                del obj._parameters[leaf_name]
            elif hasattr(obj, leaf_name) and leaf_name not in obj._buffers:
                delattr(obj, leaf_name)
            obj.register_buffer(leaf_name, tensor)

    def _load_zero_copy_mode(self, model_config, device_config, entries) -> nn.Module:
        """Zero-copy load: map IPC tensors directly as ``param.data``.

        The model is initialized on the meta device (no memory allocation), then
        each parameter's data is replaced with the IPC-mapped GPU tensor. The
        engine and daemon share the same physical GPU memory via CUDA IPC.
        """
        from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype

        # Initialize the model on the meta device to avoid any GPU/CPU memory
        # allocation. This creates the model structure with the correct
        # parameter shapes/dtypes but without allocating actual storage.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(model_config, self.load_config)

        # Build lookup dicts of existing parameter/buffer names in the
        # meta-device model. Post-quantization parameters (e.g. weight_scale from
        # FP8) are created by process_weights_after_loading, which the daemon
        # already ran. Those params exist in the daemon's entries but NOT in the
        # meta-device model — we must register them as new attrs. Use dicts (not
        # sets) so we can do O(1) shape/dtype validation without re-traversing
        # the model tree on every lookup. remove_duplicate=False mirrors the
        # daemon's export (which keys tied weights under every name).
        existing_params = {
            name: param
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        existing_buffers = {name: buf for name, buf in model.named_buffers()}
        existing_names = set(existing_params) | set(existing_buffers)

        imported_refs = []
        imported_count = 0
        mismatched: list[str] = []
        new_params_count = 0
        map_tic = time.perf_counter()

        # Iterate over ALL daemon entries (not just model params/buffers). This
        # ensures post-quantization parameters (weight_scale, etc.) that were
        # created by process_weights_after_loading are also mapped.
        for name, entry in entries.items():
            imported_tensor = MultiprocessingSerializer.deserialize(entry["handle"])
            is_param = entry.get("is_param", True)

            if name in existing_names:
                if name in existing_params:
                    ref_param = existing_params[name]
                else:
                    ref_param = existing_buffers[name]
                if (
                    imported_tensor.shape != ref_param.shape
                    or imported_tensor.dtype != ref_param.dtype
                ):
                    mismatched.append(
                        f"  {name}: IPC={imported_tensor.shape}/{imported_tensor.dtype} "
                        f"vs model={ref_param.shape}/{ref_param.dtype}"
                    )
                    del imported_tensor
                    continue

            self._set_module_tensor(model, name, imported_tensor, is_param=is_param)
            imported_refs.append(imported_tensor)
            imported_count += 1

            if name not in existing_names:
                new_params_count += 1

        if mismatched:
            raise RuntimeError(
                f"[IpcModelLoader] {len(mismatched)} tensor(s) have shape/dtype "
                f"mismatch between the IPC daemon and the meta-initialized "
                f"model. The quantization method passed the IPC allowlist gate "
                f"(check_ipc_quant_support), so this is NOT an unsupported-quant "
                f"case — it indicates the daemon's weight fingerprint is "
                f"incomplete or the daemon/client configs drifted (a bug to "
                f"fix), not merely uninitialized weights:\n" + "\n".join(mismatched)
            )

        # After mapping every daemon entry, any tensor still on the meta device
        # is one the daemon did NOT provide. Filling it with torch.empty() would
        # hand the model uninitialized GPU memory — silently producing wrong
        # output, the worst failure mode for a load path. Hard-error and list
        # the offenders instead.
        still_on_meta_params = [
            name
            for name, param in model.named_parameters()
            if param.device.type == "meta"
        ]
        still_on_meta_buffers = [
            name for name, buf in model.named_buffers() if buf.device.type == "meta"
        ]

        if still_on_meta_params or still_on_meta_buffers:
            raise RuntimeError(
                f"[IpcModelLoader] After IPC mapping, "
                f"{len(still_on_meta_params)} parameter(s) and "
                f"{len(still_on_meta_buffers)} buffer(s) remain on the meta "
                f"device — the daemon did not export them. Refusing to fill "
                f"them with uninitialized memory, which would silently produce "
                f"wrong output. This means the daemon's export is incomplete, "
                f"or a recomputable buffer needs explicit recompute logic here.\n"
                f"  params: {still_on_meta_params[:10]}"
                f"{'...' if len(still_on_meta_params) > 10 else ''}\n"
                f"  buffers: {still_on_meta_buffers[:10]}"
                f"{'...' if len(still_on_meta_buffers) > 10 else ''}"
            )

        map_elapsed = time.perf_counter() - map_tic

        # Stash IPC refs on the model to prevent GC (which would unmap the memory).
        if imported_refs:
            model._ipc_imported_tensors = imported_refs

        logger.info(
            "[IpcModelLoader] Zero-copy: mapped %d tensors (%d new post-quant), "
            "time=%.3fs",
            imported_count,
            new_params_count,
            map_elapsed,
        )
        return model

    def _fetch_from_cache(self, model_config) -> dict | None:
        """Connect to daemon, validate config, fetch IPC handles.

        Returns the daemon response dict on success, None if the daemon is
        genuinely absent (socket file doesn't exist). Raises on all other
        failures so they are never silently swallowed as a disk-load fallback.
        """
        import socket as socket_mod

        # Only connect to a real socket node owned by us: reject a symlink, a
        # plain file, or another user's socket planted at this /tmp path. An
        # absent socket means no daemon -> fall back to disk (return None).
        try:
            st = os.lstat(self.socket_path)
        except FileNotFoundError:
            logger.info(
                "[IpcModelLoader] Daemon socket not found at %s.", self.socket_path
            )
            return None
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise RuntimeError(
                f"[IpcModelLoader] Refusing to connect: {self.socket_path} is "
                f"not a socket owned by this user."
            )

        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        try:
            sock.settimeout(30)
            sock.connect(self.socket_path)
        except FileNotFoundError:
            # Raced: socket removed between lstat and connect -> treat as absent.
            sock.close()
            return None
        except ConnectionRefusedError:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Daemon socket exists at {self.socket_path} "
                f"but refused the connection. The daemon may have crashed after "
                f"creating the socket. Check daemon logs."
            )
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[IpcModelLoader] Failed to connect to daemon at "
                f"{self.socket_path}: {e}"
            ) from e

        try:
            engine_config = self._build_engine_config(model_config)
            logger.info(
                "[IpcModelLoader] Requesting weights from daemon at %s with "
                "config: model=%s, arch=%s, world=%d rank=%d, quant=%s, dtype=%s",
                self.socket_path,
                engine_config.model_path,
                engine_config.model_arch,
                engine_config.world_size,
                engine_config.rank,
                engine_config.quant_method,
                engine_config.dtype,
            )

            send_msg(sock, {"type": "fetch_state", "config": engine_config.to_dict()})
            result = recv_msg(sock)

            if result.get("status") != "ok":
                daemon_config = result.get("daemon_config", {})
                raise RuntimeError(
                    f"[IpcModelLoader] Daemon config mismatch!\n"
                    f"  Engine config: {engine_config.to_dict()}\n"
                    f"  Daemon config: {daemon_config}"
                )

            return result
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[IpcModelLoader] Error communicating with daemon at "
                f"{self.socket_path}: {e}"
            ) from e
        finally:
            sock.close()

    def _build_engine_config(self, model_config) -> CacheConfig:
        """Build this engine rank's CacheConfig fingerprint from its mapping."""
        mapping = model_config.mapping
        quant_method, quant_config = self._resolve_engine_quant(model_config)
        return CacheConfig(
            model_path=model_config.model_path,
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
            revision=getattr(model_config, "revision", None) or "",
            **compute_env_stamp(),
        )

    def _fallback_load(self, model_config, device_config) -> nn.Module:
        """Fall back to :class:`DefaultModelLoader` for disk-based loading."""
        from tokenspeed.runtime.model_loader.loader import DefaultModelLoader

        fallback_config = LoadConfig(
            load_format=self._fallback_load_format,
            download_dir=self.load_config.download_dir,
            model_loader_extra_config=self.load_config.model_loader_extra_config,
            ext_yaml=self.load_config.ext_yaml,
        )
        loader_cls = self._fallback_loader_cls or DefaultModelLoader
        fallback = loader_cls(fallback_config)
        return fallback.load_model(
            model_config=model_config, device_config=device_config
        )
