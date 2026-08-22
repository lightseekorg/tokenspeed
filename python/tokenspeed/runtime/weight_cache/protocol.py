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

"""Protocol definitions for the weight cache daemon.

Defines :class:`CacheConfig` (a fingerprint used to validate that a daemon's
cached weights are compatible with a requesting engine process) and the small
length-prefixed socket message protocol the daemon and its clients share.

Kept intentionally dependency-light so it stays cheap to import and usable on
CPU-only hosts: ``torch`` is imported lazily inside :func:`compute_env_stamp`.
"""

import hashlib
import json
import os
import pickle
import signal
import struct
from typing import Any

import msgspec

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# Socket / ready-file path templates for weight cache daemons, keyed by the
# global rank (``mapping.rank``) so multi-node / multi-DP layouts don't collide.
WEIGHT_CACHE_SOCKET_TEMPLATE = "/tmp/tokenspeed_weight_cache_rank{rank}.sock"
WEIGHT_CACHE_READY_TEMPLATE = "/tmp/tokenspeed_weight_cache_rank{rank}.ready"


class CacheConfig(msgspec.Struct):
    """Fingerprint of the cached weights.

    Used to validate compatibility between a daemon's cached state and a
    requesting engine process. Any mismatch triggers a fallback to disk
    loading (client mode) or a hard error (daemon mode). The parallelism
    fields mirror tokenspeed's :class:`~tokenspeed.runtime.distributed.mapping.Mapping`
    so a shard produced under one topology is never handed to an engine that
    expects a different one.
    """

    model_path: str
    model_arch: str
    # Parallelism topology (tokenspeed Mapping): a shard is uniquely identified
    # by the world size + global rank, but the individual sizes are compared too
    # so a config drift surfaces as a readable mismatch rather than a silent one.
    world_size: int
    rank: int
    attn_tp_size: int
    attn_tp_rank: int
    dense_tp_size: int
    moe_ep_size: int
    moe_ep_rank: int
    dp_size: int
    quant_method: str  # e.g. "fp8", "" for unquantized
    quant_config_hash: str  # SHA-256 hash of the quantization config
    dtype: str  # e.g. "torch.bfloat16"
    revision: str  # model revision the weights were loaded from ("" if unset)
    # Environment stamp: a daemon and a client that ran different post-processing
    # branches (different GPU compute capability or torch/kernel version) can
    # produce incompatible weights that would map cleanly yet serve garbage.
    # Comparing these turns that into a clean mismatch. See compute_env_stamp().
    device_capability: str  # local compute capability, e.g. "9.0" ("" if N/A)
    torch_version: str  # torch.__version__ of the process that built the weights

    def matches(self, other: "CacheConfig") -> bool:
        """Check if two configs are compatible for weight sharing."""
        return self == other

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__struct_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CacheConfig":
        return cls(**{k: d[k] for k in cls.__struct_fields__ if k in d})


def hash_quant_config(quant_config: Any) -> str:
    """Compute a stable hash of the quantization config.

    Avoids ``str()``/``repr()`` on arbitrary objects because those embed memory
    addresses (e.g. ``"at 0x7f..."``), producing different hashes across
    processes and causing a permanent, spurious config mismatch.
    """
    if quant_config is None:
        return ""
    try:
        if hasattr(quant_config, "to_dict"):
            config_str = json.dumps(quant_config.to_dict(), sort_keys=True)
        elif isinstance(quant_config, dict):
            config_str = json.dumps(quant_config, sort_keys=True)
        elif hasattr(quant_config, "__dict__"):
            config_str = (
                type(quant_config).__name__
                + ":"
                + json.dumps(
                    {
                        k: v
                        for k, v in sorted(quant_config.__dict__.items())
                        if not k.startswith("_")
                        and isinstance(
                            v, (str, int, float, bool, type(None), list, dict)
                        )
                    },
                    sort_keys=True,
                )
            )
        else:
            config_str = type(quant_config).__name__
        return hashlib.sha256(config_str.encode()).hexdigest()
    except Exception:
        config_str = type(quant_config).__name__
        return hashlib.sha256(config_str.encode()).hexdigest()


def get_quant_method_name(quant_config: Any) -> str:
    """Extract the quantization method name from a config object or dict."""
    if quant_config is None:
        return ""
    if isinstance(quant_config, str):
        return quant_config
    if isinstance(quant_config, dict):
        for key in ("quant_method", "quant_algo", "method"):
            value = quant_config.get(key)
            if value:
                return str(value)
        return ""
    if hasattr(quant_config, "get_name"):
        return quant_config.get_name()
    if hasattr(quant_config, "name"):
        return quant_config.name
    return type(quant_config).__name__


# ---------------------------------------------------------------------------
# IPC quantization-method allowlist
# ---------------------------------------------------------------------------
#
# CUDA IPC zero-copy sharing exports ONLY raw tensor data, so it is correct only
# when process_weights_after_loading's entire effect is captured by that data.
# Methods that stamp Python-side metadata or repack/transpose weights into
# shapes the meta-init client can't reproduce (per-tensor FP8, Marlin, AWQ/GPTQ)
# would serve silently-wrong numerics. Only methods verified to round-trip
# through pure tensor export are allowed; every other method hard-errors. Extend
# the registry below only after verifying a method end-to-end.


class UnsupportedQuantForIPCError(RuntimeError):
    """Raised when a quantization method is not on the verified allowlist for
    CUDA IPC zero-copy weight sharing."""


def _get_quant_field(quant_config: Any, key: str) -> Any:
    """Read a field from a quant config that may be a dict or an object."""
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        return quant_config.get(key)
    return getattr(quant_config, key, None)


def _fp8_round_trips_via_ipc(quant_config: Any) -> bool:
    """Only block-wise FP8 is verified.

    Block-wise FP8 (``weight_block_size`` set) preserves weight shape and the
    only post-load metadata it stamps is accounted for. Per-tensor FP8
    transposes ``layer.weight`` during post-processing, a shape change the
    meta-init client cannot reproduce, so it is not supported.
    """
    return _get_quant_field(quant_config, "weight_block_size") is not None


# quant_method name -> predicate(quant_config) -> bool (True == verified safe).
# A method absent from this registry is unsupported and hard-errors.
IPC_QUANT_ALLOWLIST = {
    "": lambda _quant_config: True,  # unquantized
    "fp8": _fp8_round_trips_via_ipc,  # only block-wise FP8 verified
}


def is_ipc_quant_supported(quant_method: str, quant_config: Any) -> bool:
    """Return True if ``quant_method`` is verified safe for IPC zero-copy sharing."""
    predicate = IPC_QUANT_ALLOWLIST.get(quant_method)
    if predicate is None:
        return False
    return bool(predicate(quant_config))


def check_ipc_quant_support(
    quant_method: str, quant_config: Any, *, where: str
) -> None:
    """Hard-error unless ``quant_method`` is verified safe for IPC zero-copy sharing.

    ``where`` is a short tag (e.g. ``"daemon"``/``"client"``) used only in the
    error message. Raises :class:`UnsupportedQuantForIPCError` with an
    actionable message.
    """
    if is_ipc_quant_supported(quant_method, quant_config):
        return
    verified = ", ".join(
        (repr(m) if m else "'' (unquantized)") for m in IPC_QUANT_ALLOWLIST
    )
    raise UnsupportedQuantForIPCError(
        f"[weight_cache:{where}] quantization method {quant_method!r} is not "
        f"verified for CUDA IPC zero-copy weight sharing. Its "
        f"process_weights_after_loading may stamp Python-side metadata or "
        f"repack/transpose weights into shapes the meta-initialized client "
        f"cannot reproduce, which would silently serve wrong-numerics weights. "
        f"Verified methods: {verified}. Note: FP8 is only verified for "
        f"block-wise configs (weight_block_size set), not per-tensor FP8. "
        f"Disable the weight cache (--weight-cache-mode off) for this model."
    )


# ---------------------------------------------------------------------------
# Socket protocol helpers
# ---------------------------------------------------------------------------

MAX_MSG_SIZE = 256 * 1024 * 1024  # 256 MiB


def send_msg(sock, obj: Any) -> None:
    """Send a length-prefixed pickled message over a socket."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_msg(sock) -> Any:
    """Receive a length-prefixed pickled message from a socket.

    The peer is a Unix-domain socket whose ownership is validated (uid + socket
    type) by the client before connecting and created with a restrictive umask
    by the daemon, so only same-user processes can talk on it.
    """
    header = _recv_exact(sock, 4)
    if header is None:
        raise ConnectionError("Connection closed while reading message header")
    length = struct.unpack("!I", header)[0]
    if length > MAX_MSG_SIZE:
        raise ValueError(f"Message size {length} exceeds {MAX_MSG_SIZE} byte cap")
    data = _recv_exact(sock, length)
    if data is None:
        raise ConnectionError("Connection closed while reading message body")
    return pickle.loads(data)


def _recv_exact(sock, n: int) -> bytes | None:
    """Receive exactly n bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def compute_env_stamp() -> dict[str, str]:
    """Local environment fingerprint for the IPC weight cache.

    Returns the device compute capability and torch version of the current
    process. A daemon and a connecting client that differ on either may have
    run different post-processing / kernel-selection branches, producing
    weights that map cleanly over IPC yet serve garbage; stamping these into
    :class:`CacheConfig` turns that into a clean mismatch. Imported lazily so
    this module stays cheap to import and usable on CPU-only hosts (both fields
    degrade to "").
    """
    device_capability = ""
    torch_version = ""
    try:
        import torch

        torch_version = str(torch.__version__)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            device_capability = f"{major}.{minor}"
    except Exception:
        pass
    return {
        "device_capability": device_capability,
        "torch_version": torch_version,
    }


def get_socket_path(rank: int) -> str:
    """Get the Unix socket path for the weight cache daemon of a global rank."""
    return WEIGHT_CACHE_SOCKET_TEMPLATE.format(rank=rank)


def get_ready_path(rank: int) -> str:
    """Get the ready-file path for the weight cache daemon of a global rank."""
    return WEIGHT_CACHE_READY_TEMPLATE.format(rank=rank)


def _read_ready_pid(ready_path: str) -> int | None:
    """Read the daemon PID from a .ready file. Returns None if unreadable."""
    try:
        with open(ready_path) as f:
            for line in f:
                if line.startswith("pid="):
                    return int(line.strip().split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cleanup_stale_daemon_files(rank: int, *, force: bool = False) -> None:
    """Validate and clean up .ready/.sock files for a daemon rank.

    If the .ready file exists and the recorded PID is still alive, the daemon
    is still running — raise :class:`RuntimeError` so the caller doesn't clobber
    it, unless ``force`` is set, in which case the running daemon is killed and
    its files are taken over (stale-takeover path for a wedged/orphaned daemon).
    If the PID is dead (or unreadable), the files are stale leftovers from a
    crashed/killed daemon and are safe to remove.
    """
    ready_path = get_ready_path(rank)
    socket_path = get_socket_path(rank)

    if not os.path.exists(ready_path) and not os.path.exists(socket_path):
        return

    pid = _read_ready_pid(ready_path) if os.path.exists(ready_path) else None

    if pid is not None and _is_pid_alive(pid):
        if not force:
            raise RuntimeError(
                f"Weight cache daemon for rank {rank} is already running "
                f"(pid={pid}, ready={ready_path}). Stop the existing daemon "
                f"before launching a new one, or pass force=True (--force) to "
                f"kill it and take over."
            )
        logger.warning(
            "[weight_cache] force takeover: killing existing daemon pid=%s "
            "for rank %s and reclaiming its socket/ready files.",
            pid,
            rank,
        )
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for path in (ready_path, socket_path):
        if os.path.exists(path):
            os.unlink(path)
            logger.info("Removed stale daemon file: %s", path)
