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

"""Launcher environment defaults and cluster topology discovery."""

from __future__ import annotations

import dataclasses
import os
import re
import socket
import subprocess
import sys

import psutil

SLURM_STEP_NUM_NODES = "SLURM_STEP_NUM_NODES"
SLURM_NODEID = "SLURM_NODEID"
SLURM_STEP_NODELIST = "SLURM_STEP_NODELIST"

# Every node must compute the same rendezvous port without talking to any other
# node, so it is a constant rather than a function of any per-node value.
# TODO: a constant can still collide on the head node. The clean fix is for rank
# 0 to bind port 0 and publish the result to the followers, retrying on the
# close()->bind() race.
DIST_INIT_DEFAULT_PORT = 23456

_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_EPHEMERAL_RANGE_PATH = "/proc/sys/net/ipv4/ip_local_port_range"


def _supports_allocator_setting(setting: str) -> bool:
    """Check the native parser without changing the launcher's allocator state."""
    # Even reading a snapshot initializes allocator settings, so keep capability
    # detection in a child without visible GPUs before worker processes fork.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import torch; torch.cuda.memory._set_allocator_settings({setting!r})",
        ],
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        return True
    name = setting.split(":", 1)[0]
    if (
        f"Unrecognized CachingAllocator option: {name}" in result.stderr
        or f"Unrecognized key '{name}'" in result.stderr
    ):
        return False
    raise RuntimeError(
        f"Cannot probe pinned allocator setting: {result.stderr.strip()}"
    )


def set_pinned_allocator_default() -> None:
    """Default the pinned host allocator for fresh worker processes.

    Large pinned buffers take exact sizes. A helper thread checks freed
    blocks' events: when the forward thread checks them itself, for its
    per-step staging copies, a CUDA host function in flight (the captured
    grammar fill) can hold its ``cudaEventQuery`` for tens of milliseconds.
    Each setting applies only where the native allocator parses it.
    """
    # PyTorch reads the first defined variable in this order.
    key = next(
        (
            name
            for name in (
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_HIP_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
            )
            if name in os.environ
        ),
        "PYTORCH_ALLOC_CONF",
    )
    config = os.environ.get(key, "").strip().rstrip(",")
    added = False
    for default in (
        "pinned_max_round_threshold_mb:1024",
        "pinned_use_background_threads:True",
    ):
        if default.split(":", 1)[0] not in config and _supports_allocator_setting(
            default
        ):
            config = f"{config},{default}" if config else default
            added = True
    if added:
        os.environ[key] = config


def set_rocm_tool_registration_default() -> None:
    """Default ROCm worker processes to no profiler-tool registration with ROCr.

    Once a profiler tool registers with ROCr -- PyTorch's kineto does on every
    import -- ROCm 7.2 wraps each queue in an ``InterceptQueue`` whose staging
    buffer holds 256 packets. A larger backlog that wraps the ring is copied
    past that buffer and corrupts the host heap (fixed upstream after 7.2 by
    ROCm/rocm-systems#4066). Serving builds such a backlog whenever the host
    enqueues far ahead of a stalled stream, e.g. a forward that waits on a
    large Host KV cache load, and then crashes inside the HIP runtime.

    Unregistered, GPU profilers record no kernels. An explicit
    ``HSA_TOOLS_DISABLE_REGISTER`` (``0`` keeps GPU profiling, and the bug) and
    a tool attached at launch through ``ROCP_TOOL_LIBRARIES`` (``rocprofv3``)
    are kept.
    """
    import torch

    if not torch.version.hip:
        return
    if (
        "HSA_TOOLS_DISABLE_REGISTER" in os.environ
        or "ROCP_TOOL_LIBRARIES" in os.environ
    ):
        return
    os.environ["HSA_TOOLS_DISABLE_REGISTER"] = "1"


def set_rocm_hw_queue_default() -> None:
    """Default ROCm worker processes to two hardware queues per stream priority.

    HIP binds each stream at its first launch to one of ``GPU_MAX_HW_QUEUES``
    queues (default 4) per priority. On MI350X queues k and k+4 share a
    compute pipe, and a launch on one waits about 30 us while the other runs
    a kernel. Kimi-K3's prefill runs Iris side kernels on high-priority
    streams beside the forward's GEMMs, so four queues per priority can pair
    them on one pipe and slow each prefill chunk by about a third; two give
    one queue per pipe. An explicit ``GPU_MAX_HW_QUEUES`` is kept.
    """
    import torch

    if torch.version.hip:
        os.environ.setdefault("GPU_MAX_HW_QUEUES", "2")


def _split_top_level(hostlist: str) -> list[str]:
    """Split a hostlist on commas that are outside ``[...]`` groups."""
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for char in hostlist:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(char)
    tokens.append("".join(current))
    return [token.strip() for token in tokens if token.strip()]


def first_host(hostlist: str) -> str:
    """Return the first hostname of a Slurm-compressed hostlist.

    Args:
        hostlist: A hostlist such as ``cn10``, ``cn[10,15]``, ``cn[01-03,07]``
            or ``cn[01-02],dgx[1-2]``.

    Returns:
        The first hostname, with any range notation replaced by its lowest
        member and zero padding preserved.

    Raises:
        ValueError: If ``hostlist`` is empty.
    """
    tokens = _split_top_level(hostlist)
    if not tokens:
        raise ValueError(f"cannot parse an empty hostlist: {hostlist!r}")

    def take_first(match: re.Match[str]) -> str:
        spec = match.group(1)
        return spec.split(",")[0].split("-")[0]

    return _BRACKET_RE.sub(take_first, tokens[0])


def local_ipv4_addresses() -> set[str]:
    """Return every IPv4 address bound to a local interface."""
    return {
        addr.address
        for addrs in psutil.net_if_addrs().values()
        for addr in addrs
        if addr.family == socket.AF_INET
    }


def _interface_ipv4(iface: str) -> str | None:
    for addr in psutil.net_if_addrs().get(iface, ()):
        if addr.family == socket.AF_INET:
            return addr.address
    return None


def _local_routable_ipv4(env: dict[str, str]) -> str | None:
    """Return this node's own routable IPv4 address, or ``None``."""
    iface = env.get("NCCL_SOCKET_IFNAME")
    if iface:
        address = _interface_ipv4(iface)
        if address and not address.startswith("127."):
            return address
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # No packet is sent; this only asks the kernel which source
            # address the default route would use.
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") else address


def interface_for_host(host: str) -> str | None:
    """Return the local interface that carries the route to ``host``.

    Args:
        host: Address or hostname of the peer to be reached.

    Returns:
        The interface name, or ``None`` if no local interface owns the source
        address the kernel would use.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # No packet is sent; this only asks the kernel which source
            # address it would route from.
            sock.connect((host, 9))
            source = sock.getsockname()[0]
    except OSError:
        return None

    for name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == source:
                return name
    return None


def _resolve_head_ipv4(host: str, node_rank: int, env: dict[str, str]) -> str:
    try:
        address = socket.gethostbyname(host)
    except OSError as exc:
        raise ValueError(
            f"cannot resolve the head node {host!r} to an address "
            f"({exc}); pass --dist-init-addr <host>:<port> explicitly"
        ) from exc

    if address.startswith("127."):
        if node_rank != 0:
            raise ValueError(
                f"the head node {host!r} resolves to loopback ({address}) on this "
                "node, so it cannot be reached; pass --dist-init-addr "
                "<host>:<port> explicitly"
            )
        address = _local_routable_ipv4(env)
        if address is None:
            raise ValueError(
                f"the head node {host!r} resolves to loopback and no routable "
                "local address was found; set NCCL_SOCKET_IFNAME or pass "
                "--dist-init-addr <host>:<port> explicitly"
            )

    if node_rank == 0 and address not in local_ipv4_addresses():
        raise ValueError(
            f"derived head address {address} (from {host!r}) is not bound to any "
            "local interface, but this is node rank 0; pass --dist-init-addr "
            "<host>:<port> explicitly"
        )
    return address


def _ephemeral_port_range() -> tuple[int, int] | None:
    try:
        with open(_EPHEMERAL_RANGE_PATH) as handle:
            low, high = handle.read().split()[:2]
        return int(low), int(high)
    except (OSError, ValueError):
        return None


def check_dist_init_port(port: int) -> None:
    """Reject a rendezvous port the kernel may hand out to something else.

    Args:
        port: The port the distributed rendezvous would bind.

    Raises:
        ValueError: If ``port`` lies in the local ephemeral port range.
    """
    span = _ephemeral_port_range()
    if span is None:
        return
    low, high = span
    if low <= port <= high:
        raise ValueError(
            f"rendezvous port {port} lies in this host's ephemeral port range "
            f"({low}-{high}) and may be taken by an unrelated connection; pass "
            "--dist-init-addr <host>:<port> with a port outside that range"
        )


@dataclasses.dataclass(frozen=True)
class LauncherTopology:
    """Multi-node topology as reported by the launcher."""

    nnodes: int
    node_rank: int
    head_host: str
    dist_init_port: int
    source: str

    @property
    def dist_init_addr(self) -> str:
        return f"{self.head_host}:{self.dist_init_port}"


def _require_int(env: dict[str, str], name: str) -> int:
    raw = env.get(name)
    if raw is None:
        raise ValueError(
            f"{name} is unset but a multi-node step was detected; pass "
            "--nnodes/--node-rank/--dist-init-addr explicitly"
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


def detect_topology(env: dict[str, str] | None = None) -> LauncherTopology | None:
    """Derive the multi-node topology from the launcher environment.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        A :class:`LauncherTopology`, or ``None`` when the environment shows no
        evidence of a multi-node launch.

    Raises:
        ValueError: If a multi-node launch is detected but the topology cannot
            be resolved from it.
    """
    env = os.environ if env is None else env

    if env.get(SLURM_STEP_NUM_NODES) is None:
        return None
    nnodes = _require_int(env, SLURM_STEP_NUM_NODES)
    if nnodes <= 1:
        return None

    node_rank = _require_int(env, SLURM_NODEID)
    if not 0 <= node_rank < nnodes:
        raise ValueError(
            f"{SLURM_NODEID}={node_rank} is out of range for {SLURM_STEP_NUM_NODES}={nnodes}"
        )

    nodelist = env.get(SLURM_STEP_NODELIST)
    if not nodelist:
        raise ValueError(
            f"{SLURM_STEP_NODELIST} is unset but a multi-node step was detected; "
            "pass --dist-init-addr <host>:<port> explicitly"
        )

    return LauncherTopology(
        nnodes=nnodes,
        node_rank=node_rank,
        head_host=_resolve_head_ipv4(first_host(nodelist), node_rank, env),
        dist_init_port=DIST_INIT_DEFAULT_PORT,
        source="Slurm",
    )
