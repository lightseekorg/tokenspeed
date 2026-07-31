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

"""Cluster topology discovery from the launcher environment."""

from __future__ import annotations

import dataclasses
import os
import re
import socket

SLURM_NNODES = "SLURM_NNODES"
SLURM_NODEID = "SLURM_NODEID"
SLURM_STEP_NODELIST = "SLURM_STEP_NODELIST"
SLURM_JOB_NODELIST = "SLURM_JOB_NODELIST"

# Every node must compute the same rendezvous port without talking to any other
# node, so it is a constant rather than a function of any per-node value.
# TODO: a constant can still collide on the head node. The clean fix is for rank
# 0 to bind port 0 and publish the result to the followers, retrying on the
# close()->bind() race.
DIST_INIT_DEFAULT_PORT = 23456

_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_EPHEMERAL_RANGE_PATH = "/proc/sys/net/ipv4/ip_local_port_range"


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
    import psutil

    return {
        addr.address
        for addrs in psutil.net_if_addrs().values()
        for addr in addrs
        if addr.family == socket.AF_INET
    }


def _interface_ipv4(iface: str) -> str | None:
    import psutil

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

    if env.get(SLURM_NNODES) is None:
        return None
    nnodes = _require_int(env, SLURM_NNODES)
    if nnodes <= 1:
        return None

    node_rank = _require_int(env, SLURM_NODEID)
    if not 0 <= node_rank < nnodes:
        raise ValueError(
            f"{SLURM_NODEID}={node_rank} is out of range for {SLURM_NNODES}={nnodes}"
        )

    nodelist = env.get(SLURM_STEP_NODELIST) or env.get(SLURM_JOB_NODELIST)
    if not nodelist:
        raise ValueError(
            f"{SLURM_STEP_NODELIST} is unset but a multi-node step was detected; "
            "pass --dist-init-addr <host>:<port> explicitly"
        )

    check_dist_init_port(DIST_INIT_DEFAULT_PORT)
    return LauncherTopology(
        nnodes=nnodes,
        node_rank=node_rank,
        head_host=_resolve_head_ipv4(first_host(nodelist), node_rank, env),
        dist_init_port=DIST_INIT_DEFAULT_PORT,
        source="Slurm",
    )
