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

"""Cumulative host capabilities and serving-rank version agreement.

Protocol 2 agrees on weight-load prerequisites and commits. Protocol 3 keeps
those guarantees and also coordinates memory transitions across the replica.
Their address-gather envelopes differ, so ranks must select the same version.
"""

import torch.distributed as dist

# v2 coordinates weight-load prerequisites, outcomes and commits. It retains
# the mainline memory drain/synchronization behavior.
WEIGHT_UPDATE_PROTOCOL = 2
# v3 retains v2 and adds replica-wide release/resume request agreement,
# preflight/drain checks and device-completion acknowledgements.
WEIGHT_UPDATE_WITH_MEMORY_TRANSITIONS_PROTOCOL = 3


def supports_replica_memory_transitions(
    *, checked_startup: bool, memory_saver: bool, backend_capable: bool, role: str
) -> bool:
    """Select v3 for checked startup and an integrated memory-saver backend."""
    return checked_startup and memory_saver and backend_capable and role != "encode"


def gather_collective_addresses(
    group: dist.ProcessGroup,
    address: str,
    *,
    replica_memory_transitions: bool,
) -> list[str]:
    """Return the mesh addresses once every serving rank agrees on the protocol."""
    protocol = (
        WEIGHT_UPDATE_WITH_MEMORY_TRANSITIONS_PROTOCOL
        if replica_memory_transitions
        else WEIGHT_UPDATE_PROTOCOL
    )
    # Keep v2's address string. A legacy runtime also joins this Gloo gather,
    # and its host rejects a v3 tuple when extracting the address string list.
    local = (protocol, address) if replica_memory_transitions else address
    gathered: list[object] = [None] * group.size()
    dist.all_gather_object(gathered, local, group=group)

    addresses = []
    versions = []
    for rank, entry in enumerate(gathered):
        if isinstance(entry, str):
            versions.append(WEIGHT_UPDATE_PROTOCOL)
            addresses.append(entry)
        elif (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], int)
            and isinstance(entry[1], str)
        ):
            version, peer_address = entry
            versions.append(version)
            addresses.append(peer_address)
        else:
            raise RuntimeError(f"Invalid mesh address from rank {rank}: {entry!r}")
    if any(version != protocol for version in versions):
        raise RuntimeError(
            "Memory protocol mismatch before serving. "
            f"Versions in rank order: {versions}. "
            "Use the same memory protocol on every serving rank."
        )
    return addresses
