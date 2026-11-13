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

"""Version agreement during the existing serving-rank address rendezvous."""

import torch.distributed as dist


def gather_collective_addresses(
    group: dist.ProcessGroup,
    address: str,
    *,
    replica_memory_transitions: bool,
) -> list[str]:
    """Return the mesh addresses once every serving rank agrees on the protocol."""
    protocol = 3 if replica_memory_transitions else 2
    # Keep v2's address string. A legacy runtime also joins this Gloo gather,
    # and its host rejects a v3 tuple when extracting the address string list.
    local = (protocol, address) if replica_memory_transitions else address
    gathered: list[object] = [None] * group.size()
    dist.all_gather_object(gathered, local, group=group)

    addresses = []
    versions = []
    for rank, entry in enumerate(gathered):
        if isinstance(entry, str):
            versions.append(2)
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
