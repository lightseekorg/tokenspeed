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

"""SMs held by a long-running kernel that overlaps compute on another stream.

A persistent kernel compiles its grid for every cluster slot of the GPU. When
another kernel holds some SMs for the whole forward, such as the DWDP weight
fetch, the CTAs that do not fit run as a second wave and the kernel takes about
twice as long. Persistent kernels size their grids with ``max_active_clusters``
so they fit beside the reserved SMs.

Reserved SMs are counted in clusters of ``CLUSTER_SIZE`` CTAs: the holding
kernel must launch in clusters of that size, so each one takes a whole 8-SM
slot instead of fragmenting the slots of larger clusters.
"""

from __future__ import annotations

import functools

import torch

CLUSTER_SIZE = 8
_reserved: dict[int, int] = {}


def reserve_sm_clusters(device: torch.device, clusters: int) -> None:
    """Record ``clusters`` CTA clusters of ``CLUSTER_SIZE`` held on ``device``.

    Call before persistent kernels compile; kernels compiled earlier keep
    their grids until they compile again.
    """
    if device.type != "cuda" or device.index is None:
        raise ValueError("SM reservations need an explicit CUDA device index")
    if clusters < 0:
        raise ValueError("Reserved clusters must be non-negative")
    _reserved[device.index] = clusters


def reserved_sm_clusters(device: torch.device) -> int:
    """Clusters of ``CLUSTER_SIZE`` CTAs reserved on ``device``."""
    return _reserved.get(device.index, 0)


def max_active_clusters(device: torch.device, cluster_size: int) -> int:
    """Clusters of ``cluster_size`` CTAs that fit beside the reserved SMs.

    Each reserved cluster takes ``CLUSTER_SIZE / cluster_size`` slots of a
    smaller cluster size and one slot of a larger one.
    """
    import cutlass.utils as utils

    slots = utils.HardwareInfo(device.index).get_max_active_clusters(cluster_size)
    taken = reserved_sm_clusters(device) * max(1, CLUSTER_SIZE // cluster_size)
    if taken >= slots:
        raise ValueError(
            f"Reserved SMs leave no room for clusters of {cluster_size} CTAs"
        )
    return slots - taken


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def free_sm_count(device: torch.device) -> int:
    """SMs left beside the reserved clusters, for one-CTA-per-SM persistent grids."""
    free = _sm_count(device.index) - reserved_sm_clusters(device) * CLUSTER_SIZE
    if free <= 0:
        raise ValueError("Reserved SMs leave no SM for persistent grids")
    return free
