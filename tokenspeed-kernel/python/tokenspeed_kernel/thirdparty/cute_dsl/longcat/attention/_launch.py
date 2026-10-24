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

"""Pointer caching and residency checks for LongCat kernel launches."""

from __future__ import annotations

import cutlass.cute as cute
import cutlass.utils
import torch
from flashinfer.cute_dsl.utils import make_ptr

# Devices already validated for a (cluster size, cluster count, caller) triple;
# the query itself costs a driver round-trip, so it runs once per launch site.
_CHECKED: set[tuple[int, int, int, str]] = set()


def assert_cluster_residency(
    device_index: int, cluster_size: int, clusters: int, what: str
) -> None:
    """Fail loudly when this device cannot co-schedule ``clusters`` clusters.

    Args:
        device_index: CUDA device ordinal to validate (cached per device).
        cluster_size: CTAs per cluster of the launch.
        clusters: number of clusters the launch needs resident at once.
        what: caller description used as the error message's subject.

    Raises:
        RuntimeError: when the device reports fewer co-resident clusters.
    """
    key = (device_index, cluster_size, clusters, what)
    if key in _CHECKED:
        return
    hw_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
        cluster_size=cluster_size
    )
    if hw_clusters < clusters:
        raise RuntimeError(
            f"{what}: needs {clusters} co-resident {cluster_size}-CTA clusters, "
            f"but this device reports only {hw_clusters} at one CTA/SM; the "
            "surplus queues silently behind the resident clusters."
        )
    _CHECKED.add(key)


def _ptr(tensor: torch.Tensor, cutlass_dtype, align: int = 16):
    return make_ptr(
        cutlass_dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _cached_weight(cache: dict, name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Retain contiguous weight rows and their source for a stable TMA view."""
    key = (tensor.data_ptr(), tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))
    slot = ("weight", name)
    entry = cache.get(slot)
    if entry is None or entry[0] != key:
        entry = (key, tensor.contiguous(), tensor)
        cache[slot] = entry
    return entry[1]


def _cached_ptr(
    cache: dict, name: str, tensor: torch.Tensor, cutlass_dtype, align: int = 16
):
    """Return the pointer wrapper of ``tensor``, cached per argument name.

    The cache entry for ``name`` holds ``((data_ptr, dtype), pointer)``; a call
    passing a different tensor (or dtype) for the same argument misses and
    rebuilds, so every step-invariant argument (weights, workspace buffers,
    transport tables) is wrapped only once.
    """
    key = (tensor.data_ptr(), cutlass_dtype)
    entry = cache.get(name)
    if entry is None or entry[0] != key:
        entry = (key, _ptr(tensor, cutlass_dtype, align))
        cache[name] = entry
    return entry[1]


def row_major(ptr, shape):
    strides = []
    acc = 1
    for extent in reversed(shape):
        strides.append(acc)
        acc *= extent
    return cute.make_tensor(
        ptr, cute.make_layout(tuple(shape), stride=tuple(reversed(strides)))
    )
