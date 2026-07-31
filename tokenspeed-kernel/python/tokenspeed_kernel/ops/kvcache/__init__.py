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

"""KV cache kernel entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import format_signature

__all__ = [
    "DirectH2DScatterPlan",
    "prepare_kv_direct_h2d_scatter_plan",
]

DirectH2DScatterPlan = Callable[[torch.Tensor, torch.Tensor, int, int], Any]


def prepare_kv_direct_h2d_scatter_plan(
    src_layers: list[torch.Tensor],
    dst_layers: list[torch.Tensor],
    entry_ids: list[int],
) -> tuple[DirectH2DScatterPlan | None, str]:
    """Select a backend and prepare metadata for repeated H2D scatters.

    Args:
        src_layers: GPU-visible CPU tensors whose leading dimension indexes
            complete cache-page records.
        dst_layers: Device tensors paired with ``src_layers``.
        entry_ids: Logical entry ID, typically a layer ID, for each tensor pair.

    Returns:
        A backend-owned plan and an empty reason on success. If no compatible
        backend is available, returns ``None`` and a stable fallback reason.
    """
    try:
        kernel = select_kernel(
            "kvcache",
            "h2d_scatter",
            format_signature(),
        )
    except NoKernelFoundError:
        return None, "kernel_unavailable"
    return kernel(
        src_layers=src_layers,
        dst_layers=dst_layers,
        entry_ids=entry_ids,
    )


# Import backend modules after defining the public plan contract.
import tokenspeed_kernel.ops.kvcache.cuda  # noqa: E402,F401
