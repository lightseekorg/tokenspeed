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

"""Which weights the Kimi-K3 TP8 decode chain prefetches into L2, and when.

Decode at a few rows is bound by HBM bandwidth, and a kernel whose weights are
not in L2 yet starts stalled on DRAM while the attention core and the
all-reduces leave bandwidth idle. At fixed points of each layer the chain
prefetches the byte ranges the next kernels read on a side stream; at the
``EXPERTS_AHEAD`` row counts the routed experts prefetch them in-kernel instead,
during their drain. ``k3_decode`` launches what these tables describe; this
module has no ``cutlass`` import, so the tables are testable anywhere.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common import policy

MIB = 1 << 20
# Rows whose o-projection exit prefetches the MoE front's packed weight.
OPROJ_DOWN = "3-8:1"
# Rows whose attention prefetches its o-projection and the MoE layer's BF16 weights.
ATTENTION_MOE = {"kda": "5-8:1", "mla": "8:1"}
# Next attention's weights once the routed experts launch: MiB cap or cap.carveout.
NEXT_ATTENTION = "1:96.100,2-3:96,4-7:64,8:32"
# Rows whose routed experts prefetch the next attention's weights in-kernel: MiB cap.
EXPERTS_AHEAD = "4-7:16"


def _turned_on(spec: str, served: tuple[int, ...]) -> tuple[int, ...]:
    table = policy.batch_table(spec)
    return tuple(b for b in served if table.get(b))


def oproj_down_batches(served: tuple[int, ...]) -> tuple[int, ...]:
    """The ``served`` row counts whose o-projection exit prefetches the front."""
    return _turned_on(OPROJ_DOWN, served)


def attention_moe_batches(mla: bool, served: tuple[int, ...]) -> tuple[int, ...]:
    """The ``served`` row counts with an ``ATTENTION_MOE`` prefetch.

    At those row counts the MLA (or KDA) attention prefetches its o-projection
    and the MoE layer's BF16 weights.
    """
    return _turned_on(ATTENTION_MOE["mla" if mla else "kda"], served)


def next_attention_groups(
    served: tuple[int, ...],
) -> dict[tuple[int, int | None], tuple[int, ...]]:
    """The ``served`` row counts with a ``NEXT_ATTENTION`` launch.

    They are grouped by (byte cap, shared-memory carveout or None);
    ``EXPERTS_AHEAD`` row counts take none.
    """
    ahead, groups = policy.batch_table(EXPERTS_AHEAD), {}
    for batch, spec in policy.batch_table(NEXT_ATTENTION).items():
        cap, carveout = spec if isinstance(spec, tuple) else (spec, None)
        if batch in served and batch not in ahead and cap:
            groups.setdefault((cap * MIB, carveout), []).append(batch)
    return {key: tuple(batches) for key, batches in groups.items()}


def experts_ahead_caps(served: tuple[int, ...]) -> dict[int, int]:
    """Byte cap of the next attention's weights per ``EXPERTS_AHEAD`` row count."""
    table = policy.batch_table(EXPERTS_AHEAD)
    return {b: table[b] * MIB for b in served if table.get(b)}


def weight_ranges(weights, byte_cap: int) -> list[tuple[int, int]]:
    """(address, bytes) of each weight's storage span, in order.

    Args:
        weights: Tensors whose storage keeps its address while the ranges are used.
        byte_cap: Total bytes covered, 0 for all; capped spans round down to 16 bytes.

    Returns:
        One range per non-empty weight, up to the cap.

    Raises:
        ValueError: A weight does not start 16-byte aligned (the prefetches move
            16-byte chunks).
    """
    ranges, remaining = [], byte_cap
    for weight in weights:
        if weight.numel() == 0:
            continue
        if weight.data_ptr() % 16:
            raise ValueError("L2 prefetch ranges must start 16-byte aligned")
        span = 1 + sum((n - 1) * s for n, s in zip(weight.shape, weight.stride()))
        size = span * weight.element_size()
        if byte_cap:
            size = min(size, remaining) // 16 * 16
            remaining -= size
        if size:
            ranges.append((weight.data_ptr(), size))
        if byte_cap and remaining < 16:
            break
    return ranges


def ahead_table(first, weights, byte_cap: int, rows: int, device) -> torch.Tensor:
    """The routed experts' in-kernel prefetch table.

    Args:
        first: Weights covered whole, before ``weights``.
        weights: Weights covered up to ``byte_cap`` bytes.
        byte_cap: The cap on ``weights``' bytes (not on ``first``'s).
        rows: The table rows the experts kernel reads; later ranges are dropped.
        device: Where the table lives.

    Returns:
        [rows, 2] int64 (address, bytes); unused rows stay zero, which the
        kernel skips.
    """
    ranges = (weight_ranges(first, 0) + weight_ranges(weights, byte_cap))[:rows]
    table = torch.zeros((rows, 2), dtype=torch.int64)
    if ranges:
        table[: len(ranges)] = torch.tensor(ranges, dtype=torch.int64)
    return table.to(device)
