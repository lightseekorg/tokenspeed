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

"""Pipeline-stage boundary state and layer-window helpers.

A pipeline stage's forward either consumes a :class:`PPStageState` received
from the upstream stage (mid-pipeline) or produces one for the downstream
stage. The tensor geometry is fully derivable from the token count plus model
config on both sides — every PP rank runs the same deterministic scheduler —
so the wire protocol is a fixed-order sequence of raw tensor sends with no
metadata exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from tokenspeed.runtime.distributed.mapping import Mapping


def pp_stage_windows(
    num_layers: int,
    pp_size: int,
    partition: tuple[int, ...] | None = None,
) -> list[tuple[int, int]]:
    """All stages' [start, end) layer windows.

    The single source of the stage-split arithmetic: the model build, the KV
    transfer route, and the Decode-side peer planner must all agree on it.

    Args:
        num_layers: Total model layer count.
        pp_size: Number of pipeline stages.
        partition: Optional explicit per-stage layer counts (front to back),
            e.g. ``(8, 11, 11, 8)``. When omitted, layers split as evenly as
            possible with the remainder on the front stages.

    Returns:
        One ``[start, end)`` window per stage, covering ``0..num_layers``.

    Raises:
        ValueError: the partition length or sum does not match.
    """
    if partition is not None:
        if len(partition) != pp_size:
            raise ValueError(
                f"pp layer partition {partition} has {len(partition)} entries "
                f"for {pp_size} pipeline stages"
            )
        if any(count <= 0 for count in partition):
            raise ValueError(
                f"pp layer partition {partition} must give every stage at "
                "least one layer"
            )
        if sum(partition) != num_layers:
            raise ValueError(
                f"pp layer partition {partition} sums to {sum(partition)} "
                f"but the model has {num_layers} layers"
            )
        counts = partition
    else:
        base = num_layers // pp_size
        remainder = num_layers % pp_size
        counts = tuple(
            base + (1 if stage < remainder else 0) for stage in range(pp_size)
        )
    windows = []
    start = 0
    for length in counts:
        windows.append((start, start + length))
        start += length
    return windows


def pp_layer_window(num_hidden_layers: int, mapping: Mapping) -> tuple[int, int]:
    """Return this stage's [start, end) global layer window.

    Honors ``mapping.pp_layer_partition`` when set (explicit per-stage layer
    counts, e.g. to lighten the embed/lm_head stages). Otherwise layers split
    as evenly as possible with the remainder on the EARLIER stages.
    """
    pp_size = mapping.pp_size
    pp_rank = mapping.pp_rank if pp_size > 1 else 0
    partition = getattr(mapping, "pp_layer_partition", None)
    return pp_stage_windows(num_hidden_layers, pp_size, partition)[pp_rank]


@dataclass
class PPStageState:
    """Inter-stage tensor bundle in a fixed wire order.

    Fields are declared in wire order; ``tensors()`` and ``from_tensors``
    round-trip them so the executor can send/recv without knowing the model.
    ``None`` fields are skipped on the wire — the spec on the receive side
    must produce the same skip pattern (both sides derive it from config).
    """

    hidden_states: torch.Tensor
    hc_x: torch.Tensor | None = None
    hc_post: torch.Tensor | None = None
    hc_comb: torch.Tensor | None = None
    # K3 AttnRes: the valid prefix of the block-residual snapshot buffer,
    # [num_valid_blocks, num_tokens, hidden]. The downstream stage seeds its
    # own (full-size) buffer with these rows; its block-write layers fill the
    # rest.
    block_residual: torch.Tensor | None = None

    def tensors(self) -> list[torch.Tensor]:
        out = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                out.append(value)
        return out

    @classmethod
    def from_tensors(cls, tensors: list[torch.Tensor], field_names: list[str]):
        kwargs = dict(zip(field_names, tensors, strict=True))
        return cls(**kwargs)
