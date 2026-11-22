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

"""The per-rank dimensions of an Inkling2 layer that its monokernel serves.

A plan carries its :class:`Dims` (``Plan.dims``), and the stages read them
from the plan at compile time (``schedule.dims(plan)``), so one kernel source
serves every geometry and each geometry compiles its own launches. The fields
are the dimensions that differ between the served checkpoints; the ones every
Inkling2 checkpoint shares (128-wide heads, 16 relative channels, a
1024-token window and relative extent, 8 routes, 4 conv taps) stay constants
of the stage modules, and the packing checks them on the weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.fp32_output import (
    projection_split_k,
)

HEAD_DIM = 128
REL = 16  # relative-logit channels a head
CONV_TAPS = 4  # taps of the short convolutions
# Rows of the QKV projection a rank packs for its GEMV tasks (64 rows a task
# of many rows) and of the K and V source projections.
_TILE_ROWS = 64


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


@dataclass(frozen=True)
class Dims:
    """One rank's dimensions of an Inkling2 decoder layer (attention TP =
    expert EP = ``world``).

    Args:
        hidden: Model width; every rank holds full rows of the residual state.
        q_heads: Query heads a rank (local and full layers alike).
        kv_heads: KV heads a rank.
        routed: Routed experts of the model (every rank routes over all).
        inter: Expert intermediate width.
        world: Ranks of the attention TP group (and the expert EP group).
    """

    hidden: int
    q_heads: int
    kv_heads: int
    routed: int
    inter: int
    world: int

    def __post_init__(self) -> None:
        sizes = (self.hidden, self.q_heads, self.kv_heads, self.routed, self.inter)
        if (
            min(sizes) < 1
            or self.world < 1
            or self.q_heads % self.kv_heads
            or self.routed % self.world
        ):
            raise ValueError(f"no Inkling2 monokernel geometry {self}")

    @property
    def group(self) -> int:
        """Query heads a KV head."""
        return self.q_heads // self.kv_heads

    @property
    def local_experts(self) -> int:
        """Routed experts a rank runs."""
        return self.routed // self.world

    @property
    def q_dim(self) -> int:
        """Width of a rank's queries, gate and attention output."""
        return self.q_heads * HEAD_DIM

    @property
    def kv_dim(self) -> int:
        """Width of a rank's keys (and of its values)."""
        return self.kv_heads * HEAD_DIM

    # The QKV projection row of a rank: q | k | v | rel | gate | sink | pad
    # (a full layer's adds its K and V conv taps after the sink).
    @property
    def k_off(self) -> int:
        return self.q_dim

    @property
    def v_off(self) -> int:
        return self.k_off + self.kv_dim

    @property
    def rel_off(self) -> int:
        return self.v_off + self.kv_dim

    @property
    def gate_off(self) -> int:
        return self.rel_off + self.q_heads * REL

    @property
    def sink_off(self) -> int:
        return self.gate_off + self.q_dim

    @property
    def taps_off(self) -> int:
        """A full layer's first K tap row; the V taps follow."""
        return self.sink_off + self.q_heads

    @property
    def qkvr(self) -> int:
        """A local layer's projection rows, padded to 8 as the model pads."""
        return _round_up(self.taps_off, 8)

    @property
    def full_qkvr(self) -> int:
        """A full layer's projection rows: the local ones plus the taps."""
        return _round_up(self.taps_off + 2 * self.kv_heads * CONV_TAPS, 8)

    @property
    def qkvr_padded(self) -> int:
        """The packed projection weight's rows (zero rows past the layer's)."""
        return _round_up(self.full_qkvr, _TILE_ROWS)

    @property
    def kv_proj(self) -> int:
        """A full layer's K (or V) source projection: values and their taps."""
        return self.kv_dim + self.kv_heads * CONV_TAPS

    @property
    def kv_proj_padded(self) -> int:
        return _round_up(self.kv_proj, _TILE_ROWS)

    # Channels of the conv ring a request: K | V | attention output | MLP
    # output (``inkling_conv_stream_layout``).
    @property
    def attn_channel0(self) -> int:
        return 2 * self.kv_dim

    @property
    def mlp_channel0(self) -> int:
        return self.attn_channel0 + self.hidden

    @property
    def ring_width(self) -> int:
        return self.mlp_channel0 + self.hidden

    # The regular FP32-output projections' K splits, which the GEMVs follow.
    @property
    def qkvr_splits(self) -> int:
        return projection_split_k(self.qkvr, self.hidden)

    @property
    def router_splits(self) -> int:
        return projection_split_k(self.routed, self.hidden)

    @property
    def wo_splits(self) -> int:
        return projection_split_k(self.hidden, self.q_dim)


def tp4(hidden: int, heads: int, kv_heads: int, routed: int, inter: int) -> Dims:
    """The dimensions of a rank at attention TP4 / experts EP4 of a model of
    ``heads`` query and ``kv_heads`` KV heads."""
    return Dims(hidden, heads // 4, kv_heads // 4, routed, inter, 4)


__all__ = ["CONV_TAPS", "Dims", "HEAD_DIM", "REL", "tp4"]
