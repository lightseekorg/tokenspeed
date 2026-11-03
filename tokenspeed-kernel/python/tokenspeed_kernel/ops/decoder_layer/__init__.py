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

"""Fused decoder layers: one persistent launch per decoder layer.

A fused decoder layer runs one decoder layer of a model recipe (attention, its
output projection and all-reduce, the router, the experts and their exchange,
the post-expert projection, the FFN boundary) and the next layer's query
projection, attention prologue and KV write in one plain launch that keeps one
workgroup on every CU. The kernel boundary between two launches orders the KV
write before the next layer's attention.

The registry entry is ``("decoder_layer", "prepare")``. A leaf compiles every
plan variant it serves at prepare time and returns a :class:`FusedDecoderLayer`;
per-step calls go to that object. Vendor bodies live in private,
solution-named packages (``_gluon/`` for gfx950).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

LAYER_KINDS = ("sliding", "full")


@dataclass(frozen=True)
class PlanKey:
    """The compile-time shape of one fused-layer launch.

    Attributes:
        requests: Requests in the step (rows = requests x verify width).
        layer_kind: This layer's attention kind, one of :data:`LAYER_KINDS`.
        next_kind: The next layer's attention kind, or ``"regular"`` when the
            next layer runs on the regular path (the launch then ends at the
            FFN boundary and writes the regular boundary rows).
    """

    requests: int
    layer_kind: str
    next_kind: str

    def __post_init__(self) -> None:
        if self.layer_kind not in LAYER_KINDS:
            raise ValueError(
                f"layer_kind must be one of {LAYER_KINDS}, got {self.layer_kind!r}"
            )
        if self.next_kind not in LAYER_KINDS + ("regular",):
            raise ValueError(
                f"next_kind must be one of {LAYER_KINDS + ('regular',)}, got {self.next_kind!r}"
            )


class FusedDecoderLayer(Protocol):
    """A prepared fused decoder layer (one per decoder layer of a model)."""

    def decline_reasons(self) -> list[str]:
        """Rank-local reasons this layer cannot serve (compile, residency,
        memory checks); empty when it can. The caller agrees on them across the
        group before capture."""

    def forward(self, prepared_query, residual, ctx):
        """Run this layer and the next layer's prologue.

        Returns:
            ``(next_prepared_query, residual)``.
        """

    def forward_last(self, prepared_query, residual, ctx):
        """Run the last fused layer, whose next layer is regular.

        Returns:
            ``(normed_rows, residual)`` in the regular boundary layout.
        """
