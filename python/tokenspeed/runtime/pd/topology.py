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

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _AttentionParallelMapping(Protocol):
    tp_size: int
    tp_rank: int
    cp_size: int
    cp_rank: int
    dp_size: int
    dp_rank: int


class _PDMapping(Protocol):
    rank: int
    world_size: int
    attn: _AttentionParallelMapping


@dataclass(frozen=True)
class PDParallelTopology:
    """Attention parallel coordinates used by the PD cache-transfer path."""

    tp_size: int
    tp_rank: int
    cp_size: int
    cp_rank: int
    dp_size: int
    dp_rank: int
    world_size: int
    global_rank: int
    # Prefill chunk-pipeline coordinates. tp/cp/dp are INTRA-stage; the world
    # is pp stages of tp*cp*dp ranks each.
    pp_size: int = 1
    pp_rank: int = 0
    # Optional explicit per-stage layer counts (front to back). Registered
    # with the bootstrap so the Decode side plans over the same windows.
    pp_layer_partition: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate parallel sizes and rank coordinates."""
        for name in ("tp", "cp", "dp", "pp"):
            size = getattr(self, f"{name}_size")
            rank = getattr(self, f"{name}_rank")
            if size <= 0:
                raise ValueError(f"{name}_size must be greater than 0, got {size}")
            if not 0 <= rank < size:
                raise ValueError(f"{name}_rank must be in [0, {size}), got {rank}")

        expected_world_size = self.tp_size * self.cp_size * self.dp_size * self.pp_size
        if self.world_size != expected_world_size:
            raise ValueError(
                "world_size must equal tp_size * cp_size * dp_size * pp_size: "
                f"expected {expected_world_size}, got {self.world_size}"
            )
        if not 0 <= self.global_rank < self.world_size:
            raise ValueError(
                f"global_rank must be in [0, {self.world_size}), "
                f"got {self.global_rank}"
            )

    @classmethod
    def from_mapping(cls, mapping: _PDMapping) -> PDParallelTopology:
        """Build PD coordinates from a runtime mapping.

        Args:
            mapping: Runtime mapping whose ``attn`` member exposes TP/CP/DP
                sizes and ranks.

        Returns:
            The immutable PD attention topology for the current process.
        """
        attention = mapping.attn
        return cls(
            tp_size=attention.tp_size,
            tp_rank=attention.tp_rank,
            cp_size=attention.cp_size,
            cp_rank=attention.cp_rank,
            dp_size=attention.dp_size,
            dp_rank=attention.dp_rank,
            world_size=mapping.world_size,
            global_rank=mapping.rank,
            pp_size=getattr(mapping, "pp_size", 1),
            pp_rank=(mapping.pp_rank if getattr(mapping, "pp_size", 1) > 1 else 0),
            pp_layer_partition=getattr(mapping, "pp_layer_partition", None),
        )

    def require_cache_pd_supported(self) -> None:
        """Reject attention topologies unsupported by paged-cache PD."""
        if self.cp_size != 1:
            raise ValueError(
                "CachePD does not support context parallelism: "
                f"cp_size={self.cp_size}; cp_size must be 1"
            )
