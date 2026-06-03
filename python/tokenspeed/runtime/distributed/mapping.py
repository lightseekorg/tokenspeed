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

import math
from functools import cached_property

Group = tuple[int, ...]


def _resolve_parallelism_sizes(world_size: int, *sizes: int | None) -> tuple[int, ...]:
    """Resolve the parallelism sizes given world_size.

    `sizes` is ordered innermost (fastest-varying) to outermost.
    """
    assert all(x is None or x > 0 for x in sizes)

    resolved = [x for x in sizes]
    num_to_resolve = sum(x is None for x in sizes)
    if num_to_resolve > 0:
        provided_size = math.prod(x for x in sizes if x is not None)
        assert provided_size <= world_size
        assert world_size % provided_size == 0
        resolved_size = world_size // provided_size

        for index, size in enumerate(resolved):
            if size is None:
                resolved[index] = resolved_size
                resolved_size = 1

    assert math.prod(resolved) == world_size
    return tuple(resolved)


def _make_parallelism_rank(rank: int, size: int, stride: int = 1) -> int:
    """Return the rank of given size and stride."""
    return (rank // stride) % size


def _make_parallelism_group(rank: int, size: int, stride: int = 1) -> Group:
    """Return the group of ranks of given size and stride."""
    base = rank - (rank // stride % size) * stride
    return tuple(base + j * stride for j in range(size))


class MappingBase:

    def __init__(self, rank: int | None = None, world_size: int = 1):
        assert rank is None or rank >= 0
        self._rank = rank
        assert world_size > 0
        self._world_size = world_size

    @property
    def rank(self) -> int:
        assert self._rank is not None, "rank is not initialized"
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        assert self._rank is None, "rank is already initialized"
        assert rank >= 0
        self._rank = rank
        self._on_rank_initialized(rank)

    def _on_rank_initialized(self, rank: int):
        return None

    @property
    def world_size(self) -> int:
        return self._world_size

    @cached_property
    def world_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.world_size, stride=1)


class DenseLayerMapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


class AttentionLayerMapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        cp_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.cp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, cp_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_cp(self) -> bool:
        return self.cp_size > 1

    @cached_property
    def cp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.cp_size, stride=self.tp_size)

    @cached_property
    def cp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.cp_size, stride=self.tp_size)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.dp_size, stride=self.tp_size * self.cp_size
        )

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(
            self.rank, self.dp_size, stride=self.tp_size * self.cp_size
        )

    def scatter_index(self, rank: int) -> int:
        """Index of ``rank`` in a dp-major/tp-minor scattered token count
        table; cp peers share their dp group's tp split."""
        tp_rank = _make_parallelism_rank(rank, self.tp_size, stride=1)
        dp_rank = _make_parallelism_rank(
            rank, self.dp_size, stride=self.tp_size * self.cp_size
        )
        return dp_rank * self.tp_size + tp_rank


class MoeLayerMapping(MappingBase):
    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.ep_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, ep_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_ep(self) -> bool:
        return self.ep_size > 1

    @cached_property
    def ep_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.ep_size, stride=self.tp_size)

    @cached_property
    def ep_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.ep_size, stride=self.tp_size)

    @cached_property
    def has_tp_ep(self) -> bool:
        return self.tp_ep_size > 1

    @cached_property
    def tp_ep_size(self) -> int:
        return self.tp_size * self.ep_size

    @cached_property
    def tp_ep_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_ep_size, stride=1)

    @cached_property
    def tp_ep_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_ep_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.dp_size, stride=self.tp_size * self.ep_size
        )

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(
            self.rank, self.dp_size, stride=self.tp_size * self.ep_size
        )


class VisionTowerMapping(MappingBase):
    """Parallel mapping for vision encoders. Vision layers run colocated and
    share the attention TP group; non-colocated deployments should run the
    encoder out-of-engine (EPD-style workers + gateway dispatch).
    """

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        (self.tp_size,) = _resolve_parallelism_sizes(self.world_size, tp_size)

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)


class Mapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        *,
        attn_tp_size: int | None = None,
        attn_cp_size: int | None = None,
        attn_dp_size: int | None = None,
        dense_tp_size: int | None = None,
        dense_dp_size: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        moe_dp_size: int | None = None,
        nprocs_per_node: int | None = None,
        nnodes: int | None = None,
        base_gpu_id: int = 0,
        gpu_id_step: int = 1,
    ):
        super().__init__(rank, world_size)
        self.attn = AttentionLayerMapping(
            rank=rank,
            world_size=world_size,
            tp_size=attn_tp_size,
            cp_size=attn_cp_size,
            dp_size=attn_dp_size,
        )
        self.dense = DenseLayerMapping(
            rank=rank,
            world_size=world_size,
            tp_size=dense_tp_size,
            dp_size=dense_dp_size,
        )
        self.moe = MoeLayerMapping(
            rank=rank,
            world_size=world_size,
            tp_size=moe_tp_size,
            ep_size=moe_ep_size,
            dp_size=moe_dp_size,
        )
        # Vision tower runs colocated on the attention TP group.
        self.vision = VisionTowerMapping(
            rank=rank,
            world_size=self.attn.tp_size,
            tp_size=self.attn.tp_size,
        )
        self.nprocs_per_node, self.nnodes = _resolve_parallelism_sizes(
            self.world_size, nprocs_per_node, nnodes
        )
        assert base_gpu_id >= 0
        assert gpu_id_step > 0
        self.base_gpu_id = base_gpu_id
        self.gpu_id_step = gpu_id_step

    def _on_rank_initialized(self, rank: int):
        self.attn.rank = rank
        self.dense.rank = rank
        self.moe.rank = rank
        self.vision.rank = rank

    @cached_property
    def has_attn_tp(self) -> bool:
        return self.attn.has_tp

    @cached_property
    def has_attn_cp(self) -> bool:
        return self.attn.has_cp

    @cached_property
    def has_attn_dp(self) -> bool:
        return self.attn.has_dp

    @cached_property
    def node_rank(self) -> int:
        return self.rank // self.nprocs_per_node

    @cached_property
    def local_rank(self) -> int:
        return self.rank % self.nprocs_per_node

    @cached_property
    def gpu_id(self) -> int:
        return self.base_gpu_id + self.local_rank * self.gpu_id_step

    def __repr__(self) -> str:
        rank_str = str(self._rank) if self._rank is not None else "?"
        lines = [
            f"Mapping(rank={rank_str}, world_size={self.world_size})",
            f"  Cluster : {self.nnodes} node(s) x {self.nprocs_per_node} proc(s)",
            f"  Attention: tp={self.attn.tp_size}  cp={self.attn.cp_size}  dp={self.attn.dp_size}",
            f"    Vision: tp={self.vision.tp_size}",
            f"  Dense   : tp={self.dense.tp_size}  dp={self.dense.dp_size}",
            f"  MoE     : tp={self.moe.tp_size}  ep={self.moe.ep_size}  dp={self.moe.dp_size}",
        ]
        return "\n".join(lines)
