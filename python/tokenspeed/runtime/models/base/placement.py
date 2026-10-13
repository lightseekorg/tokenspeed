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

"""DTensor-inspired Placement type system for describing tensor distribution.

Instead of using torch.DTensor directly (designed for training, high overhead),
we use lightweight annotations + compile-time static analysis to describe how
tensors are distributed across parallel groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tokenspeed.runtime.distributed.mapping import Mapping


class PlacementType(Enum):
    """Describes the distribution state of a tensor on a parallel dimension."""

    REPLICATE = auto()  # Every rank holds a full copy
    SHARD = auto()  # Scattered along the token dimension across ranks
    PARTIAL = auto()  # Each rank holds a partial sum; needs reduce to be complete


class ParallelGroup(Enum):
    """The parallel group a communication belongs to."""

    ATTN_TP = auto()
    # Head-sharded attention projections over attention-DP ranks.
    ATTN_HEAD_TP = auto()
    DENSE_TP = auto()
    MOE_TP_EP = auto()
    # Vocab-sharded LM head over attention-DP ranks.
    LM_HEAD_TP = auto()


@dataclass(frozen=True, slots=True)
class Placement:
    """Describes the distribution state of a tensor."""

    type: PlacementType
    group: ParallelGroup

    def __repr__(self) -> str:
        return f"Placement({self.type.name}, {self.group.name})"

    @classmethod
    def replicate(cls, group: ParallelGroup) -> Placement:
        return cls(PlacementType.REPLICATE, group)

    @classmethod
    def shard(cls, group: ParallelGroup) -> Placement:
        return cls(PlacementType.SHARD, group)

    @classmethod
    def partial(cls, group: ParallelGroup) -> Placement:
        return cls(PlacementType.PARTIAL, group)


Replicate = Placement.replicate
Shard = Placement.shard
Partial = Placement.partial


def group_tp_size(mapping: Mapping, group: ParallelGroup) -> int:
    if group == ParallelGroup.ATTN_TP:
        return mapping.attn.tp_size
    elif group == ParallelGroup.ATTN_HEAD_TP:
        return mapping.attn.head_tp_size
    elif group == ParallelGroup.DENSE_TP:
        return mapping.dense.tp_size
    elif group == ParallelGroup.MOE_TP_EP:
        return mapping.moe.tp_ep_size
    elif group == ParallelGroup.LM_HEAD_TP:
        return mapping.lm_head.tp_size
    else:
        raise ValueError(f"Unknown group: {group}")


def group_has_parallel(mapping: Mapping, group: ParallelGroup) -> bool:
    if group == ParallelGroup.ATTN_TP:
        return mapping.has_attn_tp
    elif group == ParallelGroup.ATTN_HEAD_TP:
        return mapping.attn.has_head_tp
    elif group == ParallelGroup.DENSE_TP:
        return mapping.dense.has_tp
    elif group == ParallelGroup.MOE_TP_EP:
        return mapping.moe.has_tp_ep
    elif group == ParallelGroup.LM_HEAD_TP:
        return mapping.lm_head.has_tp
    else:
        raise ValueError(f"Unknown group: {group}")


def use_all_reduce(
    mapping: Mapping, src_group: ParallelGroup, dst_group: ParallelGroup
) -> bool:
    return group_tp_size(mapping, src_group) == group_tp_size(mapping, dst_group)


def can_fuse_reduce_norm(
    mapping: Mapping,
    src_group: ParallelGroup,
    dst_group: ParallelGroup,
) -> bool:
    return use_all_reduce(mapping, src_group, dst_group) and mapping.has_attn_tp
