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

"""Utilities for context-parallel layer helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from tokenspeed.runtime.distributed.comm_ops import token_all_gather
from tokenspeed.runtime.utils import get_bool_env_var


def get_layer_id(weight_name: str) -> int | None:
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


# Attention CP utils
@dataclass
class ContextParallelMetadata:
    split_list: list[int] | None = None
    inverse_split_list: list[int] | None = None
    max_token_len_in_block: int = -1
    zigzag_index: list[int] | None = None
    per_rank_actual_token: list[int] | None = None
    prefix_sum_tokens_prev: int = -1
    prefix_sum_tokens_cur: int = -1
    tokens_prev: int = -1
    tokens_cur: int = -1
    total_token_len: int = -1


class CPMetadataContainer:
    """Container for storing global CP metadata."""

    def __init__(self):
        self.value: ContextParallelMetadata | None = None

    def set(self, metadata: ContextParallelMetadata | None) -> None:
        self.value = metadata

    def get(self) -> ContextParallelMetadata | None:
        return self.value

    def __bool__(self) -> bool:
        """Support ``if CP_METADATA`` syntax."""
        return self.value is not None


CP_METADATA = CPMetadataContainer()
ENABLE_CP = get_bool_env_var("ENABLE_CP", "false")


def cp_split_and_rebuild_data(x: torch.Tensor, split_list, zigzag_index):
    split_tensors = list(torch.split(x, split_list, dim=0))
    return torch.cat([split_tensors[i] for i in zigzag_index], dim=0)


def cp_all_gather_rerange_output(
    x, cp_metadata: ContextParallelMetadata, rank: int, group: tuple
):
    """
    |   +-----------before allgather------------+|
    |   | cp_rank0: block0, block7 |
    |   | cp_rank1: block1, block6 |
    |   | cp_rank2: block2, block5 |
    |   | cp_rank3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    x = token_all_gather(
        x,
        group,
        scattered_num_tokens=cp_metadata.per_rank_actual_token,
    )
    cp_segment_num = len(cp_metadata.split_list)
    inverse_index = list(range(0, cp_segment_num, 2)) + list(
        range(cp_segment_num - 1, 0, -2)
    )
    x_list = torch.split(x, cp_metadata.inverse_split_list)
    output = torch.cat([x_list[i] for i in inverse_index])
    return output
