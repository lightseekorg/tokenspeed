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

"""Collectives for a process that emulates one rank of a larger layout."""

import torch

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend
from tokenspeed.runtime.distributed.mapping import Group


def _tile_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """``rows`` rows repeating ``tensor``'s, or zeros if it has none."""
    if tensor.shape[0] == 0:
        return tensor.new_zeros((rows, *tensor.shape[1:]))
    repeats = (rows + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat(repeats, *([1] * (tensor.dim() - 1)))[:rows]


class EmulatedRankBackend(CommBackend):
    """Local stand-ins for the collectives of a rank that has no peers.

    ``--emulate-rank-zero`` runs one rank of a multi-rank layout in a single
    process. Every collective here completes on that process: results have the
    shapes and dtypes the real collective returns on this rank, but their values
    come from this rank's operand alone. The surrounding kernels therefore see
    the deployment's problem sizes while the outputs are not meaningful, and no
    time is spent communicating.

    The fused-collective probes keep the base answers (nothing to prepare, no
    producer-direct outputs), so callers take their unfused paths.
    """

    def __init__(self, rank: int) -> None:
        self._rank = rank

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(tensor, torch.Tensor):
            return tensor
        return super().all_reduce(tensor, group, op=op)

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        return torch.cat([tensor] * len(group), dim=dim)

    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        output.view(len(group), -1).copy_(input.reshape(1, -1))

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        rows = tensor.shape[0] // len(group)
        start = group.index(self._rank) * rows
        return tensor[start : start + rows].clone()

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: Group,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
    ) -> None:
        if output_split_sizes is None and input_split_sizes is None:
            output.copy_(input)
            return
        # Peers may send this rank more rows than it sends.
        output.copy_(_tile_rows(input, output.shape[0]))

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        # Peers may own more rows than this rank, so tile the local rows.
        return _tile_rows(tensor, sum(scattered_num_tokens))

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        index = group.index(self._rank)
        start = sum(scattered_num_tokens[:index])
        return tensor[start : start + scattered_num_tokens[index]].clone()
