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

"""HCCL communication backend."""

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


class HcclBackend(CommBackend):
    """Collectives implemented by torch.distributed over HCCL."""

    @staticmethod
    def _process_group(group: Group):
        return pg_manager.get_process_group("hccl", group)

    def all_reduce(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        group: Group,
        op=None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(tensor, torch.Tensor):
            return super().all_reduce(tensor, group, op=op)
        if len(group) > 1:
            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM if op is None else op,
                group=self._process_group(group),
            )
        return tensor

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        if len(group) == 1:
            return tensor
        if dim < 0:
            dim += tensor.dim()

        input_size = tensor.size()
        gathered = torch.empty(
            (input_size[0] * len(group),) + input_size[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        self.all_gather_into_tensor(gathered, tensor, group)
        return (
            gathered.reshape((len(group),) + input_size)
            .movedim(0, dim)
            .reshape(
                input_size[:dim]
                + (len(group) * input_size[dim],)
                + input_size[dim + 1 :]
            )
        )

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        if len(group) == 1:
            output.copy_(input)
            return
        dist.all_gather_into_tensor(output, input, group=self._process_group(group))

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        if len(group) == 1:
            return tensor
        output = torch.empty(
            (tensor.shape[0] // len(group),) + tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.reduce_scatter_tensor(output, tensor, group=self._process_group(group))
        return output

    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        if len(group) == 1:
            output.copy_(input)
            return
        dist.all_to_all_single(output, input, group=self._process_group(group))

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        max_tokens = max(scattered_num_tokens)
        if tensor.shape[0] < max_tokens:
            tensor = torch.cat(
                (
                    tensor,
                    tensor.new_zeros(max_tokens - tensor.shape[0], tensor.shape[-1]),
                )
            )

        gathered = tensor.new_empty(len(group) * max_tokens, tensor.shape[-1])
        self.all_gather_into_tensor(gathered, tensor.contiguous(), group)
        return torch.cat(
            [
                gathered[rank * max_tokens : rank * max_tokens + tokens]
                for rank, tokens in enumerate(scattered_num_tokens)
            ]
        )

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        max_tokens = max(scattered_num_tokens)
        padded = tensor.new_zeros(
            len(group) * max_tokens,
            tensor.shape[-1],
        )
        offset = 0
        for rank, tokens in enumerate(scattered_num_tokens):
            padded[rank * max_tokens : rank * max_tokens + tokens].copy_(
                tensor[offset : offset + tokens]
            )
            offset += tokens

        output = tensor.new_empty(max_tokens, tensor.shape[-1])
        if len(group) == 1:
            output.copy_(padded)
        else:
            dist.reduce_scatter_tensor(
                output,
                padded.contiguous(),
                group=self._process_group(group),
            )
        rank = group.index(dist.get_rank())
        return output[: scattered_num_tokens[rank]].contiguous()

    def send(self, tensor: torch.Tensor, dst: int, group: Group) -> None:
        dist.send(tensor, group[dst], group=self._process_group(group))

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        group: Group,
    ) -> torch.Tensor:
        tensor = torch.empty(size, dtype=dtype, device=device)
        dist.recv(tensor, group[src], group=self._process_group(group))
        return tensor
