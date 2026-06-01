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

"""NCCL communication backend.

Looks up pre-created process groups from pg_manager. Optionally uses
PyNccl communicators for better performance. Supports torch.compile
via custom ops.
"""

import torch
import torch.distributed

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend, Group


class NcclBackend(CommBackend):
    """Backend using NCCL via PyNccl or torch.distributed.

    Caches per-group resources (process group handle, PyNccl comm)
    keyed by group tuple. Process groups are looked up from pg_manager
    on first use.
    """

    def __init__(self):
        self._resources = {}  # group_tuple → {pynccl_comm, device_group, world_size}
        self._use_pynccl = False

    def configure(self, use_pynccl: bool = False) -> None:
        self._use_pynccl = use_pynccl

    def _get_or_create_resources(self, group: Group):
        if group in self._resources:
            return self._resources[group]

        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        device_group = pg_manager.get_process_group("nccl", group)
        world_size = len(group)

        pynccl_comm = None
        if self._use_pynccl and world_size > 1:
            try:
                from tokenspeed.runtime.distributed.device_communicators.pynccl import (
                    PyNcclCommunicator,
                )

                gloo_group = pg_manager.get_process_group("gloo", group)
                pynccl_comm = PyNcclCommunicator(
                    group=gloo_group,
                    device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                )
            except Exception:
                pynccl_comm = None

        self._resources[group] = {
            "pynccl_comm": pynccl_comm,
            "device_group": device_group,
            "world_size": world_size,
        }
        return self._resources[group]

    # ---- Public CommBackend interface ----

    def all_reduce(self, tensor: torch.Tensor, group: Group, op=None) -> torch.Tensor:
        res = self._get_or_create_resources(group)
        if res["world_size"] == 1:
            return tensor
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        pynccl = res["pynccl_comm"]
        if pynccl is not None and not pynccl.disabled:
            pynccl.all_reduce(tensor, op=op)
        else:
            torch.distributed.all_reduce(tensor, op=op, group=res["device_group"])
        return tensor

    def all_gather(
        self, tensor: torch.Tensor, group: Group, dim: int = 0
    ) -> torch.Tensor:
        res = self._get_or_create_resources(group)
        ws = res["world_size"]
        if ws == 1:
            return tensor
        if dim < 0:
            dim += tensor.dim()

        input_size = tensor.size()
        output_size = (input_size[0] * ws,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=tensor.dtype, device=tensor.device
        )

        self.all_gather_into_tensor(output_tensor, tensor, group)

        output_tensor = output_tensor.reshape((ws,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (ws * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor, group: Group
    ) -> None:
        res = self._get_or_create_resources(group)
        pynccl = res["pynccl_comm"]
        if pynccl is not None and not pynccl.disabled:
            pynccl.all_gather(output, input)
        else:
            torch.distributed.all_gather_into_tensor(
                output, input, group=res["device_group"]
            )

    def reduce_scatter(self, tensor: torch.Tensor, group: Group) -> torch.Tensor:
        res = self._get_or_create_resources(group)
        ws = res["world_size"]
        if ws == 1:
            return tensor
        input_size = tuple(tensor.size())
        output_tensor = torch.empty(
            (input_size[0] // ws,) + input_size[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        pynccl = res["pynccl_comm"]
        if pynccl is not None and not pynccl.disabled:
            pynccl.reduce_scatter(output_tensor, tensor)
        else:
            torch.distributed.reduce_scatter_tensor(
                output_tensor, tensor, group=res["device_group"]
            )
        return output_tensor

    def send(self, tensor: torch.Tensor, dst: int, group: Group) -> None:
        res = self._get_or_create_resources(group)
        pynccl = res["pynccl_comm"]
        if pynccl is not None and not pynccl.disabled:
            pynccl.send(tensor, dst)
        else:
            torch.distributed.send(tensor, group[dst], group=res["device_group"])

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        group: Group,
    ) -> torch.Tensor:
        res = self._get_or_create_resources(group)
        tensor = torch.empty(size, dtype=dtype, device=device)
        pynccl = res["pynccl_comm"]
        if pynccl is not None and not pynccl.disabled:
            pynccl.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, group[src], group=res["device_group"])
        return tensor

    def token_all_gather(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        """NCCL token_all_gather with padding for uneven token distribution.

        Pads each rank's slice to max_tokens rows, all-gathers, then strips padding.
        """
        tp_size = len(scattered_num_tokens)
        max_tokens = max(scattered_num_tokens)
        hidden = tensor.size(-1)

        local_tokens = tensor.size(0)
        if local_tokens < max_tokens:
            pad = torch.zeros(
                max_tokens - local_tokens,
                hidden,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded = torch.cat([tensor, pad], dim=0)
        else:
            padded = tensor

        output = torch.empty(
            tp_size * max_tokens, hidden, dtype=tensor.dtype, device=tensor.device
        )
        self.all_gather_into_tensor(output, padded.contiguous(), group)

        chunks = []
        for i, n in enumerate(scattered_num_tokens):
            chunks.append(output[i * max_tokens : i * max_tokens + n])
        return torch.cat(chunks, dim=0)

    def token_reduce_scatter(
        self,
        tensor: torch.Tensor,
        group: Group,
        scattered_num_tokens: list[int],
    ) -> torch.Tensor:
        """NCCL token_reduce_scatter with padding for uneven token distribution.

        Pads the gathered tensor to a uniform layout, reduce-scatters, then strips padding.
        """
        tp_size = len(scattered_num_tokens)
        max_tokens = max(scattered_num_tokens)
        hidden = tensor.size(-1)

        padded_input = torch.zeros(
            tp_size * max_tokens, hidden, dtype=tensor.dtype, device=tensor.device
        )
        offset = 0
        for i, n in enumerate(scattered_num_tokens):
            padded_input[i * max_tokens : i * max_tokens + n].copy_(
                tensor[offset : offset + n]
            )
            offset += n

        output = torch.empty(
            max_tokens, hidden, dtype=tensor.dtype, device=tensor.device
        )
        res = self._get_or_create_resources(group)
        torch.distributed.reduce_scatter_tensor(
            output, padded_input.contiguous(), group=res["device_group"]
        )
        rank = group.index(torch.distributed.get_rank())
        return output[: scattered_num_tokens[rank]].contiguous()
