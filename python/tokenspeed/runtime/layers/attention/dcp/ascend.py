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

"""Ascend collectives shared by dense MLA and sparse DSA.

Cache ownership and page compaction live in dcp.placement and dcp.metadata.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist


class AscendDCPContext:
    """Own group order and partial-result exchange."""

    def __init__(
        self,
        *,
        degree: int,
        rank: int,
        ranks: tuple[int, ...],
        auxiliary_namespace: str | None,
    ):
        if degree < 1 or len(ranks) != degree or len(set(ranks)) != degree:
            raise ValueError("invalid DCP group")
        if not 0 <= rank < degree:
            raise ValueError("invalid DCP rank")
        self.degree, self.rank, self.ranks = degree, rank, ranks
        self.primary_process_group = self.auxiliary_process_group = None
        self.stream_fork = None
        if degree > 1 and dist.is_initialized():
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            # Import the manager used by the host, without imposing runtime
            # imports on the standalone address/reference tests.
            self.primary_process_group = pg_manager.get_device_process_group(ranks)
            if auxiliary_namespace is None:
                return
            self.auxiliary_process_group = pg_manager.get_dedicated_device_group(
                ranks, auxiliary_namespace
            )
            from tokenspeed.runtime.utils.device_stream import (
                StreamFork,
                new_device_stream,
            )

            self.stream_fork = StreamFork(new_device_stream())

    @property
    def ready(self):
        return self.degree == 1 or self.primary_process_group is not None

    @property
    def has_auxiliary(self):
        return self.auxiliary_process_group is not None

    def _process_group(self, *, auxiliary: bool):
        if not self.ready:
            raise RuntimeError("DCP process group was not configured")
        return (
            self.auxiliary_process_group
            if auxiliary and self.has_auxiliary
            else self.primary_process_group
        )

    def all_gather(self, tensor, *, auxiliary: bool):
        group = self._process_group(auxiliary=auxiliary)
        if self.degree == 1:
            return tensor
        output = torch.empty(
            (self.degree * tensor.shape[0],) + tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(output, tensor.contiguous(), group=group)
        return output

    def gather_query(self, query):
        """Q[T,h,D] -> Q[T,P*h,D], heads in group-rank order."""
        if self.degree == 1:
            return query
        gathered = self.all_gather(query, auxiliary=True)
        return (
            gathered.view(self.degree, *query.shape)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
            .contiguous()
        )

    def all_to_all(self, output, input_tensor, *, auxiliary: bool):
        group = self._process_group(auxiliary=auxiliary)
        if self.degree == 1:
            output.copy_(input_tensor)
        else:
            dist.all_to_all_single(output, input_tensor.contiguous(), group=group)

    def combine(self, output, lse, *, merge: Callable):
        """Partial O[T,P*h,D], LSE[T,P*h] -> TP-local O[T,h,D]."""
        if output.ndim != 3 or output.shape[:-1] != lse.shape:
            raise ValueError("expected O[T,H,D] and LSE[T,H]")
        tokens, heads, dim = output.shape
        if heads % self.degree:
            raise ValueError("query heads must divide the DCP group")
        local_heads = heads // self.degree
        # Also sanitize the single-rank case and native merge inputs.
        output = torch.where(torch.isneginf(lse)[..., None], 0, output)
        if self.degree == 1:
            return output
        send_o = (
            output.transpose(0, 1)
            .reshape(self.degree, local_heads, tokens, dim)
            .contiguous()
        )
        send_lse = lse.float().T.reshape(self.degree, local_heads, tokens).contiguous()
        recv_o, recv_lse = torch.empty_like(send_o), torch.empty_like(send_lse)
        from tokenspeed.runtime.execution.forward_step import (
            get_is_capture_mode,
            get_is_cuda_graph_phase,
        )
        from tokenspeed.runtime.utils.device_stream import limit_stream_cores

        graph_phase = get_is_cuda_graph_phase()
        fork = self.stream_fork
        if graph_phase and self.has_auxiliary and fork is not None:
            main = fork.device_module.current_stream()
            with (
                limit_stream_cores(main, cube_num=12, vector_num=24, enable=True),
                limit_stream_cores(
                    fork.aux_stream, cube_num=12, vector_num=24, enable=True
                ),
                fork.scope(enable=True, overlap=get_is_capture_mode()) as branch,
            ):
                with branch.branch():
                    self.all_to_all(recv_lse, send_lse, auxiliary=True)
                self.all_to_all(recv_o, send_o, auxiliary=False)
        else:
            self.all_to_all(recv_o, send_o, auxiliary=False)
            self.all_to_all(recv_lse, send_lse, auxiliary=True)
        result = merge(
            recv_o.reshape(self.degree, -1, dim), recv_lse.reshape(self.degree, -1)
        )
        return result.view(local_heads, tokens, dim).transpose(0, 1).contiguous()
