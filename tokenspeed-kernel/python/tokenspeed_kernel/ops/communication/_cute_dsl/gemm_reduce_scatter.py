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

"""Fused GEMM + reduce-scatter through NVLink multicast on SM100."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.thirdparty.cute_dsl.gemm_reduce_scatter._kernel import (
    Sm100PersistentDenseGemmReduceScatterLDMCKernel,
)

# CTA tile; every rank's row block is padded to whole M tiles.
_TILE_M, _TILE_N = 128, 256


def _cute_tensor(x: torch.Tensor) -> cute.Tensor:
    # Matrices go in as (rows, columns, batch) with dynamic rows; flags stay 1-D.
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    return from_dlpack(x, assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if x.dim() == 3 else 0
    )


class GemmReduceScatterState:
    """Symmetric workspace and compiled kernel for one group's
    ``reduce_scatter(x @ weight.T)`` with BF16 operands.

    Each rank stores its BF16 partial tiles in symmetric memory; the owner of
    a row block sums all ranks' tiles in FP32 with one multicast load per
    tile and rounds once. Every rank builds the state together, eagerly and
    outside graph capture; the kernel compiles here with dynamic row counts.

    Args:
        group: Two, four, or eight NVLink-connected ranks.
        device: This rank's SM100 or SM103 CUDA device.
        max_rows: Most rows of ``x`` (all ranks' row blocks) one call takes.
        out_features: ``weight`` rows N, divisible by 256.
        in_features: ``weight`` columns K, divisible by 64.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        max_rows: int,
        out_features: int,
        in_features: int,
    ) -> None:
        if (
            group.size() not in (2, 4, 8)
            or out_features % _TILE_N
            or in_features % 64
            or torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3))
        ):
            raise ValueError(
                "GEMM reduce-scatter requires TP2/4/8 SM100/SM103 GPUs, "
                "N divisible by 256 and K by 64"
            )
        self.world_size = group.size()
        self.out_features = out_features
        self.in_features = in_features
        block = -(-max_rows // self.world_size)
        self.block_capacity = -(-block // _TILE_M) * _TILE_M
        capacity = self.block_capacity * self.world_size
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.storage = symm_mem.empty(
            (capacity, out_features), dtype=torch.bfloat16, device=device
        )
        self.handle = symm_mem.rendezvous(self.storage, group.group_name)
        if not self.handle.multicast_ptr:
            raise RuntimeError("GEMM reduce-scatter requires NVLink multicast")
        self.multicast = cutlass_torch.as_tensor(
            self.handle.multicast_ptr, self.storage.shape, torch.bfloat16
        )
        flags = capacity // _TILE_M * (out_features // _TILE_N) + sms + 8
        self.flags = symm_mem.empty((flags,), dtype=torch.int32, device=device)
        self.flags.zero_()
        flag_handle = symm_mem.rendezvous(self.flags, group.group_name)
        if not flag_handle.multicast_ptr:
            raise RuntimeError("GEMM reduce-scatter requires NVLink multicast")
        self.flag_multicast = cutlass_torch.as_tensor(
            flag_handle.multicast_ptr, self.flags.shape, torch.int32
        )
        self.peer_flags = tuple(
            flag_handle.get_buffer(peer, self.flags.shape, torch.int32)
            for peer in range(self.world_size)
        )
        kernel = Sm100PersistentDenseGemmReduceScatterLDMCKernel(
            cutlass.Float32,
            cutlass.BFloat16,
            False,
            (_TILE_M, _TILE_N),
            (1, 1),
            True,
            group.rank(),
            self.world_size,
        )
        kernel.can_implement(
            (capacity, out_features, in_features, 1),
            cutlass.BFloat16,
            cutlass.BFloat16,
            "k",
            "k",
            "n",
        )
        x = torch.empty((capacity, in_features), dtype=torch.bfloat16, device=device)
        weight = torch.empty(
            (out_features, in_features), dtype=torch.bfloat16, device=device
        )
        out = torch.empty(
            (self.block_capacity, out_features), dtype=torch.bfloat16, device=device
        )
        self._kernel = cute.compile(
            kernel,
            **self._arguments(x, weight, out),
            max_active_clusters=sms,
        )
        self.handle.barrier(channel=0)

    def _arguments(
        self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> dict:
        rows = x.shape[0]
        return dict(
            a=_cute_tensor(x),
            b=_cute_tensor(weight),
            c=_cute_tensor(self.storage[:rows]),
            comm_in_multicast_tensor=_cute_tensor(self.multicast[:rows]),
            comm_out_unicast_tensor=_cute_tensor(out),
            barrier_flag_unicast=_cute_tensor(self.flags),
            barrier_flag_multicast=_cute_tensor(self.flag_multicast),
            barrier_flag_unicast_per_peer=tuple(
                _cute_tensor(flags) for flags in self.peer_flags
            ),
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )

    def run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """This rank's block of ``reduce_scatter(x @ weight.T)``.

        Args:
            x: Contiguous BF16 ``[world_size * B, K]``, rank ``r``'s rows
                ``[r * B, (r + 1) * B)``.
            weight: Contiguous BF16 ``[N, K]``.

        Returns:
            BF16 ``[B, N]``: the sum over ranks of their partials' rows of
            this rank's block.
        """
        block = x.shape[0] // self.world_size
        padded = -(-block // _TILE_M) * _TILE_M
        assert x.shape[0] == block * self.world_size and padded <= self.block_capacity
        if padded != block:
            x = torch.nn.functional.pad(
                x.view(self.world_size, block, -1), (0, 0, 0, padded - block)
            ).view(self.world_size * padded, -1)
        out = x.new_empty((padded, self.out_features))
        # Exit-flag offsets follow the row count: no peer may still reset the last call's.
        self.handle.barrier(channel=0)
        self._kernel(**self._arguments(x, weight, out))
        return out[:block]
