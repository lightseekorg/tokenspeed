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

"""Model-owned symmetric storage and precompiled projection row buckets."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection_kernel import (
    FusedTPProjectionKernel,
)


def _row_bucket(rows: int) -> int:
    """Keep small shared tiles; align larger owners to whole 128-row tiles."""
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    # Eight intervals per power-of-two range above 2048 bound startup work.
    step = max(128, 1 << max(0, (rows - 1).bit_length() - 4))
    return (rows + step - 1) // step * step


class FusedTPProjectionState:
    """Collectively prepare a TP2/4/8 projection before graph capture.

    Args:
        group: Two, four, or eight node-local GB300 ranks in output-shard order.
        device: This rank's CUDA device, with 152 SMs available to the grid.
        max_rows: Maximum physical rows on any owner.
        input_size: Full input channel count, divisible by 128.
        output_size: Padded output width, divisible by group size times 128.

    Tile-aligned row buckets share storage and epochs and compile before use.
    Above 2048 rows, bucket spacing grows with capacity to bound compilation.
    Sequential layers share this storage, but supply their own weights to run.
    Concurrent models/streams require separate states. No layer weight or final
    output pointer is retained. Call close collectively after destroying graphs.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        max_rows: int,
        input_size: int,
        output_size: int,
    ) -> None:
        if (
            group.size() not in (2, 4, 8)
            or max_rows < 1
            or input_size < 128
            or input_size % 128
            or output_size < group.size() * 128
            or output_size % (group.size() * 128)
            or torch.cuda.get_device_capability(device) != (10, 3)
            or torch.cuda.get_device_properties(device).multi_processor_count != 152
        ):
            raise ValueError(
                "Fused TP projection requires aligned widths and TP2/4/8 GB300 GPUs"
            )
        self.group = group
        self.device = device
        self.rank = group.rank()
        self.world_size = group.size()
        self.input_size = input_size
        self.output_size = output_size
        self.shard_size = output_size // self.world_size
        self.closed = False
        self.max_rows = max_rows
        capacity = _row_bucket(max_rows)
        self.a = symm_mem.empty(
            (capacity * self.world_size, input_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.sa = symm_mem.empty(
            (capacity * self.world_size, input_size // 128),
            dtype=torch.float32,
            device=device,
        )
        self.receive = symm_mem.empty(
            (capacity, output_size), dtype=torch.bfloat16, device=device
        )
        self.ready = symm_mem.empty((168,), dtype=torch.int64, device=device)
        self.ack = symm_mem.empty((168,), dtype=torch.int64, device=device)
        self.input_flags = symm_mem.empty((168,), dtype=torch.int64, device=device)
        for flags in (self.ready, self.ack, self.input_flags):
            flags.zero_()
        tensors = (
            self.a,
            self.sa,
            self.receive,
            self.ready,
            self.ack,
            self.input_flags,
        )
        self.handles = [symm_mem.rendezvous(tensor, group) for tensor in tensors]
        self.peer_tensors = tuple(
            tuple(
                handle.get_buffer(
                    (self.rank + peer) % self.world_size, tensor.shape, tensor.dtype
                )
                for peer in range(self.world_size)
            )
            for handle, tensor in zip(self.handles, tensors)
        )
        self.input_peers, self.scale_peers = (
            tuple(from_dlpack(t.unsqueeze(-1), assumed_align=16) for t in peers)
            for peers in self.peer_tensors[:2]
        )
        # Every bucket addresses the same allocations with its own logical
        # row stride. Stream order and the final peer acknowledgements protect
        # those views when consecutive calls use different row buckets.
        buckets = [32]
        while buckets[-1] < capacity:
            buckets.append(_row_bucket(buckets[-1] + 1))
        self.row_views = {
            rows: (
                from_dlpack(
                    self.a[: rows * self.world_size].unsqueeze(-1), assumed_align=16
                ),
                from_dlpack(
                    self.sa[: rows * self.world_size].unsqueeze(-1), assumed_align=16
                ),
                (
                    from_dlpack(
                        self.receive[:rows].view(
                            rows * self.world_size, self.shard_size, 1
                        ),
                        assumed_align=16,
                    ),
                    from_dlpack(self.receive[:rows].unsqueeze(-1), assumed_align=16),
                    *(
                        from_dlpack(
                            t[
                                :rows,
                                self.rank
                                * self.shard_size : (self.rank + 1)
                                * self.shard_size,
                            ].unsqueeze(-1),
                            assumed_align=16,
                        )
                        for t in self.peer_tensors[2]
                    ),
                ),
            )
            for rows in buckets
        }
        self.peer_ready, self.peer_ack, self.peer_input_flags = (
            tuple(from_dlpack(t, assumed_align=16) for t in peers)
            for peers in self.peer_tensors[3:]
        )
        self.ready_cute = from_dlpack(self.ready, assumed_align=16)
        self.ack_cute = from_dlpack(self.ack, assumed_align=16)
        self.input_flags_cute = from_dlpack(self.input_flags, assumed_align=16)
        weight = torch.zeros(
            (self.shard_size, input_size), dtype=torch.float8_e4m3fn, device=device
        )
        scales = torch.ones(
            (self.shard_size // 128, input_size // 128),
            dtype=torch.float32,
            device=device,
        )
        # A peer must finish zeroing its flags before another rank can publish.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        self.compiled = {}
        for rows in self.row_views:
            inputs = torch.zeros(
                (rows, input_size), dtype=torch.bfloat16, device=device
            )
            out = torch.empty((rows, output_size), dtype=torch.bfloat16, device=device)
            kernel = FusedTPProjectionKernel(cutlass.Float32, False, (128, 128), (1, 1))
            args = self._arguments(inputs, weight, scales, rows, out)
            self.compiled[rows] = cute.compile(
                kernel, *args[:5], 152, *args[5:], lambda v: v, options="--opt-level=2"
            )
            # Warm every supported bucket before capture; they share device epochs.
            self.compiled[rows](*args)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _arguments(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        bucket: int,
        out: torch.Tensor,
    ) -> tuple:
        a, sa, peers = self.row_views[bucket]
        return (
            a,
            from_dlpack(weight.unsqueeze(-1), assumed_align=16),
            peers[0],
            sa,
            from_dlpack(scales.unsqueeze(-1), assumed_align=16),
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
            from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=out.dim_order()
            ),
            peers,
            self.ready_cute,
            self.peer_ready,
            self.ack_cute,
            self.peer_ack,
            cutlass.Int32(self.rank),
            from_dlpack(inputs, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=inputs.dim_order()
            ),
            self.input_peers,
            self.scale_peers,
            self.input_flags_cute,
            self.peer_input_flags,
        )

    def run(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Launch with this layer's tensors and return its owned destination."""
        if self.closed:
            raise RuntimeError("Fused TP projection workspace is closed")
        if (
            len(owner_rows) != self.world_size
            or min(owner_rows) < 0
            or not 0 < max(owner_rows) <= self.max_rows
        ):
            raise ValueError("Fused TP projection counts exceed prepared capacity")
        rows = owner_rows[self.rank]
        bucket = _row_bucket(max(owner_rows))
        for tensor, shape, dtype in (
            (inputs, (rows, self.input_size), torch.bfloat16),
            (weight, (self.shard_size, self.input_size), torch.float8_e4m3fn),
            (
                weight_scales,
                (self.shard_size // 128, self.input_size // 128),
                torch.float32,
            ),
            (out, (rows, self.output_size), torch.bfloat16),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError("Incompatible fused TP projection tensor")
        self.compiled[bucket](
            *self._arguments(inputs, weight, weight_scales, bucket, out)
        )
        return out

    def close(self) -> None:
        """Release peer mappings collectively after all graph consumers finish."""
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.compiled = {}
        self.row_views = {}
        self.input_peers = self.scale_peers = ()
        self.peer_ready = self.peer_ack = self.peer_input_flags = ()
        self.ready_cute = self.ack_cute = self.input_flags_cute = None
        self.peer_tensors = ()
        self.handles = []
        self.a = self.sa = self.receive = self.ready = self.ack = self.input_flags = (
            None
        )
        self.closed = True
