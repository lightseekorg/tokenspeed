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
import cutlass.utils as utils
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.ops.communication._cuda.symmetric_memory import (
    empty_owned_symmetric,
)
from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection_kernel import (
    FusedTPProjectionKernel,
)


def _row_bucket(rows: int, row_parallel: bool, world_size: int) -> int:
    """Choose prepared row buckets without specializing each batch size."""
    if rows <= 32:
        return 32
    if rows <= 64:
        return 64
    if row_parallel and 128 < rows <= 160:
        return 160
    if row_parallel and 160 < rows <= 192:
        return 192
    if row_parallel and world_size == 4 and 192 < rows <= 224:
        return 224
    if row_parallel and world_size == 4 and 512 < rows <= 576:
        return 576
    if row_parallel and world_size == 4 and 896 < rows <= 960:
        return 960
    # Eight intervals per power-of-two range above 2048 bound startup work.
    step = max(128, 1 << max(0, (rows - 1).bit_length() - 4))
    return (rows + step - 1) // step * step


class FusedTPProjectionState:
    """Collectively prepare a TP2/4/8 projection before graph capture.

    Args:
        group: NVLink-connected SM100/SM103 ranks in weight-shard order:
            TP2/4 for rows, TP2/4/8 for columns.
        device: This rank's CUDA device, whose SM count bounds the resident grid.
        max_rows: Maximum physical rows on any owner.
        input_size: Full input channel count, divisible by 128.
        output_size: Padded output width, divisible by group size times 128.
        row_parallel: Shard GEMM input channels and reduce output partials.
        quantized: Use block-FP8 operands and scales; False keeps operands BF16.

    Prepared row buckets share storage and epochs and compile before use.
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
        *,
        row_parallel: bool,
        quantized: bool,
    ) -> None:
        device_arch = torch.cuda.get_device_capability(device)
        column_96 = (
            not quantized
            and not row_parallel
            and group.size() == 4
            and (input_size, output_size) == (7168, 14592)
        )
        if (
            group.size() not in (2, 4, 8)
            or max_rows < 1
            or input_size < 128
            or input_size % 128
            or output_size < group.size() * 128
            or (output_size % (group.size() * 128) and not column_96)
            or device_arch not in ((10, 0), (10, 3))
        ):
            raise ValueError(
                "Fused TP projection requires aligned widths and TP2/4/8 SM100/SM103 GPUs"
            )
        if row_parallel and (
            group.size() not in (2, 4) or input_size % (128 * group.size())
        ):
            raise ValueError("Row projection requires TP2/4 and aligned input shards")
        self.row_parallel = row_parallel
        self.quantized = quantized
        self.gemm_input_size = (
            input_size // group.size() if row_parallel else input_size
        )
        self.group = group
        self.device = device
        self.rank = group.rank()
        self.world_size = group.size()
        self.input_size = input_size
        self.output_size = output_size
        self.shard_size = (
            output_size if row_parallel else output_size // self.world_size
        )
        self.receive_width = self.shard_size * self.world_size
        self.closed = False
        self.max_rows = max_rows
        self.active_ctas = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
        # Reserve a control cache line before CTA flags and align the peer epoch banks.
        flag_words = 16 + self.active_ctas
        epoch_offset = (flag_words + 15) // 16 * 16
        capacity = _row_bucket(max_rows, row_parallel, self.world_size)
        self.a = empty_owned_symmetric(
            (capacity * self.world_size, self.gemm_input_size),
            dtype=torch.float8_e4m3fn if quantized else torch.bfloat16,
            device=device,
        )
        self.sa = (
            empty_owned_symmetric(
                (capacity * self.world_size, self.gemm_input_size // 128),
                dtype=torch.float32,
                device=device,
            )
            if quantized
            else None
        )
        self.receive = empty_owned_symmetric(
            (capacity, self.receive_width), dtype=torch.bfloat16, device=device
        )
        if row_parallel:
            self.receive.view(torch.int16).fill_(-32768)
        self.ready = empty_owned_symmetric(
            (flag_words,), dtype=torch.int64, device=device
        )
        self.ack = empty_owned_symmetric(
            (flag_words,), dtype=torch.int64, device=device
        )
        self.input_flags = empty_owned_symmetric(
            (epoch_offset + self.world_size * 16 if row_parallel else flag_words,),
            dtype=torch.int64,
            device=device,
        )
        for flags in (self.ready, self.ack, self.input_flags):
            flags.zero_()
        tensors = (self.a, self.receive, self.ready, self.ack, self.input_flags)
        if quantized:
            tensors = (self.a, self.sa, *tensors[1:])
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
        if not quantized:
            # Keep the peer-role indices fixed without reserving a scale buffer.
            self.peer_tensors = (self.peer_tensors[0], (), *self.peer_tensors[1:])
        self.input_peers = tuple(
            from_dlpack(t.unsqueeze(-1), assumed_align=16) for t in self.peer_tensors[0]
        )
        self.scale_peers = tuple(
            from_dlpack(t.unsqueeze(-1), assumed_align=16) for t in self.peer_tensors[1]
        )
        # Every bucket addresses the same allocations with its own logical
        # row stride. Stream order and peer acknowledgements protect reuse;
        # the kernel joins peers before changing the physical row stride.
        buckets = [32]
        while buckets[-1] < capacity:
            buckets.append(
                _row_bucket(buckets[-1] + 1, self.row_parallel, self.world_size)
            )
        self.row_views = {
            rows: (
                from_dlpack(
                    self.a[: rows * self.world_size].unsqueeze(-1), assumed_align=16
                ),
                (
                    from_dlpack(
                        self.sa[: rows * self.world_size].unsqueeze(-1),
                        assumed_align=16,
                    )
                    if quantized
                    else None
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
                            (
                                t[:rows].view(self.world_size, rows, self.shard_size)[
                                    self.rank
                                ]
                                if row_parallel
                                else t[
                                    :rows,
                                    self.rank
                                    * self.shard_size : (self.rank + 1)
                                    * self.shard_size,
                                ]
                            ).unsqueeze(-1),
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
        # Each rank has one cache line for input and consumption epochs.
        # Keep remote polling available when multicast is unsupported.
        self.input_multicast_ptr = (
            int(self.handles[-1].multicast_ptr or 0)
            if row_parallel or (not quantized and self.world_size == 4)
            else 0
        )
        self.peer_input_flags_local = self.peer_input_flags
        self.peer_ack_local = self.peer_ack
        if row_parallel:
            self.peer_input_flags = tuple(
                from_dlpack(t[:flag_words], assumed_align=16)
                for t in self.peer_tensors[5]
            )
            epoch_banks = self.input_flags[epoch_offset:].view(self.world_size, 16)
            peer_epoch_banks = tuple(
                epoch_banks[(self.rank + peer) % self.world_size]
                for peer in range(self.world_size)
            )
            self.peer_input_flags_local = tuple(
                from_dlpack(bank, assumed_align=16) for bank in peer_epoch_banks
            )
            # The second word records consumption; multicast makes it local.
            if self.input_multicast_ptr:
                self.peer_ack_local = tuple(
                    from_dlpack(bank[1:2], assumed_align=8) for bank in peer_epoch_banks
                )
        self.ready_cute = from_dlpack(self.ready, assumed_align=16)
        self.ack_cute = from_dlpack(self.ack, assumed_align=16)
        self.input_flags_cute = from_dlpack(self.input_flags, assumed_align=16)
        weight = torch.zeros(
            (self.shard_size, self.gemm_input_size),
            dtype=torch.float8_e4m3fn if quantized else torch.bfloat16,
            device=device,
        )
        scales = (
            torch.ones(
                (self.shard_size // 128, self.gemm_input_size // 128),
                dtype=torch.float32,
                device=device,
            )
            if quantized
            else None
        )
        # A peer must finish zeroing its flags before another rank can publish.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        self.compiled = {}
        cluster_ctas = self.active_ctas
        # Use SM103 cooperative schedules for the tuned TP4 projection shapes.
        cooperative_projection = (
            device_arch == (10, 3)
            and not quantized
            and self.world_size == 4
            and (
                column_96
                or (row_parallel and input_size == 12288 and output_size == 7168)
            )
        )
        cooperative_cluster_n = (
            2 if not row_parallel and self.input_multicast_ptr else 1
        )
        if cooperative_projection and capacity >= 128:
            cluster_size = 2 * cooperative_cluster_n
            active = (
                utils.HardwareInfo(device_id=device.index).get_max_active_clusters(
                    cluster_size
                )
                * cluster_size
            )
            active = torch.tensor(active, dtype=torch.int32, device=device)
            dist.all_reduce(active, op=dist.ReduceOp.MIN, group=group)
            cluster_ctas = int(active.item()) // 4 * 4
        column_512_ctas = 0
        if column_96 and self.input_multicast_ptr and 512 in self.row_views:
            active_pairs = utils.HardwareInfo(
                device_id=device.index
            ).get_max_active_clusters(2)
            active = torch.tensor(
                min(active_pairs * 2, self.active_ctas),
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(active, op=dist.ReduceOp.MIN, group=group)
            # The preferred launch groups two MMA pairs into a four-CTA cluster.
            column_512_ctas = int(active.item()) // 4 * 4
        # The first eight CTAs include input publishers; arrival counts are bytes.
        column_512_cooperative = 8 <= column_512_ctas < 256
        self.can_borrow_column_512 = (
            column_512_cooperative and column_512_ctas == self.active_ctas
        )
        for rows in self.row_views:
            resident_ctas = (
                128
                if quantized
                and row_parallel
                and self.world_size == 2
                and input_size == 12288
                and output_size == 7168
                and rows < 512
                else self.active_ctas
            )
            # Keep eight GEMM waves at 384 rows; align the 512-row grid with
            # its sixteen M tiles so each CTA retains the same input tile.
            if (
                quantized
                and not row_parallel
                and self.world_size == 4
                and input_size == 7168
                and output_size == 49664
            ):
                if rows == 384:
                    resident_ctas = 148
                elif rows == 512:
                    resident_ctas = 144
            if (
                quantized
                and row_parallel
                and self.world_size == 4
                and input_size == 12288
                and output_size == 7168
                and rows == 192
            ):
                # Keep each CTA on the same M tile and reduce idle grid work.
                resident_ctas = 132
            resident_ctas = min(resident_ctas, self.active_ctas)
            inputs = torch.zeros(
                (rows, input_size), dtype=torch.bfloat16, device=device
            )
            out = torch.empty((rows, output_size), dtype=torch.bfloat16, device=device)
            # At 32 rows per rank, TP4 needs more GEMM tiles to occupy the GPU.
            tile_m = 64 if row_parallel and self.world_size == 4 and rows == 32 else 128
            # Pair adjacent N tiles to share activation loads through TMA.
            cluster_shape = (
                (1, 2)
                if quantized
                and row_parallel
                and self.world_size == 4
                and input_size == 12288
                and output_size == 7168
                and rows in (2048, 2304, 4096, 4608)
                else (1, 1)
            )
            tile_n = 96 if column_96 and self.shard_size % 128 else 128
            column_512 = column_512_cooperative and rows == 512
            cooperative_bf16 = column_512 or (
                cooperative_projection
                and (rows == 128 or (row_parallel and rows == 512))
            )
            if cooperative_bf16:
                tile_m = 256
                cluster_n = 1 if column_512 else cooperative_cluster_n
                if column_512:
                    tile_n = 192
                elif cluster_n == 2:
                    tile_n = 128
                cluster_shape = (2, cluster_n)
                resident_ctas = min(
                    resident_ctas, column_512_ctas if column_512 else cluster_ctas
                )
                if (
                    row_parallel
                    and rows == 512
                    and self.input_multicast_ptr
                    and resident_ctas == 152
                ):
                    tile_n = 256
            kernel = FusedTPProjectionKernel(
                cutlass.Float32,
                cooperative_bf16,
                (tile_m, tile_n),
                cluster_shape,
                row_parallel=row_parallel,
                accumulator_groups=(
                    2
                    if (row_parallel or self.world_size == 4)
                    and rows >= 128
                    and tile_n != 96
                    else 1
                ),
                use_input_multicast=bool(self.input_multicast_ptr),
                prefetch_weights=(
                    not quantized
                    and self.world_size == 4
                    and (
                        rows == 128
                        or (cooperative_projection and row_parallel and rows == 512)
                    )
                ),
                quantized=quantized,
            )
            input_dtypes = (
                (torch.bfloat16, torch.float8_e4m3fn)
                if row_parallel and quantized
                else (torch.bfloat16,)
            )
            for input_dtype in input_dtypes:
                inputs = inputs.to(input_dtype)
                input_scales = (
                    torch.ones(
                        (rows, input_size // 128), dtype=torch.float32, device=device
                    )
                    if input_dtype == torch.float8_e4m3fn
                    else None
                )
                args = self._arguments(
                    inputs, input_scales, weight, scales, rows, out, rows
                )
                key = (rows, input_dtype)
                self.compiled[key] = cute.compile(
                    kernel,
                    *args[:5],
                    resident_ctas,
                    *args[5:],
                    lambda v: v,
                    options="--opt-level=2",
                )
                # Warm both input formats before capture; they share device epochs.
                self.compiled[key](*args)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _arguments(
        self,
        inputs: torch.Tensor,
        input_scales: torch.Tensor | None,
        weight: torch.Tensor,
        scales: torch.Tensor | None,
        bucket: int,
        out: torch.Tensor,
        live_rows: int,
    ) -> tuple:
        a, sa, peers = self.row_views[bucket]
        return (
            a,
            from_dlpack(weight.unsqueeze(-1), assumed_align=16),
            peers[0],
            sa,
            (
                from_dlpack(scales.unsqueeze(-1), assumed_align=16)
                if scales is not None
                else None
            ),
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
            from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=out.dim_order()
            ),
            peers,
            self.ready_cute,
            self.peer_ready,
            self.ack_cute,
            self.peer_ack_local,
            cutlass.Int32(self.rank),
            from_dlpack(inputs, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=inputs.dim_order()
            ),
            (
                from_dlpack(input_scales, assumed_align=16).mark_compact_shape_dynamic(
                    mode=0, stride_order=input_scales.dim_order()
                )
                if input_scales is not None
                else None
            ),
            self.input_peers,
            self.scale_peers if self.quantized else None,
            self.input_flags_cute,
            (
                self.peer_input_flags_local
                if self.row_parallel and bucket >= 160 and self.input_multicast_ptr
                else self.peer_input_flags
            ),
            cutlass.Int64(self.input_multicast_ptr),
            cutlass.Int32(live_rows),
        )

    def borrow_column_output(self, owner_rows: tuple[int, ...]) -> torch.Tensor | None:
        """Return the native column receive view for a same-stream consumer.

        Consume it before the next projection using this workspace. Each peer
        waits for the destination's next input publication before overwriting it.
        """
        if self.closed or not self.can_borrow_column_512:
            return None
        if owner_rows == (512,) * self.world_size:
            return self.receive[:512]
        return None

    def run(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor | None,
        owner_rows: tuple[int, ...],
        out: torch.Tensor,
        *,
        input_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch the prepared BF16 or block-FP8 projection specialization."""
        if self.closed:
            raise RuntimeError("Fused TP projection workspace is closed")
        if (
            len(owner_rows) != self.world_size
            or min(owner_rows) < 0
            or max(owner_rows) > self.max_rows
        ):
            raise ValueError("Fused TP projection counts exceed prepared capacity")
        rows = owner_rows[self.rank]
        live_rows = max(owner_rows)
        bucket = _row_bucket(live_rows, self.row_parallel, self.world_size)
        if input_scales is not None and not self.row_parallel:
            raise ValueError("Prequantized input is supported for row projection")
        if not self.quantized and (
            weight_scales is not None or input_scales is not None
        ):
            raise ValueError("BF16 projection does not accept quantization scales")
        if self.quantized and weight_scales is None:
            raise ValueError("Block-FP8 projection requires weight scales")
        tensors = [
            (
                "inputs",
                inputs,
                (rows, self.input_size),
                torch.float8_e4m3fn if input_scales is not None else torch.bfloat16,
            ),
            (
                "weight",
                weight,
                (self.shard_size, self.gemm_input_size),
                torch.float8_e4m3fn if self.quantized else torch.bfloat16,
            ),
            ("out", out, (rows, self.output_size), torch.bfloat16),
        ]
        if weight_scales is not None:
            tensors.append(
                (
                    "weight_scales",
                    weight_scales,
                    (self.shard_size // 128, self.gemm_input_size // 128),
                    torch.float32,
                )
            )
        if input_scales is not None:
            tensors.append(
                (
                    "input_scales",
                    input_scales,
                    (rows, self.input_size // 128),
                    torch.float32,
                )
            )
        for name, tensor, shape, dtype in tensors:
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.device
            ):
                raise ValueError(
                    f"Fused TP projection {name} must have shape {shape} "
                    f"and dtype {dtype} on {self.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"Fused TP projection {name} must be contiguous")
            if tensor.data_ptr() % 16:
                raise ValueError(f"Fused TP projection {name} must be 16-byte aligned")
        if live_rows == 0:
            return out
        self.compiled[bucket, inputs.dtype](
            *self._arguments(
                inputs, input_scales, weight, weight_scales, bucket, out, live_rows
            )
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
        self.peer_input_flags_local = self.peer_ack_local = ()
        self.input_multicast_ptr = 0
        self.ready_cute = self.ack_cute = self.input_flags_cute = None
        self.peer_tensors = ()
        self.handles = []
        self.a = self.sa = self.receive = self.ready = self.ack = self.input_flags = (
            None
        )
        self.closed = True
