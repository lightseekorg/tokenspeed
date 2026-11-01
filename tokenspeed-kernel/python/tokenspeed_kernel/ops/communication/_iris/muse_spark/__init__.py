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


"""Muse Spark's prepared TP4/TP8 policy and decoder fusions."""

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.communication._iris import (
    HEAP_HEADROOM_BYTES,
    MIN_HEAP_BYTES,
    IrisGroup,
    allocation_bytes,
    get_or_create_context,
)
from tokenspeed_kernel.ops.communication._iris.all_gather import (
    IrisAllGather,
    all_gather_columns_can_run,
    iris_fused_lamport_allgather_rmsnorm_mxfp4,
    iris_fused_pull_allgather_rmsnorm_mxfp4,
    iris_pull_allgather_rows,
)
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    ProducerAllReduce,
    iris_arch_policy,
)
from tokenspeed_kernel.ops.communication._iris.broadcast import IrisBroadcast
from tokenspeed_kernel.ops.communication._iris.muse_spark.all_gather import (
    iris_pack_moe_inputs,
    iris_pull_allgather_moe_inputs,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.all_to_all import (
    iris_a2a_transpose_qkvg,
    iris_fused_a2a_transpose_qkvg_norm_rope,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
    iris_fused_lamport_allreduce_muse_attntail,
    iris_fused_pull_allgather_muse_attntail,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (
    iris_fused_gemv_lamport_allgather_muse_moetail,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
    iris_fused_lamport_allreduce_routed_rmsnorm_sum,
    iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum,
    iris_routed_rmsnorm_weighted_sum_write_through,
)
from tokenspeed_kernel.ops.communication._iris.reduce_scatter import (
    iris_pull_reduce_scatter,
)
from tokenspeed_kernel.ops.communication.iris import IrisAllReduce
from tokenspeed_kernel.ops.moe import gather_moe_routes
from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        gluon_rowcta_bf16_gemv_geometry_gfx950,
    )

_MUSE_BOUNDARY_MAX_ROWS = 26
_MUSE_WEIGHTED_BOUNDARY_MAX_ROWS = 25
_MUSE_BOUNDARY_WIDTH = 8192
# Workgroups that push each small-boundary row; one more waits and normalizes.
_MUSE_BOUNDARY_SENDERS = 8
# From this many rows, staggering each sender's peer order spreads link load.
_MUSE_BOUNDARY_ROTATE_MIN_ROWS = 5
# Largest row count for the fused routed Lamport when the TP8 Lamport two-shot
# is prepared; above it, the separate routed norm plus the two-shot is faster.
_MUSE_ROUTED_LAMPORT_TWO_SHOT_MAX_ROWS = 32
_MUSE_ROW_BOUNDARY_MIN_ROWS = 512
_MUSE_OWNED_MIN_ROWS = 192
_MUSE_ROW_SCATTER_PROGRAMS = 24
_MUSE_ROW_GATHER_PROGRAMS = 128
_MUSE_TOPK = 8
_MUSE_ROUTE_WIDTH = 4096
# Most decode rows for the fused post-expert tail at TP8; it lost to the
# replicated projection plus boundary at 16 rows in 8-rank graph replay.
# Rows match the single-row tail bit for bit within 1-8 rows; 16 rows take
# the replicated path.
_MUSE_MOETAIL_MAX_ROWS = 8


class MuseIrisState:
    """Compose prepared generic collectives and the model's boundary storage.

    All layers reuse these buffers in stream order. Reductions and gathers can
    return borrowed activations; a consumer must finish before the next write.
    The generic all-reduce retains its independent-output contract.
    Expert TP and EP use the same owned-row token cutoffs; each owner
    holds ceil(global rows / world size) rows, with padding masked.
    """

    @staticmethod
    def heap_bytes(
        max_numel: int,
        *,
        world_size: int,
        enable_lamport: bool,
        owned_route_max_rows: int,
    ) -> int:
        rows = min(max_numel // _MUSE_BOUNDARY_WIDTH, _MUSE_BOUNDARY_MAX_ROWS)
        size = ProducerAllReduce.heap_bytes(
            max_numel, world_size=world_size, enable_lamport=enable_lamport
        ) + IrisAllGather.heap_bytes(world_size)
        if enable_lamport:
            size += IrisBroadcast.heap_bytes(max_numel)
        if rows:
            size += allocation_bytes(
                (3, world_size, rows, _MUSE_BOUNDARY_WIDTH), torch.bfloat16
            )
        if max_numel >= _MUSE_ROW_BOUNDARY_MIN_ROWS * _MUSE_BOUNDARY_WIDTH:
            size += allocation_bytes(
                (_MUSE_ROW_SCATTER_PROGRAMS, world_size), torch.uint32
            )
            size += allocation_bytes(
                (2, _MUSE_ROW_GATHER_PROGRAMS, world_size), torch.uint32
            )
        if owned_route_max_rows:
            size += allocation_bytes(
                (owned_route_max_rows, _MUSE_TOPK, _MUSE_ROUTE_WIDTH), torch.bfloat16
            )
            size += allocation_bytes(
                (_MUSE_ROW_SCATTER_PROGRAMS, world_size), torch.uint32
            )
        return size

    def __init__(
        self,
        all_reduce: IrisAllReduce,
        *,
        enable_lamport: bool,
        owned_route_max_rows: int,
    ):
        self._arch = iris_arch_policy()
        self.all_reduce = all_reduce
        self.world_size = all_reduce.world_size
        self.enable_lamport = enable_lamport
        self.reduction = ProducerAllReduce(
            all_reduce.producer_pull, enable_lamport=enable_lamport
        )
        self.gather = IrisAllGather(all_reduce.producer)
        self.routed_lamport_max_rows = (
            _MUSE_ROUTED_LAMPORT_TWO_SHOT_MAX_ROWS
            if self.reduction.lamport_two_shot_max_numel
            else self.reduction.lamport_max_numel // _MUSE_ROUTE_WIDTH
        )
        peers = all_reduce.peers
        capacity = all_reduce.producer_direct_max_numel
        self.broadcast = IrisBroadcast(peers, capacity) if enable_lamport else None
        self.boundary_max_rows = min(
            capacity // _MUSE_BOUNDARY_WIDTH, _MUSE_BOUNDARY_MAX_ROWS
        )
        self.row_boundary_supported = (
            capacity >= _MUSE_ROW_BOUNDARY_MIN_ROWS * _MUSE_BOUNDARY_WIDTH
            and all_reduce.producer_pull.scratch.numel()
            >= triton.cdiv(_MUSE_ROW_BOUNDARY_MIN_ROWS, self.world_size)
            * _MUSE_BOUNDARY_WIDTH
        )
        # Three rotating Lamport slots per row; 0x8000 (negative zero) marks empty.
        self.boundary_mailbox = (
            peers.ctx.empty(
                (3, self.world_size, self.boundary_max_rows, _MUSE_BOUNDARY_WIDTH),
                dtype=all_reduce.dtype,
            )
            if self.boundary_max_rows
            else None
        )
        if self.boundary_mailbox is not None:
            self.boundary_mailbox.view(torch.int16).fill_(-32768)
        self.boundary_epochs = (
            torch.zeros(
                (self.boundary_max_rows, _MUSE_BOUNDARY_SENDERS + 1),
                dtype=torch.int32,
                device=peers.device,
            )
            if self.boundary_max_rows
            else None
        )
        self.row_scatter_flags = (
            peers.ctx.zeros(
                (_MUSE_ROW_SCATTER_PROGRAMS, self.world_size), dtype=torch.uint32
            )
            if self.row_boundary_supported
            else None
        )
        self.row_gather_flags = (
            peers.ctx.zeros(
                (2, _MUSE_ROW_GATHER_PROGRAMS, self.world_size), dtype=torch.uint32
            )
            if self.row_boundary_supported
            else None
        )
        # Expert TP stages raw routes separately from the producer buffer.
        self.owned_route_max_rows = owned_route_max_rows
        self.owned_route_input = (
            peers.ctx.empty(
                (owned_route_max_rows, _MUSE_TOPK, _MUSE_ROUTE_WIDTH),
                dtype=torch.bfloat16,
            )
            if owned_route_max_rows
            else None
        )
        self.owned_route_output = (
            torch.empty(
                (
                    triton.cdiv(owned_route_max_rows, self.world_size),
                    _MUSE_TOPK,
                    _MUSE_ROUTE_WIDTH,
                ),
                dtype=torch.bfloat16,
                device=peers.device,
            )
            if owned_route_max_rows
            else None
        )
        self.owned_route_scatter_flags = (
            peers.ctx.zeros(
                (_MUSE_ROW_SCATTER_PROGRAMS, self.world_size), dtype=torch.uint32
            )
            if owned_route_max_rows
            else None
        )

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce an ordinary tensor in place using the prepared TP4/TP8 policy."""
        numel = tensor.numel()
        assert tensor.dtype == self.all_reduce.dtype and tensor.is_contiguous()
        capacity = self.all_reduce.producer_direct_max_numel
        capacity -= capacity % (self.world_size * 8)
        assert capacity > 0 and numel % (self.world_size * 8) == 0
        # Oversized full-row fallbacks retain Iris without growing prepared state.
        flat = tensor.view(-1)
        for begin in range(0, numel, capacity):
            chunk = flat[begin : begin + capacity]
            reduced = self.reduction.run(chunk, chunk, chunk.numel())
            if reduced.data_ptr() != chunk.data_ptr():
                chunk.copy_(reduced[: chunk.numel()])
        return tensor

    def reduce_outputs(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Consume consecutive producer views and return borrowed reduced views."""
        assert self.all_reduce.owns_outputs(tensors)
        shapes = tuple(tuple(t.shape) for t in tensors)
        numel = sum(t.numel() for t in tensors)
        if numel % (self.world_size * 8):
            return self.all_reduce.all_reduce_symmetric(tensors)
        reduced = self.reduction.run(
            self.all_reduce.producer.data, self.gather.output, numel
        )
        return self.all_reduce.producer.views(reduced, shapes)

    def can_use_owned_rows(self, rows: int) -> bool:
        """Check capacity for splitting token rows across the group."""
        local_elements = triton.cdiv(rows, self.world_size) * _MUSE_BOUNDARY_WIDTH
        return (
            self.row_boundary_supported
            and _MUSE_OWNED_MIN_ROWS <= rows
            and rows * _MUSE_BOUNDARY_WIDTH <= self.all_reduce.producer_direct_max_numel
            and local_elements <= self.all_reduce.producer_pull.scratch.numel()
            and self.all_reduce.producer_direct_max_numel // 2 + local_elements
            <= self.all_reduce.producer_direct_max_numel
        )

    def acquire_owned_qkvg_output(self, rows: int) -> torch.Tensor:
        """Borrow the producer prefix until the QKVG transpose has completed."""
        assert self.can_use_owned_rows(rows)
        local_rows = triton.cdiv(rows, self.world_size)
        elements = local_rows * 18432
        assert elements <= self.all_reduce.producer_direct_max_numel
        return self.all_reduce.producer.data[:elements].view(local_rows, 18432)

    def all_to_all_transpose_qkvg(
        self, tensor: torch.Tensor, rows: int
    ) -> torch.Tensor:
        """Turn owned token rows into this rank's Q/K/V/gate head shard for all tokens.

        The source must be the completed producer-direct QKVG projection. The
        result owns its storage; the transpose exit protects input reuse by O.
        """
        expected = self.acquire_owned_qkvg_output(rows)
        assert (
            tensor.shape == expected.shape and tensor.data_ptr() == expected.data_ptr()
        )
        assert tensor.is_contiguous() and tensor.dtype == torch.bfloat16
        assert tensor.device == self.all_reduce.device
        output = torch.empty(
            (rows, 18432 // self.world_size),
            dtype=tensor.dtype,
            device=self.all_reduce.device,
        )
        iris_a2a_transpose_qkvg[(_MUSE_ROW_GATHER_PROGRAMS,)](
            tensor,
            output,
            self.row_gather_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            rows,
            RANK=self.all_reduce.rank_in_group,
            FLAG_ROWS=_MUSE_ROW_GATHER_PROGRAMS,
            BLOCK=2048,
            NUM_WARPS=4,
            num_warps=4,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return output

    def all_to_all_transpose_qkvg_norm_rope(
        self,
        tensor: torch.Tensor,
        rows: int,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        eps: float,
        q_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return owned QKVG as contiguous, normalized prefill Q/K and V/gate.

        PagedAttention stores these prepared K/V rows through its prologue.
        Peer reads finish before the output projection reuses the input.
        """
        expected = self.acquire_owned_qkvg_output(rows)
        assert (
            tensor.shape == expected.shape and tensor.data_ptr() == expected.data_ptr()
        )
        assert tensor.is_contiguous() and tensor.dtype == torch.bfloat16
        assert tensor.device == self.all_reduce.device
        assert positions.shape == (rows,) and positions.is_contiguous()
        assert (
            positions.dtype in (torch.int32, torch.int64)
            and positions.device == self.all_reduce.device
        )
        assert cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] in (0, 64)
        assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()
        assert cos_sin_cache.device == self.all_reduce.device
        outputs = tuple(
            torch.empty(
                (rows, width // self.world_size),
                dtype=tensor.dtype,
                device=self.all_reduce.device,
            )
            for width in (8192, 1024, 1024, 8192)
        )
        iris_fused_a2a_transpose_qkvg_norm_rope[(_MUSE_ROW_GATHER_PROGRAMS,)](
            tensor,
            *outputs,
            positions,
            cos_sin_cache,
            self.row_gather_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            rows,
            RANK=self.all_reduce.rank_in_group,
            FLAG_ROWS=_MUSE_ROW_GATHER_PROGRAMS,
            ROPE=cos_sin_cache.shape[1] == 64,
            EPS=eps,
            Q_SCALE=q_scale,
            NUM_WARPS=8,
            num_warps=8,
            enable_fp_fusion=False,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return outputs

    def acquire_owned_post_expert_output(self, rows: int, width: int) -> torch.Tensor:
        """Borrow the producer buffer after the region holding pre-expert rows."""
        assert self.can_use_owned_rows(rows) and width == _MUSE_BOUNDARY_WIDTH
        local_rows = triton.cdiv(rows, self.world_size)
        start = self.all_reduce.producer_direct_max_numel // 2
        return self.all_reduce.producer.data[start : start + local_rows * width].view(
            local_rows, width
        )

    def reduce_scatter_owned_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce full BF16 rows and finish all peer reads before returning."""
        rows, width = tensor.shape
        assert self.can_use_owned_rows(rows) and width in (4096, 8192)
        assert tensor.is_contiguous() and tensor.dtype == torch.bfloat16
        assert tensor.device == self.all_reduce.device
        if tensor.data_ptr() != self.all_reduce.producer.data.data_ptr():
            self.all_reduce.producer.data[: tensor.numel()].copy_(tensor.view(-1))
        local_rows = triton.cdiv(rows, self.world_size)
        output = self.all_reduce.producer_pull.scratch[: local_rows * width].view(
            local_rows, width
        )
        iris_pull_reduce_scatter[(_MUSE_ROW_SCATTER_PROGRAMS,)](
            self.all_reduce.producer.data,
            output,
            self.row_scatter_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            ROWS=rows,
            RANK=self.all_reduce.rank_in_group,
            WIDTH=width,
            BLOCK=2048,
            PROGRAMS=_MUSE_ROW_SCATTER_PROGRAMS,
            NUM_WARPS=4,
            EXIT_BARRIER=True,
            PRODUCER_WRITES_THROUGH=False,
            num_warps=4,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return output

    def acquire_owned_route_input(self, rows: int) -> torch.Tensor:
        """Borrow BF16 ``[rows * 8, 4096]`` route staging for an expert producer.

        Write it on the stream of the next ``reduce_scatter_owned_routes``, which
        then skips staging.
        """
        assert 0 < rows <= self.owned_route_max_rows
        return self.owned_route_input[:rows].view(rows * _MUSE_TOPK, _MUSE_ROUTE_WIDTH)

    def reduce_scatter_owned_routes(
        self, routes: torch.Tensor, route_map: torch.Tensor | None, rows: int
    ) -> torch.Tensor:
        """Sum expert TP route partials onto token owners before route norms.

        ``routes`` holds this rank's partial outputs; ``route_map`` maps
        canonical token/route order to its rows, else ``routes`` is canonical
        and contiguous; an ``acquire_owned_route_input`` view skips staging, other
        overlapping sources are rejected. Returns borrowed BF16 ``[ceil(rows/P), 8, 4096]``
        routes with zero padding rows; consume them before the next call.
        """
        assert 0 < rows <= self.owned_route_max_rows
        assert (
            routes.dtype == torch.bfloat16 and routes.device == self.all_reduce.device
        )
        staged = self.owned_route_input[:rows]
        if route_map is None:
            assert routes.is_contiguous() and routes.numel() == staged.numel()
            if routes.data_ptr() != staged.data_ptr():
                size = staged.numel() * staged.element_size()
                if (
                    routes.data_ptr() < staged.data_ptr() + size
                    and staged.data_ptr() < routes.data_ptr() + size
                ):
                    raise ValueError("route staging must not overlap its source")
                staged.copy_(routes.view_as(staged))
        else:
            gather_moe_routes(
                routes, route_map, rows, _MUSE_TOPK, _MUSE_ROUTE_WIDTH, output=staged
            )
        output = self.owned_route_output[: triton.cdiv(rows, self.world_size)]
        # Partition whole tokens: all eight routes must reach the same owner.
        iris_pull_reduce_scatter[(_MUSE_ROW_SCATTER_PROGRAMS,)](
            staged,
            output,
            self.owned_route_scatter_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            ROWS=rows,
            RANK=self.all_reduce.rank_in_group,
            WIDTH=_MUSE_TOPK * _MUSE_ROUTE_WIDTH,
            BLOCK=2048,
            PROGRAMS=_MUSE_ROW_SCATTER_PROGRAMS,
            NUM_WARPS=4,
            EXIT_BARRIER=True,
            PRODUCER_WRITES_THROUGH=False,
            num_warps=4,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return output

    def reduce_scatter_owned_routed_rmsnorm_sum(
        self,
        routes: torch.Tensor,
        route_map: torch.Tensor,
        num_tokens: int,
        topk: int,
        route_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        weight_offset: float,
    ) -> torch.Tensor | None:
        """Normalize and weight expert routes, then reduce to this rank's rows.

        Expert inputs must not overlap the shared producer buffer. The result
        borrows pull scratch until the next producer or collective call.
        """
        if not (
            num_tokens >= 192
            and self.can_use_owned_rows(num_tokens)
            and topk == 8
            and routes.ndim == 2
            and routes.shape[1] == 4096
            and routes.is_contiguous()
            and routes.dtype == torch.bfloat16
            and route_map.shape == (num_tokens * topk,)
            and route_map.dtype == torch.int32
            and route_map.is_contiguous()
            and route_weights.shape == (num_tokens, topk)
            and route_weights.dtype == torch.float32
            and route_weights.is_contiguous()
            and norm_weight.shape == (4096,)
            and norm_weight.dtype == torch.bfloat16
            and norm_weight.is_contiguous()
            and all(
                value.device == self.all_reduce.device
                for value in (routes, route_map, route_weights, norm_weight)
            )
        ):
            return None
        begin = self.all_reduce.producer.data.data_ptr()
        end = begin + num_tokens * 4096 * 2
        assert all(
            value.data_ptr() + value.numel() * value.element_size() <= begin
            or value.data_ptr() >= end
            for value in (routes, route_map, route_weights, norm_weight)
        ), "expert inputs must not alias the owned routed-sum prefix"
        iris_routed_rmsnorm_weighted_sum_write_through[(num_tokens,)](
            routes,
            route_map,
            route_weights,
            norm_weight,
            self.all_reduce.producer.data,
            routes.stride(0),
            EPS=eps,
            WEIGHT_OFFSET=weight_offset,
            num_warps=8,
            WAIT_STORES=self._arch.wait_stores,
        )
        local_rows = triton.cdiv(num_tokens, self.world_size)
        output = self.all_reduce.producer_pull.scratch[: local_rows * 4096].view(
            local_rows, 4096
        )
        # The local producer uses write-through stores; peer reads need no cache writeback.
        iris_pull_reduce_scatter[(_MUSE_ROW_SCATTER_PROGRAMS,)](
            self.all_reduce.producer.data,
            output,
            self.row_scatter_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            ROWS=num_tokens,
            RANK=self.all_reduce.rank_in_group,
            WIDTH=4096,
            BLOCK=2048,
            PROGRAMS=_MUSE_ROW_SCATTER_PROGRAMS,
            NUM_WARPS=4,
            EXIT_BARRIER=True,
            PRODUCER_WRITES_THROUGH=True,
            num_warps=4,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return output

    def all_gather_owned_rows(
        self,
        tensor: torch.Tensor,
        rows: int,
        producer_writes_through: bool,
    ) -> torch.Tensor:
        """Gather each rank's BF16 token rows into complete projection inputs."""
        assert self.can_use_owned_rows(rows)
        local_rows = triton.cdiv(rows, self.world_size)
        assert tensor.shape == (local_rows, _MUSE_BOUNDARY_WIDTH)
        assert tensor.dtype == torch.bfloat16 and tensor.is_contiguous()
        assert tensor.device == self.all_reduce.device
        pointer = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        symmetric = any(
            buffer.data_ptr() <= pointer
            and pointer + size <= buffer.data_ptr() + buffer.numel() * 2
            for buffer in (
                self.all_reduce.producer.data,
                self.all_reduce.producer_pull.scratch,
            )
        )
        if not symmetric:
            staging = self.all_reduce.producer_pull.scratch[: tensor.numel()].view_as(
                tensor
            )
            staging.copy_(tensor)
            tensor = staging
            producer_writes_through = False
        output = torch.empty(
            (rows, _MUSE_BOUNDARY_WIDTH),
            dtype=tensor.dtype,
            device=self.all_reduce.device,
        )
        iris_pull_allgather_rows[(_MUSE_ROW_GATHER_PROGRAMS,)](
            tensor,
            output,
            self.row_gather_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            rows,
            RANK=self.all_reduce.rank_in_group,
            WIDTH=tensor.shape[1],
            PRODUCER_WRITES_THROUGH=producer_writes_through,
            FLAG_ROWS=_MUSE_ROW_GATHER_PROGRAMS,
            BLOCK=2048,
            NUM_WARPS=4,
            num_warps=4,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return output

    def gather_owned_moe_inputs(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ids: torch.Tensor,
        weights: torch.Tensor,
        rows: int,
        valid_rows: int,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """Gather pre-expert activations, expert IDs, and weights in one packed buffer."""
        assert self.can_use_owned_rows(rows)
        local_rows = triton.cdiv(rows, self.world_size)
        assert valid_rows == min(
            local_rows, max(0, rows - self.all_reduce.rank_in_group * local_rows)
        )
        quantized = isinstance(inputs, tuple)
        if quantized:
            data, scales = inputs
            assert data.shape == (local_rows, 2048) and data.dtype == torch.uint8
            assert scales.shape == (4096, triton.cdiv(local_rows, 32))
            assert scales.dtype == torch.uint8 and scales.stride() == (1, 4096)
            output = torch.empty(
                (rows, 2048), dtype=torch.uint8, device=self.all_reduce.device
            )
            output_scales = torch.empty(
                (triton.cdiv(rows, 32), 4096),
                dtype=torch.uint8,
                device=self.all_reduce.device,
            ).T
            result = (output, output_scales)
        else:
            data = inputs
            assert data.shape == (local_rows, 4096) and data.dtype == torch.bfloat16
            scales = data.view(torch.uint8)
            output = torch.empty(
                (rows, 4096), dtype=torch.bfloat16, device=self.all_reduce.device
            )
            output_scales = output.view(torch.uint8)
            result = output
        assert ids.shape == weights.shape == (local_rows, 8)
        assert ids.dtype == torch.int32 and weights.dtype == torch.float32
        assert all(
            value.device == self.all_reduce.device
            for value in (data, scales, ids, weights)
        )
        assert all(value.is_contiguous() for value in (data, ids, weights))
        packet_bytes = 2240 if quantized else 8256
        byte_count = local_rows * packet_bytes
        # A completed pull proves all old producer-input reads have finished;
        # peers may still read its scratch until the next synchronization.
        packet = self.acquire_owned_post_expert_output(rows, 8192).view(torch.uint8)
        packet = packet.reshape(-1)[:byte_count]
        packet_begin, packet_end = packet.data_ptr(), packet.data_ptr() + byte_count
        assert all(
            value.data_ptr() + value.numel() * value.element_size() <= packet_begin
            or value.data_ptr() >= packet_end
            for value in (data, scales, ids, weights)
        ), "pre-expert inputs and routes must not alias the packet staging region"
        route_storage = torch.empty(
            (2, rows, 8), dtype=torch.int32, device=self.all_reduce.device
        )
        output_ids, output_weights = (
            route_storage[0],
            route_storage[1].view(torch.float32),
        )
        iris_pack_moe_inputs[(triton.cdiv(byte_count, 2048),)](
            data.view(torch.uint8),
            scales,
            ids.view(torch.uint8),
            weights.view(torch.uint8),
            packet,
            local_rows,
            valid_rows,
            QUANTIZED=quantized,
            BLOCK=2048,
            NUM_WARPS=4,
            num_warps=4,
        )
        iris_pull_allgather_moe_inputs[(_MUSE_ROW_GATHER_PROGRAMS,)](
            packet,
            output.view(torch.uint8),
            output_scales,
            output_ids.view(torch.uint8),
            output_weights.view(torch.uint8),
            self.row_gather_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            rows,
            RANK=self.all_reduce.rank_in_group,
            QUANTIZED=quantized,
            FLAG_ROWS=_MUSE_ROW_GATHER_PROGRAMS,
            BLOCK=2048,
            NUM_WARPS=4,
            num_warps=4,
            WAIT_VMEM=self._arch.wait_vmem,
        )
        return result, output_ids, output_weights

    def can_all_reduce_symmetric_boundary(
        self,
        branch: torch.Tensor,
        stream: torch.Tensor,
        post_weight: torch.Tensor | None,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_weight: torch.Tensor,
    ) -> bool:
        """Check the measured small push or large row-scatter boundary policy."""
        max_rows = (
            min(self.boundary_max_rows, _MUSE_WEIGHTED_BOUNDARY_MAX_ROWS)
            if post_weight is not None
            else self.boundary_max_rows
        )
        if (
            branch.ndim != 2
            or not (
                0 < branch.shape[0] <= max_rows
                or (
                    self.row_boundary_supported
                    and post_weight is None
                    and branch.shape[0] >= _MUSE_ROW_BOUNDARY_MIN_ROWS
                    and triton.cdiv(branch.shape[0], self.world_size)
                    * _MUSE_BOUNDARY_WIDTH
                    <= self.all_reduce.producer_pull.scratch.numel()
                )
            )
            or branch.shape[1] != _MUSE_BOUNDARY_WIDTH
            or branch.dtype != torch.bfloat16
            or not self.all_reduce.owns_outputs((branch,))
            or stream.shape != branch.shape
            or stream.dtype != torch.float32
            or stream.device != branch.device
            or not stream.is_contiguous()
        ):
            return False
        for value, dtype in (
            (pre_weight, torch.bfloat16),
            (alpha, torch.float32),
            (beta, torch.float32),
        ):
            if (
                value.shape != (_MUSE_BOUNDARY_WIDTH,)
                or value.dtype != dtype
                or value.device != branch.device
                or not value.is_contiguous()
            ):
                return False
        return post_weight is None or (
            post_weight.shape == (_MUSE_BOUNDARY_WIDTH,)
            and post_weight.dtype == torch.bfloat16
            and post_weight.device == branch.device
            and post_weight.is_contiguous()
        )

    def all_reduce_symmetric_boundary(
        self,
        branch: torch.Tensor,
        stream: torch.Tensor,
        post_weight: torch.Tensor | None,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_weight: torch.Tensor,
        post_eps: float,
        post_weight_offset: float,
        pre_eps: float,
        pre_weight_offset: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce the attention output, update the residual, and normalize for MoE.

        The FP32 normalized rows feed the router; BF16 rows feed the pre-expert projection.
        """
        assert self.can_all_reduce_symmetric_boundary(
            branch, stream, post_weight, alpha, beta, pre_weight
        )
        rows, width = branch.shape
        if rows >= _MUSE_ROW_BOUNDARY_MIN_ROWS:
            assert self.all_reduce.producer_pull.scratch is not None
            assert self.row_scatter_flags is not None
            assert self.row_gather_flags is not None
            local_rows = triton.cdiv(rows, self.world_size)
            shard = self.all_reduce.producer_pull.scratch[: local_rows * width]
            iris_pull_reduce_scatter[(_MUSE_ROW_SCATTER_PROGRAMS,)](
                branch,
                shard,
                self.row_scatter_flags,
                self.all_reduce.peers.addresses[: self.world_size],
                ROWS=rows,
                RANK=self.all_reduce.rank_in_group,
                WIDTH=width,
                BLOCK=2048,
                PROGRAMS=_MUSE_ROW_SCATTER_PROGRAMS,
                NUM_WARPS=4,
                EXIT_BARRIER=False,
                PRODUCER_WRITES_THROUGH=False,
                num_warps=4,
                WAIT_VMEM=self._arch.wait_vmem,
            )
            out_stream = torch.empty_like(stream)
            pre_fp32 = torch.empty_like(stream)
            pre_bf16 = torch.empty_like(branch)
            iris_fused_pull_allgather_muse_attntail[(_MUSE_ROW_GATHER_PROGRAMS,)](
                shard,
                stream,
                alpha,
                beta,
                pre_weight,
                out_stream,
                pre_fp32,
                pre_bf16,
                self.row_gather_flags,
                self.all_reduce.peers.addresses[: self.world_size],
                rows=rows,
                local_rows=local_rows,
                RANK=self.all_reduce.rank_in_group,
                WIDTH=width,
                POST_EPS=post_eps,
                PRE_EPS=pre_eps,
                PRE_WEIGHT_OFFSET=pre_weight_offset,
                FLAG_ROWS=_MUSE_ROW_GATHER_PROGRAMS,
                ELEMENTS_PER_THREAD=8,
                NUM_WARPS=16,
                num_warps=16,
                WAIT_VMEM=self._arch.wait_vmem,
            )
            return out_stream, pre_fp32, pre_bf16
        assert self.boundary_mailbox is not None
        assert self.boundary_epochs is not None
        out_stream = torch.empty_like(stream)
        pre_fp32 = torch.empty_like(stream)
        pre_bf16 = torch.empty_like(branch)
        iris_fused_lamport_allreduce_muse_attntail[(rows, _MUSE_BOUNDARY_SENDERS + 1)](
            branch,
            self.boundary_mailbox,
            stream,
            post_weight if post_weight is not None else branch,
            alpha,
            beta,
            pre_weight,
            out_stream,
            pre_fp32,
            pre_bf16,
            self.boundary_epochs,
            self.all_reduce.peers.addresses[: self.world_size],
            RANK=self.all_reduce.rank_in_group,
            WORLD_SIZE=self.all_reduce.world_size,
            WIDTH=width,
            MAX_ROWS=self.boundary_max_rows,
            POST_EPS=post_eps,
            POST_WEIGHT_OFFSET=post_weight_offset,
            PRE_EPS=pre_eps,
            PRE_WEIGHT_OFFSET=pre_weight_offset,
            POST_HAS_WEIGHT=post_weight is not None,
            SENDERS=_MUSE_BOUNDARY_SENDERS,
            ROTATE_PEERS=rows >= _MUSE_BOUNDARY_ROTATE_MIN_ROWS,
            ELEMENTS_PER_THREAD=8,
            NUM_WARPS=16,
            PEER_SIGNAL_STORE_CACHE=self._arch.peer_signal_store,
            num_warps=16,
        )
        return out_stream, pre_fp32, pre_bf16

    def _moetail_geometry(
        self,
        rows: int,
        weight: torch.Tensor,
        stream: torch.Tensor,
        post_weight: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_weight: torch.Tensor | None,
    ) -> tuple[int, int] | None:
        """Return the senders' GEMV geometry if the fused tail applies, else None."""
        reduction = self.reduction
        # TP4 takes one row: more rows were not measured there.
        max_rows = _MUSE_MOETAIL_MAX_ROWS if self.world_size == 8 else 1
        if (
            not current_platform().is_cdna4
            or self.world_size not in (4, 8)
            or not 1 <= rows <= max_rows
            or reduction.lamport_region is None
            or (1 << (rows - 1).bit_length()) * _MUSE_BOUNDARY_WIDTH // self.world_size
            > reduction.lamport_max_numel
            or weight.shape != (_MUSE_BOUNDARY_WIDTH, _MUSE_ROUTE_WIDTH)
            or weight.dtype != torch.bfloat16
            or weight.device != self.all_reduce.device
            or not weight.is_contiguous()
            or stream.shape != (rows, _MUSE_BOUNDARY_WIDTH)
            or stream.dtype != torch.float32
            or stream.device != weight.device
            or not stream.is_contiguous()
        ):
            return None
        vectors = [(post_weight, torch.bfloat16), (alpha, torch.float32)]
        vectors.append((beta, torch.float32))
        if pre_weight is not None:
            vectors.append((pre_weight, torch.bfloat16))
        if not all(
            value is not None
            and value.shape == (_MUSE_BOUNDARY_WIDTH,)
            and value.dtype == dtype
            and value.device == weight.device
            and value.is_contiguous()
            for value, dtype in vectors
        ):
            return None
        # Senders repeat the replicated GEMV's per-output reduction, each
        # storing an even number of outputs in its 16 subgroups.
        block_k, waves = gluon_rowcta_bf16_gemv_geometry_gfx950(*weight.shape)
        if 8 % waves or block_k % (64 * waves) or _MUSE_ROUTE_WIDTH % block_k:
            return None
        return block_k, waves

    def can_gemv_all_gather_boundary(
        self,
        rows: int,
        weight: torch.Tensor,
        stream: torch.Tensor,
        post_weight: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_weight: torch.Tensor | None,
    ) -> bool:
        """Check the decode post-expert projection, gather and boundary contract.

        ``weight`` is the replicated ``[8192, 4096]`` projection; the rows are
        gathered through the Lamport one-shot mailbox. TP8 takes up to
        ``_MUSE_MOETAIL_MAX_ROWS`` rows, TP4 one.
        """
        return (
            self._moetail_geometry(
                rows, weight, stream, post_weight, alpha, beta, pre_weight
            )
            is not None
        )

    def gemv_all_gather_boundary(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        stream: torch.Tensor,
        post_weight: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        pre_weight: torch.Tensor | None,
        post_eps: float,
        post_weight_offset: float,
        pre_eps: float,
        pre_weight_offset: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project this rank's columns of each row, gather the rows, update the residual.

        Returns the FP32 residual and, with ``pre_weight``, the BF16 input of
        the next layer; each row is the replicated single-row projection of
        that row followed by the Triton boundary.
        """
        geometry = self._moetail_geometry(
            x.shape[0], weight, stream, post_weight, alpha, beta, pre_weight
        )
        assert geometry is not None
        rows = x.shape[0]
        assert x.shape == (rows, _MUSE_ROUTE_WIDTH) and x.dtype == torch.bfloat16
        assert x.is_contiguous() and x.device == weight.device
        block_k, waves = geometry
        rank = self.all_reduce.rank_in_group
        local_width = _MUSE_BOUNDARY_WIDTH // self.world_size
        reduction = self.reduction
        out_stream = torch.empty_like(stream)
        pre_bf16 = (
            torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
            if pre_weight is not None
            else None
        )
        # Sixteen-subgroup programs of 16 // waves outputs each, then one
        # reducer per row.
        iris_fused_gemv_lamport_allgather_muse_moetail[
            (local_width * waves // 16 + rows,)
        ](
            x,
            weight[rank * local_width : (rank + 1) * local_width],
            reduction.lamport_region,
            reduction.lamport_epochs,
            reduction.lamport_peer_addresses[: self.world_size],
            stream,
            post_weight,
            alpha,
            beta,
            pre_weight if pre_weight is not None else post_weight,
            out_stream,
            pre_bf16 if pre_bf16 is not None else x,
            rows,
            RANK=rank,
            WORLD_SIZE=self.world_size,
            K=_MUSE_ROUTE_WIDTH,
            LOCAL_WIDTH=local_width,
            MAX_ELEMENTS=reduction.lamport_max_numel,
            BLOCK_M=1 << (rows - 1).bit_length(),
            BLOCK_K=block_k,
            WAVES_PER_OUTPUT=waves,
            POST_EPS=post_eps,
            POST_WEIGHT_OFFSET=post_weight_offset,
            PRE_EPS=pre_eps,
            PRE_WEIGHT_OFFSET=pre_weight_offset,
            HAS_PRE=pre_weight is not None,
            PEER_SIGNAL_STORE_CACHE=self._arch.peer_signal_store,
            num_warps=16,
        )
        return out_stream, pre_bf16

    def can_all_reduce_routed_rmsnorm_sum(
        self,
        routes: torch.Tensor,
        route_map: torch.Tensor,
        num_tokens: int,
        topk: int,
        route_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        weight_offset: float,
    ) -> bool:
        """Check the gfx950 TP4/TP8 routed-normalization collective contract."""
        width = int(norm_weight.shape[0]) if norm_weight.ndim == 1 else 0
        return (
            current_platform().is_cdna4
            and self.world_size in (4, 8)
            and self.all_reduce.dtype == torch.bfloat16
            and 0 < num_tokens
            and (
                num_tokens <= 16
                or (
                    num_tokens <= self.routed_lamport_max_rows
                    and topk == 8
                    and width == 4096
                    and weight_offset == 1.0
                    and self.reduction.lamport_region is not None
                    and self.reduction.lamport_epochs is not None
                    and num_tokens * width <= self.reduction.lamport_max_numel
                )
            )
            and topk > 0
            and routes.ndim == 2
            and routes.dtype == torch.bfloat16
            and routes.device == self.all_reduce.device
            and routes.stride(1) == 1
            and routes.shape[0] >= num_tokens * topk
            and routes.shape[1] >= width
            and route_map.shape == (num_tokens * topk,)
            and route_map.dtype == torch.int32
            and route_map.device == self.all_reduce.device
            and route_map.is_contiguous()
            and route_weights.shape == (num_tokens, topk)
            and route_weights.dtype == torch.float32
            and route_weights.device == self.all_reduce.device
            and route_weights.is_contiguous()
            and width > 0
            and width <= 65536
            and norm_weight.dtype == torch.bfloat16
            and norm_weight.device == self.all_reduce.device
            and norm_weight.is_contiguous()
            and self.all_reduce.producer.data is not None
            and self.gather.output is not None
            and self.all_reduce.producer_pull.ready_flags is not None
            and num_tokens * width <= self.all_reduce.producer_direct_max_numel
            and num_tokens <= self.all_reduce.producer_pull.ready_flags.shape[0]
        )

    def all_reduce_routed_rmsnorm_sum(
        self,
        routes: torch.Tensor,
        route_map: torch.Tensor,
        num_tokens: int,
        topk: int,
        route_weights: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        weight_offset: float,
    ) -> torch.Tensor:
        """Normalize and sum local routes while reducing the result across TP4/TP8."""
        if not self.can_all_reduce_routed_rmsnorm_sum(
            routes,
            route_map,
            num_tokens,
            topk,
            route_weights,
            norm_weight,
            weight_offset,
        ):
            raise RuntimeError("unsupported Iris routed-normalization collective")
        if (
            topk == 8
            and norm_weight.shape[0] == 4096
            and weight_offset == 1.0
            and self.reduction.lamport_region is not None
            and self.reduction.lamport_epochs is not None
            and num_tokens * 4096 <= self.reduction.lamport_max_numel
        ):
            assert self.gather.output is not None
            iris_fused_lamport_allreduce_routed_rmsnorm_sum[
                (num_tokens * (4096 // 512),)
            ](
                routes,
                route_map,
                route_weights,
                norm_weight,
                self.gather.output,
                self.reduction.lamport_region,
                self.reduction.lamport_epochs,
                self.reduction.lamport_peer_addresses[: self.world_size],
                RANK=self.all_reduce.rank_in_group,
                WIDTH=4096,
                row_stride=routes.stride(0),
                EPS=eps,
                MAX_ELEMENTS=self.reduction.lamport_max_numel,
                PORTABLE_MATH=num_tokens > 16,
                PEER_SIGNAL_STORE_CACHE=self._arch.peer_signal_store,
                num_warps=1,
                WAIT_STORES=self._arch.wait_stores,
            )
            return self.gather.output[: num_tokens * 4096].view(num_tokens, 4096)
        assert self.all_reduce.producer.data is not None
        assert self.gather.output is not None
        assert self.all_reduce.producer_pull.ready_flags is not None
        width = int(norm_weight.shape[0])
        block = triton.next_power_of_2(width)
        num_warps = 8
        iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum[(num_tokens,)](
            routes,
            route_map,
            route_weights,
            norm_weight,
            self.all_reduce.producer.data,
            self.gather.output,
            self.all_reduce.producer_pull.ready_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            rank=self.all_reduce.rank_in_group,
            width=width,
            row_stride=routes.stride(0),
            topk=topk,
            eps=eps,
            weight_offset=weight_offset,
            block=block,
            num_warps=num_warps,
        )
        return self.gather.output[: num_tokens * width].view(num_tokens, width)

    def can_all_gather_rmsnorm_mxfp4_quant(
        self,
        local: torch.Tensor,
        weight: torch.Tensor,
    ) -> bool:
        """Check the TP4/TP8 gather-and-quantize contract."""
        if (
            not current_platform().is_cdna4
            or self.world_size not in (4, 8)
            or local.ndim != 2
            or local.dtype != torch.bfloat16
            or self.all_reduce.dtype != torch.bfloat16
            or not local.is_contiguous()
            or weight.shape != (local.shape[1] * self.all_reduce.world_size,)
            or weight.dtype != local.dtype
            or weight.device != local.device
            or not weight.is_contiguous()
            or self.all_reduce.producer.data is None
            or self.gather.ready_flags is None
        ):
            return False
        return local.shape[0] <= self.gather.ready_flags.shape[
            0
        ] and all_gather_columns_can_run(
            world_size=self.all_reduce.world_size,
            num_tokens=local.shape[0],
            local_hidden_size=local.shape[1],
            dtype=local.dtype,
            max_bytes=self.all_reduce.producer_direct_max_numel * local.element_size(),
        )

    def all_gather_rmsnorm_mxfp4_quant(
        self,
        local: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        weight_offset: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather feature shards and return normalized MXFP4 rows."""
        if not self.can_all_gather_rmsnorm_mxfp4_quant(local, weight):
            raise RuntimeError("unsupported Iris gather-and-MXFP4 fusion")
        assert self.all_reduce.producer.data is not None
        assert self.gather.ready_flags is not None
        rows, local_width = local.shape
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
            empty_swizzled_cdna4_mxfp4_scale,
        )

        total_width = local_width * self.world_size
        packed = torch.empty(
            (rows, total_width // 2),
            dtype=torch.uint8,
            device=self.all_reduce.producer.data.device,
        )
        scales = empty_swizzled_cdna4_mxfp4_scale(
            rows,
            total_width // 32,
            device=self.all_reduce.producer.data.device,
        )
        block_k = triton.next_power_of_2(total_width)
        if (
            self.reduction.lamport_region is not None
            and self.reduction.lamport_epochs is not None
            and local_width % 512 == 0
            and block_k == total_width
            and total_width % 1024 == 0
            and rows * local_width <= self.reduction.lamport_max_numel
        ):
            if local.data_ptr() % 4:
                # Rank-local fallback could split ranks across protocols.
                raise ValueError(
                    "iris_fused_lamport_allgather_rmsnorm_mxfp4 reads BF16 pairs as "
                    f"32-bit words; local starts at byte {local.data_ptr() % 4} of a word"
                )
            # Sentinel mailboxes replace both peer rendezvous of the pull path.
            iris_fused_lamport_allgather_rmsnorm_mxfp4[(rows,)](
                local,
                weight,
                packed,
                scales,
                self.reduction.lamport_region,
                self.reduction.lamport_epochs,
                self.reduction.lamport_peer_addresses[: self.world_size],
                rank=self.all_reduce.rank_in_group,
                local_width=local_width,
                total_width=total_width,
                eps=eps,
                weight_offset=weight_offset,
                scale_stride_linear=scales.stride(0),
                scale_stride_mblock=scales.stride(1),
                max_elements=self.reduction.lamport_max_numel,
                PEER_SIGNAL_STORE_CACHE=self._arch.peer_signal_store,
                WAIT_STORES=self._arch.wait_stores,
                num_warps=1,
            )
            return packed, scales
        iris_fused_pull_allgather_rmsnorm_mxfp4[(rows,)](
            local,
            self.all_reduce.producer.data,
            weight,
            packed,
            scales,
            self.gather.ready_flags,
            self.all_reduce.peers.addresses[: self.world_size],
            rank=self.all_reduce.rank_in_group,
            local_width=local_width,
            total_width=total_width,
            eps=eps,
            weight_offset=weight_offset,
            scale_stride_linear=scales.stride(0),
            scale_stride_mblock=scales.stride(1),
            block_k=block_k,
            input_published=(
                rows == 1
                and local.data_ptr()
                == self.all_reduce.producer.data.data_ptr()
                + self.all_reduce.rank_in_group * local_width * local.element_size()
            ),
            num_warps=8,
        )
        return packed, scales


def prepare_muse_communication(
    *,
    group: dist.ProcessGroup,
    rank_in_group: int,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    enable_lamport: bool,
    owned_route_max_rows: int,
    all_reduce: IrisAllReduce | None,
    device: torch.device,
) -> MuseIrisState:
    """Prepare one group's generic operations and Muse fusions before capture.

    Capacities count elements. An existing generic all-reduce can be reused only
    if its fixed capacities cover the request. The first preparation sizes the
    shared heap; Iris's allocator enforces later allocations.
    """
    if all_reduce is not None:
        if (
            all_reduce.staged_max_numel < staged_max_numel
            or all_reduce.producer_direct_max_numel < producer_direct_max_numel
        ):
            raise RuntimeError("Iris all-reduce was prepared below Muse's capacities")
        if (
            all_reduce.group is not group
            or all_reduce.device != device
            or all_reduce.dtype != torch.bfloat16
        ):
            raise ValueError("Muse requires the prepared group's BF16 all-reduce")
        capacity = all_reduce.producer_direct_max_numel
    else:
        capacity = producer_direct_max_numel
    size = MuseIrisState.heap_bytes(
        capacity,
        world_size=group.size(),
        enable_lamport=enable_lamport,
        owned_route_max_rows=owned_route_max_rows,
    )
    if all_reduce is None:
        size += IrisAllReduce.heap_bytes(
            staged_max_numel, capacity, group.size(), torch.bfloat16
        )
        ctx = get_or_create_context(
            heap_size=max(MIN_HEAP_BYTES, size + HEAP_HEADROOM_BYTES)
        )
        peers = IrisGroup(ctx, group, rank_in_group, device)
        all_reduce = IrisAllReduce(peers, staged_max_numel, capacity, torch.bfloat16)
    state = MuseIrisState(
        all_reduce,
        enable_lamport=enable_lamport,
        owned_route_max_rows=owned_route_max_rows,
    )
    all_reduce.peers.finish_prepare()
    return state
