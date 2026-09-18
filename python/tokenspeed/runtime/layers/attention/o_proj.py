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

"""Redistribute locally owned full-channel tokens for row-parallel projection.

Every subgroup rank participates, even with zero local tokens. Outputs return
to their original owners before residual/normalization processing. This wrapper
uses RowParallelLinear for GEMM; its own reduction must not be duplicated by
that Linear. Workspaces are shared only by serialized calls on one stream and
must outlive all captured graphs that reference them.
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.flashinfer import (
    create_projection_a2a,
    flashinfer_projection_a2a,
    flashinfer_projection_a2a_borrowed,
    prepare_borrowed_projection_a2a,
)
from tokenspeed_kernel.ops.communication.triton import (
    triton_pack_channel_shards_for_a2a,
)
from tokenspeed_kernel.ops.communication.triton_projection import (
    ProjectionPeerState,
    triton_projection_reduce_scatter,
    triton_projection_reduce_scatter_after_a2a,
)

from tokenspeed.runtime.distributed.comm_ops import all_to_all_single, reduce_scatter
from tokenspeed.runtime.distributed.mapping import (
    AttentionLayerMapping,
    DenseLayerMapping,
    LinearAttnLayerMapping,
    Mapping,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.linear import RowParallelLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext

logger = logging.getLogger(__name__)
FLASHINFER_MAX_TOKENS = 512
PEER_MAX_TOKENS = 8192


def projection_mapping(rank: int, world_size: int, tp_size: int) -> DenseLayerMapping:
    """Return a contiguous projection subgroup without changing model mappings."""
    if tp_size < 1 or world_size % tp_size:
        raise ValueError("Projection TP size must be a positive divisor of world size")
    return DenseLayerMapping(
        rank=rank, world_size=world_size, tp_size=tp_size, dp_size=world_size // tp_size
    )


def validate_projection_settings(
    mapping: Mapping, value: str, a2a_value: str, rs_value: str
) -> DenseLayerMapping:
    """Validate explicit TP/backend settings across ranks and return the mapping.

    Args:
        mapping: Model mapping supplying the world group and rank.
        value: Requested projection TP size, kept as a string until ranks agree.
        a2a_value: Selected A2A backend.
        rs_value: Selected reduction backend.

    Returns:
        The contiguous projection mapping. The model must validate its topology
        before calling initialize_projection_group; this function creates no
        projection subgroup and reads no environment variables.
    """
    if dist.is_initialized() and mapping.world_size > 1:
        # Every rank participates, even when its local setting is disabled or
        # malformed. Reject disagreement before entering differently sized groups.
        pg_manager.init_process_group(mapping.world_group, backend="gloo")
        values = [None] * mapping.world_size
        dist.all_gather_object(
            values,
            (value, a2a_value, rs_value),
            group=pg_manager.get_process_group("gloo", mapping.world_group),
        )
        if len(set(values)) != 1:
            raise ValueError(
                f"Projection TP/A2A/RS settings differ across ranks: {values}"
            )
    if a2a_value not in ("nccl", "auto", "flashinfer"):
        raise ValueError(f"Invalid output projection A2A backend: {a2a_value}")
    if rs_value not in ("nccl", "triton_peer"):
        raise ValueError(
            "Output projection reduction backend must be nccl or triton_peer"
        )
    try:
        size = int(value)
    except ValueError as exc:
        raise ValueError("Projection TP size must be a positive integer") from exc
    return projection_mapping(mapping.rank, mapping.world_size, size)


def initialize_projection_group(parallel: DenseLayerMapping) -> None:
    """Materialize subgroup collectives before weight loading or graph capture.

    Callers must agree on the mapping and backend settings across ranks first.
    """
    if parallel.tp_size > 1:
        pg_manager.init_process_group(parallel.tp_group, backend=None)
        probe = torch.zeros((parallel.tp_size, 1), dtype=torch.bfloat16, device="cuda")
        received = torch.empty_like(probe)
        all_to_all_single(received, probe, parallel.tp_group, backend=None)
        reduce_scatter(received, parallel.tp_group, backend=None)


class ProjectionWorkspace:
    """Reusable exchange scratch, shared by sequential attention layers."""

    def __init__(
        self,
        max_tokens: int,
        max_input_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # All-to-all exchanges P * M * (K/P) elements: M*K, not P*M*K.
        self.max_tokens = max_tokens
        self.send = torch.empty(max_tokens * max_input_size, dtype=dtype, device=device)
        self.recv = torch.empty_like(self.send)
        self.a2a = None
        self.a2a_reason = None
        self._a2a_initialized = False
        self.peer_states = {}
        self.borrowed_a2a = None
        self._rs_initialized = False

    def initialize_reduce_scatter(
        self, parallel: DenseLayerMapping, output_sizes: list[int], backend: str
    ) -> None:
        """Allocate bounded symmetric scratch per output width before capture.

        Explicit opt-in accepts a different BF16 reduction order. Initialization
        failures are fatal; never retry a failed collective with
        another backend inside forward.
        """
        if self._rs_initialized:
            return
        if backend not in ("nccl", "triton_peer"):
            raise ValueError("Projection reduction backend must be nccl or triton_peer")
        if backend == "triton_peer":
            if parallel.tp_size != 4:
                logger.warning("Projection peer reduction requires TP4; using NCCL")
                self._rs_initialized = True
                return
            if self.send.dtype != torch.bfloat16 or any(
                width <= 0 for width in output_sizes
            ):
                raise ValueError(
                    "Projection peer reduction requires BF16 and positive widths"
                )
            group = pg_manager.get_process_group("nccl", parallel.tp_group)
            if self.a2a is not None:
                self.borrowed_a2a = prepare_borrowed_projection_a2a(self.a2a, group)
            for width in sorted(set(output_sizes)):
                state = ProjectionPeerState(
                    group,
                    min(self.max_tokens, PEER_MAX_TOKENS),
                    width,
                    self.send.device,
                )
                probe = state.input_buffer(1)
                probe.zero_()
                triton_projection_reduce_scatter(state, probe, 1)
                self.peer_states[width] = state
            logger.info(
                "Projection ReduceScatter: triton_peer, up to %s rows/rank",
                min(self.max_tokens, PEER_MAX_TOKENS),
            )
        self._rs_initialized = True

    def peer_state(self, output_size: int, rows: int):
        """Select the same prepared fast path from shared physical counts."""
        if 0 < rows <= min(self.max_tokens, PEER_MAX_TOKENS):
            return self.peer_states.get(output_size)
        return None

    def reduce_scatter(
        self,
        partial: torch.Tensor,
        parallel: DenseLayerMapping,
        max_tokens: int,
    ) -> torch.Tensor:
        """Return owned output rows; never expose reusable communication storage."""
        peer = self.peer_state(partial.shape[1], max_tokens)
        if peer is not None:
            return triton_projection_reduce_scatter(peer, partial, max_tokens)
        return reduce_scatter(partial, parallel.tp_group, backend=None)

    def initialize_a2a(
        self, parallel: DenseLayerMapping, max_input_size: int, backend: str
    ) -> None:
        """Collectively prepare optional IPC/JIT resources before graph capture.

        The communicator is shared by sequential layers and lives as long as
        this workspace. Never replace it while captured graphs reference it.
        """
        if self._a2a_initialized:
            return
        if backend not in ("nccl", "auto", "flashinfer"):
            raise ValueError("Invalid projection A2A backend")
        if backend != "nccl":
            self.a2a, self.a2a_reason = create_projection_a2a(
                group=pg_manager.get_process_group("nccl", parallel.tp_group),
                max_elems=min(self.max_tokens, FLASHINFER_MAX_TOKENS) * max_input_size,
                dtype=self.send.dtype,
                device=self.send.device,
                backend=backend,
            )
            logger.info(
                "Projection A2A: %s (%s)",
                "flashinfer" if self.a2a is not None else "nccl",
                self.a2a_reason or "NVLink",
            )
        self._a2a_initialized = True

    def use_flashinfer(
        self, parallel: DenseLayerMapping, counts: list[int], input_size: int
    ) -> bool:
        """Choose identically on every subgroup rank, using only host metadata."""
        rows = counts[parallel.rank]
        return (
            self.a2a is not None
            and 0 < rows <= FLASHINFER_MAX_TOKENS
            and input_size % (8 * parallel.tp_size) == 0
            and all(counts[r] == rows for r in parallel.tp_group)
        )

    def close(self) -> None:
        """Collectively release IPC resources after all referencing graphs die."""
        if self.peer_states:
            torch.cuda.synchronize(self.send.device)
            dist.barrier(group=next(iter(self.peer_states.values())).group)
        self.peer_states.clear()
        self.borrowed_a2a = None
        if self.a2a is not None:
            self.a2a.close()
            self.a2a = None


class DistributedOutputProjection:
    """Run a sharded projection and return outputs to the original DP owner.

    The row-parallel Linear remains registered as self_attn.o_proj, preserving
    checkpoint names and the existing packed-weight/scale shard loaders.
    """

    def __init__(self, parallel: DenseLayerMapping, input_size: int) -> None:
        self.parallel = parallel
        self.input_size = input_size
        self.workspace: ProjectionWorkspace | None = None

    def forward(
        self, inputs: torch.Tensor, linear: RowParallelLinear, counts: list[int]
    ) -> torch.Tensor:
        """Project local [tokens, channels] rows using subgroup token counts.

        Returns complete [local_tokens, hidden] rows in the original order.
        All subgroup ranks must call, including ranks with no local tokens.
        """
        parallel = self.parallel
        if (
            len(counts) != parallel.world_size
            or counts[parallel.rank] != inputs.shape[0]
            or any(count < 0 for count in counts)
            or inputs.ndim != 2
            or inputs.shape[1] != self.input_size
        ):
            raise ValueError(
                "Output projection requires matching collective token counts"
            )
        max_tokens = max(counts[r] for r in parallel.tp_group)
        if max_tokens == 0:
            return inputs.new_empty((0, linear.output_size))
        workspace = self.workspace
        if workspace is None or max_tokens > workspace.max_tokens:
            raise RuntimeError(
                "Output projection workspace must be prepared before forward"
            )
        if (
            inputs.dtype != workspace.send.dtype
            or inputs.device != workspace.send.device
        ):
            raise ValueError(
                "Output projection inputs must match the prepared workspace"
            )
        size = parallel.tp_size
        shard = self.input_size // size
        elements = max_tokens * self.input_size
        send = workspace.send[:elements].view(size, max_tokens, shard)
        recv = workspace.recv[:elements].view(size * max_tokens, shard)
        # Equal-sized messages permit capture and avoid device-to-host counts.
        # Rank-major output segments are exactly reduce-scatter's owner ordering.
        # One row already has rank-major byte order. Other shapes fuse the
        # transpose and padding, preserving the original BF16/FP16 values.
        fused_a2a = workspace.use_flashinfer(parallel, counts, self.input_size)
        peer = workspace.peer_state(linear.output_size, max_tokens)
        if peer is not None and fused_a2a:
            recv = flashinfer_projection_a2a_borrowed(
                workspace.borrowed_a2a, inputs.contiguous()
            )
        elif fused_a2a:
            recv = flashinfer_projection_a2a(workspace.a2a, inputs.contiguous())
        else:
            packed = triton_pack_channel_shards_for_a2a(inputs, send)
            all_to_all_single(recv, packed, parallel.tp_group, backend=None)
        if peer is not None:
            if not fused_a2a:
                # A preceding FI reduction may have deferred its reuse fence.
                # NCCL A2A is not that fence: protect direct GEMM writes here,
                # including transitions through empty or NCCL-only batches.
                peer.handle.barrier(channel=1)
            partial, _ = linear.forward_into(recv, None, peer.input_buffer(max_tokens))
            if fused_a2a:
                # Next FI A2A or the NCCL path's pre-write fence protects reuse.
                output = triton_projection_reduce_scatter_after_a2a(
                    peer, partial, max_tokens
                )
            else:
                output = triton_projection_reduce_scatter(peer, partial, max_tokens)
        else:
            partial, _ = linear(recv)
            output = workspace.reduce_scatter(
                partial.contiguous(), parallel, max_tokens
            )
        return output[: inputs.shape[0]]


def make_output_projection(
    *,
    parallel: DenseLayerMapping | None,
    input_size: int,
    output_size: int,
    quant_config: QuantizationConfig | None,
    prefix: str,
    default_parallel: AttentionLayerMapping | LinearAttnLayerMapping,
    reduce_results: bool,
) -> tuple[RowParallelLinear, DistributedOutputProjection | None]:
    """Construct projection shards and optional DP/TP exchange, without full weights."""
    enabled = parallel is not None and parallel.tp_size > 1
    selected = parallel if enabled else default_parallel
    if input_size % selected.tp_size:
        raise ValueError("Output projection channels must divide projection TP size")
    linear = RowParallelLinear(
        input_size,
        output_size,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
        params_dtype=None,
        reduce_results=False if enabled else reduce_results,
        quant_config=quant_config,
        prefix=prefix,
        tp_rank=selected.tp_rank,
        tp_size=selected.tp_size,
        tp_group=selected.tp_group,
        use_presharded_weights=False,
        override_kernel_name=None,
        interleave_linear_and_gate=False,
    )
    if enabled:
        # Mixed checkpoints resolve quantization per Linear. Inspect that
        # resolved method, not the model-level default (e.g. MXFP8 vs FP8).
        resolved = getattr(linear.quant_method, "quant_config", None)
        block = getattr(resolved, "weight_block_size", None)
        alignment = block[-1] if block else getattr(resolved, "group_size", 1)
        if linear.input_size_per_partition % alignment:
            raise ValueError("Output projection TP shard splits a quantization block")
    return linear, (
        DistributedOutputProjection(parallel, input_size) if enabled else None
    )


def project_attention_output(
    inputs: torch.Tensor,
    linear: RowParallelLinear,
    exchange: DistributedOutputProjection | None,
    ctx: "ForwardContext",
) -> torch.Tensor:
    """Apply the ordinary projection or its projection-only TP exchange."""
    if exchange is None:
        output, _ = linear(inputs)
        return output
    counts = ctx.collective_global_num_tokens
    if counts is None:
        counts = ctx.global_num_tokens
    if counts is None:
        raise ValueError("Output projection TP requires collective token counts")
    return exchange.forward(inputs, linear, counts)
