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

"""Shared collectives and scratch for data-parallel projection token owners.

Linear modules own parameters and GEMM dispatch. These non-module objects own
only communication resources; sequential layers may share them across capture
and replay without adding buffers to every layer or to the state dict.
"""

import logging
import socket

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.cuda import (
    TokenSpeedA2ALamportState,
    tokenspeed_a2a_lamport,
)
from tokenspeed_kernel.ops.communication.triton import (
    ProjectionPeerState,
    triton_projection_reduce_scatter,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherQuantState,
    TrtllmAllGatherState,
    TrtllmReduceScatterState,
    trtllm_allgather,
    trtllm_reduce_scatter,
)
from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale

from tokenspeed.runtime.distributed.comm_ops import (
    all_gather_single,
    all_to_all_single,
    reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import DenseLayerMapping, Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)

logger = logging.getLogger(__name__)
A2A_LAMPORT_MAX_TOKENS = 512
PEER_MAX_TOKENS = 8192
LAMPORT_MAX_TOKENS = 128


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
    if a2a_value not in ("nccl", "tokenspeed_a2a_lamport"):
        raise ValueError(f"Invalid output projection A2A backend: {a2a_value}")
    if rs_value not in ("nccl", "triton_peer", "trtllm_lamport"):
        raise ValueError(
            "Output projection reduction backend must be nccl, triton_peer or trtllm_lamport"
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


def prepare_lamport_projection_a2a(group, max_tokens, channels, dtype, device):
    """Prepare TokenSpeed TP4 BF16 A2A, or select NCCL for other topologies.

    Shape-specific state owns packet/chunk rings and local output. Bound rows
    to keep startup memory predictable; larger calls use the existing NCCL path.
    Initialization failures on a supported topology remain fatal on every rank.
    """
    if group.size() != 4 or dtype != torch.bfloat16 or channels % 8:
        return None
    hosts = [None] * group.size()
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        return None
    state = TokenSpeedA2ALamportState(
        group,
        min(max_tokens, A2A_LAMPORT_MAX_TOKENS),
        channels,
        device,
        min(128, torch.cuda.get_device_properties(device).multi_processor_count),
    )
    if channels % 32 == 0:
        # Exactly 8 MiB retains packet exchange; larger messages avoid its
        # doubled link traffic using the vectorized chunk protocol.
        state.prepare_chunk_exchange(threshold_bytes=8 * 2**20 + 1)
    return state


def column_projection_width(output_size: int, tp_size: int, block_rows: int) -> int:
    """Pad output width so each contiguous TP shard contains whole scale blocks."""
    if min(output_size, tp_size, block_rows) <= 0:
        raise ValueError("Projection sizes and scale-block rows must be positive")
    alignment = tp_size * block_rows
    return (output_size + alignment - 1) // alignment * alignment


class DPRowParallelCommunication:
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
        self.lamport_a2a_states = {}
        self._a2a_initialized = False
        self.peer_states = {}
        self.lamport_states = {}
        self._rs_initialized = False

    def initialize_reduce_scatter(
        self, parallel: DenseLayerMapping, output_sizes: list[int], backend: str
    ) -> None:
        """Allocate bounded symmetric scratch per output width before capture.

        Unsupported group sizes/dtypes use NCCL. Initialization failures on a
        supported topology are fatal; never retry a failed collective with
        another backend inside forward. Fast paths may change BF16 summation order.
        """
        if self._rs_initialized:
            return
        if backend not in ("nccl", "triton_peer", "trtllm_lamport"):
            raise ValueError("Invalid projection reduction backend")
        if backend != "nccl" and (
            parallel.tp_size != 4 or self.send.dtype != torch.bfloat16
        ):
            logger.warning(
                f"Projection {backend} reduction requires TP4 BF16; using NCCL"
            )
            self._rs_initialized = True
            return
        if backend == "trtllm_lamport":
            group = pg_manager.get_process_group("nccl", parallel.tp_group)
            for width in sorted(set(output_sizes)):
                state = TrtllmReduceScatterState(
                    group,
                    min(self.max_tokens, LAMPORT_MAX_TOKENS),
                    width,
                    self.send.device,
                )
                self.lamport_states[width] = state
                probe = state.input_buffer(1)
                probe.zero_()
                trtllm_reduce_scatter(state, probe, 1)
            logger.info(
                "Projection ReduceScatter: trtllm_lamport, up to "
                f"{min(self.max_tokens, LAMPORT_MAX_TOKENS)} rows/rank; NCCL above"
            )
        if backend == "triton_peer":
            group = pg_manager.get_process_group("nccl", parallel.tp_group)
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
                "Projection ReduceScatter: triton_peer, up to "
                f"{min(self.max_tokens, PEER_MAX_TOKENS)} rows/rank"
            )
        self._rs_initialized = True

    def peer_state(self, output_size: int, rows: int):
        """Select the same prepared fast path from shared physical counts."""
        if 0 < rows <= min(self.max_tokens, PEER_MAX_TOKENS):
            return self.peer_states.get(output_size)
        return None

    def lamport_state(self, output_size: int, rows: int):
        """Select one-shot IPC scratch using shared physical row counts."""
        if 0 < rows <= min(self.max_tokens, LAMPORT_MAX_TOKENS):
            return self.lamport_states.get(output_size)
        return None

    def reduce_scatter(
        self,
        partial: torch.Tensor,
        parallel: DenseLayerMapping,
        max_tokens: int,
    ) -> torch.Tensor:
        """Return owned output rows; never expose reusable communication storage."""
        lamport = self.lamport_state(partial.shape[1], max_tokens)
        if lamport is not None:
            return trtllm_reduce_scatter(lamport, partial, max_tokens)
        peer = self.peer_state(partial.shape[1], max_tokens)
        if peer is not None:
            return triton_projection_reduce_scatter(peer, partial, max_tokens)
        return reduce_scatter(partial, parallel.tp_group, backend=None)

    def initialize_a2a(
        self, parallel: DenseLayerMapping, input_sizes: list[int], backend: str
    ) -> None:
        """Collectively prepare optional IPC/JIT resources before graph capture.

        The communicator is shared by sequential layers and lives as long as
        this workspace. Never replace it while captured graphs reference it.
        """
        if self._a2a_initialized:
            return
        if backend not in ("nccl", "tokenspeed_a2a_lamport"):
            raise ValueError("Invalid projection A2A backend")
        if backend == "tokenspeed_a2a_lamport":
            group = pg_manager.get_process_group("nccl", parallel.tp_group)
            for width in sorted(set(input_sizes)):
                state = prepare_lamport_projection_a2a(
                    group, self.max_tokens, width, self.send.dtype, self.send.device
                )
                if state is not None:
                    if width % 512 == 0:
                        state.prepare_fp8_quantization()
                    self.lamport_a2a_states[width] = state
            logger.info(
                f"Projection A2A: tokenspeed_a2a_lamport for widths {sorted(self.lamport_a2a_states)}, "
                f"up to {min(self.max_tokens, A2A_LAMPORT_MAX_TOKENS)} rows/rank; "
                "NCCL for other shapes/topologies"
            )
        self._a2a_initialized = True

    def lamport_a2a_state(self, input_size: int, rows: int):
        """Select by subgroup physical capacity, including uneven/empty owners."""
        state = self.lamport_a2a_states.get(input_size)
        return state if state is not None and 0 < rows <= state.max_rows else None

    def close(self) -> None:
        """Collectively release IPC resources after all referencing graphs die."""
        if self.lamport_a2a_states:
            torch.cuda.synchronize(self.send.device)
            dist.barrier(group=next(iter(self.lamport_a2a_states.values())).group)
        self.lamport_a2a_states.clear()
        for state in self.lamport_states.values():
            state.close()
        self.lamport_states.clear()
        if self.peer_states:
            torch.cuda.synchronize(self.send.device)
            dist.barrier(group=next(iter(self.peer_states.values())).group)
        self.peer_states.clear()


class DPColumnParallelCommunication:
    """Shared AllGather/A2A buffers for sequential DP column-parallel layers.

    Args:
        parallel: Prepared projection-only subgroup mapping.
        input_size: Complete input width K on every token owner.
        padded_output_size: Stored width N, divisible by TP and scale alignment.
        max_tokens: Prepared physical row capacity per rank.
        dtype: BF16 activation/output storage type.
        device: Current rank's selected CUDA device.
        allgather_backend: Explicit 'nccl' or 'trtllm'.
        a2a_backend: Explicit 'nccl' or 'tokenspeed_a2a_lamport'.

    Construct collectively before capture/cache budgeting. Calls and consumers
    must be serialized on one stream; close only after referencing graphs die.
    """

    def __init__(
        self,
        parallel: DenseLayerMapping,
        input_size: int,
        padded_output_size: int,
        max_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        allgather_backend: str,
        a2a_backend: str,
    ):
        if allgather_backend not in ("nccl", "trtllm"):
            raise ValueError("Invalid column projection all-gather backend")
        if a2a_backend not in ("nccl", "tokenspeed_a2a_lamport"):
            raise ValueError("Invalid column projection A2A backend")
        if (
            parallel.tp_size not in (2, 4, 8, 16)
            or min(input_size, padded_output_size, max_tokens) <= 0
            or padded_output_size % parallel.tp_size
            or dtype != torch.bfloat16
        ):
            raise ValueError("Invalid BF16 column projection dimensions")
        self.parallel = parallel
        self.input_size = input_size
        self.padded_output_size = padded_output_size
        self.max_tokens = max_tokens
        self.closed = False
        self.send = torch.empty((max_tokens, input_size), dtype=dtype, device=device)
        self.gathered = torch.empty(
            (parallel.tp_size * max_tokens, input_size), dtype=dtype, device=device
        )
        self.received = torch.empty(
            max_tokens * padded_output_size, dtype=dtype, device=device
        )
        group = pg_manager.get_process_group("nccl", parallel.tp_group)
        self.gather_state = None
        if allgather_backend == "trtllm":
            # Fused gather/quantization has TP2/TP4 numerical coverage. Its
            # inherited BF16 gather shares the IPC ring for other linear plans;
            # larger groups retain the ordinary gather until validated.
            if parallel.tp_size in (2, 4) and has_flashinfer_fp8_blockscale():
                self.gather_state = TrtllmAllGatherQuantState(
                    group,
                    min(max_tokens, 128),
                    input_size,
                    device,
                    torch.cuda.get_device_properties(device).multi_processor_count,
                )
            else:
                self.gather_state = TrtllmAllGatherState(
                    group, min(max_tokens, 128), input_size, device, True
                )
        self.lamport_a2a = None
        if a2a_backend == "tokenspeed_a2a_lamport":
            self.lamport_a2a = prepare_lamport_projection_a2a(
                group, max_tokens, padded_output_size, dtype, device
            )

    def padded_inputs(self, inputs: torch.Tensor, rows: int) -> torch.Tensor:
        """Use identical owner padding for BF16 and fused-quantized gathers."""
        if inputs.shape[0] == rows and inputs.is_contiguous():
            return inputs
        send = self.send[:rows]
        send.zero_()
        send[: inputs.shape[0]].copy_(inputs)
        return send

    def gather_inputs(self, inputs: torch.Tensor, rows: int) -> torch.Tensor:
        """Gather equal padded rank-major token blocks; return borrowed storage."""
        send = self.padded_inputs(inputs, rows)
        if self.gather_state is not None and rows <= self.gather_state.max_rows:
            return trtllm_allgather(self.gather_state, send)
        gathered = self.gathered[: self.parallel.tp_size * rows]
        all_gather_single(gathered, send, self.parallel.tp_group, backend=None)
        return gathered

    def restore_outputs(self, local: torch.Tensor, rows: int) -> torch.Tensor:
        """Exchange output channel shards and return owned padded [rows,N]."""
        size = self.parallel.tp_size
        shard = self.padded_output_size // size
        if self.lamport_a2a is not None and rows <= self.lamport_a2a.max_rows:
            # Write directly into owned storage: later layers may reuse the
            # communicator while a caller still retains this projection result.
            output = local.new_empty((rows, self.padded_output_size))
            return tokenspeed_a2a_lamport(
                self.lamport_a2a, local, inverse=True, out=output
            )
        received = self.received[: rows * self.padded_output_size].view(
            size * rows, shard
        )
        all_to_all_single(received, local, self.parallel.tp_group, backend=None)
        # Received messages are source-channel-rank major. The copy restores
        # token-major channel order and gives the caller independent storage.
        return (
            received.view(size, rows, shard)
            .transpose(0, 1)
            .clone(memory_format=torch.contiguous_format)
            .view(rows, self.padded_output_size)
        )

    def close(self) -> None:
        """Collectively close communicators after all referencing graphs die."""
        if self.closed:
            return
        torch.cuda.synchronize(self.send.device)
        if self.lamport_a2a is not None:
            dist.barrier(group=self.lamport_a2a.group)
            self.lamport_a2a = None
        if self.gather_state is not None:
            self.gather_state.close()
        self.closed = True
