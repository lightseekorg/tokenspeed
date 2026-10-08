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

"""Projection collective dispatch with separately owned persistent workspaces."""

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Group
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils.tensor import prepare_padded_rows

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.comm_backend.base import CommBackend
    from tokenspeed.runtime.execution.workspace import WorkspacePool


@dataclass(frozen=True)
class ProjectionSpec:
    """Static layout and allocation bounds, identical on every subgroup peer.

    Sequential layers may share a prepared object with the same spec. Separate
    streams or concurrently executing models must prepare separate objects.
    Sizes are full (unsharded) widths; max_tokens is physical rows per owner.
    """

    group: Group
    kind: Literal["column", "row"]
    input_size: int
    output_size: int
    max_tokens: int
    dtype: torch.dtype
    device: torch.device

    @property
    def scratch_specs(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        """Simultaneously live send, receive and GEMM buffers for this layout."""
        width = self.input_size if self.kind == "row" else self.output_size
        gemm_width = self.input_size if self.kind == "column" else self.output_size
        return (
            ((self.max_tokens * self.input_size,), self.dtype),
            ((self.max_tokens * width,), self.dtype),
            ((len(self.group) * self.max_tokens, gemm_width), self.dtype),
        )


class ProjectionWorkspace:
    """Own scratch tensors and native resources for one prepared projection.

    The backend installs eligible native resources during preparation. This
    object keeps their storage alive until collective teardown; dispatch and
    layout conversion belong to the backend. Sequential layers may share
    generic scratch across shapes while retaining separate native resources.
    Intermediate tensors borrow storage only for the current projection;
    consumers finish on the same stream before another projection reuses it.
    Independent preparations own separate allocations. Final inverse-A2A/RS
    outputs are caller-owned. Destroy referencing CUDA graphs before close().
    """

    def __init__(
        self,
        spec: ProjectionSpec,
        backend: "CommBackend",
        scratch_pool: "WorkspacePool | None" = None,
    ):
        if (
            spec.kind not in ("column", "row")
            or len(spec.group) < 2
            or min(spec.input_size, spec.output_size, spec.max_tokens) <= 0
        ):
            raise ValueError("Invalid prepared projection dimensions")
        width = spec.input_size if spec.kind == "row" else spec.output_size
        if width % len(spec.group):
            raise ValueError("Projection channel width must be divisible by TP")
        self.spec = spec
        self.backend = backend
        self.closed = False
        if scratch_pool is None:
            # env -> server_args -> comm_ops imports this module before envs
            # exists. Load the env-dependent allocator only at preparation.
            from tokenspeed.runtime.execution.workspace import WorkspacePool

            scratch_pool = WorkspacePool(spec.device, initial_nbytes=0)
            scratch_pool.allocate(*spec.scratch_specs)
            scratch_pool.freeze()
        if scratch_pool.device != spec.device or not scratch_pool.frozen:
            raise ValueError("Projection scratch must be frozen on the same device")
        # Projection retains these views until teardown. Keep their addresses
        # independent of the executor-managed global pool, which can be unfrozen
        # and grown during reconfiguration.
        self.scratch_pool: "WorkspacePool | None" = scratch_pool
        self.send, self.received, gemm = scratch_pool.allocate(*spec.scratch_specs)
        self.gathered = gemm if spec.kind == "column" else None
        self.partial = gemm if spec.kind == "row" else None
        self.gather = None
        self.gather_quant = None
        self.a2a = None
        self.reduction = None

    def close(self) -> None:
        """Collectively release native resources and owned scratch references.

        Call on every subgroup rank after consumers finish and referencing CUDA
        graphs are destroyed, before process-group teardown. Repeated calls are
        harmless. Borrowed tensor views must also be dropped to reclaim storage;
        caller-owned final outputs remain valid.
        """
        if self.closed:
            return
        if self.spec.device.type == "cuda":
            torch.cuda.synchronize(self.spec.device)
        dist.barrier(group=pg_manager.get_device_process_group(self.spec.group))
        self.a2a = None
        if self.gather is not None:
            self.gather.close()
            self.gather = self.gather_quant = None
        if self.reduction is not None:
            self.reduction.close()
            self.reduction = None
        self.send = self.received = self.gathered = self.partial = None
        self.scratch_pool = None
        self.closed = True


def prepare_projection_workspace(
    spec: ProjectionSpec,
    backend: "CommBackend",
    scratch_pool: "WorkspacePool | None",
) -> ProjectionWorkspace:
    """Allocate and warm model-owned scratch using backend's ordinary collectives."""
    workspace = ProjectionWorkspace(spec, backend, scratch_pool)
    _warmup_projection(workspace)
    return workspace


def _warmup_projection(workspace: ProjectionWorkspace) -> None:
    spec = workspace.spec
    backend = workspace.backend
    # Exercise the selected paths before capture, including optional fusion.
    # Generic backends use ordinary collectives; Auto may use native resources.
    probe = workspace.send[: spec.input_size].view(1, spec.input_size)
    probe.zero_()
    if spec.kind == "column":
        backend.projection_all_gather(probe, 1, False, workspace)
        backend.projection_all_gather(probe, 1, True, workspace)
        local = probe.new_zeros(len(spec.group), spec.output_size // len(spec.group))
        backend.projection_all_to_all(
            local, 1, True, False, probe.new_empty(1, spec.output_size), workspace
        )
    else:
        backend.projection_all_to_all(probe, 1, False, False, None, workspace)
        backend.projection_all_to_all(probe, 1, False, True, None, workspace)
        partial = backend.acquire_projection_output(1, workspace)
        partial.zero_()
        backend.projection_reduce_scatter(partial, 1, workspace)
    # Warm fallbacks even when all probes selected a small-message kernel.
    small = probe.new_zeros(len(spec.group), 1)
    backend.all_to_all_single(
        torch.empty_like(small),
        small,
        spec.group,
        output_split_sizes=None,
        input_split_sizes=None,
    )
    backend.all_gather_single(small, probe.new_zeros(1, 1), spec.group)
    backend.reduce_scatter(small, spec.group)


def _check_workspace(
    rows: int, workspace: ProjectionWorkspace, backend: "CommBackend"
) -> None:
    if workspace.backend is not backend:
        raise ValueError("Projection workspace belongs to another backend")
    if workspace.closed or not 0 < rows <= workspace.spec.max_tokens:
        raise RuntimeError("Projection communication is closed or exceeds capacity")


def _check_a2a_options(inverse: bool, quantize: bool, out: torch.Tensor | None) -> None:
    if inverse and (quantize or out is None):
        raise ValueError("Inverse A2A requires owned output and no quantization")
    if not inverse and out is not None:
        # Forward can return BF16 or FP8 borrowed scratch, not an owned output.
        raise ValueError("Forward A2A returns borrowed output; out must be None")


def projection_all_gather(
    inputs: torch.Tensor,
    rows: int,
    quantize: bool,
    workspace: ProjectionWorkspace,
    backend: "CommBackend",
) -> tuple[torch.Tensor, None]:
    """Gather padded owner rows through backend into borrowed activation scratch.

    quantize permits fusion; this generic path returns unquantized values and
    None for scales so the Linear uses its ordinary quantization/GEMM path.
    """
    _check_workspace(rows, workspace, backend)
    send = prepare_padded_rows(inputs, rows, workspace.send, alignment_bytes=16)
    gathered = workspace.gathered[: len(workspace.spec.group) * rows]
    backend.all_gather_single(gathered, send, workspace.spec.group)
    return gathered, None


def projection_all_to_all(
    inputs: torch.Tensor,
    rows: int,
    inverse: bool,
    quantize: bool,
    out: torch.Tensor | None,
    workspace: ProjectionWorkspace,
    backend: "CommBackend",
) -> tuple[torch.Tensor, None]:
    """Exchange token/channel axes using backend's ordinary AllToAll.

    Forward returns borrowed padded channel shards without fused quantization.
    Inverse restores complete owner rows into the caller-provided out tensor.
    """
    from tokenspeed_kernel.ops.communication.triton import (
        triton_pack_channel_shards_for_a2a,
    )

    _check_workspace(rows, workspace, backend)
    _check_a2a_options(inverse, quantize, out)
    spec = workspace.spec
    size = len(spec.group)
    width = spec.output_size if inverse else spec.input_size
    shard = width // size
    received = workspace.received[: rows * width].view(size * rows, shard)
    if inverse:
        sent = inputs.contiguous()
    else:
        scratch = workspace.send[: rows * width].view(size, rows, shard)
        sent = triton_pack_channel_shards_for_a2a(inputs, scratch)
    backend.all_to_all_single(
        received,
        sent,
        spec.group,
        output_split_sizes=None,
        input_split_sizes=None,
    )
    if inverse:
        out.view(rows, size, shard).copy_(
            received.view(size, rows, shard).transpose(0, 1)
        )
        return out, None
    return received, None


def acquire_projection_output(
    rows: int, workspace: ProjectionWorkspace, backend: "CommBackend"
) -> torch.Tensor:
    """Borrow [TP*rows,N] ordinary GEMM scratch for a following ReduceScatter."""
    _check_workspace(rows, workspace, backend)
    return workspace.partial[: len(workspace.spec.group) * rows]


def projection_reduce_scatter(
    partial: torch.Tensor,
    rows: int,
    workspace: ProjectionWorkspace,
    backend: "CommBackend",
) -> torch.Tensor:
    """Reduce [TP*rows,N] partials through backend into owned [rows,N] outputs."""
    _check_workspace(rows, workspace, backend)
    return backend.reduce_scatter(partial, workspace.spec.group)


class ProjectionBackend:
    """Dispatch prepared projection operations through native or fallback kernels.

    AutoBackend composes this dispatcher, like TritonRSAGBackend. Generic
    fallbacks above reuse its ordinary collectives and numerics. Workspaces are
    passed explicitly, so one dispatcher can serve independent models/streams.
    Physical subgroup rows select the same path on every peer, including empty
    owners. All workspace allocation and warmup precedes capture.
    """

    def __init__(self, fallback: "CommBackend"):
        self._fallback = fallback

    def prepare(
        self,
        spec: ProjectionSpec,
        use_lamport: bool,
        use_lamport_reduction: bool,
        scratch_pool: "WorkspacePool | None",
    ) -> ProjectionWorkspace:
        """Allocate and warm a workspace bound to this dispatcher's backend."""
        workspace = ProjectionWorkspace(spec, self._fallback, scratch_pool)
        if use_lamport:
            # CUDA IPC requires a node-local group. Agree on topology before
            # any rank attempts collective native allocation.
            hosts = [None] * len(spec.group)
            dist.all_gather_object(
                hosts,
                socket.gethostname(),
                group=pg_manager.get_process_group("gloo", spec.group),
            )
            if len(set(hosts)) == 1:
                self._prepare_lamport(workspace, use_lamport_reduction)

        _warmup_projection(workspace)
        return workspace

    def _prepare_lamport(
        self, workspace: ProjectionWorkspace, use_reduction: bool
    ) -> None:
        from tokenspeed_kernel.ops.communication.cuda import TokenSpeedA2ALamportState
        from tokenspeed_kernel.ops.communication.trtllm import (
            TrtllmAllGatherQuantState,
            TrtllmAllGatherState,
            TrtllmReduceScatterState,
        )
        from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale

        spec = workspace.spec
        size = len(spec.group)
        group = pg_manager.get_device_process_group(spec.group)
        sms = torch.cuda.get_device_properties(spec.device).multi_processor_count
        if (
            spec.kind == "column"
            and size in (2, 4, 8, 16)
            and spec.input_size % 128 == 0
        ):
            if size in (2, 4) and has_flashinfer_fp8_blockscale():
                workspace.gather_quant = TrtllmAllGatherQuantState(
                    group, min(spec.max_tokens, 128), spec.input_size, spec.device, sms
                )
                workspace.gather = workspace.gather_quant
            else:
                workspace.gather = TrtllmAllGatherState(
                    group, min(spec.max_tokens, 128), spec.input_size, spec.device, True
                )
        if (
            spec.kind == "row"
            and use_reduction
            and size in (2, 4, 8, 16)
            and spec.output_size % 8 == 0
        ):
            workspace.reduction = TrtllmReduceScatterState(
                group, min(spec.max_tokens, 128), spec.output_size, spec.device
            )
        channels = spec.input_size if spec.kind == "row" else spec.output_size
        if size == 4 and channels % 8 == 0:
            workspace.a2a = TokenSpeedA2ALamportState(
                group, min(spec.max_tokens, 512), channels, spec.device, min(128, sms)
            )
            if channels % 32 == 0:
                workspace.a2a.prepare_chunk_exchange(threshold_bytes=8 * 2**20 + 1)
            if spec.kind == "row" and channels % 512 == 0:
                workspace.a2a.prepare_fp8_quantization()

    def all_gather(
        self,
        inputs: torch.Tensor,
        rows: int,
        quantize: bool,
        workspace: ProjectionWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if workspace.gather is None or rows > workspace.gather.max_rows:
            return projection_all_gather(
                inputs, rows, quantize, workspace, self._fallback
            )
        from tokenspeed_kernel.ops.communication.trtllm import (
            trtllm_allgather,
            trtllm_allgather_fp8_quantize,
        )

        _check_workspace(rows, workspace, self._fallback)
        send = prepare_padded_rows(inputs, rows, workspace.send, alignment_bytes=16)
        if quantize and workspace.gather_quant is not None:
            return trtllm_allgather_fp8_quantize(workspace.gather_quant, send)
        return trtllm_allgather(workspace.gather, send), None

    def all_to_all(
        self,
        inputs: torch.Tensor,
        rows: int,
        inverse: bool,
        quantize: bool,
        out: torch.Tensor | None,
        workspace: ProjectionWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if workspace.a2a is None or rows > workspace.a2a.max_rows:
            return projection_all_to_all(
                inputs, rows, inverse, quantize, out, workspace, self._fallback
            )
        from tokenspeed_kernel.ops.communication.cuda import (
            tokenspeed_a2a_lamport,
            tokenspeed_a2a_lamport_fp8_quantize,
        )

        _check_workspace(rows, workspace, self._fallback)
        _check_a2a_options(inverse, quantize, out)
        sent = (
            inputs.contiguous()
            if inverse
            else prepare_padded_rows(inputs, rows, workspace.send, alignment_bytes=16)
        )
        if quantize and workspace.a2a.fp8_output is not None:
            return tokenspeed_a2a_lamport_fp8_quantize(workspace.a2a, sent)
        return (
            tokenspeed_a2a_lamport(workspace.a2a, sent, inverse=inverse, out=out),
            None,
        )

    def acquire_output(self, rows: int, workspace: ProjectionWorkspace) -> torch.Tensor:
        if workspace.reduction is None or rows > workspace.reduction.max_rows:
            return acquire_projection_output(rows, workspace, self._fallback)
        _check_workspace(rows, workspace, self._fallback)
        return workspace.reduction.input_buffer(rows)

    def reduce_scatter(
        self, partial: torch.Tensor, rows: int, workspace: ProjectionWorkspace
    ) -> torch.Tensor:
        if workspace.reduction is None or rows > workspace.reduction.max_rows:
            return projection_reduce_scatter(partial, rows, workspace, self._fallback)
        from tokenspeed_kernel.ops.communication.trtllm import trtllm_reduce_scatter

        _check_workspace(rows, workspace, self._fallback)
        return trtllm_reduce_scatter(workspace.reduction, partial, rows)
