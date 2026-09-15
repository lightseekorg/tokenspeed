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

"""Experimental TP8 shared RS with caller-visible symmetric producer inputs."""

import re
import tempfile
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.shared_rs_contract import SharedRsTuning
from tokenspeed_kernel.ops.communication.triton import create_hidden_rsag_state
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _vote(group: dist.ProcessGroup, identity: Any, error: str | None) -> None:
    records = [None] * dist.get_world_size(group)
    dist.all_gather_object(records, (identity, error), group=group)
    if any(r[1] is not None or r[0] != records[0][0] for r in records):
        raise ValueError(f"collective shared-RS validation failed: {records}")


def _cuda_result(result: tuple) -> Any:
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA driver error: {result}")
    return result[1] if len(result) == 2 else result[1:]


@dataclass
class SharedRsWorkspace:
    """Own input/shard/signal allocations for one non-overlapping stream.

    Allocate outside capture. Independent graphs own separate workspaces;
    sequential layers of one graph may reuse a workspace. The strict RS exit
    protects peer reads before the next producer overwrites the input.
    """

    state: Any

    @classmethod
    def allocate(cls, group: dist.ProcessGroup, max_tokens: int, device: torch.device):
        """Collectively allocate TP8 BF16 input/shard storage outside capture.

        Args:
            group: Eight-rank NVLS-capable process group.
            max_tokens: Capacity in [1,8192], identical on every rank.
            device: This rank's CUDA device.

        Returns:
            Independently owned workspace; never shared by concurrent graphs.
        """
        error = None
        if dist.get_world_size(group) != 8 or max_tokens < 1 or max_tokens > 8192:
            error = "requires TP8 and capacity in [1,8192]"
        if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            error = "allocation requires CUDA outside graph capture"
        _vote(group, max_tokens, error)
        return cls(
            create_hidden_rsag_state(
                group,
                dist.get_rank(group),
                max_tokens,
                7168,
                device,
            )
        )


class BoundSharedRs:
    """Prepared input [M,7168] -> owned shard [M,896], without staging.

    Pass input_view as the producer's out= or use stage() for copied controls.
    run() does not allocate, validate collectively, compile, or change pointers.
    Keep this plan/workspace alive while a captured graph references it.
    Cross-stream execution and concurrent workspace reuse are unsupported.
    """

    @classmethod
    def prepare(
        cls,
        workspace: SharedRsWorkspace,
        m: int,
        tuning: SharedRsTuning,
        protected: tuple[torch.Tensor, ...],
    ):
        """Collectively validate, compile and bind exact-M producer/shard views.

        Args:
            workspace: Allocation owner kept alive throughout graph replay.
            m: Identical rank-local row count in [0,capacity].
            tuning: Explicit rank-identical geometry and request layout.
            protected: Residual, weight and final-output storage that cannot alias
                either RS buffer. The sequence may be empty for standalone RS.

        Returns:
            Bound plan with input_view [M,7168] and output_view [M,896].
        """
        state = workspace.state
        error = None
        try:
            tuning.validate(state.hidden_rsag_max_blocks)
            if not 0 <= m <= state.max_token_num:
                raise ValueError("M exceeds prepared capacity")
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("prepare must precede graph capture")
            for tensor, width in ((state.comm_buff, 7168), (state.local_buff, 896)):
                if (
                    tensor.dtype != torch.bfloat16
                    or tensor.device != state.device
                    or tuple(tensor.shape) != (state.max_token_num, width)
                    or not tensor.is_contiguous()
                    or tensor.data_ptr() % 16
                ):
                    raise ValueError(
                        "workspace buffers must be aligned contiguous CUDA BF16"
                    )
            buffers = (state.comm_buff, state.local_buff)
            for tensor, handle, width in (
                (state.comm_buff, state.symm_mem_hdl, 7168),
                (state.local_buff, state.local_symm_mem_hdl, 896),
            ):
                physical = handle.get_buffer(
                    state.rank_in_group,
                    (state.max_token_num, width),
                    torch.bfloat16,
                    storage_offset=0,
                )
                if physical.data_ptr() != tensor.data_ptr():
                    raise ValueError(
                        "workspace buffer does not match its symmetric mapping"
                    )
            for i, lhs in enumerate(buffers):
                for rhs in (*buffers[i + 1 :], *protected):
                    if (
                        lhs.untyped_storage().data_ptr()
                        == rhs.untyped_storage().data_ptr()
                    ):
                        raise ValueError(
                            "RS input/shard/protected tensors must not alias"
                        )
        except Exception as exc:
            error = str(exc)
        _vote(state.group, (m, tuning.key()), error)
        plan = cls()
        plan.workspace, plan.m, plan.tuning = workspace, m, tuning
        plan.input_view = state.comm_buff[:m]
        plan.output_view = state.local_buff[:m]
        plan.resources = {}
        plan.stream = torch.cuda.current_stream(state.device).cuda_stream
        if m:
            error = None
            try:
                plan._compile()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            _vote(state.group, (m, tuning.key()), error)
        return plan

    def _compile(self) -> None:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from tokenspeed_kernel.thirdparty.cute_dsl.shared_rs.kernel import (
            SharedRsKernel,
        )

        state = self.workspace.state
        signals = state.symm_mem_hdl.signal_pad_ptrs_dev
        signals_address = (
            signals.data_ptr() if isinstance(signals, torch.Tensor) else int(signals)
        )
        self.cuda, self.cutlass = cuda, cutlass
        self.args = (
            cutlass.Int64(int(state.symm_mem_hdl.multicast_ptr)),
            cutlass.Int64(self.output_view.data_ptr()),
            cutlass.Int64(signals_address),
            cutlass.Int32(self.m),
        )
        t = self.tuning
        # Eight compiler processes may share cwd. Keep intermediate artifacts
        # private so --keep-ptx/--keep-cubin cannot race another rank/config.
        self._compile_dump = tempfile.TemporaryDirectory(
            prefix=f"shared-rs-r{state.rank_in_group}-"
        )
        self.compiled = cute.compile(
            SharedRsKernel(
                t.ctas,
                t.threads,
                t.vectors_per_thread,
                t.layout == "row",
                state.rank_in_group,
            ),
            *self.args,
            cuda.CUstream(self.stream),
            options=f"--dump-dir={self._compile_dump.name} --keep-ptx --keep-cubin",
        )
        # Query the actual compiled function before allowing any barrier launch.
        ptx, cubin = self.compiled.__ptx__, self.compiled.__cubin__
        if not ptx or not cubin:
            raise RuntimeError(
                "compiled PTX/CUBIN are required for residency validation"
            )
        symbols = re.findall(r"\.entry\s+([A-Za-z0-9_$]+)", ptx)
        if len(symbols) != 1:
            raise RuntimeError(f"expected one RS CUDA function, got {symbols}")
        module = _cuda_result(cuda.cuModuleLoadData(cubin))
        try:
            function = _cuda_result(
                cuda.cuModuleGetFunction(module, symbols[0].encode())
            )
            for key, enum in (
                ("registers", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS),
                (
                    "local_bytes",
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                ),
                (
                    "shared_bytes",
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                ),
                (
                    "max_threads",
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                ),
            ):
                self.resources[key] = int(
                    _cuda_result(cuda.cuFuncGetAttribute(enum, function))
                )
            active = int(
                _cuda_result(
                    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        function,
                        t.threads,
                        0,
                    )
                )
            )
            self.resources["active_ctas_per_sm"] = active
            if active < 1 or self.resources["max_threads"] < t.threads:
                raise ValueError(
                    "compiled function cannot support one resident CTA per SM"
                )
        finally:
            _cuda_result(cuda.cuModuleUnload(module))

    def stage(self, source: torch.Tensor) -> torch.Tensor:
        """Copy a same-shape BF16 partial to the exact symmetric input view."""
        if (
            source.shape != self.input_view.shape
            or source.dtype != torch.bfloat16
            or source.device != self.input_view.device
            or not source.is_contiguous()
        ):
            raise ValueError("stage expects contiguous colocated BF16 [M,7168]")
        if (
            source.untyped_storage().data_ptr()
            == self.output_view.untyped_storage().data_ptr()
        ):
            raise ValueError("staging source must not alias the RS output")
        self.input_view.copy_(source)
        return self.input_view

    def run(self) -> torch.Tensor:
        """Return the bound contiguous BF16 shard after strict RS completion."""
        return shared_rs(self)


@register_kernel(
    "communication",
    "shared_rs",
    name="cutedsl_shared_rs",
    features={"mnnvl", "cuda_graph", "producer_direct"},
    solution="cutedsl",
    signatures=frozenset({format_signature(input=dense_tensor_format(torch.bfloat16))}),
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    priority=Priority.SPECIALIZED,
    tags={"blackwell", "throughput", "experimental"},
)
def shared_rs(plan: BoundSharedRs) -> torch.Tensor:
    """Execute a prepared experimental plan; zero-row plans are explicit no-ops."""
    if plan.m:
        current = torch.cuda.current_stream(plan.input_view.device).cuda_stream
        # CUDA capture uses a capture stream; the caller guarantees that this
        # workspace is never concurrently used on another stream.
        plan.compiled(*plan.args, plan.cuda.CUstream(current))
    return plan.output_view
