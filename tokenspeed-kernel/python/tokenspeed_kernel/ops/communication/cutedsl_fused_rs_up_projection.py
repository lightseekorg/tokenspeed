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

"""Shared live-pointer launcher for fused shared-RS/up-projection/AG profiles.

Allocations and compilation precede capture. Live operand descriptors are
host-only, non-synchronizing DLPack views, so a graph records its own latent
and residual pointers without additional device copies. The device body and
both publication/completion barriers are unchanged.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.ops.communication.fused_rs_workspace import _vote
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    SymmetricUpProjectionOutput,
    _overlaps,
    _rank_barrier_kernel,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@dataclass
class _PreparedLaunch:
    compiled: Any
    cuda: Any
    cutlass: Any
    fixed_args: tuple
    weight: torch.Tensor
    output: SymmetricUpProjectionOutput
    compile_dump: Any
    capacity_records: list
    kernel: Any
    max_active_clusters: int


class FusedRsUpProjectionServingBase(ABC):
    """Share validation and live launches without selecting a token policy.

    Concrete profiles own output allocation, input views and compilation.
    This base cannot be instantiated as a separate serving entry point.
    """

    @abstractmethod
    def input_view(self, num_tokens: int) -> torch.Tensor:
        """Return the profile's symmetric shared-producer destination."""
        raise NotImplementedError

    @abstractmethod
    def _prepare(self, m, latent, weight, residual) -> _PreparedLaunch:
        """Bind and compile the selected profile before graph capture."""
        raise NotImplementedError

    def _validate(self, latent, weight, residual, shared):
        if latent.ndim != 2:
            raise ValueError("latent must have rank two")
        m = latent.shape[0]
        raw = self.input_view(m)
        for tensor, shape in (
            (latent, (m, 3584)),
            (weight, (896, 3584)),
            (residual, (m, 7168)),
            (shared, (m, 7168)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != self.output.tensor.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(f"fused serving requires aligned BF16 {shape}")
        if shared.data_ptr() != raw.data_ptr():
            raise ValueError("shared producer must write the bound symmetric input")
        for protected in (latent, weight, residual):
            if _overlaps(raw, protected) or _overlaps(self.output.tensor, protected):
                raise ValueError("fused serving storage aliases a live input")
        return m

    def __call__(self, latent, weight, residual, shared):
        """Run with live operands; return this layer's persistent output view."""
        return cutedsl_fused_rs_up_projection_ag(self, latent, weight, residual, shared)


@register_kernel(
    "communication",
    "fused_rs_up_projection_ag",
    name="cutedsl_fused_rs_up_projection_ag",
    features={"mnnvl", "multicast_gemm", "cuda_graph", "symmetric_output"},
    solution="cutedsl",
    signatures=frozenset(
        {
            format_signature(
                latent=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
                residual=dense_tensor_format(torch.bfloat16),
                shared=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
    ),
    priority=Priority.SPECIALIZED,
    tags={"blackwell", "throughput", "integrated"},
)
def cutedsl_fused_rs_up_projection_ag(plan, latent, weight, residual, shared):
    """Execute the fused collective with the caller's live graph operands.

    Args:
        plan: Concrete serving profile with persistent output/raw owners.
        latent: Replicated contiguous CUDA BF16 [M,3584], produced by routed BT/HT.
        weight: This layer's fixed contiguous CUDA BF16 [896,3584] owner weight.
        residual: Replicated contiguous CUDA BF16 [M,7168], added once.
        shared: Exact raw input view returned by plan.input_view(M), already
            filled by the shared producer on the same ordered stream.

    Returns:
        Persistent symmetric BF16 [M,7168] output, valid until the next call
        using the same output slot. The concrete profile selects supported M;
        the integrated profile covers the continuous interval (32,8192].
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import to_cute

    m, error = None, None
    try:
        m = plan._validate(latent, weight, residual, shared)
        existing = plan._plans.get(m)
        if existing is not None and weight.data_ptr() != existing.weight.data_ptr():
            raise ValueError("up-projection weight changed; rebuild the serving plan")
    except (ValueError, AttributeError, IndexError) as exc:
        error = str(exc)
    if not torch.cuda.is_current_stream_capturing():
        # Warmup is collective, including validation failures. Replay executes
        # only the recorded device graph; capture never enters this agreement.
        _vote(plan.workspace.state.group, ("fused-serving", m), error)
    elif error is not None:
        raise ValueError(error)
    prepared = plan._plans.get(m)
    if prepared is None:
        prepared = plan._prepare(m, latent, weight, residual)
    rank = prepared.output.rank
    owner = slice(rank * 896, (rank + 1) * 896)
    # to_cute uses stream=-1: no implicit producer synchronization or storage
    # allocation. The framework orders warmup, capture and replay explicitly.
    arguments = (
        to_cute(latent.unsqueeze(-1), 16),
        *prepared.fixed_args,
        to_cute(residual[:, owner].unsqueeze(-1), 16),
    )
    _rank_barrier_kernel[(1,)](
        plan.workspace.state.symm_mem_hdl.signal_pad_ptrs_dev,
        RANK=rank,
        ENABLE_PDL=False,
        num_warps=4,
    )
    prepared.compiled(
        *arguments,
        prepared.cutlass.Int64(prepared.output.handle.multicast_ptr + rank * 896 * 2),
        prepared.cutlass.Boolean(True),
        prepared.cuda.CUstream(torch.cuda.current_stream(latent.device).cuda_stream),
    )
    _rank_barrier_kernel[(1,)](
        prepared.output.handle.signal_pad_ptrs_dev,
        RANK=rank,
        ENABLE_PDL=False,
        num_warps=4,
    )
    return prepared.output.tensor
