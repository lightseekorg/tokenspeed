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

"""Serving adapter for the qualified fused shared-RS/up-projection/AG kernel.

Allocations and compilation precede capture. Live operand descriptors are
host-only, non-synchronizing DLPack views, so a graph records its own latent
and residual pointers without additional device copies. The device body and
both publication/completion barriers are unchanged.
"""

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.ops.communication.fused_rs_up_projection import (
    prepare_fused_rs_up_projection,
)
from tokenspeed_kernel.ops.communication.fused_rs_up_projection_config import (
    fused_rs_up_projection_config,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_shared_rs import (
    SharedRsWorkspace,
    _vote,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    SymmetricUpProjectionOutput,
    _overlaps,
    _rank_barrier_kernel,
    allocate_symmetric_up_projection_output,
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


class FusedRsUpProjectionServing:
    """Per-layer output owner using one sequential model's shared raw workspace.

    Allocate during model initialization, before KV-cache sizing. Lazy compile
    is allowed during prefill warmup only, never during CUDA graph capture.
    Different buckets may reuse this layer's output because the serving engine
    executes them sequentially on its execution stream. Concurrent graphs or
    model instances must not share this object or its raw workspace.
    """

    def __init__(self, workspace: SharedRsWorkspace, max_tokens: int):
        """Allocate a layer output for workspace's TP8 group and explicit capacity.

        Both allocations must outlive every graph using the layer. Call outside
        capture, collectively in the same model-construction order on all ranks.
        """
        fused_rs_up_projection_config(max_tokens)
        self.workspace = workspace
        self.output = allocate_symmetric_up_projection_output(
            workspace.state.group, max_tokens, device=workspace.state.device
        )
        self._plans: dict[int, _PreparedLaunch] = {}

    def input_view(self, num_tokens: int) -> torch.Tensor:
        """Return the exact symmetric shared down-projection out= view."""
        fused_rs_up_projection_config(num_tokens)
        if num_tokens > self.output.tensor.shape[0]:
            raise ValueError("fused serving token count exceeds allocated capacity")
        return self.workspace.state.comm_buff[:num_tokens]

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

    def _prepare(self, m, latent, weight, residual):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("fused serving compilation must finish during warmup")
        output = SymmetricUpProjectionOutput(
            self.output.tensor[:m],
            self.output.handle,
            self.output.group,
            self.output.rank,
        )
        facade = prepare_fused_rs_up_projection(
            latent,
            weight,
            residual,
            self.workspace,
            output,
            residual_is_replicated=True,
        )
        bound = facade.plan
        # Retain only fixed weight/output/layout descriptors, not a warmup
        # residual allocation or its pointer. Every launch supplies live inputs.
        result = _PreparedLaunch(
            compiled=bound.compiled,
            cuda=bound._cuda,
            cutlass=bound._cutlass,
            fixed_args=bound._cute_args[1:4],
            weight=weight,
            output=output,
            compile_dump=bound._compile_dump,
            capacity_records=bound.qualified_capacity_records,
            kernel=bound.kernel,
            max_active_clusters=bound.max_active_clusters,
        )
        self._plans[m] = result
        return result

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
    tags={"blackwell", "throughput", "opt_in"},
)
def cutedsl_fused_rs_up_projection_ag(plan, latent, weight, residual, shared):
    """Execute the fused collective with the caller's live graph operands.

    Args:
        plan: Per-layer FusedRsUpProjectionServing with persistent output/raw owners.
        latent: Replicated contiguous CUDA BF16 [M,3584], produced by routed HT.
        weight: This layer's fixed contiguous CUDA BF16 [896,3584] owner weight.
        residual: Replicated contiguous CUDA BF16 [M,7168], added once.
        shared: Exact raw input view returned by plan.input_view(M), already
            filled by the shared producer on the same ordered stream.

    Returns:
        Persistent symmetric BF16 [M,7168] output, valid until the next call
        using this layer's output. Only M4096 and M8192 are supported.
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
