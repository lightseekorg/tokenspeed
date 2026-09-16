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

"""Functionally named, explicitly bound large-M shared-RS/up-projection/AG API.

Preparation is collective and outside capture. This entry point does not
change model/runtime dispatch. The device implementation is the qualified
persistent GEMM; no independent RS kernel or output materializer is launched.
"""

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.fused_rs_up_projection_config import (
    fused_rs_up_projection_config,
)
from tokenspeed_kernel.ops.communication.fused_rs_workspace import _vote
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_fused_rs_up_projection import (
    BoundFusedRsUpProjection,
    FusedRsUpProjectionTuning,
)
from tokenspeed_kernel.ops.communication.mnnvl_fused_rs_cluster_cap import (
    BoundFusedRsClusterCap,
    FusedRsClusterCapTuning,
)


class FusedRsUpProjection:
    """A one-stream facade owning a qualified bound collective plan.

    Keep the plan, all inputs, workspace and output alive through graph use.
    input_view is the exact producer out= destination; tensor contents may be
    updated between ordered calls, but pointers and shapes must not change.
    Calls return aliased persistent output, never a newly allocated result.
    Independent/concurrent graphs must own independent allocations.
    """

    def __init__(self, plan, config, stream):
        self.plan = plan
        self.config = config
        self.stream = stream

    @property
    def input_view(self):
        """Return BF16 [M,7168] input for a same-stream copy or producer out=."""
        return self.plan.input_view

    @property
    def output(self):
        """Return the caller-owned symmetric output allocation and its handle."""
        return self.plan.output

    def _check_stream(self):
        current = torch.cuda.current_stream(self.output.tensor.device).cuda_stream
        if current != self.stream:
            raise ValueError("fused RS/up/AG must execute on its prepared stream")

    def stage(self, partial):
        """Copy a contiguous BF16 [M,7168] partial on the prepared stream."""
        self._check_stream()
        return self.plan.stage(partial)

    def __call__(self):
        """Publish raw input, execute fused RS/up/AG, finish all ranks, return out."""
        self._check_stream()
        return self.plan()


def prepare_fused_rs_up_projection(
    latent, weight, residual, workspace, output, *, residual_is_replicated
):
    """Bind the qualified endpoint without allocating or launching a producer.

    Args:
        latent: Replicated contiguous CUDA BF16 [M,3584], updated by routed HT.
        weight: Owner-local contiguous CUDA BF16 [896,3584].
        residual: Bitwise-replicated contiguous CUDA BF16 [M,7168].
        workspace: SharedRsWorkspace with symmetric input capacity at least M.
        output: Separate SymmetricUpProjectionOutput for this live layer/graph.
        residual_is_replicated: Explicit True on every TP8 rank.
    Returns:
        A callable one-stream FusedRsUpProjection exposing its exact input_view.
        All validation, compilation and cross-rank agreement precede capture.
    """
    config, error = None, None
    try:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("fused RS/up/AG preparation must precede capture")
        config = fused_rs_up_projection_config(output.tensor.shape[0])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _vote(output.group, ("fused-rs-up-ag", config, residual_is_replicated), error)
    binding = BoundFusedRsUpProjection
    values = config["tuning"]
    tuning = FusedRsUpProjectionTuning(**values)
    if config["cluster_cap"] is not None:
        tuning = FusedRsClusterCapTuning(**values, cluster_cap=config["cluster_cap"])
        binding = BoundFusedRsClusterCap
    plan = binding.prepare_fused(
        latent,
        weight,
        residual,
        workspace,
        output,
        residual_is_replicated=residual_is_replicated,
        tuning=tuning,
    )
    local = {
        "rank": output.rank,
        "hardware_capacity": (
            plan.hardware_cluster_capacity
            if config["cluster_cap"] is not None
            else plan.max_active_clusters
        ),
        "launched_clusters": plan.max_active_clusters,
    }
    records = [None] * dist.get_world_size(output.group)
    dist.all_gather_object(records, local, group=output.group)
    error = None
    if (
        len(records) != 8
        or min(row["hardware_capacity"] for row in records) < plan.max_active_clusters
        or len({row["launched_clusters"] for row in records}) != 1
    ):
        error = "fused RS/up/AG requires TP8 uniform admitted cluster capacity"
    _vote(output.group, ("fused-rs-up-ag-capacity", config), error)
    plan.qualified_capacity_records = records
    stream = torch.cuda.current_stream(output.tensor.device).cuda_stream
    return FusedRsUpProjection(plan, config, stream)
