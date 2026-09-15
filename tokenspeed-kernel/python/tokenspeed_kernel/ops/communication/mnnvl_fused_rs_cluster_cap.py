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

"""Explicit persistent-cluster cap experiment; no runtime registration."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_fused_rs_up_projection import (
    BoundFusedRsUpProjection,
    FusedRsUpProjectionTuning,
)
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    _vote,
)

CAPS = (38, 56, 64, 76)


@dataclass(frozen=True)
class FusedRsClusterCapTuning(FusedRsUpProjectionTuning):
    """Keep current N128 grouped RS/up and explicitly lower only its launch cap.

    Args:
        cluster_cap: Exactly 38, 56, 64 or 76 two-CTA clusters. This does not
            change hardware capacity, MMA shape, stage depths or HT geometry.
    """

    cluster_cap: int

    def validate(self):
        super().validate()
        if (
            type(self.cluster_cap) is not int
            or self.cluster_cap not in CAPS
            or self.tile_m != 256
            or self.tile_n != 128
            or not self.two_cta
            or (self.cluster_m, self.cluster_n) != (2, 1)
            or (self.epilogue_m, self.epilogue_n) != (128, 64)
            or self.c_stages != 0
            or self.prefetch_acc_tile
            or self.addend_stages != 1
            or self.reduce_vectors != 4
            or self.enable_pdl
            or self.store_kind != "tma"
            or not self.release_acc_early
            or not self.acquire_before_store
            or self.addend_cache_policy != "no_allocate"
            or self.paired_addend_loads
        ):
            raise ValueError(
                "cap sweep fixes current N128/group4/stage1/auto-C and cap38/56/64/76"
            )


class BoundFusedRsClusterCap(BoundFusedRsUpProjection):
    """Inherit the original complete RS/up publication and output lifecycle."""

    @classmethod
    def prepare_fused(
        cls,
        latent,
        weight,
        residual,
        workspace,
        output,
        *,
        residual_is_replicated,
        tuning,
    ):
        """Collectively bind one exact-M4096 cap before compilation/capture.

        The tensor/workspace/return contract is the original fused binding.
        All ranks must explicitly request the same typed tuning. Capacity is
        queried on each rank; a cap above their minimum is rejected, never clamped.
        """
        local, error = None, None
        try:
            if type(tuning) is not FusedRsClusterCapTuning:
                raise ValueError("requires the explicit cluster-cap tuning type")
            tuning.validate()
            if output.tensor.shape[0] != 4096:
                raise ValueError("cluster-cap experiment requires exact M4096")
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("cap admission must precede graph capture")
            if dist.get_world_size(output.group) != 8:
                raise ValueError("cap admission requires TP8")
            import cutlass.utils as utils

            with torch.cuda.device(output.tensor.device):
                capacity = utils.HardwareInfo().get_max_active_clusters(2)
                properties = torch.cuda.get_device_properties(output.tensor.device)
            if type(capacity) is not int or capacity < 1:
                raise ValueError("hardware cluster capacity must be a positive integer")
            local = {
                "rank": dist.get_rank(output.group),
                "hardware_cluster_capacity": capacity,
                "device": str(output.tensor.device),
                "sm_count": properties.multi_processor_count,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        identity = ("fused-cluster-cap-admission-v1", 4096, tuning)
        _vote(output.group, identity, error)
        records = [None] * 8
        dist.all_gather_object(records, local, group=output.group)
        minimum = min(record["hardware_cluster_capacity"] for record in records)
        error = None
        if tuning.cluster_cap > minimum:
            error = f"requested cap{tuning.cluster_cap} exceeds all-rank minimum hardware capacity{minimum}"
        _vote(output.group, identity, error)
        plan = super().prepare_fused(
            latent,
            weight,
            residual,
            workspace,
            output,
            residual_is_replicated=residual_is_replicated,
            tuning=tuning,
        )
        plan.cap_admission_records = records
        plan.minimum_hardware_capacity = minimum
        plan._cap_stream = torch.cuda.current_stream(output.tensor.device).cuda_stream
        return plan

    def _select_max_active_clusters(self, hardware_capacity):
        # Called inside the original compile try/outer collective vote. No new
        # host collective here: a peer could have failed earlier in that try.
        selected = self.tuning.cluster_cap
        if type(hardware_capacity) is not int or not 1 <= selected <= hardware_capacity:
            raise ValueError("cluster capacity changed or cannot support requested cap")
        self.hardware_cluster_capacity = hardware_capacity
        self.launched_cluster_cap = selected
        return selected

    def _check_stream(self):
        if (
            torch.cuda.current_stream(self.output.tensor.device).cuda_stream
            != self._cap_stream
        ):
            raise ValueError("cluster-cap plan must run/capture on its bound stream")

    def stage(self, partial):
        """Copy the original partial on the explicitly bound stream."""
        self._check_stream()
        return super().stage(partial)

    def __call__(self):
        self._check_stream()
        return super().__call__()
