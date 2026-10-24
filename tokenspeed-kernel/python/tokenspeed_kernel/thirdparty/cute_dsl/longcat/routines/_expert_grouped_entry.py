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

"""Expert-grouped MoE GateUp/Down on a fixed number of fill CTAs.

The verify-8 MoE step routes 8 tokens to at most 96 distinct experts.  Instead
of one GEMM per (token, slot) route (which streams an expert's weights once per
token that picked it), the work is grouped **per distinct expert**: group ``g``
multiplies the shared ``[8, 8192]`` NVFP4 activation (all eight tokens -- the
MMA tile is 128 rows anyway) against expert ``ids[g]``, so every expert shard
is read exactly once.  Membership is applied afterwards through
``weight_table[g, t]`` (route weight, 0 for tokens that did not pick the
expert).

GateUp streams weight-as-M operands and publishes one arrival per group. Down
owners partition the hidden tiles into slabs and accumulate weighted expert
outputs into ``acc[8, 8192]``. Both families use independent one-CTA owners;
the launch contains ``2 * (gate_clusters + down_clusters)`` CTAs. Tickets are
self-generational, so repeated launches need no reset.
"""

from __future__ import annotations

from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._down_kernel import (
    LongcatPersistentDownKStageKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._expert_gate_v2_kernel import (
    LongcatExpertGateV2Kernel,
)

TOKENS = 8
GROUPS = 96
HIDDEN = 8192
INTERMEDIATE = 256
_GATE_SCALE_K = HIDDEN // 16
_OUTPUT_TILES_MAX = HIDDEN // 128
_DOWN_SCALE_K = INTERMEDIATE // 16
# Weight-as-M GateUp (``gate_v2``) pipeline geometry: K/stage 256, ten stages.
# The pipeline holds approximately 160 KB of weight bytes within the SMEM cap.
GATE_V2_K_BLOCKS_PER_STAGE = 4
GATE_V2_AB_STAGES = 10


class LongcatExpertGroupedEntry:
    """Prepare the expert routines hosted by the layer."""

    threads_per_cta = 192

    def __init__(
        self,
        *,
        gate_clusters: int,
        down_clusters: int,
        down_weight_prefetch_depth: int = 1,
        gate_v2_k_blocks: int = GATE_V2_K_BLOCKS_PER_STAGE,
        gate_v2_ab_stages: int = GATE_V2_AB_STAGES,
        down_group_splits: int = 2,
    ) -> None:
        if (gate_clusters, down_clusters, down_group_splits) != (16, 8, 2):
            raise ValueError("the layer expert topology is 16x8x2")
        if (down_weight_prefetch_depth, gate_v2_k_blocks, gate_v2_ab_stages) != (
            1,
            4,
            10,
        ):
            raise ValueError(
                "the layer expert pipeline requires Down prefetch=1 "
                "and GateUp stages=4x10"
            )
        self.gate_clusters = gate_clusters
        self.down_clusters = down_clusters
        self.down_group_splits = down_group_splits
        self.gate_v2_up = LongcatExpertGateV2Kernel(
            num_workers=gate_clusters * 2,
            k_blocks_per_stage=gate_v2_k_blocks,
            ab_stages=gate_v2_ab_stages,
        )
        self.physical_cluster_count = gate_clusters + down_clusters
        self.down = LongcatPersistentDownKStageKernel(
            sf_vec_size=16,
            mma_tiler_mn=(128, 8),
            cluster_shape_mn=(1, 1),
            use_prefetch=down_weight_prefetch_depth > 0,
            gate_stage_producers_per_generation=4,
            weight_prefetch_depth=down_weight_prefetch_depth,
        )


__all__ = ["GROUPS", "LongcatExpertGroupedEntry"]
