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

"""Medium-M host specialization of the same fused RS/GEMM/AG device pattern."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.fused_shared_rs import (
    FusedSharedRsUpProjectionGemm,
)
from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.gemm import (
    SymmetricUpProjectionGemm,
)


class MediumFusedSharedRsUpProjectionGemm(FusedSharedRsUpProjectionGemm):
    """Change geometry only; inherit vector reduction, pipeline and final stores.

    The wider validation is isolated here. The qualified large-M constructors
    retain their frozen geometry and validation. Every medium configuration
    requires numerical and compiled-resource admission before performance use.
    """

    def __init__(self, raw_multicast_address, rank, tuning):
        tuning.validate()
        if (
            type(raw_multicast_address) is not int
            or raw_multicast_address <= 0
            or raw_multicast_address % 16
        ):
            raise ValueError("raw multicast address must be positive and aligned")
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("requires an explicit TP8 owner rank")
        SymmetricUpProjectionGemm.__init__(
            self,
            cutlass.Float32,
            tuning.two_cta,
            (tuning.tile_m, tuning.tile_n),
            (tuning.cluster_m, tuning.cluster_n),
            True,
            False,
            "tma",
        )
        self.configure_output_pipeline(tuning.c_stages, True, True, False)
        self.requested_ab_stages = tuning.ab_stages
        self.medium_scheduler_type = tuning.scheduler_type
        self.configure_epilogue_tile(tuning.epilogue_m, tuning.epilogue_n)
        self.configure_addend_loads("no_allocate", False)
        self.configure_addend_pipeline(tuning.addend_stages)
        self.fused_shared_rs = True
        self.raw_multicast_address = raw_multicast_address
        self.raw_owner_rank = rank
        self.fused_reduce_vectors = tuning.reduce_vectors

    def configure_addend_pipeline(self, stages):
        """Enable generic coalesced BF16x8 partitions for full-CTA-M epilogues."""
        cta_m = self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1)
        if (
            type(stages) is not int
            or stages not in (1, 2)
            or cta_m not in (64, 128)
            or self.mma_tiler_mn[1] not in (64, 128)
            or self.requested_epilogue_tile != (cta_m, 64)
            or self.cluster_shape_mn != ((2, 1) if self.use_2cta_instrs else (1, 1))
            or self.store_kind != "tma"
            or self.addend_cache_policy != "no_allocate"
            or self.paired_addend_loads
        ):
            raise ValueError("medium addends require full CTA64/128 x64 epilogues")
        self.addend_stages = stages

    def _setup_attributes(self):
        super()._setup_attributes()
        if self.requested_ab_stages:
            # Rebuild only the A/B staged layouts, before __call__ constructs
            # TMA partitions or the device kernel. C is explicit in this mode
            # so reducing A/B storage cannot refill the saved space with C.
            # The inherited ACC2/TMEM allocation and every device pipeline
            # consume the updated stage count through the existing attributes.
            tiled_mma = self._create_tiled_mma()
            self.num_ab_stage = self.requested_ab_stages
            self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
            )
            self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
            )
        # Retain plain byte counts while the MLIR layouts are still live. The
        # admission query uses a conservative reserve for metadata/alignment,
        # not a fabricated observed launch size or zero-shared occupancy query.
        self.medium_a_smem_bytes = cute.size_in_bytes(
            self.a_dtype, self.a_smem_layout_staged
        )
        self.medium_b_smem_bytes = cute.size_in_bytes(
            self.b_dtype, self.b_smem_layout_staged
        )
        self.medium_c_smem_bytes = cute.size_in_bytes(
            self.c_dtype, self.c_smem_layout_staged
        )
        self.medium_smem_budget_bytes = (
            1024
            + self.medium_a_smem_bytes
            + self.medium_b_smem_bytes
            + self.medium_c_smem_bytes
            + self.addend_smem_bytes
        )
        if 16 * self.num_ab_stage + 16 * self.num_acc_stage + 12 + 5 * 127 > 1024:
            raise ValueError("A/B stages exceed the metadata/alignment reserve")
        if self.medium_smem_budget_bytes > self.smem_capacity:
            raise ValueError(
                "explicit A/B/C/addend layouts exceed device shared memory"
            )

    def _compute_grid(
        self, c, cta_tile_shape_mnk, cluster_shape_mn, max_active_clusters
    ):
        # Keep the existing scheduler params, cluster-shaped X/Y coordinates,
        # work indexing from blockIdx.z and all device count/phase transitions.
        # Giving the static scheduler every output work tile makes each cluster
        # execute one iteration; it does not require all clusters to be resident.
        if self.medium_scheduler_type == "full_grid":
            max_active_clusters = (
                cute.ceil_div(c.shape[0], self.mma_tiler_mn[0])
                * cute.ceil_div(c.shape[1], self.mma_tiler_mn[1])
                * c.shape[2]
            )
        return SymmetricUpProjectionGemm._compute_grid(
            c, cta_tile_shape_mnk, cluster_shape_mn, max_active_clusters
        )
