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

"""N64-only host specialization of the unchanged fused reduction/epilogue math."""

from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.fused_shared_rs import (
    FusedSharedRsUpProjectionGemm,
)


class FusedSharedRsN64UpProjectionGemm(FusedSharedRsUpProjectionGemm):
    """Retain CTA128x64 epilogue and exact NVLS bits; change MMA N only."""

    def __init__(
        self,
        raw_multicast_address,
        rank,
        c_stages,
        prefetch_acc_tile,
        addend_stages,
        reduce_vectors,
    ):
        if (c_stages, prefetch_acc_tile, addend_stages, reduce_vectors) != (
            0,
            False,
            1,
            4,
        ):
            raise ValueError("N64 experiment fixes auto C, no prefetch, addend1/group4")
        super().__init__(
            raw_multicast_address,
            rank,
            c_stages,
            prefetch_acc_tile,
            addend_stages,
            reduce_vectors,
        )
        # All constructor state is host-only; no descriptor, layout or compiled
        # MMA exists yet. Revalidate the selected geometry before JIT constructs
        # its actual hardware MMA and TMA/TMEM partitions.
        self.experimental_fused_n64 = True
        self.mma_tiler_mn = (256, 64)
        self.mma_tiler = (256, 64, 1)
        self.configure_epilogue_tile(128, 64)
        self.configure_addend_pipeline(1)
