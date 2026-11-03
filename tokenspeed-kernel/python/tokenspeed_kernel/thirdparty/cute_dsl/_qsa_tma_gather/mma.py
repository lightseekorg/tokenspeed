# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims


@cute.jit
def mma_ss(desc, desc_a_base, desc_b_base, tmem_c, *, accumulate: bool):
    """Issue a full K tile of single-CTA, unscaled shared/shared MMA."""
    enable_input_d = accumulate
    for k in cutlass.range_constexpr(desc.num_k_steps):
        ki_a, s_a = k % desc.sps_A, k // desc.sps_A
        ki_b, s_b = k % desc.sps_B, k // desc.sps_B
        offset_a_bytes = desc.smem_advance_A_intra * ki_a + desc.smem_subtile_A * s_a
        offset_b_bytes = desc.smem_advance_B_intra * ki_b + desc.smem_subtile_B * s_b
        # CUTLASS converts byte offsets to the descriptor's 16-byte address units.
        if prims.elect_sync():
            prims.tcgen05_mma(
                desc.kind,
                prims.CTAGroup.CTA_1,
                tmem_c,
                desc_a_base.advance_start_address(offset_a_bytes),
                desc_b_base.advance_start_address(offset_b_bytes),
                desc.idesc,
                enable_input_d,
            )
        enable_input_d = True
