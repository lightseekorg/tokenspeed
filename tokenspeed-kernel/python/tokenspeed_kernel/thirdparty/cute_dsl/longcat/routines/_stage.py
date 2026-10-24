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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Shared NVFP4 TMA and stage-layout construction."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync


class BlockScaledStages:
    """Shared layout construction for the existing NVFP4 pipelines."""

    @staticmethod
    def smem_views(storage, a_layout, b_layout, sfa_layout, sfb_layout):
        return (
            storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner),
            storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner),
            storage.sSFA.get_tensor(sfa_layout),
            storage.sSFB.get_tensor(sfb_layout),
        )

    @staticmethod
    def stage_byte_size(dtype, layout):
        return cute.size_in_bytes(dtype, cute.slice_(layout, (None, None, None, 0)))

    def stage_pipelines(
        self, storage, cluster_layout, *, tx_bytes, acc_stages, ab_consumers
    ):
        ab = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, ab_consumers
            ),
            tx_count=tx_bytes,
            cta_layout_vmnk=cluster_layout,
            defer_sync=True,
        )
        acc = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout,
            defer_sync=True,
        )
        return ab, acc

    def partition_operand(
        self, tensor, smem, atom, cluster_layout, thr_mma, coordinate, *, operand, scale
    ):
        # These selectors are Python constants at every call site.
        tiler = self.mma_tiler_sfb if operand == "B" and scale else self.mma_tiler
        modes = (None, 0, None) if operand == "A" else (0, None, None)
        gmem = cute.local_tile(tensor, cute.slice_(tiler, modes), (None, None, None))
        partition = (
            thr_mma.partition_A(gmem) if operand == "A" else thr_mma.partition_B(gmem)
        )
        cluster_modes = (0, 0, None, 0) if operand == "A" else (0, None, 0, 0)
        cta_layout = cute.make_layout(cute.slice_(cluster_layout, cluster_modes).shape)
        smem_part, gmem_part = cpasync.tma_partition(
            atom,
            coordinate,
            cta_layout,
            cute.group_modes(smem, 0, 3),
            cute.group_modes(partition, 0, 3),
        )
        if scale:
            smem_part = cute.filter_zeros(smem_part)
            gmem_part = cute.filter_zeros(gmem_part)
        return gmem, smem_part, gmem_part

    def sf_tmem_tensor(self, address, mma, layout, *, operand):
        make_layout = (
            blockscaled_utils.make_tmem_layout_sfa
            if operand == "A"
            else blockscaled_utils.make_tmem_layout_sfb
        )
        return cute.make_tensor(
            cute.recast_ptr(address, dtype=self.sf_dtype),
            make_layout(
                mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(layout, (None, None, None, 0)),
            ),
        )

    def weight_tma_atoms(self, mma, cluster_layout, cluster_shape, weight, weight_sf):
        weight_sf = cute.make_tensor(
            weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(weight.shape, self.sf_vec_size),
        )
        op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape, mma.thr_id)
        a, ga = cute.nvgpu.make_tiled_tma_atom_A(
            op,
            weight,
            cute.slice_(self.a_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            mma,
            cluster_layout.shape,
        )
        sfa, gsfa = cute.nvgpu.make_tiled_tma_atom_A(
            op,
            weight_sf,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            mma,
            cluster_layout.shape,
            internal_type=cutlass.Int16,
        )
        return a, ga, sfa, gsfa
