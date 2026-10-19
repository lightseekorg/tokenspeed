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

"""Per-route input-scale RMSNorm and weighted top-k sum on one cluster per token.

CTA ``k`` normalizes route ``k`` and publishes its inverse RMS to every peer;
each CTA then reduces its column slice over the routes in the Triton order.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.typing import Float32, Int32, Int64
from tokenspeed_kernel.thirdparty.cute_dsl.argmax import elem_pointer
from tokenspeed_kernel.thirdparty.cute_dsl.cluster_boundary import (
    publish_to_cluster,
    rsqrt_approx,
    warp_sum,
)

# One CTA per route, eight elements per thread of a 4096-wide route's variance.
ROUTED_CTA_THREADS = 512
_WARPS = ROUTED_CTA_THREADS // 32


class ClusterRoutedSumKernel:
    """Grid ``(tokens, topk)`` with cluster ``(1, topk, 1)``; CTA ``k`` owns route ``k``.

    Args:
        width: Route width, a power of two divisible by ``topk * ROUTED_CTA_THREADS``.
        topk: Routes per token, the cluster size, at most 16.
        eps: Epsilon for each route's variance.
        weight_offset: Center added to the checkpoint's norm weight.
        use_pdl: Launch with programmatic dependent launch.

    The call takes BF16 ``routes [rows, padded_width]``, the int32 permute map,
    FP32 ``[tokens * topk]`` route weights, the BF16 norm weight, the BF16
    ``[tokens, width]`` output, ``tokens``, the routes' row stride and the stream.
    """

    def __init__(
        self,
        width: int,
        topk: int,
        eps: float,
        weight_offset: float,
        use_pdl: bool,
    ) -> None:
        self.width = width
        self.topk = topk
        self.per_thread = width // ROUTED_CTA_THREADS
        self.slice = width // topk
        self.per_thread_out = self.slice // ROUTED_CTA_THREADS
        # Warp partials, cluster slots, then one 8-byte aligned mbarrier.
        self.smem_bytes = ((_WARPS + topk) * 4 + 7) // 8 * 8 + 8
        self.eps = eps
        self.weight_offset = weight_offset
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        routes: cute.Tensor,
        permute_map: cute.Tensor,
        route_weights: cute.Tensor,
        norm_weight: cute.Tensor,
        output: cute.Tensor,
        tokens: Int32,
        row_stride: Int32,
        cu_stream: cuda.CUstream,
    ):
        """Launch one cluster per token."""
        self.kernel(
            routes, permute_map, route_weights, norm_weight, output, row_stride
        ).launch(
            grid=[tokens, self.topk, 1],
            block=[ROUTED_CTA_THREADS, 1, 1],
            cluster=[1, self.topk, 1],
            smem=self.smem_bytes,
            stream=cu_stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routes: cute.Tensor,
        permute_map: cute.Tensor,
        route_weights: cute.Tensor,
        norm_weight: cute.Tensor,
        output: cute.Tensor,
        row_stride: Int32,
    ):
        """One route's CTA for one token."""
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        rank = cute.arch.block_idx_in_cluster()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        per_thread: cutlass.Constexpr = self.per_thread
        per_thread_out: cutlass.Constexpr = self.per_thread_out
        width: cutlass.Constexpr = self.width
        topk: cutlass.Constexpr = self.topk
        inv_width: cutlass.Constexpr = 1.0 / self.width

        smem = cutlass.utils.SmemAllocator()
        warp_partials = smem.allocate_tensor(
            Float32, cute.make_layout((_WARPS,)), byte_alignment=4
        )
        inv_rms = smem.allocate_tensor(
            Float32, cute.make_layout((topk,)), byte_alignment=4
        )
        mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            # Dependents may launch now; their own wait still covers this grid's completion.
            cute.arch.griddepcontrol_launch_dependents()

        # This CTA's route: -1 marks a route another rank owns, all zeros.
        route_index = token * topk + rank
        row = permute_map[route_index]
        owned = row >= 0
        if not owned:
            row = Int32(0)
        row_layout = cute.make_layout((width,))
        # 64-bit offsets: the permuted expert output can exceed 2^31 elements.
        route_row = cute.make_tensor(
            routes.iterator + Int64(row) * Int64(row_stride), row_layout
        )
        col0 = tidx * per_thread
        partial = Float32(0.0)
        for i in cutlass.range_constexpr(per_thread):
            value = Float32(0.0)
            if owned:
                value = route_row[col0 + i].to(Float32)
            weight = (
                (norm_weight[col0 + i].to(Float32) + self.weight_offset)
                .to(cutlass.BFloat16)
                .to(Float32)
            )
            scaled = value * weight
            partial = partial + scaled * scaled
        partial = warp_sum(partial)
        if lane == 0:
            warp_partials[warp] = partial
        cute.arch.barrier()
        total = Float32(0.0)
        for w in cutlass.range_constexpr(_WARPS):
            total = total + warp_partials[w]
        variance = total * inv_width
        own_inv_rms = rsqrt_approx(variance + self.eps)

        cute.arch.cluster_wait()
        publish_to_cluster(
            own_inv_rms, elem_pointer(inv_rms, (rank,)), mbar_ptr, topk, warp, lane
        )

        # The token's rows and route weights are loaded once, not per column.
        peer_rows = cute.make_rmem_tensor((topk,), Int32)
        peer_weights = cute.make_rmem_tensor((topk,), Float32)
        for k in cutlass.range_constexpr(topk):
            peer_rows[k] = permute_map[token * topk + k]
            peer_weights[k] = route_weights[token * topk + k]
        out_row = cute.make_tensor(output.iterator + token * width, row_layout)
        base = rank * self.slice + tidx * per_thread_out
        for i in cutlass.range_constexpr(per_thread_out):
            col = base + i
            weight = (
                (norm_weight[col].to(Float32) + self.weight_offset)
                .to(cutlass.BFloat16)
                .to(Float32)
            )
            acc = Float32(0.0)
            for k in cutlass.range_constexpr(topk):
                peer_row = peer_rows[k]
                value = Float32(0.0)
                if peer_row >= 0:
                    peer_tensor = cute.make_tensor(
                        routes.iterator + Int64(peer_row) * Int64(row_stride),
                        row_layout,
                    )
                    value = peer_tensor[col].to(Float32)
                # Operation order pinned to the Triton sum.
                acc = acc + (value * weight) * inv_rms[k] * peer_weights[k]
            out_row[col] = acc.to(cutlass.BFloat16)

        # Redundant with the mbarrier wait, but keeps the cluster exit ordered.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
