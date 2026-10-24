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

"""LongCat boundary and publication rules for the shared distributed argmax."""

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.argmax import (
    ptx_redux_sync_min_u32,
    warp_argmax_redux,
    warp_reduce_argmax,
)


@dsl_user_op
def ptx_fence_proxy_alias(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.alias;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class LongcatArgmaxPolicy:
    invalid_index = 0x7FFFFFFF

    @cute.jit
    def wins(
        self,
        value: cutlass.Float32,
        idx: Int32,
        best: cutlass.Float32,
        best_idx: Int32,
    ):
        return idx != Int32(0x7FFFFFFF) and (
            (value != value and (best == best or idx < best_idx))
            or (
                value == value
                and best == best
                and (value > best or (value == best and idx < best_idx))
            )
        )

    @cute.jit
    def warp_argmax(
        self, value: cutlass.Float32, idx: Int32, use_redux: cutlass.Constexpr
    ):
        nan_idx = Int32(0x7FFFFFFF)
        warp_max, warp_idx = value, idx
        if value != value:
            nan_idx = idx
        nan_idx = ptx_redux_sync_min_u32(nan_idx)
        if nan_idx != Int32(0x7FFFFFFF):
            warp_max = Int32(0x7FC00000).bitcast(cutlass.Float32)
            warp_idx = nan_idx
        else:
            if cutlass.const_expr(use_redux):
                warp_max, warp_idx = warp_argmax_redux(value, idx)
            else:
                warp_max, warp_idx = warp_reduce_argmax(value, idx)
        return warp_max, warp_idx

    @cute.jit
    def index_to_scratch(self, idx: Int32):
        return idx.bitcast(cutlass.Float32)

    @cute.jit
    def index_from_scratch(self, value: cutlass.Float32):
        return value.bitcast(Int32)

    @cute.jit
    def after_publish(self):
        # Order lane0 multicast publication before every unicast consumer lane.
        cute.arch.sync_warp()
        ptx_fence_proxy_alias()

    @staticmethod
    def validate_handle(hdl, rank, world_size, group, device, strict):
        available = torch.tensor(
            [
                int(
                    hdl.rank == rank
                    and hdl.world_size == world_size
                    and bool(hdl.multicast_ptr)
                )
            ],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(available, op=dist.ReduceOp.MIN, group=group)
        if available.item():
            return True
        if strict:
            raise RuntimeError(
                "distributed_argmax requires a valid CUDA multicast / NVLS handle "
                "on every rank; at least one rank has an unavailable multicast "
                "mapping or a mismatched rank/world size"
            )
        return False
