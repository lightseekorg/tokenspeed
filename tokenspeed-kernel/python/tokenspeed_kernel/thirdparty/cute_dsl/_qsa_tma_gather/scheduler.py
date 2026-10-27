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


from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import pipeline
from cutlass.experimental import primitives as prims
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_tma_gather.barrier import (
    arrive_expect_tx,
    wait,
)


@cute.jit
def read_tile_id_arrive(mb):
    """Credit one consumer warp after decoding the single-CTA CLC response."""
    if prims.elect_sync():
        prims.mbarrier_arrive(mb)


class Sched(NamedTuple):
    mb_scheduler: object
    mb_read_tile_id: object
    tile_id_smem: object
    bidx_init: object
    bidy_init: object
    bidz_init: object


@cute.jit
def read_clc_payload(sched, base_word, *, warp_broadcast: cutlass.Constexpr[bool]):
    """Decode one scheduler response slot before returning its slot credit."""
    if cutlass.const_expr(warp_broadcast):
        # The local persistent producer must wait for the last payload reader.
        # Read through lane zero and broadcast before returning slot credit,
        # rather than letting independently scheduled lanes
        # retain shared-memory reads after the elected release-arrive.
        x = cutlass.Int32(0)
        y = cutlass.Int32(0)
        valid = cutlass.Int32(0)
        if (cute.arch.thread_idx()[0] & cutlass.Int32(31)) == 0:
            vec = sched.tile_id_smem.load(base_word, vector_size=4, alignment=16)
            x, y, valid = vec[0], vec[1], vec[2]
        x = cute.arch.shuffle_sync(x, 0)
        y = cute.arch.shuffle_sync(y, 0)
        valid = cute.arch.shuffle_sync(valid, 0)
    else:
        vec = sched.tile_id_smem.load(base_word, vector_size=4, alignment=16)
        x, y, valid = vec[0], vec[1], vec[2]
    prims.fence_proxy("async.shared", space="cta")
    return x, y, valid & cutlass.Int32(1)


@cute.jit
def scheduler_warp_loop(sched, sched_stages: int):
    """Publish single-CTA CLC responses after every consumer returns slot credit."""
    state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, sched_stages
    )
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.index), state.phase)

        if prims.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.index), 16)
            prims.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.index * cutlass.Int32(8)),
                sched.mb_scheduler.subview(state.index),
                multicast=1,
            )
        prims.fence_proxy("async.shared", space="cta")
        prims.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(state.index), state.phase)
        _m, _n, is_valid = read_clc_payload(
            sched, state.index * cutlass.Int32(8), warp_broadcast=False
        )

        state.advance()
