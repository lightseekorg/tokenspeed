# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native double- and single-buffer up/gate projection procedures."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import amdgcn_s_waitcnt_barrier
from lib.moe.rocm.fused_moe import ClearMat, HotLoopScheduler, SiluDot
from lib.moe.rocm.memory_ops import (
    InputLayout,
    _uninitialized_like,
    _uninitialized_uint4_array,
)
from triton.experimental.gluon import language as l


class FusedMoEBlockScaleFP8Stage1DoubleBufferOp:
    __triton_builtin__ = True
    kStage = 2
    Input = InputLayout
    kGroupDim = Input.kGroupDim
    kTokenBatch = Input.kTokenBatch
    kAccumFragments = 8
    kActivationFragments = Input.kActivationFragments
    # Native Shm::x[kStage] { act[kShmInputElements], scale[kThreads] }.
    kShmStageWords = Input.kShmInputElements + Input.kThreads
    Shm = (kStage, (Input.kShmInputElements, Input.kThreads))

    @g.jit
    def Run(
        shm, state, dim, tid, wid, wtid, token_select, tokens, m, Trait: l.constexpr
    ):
        Op: l.constexpr = FusedMoEBlockScaleFP8Stage1DoubleBufferOp
        Input: l.constexpr = Trait.Input
        kStage: l.constexpr = Op.kStage
        kAccumFragments: l.constexpr = Trait.kAccumFragments
        kActivationFragments: l.constexpr = Trait.kActivationFragments
        t_gate = ClearMat(tid, kAccumFragments)
        t_up = ClearMat(tid, kAccumFragments)
        tile = _uninitialized_uint4_array(tid, kActivationFragments)
        x = (tile,) * kStage
        scale = (_uninitialized_like(tid, l.float32),) * 4
        scale_x = (scale,) * kStage
        state = Trait.PrefetchInput(
            state,
            shm,
            shm + Input.kShmInputElements,
            wid,
            wtid,
            token_select,
            tokens,
            m,
        )
        state = Trait.LoadInitial(state, tid, wid, wtid)
        amdgcn_s_waitcnt_barrier(0)
        tile, scale = Trait.ReadInput(shm, shm + Input.kShmInputElements, wtid)
        x = (tile, x[1])
        scale_x = (scale, scale_x[1])
        for d in range(0, dim, 2 * Op.kGroupDim):
            l.barrier()
            for curr in l.static_range(kStage):
                # Gluon requires constexpr tuple indices and cannot break an
                # unrolled loop. This guard implements native's post-Matmul break.
                if curr == 0 or d + Op.kGroupDim < dim:
                    HotLoopScheduler(128, 6, 0, 0, 2)
                    p = shm + (1 - curr) * Op.kShmStageWords
                    state = Trait.PrefetchInput(
                        state,
                        p,
                        p + Input.kShmInputElements,
                        wid,
                        wtid,
                        token_select,
                        tokens,
                        m,
                    )
                    t_gate, t_up, state = Trait.Matmul(
                        t_gate, t_up, x[curr], scale_x[curr], state, tid, wid, wtid
                    )
                    if d + Op.kGroupDim < dim:
                        amdgcn_s_waitcnt_barrier(0)
                        tile, scale = Trait.ReadInput(
                            p, p + Input.kShmInputElements, wtid
                        )
                        x = x[: 1 - curr] + (tile,) + x[2 - curr :]
                        scale_x = scale_x[: 1 - curr] + (scale,) + scale_x[2 - curr :]
        h = ()
        for i in l.static_range(kAccumFragments):
            h += (SiluDot(t_gate[i], t_up[i]),)
        return h


class FusedMoEBlockScaleFP8Stage1SingleBufferOp:
    __triton_builtin__ = True
    kStage = 1
    Input = InputLayout
    kGroupDim = Input.kGroupDim
    kTokenBatch = Input.kTokenBatch
    kAccumFragments = 8
    kActivationFragments = Input.kActivationFragments
    kShmStageWords = Input.kShmInputElements + Input.kThreads
    Shm = (kStage, (Input.kShmInputElements, Input.kThreads))

    @g.jit
    def Run(
        shm, state, dim, tid, wid, wtid, token_select, tokens, m, Trait: l.constexpr
    ):
        Op: l.constexpr = FusedMoEBlockScaleFP8Stage1SingleBufferOp
        Input: l.constexpr = Trait.Input
        kStage: l.constexpr = Op.kStage
        kAccumFragments: l.constexpr = Trait.kAccumFragments
        t_gate = ClearMat(tid, kAccumFragments)
        t_up = ClearMat(tid, kAccumFragments)
        state = Trait.LoadInitial(state, tid, wid, wtid)
        for d in range(0, dim, Op.kGroupDim):
            state = Trait.PrefetchInput(
                state,
                shm,
                shm + Input.kShmInputElements,
                wid,
                wtid,
                token_select,
                tokens,
                m,
            )
            amdgcn_s_waitcnt_barrier(0)
            x, scale_x = Trait.ReadInput(shm, shm + Input.kShmInputElements, wtid)
            t_gate, t_up, state = Trait.Matmul(
                t_gate, t_up, x, scale_x, state, tid, wid, wtid
            )
        h = ()
        for i in l.static_range(kAccumFragments):
            h += (SiluDot(t_gate[i], t_up[i]),)
        return h
