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

"""Native single-kernel MoE orchestration and persistent route launch geometry."""

from functools import cache
from hashlib import sha256
from pathlib import Path

import torch
import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    _amdgcn_dequant_library,
    _amdgcn_intrinsics_library,
    _amdgcn_moe_library,
    amdgcn_readfirstlane,
    kWarpSize,
)
from lib.moe.rocm.memory_ops import InputLayout, MakeBufferResource
from lib.streams import native_stream
from triton.experimental.gluon import language as l


class OnestageFusedMoEBlockScaleFP8:
    __triton_builtin__ = True
    kGroupM = 32
    kGroupDim = 256
    kNumWarps = 4
    kThreads = kNumWarps * kWarpSize.value
    kScaleBlockSize = 128
    kTokenBatch = 8
    kSubGroupSize = 16
    kRoutesPerBlock = kTokenBatch * kNumWarps
    kRefBufferRange = 0xFFFFFFF0

    @g.jit
    def LoadSortedWeights(sorted_weights, tid, route_base):
        br = MakeBufferResource(
            sorted_weights + route_base, l.full((), 0xFFFFFFF0, l.uint32)
        )
        zero = l.full((), 0, l.int32)
        x = BufferResource.LoadU32(br, (tid % 16) * 4, zero, 0).to(
            l.float32, bitcast=True
        )
        y = BufferResource.LoadU32(br, (tid % 16 + 16) * 4, zero, 0).to(
            l.float32, bitcast=True
        )
        return x, y

    @g.jit
    def QuantizeAndShuffle(shm_max, shm_q, tid, wid, wtid, h, Trait: l.constexpr):
        ReadLayout: l.constexpr = Trait.QuantizationShuffleReadLayout
        return Trait.QuantizeAndShuffleOp.Run(
            shm_max, shm_q, h, tid, wid, wtid, ReadLayout
        )

    @g.jit
    def Stage1(
        shm,
        inp,
        w1,
        w3,
        dim,
        wid,
        wtid,
        tid,
        token_select,
        tokens,
        m,
        Trait: l.constexpr,
    ):
        StageTrait: l.constexpr = Trait.Stage1Trait
        state = StageTrait.Construct(inp, w1, w3, tid)
        return Trait.Stage1Op.Run(
            shm, state, dim, tid, wid, wtid, token_select, tokens, m, StageTrait
        )

    @g.jit
    def Stage2(
        out,
        shm,
        w2,
        dim,
        quant_h,
        dq_act,
        weights,
        tokens,
        invalid,
        tid,
        wid,
        wtid,
        Trait: l.constexpr,
    ):
        StageTrait: l.constexpr = Trait.Stage2Trait
        state = StageTrait.Construct(w2, tid)
        return Trait.Stage2Op.Run(
            out,
            shm,
            state,
            dim,
            quant_h,
            dq_act,
            weights,
            tokens,
            invalid,
            tid,
            wid,
            wtid,
            StageTrait,
        )

    @g.jit
    def Compute(
        out,
        act,
        w13,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        scales_act,
        scales_w13,
        scales_w2,
        num_valid_ids,
        topk,
        m,
        dim,
        inter_dim,
        num_experts,
        persistent_route_step,
        Trait: l.constexpr,
    ):
        Kernel: l.constexpr = OnestageFusedMoEBlockScaleFP8
        kTokenBatch: l.constexpr = Kernel.kTokenBatch
        n_blocks = dim // Kernel.kScaleBlockSize
        k_blocks = inter_dim // Kernel.kScaleBlockSize
        tid = l.arange(0, Kernel.kThreads, layout=l.BlockedLayout([1], [64], [4], [0]))
        wid = amdgcn_readfirstlane(tid // kWarpSize)
        wtid = tid % kWarpSize
        tile_k = l.program_id(0)
        col_id = tid % Kernel.kSubGroupSize
        X_WORDS: l.constexpr = Trait.Stage1Op.kShmStageWords * Trait.Stage1Op.kStage
        MAX_WORDS: l.constexpr = (
            Trait.QuantizeAndShuffleOp.MaxShm[0] * Trait.QuantizeAndShuffleOp.MaxShm[1]
        )
        Q_WORDS: l.constexpr = Trait.QuantizeAndShuffleOp.Shm[0]
        RET_WORDS: l.constexpr = Trait.Stage2Op.Shm[0] * Trait.Stage2Op.Shm[1]
        TOTAL: l.constexpr = X_WORDS + MAX_WORDS + Q_WORDS + RET_WORDS
        storage = l.allocate_shared_memory(
            l.uint32, [TOTAL], l.SwizzledSharedLayout(1, 1, 1, [0])
        )
        shm = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
        shm_max = (shm + X_WORDS).to(l.pointer_type(l.float32, 3))
        shm_q = shm + X_WORDS + MAX_WORDS
        shm_ret = shm + X_WORDS + MAX_WORDS + Q_WORDS
        num_valid_ids_value = l.load(num_valid_ids)
        num_tokens = l.load(num_valid_ids + 1)
        route_group_limit = (
            num_valid_ids_value + Kernel.kRoutesPerBlock - 1
        ) // Kernel.kRoutesPerBlock
        route_group_begin = l.program_id(1)
        route_group_end = l.where(
            persistent_route_step != 0, route_group_limit, route_group_begin + 1
        )
        route_group_increment = l.where(
            persistent_route_step != 0, persistent_route_step, 1
        )
        w2_offset = l.full((), 0, l.uint32)
        for route_group in range(
            route_group_begin, route_group_end, route_group_increment
        ):
            route_base = route_group * Kernel.kRoutesPerBlock
            if route_group < route_group_limit and route_base < num_valid_ids_value:
                expert_id = l.load(sorted_expert_ids + route_group)
                valid_expert = num_experts == 0 or expert_id < num_experts
                if valid_expert:
                    scale_act_ptr = scales_act + (wid // 2) * m * 4
                    input_value_range = m * dim
                    input_scale_range = (n_blocks - (wid // 2)) * m * 4
                    input_ = Trait.Input.Initialize(
                        act, input_value_range, scale_act_ptr, input_scale_range, dim
                    )
                    w1_, w3_, w2_ = Trait.InitializeWeights(
                        w13,
                        w2,
                        scales_w13,
                        scales_w2,
                        expert_id,
                        tile_k,
                        dim,
                        inter_dim,
                        n_blocks,
                        k_blocks,
                        w2_offset,
                    )
                    sorted_token_br_ = MakeBufferResource(
                        sorted_token_ids + route_base,
                        l.full((), Kernel.kRefBufferRange, l.uint32),
                    )
                    token_select = (
                        l.load(sorted_token_ids + route_base + col_id) & 0x00FFFFFF,
                        l.load(
                            sorted_token_ids
                            + route_base
                            + col_id
                            + Kernel.kSubGroupSize
                        )
                        & 0x00FFFFFF,
                    )
                    tokens = ()
                    for i in l.static_range(kTokenBatch):
                        token_idx = wid + i * 4
                        token = (
                            BufferResource.LoadU32(
                                sorted_token_br_,
                                l.full((), 0, l.int32),
                                token_idx * 4,
                                0,
                            )
                            & 0x00FFFFFF
                        )
                        tokens += (token,)
                    safe_token_select = (
                        l.where(token_select[0] < num_tokens, token_select[0], 0),
                        l.where(token_select[1] < num_tokens, token_select[1], 0),
                    )
                    safe_tokens = ()
                    for i in l.static_range(kTokenBatch):
                        safe_tokens += (l.where(tokens[i] < num_tokens, tokens[i], 0),)
                    route_weights = OnestageFusedMoEBlockScaleFP8.LoadSortedWeights(
                        sorted_weights, tid, route_base
                    )
                    h = OnestageFusedMoEBlockScaleFP8.Stage1(
                        shm,
                        input_,
                        w1_,
                        w3_,
                        dim,
                        wid,
                        wtid,
                        tid,
                        safe_token_select,
                        safe_tokens,
                        m,
                        Trait,
                    )
                    l.barrier()
                    invalid_token_mask = l.full(
                        [256], 0, l.uint32, l.BlockedLayout([1], [64], [4], [0])
                    )
                    for i in l.static_range(kTokenBatch):
                        invalid_token_mask |= (tokens[i] >= num_tokens).to(
                            l.uint32
                        ) << i
                    invalid_token_mask = amdgcn_readfirstlane(invalid_token_mask)
                    quant_h, dq_act = OnestageFusedMoEBlockScaleFP8.QuantizeAndShuffle(
                        shm_max, shm_q, tid, wid, wtid, h, Trait
                    )
                    w2_offset = OnestageFusedMoEBlockScaleFP8.Stage2(
                        out,
                        shm_ret,
                        w2_,
                        dim,
                        quant_h,
                        dq_act,
                        route_weights,
                        safe_tokens,
                        invalid_token_mask,
                        tid,
                        wid,
                        wtid,
                        Trait,
                    )
            l.barrier()
        storage._keep_alive()


@g.jit
def OnestageFusedMoEBlockScaleFP8Compute(
    out,
    act,
    w13,
    w2,
    sorted_token_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    scales_act,
    scales_w13,
    scales_w2,
    m,
    n,
    k,
    num_experts,
    persistent_route_step,
    Config: l.constexpr,
    Trait: l.constexpr,
    SOURCE_KEY: l.constexpr,
):
    OnestageFusedMoEBlockScaleFP8.Compute(
        out,
        act.to(l.pointer_type(l.uint8)),
        w13.to(l.pointer_type(l.uint8)),
        w2.to(l.pointer_type(l.uint8)),
        sorted_token_ids.to(l.pointer_type(l.uint32)),
        sorted_weights.to(l.pointer_type(l.uint32)),
        sorted_expert_ids.to(l.pointer_type(l.uint32)),
        scales_act.to(l.pointer_type(l.uint8)),
        scales_w13.to(l.pointer_type(l.uint8)),
        scales_w2.to(l.pointer_type(l.uint8)),
        num_valid_ids.to(l.pointer_type(l.uint32)),
        topk,
        m,
        n,
        k,
        num_experts,
        persistent_route_step,
        Trait,
    )


def LaunchOnestageFusedMoEBlockScaleFP8(
    out,
    act,
    w13,
    w2,
    sorted_token_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    scales_act,
    scales_w13,
    scales_w2,
    max_num_m_blocks,
    m,
    n,
    k,
    num_experts,
    stream,
    num_persistent_tgs,
    Config,
    Trait,
):
    """Launch one native-shaped grid; return the compiled kernel for inspection."""
    if m == 0 or n == 0 or k == 0 or topk == 0 or max_num_m_blocks == 0:
        return None
    split_k = (k + Config.kGroupDim - 1) // Config.kGroupDim
    routes = max_num_m_blocks
    step = 0
    if num_persistent_tgs > 0:
        routes = min(routes, max(1, (num_persistent_tgs + split_k - 1) // split_k))
        step = routes
    execution_stream = native_stream(stream, out.device)
    with torch.cuda.stream(execution_stream):
        return OnestageFusedMoEBlockScaleFP8Compute[(split_k, routes)](
            out,
            act,
            w13,
            w2,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            scales_act,
            scales_w13,
            scales_w2,
            m,
            n,
            k,
            num_experts,
            step,
            Config,
            Trait,
            _implementation_key(),
            num_warps=4,
            enable_fp_fusion=False,
            extern_libs={
                "petit": _amdgcn_intrinsics_library(4),
                "moe": _amdgcn_moe_library(),
                "petit_dequant": _amdgcn_dequant_library(),
            },
        )


@cache
def _implementation_key():
    """Trait-class methods need an explicit source key in Triton's disk cache."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / name
        for name in (
            "fused_moe.py",
            "memory_ops.py",
            "quantization.py",
            "warp_schedule.py",
            "fused_moe_blockscale_fp8_grid.py",
            "fused_moe_blockscale_fp8_fp4_grid.py",
            "ops/onestage_blockscale_fp8_stage1.py",
            "ops/onestage_blockscale_fp8_stage2.py",
            "ops/onestage_blockscale_quantization.py",
            "ops/onestage_fused_moe_blockscale_fp8.py",
        )
    ]
    paths.append(root.parents[1] / "gemm/rocm/amd_intrinsics.py")
    paths.extend(
        root.parents[1] / "tal/tensor" / name for name in ("layout.py", "stride.py")
    )
    return sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
