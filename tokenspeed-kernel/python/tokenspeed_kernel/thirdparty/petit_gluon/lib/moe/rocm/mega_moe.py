"""Native MegaMoE solution registry, launch validation, and input quantizer."""

from dataclasses import dataclass
from functools import cache

import triton.experimental.gluon as g
from lib.moe.rocm.fused_moe import (
    FusedMoESolutionId,
    IsGfx950,
    MegaMoEProducerGeometry,
    MegaMoETileShape,
    kFusedMoEErrorInvalidArgument,
    kFusedMoEErrorInvalidSolution,
    kFusedMoEErrorUnsupported,
)
from lib.moe.rocm.mega_moe.mega_moe_two_stage_kernel import (
    MegaMoECombine,
    MegaMoECombineKernel,
    MegaMoEStage1,
    MegaMoEStage2,
    MegaMoETwoStageCommComputeKernel,
)
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.moe.rocm.mega_moe_config_selector import (
    MegaMoEConfigSelector,
    MegaMoEStage1M64W4Config,
    MegaMoEStage1M64W8Config,
    kMegaMoETwoStageMxFp4KimiSituSolutionId,
    kMegaMoETwoStageMxFp4SiluSolutionId,
    kMegaMoETwoStageMxFp4SolutionId,
)
from lib.moe.rocm.memory_ops import _load_vector4, _store_vector4
from lib.moe.rocm.quantization import MxFp4Scale, NativeMxFp4Quantization
from lib.tal.thread_jit import thread_jit
from triton.experimental.gluon import language as l

kMegaMoEProfileCtas, kMegaMoEProfileCounterCount = 3072, 35


@dataclass
class MegaMoEParams:
    out: object
    output_row_stride: int
    w13: object
    w2: object
    scales_w13: object
    scales_w2: object
    input_tokens: object
    input_topk_ids: object
    input_topk_weights: object
    num_tokens: int
    hidden_size: int
    inter_dim: int
    w13_bias: object
    w2_bias: object
    workspace: object
    rank: int
    stream: object = None
    profile: object = None


@dataclass(frozen=True)
class MegaMoEWorkspaceInfo:
    barrier_record_bytes: int
    rank_sym_buffer_base: int
    rank_slot_bytes: int
    local_offset: int
    local_bytes: int
    input_tokens_offset: int
    input_topk_expert_id_offset: int
    input_topk_expert_weight_offset: int
    input_token_bytes: int
    max_tokens_per_rank: int
    num_ranks: int
    num_experts: int
    topk: int
    hidden_size: int
    compute_hidden_size: int
    act_dtype: object


@cache
def _MegaMoESolutions():
    entries = []

    def register(base, ranks, experts, topk, hidden, inter, producers):
        for producer in producers:
            solution = base.WithMegaMoEConfig(
                ranks, experts, topk, hidden, inter, producer, MegaMoETileShape.kN256
            )
            entries.append((solution.Repr(), MegaMoESolutionAdapter(solution)))

    P = MegaMoEProducerGeometry
    register(
        kMegaMoETwoStageMxFp4KimiSituSolutionId, 8, 896, 16, 3584, 3072, (P.kCta56,)
    )
    for ranks in (2, 4, 8):
        register(kMegaMoETwoStageMxFp4SolutionId, ranks, 32, 4, 2880, 3072, (P.kCta56,))
    register(
        kMegaMoETwoStageMxFp4SolutionId,
        8,
        128,
        4,
        2880,
        3072,
        (P.kCta56, P.kCta64, P.kCta128),
    )
    register(
        kMegaMoETwoStageMxFp4SiluSolutionId,
        8,
        256,
        8,
        7168,
        2048,
        (P.kCta56, P.kCta128, P.kCta192),
    )
    register(
        kMegaMoETwoStageMxFp4SiluSolutionId,
        8,
        384,
        6,
        7168,
        3072,
        (P.kCta56, P.kCta192),
    )
    return dict(entries)


class MegaMoESolutionAdapter:
    def __init__(self, solution):
        self.Config = MegaMoEConfigSelector(solution)
        self.Workspace = MegaMoEWorkspace(self.Config)

    @cache
    def Kernels(self, external, profile, m64, w8):
        Config = self.Config
        SelectedStage1 = (
            MegaMoEStage1M64W8Config(Config)
            if w8
            else MegaMoEStage1M64W4Config(Config) if m64 else Config
        )
        return (
            MegaMoETwoStageCommComputeKernel(SelectedStage1, external, profile),
            MegaMoETwoStageCommComputeKernel(Config, external, profile),
            MegaMoECombineKernel(Config, profile),
        )

    def GetWorkspaceInfo(self, rank):
        W, C = self.Workspace, self.Config
        return MegaMoEWorkspaceInfo(
            W.XGpuBarrierRecordBytes(),
            W.RankSymBufferBase(),
            W.RankSymBufferSlotBytes(),
            W.LocalOffsetBase(),
            W.kLocalBytes,
            W.InputTokensOffset(),
            W.InputTokenTopKExpertIDOffset(),
            W.InputTokenTopKExpertWeightOffset(),
            C.kInputTokenBytes,
            C.kMaxTokensPerRank,
            C.kNumRanks,
            C.kNumExperts,
            C.kTopK,
            C.kHiddenSize,
            C.kComputeHiddenSize,
            C.kActDType,
        )

    def Invoke(self, params):
        C, p = self.Config, params
        external_input_count = sum(
            x is not None
            for x in (p.input_tokens, p.input_topk_ids, p.input_topk_weights)
        )
        if (
            (p.num_tokens != 0 and p.out is None)
            or any(
                x is None for x in (p.w13, p.w2, p.scales_w13, p.scales_w2, p.workspace)
            )
            or p.hidden_size != C.kComputeHiddenSize
            or p.output_row_stride not in (C.kHiddenSize, C.kComputeHiddenSize)
            or p.inter_dim != C.kInterDim
            or external_input_count not in (0, 3)
        ):
            return kFusedMoEErrorInvalidArgument
        kM64MinTokens = 128 if C.kNumExperts == 256 else 256
        kM64W8MinTokens = 1024 if C.kNumExperts == 128 else kM64MinTokens
        stage1, stage2, combine = self.Kernels(
            external_input_count == 3,
            p.profile is not None,
            p.num_tokens >= kM64MinTokens,
            p.num_tokens >= kM64W8MinTokens,
        )
        import torch
        from lib.streams import native_stream

        with torch.cuda.stream(native_stream(p.stream, p.workspace.device)):
            MegaMoEStage1[(C.kNumSMs,)](
                p.w13,
                p.scales_w13,
                p.num_tokens,
                p.w13_bias,
                p.workspace,
                p.rank,
                p.input_tokens,
                p.input_topk_ids,
                p.input_topk_weights,
                stage1,
                p.profile,
                num_warps=stage1.kNumWarps,
                enable_fp_fusion=False,
            )
            MegaMoEStage2[(stage2.kStage2GridBlocks,)](
                p.w2,
                p.scales_w2,
                p.w2_bias,
                p.workspace,
                p.rank,
                stage2,
                p.profile,
                num_warps=stage2.kNumWarps,
                enable_fp_fusion=False,
            )
            MegaMoECombine[(combine.kNumSMs,)](
                p.out,
                p.num_tokens,
                p.output_row_stride,
                p.workspace,
                p.rank,
                combine,
                p.profile,
                num_warps=combine.kNumWarps,
                enable_fp_fusion=False,
            )
        return 0


def MegaMoECompute(params, solution_id):
    adapter = _MegaMoESolutions().get(int(solution_id))
    if adapter is None:
        return kFusedMoEErrorInvalidSolution
    if not IsGfx950(params.stream, params.workspace.device):
        return kFusedMoEErrorUnsupported
    return adapter.Invoke(params)


def GetMegaMoEWorkspaceInfo(rank, solution_id):
    adapter = _MegaMoESolutions().get(int(solution_id))
    if adapter is None:
        raise ValueError("unsupported MegaMoE solution_id")
    return adapter.GetWorkspaceInfo(rank)


@thread_jit
def _quantize_thread(input, output, groups_per_row, input_row_stride, group_col, row):
    groups_per_row = (l.full((), 0, l.uint32) + groups_per_row).to(l.uint32)
    input_row_stride = (l.full((), 0, l.uint32) + input_row_stride).to(l.uint32)
    scale_row_stride = (groups_per_row + 15) & 0xFFFFFFF0
    value_row_bytes = groups_per_row * 16
    output_row_stride = value_row_bytes + scale_row_stride
    if group_col < scale_row_stride:
        row_output = output + row * output_row_stride
        row_values = row_output.to(l.pointer_type(l.uint32))
        row_scales = row_output + value_row_bytes
        if group_col >= groups_per_row:
            l.store(row_scales + group_col, 0)
        else:
            row_input = input + row * input_row_stride
            src = row_input.to(l.pointer_type(l.uint32)) + group_col * 16
            values = ()
            for chunk in l.static_range(4):
                packed = _load_vector4(src + chunk * 4)
                pairs = ()
                for i in l.static_range(4):
                    pairs += (
                        (
                            (packed[i] << 16).to(l.float32, bitcast=True),
                            (packed[i] & 0xFFFF0000).to(l.float32, bitcast=True),
                        ),
                    )
                values += (
                    (pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]),
                    (pairs[2][0], pairs[2][1], pairs[3][0], pairs[3][1]),
                )
            max_abs = l.full((), 0.0, l.float32)
            for vector in l.static_range(8):
                max_abs = l.maximum(
                    max_abs, NativeMxFp4Quantization.MaximumAbs(values[vector])
                )
            if max_abs == 0.0:
                _store_vector4(row_values + group_col * 4, (0, 0, 0, 0))
                l.store(row_scales + group_col, 0)
            else:
                required = max_abs * (1.0 / 6.0)
                required_bits = required.to(l.uint32, bitcast=True)
                scale_byte = (required_bits >> 23) & 0xFF
                if scale_byte < 0xFF and (required_bits & 0x7FFFFF) != 0:
                    scale_byte += 1
                scale_bits = scale_byte << 23
                scale = MxFp4Scale(scale_byte, scale_bits.to(l.float32, bitcast=True))
                packed = NativeMxFp4Quantization.Pack(values, scale, 8)
                _store_vector4(row_values + group_col * 4, packed)
                l.store(row_scales + group_col, scale_byte)


@g.jit
def MegaMoEQuantizeMxFp4Kernel(input, output, groups_per_row, input_row_stride):
    tid = l.arange(0, 64, layout=l.BlockedLayout([1], [64], [1], [0])).to(l.uint32)
    group_col = l.program_id(0).to(l.uint32) * 64 + tid
    _quantize_thread(
        input,
        output,
        groups_per_row,
        input_row_stride,
        group_col,
        l.program_id(1).to(l.uint32),
    )


def MegaMoEQuantizeMxFp4(input, output, rows, cols, input_row_stride, stream=None):
    if (
        cols == 0
        or cols % 32
        or input_row_stride < cols
        or input_row_stride % 8
        or (rows and (input is None or output is None))
    ):
        return kFusedMoEErrorInvalidArgument
    if rows == 0:
        return 0
    if not IsGfx950(stream, input.device):
        return kFusedMoEErrorUnsupported
    import torch
    from lib.streams import native_stream

    with torch.cuda.stream(native_stream(stream, input.device)):
        MegaMoEQuantizeMxFp4Kernel[((cols // 32 + 63) // 64, rows)](
            input,
            output,
            cols // 32,
            input_row_stride,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return 0
