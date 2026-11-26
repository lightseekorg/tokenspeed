"""Native MegaMoE solution registry, launch validation, and input quantizer."""

import os
from dataclasses import dataclass
from functools import cache

from lib.moe.rocm.fused_moe import (
    FusedMoEDataType,
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
    MegaMoECombineResidual,
    MegaMoEStage1,
    MegaMoEStage2,
    MegaMoETwoStageCommComputeKernel,
)
from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace
from lib.moe.rocm.mega_moe_config_selector import (
    MegaMoEConfigSelector,
    MegaMoEStage1M64W4Config,
    MegaMoEStage1M64W8Config,
    MegaMoEStage1N128Config,
    MegaMoEStage2N128Config,
    kMegaMoETwoStageMxFp4ClampedSiluSolutionId,
    kMegaMoETwoStageMxFp4KimiSituSolutionId,
    kMegaMoETwoStageMxFp4SiluSolutionId,
    kMegaMoETwoStageMxFp4SolutionId,
)
from lib.moe.rocm.quantization import MxFp4Scale, NativeMxFp4Quantization
from tokenspeed_kernel._triton import gl as l
from tokenspeed_kernel._triton import gluon as g


def _stage1_n128_by_experts() -> bool:
    """C5 selector: ``TOKENSPEED_PETIT_STAGE1_N128_SELECT=experts`` (with
    ``TOKENSPEED_PETIT_STAGE1_N128=1``) chooses the N=128 stage-1 tile from the
    number of distinct experts a rank is estimated to receive in the call
    (``Stage1N128ForGlobalRows``), passed down by the caller the same on every
    rank, instead of this rank's own row count.
    """
    return (
        os.environ.get("TOKENSPEED_PETIT_STAGE1_N128", "0") == "1"
        and os.environ.get("TOKENSPEED_PETIT_STAGE1_N128_SELECT", "rows") == "experts"
    )


kStage1N128ByExperts = _stage1_n128_by_experts()


def _stage1_n128_max_tokens() -> int:
    """C5 switch: per-rank row counts up to this launch the N=128 stage-1 tile.

    ``TOKENSPEED_PETIT_STAGE1_N128=1`` turns it on (read once at import) with
    ``TOKENSPEED_PETIT_STAGE1_N128_MAX_ROWS`` rows (default 8: batch-1 DP4,
    batch 1-4 tp4ep4-petit); off (the default) returns -1 and every count
    keeps the 256-column tile. With the received-experts selector the row rule
    is off (-1) and only the caller's ``stage1_n128`` picks the tile.
    """
    if os.environ.get("TOKENSPEED_PETIT_STAGE1_N128", "0") != "1":
        return -1
    if _stage1_n128_by_experts():
        return -1
    return int(os.environ.get("TOKENSPEED_PETIT_STAGE1_N128_MAX_ROWS", "8"))


kStage1N128MaxTokens = _stage1_n128_max_tokens()


def _stage2_n128() -> bool:
    """C5b switch: ``TOKENSPEED_PETIT_STAGE2_N128=1`` (read once at import)
    launches stage 2 with the 32x128 tile (``MegaMoEStage2N128Config``) on
    every call of a bias-free solution; stage 1 and the combine keep their
    configs. Off (the default) launches exactly the old kernels."""
    return os.environ.get("TOKENSPEED_PETIT_STAGE2_N128", "0") == "1"


kStage2N128 = _stage2_n128()


def EstimatedReceivedExperts(num_rows_global, top_k, num_experts, num_ranks):
    """Distinct experts one rank is expected to receive rows for in a call.

    ``num_rows_global`` rows are routed once each (DP4: every rank's rows;
    attention TP: the replicated rows, split over the ranks), each to
    ``top_k`` distinct experts. Under uniform routing an expert is touched
    with probability ``1 - (1 - top_k / num_experts) ** rows``, so a rank's
    ``num_experts / num_ranks`` local experts expect that fraction touched.
    For few rows this is ``rows * top_k / num_ranks`` (no collisions). The
    value depends only on call-wide counts, so every rank gets the same one.
    """
    rows = int(num_rows_global)
    if rows <= 0:
        return 0
    local = num_experts // num_ranks
    touched = 1.0 - (1.0 - top_k / num_experts) ** rows
    return min(local, max(1, int(round(local * touched))))


def ReceivedExpertsBounds(num_rows_global, top_k, num_experts, num_ranks):
    """``(low, high)`` distinct experts one rank receives rows for in a call.

    ``low`` is ``EstimatedReceivedExperts`` (uniform random routing, with
    collisions); ``high`` is the collision-free count
    ``min(local, ceil(rows * top_k / num_ranks))``, which is what routing
    that spreads a call's routes over distinct experts delivers (the
    synthetic ``--routing test`` of the Petit tests and ``petit_layer.py``
    does up to all local experts). Real routing has collisions and skew, so a
    serving call lands at or below ``high``; whether it stays at or above
    ``low`` is what logging ``distinct_local_experts`` in a serving decode
    step must show.
    """
    rows = int(num_rows_global)
    if rows <= 0:
        return 0, 0
    local = num_experts // num_ranks
    high = min(local, -(-rows * top_k // num_ranks))
    low = EstimatedReceivedExperts(rows, top_k, num_experts, num_ranks)
    return min(low, high), high


def Stage1N128PaysForExperts(experts, tiles_per_expert, num_ctas):
    """Whether half-width (N=128) stage-1 tiles take fewer tile rounds.

    At decode counts each touched expert has one 32-row block, so stage 1
    runs ``t = experts * tiles_per_expert`` N=256 tiles (``2 t`` half-width
    tiles) on ``num_ctas`` CTAs, and its time is set by per-CTA tile rounds
    (r0002-pg). N=256 takes ``ceil(t / C)`` full rounds, N=128
    ``ceil(2 t / C)`` half rounds, which is ``2 ceil(t / C) - 1`` (a saving
    of half a round) exactly when ``0 < t mod C <= C / 2``; otherwise both
    take the same time and the larger tile is kept. DSv4.1 Flash at EP4
    (10 tiles per expert, 256 CTAs): N=128 for 1-12, 26-38, 52-64 and 77-89
    experts.
    """
    tiles = int(experts) * int(tiles_per_expert)
    rem = tiles % num_ctas
    return tiles > 0 and 0 < rem <= num_ctas // 2


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
    # If False, the combine copies each token's routes unsummed into out's
    # [topk, hidden] rows.
    reduce_routes: bool
    stream: object = None
    # C5 tile chosen by the caller for the whole call (received-experts
    # selector); None keeps the per-rank row rule.
    stage1_n128: object = None
    # C3 (TOKENSPEED_PETIT_FUSE_POST): a BF16 [num_tokens, hidden] row block
    # the combine adds before its one rounding (may alias out), its row
    # stride in elements, and an event the combine waits for (the shared
    # expert's aux-stream completion) or None.
    residual: object = None
    residual_row_stride: int = 0
    residual_ready: object = None


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
    register(
        kMegaMoETwoStageMxFp4SiluSolutionId,
        8,
        256,
        8,
        4096,
        4096,
        (P.kCta56, P.kCta128, P.kCta192),
    )
    # GLM-5.3 Flash: 36 experts per rank; 288 experts take the default
    # producer rule (cta128 below 12 rows, cta64 below 24, else cta56).
    register(
        kMegaMoETwoStageMxFp4ClampedSiluSolutionId,
        8,
        288,
        8,
        4096,
        2048,
        (P.kCta56, P.kCta64, P.kCta128),
    )
    for ranks in (4, 8):
        for experts, topk in ((384, 6), (128, 3)):
            register(
                kMegaMoETwoStageMxFp4ClampedSiluSolutionId,
                ranks,
                experts,
                topk,
                5120,
                2560,
                (P.kCta56, P.kCta64, P.kCta128, P.kCta192),
            )
    return dict(entries)


class MegaMoESolutionAdapter:
    def __init__(self, solution):
        self.Config = MegaMoEConfigSelector(solution)
        self.Workspace = MegaMoEWorkspace(self.Config)

    @cache
    def Kernels(
        self,
        external,
        m64,
        w8,
        reduce_routes,
        n128=False,
        stage2_n128=False,
        add_residual=False,
    ):
        Config = self.Config
        Stage2 = MegaMoEStage2N128Config(Config) if stage2_n128 else Config
        SelectedStage1 = (
            MegaMoEStage1M64W8Config(Config)
            if w8
            else (
                MegaMoEStage1M64W4Config(Config)
                if m64
                else MegaMoEStage1N128Config(Config) if n128 else Config
            )
        )
        return (
            MegaMoETwoStageCommComputeKernel(SelectedStage1, external),
            MegaMoETwoStageCommComputeKernel(Stage2, external),
            (
                MegaMoECombineKernel(Config, reduce_routes, True)
                if add_residual
                else MegaMoECombineKernel(Config, reduce_routes)
            ),
        )

    def Stage1Tile(self, num_tokens):
        """Return the stage-1 tile flags ``(m64, w8)`` for a per-rank count."""
        C = self.Config
        return num_tokens >= C.kM64MinTokens, num_tokens >= C.kM64W8MinTokens

    def Stage1N128Admits(self, num_tokens):
        """Whether this rank's ``num_tokens`` rows can launch the N=128 tile:
        the 32-row stage-1 tile and a bias-free solution (DSv4.1 Flash and its
        DSpark draft among them; biased solutions keep their tile)."""
        return (
            num_tokens < self.Config.kM64MinTokens
            and self.Config.kSolution.bias_dtype == FusedMoEDataType.kNone
        )

    def Stage1N128(self, num_tokens):
        """Whether ``num_tokens`` rows take the N=128 stage-1 tile (C5, opt-in)."""
        return num_tokens <= kStage1N128MaxTokens and self.Stage1N128Admits(num_tokens)

    def Stage1N128ForGlobalRows(self, num_rows_global):
        """The received-experts selector's call-wide N=128 decision, or None
        when the selector is off (the row rule then decides per rank)."""
        if not kStage1N128ByExperts:
            return None
        C = self.Config
        if C.kSolution.bias_dtype != FusedMoEDataType.kNone:
            return False
        # N=128 only when the tile-round rule takes it at every received-expert
        # count the call can deliver, from the collision estimate to the
        # collision-free count: the rule flips every 13 experts above one
        # round, and a wrong N=128 costs more than a right one saves (R=64:
        # estimate 61 -> N=128, delivered 96 -> +13.2 us per call, r0002-f3).
        low, high = ReceivedExpertsBounds(
            num_rows_global, C.kTopK, C.kNumExperts, C.kNumRanks
        )
        return high > 0 and all(
            Stage1N128PaysForExperts(e, C.kInterDim // C.kStage1GroupN, C.kNumSMs)
            for e in range(low, high + 1)
        )

    def Stage2N128(self):
        """Whether stage 2 takes the 32x128 tile (C5b, opt-in): the switch and
        a bias-free solution (biased solutions keep their tile)."""
        return (
            kStage2N128 and self.Config.kSolution.bias_dtype == FusedMoEDataType.kNone
        )

    def SelectStage1N128(self, num_tokens, stage1_n128):
        """The tile a launch takes: the caller's call-wide choice when given
        (still only where this rank's count admits it), else the row rule."""
        if stage1_n128 is None:
            return self.Stage1N128(num_tokens)
        return bool(stage1_n128) and self.Stage1N128Admits(num_tokens)

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

    def Invoke(self, params, compile_only=False):
        """Launch the three kernels, or with ``compile_only`` only compile them.

        ``compile_only`` builds exactly the specializations a launch with the
        same params would (tile, producer geometry, rank, Triton's integer
        classes of ``num_tokens``) and launches nothing, so it is not a
        collective and may run on one rank alone.
        """
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
        m64, w8 = self.Stage1Tile(p.num_tokens)
        # C5b: only with the switch on does the call name stage2_n128, so off
        # keeps the old Kernels cache entries and kernels.
        stage2_kw = {"stage2_n128": True} if self.Stage2N128() else {}
        if p.residual is not None:
            # C3: only reduced rows take a residual; the stage kernels are
            # the same, the combine is its residual variant.
            if not p.reduce_routes or p.residual_row_stride <= 0:
                return kFusedMoEErrorInvalidArgument
            stage2_kw["add_residual"] = True
        if self.SelectStage1N128(p.num_tokens, p.stage1_n128):
            stage1, _, _ = self.Kernels(
                external_input_count == 3, m64, w8, p.reduce_routes, n128=True
            )
            _, stage2, combine = self.Kernels(
                external_input_count == 3, m64, w8, p.reduce_routes, **stage2_kw
            )
        else:
            stage1, stage2, combine = self.Kernels(
                external_input_count == 3, m64, w8, p.reduce_routes, **stage2_kw
            )
        # Combine always runs the base config (stage2_n128 replaces stage 2 only).
        assert combine.Config is C
        import torch
        from lib.streams import native_stream

        def Launch(kernel, grid):
            return kernel.warmup if compile_only else kernel[grid]

        def Grid(blocks):
            # warmup() takes the grid as a keyword; a launch takes it above.
            return {"grid": blocks} if compile_only else {}

        with torch.cuda.stream(native_stream(p.stream, p.workspace.device)):
            Launch(MegaMoEStage1, (C.kNumSMs,))(
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
                num_warps=stage1.kNumWarps,
                enable_fp_fusion=False,
                **Grid((C.kNumSMs,)),
            )
            Launch(MegaMoEStage2, (stage2.kStage2GridBlocks,))(
                p.w2,
                p.scales_w2,
                p.w2_bias,
                p.workspace,
                p.rank,
                stage2,
                num_warps=stage2.kNumWarps,
                enable_fp_fusion=False,
                **Grid((stage2.kStage2GridBlocks,)),
            )
            if p.residual is None:
                Launch(MegaMoECombine, (combine.kNumSMs,))(
                    p.out,
                    p.num_tokens,
                    p.output_row_stride,
                    p.workspace,
                    p.rank,
                    combine,
                    num_warps=combine.kNumWarps,
                    enable_fp_fusion=False,
                    **Grid((combine.kNumSMs,)),
                )
            else:
                assert combine.kAddResidual
                if p.residual_ready is not None and not compile_only:
                    # The shared expert runs on the aux stream: its rows
                    # must be complete before the combine reads them.
                    torch.cuda.current_stream().wait_event(p.residual_ready)
                Launch(MegaMoECombineResidual, (combine.kNumSMs,))(
                    p.out,
                    p.num_tokens,
                    p.output_row_stride,
                    p.residual,
                    p.residual_row_stride,
                    p.workspace,
                    p.rank,
                    combine,
                    num_warps=combine.kNumWarps,
                    enable_fp_fusion=False,
                    **Grid((combine.kNumSMs,)),
                )
        return 0


@cache
def _MegaMoESolution(solution_id):
    """Return the adapter of a registered solution at any admitted capacity.

    The registry holds each solution at the default per-rank token capacity;
    a larger capacity (FusedMoESolutionId.MaxTokensPerRank) reuses the
    registered solution with buffers sized for it. Returns None for an
    unregistered solution and raises ValueError when the capacity's
    workspace does not fit 32-bit offsets.
    """
    solutions = _MegaMoESolutions()
    adapter = solutions.get(solution_id)
    if adapter is not None:
        return adapter
    solution = FusedMoESolutionId.FromRepr(solution_id)
    default = solution.WithMaxTokensPerRank(
        FusedMoESolutionId.kDefaultMegaMaxTokensPerRank
    )
    if default.Repr() == solution_id or default.Repr() not in solutions:
        return None
    try:
        return MegaMoESolutionAdapter(solution)
    except AssertionError as exc:
        raise ValueError(
            f"MegaMoE workspace for {solution.MaxTokensPerRank()} tokens per rank "
            f"at {1 << solution.NumRanksLog2()} ranks and "
            f"{solution.NumExperts()} experts exceeds 32-bit offsets"
        ) from exc


def MegaMoECompute(params, solution_id, compile_only=False):
    """Launch MegaMoE, or with ``compile_only`` compile its kernels only."""
    adapter = _MegaMoESolution(int(solution_id))
    if adapter is None:
        return kFusedMoEErrorInvalidSolution
    if not IsGfx950(params.stream, params.workspace.device):
        return kFusedMoEErrorUnsupported
    return adapter.Invoke(params, compile_only)


def MegaMoEStage1N128ForGlobalRows(solution_id, num_rows_global):
    """The received-experts selector's N=128 choice for a call routing
    ``num_rows_global`` rows (None when the selector is off)."""
    adapter = _MegaMoESolution(int(solution_id))
    if adapter is None:
        raise ValueError("unsupported MegaMoE solution_id")
    return adapter.Stage1N128ForGlobalRows(int(num_rows_global))


def GetMegaMoEWorkspaceInfo(rank, solution_id):
    adapter = _MegaMoESolution(int(solution_id))
    if adapter is None:
        raise ValueError("unsupported MegaMoE solution_id")
    return adapter.GetWorkspaceInfo(rank)


def MegaMoEStage1Tile(solution_id, num_tokens):
    """Name the stage-1 tile one rank launches for ``num_tokens`` rows."""
    adapter = _MegaMoESolution(int(solution_id))
    if adapter is None:
        raise ValueError("unsupported MegaMoE solution_id")
    m64, w8 = adapter.Stage1Tile(int(num_tokens))
    if adapter.Stage1N128(int(num_tokens)):
        return "m32_n128"
    return "m64_w8" if w8 else "m64_w4" if m64 else "m32"


def MegaMoEHeapBytes(solution_id):
    """Return one rank's VMM heap ``(reserved, physical)`` bytes.

    Mirrors VmmSymmetricHeap::Allocate: every region is 4 KiB aligned, the
    reserved range is ``local_offset + local_bytes`` (which must stay below
    2**32), and one rank backs its barrier record, its rank slot, the padding
    up to ``local_offset`` and its local data with physical memory.
    """
    info = GetMegaMoEWorkspaceInfo(0, solution_id)

    def page(value):
        return (value + 4095) // 4096 * 4096

    barrier, slot, local = (
        page(info.barrier_record_bytes),
        page(info.rank_slot_bytes),
        page(info.local_bytes),
    )
    shared_end = info.rank_sym_buffer_base + info.num_ranks * slot
    reserved = info.local_offset + local
    physical = barrier + slot + (info.local_offset - shared_end) + local
    return reserved, physical


@g.jit
def MegaMoEQuantizeMxFp4Kernel(input, output, groups_per_row, input_row_stride):
    groups_per_row = (l.full((), 0, l.uint32) + groups_per_row).to(l.uint32)
    input_row_stride = (l.full((), 0, l.uint32) + input_row_stride).to(l.uint32)
    # Each lane owns one 32-element quantization group, as in the original
    # thread implementation. The trailing dimension stays in lane registers.
    layout: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [1, 1], [1, 0])
    lane = l.arange(0, 64, layout=l.SliceLayout(1, layout))
    word = l.arange(0, 4, layout=l.SliceLayout(0, layout))
    group_col = l.program_id(0) * 64 + lane
    row = l.program_id(1)
    scale_row_stride = (groups_per_row + 15) & 0xFFFFFFF0
    value_row_bytes = groups_per_row * 16
    row_output = output + row * (value_row_bytes + scale_row_stride)
    row_input = (input + row * input_row_stride).to(l.pointer_type(l.uint32))
    valid = group_col < groups_per_row
    values = ()
    for chunk in l.static_range(4):
        packed = l.load(
            row_input + group_col[:, None] * 16 + chunk * 4 + word[None, :],
            mask=valid[:, None],
            other=0,
        )
        even, odd = l.split(packed.reshape([64, 2, 2]))
        x, z = l.split(even)
        y, w = l.split(odd)
        words = (
            l.convert_layout(x, l.SliceLayout(1, layout)),
            l.convert_layout(y, l.SliceLayout(1, layout)),
            l.convert_layout(z, l.SliceLayout(1, layout)),
            l.convert_layout(w, l.SliceLayout(1, layout)),
        )
        pairs = ()
        for i in l.static_range(4):
            pairs += (
                (
                    (words[i] << 16).to(l.float32, bitcast=True),
                    (words[i] & 0xFFFF0000).to(l.float32, bitcast=True),
                ),
            )
        values += (
            (pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]),
            (pairs[2][0], pairs[2][1], pairs[3][0], pairs[3][1]),
        )
    max_abs = l.full([64], 0.0, l.float32, l.SliceLayout(1, layout))
    for vector in l.static_range(8):
        max_abs = l.maximum(max_abs, NativeMxFp4Quantization.MaximumAbs(values[vector]))
    required_bits = (max_abs * (1.0 / 6.0)).to(l.uint32, bitcast=True)
    scale_byte = (required_bits >> 23) & 0xFF
    scale_byte += ((scale_byte < 0xFF) & ((required_bits & 0x7FFFFF) != 0)).to(l.uint32)
    scale_byte = l.where(max_abs == 0, 0, scale_byte)
    # A unit scale avoids division by zero for empty groups; their packed
    # representation and scale byte are explicitly zeroed below.
    scale_bits = l.where(max_abs == 0, 127, scale_byte) << 23
    scale = MxFp4Scale(scale_byte, scale_bits.to(l.float32, bitcast=True))
    packed = NativeMxFp4Quantization.Pack(values, scale, 8)
    result = l.join(l.join(packed[0], packed[2]), l.join(packed[1], packed[3]))
    result = l.convert_layout(result.reshape([64, 4]), layout)
    result = l.where((max_abs != 0)[:, None], result, 0)
    l.store(
        row_output.to(l.pointer_type(l.uint32))
        + group_col[:, None] * 4
        + word[None, :],
        result,
        mask=valid[:, None],
    )
    l.store(
        row_output + value_row_bytes + group_col,
        scale_byte,
        mask=group_col < scale_row_stride,
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
