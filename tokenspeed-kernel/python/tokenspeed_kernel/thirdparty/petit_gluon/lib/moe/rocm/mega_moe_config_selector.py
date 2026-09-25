"""Native solution selectors and registrations."""

from lib.moe.rocm.fused_moe import (
    FusedMoEActivationFunction,
    FusedMoEDataType,
    FusedMoEMfmaShape,
    FusedMoESolutionId,
    FusedMoEStage1Buffering,
    FusedMoEStage1TileShape,
    FusedMoEStages,
    FusedMoEWeightLoadPolicy,
    FusedMoEWeightOrdering,
)

kMegaMoETwoStageMxFp4SolutionId = FusedMoESolutionId.MakeMegaBase(
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kBf16,
    FusedMoEWeightOrdering.kNativeMxFp4,
    FusedMoEMfmaShape.kMfmaScaleFp4MxFp4,
    FusedMoEStages.kTwoStage,
    FusedMoEActivationFunction.kOpenAISwiGLU,
    FusedMoEStage1Buffering.kDoubleBuffer,
)

kMegaMoETwoStageMxFp4SiluSolutionId = FusedMoESolutionId.MakeMegaBase(
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kNone,
    FusedMoEWeightOrdering.kNativeMxFp4,
    FusedMoEMfmaShape.kMfmaScaleFp4MxFp4,
    FusedMoEStages.kTwoStage,
    FusedMoEActivationFunction.kSiluDot,
    FusedMoEStage1Buffering.kDoubleBuffer,
)

kMegaMoETwoStageMxFp4KimiSituSolutionId = FusedMoESolutionId.MakeMegaBase(
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kNone,
    FusedMoEWeightOrdering.kNativeMxFp4,
    FusedMoEMfmaShape.kMfmaScaleFp4MxFp4,
    FusedMoEStages.kTwoStage,
    FusedMoEActivationFunction.kKimiSitu,
    FusedMoEStage1Buffering.kDoubleBuffer,
)


from lib.moe.rocm.fused_moe import MegaMoETileShape
from lib.moe.rocm.mem.bias import (
    Bf16BiasLayout,
    MxFp4BiasLayout,
    MxFp4BiasLayoutM64N256W8,
    NoopBiasLayout,
)
from lib.moe.rocm.mem.input_mxfp4_packed import MxFp4InputPacked
from lib.moe.rocm.mem.weight_mxfp4 import MxFp4Weights
from lib.moe.rocm.memory_ops import BufferResource, MxFp4TileShape
from lib.moe.rocm.ops.activation import KimiSituOp, OpenAISwiGLUOp, SiluDotOp
from lib.moe.rocm.ops.op_stages import (
    OnestageFusedMoEStage1DoubleBufferOp,
    OnestageFusedMoEStage1SingleBufferOp,
    W2TileSchedule,
    W13TileSchedule,
)
from lib.moe.rocm.ops.schedule_tiles import NativeMxFp4TileOps
from lib.tal.device import DeviceTemplate


class MegaMoEConfigSelector(DeviceTemplate):
    def __init__(self, Solution):
        self._key = Solution.Repr()
        self.kSolution = Solution
        self.kGroupM, self.kStage1GroupN = 32, 256
        self.kGroupN = 128 if Solution.W2TileShape() == MegaMoETileShape.kN128 else 256
        self.kGroupDim, self.kTokenBatch = 256, 8
        self.kStage2GroupM, self.kStage2TokenBatch = 32, 8
        self.kNumWarps, self.kStage1WarpsM, self.kStage1WarpsN = 4, 1, 4
        self.kThreads, self.kNumSMs, self.kGridSyncSlots = 256, 256, 5
        self.kStage2GroupInterDim = self.kGroupDim
        self.kNumRanks, self.kNumExperts, self.kTopK = (
            1 << Solution.NumRanksLog2(),
            Solution.NumExperts(),
            Solution.TopK(),
        )
        self.kHiddenSize = Solution.HiddenSizeDiv64() * 64
        self.kComputeHiddenSize = (self.kHiddenSize + 511) // 512 * 512
        self.kDim, self.kInterDim = (
            self.kComputeHiddenSize,
            Solution.MegaInterDimDiv64() * 64,
        )
        self.kProducerBlocks = min(Solution.ProducerBlocks(), self.kNumExperts)
        self.kMaxTokensPerRank, self.kSortedTokenBlock, self.kWeightScaleBlockSize = (
            1024,
            32,
            128,
        )
        self.kWeightLoadAux = BufferResource.kNone
        self.kInputTokenBytes = (
            (self.kHiddenSize // 2 + self.kHiddenSize // 32 + 15) // 16 * 16
        )
        self.kMaxExpertsPerToken = min(self.kTopK, self.kNumExperts // self.kNumRanks)
        self.kRouteOutputBufferBytes = self.kTopK * self.kHiddenSize * 2
        self.kActDType, self.kWeightDType, self.kMfmaShape = (
            Solution.act_dtype,
            Solution.weight_dtype,
            Solution.mfma,
        )
        self.kValidateExpertIds = False
        self.kW13TileShape = MxFp4TileShape.kN256
        self.kW2TileShape = (
            MxFp4TileShape.kN128
            if Solution.W2TileShape() == MegaMoETileShape.kN128
            else MxFp4TileShape.kN256
        )
        assert self.kNumExperts and self.kTopK and self.kHiddenSize and self.kInterDim
        assert (
            Solution.stages == FusedMoEStages.kTwoStage
            and self.kActDType == FusedMoEDataType.kMxFp4
        )
        assert self.kNumRanks in (2, 4, 8) and self.kNumExperts % self.kNumRanks == 0
        assert self.kTopK <= self.kNumExperts and self.kInterDim % 512 == 0
        self.ActivationOp = {
            FusedMoEActivationFunction.kSiluDot: SiluDotOp,
            FusedMoEActivationFunction.kOpenAISwiGLU: OpenAISwiGLUOp,
            FusedMoEActivationFunction.kKimiSitu: KimiSituOp,
        }[Solution.activation]
        self.Input, self.Weight = MxFp4InputPacked(self), MxFp4Weights(self)
        self.W13Weights, self.W2Weights = self.Weight.W13Weights, self.Weight.W2Weights
        self.W13, self.W2 = self.Weight.W13, self.Weight.W2
        self.Bias = (
            NoopBiasLayout(4, 256)
            if Solution.bias_dtype == FusedMoEDataType.kNone
            else Bf16BiasLayout(4, 256, MxFp4BiasLayout(32, 256), 4)
        )
        self.Stage2Bias = (
            NoopBiasLayout(4, self.kGroupN)
            if Solution.bias_dtype == FusedMoEDataType.kNone
            else Bf16BiasLayout(
                4, self.kGroupN, MxFp4BiasLayout(32, self.kGroupN), self.kGroupN // 64
            )
        )
        self.Stage1Tiles = W13TileSchedule(NativeMxFp4TileOps(self, self.W13))
        self.Stage2Tiles = W2TileSchedule(NativeMxFp4TileOps(self, self.W2))
        self.Stage1Op = (
            OnestageFusedMoEStage1DoubleBufferOp
            if Solution.stage1_buffering == FusedMoEStage1Buffering.kDoubleBuffer
            else OnestageFusedMoEStage1SingleBufferOp
        )(self.Stage1Tiles)
        from lib.moe.rocm.comm.barrier import LegacyXGpuSync
        from lib.moe.rocm.mega_moe.workspace import MegaMoEWorkspace

        self.XGpuSync = LegacyXGpuSync(self, MegaMoEWorkspace(self))


class MegaMoEStage1M64W4Config(DeviceTemplate):
    def __init__(self, Base):
        self.__dict__.update(Base.__dict__)
        self._key = Base.cache_key
        self.kGroupM, self.kStage1GroupN, self.kTokenBatch = 64, 256, 16
        self.kNumWarps, self.kStage1WarpsM, self.kStage1WarpsN, self.kThreads = (
            4,
            2,
            2,
            256,
        )
        self.kW13TileShape = MxFp4TileShape.kM64N256
        self.Input, self.Weight = MxFp4InputPacked(self), MxFp4Weights(self)
        self.W13Weights, self.W13 = self.Weight.W13Weights, self.Weight.W13
        self.Bias = (
            NoopBiasLayout(4, 256)
            if Base.kSolution.bias_dtype == FusedMoEDataType.kNone
            else Bf16BiasLayout(4, 256, MxFp4BiasLayout(64, 256), 8)
        )
        self.Stage1Tiles = W13TileSchedule(NativeMxFp4TileOps(self, self.W13))
        self.Stage1Op = (
            OnestageFusedMoEStage1DoubleBufferOp
            if Base.kSolution.stage1_buffering == FusedMoEStage1Buffering.kDoubleBuffer
            else OnestageFusedMoEStage1SingleBufferOp
        )(self.Stage1Tiles)


class MegaMoEStage1M64W8Config(DeviceTemplate):
    def __init__(self, Base):
        self.__dict__.update(Base.__dict__)
        self._key = Base.cache_key
        self.kGroupM, self.kStage1GroupN, self.kTokenBatch = 64, 256, 8
        self.kNumWarps, self.kStage1WarpsM, self.kStage1WarpsN, self.kThreads = (
            8,
            2,
            4,
            512,
        )
        self.kStage1WaveM64 = True
        self.kW13TileShape = MxFp4TileShape.kM64N256W8
        self.Input, self.Weight = MxFp4InputPacked(self), MxFp4Weights(self)
        self.W13Weights, self.W13 = self.Weight.W13Weights, self.Weight.W13
        self.Bias = (
            NoopBiasLayout(8, 256)
            if Base.kSolution.bias_dtype == FusedMoEDataType.kNone
            else Bf16BiasLayout(8, 256, MxFp4BiasLayoutM64N256W8, 2, 4)
        )
        self.Stage1Tiles = W13TileSchedule(NativeMxFp4TileOps(self, self.W13))
        self.Stage1Op = (
            OnestageFusedMoEStage1DoubleBufferOp
            if Base.kSolution.stage1_buffering == FusedMoEStage1Buffering.kDoubleBuffer
            else OnestageFusedMoEStage1SingleBufferOp
        )(self.Stage1Tiles)
