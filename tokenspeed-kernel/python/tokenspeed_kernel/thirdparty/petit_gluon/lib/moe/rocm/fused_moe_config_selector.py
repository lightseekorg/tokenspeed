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

kFusedMoEBlockScaleFp8SolutionId = FusedMoESolutionId.MakeBase(
    FusedMoEDataType.kChannelScaleFp8,
    FusedMoEDataType.kBlockScaleFp8,
    FusedMoEDataType.kNone,
    FusedMoEWeightOrdering.kPetitFp8,
    FusedMoEMfmaShape.kMfmaFp816x16x32,
    FusedMoEStages.kOneStage,
    FusedMoEActivationFunction.kSiluDot,
    FusedMoEStage1Buffering.kDoubleBuffer,
)

kFusedMoEFp8PetitMxFp4SolutionId = FusedMoESolutionId.MakeBase(
    FusedMoEDataType.kChannelScaleFp8,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kNone,
    FusedMoEWeightOrdering.kPetitMxFp4,
    FusedMoEMfmaShape.kMfmaFp816x16x32,
    FusedMoEStages.kOneStage,
    FusedMoEActivationFunction.kSiluDot,
    FusedMoEStage1Buffering.kSingleBuffer,
)

kFusedMoEFp8PetitMxFp4BiasSolutionId = FusedMoESolutionId.MakeBase(
    FusedMoEDataType.kChannelScaleFp8,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kBf16,
    FusedMoEWeightOrdering.kPetitMxFp4,
    FusedMoEMfmaShape.kMfmaFp816x16x32,
    FusedMoEStages.kOneStage,
    FusedMoEActivationFunction.kOpenAISwiGLU,
    FusedMoEStage1Buffering.kDoubleBuffer,
)

kFusedMoEBf16NativeMxFp4BiasSolutionId = FusedMoESolutionId.MakeBase(
    FusedMoEDataType.kBf16,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kBf16,
    FusedMoEWeightOrdering.kNativeMxFp4,
    FusedMoEMfmaShape.kMfmaBf16MxFp4,
    FusedMoEStages.kOneStage,
    FusedMoEActivationFunction.kOpenAISwiGLU,
    FusedMoEStage1Buffering.kDoubleBuffer,
)

kFusedMoEMxFp4NativeMxFp4BiasSolutionId = FusedMoESolutionId.MakeBase(
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kBf16,
    FusedMoEWeightOrdering.kNativeMxFp4,
    FusedMoEMfmaShape.kMfmaScaleFp4MxFp4,
    FusedMoEStages.kOneStage,
    FusedMoEActivationFunction.kOpenAISwiGLU,
    FusedMoEStage1Buffering.kDoubleBuffer,
)

kFusedMoETwoStageMxFp4BiasSolutionId = FusedMoESolutionId.Make(
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kBf16,
    FusedMoEWeightOrdering.kNativeMxFp4,
    FusedMoEMfmaShape.kMfmaScaleFp4MxFp4,
    FusedMoEStages.kTwoStage,
    FusedMoEActivationFunction.kOpenAISwiGLU,
    FusedMoEStage1Buffering.kDoubleBuffer,
    3072,
    3072,
)

kFusedMoETwoStageMxFp4SiluSolutionId = FusedMoESolutionId.MakeBase(
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kMxFp4,
    FusedMoEDataType.kNone,
    FusedMoEWeightOrdering.kNativeMxFp4,
    FusedMoEMfmaShape.kMfmaScaleFp4MxFp4,
    FusedMoEStages.kTwoStage,
    FusedMoEActivationFunction.kSiluDot,
    FusedMoEStage1Buffering.kDoubleBuffer,
)


from lib.moe.rocm.mem.bias import Bf16BiasLayout, MxFp4BiasLayout, NoopBiasLayout
from lib.moe.rocm.mem.input_mxfp4 import MxFp4Input
from lib.moe.rocm.mem.weight_mxfp4 import MxFp4Weights
from lib.moe.rocm.memory_ops import (
    BufferResource,
    MxFp4TileShape,
    TargetWeightLoadPolicy,
)
from lib.moe.rocm.ops.activation import KimiSituOp, OpenAISwiGLUOp, SiluDotOp
from lib.moe.rocm.ops.op_stages import (
    OnestageFusedMoEStage1DoubleBufferOp,
    OnestageFusedMoEStage1SingleBufferOp,
    W2TileSchedule,
    W13TileSchedule,
)
from lib.moe.rocm.ops.schedule_tiles import NativeMxFp4TileOps
from lib.tal.device import DeviceTemplate


class ConfigSelector(DeviceTemplate):
    def __init__(self, id, topk=4):
        self._key = (id.Repr(), topk)
        self.kSolution = id
        self.kDim, self.kInterDim, self.kTopK = id.Dim(), id.InterDim(), topk
        self.kGroupM = id.Stage1TileM() if id.stages == FusedMoEStages.kTwoStage else 32
        self.kGroupN = 256
        self.kStage1GroupN = (
            id.Stage1TileN() // 2
            if id.stages == FusedMoEStages.kTwoStage
            else self.kGroupN
        )
        self.kGroupDim, self.kNumWarps = 256, 4
        self.kTokenBatch = self.kGroupM // self.kNumWarps
        self.kStage1WarpsM = self.kGroupM // 32
        self.kStage1WarpsN = self.kNumWarps // self.kStage1WarpsM
        self.kStage2GroupM, self.kStage2TokenBatch = 32, 8
        self.kStage1ToStage2GroupRatio = self.kGroupM // self.kStage2GroupM
        self.kThreads = self.kNumWarps * 64
        self.kStage2GroupInterDim = self.kGroupDim
        self.kUseNonTemporalWeightLoads = (
            TargetWeightLoadPolicy.kAux == BufferResource.kNTBit
            if id.stages == FusedMoEStages.kOneStage
            else id.weight_load_policy == FusedMoEWeightLoadPolicy.kNonTemporal
        )
        self.kWeightLoadAux = (
            BufferResource.kNTBit
            if self.kUseNonTemporalWeightLoads
            else BufferResource.kNone
        )
        self.kActDType, self.kWeightDType, self.kMfmaShape = (
            id.act_dtype,
            id.weight_dtype,
            id.mfma,
        )
        self.kValidateExpertIds = False
        self.kW13TileShape = {
            (32, 128): MxFp4TileShape.kN128,
            (32, 256): MxFp4TileShape.kN256,
            (64, 256): MxFp4TileShape.kM64N256,
        }[(self.kGroupM, self.kStage1GroupN)]
        self.kW2TileShape = MxFp4TileShape.kN256
        self.ActivationOp = {
            FusedMoEActivationFunction.kSiluDot: SiluDotOp,
            FusedMoEActivationFunction.kOpenAISwiGLU: OpenAISwiGLUOp,
            FusedMoEActivationFunction.kKimiSitu: KimiSituOp,
        }[id.activation]
        if (
            id.act_dtype != FusedMoEDataType.kMxFp4
            or id.weight_ordering != FusedMoEWeightOrdering.kNativeMxFp4
        ):
            raise NotImplementedError("Current selector port: native MXFP4 path only")
        self.Input = MxFp4Input(self)
        self.Weights = MxFp4Weights(self)
        self.W13Weights, self.W2Weights = (
            self.Weights.W13Weights,
            self.Weights.W2Weights,
        )
        self.W13, self.W2 = self.Weights.W13, self.Weights.W2
        if id.bias_dtype == FusedMoEDataType.kNone:
            self.Bias = NoopBiasLayout(self.kNumWarps, self.kStage1GroupN)
            self.Stage2Bias = NoopBiasLayout(self.kNumWarps, self.kGroupN)
        else:
            self.Bias = Bf16BiasLayout(
                self.kNumWarps,
                self.kStage1GroupN,
                MxFp4BiasLayout(self.kGroupM, self.kStage1GroupN),
                self.kStage1GroupN // self.kStage1WarpsN // 16,
            )
            self.Stage2Bias = Bf16BiasLayout(
                self.kNumWarps, self.kGroupN, MxFp4BiasLayout(32, self.kGroupN), 4
            )
        self.Stage1Tiles = W13TileSchedule(NativeMxFp4TileOps(self, self.W13))
        self.Stage2Tiles = W2TileSchedule(NativeMxFp4TileOps(self, self.W2))
        self.Stage1Op = (
            OnestageFusedMoEStage1DoubleBufferOp
            if id.stage1_buffering == FusedMoEStage1Buffering.kDoubleBuffer
            else OnestageFusedMoEStage1SingleBufferOp
        )(self.Stage1Tiles)
