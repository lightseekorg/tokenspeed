"""Native tile-level load and MFMA schedules."""

from typing import NamedTuple

from lib.moe.rocm.memory_ops import TargetWeightLoadPolicy
from lib.moe.rocm.ops.mxfp4_activation import MxFp4InputRegs
from lib.moe.rocm.ops.schedule_matmul import NativeMxFp4Matmul
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class MxFp4Tile(NamedTuple):
    value: object
    scale: object


class InputShm(NamedTuple):
    act: object
    scale: object


class NativeMxFp4TileOps(DeviceTemplate):
    InputRegs = MxFp4InputRegs
    Tile = MxFp4Tile

    def __init__(self, Config, Weight):
        self._key = (Config.cache_key, Weight.cache_key)
        self.Config, self.Weight, self.Input = Config, Weight, Config.Input
        self.kWaveTileN = Weight.kWaveTileN
        self.MatmulOp = NativeMxFp4Matmul(Weight.kWaveTileM, self.kWaveTileN)
        self.kNumWarps, self.kKStages = Config.kNumWarps, Config.kGroupDim // 128
        self.kActivationFragments = self.MatmulOp.kActivationFragments
        self.kScaleFragments = self.MatmulOp.kScaleFragments
        self.kAccumFragments = self.MatmulOp.kAccumFragments
        self.kOutputPacksPerToken = Config.kGroupN // 128
        self.kStage2BiasUsesTileK = False
        self.kWeightLoadAux = getattr(
            Config, "kWeightLoadAux", TargetWeightLoadPolicy.kAux
        )
        assert self.kKStages == 2
        # Stage2 uses its own consumer rather than Config.Input.
        assert Weight.kLoadGlobal == self.MatmulOp.kWeightFragments

    @device_method
    def PrefetchInput(self, input_state, shm, wid, wtid, tokens, m):
        input_state = self.Input.FetchAsync(input_state, shm.act, wid, wtid, tokens)
        self.Input.FetchScaleAsync(input_state, shm.scale, wid, wtid, tokens, m)
        return input_state

    @device_method
    def ReadInput(self, input_state, shm, wtid):
        x = self.Input.FetchToRegs(input_state, shm.act, wtid)
        scale = ()
        if self.Weight.kWaveTileM == 64:
            for m32 in l.static_range(self.kScaleFragments):
                scale += (
                    self.Input.FetchScaleToReg(input_state, shm.scale, wtid, m32),
                )
        else:
            scale = (self.Input.FetchScaleToReg(input_state, shm.scale, wtid),)
        input_state = self.Input.AdvanceScaleStep(input_state)
        return input_state, MxFp4InputRegs(x, scale)

    @device_method
    def Load(self, weight, tid, wid, wtid):
        v0 = self.Weight.LoadTile(weight, 0, wid, wtid, kAux=self.kWeightLoadAux)
        v1 = self.Weight.LoadTile(weight, 1, wid, wtid, kAux=self.kWeightLoadAux)
        scale = ()
        for n32_pair in l.static_range(self.Weight.kLoadGlobal // 2):
            scale += (self.Weight.LoadScale(weight, wid, wtid, n32_pair),)
        return MxFp4Tile((v0, v1), scale)

    @device_method
    def LoadProjection(self, weight, tid, wid, wtid, value_offset, scale_offset):
        v0 = self.Weight.LoadTile(
            weight, 0, wid, wtid, value_offset, self.kWeightLoadAux
        )
        v1 = self.Weight.LoadTile(
            weight, 1, wid, wtid, value_offset, self.kWeightLoadAux
        )
        scale = ()
        for n32_pair in l.static_range(self.Weight.kLoadGlobal // 2):
            scale += (self.Weight.LoadScale(weight, wid, wtid, n32_pair, scale_offset),)
        return MxFp4Tile((v0, v1), scale)

    @device_method
    def Matmul(self, t, tile, input_regs, wtid):
        return self.MatmulOp.Matmul(
            t, tile.value, input_regs.x, input_regs.scale, tile.scale
        )
