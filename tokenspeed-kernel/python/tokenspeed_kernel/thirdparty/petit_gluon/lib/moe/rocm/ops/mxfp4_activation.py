"""Shared MXFP4 intermediate layout and stage-two input operations."""

from dataclasses import dataclass

import triton.experimental.gluon as g
import triton.language as tl
from lib.tal.host_device import host_device
from triton.experimental.gluon import language as l


@dataclass(frozen=True)
class _MxFp4ActivationLayout:
    __triton_builtin__ = True
    cache_key = "MxFp4ActivationLayout-v1"

    @host_device
    def ValueBytes(self, rows, inter_dim):
        return rows * inter_dim // 2

    @host_device
    def PaddedScaleRows(self, rows):
        return ((rows + 255) // 256) * 256

    @host_device
    def ScaleCols(self, inter_dim):
        return ((inter_dim // 32 + 7) // 8) * 8

    @host_device
    def ScaleBytes(self, rows, inter_dim):
        return self.PaddedScaleRows(rows) * self.ScaleCols(inter_dim)

    @host_device
    def ScaleOffset(self, row, col, scale_cols):
        return row * scale_cols + col


MxFp4ActivationLayout = _MxFp4ActivationLayout()


from typing import NamedTuple

from lib.gemm.rocm.amd_intrinsics import (
    BufferResource,
    _resource_content,
    amdgcn_ds_swizzle,
    amdgcn_mov_dpp,
    amdgcn_shuffle,
    llvm_amdgcn_raw_buffer_load_v4i32,
)
from lib.moe.rocm.memory_ops import _load_vector4, _store_vector4
from lib.moe.rocm.ops.stage1_accumulator_lds import (
    StoreStage1AccumulatorLds2D,
    StoreStage1AccumulatorLdsM64,
)
from lib.moe.rocm.quantization import AiterMxFp4Quantization, QuantizeMxFp4
from lib.tal.device import DeviceTemplate, device_method


class Quantized(NamedTuple):
    value: object
    scale: object


class MxFp4ActivationQuantizer(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.Config = Config
        self.kRowsPerTile, self.kTileCols = Config.kGroupM, Config.kStage1GroupN
        self.kInputFragments = (
            self.kRowsPerTile * self.kTileCols // (Config.kNumWarps * 64) // 4
        )
        self.kWaveM64 = getattr(Config, "kStage1WaveM64", False)
        self.kQuantizeShmWords = self.kRowsPerTile * self.kTileCols

    @device_method
    def StoreAccumulator(self, shm, h, wid, wtid):
        if self.kWaveM64:
            StoreStage1AccumulatorLdsM64(
                shm, h, wid, wtid, self.kTileCols, self.Config.kNumWarps
            )
        else:
            StoreStage1AccumulatorLds2D(
                shm,
                h,
                wid,
                wtid,
                self.kRowsPerTile,
                self.kTileCols,
                self.Config.kStage1WarpsM,
                self.Config.kStage1WarpsN,
            )

    @device_method
    def Quantize(self, shm, row, col_lane):
        value = _load_vector4(shm + row * self.kTileCols + col_lane * 4)
        max_abs = AiterMxFp4Quantization.MaximumAbs(value)
        max_abs = self.ReduceMaximum(max_abs, 0x041F)
        max_abs = self.ReduceMaximum(max_abs, 0x081F)
        max_abs = self.ReduceMaximum(max_abs, 0x101F)
        packed, scale = QuantizeMxFp4((value,), max_abs, AiterMxFp4Quantization, 1)
        return Quantized(packed[0], scale)

    @device_method
    def Store(
        self,
        workspace,
        value_base,
        scale_base,
        value_row,
        scale_row,
        tile_n,
        col_lane,
        inter_dim,
        scale_cols,
        quantized,
        kStoreScope: l.constexpr = BufferResource.kNone,
        valid=True,
    ):
        col_local = col_lane * 4
        value_offset = (
            value_row * (inter_dim // 2)
            + tile_n * (self.kTileCols // 2)
            + col_local // 2
        )
        partner = amdgcn_mov_dpp(quantized.value.to(l.uint32), 0xB1, 0xF, 0xF, False)
        packed = quantized.value.to(l.uint32) | (partner << 16)
        BufferResource.StoreU32(
            workspace,
            value_offset,
            value_base,
            packed,
            kStoreScope | BufferResource.kNTBit,
            valid & ((col_lane & 1) == 0),
        )
        scale0 = amdgcn_shuffle(quantized.scale, 0, 32)
        scale1 = amdgcn_shuffle(quantized.scale, 8, 32)
        scale2 = amdgcn_shuffle(quantized.scale, 16, 32)
        scale3 = amdgcn_shuffle(quantized.scale, 24, 32)
        packed = scale0 | (scale1 << 8) | (scale2 << 16) | (scale3 << 24)
        scale_col = tile_n * (self.kTileCols // 32) + (col_lane // 32) * 4
        scale_offset = MxFp4ActivationLayout.ScaleOffset(
            scale_row, scale_col, scale_cols
        )
        BufferResource.StoreU32(
            workspace,
            scale_offset,
            scale_base,
            packed,
            kStoreScope,
            valid & ((col_lane & 31) == 0),
        )

    @device_method
    def ReduceMaximum(self, value, kPattern: l.constexpr):
        bits = value.to(l.uint32, bitcast=True)
        peer_bits = amdgcn_ds_swizzle(bits, kPattern)
        peer = peer_bits.to(l.float32, bitcast=True)
        return l.maximum(value, peer, propagate_nan=tl.PropagateNan.ALL)


class Stage2Prefetch(NamedTuple):
    value: object
    scale: object


class MxFp4InputRegs(NamedTuple):
    x: object
    scale: object


class MxFp4Stage2Input(DeviceTemplate):
    def __init__(self, Config, InputRegs=MxFp4InputRegs):
        self._key = Config.cache_key
        self.InputRegs = InputRegs
        self.kRowsPerTile, self.kGroupDim = Config.kStage2GroupM, Config.kGroupDim
        self.kVectorsPerRow = (self.kGroupDim // 2) // 16
        self.kScaleCols = MxFp4ActivationLayout.ScaleCols(Config.kInterDim)
        self.kLdsStages, self.kLdsVectorsPerRow, self.kInvalidValueOffset = (
            2,
            16,
            0xFFFFFFF0,
        )
        self.kInputShmWords = (
            self.kLdsStages * self.kRowsPerTile * self.kLdsVectorsPerRow * 4
        )

    @device_method
    def LoadTile(
        self,
        workspace,
        value_voffset,
        value_soffset,
        scale_voffset,
        scale_soffset,
        tile_k,
        valid,
        wtid,
        kAux: l.constexpr = BufferResource.kNone,
    ):
        value_offset = l.where(
            valid,
            value_voffset + tile_k * self.kGroupDim // 2,
            self.kInvalidValueOffset,
        )
        value = llvm_amdgcn_raw_buffer_load_v4i32(
            _resource_content(workspace), value_offset, value_soffset, kAux
        )
        return Stage2Prefetch(
            value,
            self.LoadScaleWord(
                workspace, scale_voffset, scale_soffset, tile_k, wtid, kAux
            ),
        )

    @device_method
    def StoreLds(self, shm, value, stage, tid):
        row = tid // self.kVectorsPerRow
        vector = tid % self.kVectorsPerRow
        _store_vector4(
            shm
            + (
                (stage * self.kRowsPerTile + row) * self.kLdsVectorsPerRow
                + (vector ^ (row & 15))
            )
            * 4,
            value,
        )

    @device_method
    def ReadLds(self, shm, stage, scale, wtid):
        x = ()
        for row_group in l.static_range(2):
            for k128 in l.static_range(2):
                row = wtid % 16 + row_group * 16
                vector = wtid // 16 + k128 * 4
                x += (
                    _load_vector4(
                        shm
                        + (
                            (stage * self.kRowsPerTile + row) * self.kLdsVectorsPerRow
                            + (vector ^ (row & 15))
                        )
                        * 4
                    ),
                )
        return MxFp4InputRegs(x, (scale,))

    @device_method
    def LoadScaleWord(
        self, workspace, scale_voffset, scale_soffset, tile_k, wtid, kAux: l.constexpr
    ):
        load_row = wtid & 31
        load_col = tile_k * 8 + (wtid >> 5) * 4
        loaded = BufferResource.LoadU32(
            workspace,
            scale_voffset + load_row * self.kScaleCols + load_col,
            scale_soffset,
            kAux,
        )
        row16 = wtid & 15
        scale4 = wtid >> 4
        shift = scale4 * 8
        return (
            ((amdgcn_shuffle(loaded, row16) >> shift) & 0xFF)
            | (((amdgcn_shuffle(loaded, row16 + 16) >> shift) & 0xFF) << 8)
            | (((amdgcn_shuffle(loaded, row16 + 32) >> shift) & 0xFF) << 16)
            | (((amdgcn_shuffle(loaded, row16 + 48) >> shift) & 0xFF) << 24)
        )
