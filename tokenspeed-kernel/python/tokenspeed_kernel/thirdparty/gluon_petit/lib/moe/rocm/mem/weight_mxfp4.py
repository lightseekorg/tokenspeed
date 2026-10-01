"""Native MXFP4 projection resources, offsets, and initialization."""

from typing import NamedTuple

from lib.moe.rocm.memory_ops import MxFp4TileShape, MxFp4WeightLayout
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


class MxFp4ConfigWeightLayouts:
    def __init__(self, Config):
        if hasattr(Config, "kW13TileShape") and hasattr(Config, "kW2TileShape"):
            self.kW13, self.kW2 = Config.kW13TileShape, Config.kW2TileShape
        else:
            self.kW13 = self.kW2 = MxFp4TileShape.kN256


class MxFp4WeightConfig:
    def __init__(self, Config):
        self.Layouts = MxFp4ConfigWeightLayouts(Config)
        self.W13 = MxFp4WeightLayout(Config.kNumWarps, self.Layouts.kW13)
        self.W2 = MxFp4WeightLayout(Config.kNumWarps, self.Layouts.kW2)
        self.kScaleGroupK, self.kScaleGroupN, self.kWeightVecSize = 128, 64, 32
        self.kW13ScaleGroupsPerOutputTile = self.W13.kGroupN // self.kScaleGroupN
        self.kScaleBytesPerWord, self.kScaleWordBytes = 4, 4


class W13State(NamedTuple):
    w1_: object


class W2State(NamedTuple):
    w2_: object


class MxFp4W13(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.Traits = MxFp4WeightConfig(Config)
        self.W13 = self.Traits.W13
        for field in (
            "kScaleGroupK",
            "kScaleGroupN",
            "kWeightVecSize",
            "kScaleBytesPerWord",
            "kScaleWordBytes",
        ):
            setattr(self, field, getattr(self.Traits, field))
        self.kScaleGroupsPerOutputTile = self.Traits.kW13ScaleGroupsPerOutputTile

    @device_method
    def Initialize(self, w13_base, scales_w13, expert_id, tile_k, dim, inter_dim):
        kRowGroupSize: l.constexpr = self.W13.kRowGroupSize
        # Match the native unsigned parameters, including uint32 wraparound.
        expert_id, tile_k = l.cast(expert_id, l.uint32), l.cast(tile_k, l.uint32)
        dim, inter_dim = l.cast(dim, l.uint32), l.cast(inter_dim, l.uint32)
        # Each native uint4 pointer addition scales its unsigned index at pointer width.
        w1_ptr = (
            w13_base.to(l.pointer_type(l.uint8))
            + (expert_id * (2 * inter_dim * dim) // self.kWeightVecSize).to(l.uint64)
            * 16
            + (tile_k * self.W13.kGroupN * dim // self.kWeightVecSize).to(l.uint64) * 16
        )
        w13_scale_words_per_expert = (
            (2 * dim * inter_dim) // kRowGroupSize // self.kScaleBytesPerWord
        )
        scale_words_per_k_tile = (
            (dim // self.kScaleGroupK) * self.kScaleGroupsPerOutputTile * 64
        )
        scale_w1_ptr = (
            scales_w13.to(l.pointer_type(l.uint32))
            + expert_id * w13_scale_words_per_expert
            + tile_k * scale_words_per_k_tile
        )
        w13_value_range = self.W13.kGroupN * dim // 2
        w13_combined_value_range = inter_dim * dim // 2 + w13_value_range
        w13_scale_range = scale_words_per_k_tile * self.kScaleWordBytes
        w13_combined_scale_range = (
            self.W13.ScaleProjectionOffsetBytes(dim, inter_dim) + w13_scale_range
        )
        return W13State(
            self.W13.Initialize(
                w1_ptr,
                w13_combined_value_range,
                scale_w1_ptr,
                w13_combined_scale_range,
                dim,
            )
        )


class MxFp4W2(DeviceTemplate):
    def __init__(self, Config):
        self._key = Config.cache_key
        self.Config = Config
        self.Traits = MxFp4WeightConfig(Config)
        self.W2 = self.Traits.W2
        self.kWeightVecSize, self.kScaleBytesPerWord, self.kScaleWordBytes = 32, 4, 4

    @device_method
    def Initialize(self, w2, scales_w2, expert_id, tile_n, tile_k):
        kRowGroupSize: l.constexpr = self.W2.kRowGroupSize
        kDim: l.constexpr = self.Config.kDim
        kInterDim: l.constexpr = self.Config.kInterDim
        expert_id = l.cast(expert_id, l.uint32)
        tile_n, tile_k = l.cast(tile_n, l.uint32), l.cast(tile_k, l.uint32)
        value_bytes_per_expert = kDim * kInterDim // 2
        scale_words_per_expert = (
            kDim * kInterDim // kRowGroupSize // self.kScaleBytesPerWord
        )
        value_tile_offset = (
            tile_n * self.W2.kGroupN * kInterDim // 2 + tile_k * 2 * 64 * 16
        )
        scale_tile_offset = (
            tile_n
            * self.W2.kGroupN
            * kInterDim
            // kRowGroupSize
            // self.kScaleBytesPerWord
            + tile_k * 64
        )
        value_ptr = (
            w2.to(l.pointer_type(l.uint8))
            + expert_id * value_bytes_per_expert
            + value_tile_offset
        )
        scale_ptr = (
            scales_w2.to(l.pointer_type(l.uint32))
            + expert_id * scale_words_per_expert
            + scale_tile_offset
        )
        return W2State(
            self.W2.Initialize(
                value_ptr,
                value_bytes_per_expert - value_tile_offset,
                scale_ptr,
                (scale_words_per_expert - scale_tile_offset) * self.kScaleWordBytes,
                kInterDim,
            )
        )


class MxFp4Weights:
    def __init__(self, Config):
        self.W13Weights, self.W2Weights = MxFp4W13(Config), MxFp4W2(Config)
        self.W13, self.W2 = self.W13Weights.W13, self.W2Weights.W2
