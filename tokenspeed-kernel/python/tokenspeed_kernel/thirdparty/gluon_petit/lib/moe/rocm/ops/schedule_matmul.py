"""Tensor matrix multiply policy with the original register layout."""

from lib.gemm.rocm.cdna4_ops import scaled_mfma_tile
from lib.tal.device import DeviceTemplate, device_method


class NativeMxFp4Matmul(DeviceTemplate):
    def __init__(self, kTileM, kTileN):
        assert kTileM in (32, 64) and kTileN % 32 == 0
        self._key = (kTileM, kTileN)
        self.kMRepeats, self.kNRepeats = kTileM // 16, kTileN // 16
        self.kKStages = 2
        self.kActivationFragments = self.kMRepeats * self.kKStages
        self.kAccumFragments = self.kMRepeats * self.kNRepeats
        self.kWeightFragments = self.kNRepeats
        self.kScaleFragments = self.kMRepeats // 2

    @device_method
    def Matmul(self, t, w, x, scale_x, scale_w):
        return scaled_mfma_tile(
            t, w, x, scale_x, scale_w, self.kMRepeats, self.kNRepeats
        )
