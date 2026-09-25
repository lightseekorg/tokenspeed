"""Native per-wave matrix multiply policies and register layouts."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    amdgcn_mov_dpp,
    amdgcn_pk_fma_f32,
    mma_m16n16k128_fp8_fp8_f32,
    mma_scale_m16n16k128_fp4_fp4_f32,
)
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


@g.jit
def Fma4(a, s, c):
    a2 = (a[:2], a[2:])
    c2 = (c[:2], c[2:])
    if isinstance(s, l.tuple):
        s2 = (s[:2], s[2:])
    else:
        s2 = ((s, s), (s, s))
    r0 = amdgcn_pk_fma_f32(s2[0], c2[0], a2[0])
    r1 = amdgcn_pk_fma_f32(s2[1], c2[1], a2[1])
    return r0 + r1


@g.jit
def GetDppValue(src, ctrl: l.constexpr):
    kDppRowNewBcastBase: l.constexpr = 0x150
    bits = src.to(l.uint32, bitcast=True)
    if ctrl == 0:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase, 0xF, 0xF, False)
    elif ctrl == 1:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 1, 0xF, 0xF, False)
    elif ctrl == 2:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 2, 0xF, 0xF, False)
    else:
        dst = amdgcn_mov_dpp(bits, kDppRowNewBcastBase + 3, 0xF, 0xF, False)
    return dst.to(src.dtype, bitcast=True)


@g.jit
def LoadFp8E8M0Scale(packed):
    f = ()
    for i in l.static_range(4):
        u = ((packed >> (8 * i)) & 0xFF) << 23
        f += (u.to(l.float32, bitcast=True) * 16384.0,)
    return f


@g.jit
def LoadE8M0ScaleByte(packed, byte_idx):
    bits = ((packed >> (byte_idx * 8)) & 0xFF) << 23
    return bits.to(l.float32, bitcast=True)


@g.jit
def ScaledMxFp4Mfma(
    opsel_a: l.constexpr, opsel_b: l.constexpr, a, scale_a, b, scale_b, acc
):
    # The native dispatch lambdas specialize both immediate selectors.
    if opsel_a == 0:
        kOpSelA: l.constexpr = 0
    elif opsel_a == 1:
        kOpSelA: l.constexpr = 1
    elif opsel_a == 2:
        kOpSelA: l.constexpr = 2
    else:
        kOpSelA: l.constexpr = 3
    if opsel_b == 0:
        kOpSelB: l.constexpr = 0
    elif opsel_b == 1:
        kOpSelB: l.constexpr = 1
    elif opsel_b == 2:
        kOpSelB: l.constexpr = 2
    else:
        kOpSelB: l.constexpr = 3
    return mma_scale_m16n16k128_fp4_fp4_f32(
        a, scale_a, b, scale_b, acc, kOpSelA, kOpSelB
    )


class MatmulTile(DeviceTemplate):
    def __init__(self, kTileN, kInputBits, kWeightBits):
        self._key = (kTileN, kInputBits, kWeightBits)
        self.kTileM, self.kTileN, self.kTileK, self.kKStages = 32, kTileN, 256, 2
        self.kWeightFragments = kTileN * 256 * kWeightBits // (8 * 16 * 64 * 2)
        self.kActivationFragments = 32 * 256 * kInputBits // (8 * 16 * 64)
        self.kAccumFragments = 32 * kTileN * 4 // (16 * 64)


class BlockScaleFp8Matmul(MatmulTile):
    def __init__(self):
        super().__init__(64, 8, 8)

    @device_method
    def Matmul(self, t, w, x, x_scale, w_scale, stage: l.constexpr):
        for i in l.static_range(4):
            w_2 = (w[i * 2][:2], w[i * 2][2:], w[i * 2 + 1][:2], w[i * 2 + 1][2:])
            for row in l.static_range(2):
                tg: l.constexpr
                tg = row * 4
                x_2 = (
                    x[tg + stage * 2][:2],
                    x[tg + stage * 2][2:],
                    x[tg + stage * 2 + 1][:2],
                    x[tg + stage * 2 + 1][2:],
                )
                z = l.full(w_scale.shape, 0.0, l.float32, w_scale.type.layout)
                m_acc = (z, z, z, z)
                m_acc = mma_m16n16k128_fp8_fp8_f32(w_2, x_2, m_acc)
                x_s = x_scale[row] if stage == 0 else x_scale[row + 2]
                w_sdpp = GetDppValue(w_scale, (stage << 1) + (i >> 1))
                result = Fma4(t[i * 2 + row], w_sdpp * x_s, m_acc)
                t = t[: i * 2 + row] + (result,) + t[i * 2 + row + 1 :]
        return t


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
        for k128 in l.static_range(2):
            for n_fragment in l.static_range(self.kWeightFragments):
                for m16 in l.static_range(self.kMRepeats):
                    t_idx: l.constexpr
                    t_idx = n_fragment * self.kMRepeats + m16
                    value = ScaledMxFp4Mfma(
                        2 * k128 + (n_fragment & 1),
                        2 * k128 + (m16 & 1),
                        w[k128][n_fragment],
                        scale_w[n_fragment // 2],
                        x[m16 * 2 + k128],
                        scale_x[m16 // 2],
                        t[t_idx],
                    )
                    t = t[:t_idx] + (value,) + t[t_idx + 1 :]
        return t
