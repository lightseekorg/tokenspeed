"""Production activation operations from moe/rocm/ops/activation.cuh."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    amdgcn_exp2f,
    amdgcn_pk_add_f32,
    amdgcn_pk_mul_f32,
    amdgcn_rcpf,
)
from triton.experimental.gluon import language as l


class SiluDotOp:
    __triton_builtin__ = True

    @g.jit
    def Apply(gate, up):
        kMinusLog2e: l.constexpr = -1.4426950408889634
        kMinusLog2e2: l.constexpr = (kMinusLog2e, kMinusLog2e)
        kOne2: l.constexpr = (1.0, 1.0)
        g2 = (gate[:2], gate[2:])
        u2 = (up[:2], up[2:])
        i2 = (
            amdgcn_pk_mul_f32(g2[0], kMinusLog2e2),
            amdgcn_pk_mul_f32(g2[1], kMinusLog2e2),
        )
        i2 = (
            (amdgcn_exp2f(i2[0][0]), amdgcn_exp2f(i2[0][1])),
            (amdgcn_exp2f(i2[1][0]), amdgcn_exp2f(i2[1][1])),
        )
        i2 = (amdgcn_pk_add_f32(i2[0], kOne2), amdgcn_pk_add_f32(i2[1], kOne2))
        i2 = (
            (amdgcn_rcpf(i2[0][0]), amdgcn_rcpf(i2[0][1])),
            (amdgcn_rcpf(i2[1][0]), amdgcn_rcpf(i2[1][1])),
        )
        i2f = i2
        r0 = amdgcn_pk_mul_f32(amdgcn_pk_mul_f32(gate[:2], i2f[0]), u2[0])
        r1 = amdgcn_pk_mul_f32(amdgcn_pk_mul_f32(gate[2:], i2f[1]), u2[1])
        return r0 + r1


class KimiSituOp:
    __triton_builtin__ = True

    @g.jit
    def Apply(gate, up):
        kMinusLog2e: l.constexpr = -1.4426950408889634
        # Native constexpr evaluation rounds each float operation to binary32.
        kGateTanhScale: l.constexpr = -0.7213475108146667
        kUpTanhScale: l.constexpr = -0.11541559547185898
        result = ()
        g, u = gate, up
        for i in l.static_range(4):
            sig = amdgcn_rcpf(1.0 + amdgcn_exp2f(kMinusLog2e * g[i]))
            tanh_gate = (
                2.0 * amdgcn_rcpf(1.0 + amdgcn_exp2f(kGateTanhScale * g[i])) - 1.0
            )
            tanh_up = 2.0 * amdgcn_rcpf(1.0 + amdgcn_exp2f(kUpTanhScale * u[i])) - 1.0
            result += ((4.0 * tanh_gate * sig) * (25.0 * tanh_up),)
        return result


class OpenAISwiGLUOp:
    __triton_builtin__ = True

    @g.jit
    def Apply(gate, up):
        kMinusAlphaLog2e: l.constexpr = -2.455307455790015
        kLimit: l.constexpr = 7.0
        r = ()
        g, u = gate, up
        for i in l.static_range(4):
            gc = l.minimum(g[i], kLimit)
            uc = l.minimum(l.maximum(u[i], -kLimit), kLimit)
            sig = amdgcn_rcpf(1.0 + amdgcn_exp2f(kMinusAlphaLog2e * gc))
            r += (gc * sig * (uc + 1.0),)
        return r
