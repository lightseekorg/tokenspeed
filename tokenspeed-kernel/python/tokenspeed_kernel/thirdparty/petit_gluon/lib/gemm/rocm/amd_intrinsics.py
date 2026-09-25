# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native amd_intrinsics.cuh operations for the supported gfx950 target.

Native per-thread vectors are tuples of lane tensors. Existing dense GEMM
accumulators and packed float2 values retain register-view adapters at the API
boundary. LLVM adapters only assemble/extract vectors and supply immargs.
"""

import re
from dataclasses import replace
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import triton
import triton.experimental.gluon as g
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language._core import (
    _unwrap_if_constexpr,
    builtin,
    distributed_type,
)
from triton.runtime.cache import get_cache_manager

# This port targets gfx950. Match the native device-compilation capability flags.
HAS_AMD_SCALE_FP4_MFMA = l.constexpr(True)
HAS_AMD_FP8_MFMA = l.constexpr(True)
HAS_AMD_BF8_FP8_MFMA = l.constexpr(True)
HAS_AMD_FP8_BF8_MFMA = l.constexpr(True)
HAS_AMD_FP8_PACK_CONVERSION = l.constexpr(True)
HAS_AMD_BF8_PACK_CONVERSION = l.constexpr(True)
HAS_AMD_SCHED_BARRIER = l.constexpr(True)
HAS_AMD_SCHED_GROUP_BARRIER = l.constexpr(True)
kWarpSize = l.constexpr(64)


@g.jit
def llvm_amdgcn_raw_buffer_load_lds(
    rsrc,
    lds_ptr,
    size: l.constexpr,
    voffset,
    soffset,
    offset: l.constexpr,
    aux: l.constexpr,
):
    _native_call(
        "llvm.amdgcn.raw.buffer.load.lds",
        "void",
        ("v4i32", "p3", "#i32", "i32", "i32", "#i32", "#i32"),
        (rsrc, lds_ptr.to(l.uint64).to(l.uint32), size, voffset, soffset, offset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_load_v4i32(rsrc, voffset, soffset, AUX: l.constexpr):
    return _native_call(
        "llvm.amdgcn.raw.buffer.load.v4i32",
        "v4i32",
        ("v4i32", "i32", "i32", "#i32"),
        (rsrc, voffset, soffset, AUX),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_store_v4i32(data, rsrc, voffset, soffset, aux: l.constexpr):
    _native_call(
        "llvm.amdgcn.raw.buffer.store.v4i32",
        "void",
        ("v4i32", "v4i32", "i32", "i32", "#i32"),
        (data, rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_load_v2i32(rsrc, voffset, soffset, aux: l.constexpr):
    return _native_call(
        "llvm.amdgcn.raw.buffer.load.v2i32",
        "v2i32",
        ("v4i32", "i32", "i32", "#i32"),
        (rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_load_i32(rsrc, voffset, soffset, aux: l.constexpr):
    return _native_call(
        "llvm.amdgcn.raw.buffer.load.i32",
        "i32",
        ("v4i32", "i32", "i32", "#i32"),
        (rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_store_v2i32(data, rsrc, voffset, soffset, aux: l.constexpr):
    return _native_call(
        "llvm.amdgcn.raw.buffer.store.v2i32",
        "void",
        ("v2i32", "v4i32", "i32", "i32", "#i32"),
        (data, rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_store_i32(data, rsrc, voffset, soffset, aux: l.constexpr):
    return _native_call(
        "llvm.amdgcn.raw.buffer.store.i32",
        "void",
        ("i32", "v4i32", "i32", "i32", "#i32"),
        (data, rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_atomic_add_i32(
    data, rsrc, voffset, soffset, aux: l.constexpr
):
    return _native_call(
        "llvm.amdgcn.raw.buffer.atomic.add.i32",
        "i32",
        ("i32", "v4i32", "i32", "i32", "#i32"),
        (data, rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_atomic_or_i32(
    data, rsrc, voffset, soffset, aux: l.constexpr
):
    return _native_call(
        "llvm.amdgcn.raw.buffer.atomic.or.i32",
        "i32",
        ("i32", "v4i32", "i32", "i32", "#i32"),
        (data, rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def llvm_amdgcn_raw_buffer_atomic_add_i64(
    data, rsrc, voffset, soffset, aux: l.constexpr
):
    return _native_call(
        "llvm.amdgcn.raw.buffer.atomic.add.i64",
        "i64",
        ("i64", "v4i32", "i32", "i32", "#i32"),
        (data, rsrc, voffset, soffset, aux),
        False,
    )


@g.jit
def amdgcn_pk_mul_f32(a, b):
    if isinstance(a, l.tuple):
        return _native_call("fmul", "v2f32", ("v2f32", "v2f32"), (a, b), True)
    else:
        return _pack_float2(
            _native_call(
                "fmul",
                "v2f32",
                ("v2f32", "v2f32"),
                (_unpack_float2(a), _unpack_float2(b)),
                True,
            )
        )


@g.jit
def amdgcn_pk_add_f32(a, b):
    return _native_call("fadd", "v2f32", ("v2f32", "v2f32"), (a, b), True)


@g.jit
def amdgcn_pk_fma_f32(a, b, c):
    return _native_call(
        "llvm.fma.v2f32", "v2f32", ("v2f32", "v2f32", "v2f32"), (a, b, c), True
    )


@g.jit
def amdgcn_exp2f(x):
    return _native_call("llvm.amdgcn.exp2.f32", "f32", ("f32",), (x,), True)


@g.jit
def amdgcn_perm_b32(hi, lo, selector):
    return _native_call(
        "llvm.amdgcn.perm", "i32", ("i32", "i32", "i32"), (hi, lo, selector), True
    )


@g.jit
def amdgcn_ds_permute_b32(index, src):
    return _native_call(
        "llvm.amdgcn.ds.permute", "i32", ("i32", "i32"), (index, src), True
    )


@g.jit
def amdgcn_cvt_pk_fp8_f32(a, b, old, WORD_HI: l.constexpr):
    return _native_call(
        "llvm.amdgcn.cvt.pk.fp8.f32",
        "i32",
        ("f32", "f32", "i32", "#i1"),
        (a, b, old, WORD_HI),
        True,
    )


@g.jit
def amdgcn_cvt_pk_f32_bf8(src, WORD_HI: l.constexpr):
    # Existing dequantization callers store native float2 in a uint64 bit view.
    return _pack_float2(
        _native_call(
            "llvm.amdgcn.cvt.pk.f32.bf8", "v2f32", ("i32", "#i1"), (src, WORD_HI), True
        )
    )


@g.jit
def amdgcn_sched_barrier(mask: l.constexpr):
    _native_call("llvm.amdgcn.sched.barrier", "void", ("#i32",), (mask,), False)


@g.jit
def amdgcn_sched_group_barrier(
    mask: l.constexpr, size: l.constexpr, sync_id: l.constexpr
):
    _native_call(
        "llvm.amdgcn.sched.group.barrier",
        "void",
        ("#i32", "#i32", "#i32"),
        (mask, size, sync_id),
        False,
    )


@g.jit
def amdgcn_pk_float22half2(a, b):
    return _native_call("llvm.amdgcn.cvt.pkrtz", "v2f16", ("f32", "f32"), (a, b), True)


@g.jit
def amdgcn_pk_add_i16(a, b):
    return _native_call("asm.v_pk_add_i16", "i32", ("i32", "i32"), (a, b), True)


@g.jit
def amdgcn_pk_mad_i16(a, b, c):
    # Triton 3.8's LLVM narrows a divergent "r" operand to its first lane.
    # Native Clang keeps that operand in a VGPR. Preserve its register class.
    if isinstance(c, l.tensor) and len(c.shape) != 0:
        return _native_call(
            "asm.v_pk_mad_i16.vector", "i32", ("i32", "i32", "i32"), (a, b, c), True
        )
    else:
        return _native_call(
            "asm.v_pk_mad_i16", "i32", ("i32", "i32", "i32"), (a, b, c), True
        )


@g.jit
def mma_m16n16k16_fp16(fa, fb, c):
    if isinstance(c, l.tuple):
        return _packed_mfma("fp16", fa, fb, c)
    else:
        return _packed_mfma_tensor_acc("fp16", fa, fb, c)


@g.jit
def mma_m16n16k16_bf16(fa, fb, c):
    if isinstance(c, l.tuple):
        return _packed_mfma("bf16", fa, fb, c)
    else:
        return _packed_mfma_tensor_acc("bf16", fa, fb, c)


@g.jit
def mma_m16n16k32_fp8_fp8_f32(fa, fb, c):
    return _packed_mfma("fp8_fp8", fa, fb, c)


@g.jit
def mma_m16n16k32_bf8_fp8_f32(fa, fb, c):
    return _packed_mfma("bf8_fp8", fa, fb, c)


@g.jit
def mma_m16n16k32_fp8_bf8_f32(fa, fb, c):
    return _packed_mfma("fp8_bf8", fa, fb, c)


@g.jit
def mma_m32n32k8_fp16(fa, fb, c):
    return _native_call(
        "llvm.amdgcn.mfma.f32.32x32x8f16",
        "v16f32",
        ("bits:v4f16", "bits:v4f16", "v16f32", "#i32", "#i32", "#i32"),
        (_pack_uint2(fa), _pack_uint2(fb), c, 0, 0, 0),
        True,
    )


@g.jit
def mma_m32n32k8_bf16(fa, fb, c):
    # Native explicitly enables this operation only on gfx942, not gfx950.
    ret = ()
    for i in l.static_range(16):
        if len(c[i].shape) == 0:
            ret += (l.full((), 0, l.float32),)
        else:
            ret += (l.full(c[i].shape, 0, l.float32, c[i].type.layout),)
    return ret


@g.jit
def mma_m16n16k128_fp8_fp8_f32(fa, fb, c):
    if HAS_AMD_SCALE_FP4_MFMA:
        return _native_call(
            "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32",
            "v4f32",
            ("v8i32", "v8i32", "v4f32", "#i32", "#i32", "#i32", "i32", "#i32", "i32"),
            (
                fa[0] + fa[1] + fa[2] + fa[3],
                fb[0] + fb[1] + fb[2] + fb[3],
                c,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            True,
        )
    elif HAS_AMD_FP8_MFMA:
        c = mma_m16n16k32_fp8_fp8_f32(fa[0], fb[0], c)
        c = mma_m16n16k32_fp8_fp8_f32(fa[1], fb[1], c)
        c = mma_m16n16k32_fp8_fp8_f32(fa[2], fb[2], c)
        c = mma_m16n16k32_fp8_fp8_f32(fa[3], fb[3], c)
        return c
    else:
        return (c[0] * 0.0, c[1] * 0.0, c[2] * 0.0, c[3] * 0.0)


@g.jit
def mma_scale_m16n16k128_fp4_fp4_f32(
    fa, scale_a, fb, scale_b, c, kOpSelA: l.constexpr, kOpSelB: l.constexpr
):
    if HAS_AMD_SCALE_FP4_MFMA:
        return _native_call(
            "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32",
            "v4f32",
            ("v8i32", "v8i32", "v4f32", "#i32", "#i32", "#i32", "i32", "#i32", "i32"),
            (
                fa + (0, 0, 0, 0),
                fb + (0, 0, 0, 0),
                c,
                4,
                4,
                kOpSelA,
                scale_a,
                kOpSelB,
                scale_b,
            ),
            True,
        )
    else:
        return c


@g.jit
def amdgcn_wave_inclusive_add(value, lane):
    remote = amdgcn_mov_dpp(value, 0x111, 0xF, 0xF, True)
    value = l.where(lane >= 1, value + remote, value)
    remote = amdgcn_mov_dpp(value, 0x112, 0xF, 0xF, True)
    value = l.where(lane >= 2, value + remote, value)
    remote = amdgcn_mov_dpp(value, 0x114, 0xF, 0xF, True)
    value = l.where(lane >= 4, value + remote, value)
    remote = amdgcn_mov_dpp(value, 0x118, 0xF, 0xF, True)
    value = l.where(lane >= 8, value + remote, value)
    source16 = (lane & 0x30) - 1
    remote = amdgcn_ds_bpermute(source16 * 4, value)
    value = l.where(lane >= 16, value + remote, value)
    source32 = (lane & 0x30) - 17
    remote = amdgcn_ds_bpermute(source32 * 4, value)
    value = l.where(lane >= 32, value + remote, value)
    return value


class BufferResourceFields(NamedTuple):
    ptr: object
    range: object
    config: object


class BufferResource(NamedTuple):
    # Native BufferResource is a union. Keep both its field and content views
    # so buffer operations reuse the descriptor formed by the constructor.
    v: BufferResourceFields
    content: object

    kDataFormatU32Config = l.constexpr(4 << 15)
    kNone = 0
    kSC0Bit = 1 << 0
    kNTBit = 1 << 1
    kSWZBit = 1 << 3
    kSC1Bit = 1 << 4
    kGLCBit = kSC0Bit
    kSLCBit = kNTBit
    kAtomicScopeAgent = kNone
    kAtomicScopeSystem = kSC1Bit

    @g.jit
    def WithRange(resource, byte_range):
        content = _resource_content(resource)
        byte_range = (l.full((), 0, l.uint32) + byte_range).to(l.uint32)
        content = (content[0], content[1], byte_range, content[3])
        return BufferResource(
            BufferResourceFields(resource.v.ptr, byte_range, resource.v.config),
            content,
        )

    @g.jit
    def WithOffset(resource, byte_offset):
        content = _resource_content(resource)
        address = (
            content[0].to(l.uint64) | (content[1].to(l.uint64) << 32)
        ) + byte_offset
        content = (
            address.to(l.uint32),
            (address >> 32).to(l.uint32),
            content[2],
            content[3],
        )
        return BufferResource(
            BufferResourceFields(address, resource.v.range, resource.v.config),
            content,
        )

    @g.jit
    def Load(resource, voffset, soffset, AUX: l.constexpr):
        v = llvm_amdgcn_raw_buffer_load_v4i32(
            _resource_content(resource), voffset, soffset, AUX
        )
        if not v[0].type.is_block():
            return v
        else:
            # Existing callers index the uint4 register array through its last axis.
            V: l.constexpr = l.BlockedLayout(
                [1, 4], [64, 1], [l.num_warps(), 1], [1, 0]
            )
            return l.convert_layout(
                l.join(l.join(v[0], v[2]), l.join(v[1], v[3])).reshape([v[0].numel, 4]),
                V,
            )

    @g.jit
    def Store(resource, voffset, soffset, data, AUX: l.constexpr, predicate=True):
        # This predicate is the calling thread's native conditional store.
        llvm_amdgcn_raw_buffer_store_v4i32(
            data,
            _resource_content(resource),
            l.where(
                predicate, (l.full((), 0, l.uint32) + voffset).to(l.uint32), 0xFFFFFFFF
            ),
            soffset,
            AUX,
        )

    @g.jit
    def LoadU64(resource, voffset, soffset, AUX: l.constexpr):
        return llvm_amdgcn_raw_buffer_load_v2i32(
            _resource_content(resource), voffset, soffset, AUX
        )

    @g.jit
    def LoadU32(resource, voffset, soffset, AUX: l.constexpr):
        return llvm_amdgcn_raw_buffer_load_i32(
            _resource_content(resource), voffset, soffset, AUX
        )

    @g.jit
    def StoreU64(resource, voffset, soffset, data, AUX: l.constexpr):
        result = llvm_amdgcn_raw_buffer_store_v2i32(
            data, _resource_content(resource), voffset, soffset, AUX
        )

    @g.jit
    def StoreU32(resource, voffset, soffset, data, AUX: l.constexpr, predicate=True):
        if isinstance(predicate, l.constexpr) and predicate:
            llvm_amdgcn_raw_buffer_store_i32(
                data, _resource_content(resource), voffset, soffset, AUX
            )
        else:
            _native_call(
                "when:llvm.amdgcn.raw.buffer.store.i32",
                "void",
                ("i32", "v4i32", "i32", "i32", "#i32", "i1"),
                (data, _resource_content(resource), voffset, soffset, AUX, predicate),
                False,
            )

    @g.jit
    def AtomicAddI32(resource, voffset, soffset, data, AUX: l.constexpr):
        result = llvm_amdgcn_raw_buffer_atomic_add_i32(
            data, _resource_content(resource), voffset, soffset, AUX
        )
        return result.to(l.int32)

    @g.jit
    def AtomicOrU32(resource, voffset, soffset, data, AUX: l.constexpr):
        result = llvm_amdgcn_raw_buffer_atomic_or_i32(
            data, _resource_content(resource), voffset, soffset, AUX
        )
        return result

    @g.jit
    def AtomicAddU64(resource, voffset, soffset, data, AUX: l.constexpr):
        result = llvm_amdgcn_raw_buffer_atomic_add_i64(
            data, _resource_content(resource), voffset, soffset, AUX
        )
        return result

    @g.jit
    def LoadLds(
        resource,
        lds_ptr,
        voffset,
        soffset,
        AUX: l.constexpr,
        SIZE: l.constexpr,
        OFFSET: l.constexpr,
        predicate=True,
    ):
        if isinstance(predicate, l.constexpr) and predicate:
            llvm_amdgcn_raw_buffer_load_lds(
                _resource_content(resource),
                lds_ptr,
                SIZE,
                voffset,
                soffset,
                OFFSET,
                AUX,
            )
        else:
            _native_call(
                "when:llvm.amdgcn.raw.buffer.load.lds",
                "void",
                ("v4i32", "p3", "#i32", "i32", "i32", "#i32", "#i32", "i1"),
                (
                    _resource_content(resource),
                    lds_ptr.to(l.uint64).to(l.uint32),
                    SIZE,
                    voffset,
                    soffset,
                    OFFSET,
                    AUX,
                    predicate,
                ),
                False,
            )


@g.jit
def GetConditionShmPtr(ptr, cond):
    return l.where(cond, ptr, l.full((), 160 * 1024, l.uint64).to(ptr.dtype))


@g.jit
def amdgcn_s_waitcnt(
    vm_cnt: l.constexpr = -1, exp_cnt: l.constexpr = -1, lgkm_cnt: l.constexpr = -1
):
    l.static_assert(vm_cnt < 64)
    l.static_assert(exp_cnt < 8)
    l.static_assert(lgkm_cnt < 16)
    vm: l.constexpr = vm_cnt & 63
    mask: l.constexpr = (
        ((vm & 48) << 10) | ((lgkm_cnt & 15) << 8) | ((exp_cnt & 7) << 4) | (vm & 15)
    )
    _native_call("llvm.amdgcn.s.waitcnt", "void", ("#i32",), (mask,), False)


@g.jit
def amdgcn_s_waitcnt_barrier(
    vm_cnt: l.constexpr = -1, exp_cnt: l.constexpr = -1, lgkm_cnt: l.constexpr = -1
):
    amdgcn_s_waitcnt(vm_cnt, exp_cnt, lgkm_cnt)
    # Keep the native s_barrier visible to Gluon before LLVM linking so LDS
    # reads cannot move across the shared-memory pipeline handoff.
    l.barrier()


@g.jit
def amdgcn_s_setprio(priority: l.constexpr):
    l.static_assert(
        priority >= 0 and priority <= 3, "s_setprio supports priority values in [0, 3]"
    )
    _native_call("llvm.amdgcn.s.setprio", "void", ("#i16",), (priority,), False)


# Additional native builtins used by the port's other headers.
@g.jit
def bitreverse(v):
    return _native_call("llvm.bitreverse.i32", "i32", ("i32",), (v,), True)


@g.jit
def amdgcn_mov_dpp(
    src,
    ctrl: l.constexpr,
    row_mask: l.constexpr,
    bank_mask: l.constexpr,
    bound_ctrl: l.constexpr,
):
    return _native_call(
        "llvm.amdgcn.mov.dpp.i32",
        "i32",
        ("i32", "#i32", "#i32", "#i32", "#i1"),
        (src, ctrl, row_mask, bank_mask, bound_ctrl),
        True,
    )


@g.jit
def amdgcn_rcpf(x):
    return _native_call("llvm.amdgcn.rcp.f32", "f32", ("f32",), (x,), True)


@g.jit
def amdgcn_readfirstlane(x):
    return _native_call("llvm.amdgcn.readfirstlane", "i32", ("i32",), (x,), True)


@g.jit
def amdgcn_ds_bpermute(index, src):
    return _native_call(
        "llvm.amdgcn.ds.bpermute", "i32", ("i32", "i32"), (index, src), True
    )


@g.jit
def amdgcn_cvt_scalef32_pk_fp4_f32(old, a, b, scale, byte: l.constexpr):
    return _native_call(
        "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
        "i32",
        ("i32", "f32", "f32", "f32", "#i32"),
        (old, a, b, scale, byte),
        True,
    )


@g.jit
def amdgcn_ds_swizzle(value, pattern: l.constexpr):
    return _native_call(
        "llvm.amdgcn.ds.swizzle", "i32", ("i32", "#i32"), (value, pattern), True
    )


@g.jit
def amdgcn_ballot(predicate):
    return _native_call("llvm.amdgcn.ballot.i64", "i64", ("i1",), (predicate,), True)


@g.jit
def amdgcn_ctz64(value):
    return _native_call("llvm.cttz.i64", "i64", ("i64", "#i1"), (value, True), True).to(
        l.uint32
    )


@g.jit
def amdgcn_thread_id(reference):
    if reference.type.is_block():
        return l.arange(
            0,
            l.num_warps() * 64,
            layout=l.BlockedLayout([1], [64], [l.num_warps()], [0]),
        ).to(l.uint32)
    else:
        return _native_call("llvm.amdgcn.workitem.id.x", "i32", (), (), True)


@g.jit
def amdgcn_shuffle(value, source_lane, width: l.constexpr = 64):
    # HIP __shfl operates inside the caller's width-sized lane subgroup.
    lane = amdgcn_thread_id(value) % 64
    source = (lane // width) * width + source_lane % width
    return amdgcn_ds_bpermute(source * 4, value.to(l.uint32, bitcast=True)).to(
        value.dtype, bitcast=True
    )


@g.jit
def amdgcn_cvt_pk_bf16_f32(a, b):
    return _native_call("asm.v_cvt_pk_bf16_f32", "i32", ("f32", "f32"), (a, b), True)


# Compiler support: native vector values are register views, not algorithms.
@builtin
def _uninitialized_like(value, dtype, _semantic):
    dtype = _unwrap_if_constexpr(dtype)
    ty = (
        distributed_type(dtype, value.type.shape, value.type.layout)
        if value.type.is_block()
        else dtype
    )
    return l.tensor(_semantic.builder.create_poison(ty.to_ir(_semantic.builder)), ty)


@builtin
def _uninitialized_scalar(dtype, _semantic):
    dtype = _unwrap_if_constexpr(dtype)
    return l.tensor(
        _semantic.builder.create_poison(dtype.to_ir(_semantic.builder)), dtype
    )


@g.jit
def _resource_content(resource):
    if len(resource) == 2:
        content = resource.content
    elif len(resource) == 4:
        content = resource
    else:
        assert len(resource) == 3
        ptr = resource[0].to(l.uint64)
        ptr_lo = _native_call(
            "llvm.amdgcn.readfirstlane", "i32", ("i32",), (ptr.to(l.uint32),), True
        )
        ptr_hi = _native_call(
            "llvm.amdgcn.readfirstlane",
            "i32",
            ("i32",),
            ((ptr >> 32).to(l.uint32),),
            True,
        )
        byte_range = _native_call(
            "llvm.amdgcn.readfirstlane",
            "i32",
            ("i32",),
            ((l.full((), 0, l.uint32) + resource[1]).to(l.uint32),),
            True,
        )
        content = (
            ptr_lo,
            ptr_hi,
            byte_range,
            (l.full((), 0, l.uint32) + resource[2]).to(l.uint32),
        )
    return content


@g.jit
def _pack_uint2(a):
    return a[0].to(l.uint32).to(l.uint64) | (a[1].to(l.uint32).to(l.uint64) << 32)


@g.jit
def _packed_mfma(kind: l.constexpr, fa, fb, c):
    if kind == "fp16":
        return _native_call(
            "llvm.amdgcn.mfma.f32.16x16x16f16",
            "v4f32",
            ("bits:v4f16", "bits:v4f16", "v4f32", "#i32", "#i32", "#i32"),
            (_pack_uint2(fa), _pack_uint2(fb), c, 0, 0, 0),
            True,
        )
    elif kind == "bf16":
        return _native_call(
            "llvm.amdgcn.mfma.f32.16x16x16bf16.1k",
            "v4f32",
            ("bits:v4i16", "bits:v4i16", "v4f32", "#i32", "#i32", "#i32"),
            (_pack_uint2(fa), _pack_uint2(fb), c, 0, 0, 0),
            True,
        )
    else:
        return _native_call(
            "llvm.amdgcn.mfma.f32.16x16x32."
            + (
                "fp8.fp8"
                if kind == "fp8_fp8"
                else "bf8.fp8" if kind == "bf8_fp8" else "fp8.bf8"
            ),
            "v4f32",
            ("i64", "i64", "v4f32", "#i32", "#i32", "#i32"),
            (_pack_uint2(fa), _pack_uint2(fb), c, 0, 0, 0),
            True,
        )


@g.jit
def _packed_mfma_tensor_acc(kind: l.constexpr, fa, fb, c):
    # Dense callers retain an MFMA-layout tensor for the native float4 array.
    # These are register views only; operands go directly to the LLVM builtin.
    NW: l.constexpr = c.type.shape[0]
    ret = c.reshape([NW, 4, 4, 16]).permute([0, 1, 3, 2]).reshape([NW * 64, 4])
    R: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [NW, 1], [1, 0])
    ret = l.convert_layout(ret, R)
    p, q = l.split(ret.reshape([NW * 64, 2, 2]))
    c0, c2 = l.split(p)
    c1, c3 = l.split(q)
    L: l.constexpr = fa[0].type.layout
    words = (
        l.convert_layout(c0, L),
        l.convert_layout(c1, L),
        l.convert_layout(c2, L),
        l.convert_layout(c3, L),
    )
    words = _packed_mfma(kind, fa, fb, words)
    ret = (
        l.join(l.join(words[0], words[2]), l.join(words[1], words[3]))
        .reshape([NW, 4, 16, 4])
        .permute([0, 1, 3, 2])
        .reshape([NW, 16, 16])
    )
    return l.convert_layout(ret, c.type.layout)


@g.jit
def _pack_float2(a):
    return a[0].to(l.uint32, bitcast=True).to(l.uint64) | (
        a[1].to(l.uint32, bitcast=True).to(l.uint64) << 32
    )


@g.jit
def _unpack_float2(a):
    return a.to(l.uint32).to(l.float32, bitcast=True), (a >> 32).to(l.uint32).to(
        l.float32, bitcast=True
    )


@cache
def _amdgcn_dequant_library():
    """Expose native vector operations to LLVM's instruction and hazard passes."""
    source = """target triple = "amdgcn-amd-amdhsa"
define i32 @petit_dequant_hmul2(i32 %a, i32 %b) alwaysinline {
  %av = bitcast i32 %a to <2 x half>
  %bv = bitcast i32 %b to <2 x half>
  %r = fmul <2 x half> %av, %bv
  %bits = bitcast <2 x half> %r to i32
  ret i32 %bits
}
"""
    manager = get_cache_manager(sha256(source.encode()).hexdigest())
    path = manager.get_file("petit_dequant.ll")
    if path is None:
        path = manager.put(source, "petit_dequant.ll", binary=False)
    return path


@cache
def _amdgcn_intrinsics_library(load_global_a):
    # Compatibility with existing launchers. Native calls carry their own library.
    return _amdgcn_dequant_library()


@cache
def _amdgcn_moe_library():
    return _amdgcn_dequant_library()


def _llvm_type(code):
    scalars = {
        "i1": ("i1", l.int1),
        "i8": ("i8", l.uint8),
        "i16": ("i16", l.uint16),
        "i32": ("i32", l.uint32),
        "i64": ("i64", l.uint64),
        "f16": ("half", l.float16),
        "f32": ("float", l.float32),
    }
    match = re.fullmatch(r"v(\d+)([if]\d+)", code)
    if match:
        n, element = int(match[1]), match[2]
        ir, dtype = scalars[element]
        return f"<{n} x {ir}>", dtype, n, ir
    if code == "void":
        return "void", l.int32, 0, "i32"
    if re.fullmatch(r"p[0-9]+", code):
        space = int(code[1:])
        bits = 32 if space in (3, 5) else 64
        return (
            f"ptr addrspace({space})",
            l.uint32 if bits == 32 else l.uint64,
            1,
            f"i{bits}",
        )
    ir, dtype = scalars[code]
    return ir, dtype, 1, ir


@cache
def _native_adapter(intrinsic, result, signature, immediates):
    """Generate a valid, specialized LLVM call with native vector types.

    Triton externs have scalar results. Tuple members are assembled/extracted
    here; LLVM inlining/CSE folds these register views into one native operation.
    No offsets, arithmetic, cache flags, or convergence semantics are invented.
    """
    guarded = intrinsic.startswith("when:")
    intrinsic = intrinsic.removeprefix("when:")
    result_ir, _, count, element_ir = _llvm_type(result)
    if guarded and count:
        raise ValueError("Conditional adapter requires a void operation")
    params, setup, operands, declared = [], [], [], []
    for index, code in enumerate(signature):
        immediate = code.startswith("#")
        bits = code.startswith("bits:")
        code = code.removeprefix("#").removeprefix("bits:")
        ir, _, n, scalar_ir = _llvm_type(code)
        declared.append(ir + (" immarg" if immediate else ""))
        name = f"%a{index}"
        if immediate:
            literal = str(int(immediates[index]))
            operands.append(f"{ir} {literal}")
            continue
        if bits:
            width = n * int(re.search(r"\d+$", code)[0])
            params.append(f"i{width} {name}")
            setup.append(f"{name}v = bitcast i{width} {name} to {ir}")
            name += "v"
        elif n > 1:
            previous = "poison"
            for j in range(n):
                params.append(f"{scalar_ir} {name}_{j}")
                setup.append(
                    f"{name}v{j} = insertelement {ir} {previous}, {scalar_ir} {name}_{j}, i32 {j}"
                )
                previous = f"{name}v{j}"
            name = previous
        elif re.fullmatch(r"p[0-9]+", code):
            params.append(f"{scalar_ir} {name}")
            setup.append(f"{name}p = inttoptr {scalar_ir} {name} to {ir}")
            name += "p"
        else:
            params.append(f"{ir} {name}")
        operands.append(f"{ir} {name}")
    if count > 1:
        params.append("i32 %word")
    declarations = []
    if guarded:
        predicate = operands.pop().split(" ", 1)[1]
        declared.pop()
        setup.extend([f"br i1 {predicate}, label %active, label %done", "active:"])
    if intrinsic == "memory.shared.i32":
        words, alignment = int(immediates[0]), int(immediates[1])
        declarations.append(
            f"@petit_shared_{words}_{alignment} = internal addrspace(3) global [{words} x i32] undef, align {alignment}"
        )
        setup.append(
            f"%result = ptrtoint ptr addrspace(3) @petit_shared_{words}_{alignment} to i32"
        )
    elif intrinsic.startswith("memory.atomic.add."):
        setup.append(
            f'%result = atomicrmw add {operands[0]}, {operands[1]} syncscope("agent") monotonic'
        )
    elif intrinsic.startswith("memory.load."):
        _, dtype, elements, _ = _llvm_type(result)
        alignment = max(1, dtype.primitive_bitwidth * elements // 8)
        if len(signature) > 1 and signature[1] == "#i32":
            alignment = int(immediates[1])
        setup.append(f"%result = load {result_ir}, {operands[0]}, align {alignment}")
    elif intrinsic == "memory.store.element.i16":
        setup.append(
            f"%element = getelementptr inbounds i16, {operands[0]}, {operands[1]}"
        )
        setup.append(
            f"store {operands[2]}, ptr addrspace(3) %element, align {int(immediates[3])}"
        )
    elif intrinsic.startswith("memory.store."):
        value_code = signature[1]
        _, dtype, elements, _ = _llvm_type(value_code)
        alignment = max(1, dtype.primitive_bitwidth * elements // 8)
        if len(signature) > 2 and signature[2] == "#i32":
            alignment = int(immediates[2])
        setup.append(f"store {operands[1]}, {operands[0]}, align {alignment}")
    elif intrinsic.startswith("fence."):
        _, ordering, scope = intrinsic.split(".")
        syncscope = ' syncscope("agent")' if scope == "agent" else ""
        setup.append(f"fence{syncscope} {ordering}")
    elif intrinsic == "buffer.atomic.pk.add.bf16":
        setup.append(
            'call void asm sideeffect "buffer_atomic_pk_add_bf16 $2, $1, $0, 0 offen", "s,v,v,~{memory}"('
            + ", ".join(operands)
            + ")"
        )
    elif intrinsic == "compiler.memory.barrier":
        setup.append('call void asm sideeffect "", "~{memory}"()')
    elif intrinsic == "buffer.wbl2.sc0.sc1":
        setup.append('call void asm sideeffect "buffer_wbl2 sc0 sc1", "~{memory}"()')
    elif intrinsic == "s.sleep.1":
        setup.append('call void asm sideeffect "s_sleep 1", "~{memory}"()')
    elif intrinsic.startswith("asm."):
        op = intrinsic.removeprefix("asm.")
        constraints = (
            "=v,v,v"
            if op in ("v_pk_add_i16", "v_cvt_pk_bf16_f32")
            else "=v,v,v,v" if op.endswith(".vector") else "=v,v,v,r"
        )
        op = op.removesuffix(".vector")
        registers = ", ".join(f"${i}" for i in range(len(operands) + 1))
        setup.append(
            f'%result = call i32 asm "{op} {registers};", "{constraints}"({", ".join(operands)}) convergent nounwind memory(none)'
        )
    elif intrinsic in ("fmul", "fadd"):
        setup.append(
            f"%result = {intrinsic} {operands[0]}, {operands[1].split('> ', 1)[1]}"
        )
    else:
        declarations.append(f"declare {result_ir} @{intrinsic}({', '.join(declared)})")
        setup.append(
            ("%result = " if count else "")
            + f"call {result_ir} @{intrinsic}({', '.join(operands)})"
        )
    if count > 1:
        setup.append(f"%value = extractelement {result_ir} %result, i32 %word")
        setup.append(f"ret {element_ir} %value")
    elif count:
        setup.append(f"ret {result_ir} %result")
    else:
        if guarded:
            setup.extend(["br label %done", "done:"])
        setup.append("ret i32 0")
    key = sha256(
        repr((guarded, intrinsic, result, signature, immediates)).encode()
    ).hexdigest()
    library = "petit_native_" + key
    symbol = library + "_call"
    convergent = any(
        s in intrinsic
        for s in (
            "asm.",
            "mfma.",
            "ds.",
            "mov.dpp",
            "readfirstlane",
            "barrier",
            "ballot",
        )
    )
    source = "\n".join(
        [
            'target triple = "amdgcn-amd-amdhsa"',
            *declarations,
            f"define {element_ir} @{symbol}({', '.join(params)}) alwaysinline"
            + (" convergent" if convergent else "")
            + " {",
            *setup,
            "}",
        ]
    )
    manager = get_cache_manager(sha256(source.encode()).hexdigest())
    path = manager.get_file("native_intrinsic.ll")
    if path is None:
        path = manager.put(source, "native_intrinsic.ll", binary=False)
    return library, path, symbol


@builtin
def _native_call(intrinsic, result, signature, args, pure, _semantic):
    intrinsic, result, pure = map(_unwrap_if_constexpr, (intrinsic, result, pure))
    signature = tuple(_unwrap_if_constexpr(x) for x in signature)
    values, immediates = [], []
    for code, value in zip(signature, args):
        if code.startswith("#"):
            immediates.append(int(_unwrap_if_constexpr(value)))
            continue
        immediates.append(None)
        bits = code.startswith("bits:")
        code = code.removeprefix("bits:")
        _, dtype, n, _ = _llvm_type(code)
        if bits:
            dtype = l.uint64 if n * dtype.primitive_bitwidth == 64 else l.uint32
            n = 1
        for member in value if n > 1 else (value,):
            member = _semantic.to_tensor(_unwrap_if_constexpr(member))
            if member.dtype.is_ptr():
                member = _semantic.cast(member, l.uint64)
            values.append(_semantic.cast(member, dtype))
    if values:
        for i in range(1, len(values)):
            values[0], values[i] = _semantic.broadcast_impl_value(values[0], values[i])
        for i in range(1, len(values)):
            values[i], _ = _semantic.broadcast_impl_value(values[i], values[0])
    library, path, symbol = _native_adapter(
        intrinsic, result, signature, tuple(immediates)
    )
    _, dtype, count, _ = _llvm_type(result)
    ty = (
        distributed_type(dtype, values[0].type.shape, values[0].type.layout)
        if values and values[0].type.is_block()
        else dtype
    )
    outputs = []
    for word in range(max(1, count)):
        handles = [v.handle for v in values]
        if count > 1:
            index = _semantic.to_tensor(word)
            if values:
                index, _ = _semantic.broadcast_impl_value(index, values[0])
            handles.append(index.handle)
        handle = _semantic.builder.create_extern_elementwise(
            library, path, symbol, handles, ty.to_ir(_semantic.builder), pure
        )
        outputs.append(l.tensor(handle, ty))
    return l.tuple(outputs) if count > 1 else outputs[0] if count else None


@builtin
def _native_load_scalar(ptr, alignment=None, _semantic=None):
    """A scalar native dereference, without a one-element LLVM vector wrapper."""
    dtype = ptr.dtype.element_ty
    code = "f32" if dtype == l.float32 else "i" + str(dtype.primitive_bitwidth)
    signature = ("p" + str(_unwrap_if_constexpr(ptr.dtype.address_space)),)
    args = (ptr,)
    if _unwrap_if_constexpr(alignment) is not None:
        signature += ("#i32",)
        args += (alignment,)
    value = _native_call(
        "memory.load." + code, code, signature, args, False, _semantic=_semantic
    )
    return _semantic.bitcast(value, dtype)


@g.jit
def _native_shared_memory(words: l.constexpr, alignment: l.constexpr = 16):
    """One native static shared allocation per kernel, retaining its LLVM base."""
    return (
        _native_call(
            "memory.shared.i32", "i32", ("#i32", "#i32"), (words, alignment), False
        )
        .to(l.uint64)
        .to(l.pointer_type(l.uint32, 3))
    )


@builtin
def _native_load_vector4(ptr, _semantic):
    """Native aligned uint4/float4 dereference, retaining its LLVM vector type."""
    dtype = ptr.dtype.element_ty
    if dtype not in (l.uint32, l.float32):
        raise TypeError("Native vector4 access requires uint32 or float32 elements")
    code = "v4f32" if dtype == l.float32 else "v4i32"
    return _native_call(
        "memory.load." + code,
        code,
        ("p" + str(_unwrap_if_constexpr(ptr.dtype.address_space)),),
        (ptr,),
        False,
        _semantic=_semantic,
    )


@builtin
def _native_store_vector4(ptr, value, _semantic):
    """Native aligned uint4/float4 assignment; ptr addresses a 16-byte element."""
    dtype = ptr.dtype.element_ty
    if dtype not in (l.uint32, l.float32):
        raise TypeError("Native vector4 access requires uint32 or float32 elements")
    code = "v4f32" if dtype == l.float32 else "v4i32"
    return _native_call(
        "memory.store." + code,
        "void",
        ("p" + str(_unwrap_if_constexpr(ptr.dtype.address_space)), code),
        (ptr, value),
        False,
        _semantic=_semantic,
    )


@builtin
def _native_load_uint2(ptr, _semantic):
    """Native aligned uint2 LDS dereference."""
    return _native_call(
        "memory.load.v2i32", "v2i32", ("p3",), (ptr,), False, _semantic=_semantic
    )


@builtin
def _native_store_uint2(ptr, value, _semantic):
    """Native aligned uint2 LDS assignment."""
    return _native_call(
        "memory.store.v2i32",
        "void",
        ("p3", "v2i32"),
        (ptr, value),
        False,
        _semantic=_semantic,
    )


@g.jit
def _native_store_ushort_component(array, index, value, component: l.constexpr):
    """Scalar ushort assignment in an eight-byte-aligned native fragment.

    Retain the array base, native in-bounds indexing and the alignment Clang
    gets from Shm::output without changing the component loop.
    """
    alignment: l.constexpr = 8 if component == 0 else 4 if component == 2 else 2
    _native_call(
        "memory.store.element.i16",
        "void",
        ("p3", "i32", "i16", "#i32"),
        (array, index, value, alignment),
        False,
    )


def _install_native_library_linker():
    """Honor per-operation LLVM libraries through Triton 3.8's pipeline hook.

    AMD 3.8 only links libraries from launch options, ignoring an extern op's
    libpath. Specializations are created during JIT, after those options exist.
    Merge only our content-addressed libraries before LLVM lowering. The hook's
    cache key includes this source; no installed Triton files are changed.
    """
    previous = triton.knobs.runtime.add_stages_inspection_hook
    source_hash = sha256(Path(__file__).read_bytes()).hexdigest()

    def hook(*args):
        if not args:
            key, digest = previous() if previous else ("", "")
            return key + source_hash, digest + source_hash
        backend, stages, options, _language, _capability = args
        if getattr(options, "backend_name", None) != "hip":
            if previous:
                previous(*args)
            return
        original = stages["llir"]

        def lower(src, metadata):
            libraries = dict(
                re.findall(
                    r'libname = "(petit_native_[0-9a-f]+)", libpath = "([^"]+)"',
                    str(src),
                )
            )
            if not libraries:
                return original(src, metadata)
            if options.arch != "gfx950":
                raise ValueError(
                    "Petit intrinsic capability flags currently target gfx950"
                )
            merged = dict(options.extern_libs)
            merged.update(libraries)
            return backend.make_llir(
                src, metadata, replace(options, extern_libs=merged)
            )

        stages["llir"] = lower
        if previous:
            previous(*args)

    triton.knobs.runtime.add_stages_inspection_hook = hook


_install_native_library_linker()
