# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Device-side program aggregates for the pipelined ragged GEMM:
MoEConfig, async-copy descriptors, and the pipelined/slice-MN/slice-N
program shapes with their descriptor factories."""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _SCALE_LOAD_MODES,
    _SCALE_PRESHUFFLE_FACTOR,
    composition,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._layouts import (
    _load_layout,
    _scale_async_blocked_layout,
    get_mfma_layout,
)


@aggregate
class MoEConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    NUM_WARPS: gl.constexpr

    DIV_FACTOR_X: gl.constexpr
    DIV_FACTOR_W: gl.constexpr
    DTYPE_X: gl.constexpr
    DTYPE_W: gl.constexpr

    W_TRANSPOSE: gl.constexpr
    W_PRESHUFFLED: gl.constexpr
    W_VIA_VGPR: gl.constexpr
    W_PREFETCH: gl.constexpr
    NUM_BUFFERS: gl.constexpr

    SCALE_BLOCK: gl.constexpr
    WITH_X_MX_SCALE: gl.constexpr
    WITH_W_MX_SCALE: gl.constexpr
    SCALE_LOAD_MODE: gl.constexpr
    X_SCALE_VIA_LDS: gl.constexpr
    W_SCALE_VIA_LDS: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    BLOCK_M_PRESHUFFLED: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr
    shared_layout_w_half_n: gl.constexpr
    shared_layout_x_half_m: gl.constexpr

    NUM_SUBTILES: gl.constexpr
    EVEN_K: gl.constexpr
    K_ITERS: gl.constexpr
    USE_GATHER: gl.constexpr
    USE_MFMA_SCALED: gl.constexpr
    NUM_LOADS_IN_BATCH: gl.constexpr

    shared_layout_x: gl.constexpr
    dot_layout_x: gl.constexpr

    shared_layout_w: gl.constexpr
    dot_layout_w: gl.constexpr

    layout_x_scale: gl.constexpr
    layout_w_scale: gl.constexpr

    shared_layout_x_scale: gl.constexpr
    shared_layout_w_scale: gl.constexpr
    load_layout_x_scale: gl.constexpr
    load_layout_w_scale: gl.constexpr

    acc_layout: gl.constexpr

    index_type: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE_X,
        DTYPE_W,
        SCALE_BLOCK,
        NUM_BUFFERS,
        W_TRANSPOSE,
        WITH_X_MX_SCALE,
        WITH_W_MX_SCALE,
        SCALE_LOAD_MODE,
        index_type,
        NUM_SUBTILES=(1, 1, 1),
        EVEN_K=True,
        K_ITERS=0,
        USE_GATHER=False,
        NUM_WARPS=4,
        W_PRESHUFFLED=False,
        W_VIA_VGPR=False,
        W_PREFETCH=True,
        X_SCALE_VIA_LDS=None,
        W_SCALE_VIA_LDS=None,
    ):
        if SCALE_LOAD_MODE not in _SCALE_LOAD_MODES:
            raise ValueError(
                f"SCALE_LOAD_MODE must be one of {_SCALE_LOAD_MODES}, "
                f"got {SCALE_LOAD_MODE!r}"
            )
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)
        self.W_TRANSPOSE = gl.constexpr(W_TRANSPOSE)
        self.W_PRESHUFFLED = gl.constexpr(W_PRESHUFFLED)
        self.W_VIA_VGPR = gl.constexpr(W_VIA_VGPR)
        self.W_PREFETCH = gl.constexpr(W_PREFETCH)
        self.WITH_X_MX_SCALE = gl.constexpr(WITH_X_MX_SCALE)
        self.WITH_W_MX_SCALE = gl.constexpr(WITH_W_MX_SCALE)
        self.SCALE_LOAD_MODE = gl.constexpr(SCALE_LOAD_MODE)
        self.SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
        self.DIV_FACTOR_X = gl.constexpr(2 if DTYPE_X == "e2m1" else 1)
        self.DIV_FACTOR_W = gl.constexpr(2 if DTYPE_W == "e2m1" else 1)
        self.DTYPE_X = gl.constexpr(DTYPE_X)
        self.DTYPE_W = gl.constexpr(DTYPE_W)

        if X_SCALE_VIA_LDS is None:
            X_SCALE_VIA_LDS = SCALE_LOAD_MODE == "swizzle" and WITH_X_MX_SCALE
        if W_SCALE_VIA_LDS is None:
            W_SCALE_VIA_LDS = SCALE_LOAD_MODE == "swizzle" and WITH_W_MX_SCALE
        _scale_via_lds = X_SCALE_VIA_LDS or W_SCALE_VIA_LDS
        self.X_SCALE_VIA_LDS = gl.constexpr(X_SCALE_VIA_LDS)
        self.W_SCALE_VIA_LDS = gl.constexpr(W_SCALE_VIA_LDS)
        self.PRESHUFFLE_FACTOR = gl.constexpr(_SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_M_PRESHUFFLED = gl.constexpr(BLOCK_M // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // _SCALE_PRESHUFFLE_FACTOR)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(
            (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR
        )

        self.NUM_SUBTILES = gl.constexpr(NUM_SUBTILES)
        self.EVEN_K = gl.constexpr(EVEN_K)
        self.K_ITERS = gl.constexpr(K_ITERS)
        self.USE_GATHER = gl.constexpr(USE_GATHER)
        _SCALED_FORMATS = ("e2m1", "e4m3", "e5m2")
        self.USE_MFMA_SCALED = gl.constexpr(
            DTYPE_X in _SCALED_FORMATS and DTYPE_W in _SCALED_FORMATS
        )
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)

        num_loads = 1  # x
        if not W_VIA_VGPR:
            num_loads += 1  # w (LDS path)
        if _scale_via_lds:
            if X_SCALE_VIA_LDS:
                num_loads += 1
            if W_SCALE_VIA_LDS:
                num_loads += 1
        self.NUM_LOADS_IN_BATCH = gl.constexpr(num_loads)

        BLOCK_K_SCALE = BLOCK_K // SCALE_BLOCK
        self.index_type = gl.constexpr(index_type)

        MFMA_LAYOUT: gl.constexpr = get_mfma_layout(
            NUM_WARPS,
            self.USE_MFMA_SCALED,
            scale_preshuffle=_scale_via_lds,
            block_m=BLOCK_M,
            w_via_vgpr=W_VIA_VGPR or W_PRESHUFFLED,
        )

        DOT_K_WIDTH_X: gl.constexpr = 16 if self.USE_MFMA_SCALED else 8
        DOT_K_WIDTH_W: gl.constexpr = 16 if self.USE_MFMA_SCALED else 8

        NUM_SUBTILES_M = self.NUM_SUBTILES[0]
        NUM_SUBTILES_N = self.NUM_SUBTILES[1]
        NUM_SUBTILES_K = self.NUM_SUBTILES[2]

        self.dot_layout_x = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=0, parent=MFMA_LAYOUT, k_width=DOT_K_WIDTH_X
            )
        )
        self.dot_layout_w = gl.constexpr(
            gl.DotOperandLayout(
                operand_index=1, parent=MFMA_LAYOUT, k_width=DOT_K_WIDTH_W
            )
        )
        if self.USE_MFMA_SCALED:
            self.layout_x_scale = gl.constexpr(
                gl.amd.cdna4.get_mfma_scale_layout(
                    self.dot_layout_x,
                    [BLOCK_M // NUM_SUBTILES_M, BLOCK_K_SCALE // NUM_SUBTILES_K],
                )
            )
            self.layout_w_scale = gl.constexpr(
                gl.amd.cdna4.get_mfma_scale_layout(
                    self.dot_layout_w,
                    [BLOCK_N // NUM_SUBTILES_N, BLOCK_K_SCALE // NUM_SUBTILES_K],
                )
            )
        else:
            self.layout_x_scale = gl.constexpr(0)
            self.layout_w_scale = gl.constexpr(0)
        self.acc_layout = gl.constexpr(MFMA_LAYOUT)

        BLOCK_K_PACKED_X_HOST = BLOCK_K // self.DIV_FACTOR_X
        BLOCK_K_PACKED_W_HOST = BLOCK_K // self.DIV_FACTOR_W

        def _row_major_offsets(H, W):
            H = int(H)
            W = int(W)
            inner = [[0, 1 << i] for i in range(W.bit_length() - 1)]
            outer = [[1 << i, 0] for i in range(H.bit_length() - 1)]
            return inner + outer

        self.shared_layout_x = gl.constexpr(
            gl.PaddedSharedLayout(
                [[1024, 32]],
                _row_major_offsets(BLOCK_M, BLOCK_K_PACKED_X_HOST),
                [],
                [BLOCK_M, BLOCK_K_PACKED_X_HOST],
            )
        )
        if W_PRESHUFFLED:
            w_shape = [BLOCK_N // 16, BLOCK_K_PACKED_W_HOST * 16]
        elif W_TRANSPOSE:
            w_shape = [BLOCK_N, BLOCK_K_PACKED_W_HOST]
        else:
            w_shape = [BLOCK_K_PACKED_W_HOST, BLOCK_N]
        self.shared_layout_w = gl.constexpr(
            gl.PaddedSharedLayout(
                [[1024, 32]],
                _row_major_offsets(w_shape[0], w_shape[1]),
                [],
                w_shape,
            )
        )

        if W_PRESHUFFLED:
            w_half_shape = [BLOCK_N // 2 // 16, BLOCK_K_PACKED_W_HOST * 16]
        elif W_TRANSPOSE:
            w_half_shape = [BLOCK_N // 2, BLOCK_K_PACKED_W_HOST]
        else:
            w_half_shape = [BLOCK_K_PACKED_W_HOST, BLOCK_N // 2]
        if (BLOCK_N // 2) >= 1 and BLOCK_K_PACKED_W_HOST >= 1:
            self.shared_layout_w_half_n = gl.constexpr(
                gl.PaddedSharedLayout(
                    [[1024, 32]],
                    _row_major_offsets(w_half_shape[0], w_half_shape[1]),
                    [],
                    w_half_shape,
                )
            )
        else:
            self.shared_layout_w_half_n = gl.constexpr(0)

        if (BLOCK_M // 2) >= 1 and BLOCK_K_PACKED_X_HOST >= 1:
            self.shared_layout_x_half_m = gl.constexpr(
                gl.PaddedSharedLayout(
                    [[1024, 32]],
                    _row_major_offsets(BLOCK_M // 2, BLOCK_K_PACKED_X_HOST),
                    [],
                    [BLOCK_M // 2, BLOCK_K_PACKED_X_HOST],
                )
            )
        else:
            self.shared_layout_x_half_m = gl.constexpr(0)

        if _scale_via_lds:
            self.shared_layout_x_scale = gl.constexpr(
                gl.SwizzledSharedLayout(4, 1, 1, order=[1, 0])
            )
            self.shared_layout_w_scale = gl.constexpr(
                gl.SwizzledSharedLayout(4, 1, 1, order=[1, 0])
            )
            self.load_layout_x_scale = gl.constexpr(
                _scale_async_blocked_layout(
                    BLOCK_M // _SCALE_PRESHUFFLE_FACTOR,
                    (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR,
                    NUM_WARPS,
                )
            )
            self.load_layout_w_scale = gl.constexpr(
                _scale_async_blocked_layout(
                    BLOCK_N // _SCALE_PRESHUFFLE_FACTOR,
                    (BLOCK_K // SCALE_BLOCK) * _SCALE_PRESHUFFLE_FACTOR,
                    NUM_WARPS,
                )
            )
        else:
            self.shared_layout_x_scale = gl.constexpr(0)
            self.shared_layout_w_scale = gl.constexpr(0)
            self.load_layout_x_scale = gl.constexpr(0)
            self.load_layout_w_scale = gl.constexpr(0)


@aggregate
class MoEProgramBase:

    @gluon.constexpr_function
    def __init__(self):
        pass

    @gluon.jit
    def mfma(self, x, scale_x, w, scale_w, accumulator):
        cfg = self.cfg
        if cfg.USE_MFMA_SCALED:
            return gl.amd.cdna4.mfma_scaled(
                x, scale_x, cfg.DTYPE_X, w, scale_w, cfg.DTYPE_W, accumulator
            )
        else:
            return gl.amd.cdna4.mfma(x, w, accumulator)

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        self.x_desc.issue_async_load(load_idx, self.x_buffer, pred, USE_MASK=USE_MASK)
        if not cfg.W_VIA_VGPR:
            self.w_desc.issue_async_load(
                load_idx, self.w_buffer, pred, USE_MASK=USE_MASK
            )
        scale_via_lds: gl.constexpr = cfg.X_SCALE_VIA_LDS or cfg.W_SCALE_VIA_LDS
        if scale_via_lds:
            if cfg.X_SCALE_VIA_LDS:
                self.x_scale_desc.issue_async_load(
                    load_idx, self.x_scale_buffer, pred, USE_MASK=USE_MASK
                )
            if cfg.W_SCALE_VIA_LDS:
                self.w_scale_desc.issue_async_load(
                    load_idx, self.w_scale_buffer, pred, USE_MASK=USE_MASK
                )
        return load_idx + 1

    @gluon.jit
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt * self.cfg.NUM_LOADS_IN_BATCH)


@gluon.aggregate
class AsyncCopyDescriptor:
    cfg: MoEConfig
    op_idx: gl.constexpr
    ptr: gl.tensor
    dtype: gl.constexpr
    stride_k: gl.tensor
    stride_nonk: gl.tensor
    offsets: gl.tensor
    off_k: gl.tensor
    off_nonk: gl.tensor
    masks_nonk: gl.tensor
    k_limit: gl.tensor
    base_offset: gl.tensor
    BLOCK_K: gl.constexpr
    cache_modifier: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        op_idx,
        BLOCK_K,
        ptr,
        dtype,
        stride_k,
        stride_nonk,
        offsets,
        off_k,
        off_nonk,
        masks_nonk,
        k_limit,
        base_offset,
        cache_modifier="",
    ):
        self.cfg = cfg
        self.op_idx = gl.constexpr(op_idx)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.ptr = ptr
        self.dtype = gl.constexpr(dtype)
        self.stride_k = stride_k
        self.stride_nonk = stride_nonk
        self.offsets = offsets
        self.off_k = off_k
        self.off_nonk = off_nonk
        self.masks_nonk = masks_nonk
        self.k_limit = k_limit
        self.base_offset = base_offset
        self.cache_modifier = gl.constexpr(cache_modifier)

    @gluon.jit
    def initialize(
        cfg: MoEConfig,
        op_idx: gl.constexpr,
        BLOCK_K: gl.constexpr,
        ptr,
        off_nonk,
        off_k,
        stride_nonk,
        stride_k,
        masks_nonk,
        k_limit,
        base_offset=0,
        cache_modifier: gl.constexpr = "",
    ):
        base_offset_t = gl.to_tensor(base_offset)
        ptr = ptr + base_offset_t
        offsets = (
            gl.expand_dims(off_k, op_idx) * stride_k
            + gl.expand_dims(off_nonk, 1 - op_idx) * stride_nonk
        )
        dtype: gl.constexpr = ptr.dtype.element_ty
        stride_k_t = gl.to_tensor(stride_k)
        stride_nonk_t = gl.to_tensor(stride_nonk)
        return AsyncCopyDescriptor(
            cfg,
            op_idx,
            BLOCK_K,
            ptr,
            dtype,
            stride_k_t,
            stride_nonk_t,
            offsets,
            off_k,
            off_nonk,
            masks_nonk,
            k_limit,
            base_offset_t,
            cache_modifier,
        )

    @gluon.jit
    def issue_async_load(
        self,
        idx,
        buffer,
        pred=1,
        USE_MASK: gl.constexpr = -1,
        COMMIT: gl.constexpr = 1,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        EVEN_K: gl.constexpr = self.cfg.EVEN_K
        if USE_MASK == -1:
            USE_MASK_RESOLVED: gl.constexpr = 0 if EVEN_K else 1
        else:
            USE_MASK_RESOLVED: gl.constexpr = USE_MASK
        CACHE_MODIFIER: gl.constexpr = self.cache_modifier
        off_k_step = idx * self.BLOCK_K
        offsets = self.offsets + off_k_step * self.stride_k
        if USE_MASK_RESOLVED == 0:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                buffer.index(idx % NUM_BUFFERS),
                self.ptr,
                offsets,
                cache_modifier=CACHE_MODIFIER,
            )
        else:
            # IMPORTANT: do not pass ``other=0`` here. A non-null
            # ``other`` causes the lowering to emit per-element
            # branches around each ``buffer.load.async.lds`` which
            # break ``SIInsertWaitcnts`` static counting and collapse
            # the async pipeline to ``s_waitcnt vmcnt(0)``. We rely on
            # the buffer descriptor's ``numRecords`` OOB check to zero
            # masked-out lanes in LDS.
            mask_k = gl.expand_dims(off_k_step + self.off_k, self.op_idx) < self.k_limit
            mask = mask_k & self.masks_nonk
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                buffer.index(idx % NUM_BUFFERS),
                self.ptr,
                offsets,
                mask=mask,
                cache_modifier=CACHE_MODIFIER,
            )
        if COMMIT == 1:
            gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def issue_local_load(
        self, idx, buffer, layout: gl.constexpr, do_permute: gl.constexpr = False
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        if do_permute:
            slot = slot.permute([1, 0])
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot, layout)

    @gluon.jit
    def issue_local_load_m_swizzle(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_M: gl.constexpr,
    ):
        if BLOCK_M == 32:
            GROUPS_M: gl.constexpr = 2
        else:
            gl.static_assert(
                BLOCK_M == 64 or BLOCK_M == 128,
                "M-swizzled local load supports BLOCK_M in {32, 64, 128}",
            )
            GROUPS_M: gl.constexpr = 4
        ROWS_PER_GROUP: gl.constexpr = BLOCK_M // GROUPS_M
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_view = (
            slot.reshape((ROWS_PER_GROUP, GROUPS_M, self.BLOCK_K))
            .permute((1, 0, 2))
            .reshape((BLOCK_M, self.BLOCK_K))
        )
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot_view, layout)

    @gluon.jit
    def issue_local_load_unswizzle(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_NONK_PS: gl.constexpr,
        BLOCK_NONK: gl.constexpr,
        BLOCK_K_SCALE: gl.constexpr,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_7d = slot.reshape((BLOCK_NONK_PS, BLOCK_K_SCALE // 8, 4, 16, 2, 2, 1))
        slot_perm = slot_7d.permute((0, 5, 3, 1, 4, 2, 6))
        slot_2d = slot_perm.reshape((BLOCK_NONK, BLOCK_K_SCALE))
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot_2d, layout)

    @gluon.jit
    def issue_local_load_unswizzle_sub(
        self,
        idx,
        buffer,
        layout: gl.constexpr,
        BLOCK_NONK_PS: gl.constexpr,
        BLOCK_NONK: gl.constexpr,
        BLOCK_K_SCALE: gl.constexpr,
        SUBTILE_NONK: gl.constexpr,
        subtile_start_nonk: gl.constexpr,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_view = (
            slot.reshape((BLOCK_NONK_PS, BLOCK_K_SCALE // 8, 4, 16, 2, 2, 1))
            .permute((0, 5, 3, 1, 4, 2, 6))
            .reshape((BLOCK_NONK, BLOCK_K_SCALE))
        )
        return gl.amd.cdna4.async_copy.load_shared_relaxed(
            slot_view.slice(subtile_start_nonk, SUBTILE_NONK, 0), layout
        )


@gluon.aggregate
class WVgprDescriptor:
    cfg: MoEConfig
    ptr: gl.tensor
    stride_k: gl.tensor  # = N (bytes between consecutive K-slabs)
    offsets: gl.tensor  # [LOAD_BN//N_LANE, BLOCK_K*N_LANE]
    pred: gl.tensor  # int1 scalar (broadcast to a per-element mask)
    BLOCK_K: gl.constexpr  # = BLOCK_K_W; mirrors AsyncCopyDescriptor
    LOAD_BN: gl.constexpr  # N width per load; SUB_BN under sliceN

    @gluon.constexpr_function
    def __init__(
        self, cfg: MoEConfig, BLOCK_K, ptr, stride_k, offsets, pred, LOAD_BN=None
    ):
        self.cfg = cfg
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.LOAD_BN = gl.constexpr(LOAD_BN if LOAD_BN is not None else cfg.BLOCK_N)
        self.ptr = ptr
        self.stride_k = stride_k
        self.offsets = offsets
        self.pred = pred

    @gluon.jit
    def issue_global_load_to_vgpr(self, idx, dot_layout: gl.constexpr):
        BLOCK_K_W: gl.constexpr = self.BLOCK_K
        LOAD_BN: gl.constexpr = self.LOAD_BN

        # idx-th K-slab; per-iter shift folds into the scalar ptr so
        # ``offsets`` stays compile-time constant.
        k_iter_offset = idx * BLOCK_K_W * self.stride_k
        ptr_iter = self.ptr + k_iter_offset

        # ``mask`` is a scalar bool; buffer_load broadcasts it to the
        # offsets layout. Hardware OOB masking returns 0 for masked
        # lanes, which is what we want when ``pred=False``.
        tile_flat = gl.amd.cdna4.buffer_load(
            ptr=ptr_iter, offsets=self.offsets, mask=self.pred
        )

        # 5-D HBM layout -> (BLOCK_K_W, LOAD_BN) MFMA-ready.
        tile_5d = tile_flat.reshape(
            LOAD_BN // 16,
            BLOCK_K_W // 64,
            4,
            16,
            16,
        )
        tile_perm = tile_5d.permute(0, 3, 1, 2, 4)
        tile_2d = tile_perm.reshape(LOAD_BN, BLOCK_K_W)
        tile_t = tile_2d.trans(1, 0)

        return gl.convert_layout(tile_t, dot_layout, assert_trivial=True)


@gluon.aggregate
class WPreshuffledLdsDescriptor:
    cfg: MoEConfig
    ptr: gl.tensor
    dtype: gl.constexpr
    stride_k: gl.tensor  # = N bytes between consecutive K slabs.
    offsets: gl.tensor  # [LOAD_BN//16, BLOCK_K*16] in preshuffled tile order.
    pred: gl.tensor
    BLOCK_K: gl.constexpr
    LOAD_BN: gl.constexpr
    load_layout: gl.constexpr
    cache_modifier: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        BLOCK_K,
        ptr,
        dtype,
        stride_k,
        offsets,
        pred,
        load_layout,
        LOAD_BN=None,
        cache_modifier="",
    ):
        self.cfg = cfg
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.LOAD_BN = gl.constexpr(LOAD_BN if LOAD_BN is not None else cfg.BLOCK_N)
        self.ptr = ptr
        self.dtype = gl.constexpr(dtype)
        self.stride_k = stride_k
        self.offsets = offsets
        self.pred = pred
        self.load_layout = gl.constexpr(load_layout)
        self.cache_modifier = gl.constexpr(cache_modifier)

    @gluon.jit
    def issue_async_load(
        self,
        idx,
        buffer,
        pred=1,
        USE_MASK: gl.constexpr = -1,
        COMMIT: gl.constexpr = 1,
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        k_iter_offset = idx * self.BLOCK_K * self.stride_k
        offsets = self.offsets + k_iter_offset
        CACHE_MODIFIER: gl.constexpr = self.cache_modifier
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            buffer.index(idx % NUM_BUFFERS),
            self.ptr,
            offsets,
            mask=self.pred,
            cache_modifier=CACHE_MODIFIER,
        )
        if COMMIT == 1:
            gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def issue_local_load(
        self, idx, buffer, layout: gl.constexpr, do_permute: gl.constexpr = False
    ):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        BLOCK_K_W: gl.constexpr = self.BLOCK_K
        LOAD_BN: gl.constexpr = self.LOAD_BN
        slot = buffer.index(idx % NUM_BUFFERS)
        slot_5d = slot.reshape((LOAD_BN // 16, BLOCK_K_W // 64, 4, 16, 16))
        slot_perm = slot_5d.permute((0, 3, 1, 2, 4))
        slot_2d = slot_perm.reshape((LOAD_BN, BLOCK_K_W))
        slot_t = slot_2d.permute((1, 0))
        return gl.amd.cdna4.async_copy.load_shared_relaxed(slot_t, layout)


@gluon.jit
def _load_scale_tile_via_gl_load(desc, mfma_idx):
    EVEN_K: gl.constexpr = desc.cfg.EVEN_K
    off_k_step = mfma_idx * desc.BLOCK_K
    base = desc.ptr + off_k_step * desc.stride_k
    if EVEN_K:
        mask = desc.masks_nonk
    else:
        mask_k = gl.expand_dims(off_k_step + desc.off_k, desc.op_idx) < desc.k_limit
        mask = mask_k & desc.masks_nonk
    return gl.load(base + desc.offsets, mask=mask, other=0)


@gluon.jit
def _load_scale_subtile_via_gl_load(
    desc, mfma_idx, subtile_start_nonk: gl.constexpr, SUBTILE_NONK: gl.constexpr
):
    EVEN_K: gl.constexpr = desc.cfg.EVEN_K
    off_k_step = mfma_idx * desc.BLOCK_K
    base = desc.ptr + off_k_step * desc.stride_k
    offsets = desc.offsets.slice(subtile_start_nonk, SUBTILE_NONK, 0)
    masks_nonk = desc.masks_nonk.slice(subtile_start_nonk, SUBTILE_NONK, 0)
    if EVEN_K:
        mask = masks_nonk
    else:
        mask_k = gl.expand_dims(off_k_step + desc.off_k, desc.op_idx) < desc.k_limit
        mask = mask_k & masks_nonk
    return gl.load(base + offsets, mask=mask, other=0)


@composition
@gluon.aggregate
class MoEPipelinedProgram:
    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc: AsyncCopyDescriptor
    w_desc: AsyncCopyDescriptor | WVgprDescriptor | WPreshuffledLdsDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer,
        w_buffer,
        x_scale_buffer,
        w_scale_buffer,
        x_desc,
        w_desc,
        x_scale_desc,
        w_scale_desc,
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.x_scale_buffer = x_scale_buffer if cfg.X_SCALE_VIA_LDS else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.W_SCALE_VIA_LDS else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS

        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(
            x_desc.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x,
        )
        # W_VIA_VGPR: skip W's LDS slot; K-loop does HBM->VGPR direct.
        if cfg.W_VIA_VGPR:
            w_buffer = gl.constexpr(0)
        elif cfg.W_PRESHUFFLED:
            w_buffer = gl.allocate_shared_memory(
                w_desc.dtype,
                shape=[NUM_BUFFERS, cfg.BLOCK_N // 16, BLOCK_K_PACKED_W * 16],
                layout=cfg.shared_layout_w,
            )
        else:
            w_buffer = gl.allocate_shared_memory(
                w_desc.dtype,
                shape=(
                    [NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
                    if cfg.W_TRANSPOSE
                    else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N]
                ),
                layout=cfg.shared_layout_w,
            )

        if cfg.X_SCALE_VIA_LDS:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_M_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.W_SCALE_VIA_LDS:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoEPipelinedProgram(
            cfg,
            x_buffer,
            w_buffer,
            x_scale_buffer,
            w_scale_buffer,
            x_desc,
            w_desc,
            x_scale_desc,
            w_scale_desc,
        )

    @gluon.jit
    def _issue_w_vgpr(self, mfma_idx):
        cfg = self.cfg
        return self.w_desc.issue_global_load_to_vgpr(
            mfma_idx,
            cfg.dot_layout_w,
        )

    @gluon.jit
    def _load_x(self, mfma_idx):
        cfg = self.cfg
        return self.x_desc.issue_local_load(
            mfma_idx,
            self.x_buffer,
            cfg.dot_layout_x,
        )

    @gluon.jit
    def _load_w(self, mfma_idx):
        cfg = self.cfg
        if cfg.W_VIA_VGPR:
            w = self._issue_w_vgpr(mfma_idx)
        else:
            w = self.w_desc.issue_local_load(
                mfma_idx,
                self.w_buffer,
                cfg.dot_layout_w,
                do_permute=cfg.W_TRANSPOSE,
            )
        return w

    @gluon.jit
    def _load_xw(self, mfma_idx):
        w = self._load_w(mfma_idx)
        x = self._load_x(mfma_idx)
        return x, w

    @gluon.jit
    def _load_xw_decode(self, mfma_idx):
        x = self._load_x(mfma_idx)
        w = self._load_w(mfma_idx)
        return x, w

    @gluon.jit
    def _load_x_scales(self, mfma_idx):
        cfg = self.cfg
        x = self.x_desc.issue_local_load(
            mfma_idx,
            self.x_buffer,
            cfg.dot_layout_x,
        )

        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_x = _load_scale_tile_via_gl_load(self.x_scale_desc, mfma_idx)
            else:
                scale_x = gl.full(
                    [cfg.BLOCK_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )
            if cfg.WITH_W_MX_SCALE:
                if cfg.W_SCALE_VIA_LDS:
                    scale_w = self.w_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.w_scale_buffer,
                        cfg.layout_w_scale,
                        cfg.BLOCK_N_PRESHUFFLED,
                        cfg.BLOCK_N,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_w = _load_scale_tile_via_gl_load(self.w_scale_desc, mfma_idx)
            else:
                scale_w = gl.full(
                    [cfg.BLOCK_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_x: gl.constexpr = 0
            scale_w: gl.constexpr = 0

        return x, scale_x, scale_w

    @gluon.jit
    def _load_scales(self, mfma_idx):
        cfg = self.cfg

        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_x = _load_scale_tile_via_gl_load(self.x_scale_desc, mfma_idx)
            else:
                scale_x = gl.full(
                    [cfg.BLOCK_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )

            if cfg.WITH_W_MX_SCALE:
                if cfg.W_SCALE_VIA_LDS:
                    scale_w = self.w_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.w_scale_buffer,
                        cfg.layout_w_scale,
                        cfg.BLOCK_N_PRESHUFFLED,
                        cfg.BLOCK_N,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_w = _load_scale_tile_via_gl_load(self.w_scale_desc, mfma_idx)
            else:
                scale_w = gl.full(
                    [cfg.BLOCK_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_x: gl.constexpr = 0
            scale_w: gl.constexpr = 0

        return scale_x, scale_w

    @gluon.jit
    def issue_local_loads(self, mfma_idx):
        scale_x, scale_w = self._load_scales(mfma_idx)
        x, w = self._load_xw(mfma_idx)

        return x, w, scale_x, scale_w

    @gluon.jit
    def issue_decode_local_loads(self, mfma_idx):
        x, w = self._load_xw_decode(mfma_idx)
        scale_x, scale_w = self._load_scales(mfma_idx)

        return x, w, scale_x, scale_w

    @gluon.jit
    def run(self, loop_k, USE_WARP_PIPELINE: gl.constexpr):
        # A single BLOCK_K tile cannot fill the double-buffered pipelines, so
        # route it to the decode schedule; otherwise pick warp vs local-prefetch.
        cfg = self.cfg
        if cfg.K_ITERS == 1:
            return self.decode_pipeline(loop_k)
        if USE_WARP_PIPELINE:
            return self.warp_pipeline(loop_k)
        return self.pipeline(loop_k)

    @gluon.jit
    def decode_pipeline(self, loop_k):
        cfg = self.cfg
        EVEN_K: gl.constexpr = cfg.EVEN_K
        load_idx = 0
        mfma_idx = 0

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)

        W_PREFETCH: gl.constexpr = cfg.W_VIA_VGPR and cfg.W_PREFETCH

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)

        if W_PREFETCH:
            w_curr = self._issue_w_vgpr(0)

        # EVEN_K: K_iters - (NUM_BUFFERS-1) all-unmasked main iters.
        # !EVEN_K: one less unmasked iter; the last is the masked tail below.
        main_iters = K_iters - (cfg.NUM_BUFFERS - 1 if EVEN_K else cfg.NUM_BUFFERS)
        gl.assume(main_iters >= 0)

        for _ in range(0, main_iters):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=0)
            self.async_wait(cfg.NUM_BUFFERS - 1)

            if W_PREFETCH:
                x, scale_x, scale_w = self._load_x_scales(mfma_idx)
                accumulator = self.mfma(x, scale_x, w_curr, scale_w, accumulator)
                w_curr = self._issue_w_vgpr(mfma_idx + 1)
            else:
                x, w, scale_x, scale_w = self.issue_decode_local_loads(mfma_idx)
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            mfma_idx += 1

        if not EVEN_K:
            # Masked tail iter (one more iter still has W to prefetch).
            load_idx = self.issue_global_loads(load_idx, USE_MASK=1)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            if W_PREFETCH:
                x, scale_x, scale_w = self._load_x_scales(mfma_idx)
                accumulator = self.mfma(x, scale_x, w_curr, scale_w, accumulator)
                w_curr = self._issue_w_vgpr(mfma_idx + 1)
            else:
                x, w, scale_x, scale_w = self.issue_decode_local_loads(mfma_idx)
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            mfma_idx += 1

        # Epilogue: drain remaining in-flight buffers; no new global loads.
        for i in gl.static_range(cfg.NUM_BUFFERS - 1):
            self.async_wait(cfg.NUM_BUFFERS - 2 - i)
            if W_PREFETCH:
                x, scale_x, scale_w = self._load_x_scales(mfma_idx)
                accumulator = self.mfma(x, scale_x, w_curr, scale_w, accumulator)
                if i < cfg.NUM_BUFFERS - 2:
                    w_curr = self._issue_w_vgpr(mfma_idx + 1)
            else:
                x, w, scale_x, scale_w = self.issue_decode_local_loads(mfma_idx)
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)
            mfma_idx += 1

        return accumulator

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        gl.static_assert(
            cfg.NUM_BUFFERS == 2,
            "current local-prefetch pipeline requires exactly two LDS buffers",
        )
        load_idx = 0
        mfma_idx = 0

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)

        # Two-buffer local-prefetch pipeline, unrolled by 2:
        #   async_copy(k + 2) -> freed LDS buffer
        #   mfma(k)
        #   wait for k + 1, leaving k + 2 in flight
        #   local_load(k + 1) -> VGPR
        #
        # The unroll alternates between two explicit operand register sets.
        # This avoids rotating the freshly loaded "next" operands into a
        # single canonical x/w variable at the loop backedge.
        for _ in gl.static_range(cfg.NUM_BUFFERS):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)

        main_iters = K_iters - cfg.NUM_BUFFERS
        gl.assume(main_iters >= 0)

        self.async_wait(cfg.NUM_BUFFERS - 1)
        x0, w0, scale_x0, scale_w0 = self.issue_local_loads(mfma_idx)
        mfma_idx += 1

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2

        for _ in range(0, unroll_pairs):
            # All waves must finish reading the previous contents before any
            # wave overwrites this LDS slot with the future async copy.
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x1, w1, scale_x1, scale_w1 = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            accumulator = self.mfma(x1, scale_x1, w1, scale_w1, accumulator)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x0, w0, scale_x0, scale_w0 = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

        if odd_main:
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
            self.async_wait(cfg.NUM_BUFFERS - 1)
            x1, w1, scale_x1, scale_w1 = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

            # Drain remaining prefetched K tiles; no new global loads.
            accumulator = self.mfma(x1, scale_x1, w1, scale_w1, accumulator)
            self.async_wait(0)
            x0, w0, scale_x0, scale_w0 = self.issue_local_loads(mfma_idx)
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
        else:
            # Drain remaining prefetched K tiles; no new global loads.
            accumulator = self.mfma(x0, scale_x0, w0, scale_w0, accumulator)
            self.async_wait(0)
            x1, w1, scale_x1, scale_w1 = self.issue_local_loads(mfma_idx)
            accumulator = self.mfma(x1, scale_x1, w1, scale_w1, accumulator)

        return accumulator

    @gluon.jit
    def warp_pipeline(self, loop_k):
        cfg = self.cfg
        gl.static_assert(
            cfg.NUM_BUFFERS >= 3,
            "warp_pipeline requires NUM_BUFFERS >= 3",
        )
        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros(
            (cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout
        )
        main_iters = gl.cdiv(loop_k, cfg.BLOCK_K) - (cfg.NUM_BUFFERS - 1)
        gl.assume(main_iters >= 0)

        # Drain oldest prologue batch into LDS; rest remain in flight.
        self.async_wait(cfg.NUM_BUFFERS - 2)

        for _ in range(0, main_iters):
            with gl.amd.warp_pipeline_stage("lds+tdm", priority=1):
                x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
                mfma_idx += 1
                load_idx = self.issue_global_loads(load_idx)

            self.async_wait(cfg.NUM_BUFFERS - 2)

            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        self.async_wait(0)
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
            mfma_idx += 1
            accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        return accumulator


@composition
@gluon.aggregate
class MoESliceMNProgram:
    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer_top: gl.shared_memory_descriptor
    x_buffer_bot: gl.shared_memory_descriptor
    w_buffer_left: gl.shared_memory_descriptor
    w_buffer_right: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc_top: AsyncCopyDescriptor
    x_desc_bot: AsyncCopyDescriptor
    w_desc_left: AsyncCopyDescriptor
    w_desc_right: AsyncCopyDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer_top,
        x_buffer_bot,
        w_buffer_left,
        w_buffer_right,
        x_scale_buffer,
        w_scale_buffer,
        x_desc_top,
        x_desc_bot,
        w_desc_left,
        w_desc_right,
        x_scale_desc,
        w_scale_desc,
    ):
        self.cfg = cfg
        self.x_buffer_top = x_buffer_top
        self.x_buffer_bot = x_buffer_bot
        self.w_buffer_left = w_buffer_left
        self.w_buffer_right = w_buffer_right
        self.x_scale_buffer = x_scale_buffer if cfg.X_SCALE_VIA_LDS else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.W_SCALE_VIA_LDS else gl.constexpr(0)
        self.x_desc_top = x_desc_top
        self.x_desc_bot = x_desc_bot
        self.w_desc_left = w_desc_left
        self.w_desc_right = w_desc_right
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(
        cfg: MoEConfig,
        x_desc_top,
        x_desc_bot,
        w_desc_left,
        w_desc_right,
        x_scale_desc,
        w_scale_desc,
    ):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer_top = gl.allocate_shared_memory(
            x_desc_top.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M // 2, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x_half_m,
        )
        x_buffer_bot = gl.allocate_shared_memory(
            x_desc_bot.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M // 2, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x_half_m,
        )
        w_buffer_left = gl.allocate_shared_memory(
            w_desc_left.dtype,
            shape=(
                [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                if cfg.W_TRANSPOSE
                else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
            ),
            layout=cfg.shared_layout_w_half_n,
        )
        w_buffer_right = gl.allocate_shared_memory(
            w_desc_right.dtype,
            shape=(
                [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                if cfg.W_TRANSPOSE
                else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
            ),
            layout=cfg.shared_layout_w_half_n,
        )

        if cfg.X_SCALE_VIA_LDS:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_M_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.W_SCALE_VIA_LDS:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceMNProgram(
            cfg,
            x_buffer_top,
            x_buffer_bot,
            w_buffer_left,
            w_buffer_right,
            x_scale_buffer,
            w_scale_buffer,
            x_desc_top,
            x_desc_bot,
            w_desc_left,
            w_desc_right,
            x_scale_desc,
            w_scale_desc,
        )

    @gluon.jit
    def issue_local_load_x_sub(self, mfma_idx, subtile_idx_m: gl.constexpr):
        cfg = self.cfg
        SUBTILE_M: gl.constexpr = cfg.BLOCK_M // 2
        subtile_start_m: gl.constexpr = subtile_idx_m * SUBTILE_M
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if subtile_idx_m == 0:
            slot = self.x_buffer_top.index(mfma_idx % cfg.NUM_BUFFERS)
        else:
            slot = self.x_buffer_bot.index(mfma_idx % cfg.NUM_BUFFERS)
        x = gl.amd.cdna4.async_copy.load_shared_relaxed(slot, cfg.dot_layout_x)

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle_sub(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                        SUBTILE_M,
                        subtile_start_m,
                    )
                else:
                    scale_x = _load_scale_subtile_via_gl_load(
                        self.x_scale_desc,
                        mfma_idx,
                        subtile_start_m,
                        SUBTILE_M,
                    )
            else:
                scale_x = gl.full(
                    [SUBTILE_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )
        else:
            scale_x: gl.constexpr = 0

        return x, scale_x

    @gluon.jit
    def issue_local_load_w_sub(self, mfma_idx, subtile_idx_n: gl.constexpr):
        cfg = self.cfg
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2
        subtile_start_n: gl.constexpr = subtile_idx_n * SUBTILE_N
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if subtile_idx_n == 0:
            slot = self.w_buffer_left.index(mfma_idx % cfg.NUM_BUFFERS)
        else:
            slot = self.w_buffer_right.index(mfma_idx % cfg.NUM_BUFFERS)
        if cfg.W_TRANSPOSE:
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(
                slot.permute([1, 0]),
                cfg.dot_layout_w,
            )
        else:
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(slot, cfg.dot_layout_w)

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_W_MX_SCALE:
                if cfg.W_SCALE_VIA_LDS:
                    scale_w = self.w_scale_desc.issue_local_load_unswizzle_sub(
                        mfma_idx,
                        self.w_scale_buffer,
                        cfg.layout_w_scale,
                        cfg.BLOCK_N_PRESHUFFLED,
                        cfg.BLOCK_N,
                        BLOCK_K_SCALE,
                        SUBTILE_N,
                        subtile_start_n,
                    )
                else:
                    scale_w = _load_scale_subtile_via_gl_load(
                        self.w_scale_desc,
                        mfma_idx,
                        subtile_start_n,
                        SUBTILE_N,
                    )
            else:
                scale_w = gl.full(
                    [SUBTILE_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_w: gl.constexpr = 0

        return w, scale_w

    @gluon.jit
    def issue_w_left(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        self.w_desc_left.issue_async_load(
            load_idx, self.w_buffer_left, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx

    @gluon.jit
    def issue_x_top(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        self.x_desc_top.issue_async_load(
            load_idx, self.x_buffer_top, pred, USE_MASK=USE_MASK, COMMIT=0
        )
        scale_via_lds: gl.constexpr = cfg.X_SCALE_VIA_LDS or cfg.W_SCALE_VIA_LDS
        if scale_via_lds:
            if cfg.X_SCALE_VIA_LDS:
                self.x_scale_desc.issue_async_load(
                    load_idx,
                    self.x_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
            if cfg.W_SCALE_VIA_LDS:
                self.w_scale_desc.issue_async_load(
                    load_idx,
                    self.w_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
        gl.amd.cdna4.async_copy.commit_group()
        return load_idx

    @gluon.jit
    def issue_x_bot(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        self.x_desc_bot.issue_async_load(
            load_idx, self.x_buffer_bot, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx

    @gluon.jit
    def issue_w_right(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        self.w_desc_right.issue_async_load(
            load_idx, self.w_buffer_right, pred, USE_MASK=USE_MASK, COMMIT=1
        )
        return load_idx + 1

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        load_idx = self.issue_w_left(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_x_top(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_x_bot(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_w_right(load_idx, pred, USE_MASK=USE_MASK)
        return load_idx

    @gluon.jit
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt * 4)

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        gl.static_assert(
            (cfg.NUM_SUBTILES[0] == 2)
            and (cfg.NUM_SUBTILES[1] == 2)
            and (cfg.NUM_SUBTILES[2] == 1),
            "MoESliceMNProgram requires NUM_SUBTILES=(2,2,1)",
        )
        gl.static_assert(NB >= 2, "MoESliceMNProgram requires NUM_BUFFERS >= 2")

        SUBTILE_M: gl.constexpr = cfg.BLOCK_M // 2
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2

        load_idx = 0
        mfma_idx = 0

        # Prologue: NB iters in flight (region 2/3 of iter 0 ds_read
        # iter 1 W_left / X_top, so NB not NB-1). Use the descriptor's
        # automatic tail-K mask for uneven K shapes; TP8 GPT-OSS GEMM2 has
        # K=384 with BLOCK_K=256, so the second preload is a K tail.
        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)

        c_tl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_bl = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_tr = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c_br = gl.zeros((SUBTILE_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        # K-tail mask absorbed via USE_MASK=-1 in-loop (no dedicated peel).
        main_iters = K_iters - NB
        gl.assume(main_iters >= 2)

        # Drain iter 0's first 2 commits so the first MFMA has data.
        gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
        x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)
        w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

        # USE_MASK=-1 + in-loop mask drops the dedicated K-tail peel.
        # Region order ``mfma -> issue -> wait -> ds_read`` lets the
        # vmem coalesce start in parallel with the wait's s_barrier
        # (raising the wait target by 1 to compensate).
        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2
        for _ in range(0, unroll_pairs):
            # iter k: 4 regions (consume buffer (m % NB), refill same).
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            load_idx = self.issue_w_left(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            # issue_x_top also refills the scale LDS slot. Read the
            # current right-W scale before that slot is reused.
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            load_idx = self.issue_w_right(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

            # iter k+1: same 4 regions, ping-ponged buffer slot.
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            load_idx = self.issue_w_left(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            load_idx = self.issue_w_right(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

        # Odd peel; same USE_MASK=-1 handles the K-tail iter.
        if odd_main:
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            load_idx = self.issue_w_left(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)
            load_idx = self.issue_x_top(load_idx, USE_MASK=-1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            load_idx = self.issue_x_bot(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 3)
            w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            load_idx = self.issue_w_right(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(4 * NB - 2)
            x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

        # Drain epilogue: NB iters of MFMA, no further async issues.
        # Mirrors v8's "iterMax-2 / iterMax-1" tail with the trailing
        # ds_reads guarded by ``i < NB - 1`` (the last-iter MFMAs use
        # the final x_top / w_left already in regs).
        gl.amd.cdna4.async_copy.wait_group(0)
        for i in gl.static_range(NB):
            c_tl = self.mfma(x_top, sx_top, w_left, sw_left, c_tl)
            x_bot, sx_bot = self.issue_local_load_x_sub(mfma_idx, 1)

            c_bl = self.mfma(x_bot, sx_bot, w_left, sw_left, c_bl)
            w_right, sw_right = self.issue_local_load_w_sub(mfma_idx, 1)

            c_tr = self.mfma(x_top, sx_top, w_right, sw_right, c_tr)
            mfma_idx += 1
            if i < NB - 1:
                w_left, sw_left = self.issue_local_load_w_sub(mfma_idx, 0)

            c_br = self.mfma(x_bot, sx_bot, w_right, sw_right, c_br)
            if i < NB - 1:
                x_top, sx_top = self.issue_local_load_x_sub(mfma_idx, 0)

        # Stitch the 4 quadrants and re-anchor to cfg.acc_layout.
        acc_top = gl.join(c_tl, c_tr).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
        acc_bot = gl.join(c_bl, c_br).permute(0, 2, 1).reshape((SUBTILE_M, cfg.BLOCK_N))
        accumulator = (
            gl.join(acc_top, acc_bot)
            .permute(2, 0, 1)
            .reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        )
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout)

        return accumulator


@composition
@gluon.aggregate
class MoESliceNProgram:
    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer_top: gl.shared_memory_descriptor | gl.constexpr
    w_buffer_bot: gl.shared_memory_descriptor | gl.constexpr
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    x_desc: AsyncCopyDescriptor
    w_desc_top: AsyncCopyDescriptor | WVgprDescriptor | WPreshuffledLdsDescriptor
    w_desc_bot: AsyncCopyDescriptor | WVgprDescriptor | WPreshuffledLdsDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr
    bottom_valid: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg: MoEConfig,
        x_buffer,
        w_buffer_top,
        w_buffer_bot,
        x_scale_buffer,
        w_scale_buffer,
        x_desc,
        w_desc_top,
        w_desc_bot,
        x_scale_desc,
        w_scale_desc,
        bottom_valid,
    ):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer_top = w_buffer_top if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.w_buffer_bot = w_buffer_bot if not cfg.W_VIA_VGPR else gl.constexpr(0)
        self.x_scale_buffer = x_scale_buffer if cfg.X_SCALE_VIA_LDS else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.W_SCALE_VIA_LDS else gl.constexpr(0)
        self.x_desc = x_desc
        self.w_desc_top = w_desc_top
        self.w_desc_bot = w_desc_bot
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)
        self.bottom_valid = bottom_valid
        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(
        cfg: MoEConfig,
        x_desc,
        w_desc_top,
        w_desc_bot,
        x_scale_desc,
        w_scale_desc,
        bottom_valid,
    ):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(
            x_desc.dtype,
            shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
            layout=cfg.shared_layout_x,
        )
        if cfg.W_VIA_VGPR:
            w_buffer_top = gl.constexpr(0)
            w_buffer_bot = gl.constexpr(0)
        elif cfg.W_PRESHUFFLED:
            w_buffer_top = gl.allocate_shared_memory(
                w_desc_top.dtype,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N // 2 // 16,
                    BLOCK_K_PACKED_W * 16,
                ],
                layout=cfg.shared_layout_w_half_n,
            )
            w_buffer_bot = gl.allocate_shared_memory(
                w_desc_bot.dtype,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N // 2 // 16,
                    BLOCK_K_PACKED_W * 16,
                ],
                layout=cfg.shared_layout_w_half_n,
            )
        else:
            w_buffer_top = gl.allocate_shared_memory(
                w_desc_top.dtype,
                shape=(
                    [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                    if cfg.W_TRANSPOSE
                    else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
                ),
                layout=cfg.shared_layout_w_half_n,
            )
            w_buffer_bot = gl.allocate_shared_memory(
                w_desc_bot.dtype,
                shape=(
                    [NUM_BUFFERS, cfg.BLOCK_N // 2, BLOCK_K_PACKED_W]
                    if cfg.W_TRANSPOSE
                    else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N // 2]
                ),
                layout=cfg.shared_layout_w_half_n,
            )

        if cfg.X_SCALE_VIA_LDS:
            x_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_M_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_x_scale,
            )
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.W_SCALE_VIA_LDS:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8,
                shape=[
                    NUM_BUFFERS,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_K_SCALE_PRESHUFFLED,
                ],
                layout=cfg.shared_layout_w_scale,
            )
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoESliceNProgram(
            cfg,
            x_buffer,
            w_buffer_top,
            w_buffer_bot,
            x_scale_buffer,
            w_scale_buffer,
            x_desc,
            w_desc_top,
            w_desc_bot,
            x_scale_desc,
            w_scale_desc,
            bottom_valid,
        )

    @gluon.jit
    def issue_local_load_x(self, mfma_idx):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
        if not cfg.W_VIA_VGPR and (
            cfg.BLOCK_M == 32 or cfg.BLOCK_M == 64 or cfg.BLOCK_M == 128
        ):
            x = self.x_desc.issue_local_load_m_swizzle(
                mfma_idx,
                self.x_buffer,
                cfg.dot_layout_x,
                cfg.BLOCK_M,
            )
        else:
            x = self.x_desc.issue_local_load(
                mfma_idx,
                self.x_buffer,
                cfg.dot_layout_x,
            )

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_X_MX_SCALE:
                if cfg.X_SCALE_VIA_LDS:
                    scale_x = self.x_scale_desc.issue_local_load_unswizzle(
                        mfma_idx,
                        self.x_scale_buffer,
                        cfg.layout_x_scale,
                        cfg.BLOCK_M_PRESHUFFLED,
                        cfg.BLOCK_M,
                        BLOCK_K_SCALE,
                    )
                else:
                    scale_x = _load_scale_tile_via_gl_load(self.x_scale_desc, mfma_idx)
            else:
                # fp8 X path: identity scale (e8m0=127 == 2^0).
                scale_x = gl.full(
                    [cfg.BLOCK_M, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_x_scale,
                )
        else:
            scale_x: gl.constexpr = 0

        return x, scale_x

    @gluon.jit
    def issue_global_load_top(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        self.x_desc.issue_async_load(
            load_idx, self.x_buffer, pred, USE_MASK=USE_MASK, COMMIT=0
        )
        scale_via_lds: gl.constexpr = cfg.X_SCALE_VIA_LDS or cfg.W_SCALE_VIA_LDS
        if scale_via_lds:
            if cfg.X_SCALE_VIA_LDS:
                self.x_scale_desc.issue_async_load(
                    load_idx,
                    self.x_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
            if cfg.W_SCALE_VIA_LDS:
                self.w_scale_desc.issue_async_load(
                    load_idx,
                    self.w_scale_buffer,
                    pred,
                    USE_MASK=USE_MASK,
                    COMMIT=0,
                )
        if not cfg.W_VIA_VGPR:
            self.w_desc_top.issue_async_load(
                load_idx, self.w_buffer_top, pred, USE_MASK=USE_MASK, COMMIT=0
            )
        gl.amd.cdna4.async_copy.commit_group()
        return load_idx

    @gluon.jit
    def issue_global_load_bot(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        cfg = self.cfg
        if cfg.W_VIA_VGPR:
            gl.amd.cdna4.async_copy.commit_group()
        else:
            self.w_desc_bot.issue_async_load(
                load_idx, self.w_buffer_bot, pred, USE_MASK=USE_MASK, COMMIT=1
            )
        return load_idx + 1

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1, USE_MASK: gl.constexpr = -1):
        load_idx = self.issue_global_load_top(load_idx, pred, USE_MASK=USE_MASK)
        load_idx = self.issue_global_load_bot(load_idx, pred, USE_MASK=USE_MASK)
        return load_idx

    @gluon.jit
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt * 2)

    @gluon.jit
    def issue_local_load_w_sub(self, mfma_idx, subtile_idx_n: gl.constexpr):
        cfg = self.cfg
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // cfg.NUM_SUBTILES[1]
        subtile_start_n: gl.constexpr = subtile_idx_n * SUBTILE_N
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        if cfg.W_VIA_VGPR:
            if subtile_idx_n == 0:
                w = self.w_desc_top.issue_global_load_to_vgpr(
                    mfma_idx, cfg.dot_layout_w
                )
            else:
                w = self.w_desc_bot.issue_global_load_to_vgpr(
                    mfma_idx, cfg.dot_layout_w
                )
        elif cfg.W_PRESHUFFLED:
            if subtile_idx_n == 0:
                w = self.w_desc_top.issue_local_load(
                    mfma_idx,
                    self.w_buffer_top,
                    cfg.dot_layout_w,
                )
            else:
                w = self.w_desc_bot.issue_local_load(
                    mfma_idx,
                    self.w_buffer_bot,
                    cfg.dot_layout_w,
                )
        else:
            if subtile_idx_n == 0:
                slot = self.w_buffer_top.index(mfma_idx % cfg.NUM_BUFFERS)
            else:
                slot = self.w_buffer_bot.index(mfma_idx % cfg.NUM_BUFFERS)

            if cfg.W_TRANSPOSE:
                w = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    slot.permute([1, 0]),
                    cfg.dot_layout_w,
                )
            else:
                w = gl.amd.cdna4.async_copy.load_shared_relaxed(slot, cfg.dot_layout_w)

        if cfg.USE_MFMA_SCALED:
            if cfg.WITH_W_MX_SCALE:
                scale_w = self.w_scale_desc.issue_local_load_unswizzle_sub(
                    mfma_idx,
                    self.w_scale_buffer,
                    cfg.layout_w_scale,
                    cfg.BLOCK_N_PRESHUFFLED,
                    cfg.BLOCK_N,
                    BLOCK_K_SCALE,
                    SUBTILE_N,
                    subtile_start_n,
                )
            else:
                scale_w = gl.full(
                    [SUBTILE_N, BLOCK_K_SCALE],
                    127,
                    gl.uint8,
                    layout=cfg.layout_w_scale,
                )
        else:
            scale_w: gl.constexpr = 0

        return w, scale_w

    @gluon.jit
    def _finish_accumulator(self, c0, c1):
        cfg = self.cfg
        accumulator = (
            gl.join(c0, c1).permute(0, 2, 1).reshape((cfg.BLOCK_M, cfg.BLOCK_N))
        )
        accumulator = gl.convert_layout(accumulator, cfg.acc_layout)
        return accumulator

    @gluon.jit
    def _pipeline_top_only(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2

        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)

        c0 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        main_iters = K_iters - NB
        gl.assume(main_iters >= 0)

        gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
        w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
        x0, sx0 = self.issue_local_load_x(mfma_idx)
        mfma_idx += 1

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2

        for _ in range(0, unroll_pairs):
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            mfma_idx += 1

            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            mfma_idx += 1

        if odd_main:
            gl.barrier()
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            mfma_idx += 1

            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(0)
        else:
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(0)

        return self._finish_accumulator(c0, c1)

    @gluon.jit
    def _pipeline_full(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        SUBTILE_N: gl.constexpr = cfg.BLOCK_N // 2

        load_idx = 0
        mfma_idx = 0

        for _ in gl.static_range(NB):
            load_idx = self.issue_global_loads(load_idx, USE_MASK=-1)

        c0 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)
        c1 = gl.zeros((cfg.BLOCK_M, SUBTILE_N), dtype=gl.float32, layout=cfg.acc_layout)

        if cfg.K_ITERS:
            K_iters: gl.constexpr = cfg.K_ITERS
        else:
            K_iters = gl.cdiv(loop_k, cfg.BLOCK_K)
        main_iters = K_iters - NB
        gl.assume(main_iters >= 0)

        # Drain iter 0's top async-copy group first. For SliceN, the top group
        # contains X, W-top, and scales; the bottom group contains W-bottom.
        # Loading X before the bottom MFMA gives the next X tile latency slack.
        # The hot loop delays W-top until after the current bottom MFMA so the
        # pre-bottom wait does not also cover an unused W-top LDS read.
        gl.amd.cdna4.async_copy.wait_group(2 * NB - 1)
        w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
        x0, sx0 = self.issue_local_load_x(mfma_idx)
        gl.amd.cdna4.async_copy.wait_group(2 * NB - 2)
        w01, sw01 = self.issue_local_load_w_sub(mfma_idx, 1)
        mfma_idx += 1

        unroll_pairs = main_iters // 2
        odd_main = main_iters - unroll_pairs * 2

        for _ in range(0, unroll_pairs):
            # The future copy reuses the slot just local-loaded into VGPRs.
            # Synchronize the CTA before any producer wave overwrites it.
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.barrier()
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            w11, sw11 = self.issue_local_load_w_sub(mfma_idx, 1)
            mfma_idx += 1

            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.barrier()
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x1, sx1, w11, sw11, c1)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            w01, sw01 = self.issue_local_load_w_sub(mfma_idx, 1)
            mfma_idx += 1

        if odd_main:
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.barrier()
            load_idx = self.issue_global_load_top(load_idx, USE_MASK=-1)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
            load_idx = self.issue_global_load_bot(load_idx, USE_MASK=-1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            gl.amd.cdna4.async_copy.wait_group(2 * (NB - 1))
            w11, sw11 = self.issue_local_load_w_sub(mfma_idx, 1)
            mfma_idx += 1

            # Drain + final NB iters of MFMAs (no more async_copy).
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w00, sw00 = self.issue_local_load_w_sub(mfma_idx, 0)
            x0, sx0 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x1, sx1, w11, sw11, c1)
            gl.amd.cdna4.async_copy.wait_group(0)
            w01, sw01 = self.issue_local_load_w_sub(mfma_idx, 1)
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
        else:
            # Drain + final NB iters of MFMAs (no more async_copy).
            c0 = self.mfma(x0, sx0, w00, sw00, c0)
            gl.amd.cdna4.async_copy.wait_group(1)
            w10, sw10 = self.issue_local_load_w_sub(mfma_idx, 0)
            x1, sx1 = self.issue_local_load_x(mfma_idx)
            c1 = self.mfma(x0, sx0, w01, sw01, c1)
            gl.amd.cdna4.async_copy.wait_group(0)
            w11, sw11 = self.issue_local_load_w_sub(mfma_idx, 1)
            c0 = self.mfma(x1, sx1, w10, sw10, c0)
            c1 = self.mfma(x1, sx1, w11, sw11, c1)

        return self._finish_accumulator(c0, c1)

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        NB: gl.constexpr = cfg.NUM_BUFFERS
        gl.static_assert(
            (cfg.NUM_SUBTILES[0] == 1)
            and (cfg.NUM_SUBTILES[1] == 2)
            and (cfg.NUM_SUBTILES[2] == 1),
            "MoESliceNProgram requires NUM_SUBTILES=(1,2,1)",
        )
        gl.static_assert(
            NB == 2,
            "current SliceN local-prefetch pipeline requires exactly two LDS buffers",
        )
        gl.static_assert(
            cfg.K_ITERS != 1,
            "SliceN requires K_ITERS >= 2; single BLOCK_K tile shapes must route "
            "to the full-N decode schedule (see _is_single_k_tile host gate)",
        )

        if self.bottom_valid:
            return self._pipeline_full(loop_k)
        return self._pipeline_top_only(loop_k)


@gluon.jit
def _make_moe_x_desc(
    cfg,
    x_ptr,
    rows_m_x,
    offs_xk,
    stride_xm,
    stride_xk,
    x_mask_nonk,
    k_limit_x,
    BLOCK_K_X: gl.constexpr,
):
    return AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        x_mask_nonk,
        k_limit_x,
    )


@gluon.jit
def _make_swizzled_scale_direct_desc(
    cfg,
    scale_ptr,
    rows_m_scale,
    offs_ks,
    stride_mblock,
    stride_kswizzled,
    mask_m_scale,
    k_limit,
    BLOCK_K_SCALE: gl.constexpr,
):
    m_block = rows_m_scale // cfg.PRESHUFFLE_FACTOR
    m_in_block = rows_m_scale % cfg.PRESHUFFLE_FACTOR
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = offs_ks // 8
    k_in_block = offs_ks % 8
    k_hi = k_in_block // 4
    k_lo = k_in_block % 4
    stride_k_t = gl.to_tensor(stride_kswizzled * cfg.PRESHUFFLE_FACTOR)
    stride_mblock_t = gl.to_tensor(stride_mblock)
    swizzled_k = (
        (((k_block[None, :] * 4 + k_lo[None, :]) * 16 + m_lo[:, None]) * 2)
        + k_hi[None, :]
    ) * 2 + m_hi[:, None]
    offsets = (
        swizzled_k * stride_kswizzled + m_block[:, None].to(gl.int64) * stride_mblock_t
    )
    return AsyncCopyDescriptor(
        cfg,
        0,
        BLOCK_K_SCALE,
        scale_ptr,
        scale_ptr.dtype.element_ty,
        stride_k_t,
        stride_mblock_t,
        offsets,
        offs_ks,
        rows_m_scale,
        mask_m_scale[:, None],
        k_limit,
        gl.to_tensor(0),
    )


@gluon.jit
def _make_slice_mn_x_descs(
    cfg,
    x_ptr,
    gather_idx_ptr,
    stride_xm,
    stride_xk,
    M_X,
    off_m,
    m_limit,
    k_limit_x,
    BLOCK_M: gl.constexpr,
    BLOCK_K_X: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    X_ELEM_BITS: gl.constexpr,
):
    SUB_BM_MN: gl.constexpr = BLOCK_M // 2

    LOAD_X_SUB_LAYOUT_MN: gl.constexpr = _load_layout(
        BLOCK_K_X, SUB_BM_MN, NUM_WARPS, [1, 0], X_ELEM_BITS
    )
    offs_xm_sub_mn = gl.arange(
        0, SUB_BM_MN, layout=gl.SliceLayout(1, LOAD_X_SUB_LAYOUT_MN)
    )
    offs_xk_sub_mn = gl.arange(
        0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_SUB_LAYOUT_MN)
    )
    rows_m_top = off_m + offs_xm_sub_mn
    rows_m_bot = off_m + SUB_BM_MN + offs_xm_sub_mn
    pre_gather_mask_top = rows_m_top < m_limit
    pre_gather_mask_bot = rows_m_bot < m_limit
    if HAS_GATHER:
        rows_m_top_safe = gl.where(
            pre_gather_mask_top, rows_m_top, gl.zeros_like(rows_m_top)
        )
        rows_m_bot_safe = gl.where(
            pre_gather_mask_bot, rows_m_bot, gl.zeros_like(rows_m_bot)
        )
        rows_m_top = gl.load(
            gather_idx_ptr + rows_m_top_safe,
            mask=pre_gather_mask_top,
            other=0,
        ).to(gl.int32)
        rows_m_bot = gl.load(
            gather_idx_ptr + rows_m_bot_safe,
            mask=pre_gather_mask_bot,
            other=0,
        ).to(gl.int32)
        mask_m_top = pre_gather_mask_top & (rows_m_top < M_X)
        mask_m_bot = pre_gather_mask_bot & (rows_m_bot < M_X)
    else:
        rows_m_top = gl.where(
            pre_gather_mask_top, rows_m_top, gl.zeros_like(rows_m_top)
        )
        rows_m_bot = gl.where(
            pre_gather_mask_bot, rows_m_bot, gl.zeros_like(rows_m_bot)
        )
        mask_m_top = pre_gather_mask_top
        mask_m_bot = pre_gather_mask_bot
    x_desc_top_mn = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        x_ptr,
        rows_m_top,
        offs_xk_sub_mn,
        stride_xm,
        stride_xk,
        mask_m_top[:, None],
        k_limit_x,
    )
    x_desc_bot_mn = AsyncCopyDescriptor.initialize(
        cfg,
        0,
        BLOCK_K_X,
        x_ptr,
        rows_m_bot,
        offs_xk_sub_mn,
        stride_xm,
        stride_xk,
        mask_m_bot[:, None],
        k_limit_x,
    )
    return x_desc_top_mn, x_desc_bot_mn


@gluon.jit
def _make_nonpreshuffled_w_desc(
    cfg,
    w_ptr,
    rows_n,
    offs_wk,
    stride_wn,
    stride_wk,
    mask_n,
    k_limit_w,
    w_base_offset,
    OP_IDX: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    return AsyncCopyDescriptor.initialize(
        cfg,
        OP_IDX,
        BLOCK_K_W,
        w_ptr,
        rows_n,
        offs_wk,
        stride_wn,
        stride_wk,
        mask_n,
        k_limit_w,
        base_offset=w_base_offset,
        cache_modifier=W_CACHE_MODIFIER,
    )


@gluon.jit
def _make_nonpreshuffled_w_half_descs(
    cfg,
    w_ptr,
    stride_wn,
    stride_wk,
    N,
    off_n,
    k_limit_w,
    w_base_offset,
    SUB_BN: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_TRANSPOSE: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    if W_TRANSPOSE:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, SUB_BN, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, SUB_BN, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        mask_n_first = (off_n + offs_wn) < N
        mask_n_second = (off_n + SUB_BN + offs_wn) < N
        w_desc_first = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_first[:, None],
            k_limit_w,
            w_base_offset,
            0,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
        w_desc_second = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + SUB_BN + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_second[:, None],
            k_limit_w,
            w_base_offset,
            0,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    else:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            SUB_BN, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, SUB_BN, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        mask_n_first = (off_n + offs_wn) < N
        mask_n_second = (off_n + SUB_BN + offs_wn) < N
        w_desc_first = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_first[None, :],
            k_limit_w,
            w_base_offset,
            1,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
        w_desc_second = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + SUB_BN + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n_second[None, :],
            k_limit_w,
            w_base_offset,
            1,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    return w_desc_first, w_desc_second


@gluon.jit
def _make_nonpreshuffled_w_full_desc(
    cfg,
    w_ptr,
    stride_wn,
    stride_wk,
    N,
    off_n,
    k_limit_w,
    w_base_offset,
    BLOCK_N: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    W_TRANSPOSE: gl.constexpr,
    W_ELEM_BITS: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
):
    if W_TRANSPOSE:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_K_W, BLOCK_N, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        mask_n = (off_n + offs_wn) < N
        w_desc = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[:, None],
            k_limit_w,
            w_base_offset,
            0,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    else:
        LOAD_W_LAYOUT: gl.constexpr = _load_layout(
            BLOCK_N, BLOCK_K_W, NUM_WARPS, [1, 0], W_ELEM_BITS
        )
        offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
        offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))
        mask_n = (off_n + offs_wn) < N
        w_desc = _make_nonpreshuffled_w_desc(
            cfg,
            w_ptr,
            off_n + offs_wn,
            offs_wk,
            stride_wn,
            stride_wk,
            mask_n[None, :],
            k_limit_w,
            w_base_offset,
            1,
            BLOCK_K_W,
            W_CACHE_MODIFIER,
        )
    return w_desc


@gluon.constexpr_function
def _preshuffled_w_read_layout(
    block_n_units: int,
    block_k_w: int,
    scale_via_lds: bool,
):
    if scale_via_lds:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 1024],
                [1, 0],
                [4, 0],
            ],
            lane_bases=[
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [0, 512],
            ],
            warp_bases=[[2, 0], [0, 0]],
            block_bases=[],
            shape=[block_n_units, block_k_w * 16],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 1024],
            [2, 0],
            [4, 0],
        ],
        lane_bases=[
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [0, 256],
            [0, 512],
        ],
        warp_bases=[[1, 0], [0, 0]],
        block_bases=[],
        shape=[block_n_units, block_k_w * 16],
    )


@gluon.constexpr_function
def _preshuffled_w_copy_layout(
    block_n_units: int,
    block_k_w: int,
    scale_via_lds: bool,
    use_all_waves_for_copy: bool,
):
    if not use_all_waves_for_copy:
        return _preshuffled_w_read_layout(block_n_units, block_k_w, scale_via_lds)
    if scale_via_lds:
        return gl.DistributedLinearLayout(
            reg_bases=[
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 1024],
                [1, 0],
            ],
            lane_bases=[
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [0, 512],
            ],
            warp_bases=[[2, 0], [4, 0]],
            block_bases=[],
            shape=[block_n_units, block_k_w * 16],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 1024],
            [2, 0],
        ],
        lane_bases=[
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [0, 256],
            [0, 512],
        ],
        warp_bases=[[1, 0], [4, 0]],
        block_bases=[],
        shape=[block_n_units, block_k_w * 16],
    )


@gluon.jit
def _make_preshuffled_w_x_desc(
    cfg,
    x_ptr,
    rows_m_x,
    offs_xk,
    stride_xm,
    stride_xk,
    M_X,
    k_limit_x,
    BLOCK_K_X: gl.constexpr,
    HAS_GATHER: gl.constexpr,
):
    if HAS_GATHER:
        # Invalid expert-tail rows were already clamped to gather row 0.
        # Keep the global token bound check, but avoid carrying the
        # per-expert tail predicate through every X async-copy element.
        x_mask_nonk = (rows_m_x < M_X)[:, None]
    else:
        x_mask_nonk = gl.to_tensor(True)
    return _make_moe_x_desc(
        cfg,
        x_ptr,
        rows_m_x,
        offs_xk,
        stride_xm,
        stride_xk,
        x_mask_nonk,
        k_limit_x,
        BLOCK_K_X,
    )


@gluon.jit
def _make_preshuffled_w_slice_offsets(
    w_base_offset,
    pid_n,
    N,
    LOAD_W_COPY_LAYOUT: gl.constexpr,
    N_LIMIT: gl.constexpr,
    SUB_BN: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
):
    offs_wn_h = gl.arange(0, SUB_BN // 16, layout=gl.SliceLayout(1, LOAD_W_COPY_LAYOUT))
    offs_wk_h = gl.arange(
        0, BLOCK_K_W * 16, layout=gl.SliceLayout(0, LOAD_W_COPY_LAYOUT)
    )
    offsets_h = gl.expand_dims(offs_wk_h, 0) + gl.expand_dims(offs_wn_h, 1) * (
        BLOCK_K_W * 16
    )
    TILE_BYTES_HALF: gl.constexpr = 128 * 128
    if N_LIMIT:
        n_block_count: gl.constexpr = (N_LIMIT + 127) // 128
        w_k_stride = gl.to_tensor(N_LIMIT)
    else:
        n_block_count = (N + 127) // 128
        w_k_stride = gl.to_tensor(N)
    bottom_valid = (2 * pid_n + 1) < n_block_count
    base_off_top = w_base_offset + 2 * pid_n * TILE_BYTES_HALF
    base_off_bot = base_off_top + TILE_BYTES_HALF
    return offsets_h, base_off_top, base_off_bot, w_k_stride, bottom_valid


@gluon.jit
def _make_preshuffled_w_full_offsets(
    w_base_offset,
    pid_n,
    LOAD_W_COPY_LAYOUT: gl.constexpr,
    BLOCK_N_LAYOUT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K_W: gl.constexpr,
):
    offs_wn = gl.arange(
        0, BLOCK_N_LAYOUT // 16, layout=gl.SliceLayout(1, LOAD_W_COPY_LAYOUT)
    )
    offs_wk = gl.arange(0, BLOCK_K_W * 16, layout=gl.SliceLayout(0, LOAD_W_COPY_LAYOUT))
    offsets_b = gl.expand_dims(offs_wk, 0) + gl.expand_dims(offs_wn, 1) * (
        BLOCK_K_W * 16
    )
    TILE_BYTES: gl.constexpr = BLOCK_K_W * BLOCK_N
    base_off_b = w_base_offset + pid_n * TILE_BYTES
    return offsets_b, base_off_b
