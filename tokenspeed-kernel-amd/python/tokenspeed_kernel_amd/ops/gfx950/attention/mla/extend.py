# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Absorbed MLA extend (prefix-cache / chunked-prefill) Gluon kernel for GFX950.

Queries are already projected into the latent space: q is ``[total_q, H,
KV_LORA_RANK + ROPE_DIM]`` and attends a paged latent cache ``[num_pages,
PAGE_SIZE, 1, KV_LORA_RANK + ROPE_DIM]`` that already holds the new tokens.
Values are the first ``KV_LORA_RANK`` dims of the same cache rows.

Each program packs ``BLOCK_Q`` query positions times ``HEAD_GROUP`` heads into
the 128-row MFMA M dimension, so one KV page is read from LDS once for all of
them. Every wave owns 32 rows: the full 32x512 fp32 accumulator (256 VGPRs)
keeps each LDS operand read feeding twice as many MFMAs as a 16-row wave
would, which is what keeps the MFMA pipe, not LDS, the limiter. The 512-wide
latent dim is processed in four 128-wide chunks so only one K or V chunk
operand is live at a time.

KV tiles stream into a four-stage LDS ring by direct-to-LDS buffer loads. For
FP8 the loads swizzle their source columns so the linear LDS writes land in a
bank-conflict-free layout for both the K and the transposed V reads. LDS
operand reads skip the per-read wait (the stage wait covers them), and group
barriers interleave the softmax exp2s into the Q @ K^T MFMAs and the next
tile's copy into the P @ V MFMAs. See ``ops/README.md`` for the full contract.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._scheduling import (
    sched_barrier,
    sched_barrier_compile_options,
    sched_group_barrier,
    wave_uniform_i32,
)
from tokenspeed_kernel_amd._triton import gl, gluon, gluon_builtin
from tokenspeed_kernel_amd.ops.gfx950.attention._common import (
    _INV_LN2_VALUE,
    _LN2,
    max,
    maximum,
    padded_shared_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import _find_seq_idx

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy

_BLOCK_M = 128
_PAGE_SIZE = 64
_NUM_WARPS = 4
_CHUNK = 128
# QK contraction step over the latent dims: one K operand read feeds one MFMA
# per 32 columns for FP8 (32x32x64). BF16 reads whole chunks: slicing a padded
# BF16 chunk in half misaddresses its K reads.
_K_PIECE_FP8 = 64
# FP8 LDS swizzle: 16-byte column groups XOR (row // per_phase) % groups.
# One KV row per phase keeps both the K reads (ds_read_b128) and the
# transposed V reads (ds_read_b64_tr_b8) free of bank conflicts; the 64-byte
# rope rows need four rows per phase for the same.
_KV_PER_PHASE = gl.constexpr(1)
_PE_PER_PHASE = gl.constexpr(4)
# KV tile stages in LDS: one being read by P @ V, one by the next tile's
# Q @ K^T, and two being filled, so each copy has a full iteration to land.
_NUM_STAGES = 4
_GFX950_CU_COUNT = 256
# A split must stream at least this many KV pages to amortize the reduce.
_MIN_PAGES_PER_SPLIT = 4
# Split partials the reduce kernel merges per step.
_REDUCE_SPLITS = 16
# Lazy rescale threshold in log2 units: the running max only moves when a tile
# raises it by more than this, so p = exp2(s - m) <= 2^8 = 256, which still
# fits FP8 e4m3 (max 448).
_RESCALE_THRESHOLD = gl.constexpr(8.0)
# sched_group_barrier instruction classes.
_SGB_MFMA = gl.constexpr(0x8)
_SGB_VMEM = gl.constexpr(0x10)
_SGB_TRANS = gl.constexpr(0x400)


@gluon_builtin
def _mfma_unscaled_fp8(a, b, acc, *, _semantic):
    # dot_scaled with None scales emits v_mfma_f32_32x32x64_f8f6f4 without
    # scale operands; plain mfma would select the K=16 FP8 instruction.
    fmt = "e4m3" if a.dtype == gl.float8e4nv else "e5m2"
    output = _semantic.dot_scaled(
        a,
        None,
        fmt,
        b,
        None,
        fmt,
        acc,
        fast_math=False,
        lhs_k_pack=True,
        rhs_k_pack=True,
        out_dtype=gl.float32,
    )
    return gl.tensor(output.handle, acc.type)


# fmt: off
@gluon.jit
def _wave_uniform_i32(value):
    # v_readfirstlane marks the value wave-uniform. A loop-carried page index
    # otherwise lands in a VGPR and every buffer load becomes a waterfall loop.
    return wave_uniform_i32(value)


@gluon.jit
def _rescale_row64(
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    x16, x17, x18, x19, x20, x21, x22, x23,
    x24, x25, x26, x27, x28, x29, x30, x31,
    x32, x33, x34, x35, x36, x37, x38, x39,
    x40, x41, x42, x43, x44, x45, x46, x47,
    x48, x49, x50, x51, x52, x53, x54, x55,
    x56, x57, x58, x59, x60, x61, x62, x63,
    a0, a1, a2, a3, a4, a5, a6, a7,
    a8, a9, a10, a11, a12, a13, a14, a15,
    a16, a17, a18, a19, a20, a21, a22, a23,
    a24, a25, a26, a27, a28, a29, a30, a31,
    a32, a33, a34, a35, a36, a37, a38, a39,
    a40, a41, a42, a43, a44, a45, a46, a47,
    a48, a49, a50, a51, a52, a53, a54, a55,
    a56, a57, a58, a59, a60, a61, a62, a63,
):
    # Every accumulator element a lane holds belongs to one row, so a0 is the
    # row's alpha. Lanes whose row kept its max skip the multiply; the whole
    # wave branches over it (s_cbranch_execz) when no lane needs it.
    if a0 != 1.0:
        x0, x1, x2, x3, x4, x5, x6, x7 = (
            x0 * a0, x1 * a0, x2 * a0, x3 * a0, x4 * a0, x5 * a0, x6 * a0, x7 * a0
        )
        x8, x9, x10, x11, x12, x13, x14, x15 = (
            x8 * a0, x9 * a0, x10 * a0, x11 * a0, x12 * a0, x13 * a0, x14 * a0, x15 * a0
        )
        x16, x17, x18, x19, x20, x21, x22, x23 = (
            x16 * a0, x17 * a0, x18 * a0, x19 * a0, x20 * a0, x21 * a0, x22 * a0, x23 * a0
        )
        x24, x25, x26, x27, x28, x29, x30, x31 = (
            x24 * a0, x25 * a0, x26 * a0, x27 * a0, x28 * a0, x29 * a0, x30 * a0, x31 * a0
        )
        x32, x33, x34, x35, x36, x37, x38, x39 = (
            x32 * a0, x33 * a0, x34 * a0, x35 * a0, x36 * a0, x37 * a0, x38 * a0, x39 * a0
        )
        x40, x41, x42, x43, x44, x45, x46, x47 = (
            x40 * a0, x41 * a0, x42 * a0, x43 * a0, x44 * a0, x45 * a0, x46 * a0, x47 * a0
        )
        x48, x49, x50, x51, x52, x53, x54, x55 = (
            x48 * a0, x49 * a0, x50 * a0, x51 * a0, x52 * a0, x53 * a0, x54 * a0, x55 * a0
        )
        x56, x57, x58, x59, x60, x61, x62, x63 = (
            x56 * a0, x57 * a0, x58 * a0, x59 * a0, x60 * a0, x61 * a0, x62 * a0, x63 * a0
        )
    return (
        x0, x1, x2, x3, x4, x5, x6, x7,
        x8, x9, x10, x11, x12, x13, x14, x15,
        x16, x17, x18, x19, x20, x21, x22, x23,
        x24, x25, x26, x27, x28, x29, x30, x31,
        x32, x33, x34, x35, x36, x37, x38, x39,
        x40, x41, x42, x43, x44, x45, x46, x47,
        x48, x49, x50, x51, x52, x53, x54, x55,
        x56, x57, x58, x59, x60, x61, x62, x63,
    )
# fmt: on


@gluon.aggregate
class ExtendConfig:
    N_HEADS: gl.constexpr
    HEAD_GROUP: gl.constexpr
    BLOCK_Q: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    PAGE_SIZE: gl.constexpr
    TILES_PER_PAGE: gl.constexpr
    NUM_STAGES: gl.constexpr
    KV_LORA_RANK: gl.constexpr
    ROPE_DIM: gl.constexpr
    CHUNK: gl.constexpr
    NUM_CHUNKS: gl.constexpr
    K_PIECE: gl.constexpr
    NUM_K_PIECES: gl.constexpr
    SM_SCALE: gl.constexpr
    IS_CAUSAL: gl.constexpr
    SPLIT: gl.constexpr
    IS_FP8: gl.constexpr
    Q_STRIDE_T: gl.constexpr
    Q_STRIDE_H: gl.constexpr
    O_STRIDE_T: gl.constexpr
    O_STRIDE_H: gl.constexpr
    KV_PAGE_STRIDE: gl.constexpr
    KV_TOKEN_STRIDE: gl.constexpr
    mma_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    load_layout: gl.constexpr
    load_pe_layout: gl.constexpr
    store_layout: gl.constexpr
    kv_smem_layout: gl.constexpr
    pe_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        N_HEADS,
        HEAD_GROUP,
        BLOCK_Q,
        BLOCK_M,
        BLOCK_N,
        PAGE_SIZE,
        KV_LORA_RANK,
        ROPE_DIM,
        SM_SCALE,
        IS_CAUSAL,
        SPLIT,
        NUM_WARPS,
        KV_DTYPE,
        Q_STRIDE_T,
        Q_STRIDE_H,
        O_STRIDE_T,
        O_STRIDE_H,
        KV_PAGE_STRIDE,
        KV_TOKEN_STRIDE,
    ):
        assert BLOCK_M == 32 * NUM_WARPS
        assert BLOCK_Q * HEAD_GROUP <= BLOCK_M
        assert KV_LORA_RANK == 4 * _CHUNK
        assert PAGE_SIZE % BLOCK_N == 0
        is_fp8 = KV_DTYPE.is_fp8()
        mma_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[32, 32, 64] if is_fp8 else [32, 32, 16],
            transposed=True,
            warps_per_cta=[NUM_WARPS, 1],
        )
        qk_kw = 16 if is_fp8 else 8
        pv_kw = 8 if is_fp8 else 4
        k_layout = gl.DotOperandLayout(1, mma_layout, k_width=qk_kw)
        v_layout = gl.DotOperandLayout(1, mma_layout, k_width=pv_kw)
        # 128-bit global->LDS copies; each warp covers whole rows of a chunk.
        vec = 16 if is_fp8 else 8
        load_layout = gl.BlockedLayout(
            [1, vec], [64 // (_CHUNK // vec), _CHUNK // vec], [NUM_WARPS, 1], [1, 0]
        )
        load_pe_layout = gl.BlockedLayout(
            [1, vec],
            [64 // (ROPE_DIM // vec), ROPE_DIM // vec],
            [NUM_WARPS, 1],
            [1, 0],
        )
        # One LDS buffer feeds both the K (k-contiguous) and V reads.
        if is_fp8:
            kv_smem_layout = gl.SwizzledSharedLayout(
                vec, _KV_PER_PHASE, _CHUNK // vec, [1, 0]
            )
            pe_smem_layout = gl.SwizzledSharedLayout(
                vec, _PE_PER_PHASE, ROPE_DIM // vec, [1, 0]
            )
        else:
            kv_smem_layout = padded_shared_layout(
                k_layout, [BLOCK_N, _CHUNK], KV_DTYPE, is_k_contig=True
            )
            pe_smem_layout = padded_shared_layout(
                k_layout, [BLOCK_N, ROPE_DIM], KV_DTYPE, is_k_contig=True
            )

        self.N_HEADS = gl.constexpr(N_HEADS)
        self.HEAD_GROUP = gl.constexpr(HEAD_GROUP)
        self.BLOCK_Q = gl.constexpr(BLOCK_Q)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.TILES_PER_PAGE = gl.constexpr(PAGE_SIZE // BLOCK_N)
        self.NUM_STAGES = gl.constexpr(_NUM_STAGES)
        self.KV_LORA_RANK = gl.constexpr(KV_LORA_RANK)
        self.ROPE_DIM = gl.constexpr(ROPE_DIM)
        self.CHUNK = gl.constexpr(_CHUNK)
        self.NUM_CHUNKS = gl.constexpr(KV_LORA_RANK // _CHUNK)
        k_piece = _K_PIECE_FP8 if is_fp8 else _CHUNK
        self.K_PIECE = gl.constexpr(k_piece)
        # Latent pieces, then the rope part as the last piece.
        self.NUM_K_PIECES = gl.constexpr(KV_LORA_RANK // k_piece + 1)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.SPLIT = gl.constexpr(SPLIT)
        self.IS_FP8 = gl.constexpr(is_fp8)
        self.Q_STRIDE_T = gl.constexpr(Q_STRIDE_T)
        self.Q_STRIDE_H = gl.constexpr(Q_STRIDE_H)
        self.O_STRIDE_T = gl.constexpr(O_STRIDE_T)
        self.O_STRIDE_H = gl.constexpr(O_STRIDE_H)
        self.KV_PAGE_STRIDE = gl.constexpr(KV_PAGE_STRIDE)
        self.KV_TOKEN_STRIDE = gl.constexpr(KV_TOKEN_STRIDE)
        self.mma_layout = gl.constexpr(mma_layout)
        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, mma_layout, k_width=qk_kw))
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, mma_layout, k_width=pv_kw))
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.load_pe_layout = gl.constexpr(load_pe_layout)
        # Each wave keeps its 32 rows; only the lane-32 partner swaps columns
        # to form eight-element (16-byte) bf16 stores.
        self.store_layout = gl.constexpr(
            gl.BlockedLayout([1, 8], [32, 2], [NUM_WARPS, 1], [0, 1])
        )
        self.kv_smem_layout = gl.constexpr(kv_smem_layout)
        self.pe_smem_layout = gl.constexpr(pe_smem_layout)


@gluon.aggregate
class ExtendProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    kv_ptr: gl.tensor
    page_table_ptr: gl.tensor
    batch: gl.tensor
    page_table_stride: gl.tensor
    head_base: gl.tensor
    q_pos_base: gl.tensor
    seq_base: gl.tensor
    seq_len: gl.tensor
    prefix: gl.tensor
    cache_len: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        kv_ptr,
        page_table_ptr,
        batch,
        page_table_stride,
        head_base,
        q_pos_base,
        seq_base,
        seq_len,
        prefix,
        cache_len,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.kv_ptr = kv_ptr
        self.page_table_ptr = page_table_ptr
        self.batch = batch
        self.page_table_stride = page_table_stride
        self.head_base = head_base
        self.q_pos_base = q_pos_base
        self.seq_base = seq_base
        self.seq_len = seq_len
        self.prefix = prefix
        self.cache_len = cache_len

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        kv_ptr,
        page_table_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
        num_seqs,
        page_table_stride,
    ):
        # Ragged grid: program_id(0) is a global q-block that binary-searches
        # cu_seqlens_q for its request (see mha/extend.py).
        pid_q = gl.program_id(0)
        batch = _find_seq_idx(cu_seqlens_q_ptr, pid_q, num_seqs, cfg.BLOCK_Q)
        seq_base = gl.load(cu_seqlens_q_ptr + batch)
        seq_len = gl.load(cu_seqlens_q_ptr + batch + 1) - seq_base
        q_pos_base = (pid_q - (seq_base // cfg.BLOCK_Q + batch)) * cfg.BLOCK_Q
        cache_len = gl.load(cache_seqlens_ptr + batch)
        return ExtendProgram(
            cfg,
            q_ptr,
            kv_ptr,
            page_table_ptr,
            batch,
            page_table_stride,
            gl.program_id(1) * cfg.HEAD_GROUP,
            q_pos_base,
            seq_base,
            seq_len,
            cache_len - seq_len,
            cache_len,
        )

    @gluon.jit
    def rows(self, layout: gl.constexpr):
        # Row m packs (query position m // HEAD_GROUP, head m % HEAD_GROUP).
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=layout)
        q_idx = offs_m // cfg.HEAD_GROUP
        q_pos = self.q_pos_base + q_idx
        head = self.head_base + offs_m % cfg.HEAD_GROUP
        valid = (q_idx < cfg.BLOCK_Q) & (q_pos < self.seq_len)
        return q_pos, head, valid

    @gluon.jit
    def load_q(self, col: gl.constexpr, width: gl.constexpr):
        # Load q[:, col:col + width] straight into the MFMA operand layout,
        # casting to the KV dtype (bf16 queries against an FP8 cache).
        cfg = self.cfg
        q_pos, head, valid = self.rows(gl.SliceLayout(1, cfg.q_layout))
        offs_d = col + gl.arange(0, width, layout=gl.SliceLayout(0, cfg.q_layout))
        offsets = (
            (self.seq_base + q_pos)[:, None] * cfg.Q_STRIDE_T
            + head[:, None] * cfg.Q_STRIDE_H
            + offs_d[None, :]
        ).to(gl.int32)
        q = cdna4.buffer_load(self.q_ptr, offsets, mask=valid[:, None], other=0.0)
        return q.to(self.kv_ptr.dtype.element_ty)

    @gluon.jit
    def load_q_pieces(self):
        # Q split along the head dim to match the K pieces of `qk`.
        cfg = self.cfg
        if cfg.K_PIECE == cfg.CHUNK:
            return (
                self.load_q(0 * cfg.CHUNK, cfg.CHUNK),
                self.load_q(1 * cfg.CHUNK, cfg.CHUNK),
                self.load_q(2 * cfg.CHUNK, cfg.CHUNK),
                self.load_q(3 * cfg.CHUNK, cfg.CHUNK),
                self.load_q(cfg.KV_LORA_RANK, cfg.ROPE_DIM),
            )
        else:
            return (
                self.load_q(0 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(1 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(2 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(3 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(4 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(5 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(6 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(7 * cfg.K_PIECE, cfg.K_PIECE),
                self.load_q(cfg.KV_LORA_RANK, cfg.ROPE_DIM),
            )

    @gluon.jit
    def load_page(self, tile, tile_end):
        # Page index holding `tile`, or 0 for a padding tile past the range.
        return gl.load(
            self.page_table_ptr
            + self.batch * self.page_table_stride
            + tile // self.cfg.TILES_PER_PAGE,
            mask=tile < tile_end,
            other=0,
        )

    @gluon.jit
    def issue_tile(self, kv_smem, pe_smem, stage, page, tile, tile_end):
        # Copy one KV tile into LDS stage `stage`. Rows past the cache end, or
        # every row of a padding tile, read as zero, so masked keys never
        # carry NaN garbage into P @ V.
        cfg = self.cfg
        num_valid = gl.where(tile < tile_end, self.cache_len - tile * cfg.BLOCK_N, 0)
        page = _wave_uniform_i32(page)
        row = (tile % cfg.TILES_PER_PAGE) * cfg.BLOCK_N
        base = (
            self.kv_ptr
            + page.to(gl.int64) * cfg.KV_PAGE_STRIDE
            + row * cfg.KV_TOKEN_STRIDE
        )
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.CHUNK, layout=gl.SliceLayout(0, cfg.load_layout))
        row_mask = (offs_n < num_valid)[:, None]
        kv_dst = kv_smem
        pe_dst = pe_smem
        cols = offs_d[None, :]
        if cfg.IS_FP8:
            # The DMA writes linear LDS addresses; swizzle the source columns
            # instead (16-byte groups XOR the row phase).
            kv_dst = kv_smem.reinterpret(
                kv_smem.dtype, kv_smem.shape, gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            )
            pe_dst = pe_smem.reinterpret(
                pe_smem.dtype, pe_smem.shape, gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            )
            phase = (offs_n[:, None] // _KV_PER_PHASE) % (cfg.CHUNK // 16)
            cols = gl.max_contiguous(
                gl.multiple_of(cols ^ (phase * 16), [1, 16]), [1, 16]
            )
        for c in gl.static_range(cfg.NUM_CHUNKS):
            offsets = offs_n[:, None] * cfg.KV_TOKEN_STRIDE + (c * cfg.CHUNK + cols)
            async_copy.buffer_load_to_shared(
                kv_dst.index(stage * cfg.NUM_CHUNKS + c),
                base,
                offsets,
                mask=row_mask,
            )
        offs_n_pe = gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_pe_layout)
        )
        offs_d_pe = cfg.KV_LORA_RANK + gl.arange(
            0, cfg.ROPE_DIM, layout=gl.SliceLayout(0, cfg.load_pe_layout)
        )
        cols_pe = offs_d_pe[None, :]
        if cfg.IS_FP8:
            phase_pe = (offs_n_pe[:, None] // _PE_PER_PHASE) % (cfg.ROPE_DIM // 16)
            cols_pe = cfg.KV_LORA_RANK + gl.max_contiguous(
                gl.multiple_of((cols_pe - cfg.KV_LORA_RANK) ^ (phase_pe * 16), [1, 16]),
                [1, 16],
            )
        offsets_pe = offs_n_pe[:, None] * cfg.KV_TOKEN_STRIDE + cols_pe
        async_copy.buffer_load_to_shared(
            pe_dst.index(stage),
            base,
            offsets_pe,
            mask=(offs_n_pe < num_valid)[:, None],
        )
        async_copy.commit_group()

    @gluon.jit
    def dot(self, a, b, acc):
        if self.cfg.IS_FP8:
            return _mfma_unscaled_fp8(a, b, acc)
        return cdna4.mfma(a, b, acc)

    @gluon.jit
    def load_k(self, kv_smem, pe_smem, stage, piece: gl.constexpr):
        # K^T rows [piece * K_PIECE, +K_PIECE): the latent pieces, then rope.
        cfg = self.cfg
        per_chunk: gl.constexpr = cfg.CHUNK // cfg.K_PIECE
        if piece + 1 < cfg.NUM_K_PIECES:
            smem = kv_smem.index(stage * cfg.NUM_CHUNKS + piece // per_chunk)
            if per_chunk > 1:
                smem = smem.slice((piece % per_chunk) * cfg.K_PIECE, cfg.K_PIECE, dim=1)
        else:
            smem = pe_smem.index(stage)
        return async_copy.load_shared_relaxed(smem.permute([1, 0]), cfg.k_layout)

    @gluon.jit
    def load_v(self, kv_smem, stage, chunk: gl.constexpr):
        cfg = self.cfg
        return async_copy.load_shared_relaxed(
            kv_smem.index(stage * cfg.NUM_CHUNKS + chunk), cfg.v_layout
        )

    @gluon.jit
    def qk(self, q, kv_smem, pe_smem, stage):
        # Q @ K^T of one tile as a chain of K_PIECE-deep steps. Each step's K
        # operand is read before the previous step's MFMAs; the caller's
        # group barriers pull the previous tile's softmax into the MFMA
        # shadows.
        cfg = self.cfg
        qk = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layout=cfg.mma_layout)
        k = self.load_k(kv_smem, pe_smem, stage, 0)
        sched_barrier()
        for j in gl.static_range(cfg.NUM_K_PIECES):
            if j + 1 < cfg.NUM_K_PIECES:
                k_next = self.load_k(kv_smem, pe_smem, stage, j + 1)
            qk = self.dot(q[j], k, qk)
            if j + 1 < cfg.NUM_K_PIECES:
                k = k_next
        return qk

    @gluon.jit
    def mask_scores(self, qk, tile, mask_from, diag):
        # Only tiles touching the causal diagonal or the cache end need masks.
        cfg = self.cfg
        start_n = tile * cfg.BLOCK_N
        if start_n + cfg.BLOCK_N > mask_from:
            col = start_n + gl.arange(
                0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.mma_layout)
            )
            visible = col[None, :] < self.cache_len
            if cfg.IS_CAUSAL:
                visible = visible & (col[None, :] <= diag[:, None])
            qk = gl.where(visible, qk, -float("inf"))
        return qk

    @gluon.jit
    def softmax(self, qk, m_i, l_i):
        # Online softmax in base-2 units with a lazy max: keep the stale max
        # unless the tile raises it by more than _RESCALE_THRESHOLD. The scale
        # folds into the exp argument (one FMA a score); it is positive, so
        # the max of raw scores is the max of scaled ones.
        cfg = self.cfg
        m_tile = maximum(m_i, max(qk, 1) * cfg.SM_SCALE)
        m_new = gl.where(m_tile - m_i > _RESCALE_THRESHOLD, m_tile, m_i)
        # A row with no visible key yet keeps m = -inf; shift by 0 instead.
        m_use = gl.where(m_new == -float("inf"), 0.0, m_new)
        p = gl.exp2(qk * cfg.SM_SCALE - m_use[:, None])
        alpha = gl.exp2(m_i - m_use)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        p = gl.convert_layout(p.to(self.kv_ptr.dtype.element_ty), cfg.p_layout)
        return p, alpha, m_new, l_i

    @gluon.jit
    def pv(
        self,
        p,
        v,
        kv_smem,
        stage,
        acc0,
        acc1,
        acc2,
        acc3,
        pe_smem,
        issue_stage,
        page,
        issue_tile,
        tile_end,
        ISSUE: gl.constexpr,
    ):
        # P @ V over the four latent chunks; `v` is chunk 0, already read.
        # Each chunk's V operand is read before the previous chunk's MFMAs.
        v_next = self.load_v(kv_smem, stage, 1)
        sched_barrier()
        acc0 = self.dot(p, v, acc0)
        if ISSUE:
            # Spread the next copy's buffer_load...lds over the chunk-0 MFMAs
            # (three VMEM issues per MFMA) instead of a burst that stalls the
            # issue port.
            self.issue_tile(kv_smem, pe_smem, issue_stage, page, issue_tile, tile_end)
            for _i in gl.static_range(4):
                sched_group_barrier(_SGB_MFMA, 1)
                sched_group_barrier(_SGB_VMEM, 3)
            sched_barrier()
        v = self.load_v(kv_smem, stage, 2)
        sched_barrier()
        acc1 = self.dot(p, v_next, acc1)
        sched_barrier()
        v_next = self.load_v(kv_smem, stage, 3)
        sched_barrier()
        acc2 = self.dot(p, v, acc2)
        sched_barrier()
        acc3 = self.dot(p, v_next, acc3)
        return acc0, acc1, acc2, acc3


@gluon.jit
def _rescale(acc, alpha):
    alpha_b = gl.broadcast(alpha[:, None], acc)[0]
    return gl.map_elementwise(_rescale_row64, acc, alpha_b, pack=64)[0]


@gluon.jit
def _store_chunk(out_ptr, o, program, col: gl.constexpr, stride_t, stride_h):
    # Store one 128-wide output chunk; o is already in its store dtype.
    cfg = program.cfg
    layout: gl.constexpr = o.type.layout
    q_pos, head, valid = program.rows(gl.SliceLayout(1, layout))
    offs_d = col + gl.arange(0, cfg.CHUNK, layout=gl.SliceLayout(0, layout))
    offsets = (
        (program.seq_base + q_pos)[:, None] * stride_t
        + head[:, None] * stride_h
        + offs_d[None, :]
    ).to(gl.int32)
    cdna4.buffer_store(o, out_ptr, offsets, mask=valid[:, None])


@gluon.jit
def _finish_chunk(out_ptr, acc, inv_l, program, col: gl.constexpr):
    cfg = program.cfg
    o = acc * inv_l[:, None]
    if cfg.SPLIT:
        # fp32 partials keep the accumulator layout: 4 contiguous floats a lane.
        _store_chunk(
            out_ptr,
            o,
            program,
            col,
            cfg.N_HEADS * cfg.KV_LORA_RANK,
            cfg.KV_LORA_RANK,
        )
    else:
        o = gl.convert_layout(o.to(out_ptr.dtype.element_ty), cfg.store_layout)
        _store_chunk(out_ptr, o, program, col, cfg.O_STRIDE_T, cfg.O_STRIDE_H)


def extend_launch_metadata(grid, kernel, args):
    """Report an upper bound on attention work without reading device lengths.

    Every query is assumed to see the page table's full width of keys; the
    causal triangle over the new tokens is ignored. Bytes count q, output and
    each request's KV pages once.
    """
    total_q, heads, qk_dim = args["q_ptr"].shape
    batch, max_pages = args["page_table_ptr"].shape
    kv_len = max_pages * args["kv_ptr"].shape[1]
    v_dim = args["KV_LORA_RANK"]
    flops = 2 * total_q * heads * kv_len * (qk_dim + v_dim)
    kv_bytes = batch * kv_len * qk_dim * args["kv_ptr"].element_size()
    q_bytes = args["q_ptr"].numel() * args["q_ptr"].element_size()
    out_bytes = total_q * heads * v_dim * args["out_ptr"].element_size()
    return {
        "name": kernel.name,
        "flops8" if args["kv_ptr"].element_size() == 1 else "flops16": flops,
        "bytes": kv_bytes + q_bytes + out_bytes,
    }


@gluon.jit(
    launch_metadata=extend_launch_metadata,
    do_not_specialize=("num_seqs", "page_table_stride", "split_stride"),
)
def gluon_mla_extend_gfx950(
    q_ptr,
    kv_ptr,
    page_table_ptr,
    out_ptr,
    lse_ptr,
    cu_seqlens_q_ptr,
    cache_seqlens_ptr,
    num_seqs,
    page_table_stride,
    split_stride,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    O_STRIDE_T: gl.constexpr,
    O_STRIDE_H: gl.constexpr,
    KV_PAGE_STRIDE: gl.constexpr,
    KV_TOKEN_STRIDE: gl.constexpr,
    SM_SCALE: gl.constexpr,
    N_HEADS: gl.constexpr,
    HEAD_GROUP: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    KV_LORA_RANK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    SPLIT: gl.constexpr,
    SCHED_LIBRARY_HASH: gl.constexpr,  # Cache dependency; not a device operand.
):
    # Grid: (q-blocks, N_HEADS // HEAD_GROUP, KV splits). With SPLIT, each
    # program streams its slice of the visible KV and writes a normalized fp32
    # partial plus base-2 LSE at out_ptr / lse_ptr + split * split_stride; the
    # reduce kernel merges them. Without SPLIT, out_ptr is the final output.
    cfg = ExtendConfig(
        N_HEADS,
        HEAD_GROUP,
        BLOCK_Q,
        BLOCK_M,
        BLOCK_N,
        PAGE_SIZE,
        KV_LORA_RANK,
        ROPE_DIM,
        SM_SCALE,
        IS_CAUSAL,
        SPLIT,
        NUM_WARPS,
        kv_ptr.dtype.element_ty,
        Q_STRIDE_T,
        Q_STRIDE_H,
        O_STRIDE_T,
        O_STRIDE_H,
        KV_PAGE_STRIDE,
        KV_TOKEN_STRIDE,
    )
    program = ExtendProgram.create(
        cfg,
        q_ptr,
        kv_ptr,
        page_table_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
        num_seqs,
        page_table_stride,
    )
    # Over-provisioned ragged block past this request's queries.
    if program.q_pos_base >= program.seq_len:
        return

    if cfg.IS_CAUSAL:
        # Keys up to the deepest query position in this block.
        kv_end = min(
            program.cache_len, program.prefix + program.q_pos_base + cfg.BLOCK_Q
        )
    else:
        kv_end = program.cache_len
    num_tiles = gl.cdiv(kv_end, cfg.BLOCK_N)
    if cfg.SPLIT:
        split = gl.program_id(2)
        tiles_per_split = gl.cdiv(num_tiles, gl.num_programs(2))
        tile_begin = split * tiles_per_split
        tile_end = min(tile_begin + tiles_per_split, num_tiles)
    else:
        tile_begin = 0
        tile_end = num_tiles

    kv_dtype: gl.constexpr = kv_ptr.dtype.element_ty
    kv_smem = gl.allocate_shared_memory(
        kv_dtype,
        [4 * cfg.NUM_CHUNKS, cfg.BLOCK_N, cfg.CHUNK],
        cfg.kv_smem_layout,
    )
    pe_smem = gl.allocate_shared_memory(
        kv_dtype, [4, cfg.BLOCK_N, cfg.ROPE_DIM], cfg.pe_smem_layout
    )

    row_layout: gl.constexpr = gl.SliceLayout(1, cfg.mma_layout)
    m_i = gl.full([cfg.BLOCK_M], -float("inf"), gl.float32, layout=row_layout)
    l_i = gl.zeros([cfg.BLOCK_M], gl.float32, layout=row_layout)
    acc0 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    acc1 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    acc2 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    acc3 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    q_pos, _, _ = program.rows(row_layout)
    diag = program.prefix + q_pos
    # Tiles from here on touch the causal diagonal or the cache end.
    if cfg.IS_CAUSAL:
        mask_from = min(program.prefix + program.q_pos_base, program.cache_len)
    else:
        mask_from = program.cache_len

    # Software pipeline, one tile deep: iteration `tile` runs Q @ K^T of tile
    # + 1 with the softmax of `tile` riding in its MFMA shadows, then P @ V of
    # `tile`. Stage (tile - tile_begin) % 4 holds `tile`; the copy of tile + 3,
    # issued inside P @ V of `tile`, fills the stage P @ V of tile - 1
    # released.
    if tile_begin < tile_end:
        page = program.load_page(tile_begin, tile_end)
        program.issue_tile(kv_smem, pe_smem, 0, page, tile_begin, tile_end)
        page = program.load_page(tile_begin + 1, tile_end)
        program.issue_tile(kv_smem, pe_smem, 1, page, tile_begin + 1, tile_end)
        page = program.load_page(tile_begin + 2, tile_end)
        program.issue_tile(kv_smem, pe_smem, 2, page, tile_begin + 2, tile_end)
        # Load the page index a tile ahead of its copy, so waiting on it does
        # not also wait for the copies issued before it (vmcnt is FIFO).
        next_page = program.load_page(tile_begin + cfg.NUM_STAGES - 1, tile_end)
        q = program.load_q_pieces()
        async_copy.wait_group(cfg.NUM_STAGES - 2)
        qk = program.qk(q, kv_smem, pe_smem, 0)
        qk = program.mask_scores(qk, tile_begin, mask_from, diag)

        for tile in range(tile_begin, tile_end - 1):
            stage = (tile - tile_begin) % cfg.NUM_STAGES
            next_stage = (stage + 1) % cfg.NUM_STAGES
            issue_stage = (stage + cfg.NUM_STAGES - 1) % cfg.NUM_STAGES
            # Tile + 1 has landed. The barrier the compiler emits after this
            # wait also covers the relaxed LDS reads: every wave has finished
            # P @ V of tile - 1 (its MFMAs consumed the V reads), so the copy
            # of tile + 3 issued below may overwrite that stage.
            async_copy.wait_group(cfg.NUM_STAGES - 3)
            page = next_page
            next_page = program.load_page(tile + cfg.NUM_STAGES, tile_end)
            sched_barrier()
            qk_next = program.qk(q, kv_smem, pe_smem, next_stage)
            v = program.load_v(kv_smem, stage, 0)
            p, alpha, m_i, l_i = program.softmax(qk, m_i, l_i)
            # Interleave the softmax exp2s into the Q @ K^T MFMA chain, two
            # per MFMA; the scheduler places the other ALU work itself.
            for _i in gl.static_range(16):
                sched_group_barrier(_SGB_MFMA, 1)
                sched_group_barrier(_SGB_TRANS, 2)
            sched_barrier()
            acc0 = _rescale(acc0, alpha)
            acc1 = _rescale(acc1, alpha)
            acc2 = _rescale(acc2, alpha)
            acc3 = _rescale(acc3, alpha)
            acc0, acc1, acc2, acc3 = program.pv(
                p,
                v,
                kv_smem,
                stage,
                acc0,
                acc1,
                acc2,
                acc3,
                pe_smem,
                issue_stage,
                page,
                tile + cfg.NUM_STAGES - 1,
                tile_end,
                True,
            )
            qk = program.mask_scores(qk_next, tile + 1, mask_from, diag)

        stage = (tile_end - 1 - tile_begin) % cfg.NUM_STAGES
        p, alpha, m_i, l_i = program.softmax(qk, m_i, l_i)
        acc0 = _rescale(acc0, alpha)
        acc1 = _rescale(acc1, alpha)
        acc2 = _rescale(acc2, alpha)
        acc3 = _rescale(acc3, alpha)
        v = program.load_v(kv_smem, stage, 0)
        acc0, acc1, acc2, acc3 = program.pv(
            p,
            v,
            kv_smem,
            stage,
            acc0,
            acc1,
            acc2,
            acc3,
            pe_smem,
            0,
            next_page,
            tile_end,
            tile_end,
            False,
        )
    async_copy.wait_group(0)

    has_kv = l_i > 0.0
    inv_l = gl.where(has_kv, 1.0 / gl.where(has_kv, l_i, 1.0), 0.0)
    if cfg.SPLIT:
        split_offset = gl.program_id(2).to(gl.int64) * split_stride
        out_ptr = out_ptr + split_offset * cfg.KV_LORA_RANK
        lse_ptr = lse_ptr + split_offset
    _finish_chunk(out_ptr, acc0, inv_l, program, 0 * cfg.CHUNK)
    _finish_chunk(out_ptr, acc1, inv_l, program, 1 * cfg.CHUNK)
    _finish_chunk(out_ptr, acc2, inv_l, program, 2 * cfg.CHUNK)
    _finish_chunk(out_ptr, acc3, inv_l, program, 3 * cfg.CHUNK)
    if cfg.SPLIT:
        # Base-2 LSE of this split; -inf marks a split with no visible key.
        _, head, valid = program.rows(row_layout)
        lse = gl.where(has_kv, m_i + gl.log2(gl.where(has_kv, l_i, 1.0)), -float("inf"))
        offsets = ((program.seq_base + q_pos) * cfg.N_HEADS + head).to(gl.int32)
        cdna4.buffer_store(lse, lse_ptr, offsets, mask=valid)


def extend_reduce_launch_metadata(grid, kernel, args):
    """Report the bytes of merging the split partials into the output."""
    part_bytes = args["part_o_ptr"].numel() * args["part_o_ptr"].element_size()
    lse_bytes = args["part_lse_ptr"].numel() * args["part_lse_ptr"].element_size()
    out_bytes = args["out_ptr"].numel() * args["out_ptr"].element_size()
    return {"name": kernel.name, "bytes": part_bytes + lse_bytes + out_bytes}


@gluon.jit(
    launch_metadata=extend_reduce_launch_metadata,
    do_not_specialize=("num_splits", "split_stride"),
)
def gluon_mla_extend_reduce_gfx950(
    part_o_ptr,
    part_lse_ptr,
    out_ptr,
    num_splits,
    split_stride,
    N_HEADS: gl.constexpr,
    O_STRIDE_T: gl.constexpr,
    O_STRIDE_H: gl.constexpr,
    KV_LORA_RANK: gl.constexpr,
    REDUCE_SPLITS: gl.constexpr,
):
    # Merge the per-split partials of one (token, head) row. Grid is
    # (total_q, N_HEADS). A first pass takes the max split LSE; the second
    # loads REDUCE_SPLITS partials at a time so their loads overlap instead of
    # one dependent load per split.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
    split_layout: gl.constexpr = gl.SliceLayout(1, layout)
    token = gl.program_id(0)
    head = gl.program_id(1)
    row = token * N_HEADS + head
    offs_s = gl.arange(0, REDUCE_SPLITS, layout=split_layout)
    offs_d = gl.arange(0, KV_LORA_RANK, layout=gl.SliceLayout(0, layout))
    m = gl.full([REDUCE_SPLITS], -float("inf"), gl.float32, layout=split_layout)
    for start in range(0, num_splits, REDUCE_SPLITS):
        split = start + offs_s
        lse = gl.load(
            part_lse_ptr + split * split_stride + row,
            mask=split < num_splits,
            other=-float("inf"),
        )
        m = gl.maximum(m, lse)
    m_max = gl.max(m, 0)
    # Every split of a row saw no key only if the row has no visible key.
    m_max = gl.where(m_max == -float("inf"), 0.0, m_max)
    acc = gl.zeros([KV_LORA_RANK], gl.float32, layout=gl.SliceLayout(0, layout))
    l_i = gl.zeros([REDUCE_SPLITS], gl.float32, layout=split_layout)
    for start in range(0, num_splits, REDUCE_SPLITS):
        split = start + offs_s
        lse = gl.load(
            part_lse_ptr + split * split_stride + row,
            mask=split < num_splits,
            other=-float("inf"),
        )
        weight = gl.exp2(lse - m_max)
        offsets = (split * split_stride + row).to(gl.int64) * KV_LORA_RANK
        # Splits without a visible key may hold no partial; skip their loads.
        part = gl.load(
            part_o_ptr + offsets[:, None] + offs_d[None, :],
            mask=(lse != -float("inf"))[:, None],
            other=0.0,
        )
        acc += gl.sum(part * weight[:, None], 0)
        l_i += weight
    l_sum = gl.sum(l_i, 0)
    out = acc / gl.where(l_sum > 0.0, l_sum, 1.0)
    gl.store(
        out_ptr + token * O_STRIDE_T + head * O_STRIDE_H + offs_d,
        out.to(out_ptr.dtype.element_ty),
    )


def _select_head_group(num_heads: int) -> int:
    # Heads packed per program: the largest divisor of num_heads whose whole
    # query positions fill all but at most 1/8 of the 128 rows (12 -> 12 x 10
    # rows, 48 -> 24 x 5, 96 -> 32 x 4). Programs of other head groups re-read
    # the same pages from L2, so fewer, wider groups also save traffic.
    for group in range(min(num_heads, _BLOCK_M), 0, -1):
        if num_heads % group == 0 and _BLOCK_M % group <= _BLOCK_M // 8:
            return group
    return 1


def _select_kv_splits(base_ctas: int, num_tiles: int) -> int:
    # Split the KV stream only when the grid leaves CUs idle. Each program is a
    # full CU (four waves at one wave per SIMD), so fill one wave of CUs without
    # spilling into a second, and keep >= _MIN_PAGES_PER_SPLIT pages a split.
    splits = _GFX950_CU_COUNT // base_ctas if base_ctas < _GFX950_CU_COUNT else 1
    splits = min(splits, num_tiles // _MIN_PAGES_PER_SPLIT)
    return splits if splits > 1 else 1


def launch_gluon_mla_extend_gfx950(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    is_causal: bool,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run absorbed MLA extend over a paged latent cache on GFX950.

    Args:
        q: ``[total_q, num_heads, kv_lora_rank + qk_rope_head_dim]`` queries in
            the latent space, the same dtype as ``kv_cache`` or bf16 against
            an FP8 e4m3 cache (cast to FP8 on load).
        kv_cache: ``[num_pages, 64, 1, kv_lora_rank + qk_rope_head_dim]``;
            must already hold the current query tokens.
        page_table: ``[batch, max_pages]`` int32 page ids.
        cache_seqlens: ``[batch]`` int32 KV lengths including the new tokens.
        cu_seqlens_q: ``[batch + 1]`` int32 query offsets.
        cu_seqlens_kv: Unused; KV lengths come from ``cache_seqlens``.
        max_seqlen_q: Unused; the grid is ragged over ``cu_seqlens_q``.
        max_seqlen_k: Upper bound of ``cache_seqlens``; sizes the KV split.
        qk_nope_head_dim: Unused by absorbed attention.
        kv_lora_rank: Latent (value) width, 512.
        qk_rope_head_dim: RoPE width, 64.
        softmax_scale: Scale applied to ``q @ k^T``.
        is_causal: Must be True; query i of a request sees keys up to
            ``cache_len - q_len + i``.
        logit_cap: Must be 0.
        return_lse: Must be False.
        out: Optional ``[total_q, num_heads, kv_lora_rank]`` output.

    Returns:
        ``[total_q, num_heads, kv_lora_rank]`` attention output (bf16 for FP8
        inputs, else the input dtype).
    """
    del cu_seqlens_kv, max_seqlen_q, qk_nope_head_dim
    if not is_causal:
        raise NotImplementedError("gluon MLA extend gfx950 requires causal attention")
    if logit_cap != 0.0:
        raise NotImplementedError("gluon MLA extend gfx950 does not support logit_cap")
    if return_lse:
        raise NotImplementedError("gluon MLA extend gfx950 does not return LSE")
    if kv_lora_rank != 512 or qk_rope_head_dim != 64:
        raise NotImplementedError(
            "gluon MLA extend gfx950 requires kv_lora_rank=512 and "
            f"qk_rope_head_dim=64, got {kv_lora_rank} and {qk_rope_head_dim}"
        )
    head_dim = kv_lora_rank + qk_rope_head_dim
    if q.ndim != 3 or q.shape[2] != head_dim or q.stride(2) != 1:
        raise ValueError(f"q must be [total_q, num_heads, {head_dim}], got {q.shape}")
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (1, head_dim):
        raise ValueError(
            f"kv_cache must be [num_pages, page_size, 1, {head_dim}], "
            f"got {tuple(kv_cache.shape)}"
        )
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous")
    page_size = kv_cache.shape[1]
    if page_size != _PAGE_SIZE:
        raise NotImplementedError(
            f"gluon MLA extend gfx950 supports page size {_PAGE_SIZE}, got {page_size}"
        )
    supported = (
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.bfloat16, torch.bfloat16),
    )
    if (q.dtype, kv_cache.dtype) not in supported:
        raise TypeError(
            f"unsupported MLA extend dtypes q={q.dtype}, kv_cache={kv_cache.dtype}"
        )
    for name, t in (
        ("page_table", page_table),
        ("cache_seqlens", cache_seqlens),
        ("cu_seqlens_q", cu_seqlens_q),
    ):
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")

    total_q, num_heads, _ = q.shape
    batch = cache_seqlens.shape[0]
    out_shape = (total_q, num_heads, kv_lora_rank)
    out_dtype = torch.bfloat16
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=q.device)
    elif out.shape != out_shape or out.dtype != out_dtype or out.stride(2) != 1:
        raise ValueError(
            f"out must be {out_shape} {out_dtype} with unit last stride, got "
            f"{tuple(out.shape)} {out.dtype}"
        )
    if total_q == 0:
        return out

    head_group = _select_head_group(num_heads)
    block_q = _BLOCK_M // head_group
    num_head_groups = num_heads // head_group
    # Ragged q-block grid: request i starts at block cu_q[i] // block_q + i.
    num_q_blocks = (total_q - 1) // block_q + batch
    # A BF16 page is split into two 32-row tiles so four stages fit in LDS.
    block_n = _PAGE_SIZE if kv_cache.element_size() == 1 else _PAGE_SIZE // 2
    num_tiles = (max_seqlen_k + block_n - 1) // block_n
    num_splits = _select_kv_splits(num_q_blocks * num_head_groups, num_tiles)
    split = num_splits > 1
    if split:
        split_stride = total_q * num_heads
        part_o = torch.empty(
            (num_splits, total_q, num_heads, kv_lora_rank),
            dtype=torch.float32,
            device=q.device,
        )
        part_lse = torch.empty(
            (num_splits, total_q, num_heads), dtype=torch.float32, device=q.device
        )
        attn_out, attn_lse = part_o, part_lse
    else:
        split_stride = 0
        attn_out, attn_lse = out, out

    gluon_mla_extend_gfx950[(num_q_blocks, num_head_groups, num_splits)](
        q,
        kv_cache,
        page_table,
        attn_out,
        attn_lse,
        cu_seqlens_q,
        cache_seqlens,
        batch,
        page_table.stride(0),
        split_stride,
        Q_STRIDE_T=q.stride(0),
        Q_STRIDE_H=q.stride(1),
        O_STRIDE_T=out.stride(0),
        O_STRIDE_H=out.stride(1),
        KV_PAGE_STRIDE=kv_cache.stride(0),
        KV_TOKEN_STRIDE=kv_cache.stride(1),
        SM_SCALE=softmax_scale * _INV_LN2_VALUE,
        N_HEADS=num_heads,
        HEAD_GROUP=head_group,
        BLOCK_Q=block_q,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        PAGE_SIZE=_PAGE_SIZE,
        KV_LORA_RANK=kv_lora_rank,
        ROPE_DIM=qk_rope_head_dim,
        NUM_WARPS=_NUM_WARPS,
        IS_CAUSAL=is_causal,
        SPLIT=split,
        num_warps=_NUM_WARPS,
        **sched_barrier_compile_options(),
    )
    if split:
        gluon_mla_extend_reduce_gfx950[(total_q, num_heads)](
            part_o,
            part_lse,
            out,
            num_splits,
            split_stride,
            N_HEADS=num_heads,
            O_STRIDE_T=out.stride(0),
            O_STRIDE_H=out.stride(1),
            KV_LORA_RANK=kv_lora_rank,
            REDUCE_SPLITS=_REDUCE_SPLITS,
            num_warps=4,
        )
    return out


__all__ = ["launch_gluon_mla_extend_gfx950"]
