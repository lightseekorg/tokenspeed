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
#
# Recurrence-phase design (row-streamed fp32 state, register/smem-resident row
# strips, warp-level dots, minimal block syncs) is derived from the Apache-2.0
# licensed KDA packed-decode CUDA reference kernel published in SGLang
# PR #32541 (``kda_packed_decode.cuh``). The conv / f_b-gate / decay / beta
# phases are a port of our own Triton megafusion
# (``tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent.
# fused_recurrent_kda_megafuse_fwd_kernel``); the math is kept op-for-op
# identical to that kernel (up to fp32 op order in the folded L2 normalizers).

"""CuTe DSL fused KDA decode (conv1d + f_b gate GEMV + delta-rule recurrence).

Drop-in replacement for the Triton
``fused_recurrent_kda_megafuse_fwd_kernel`` at decode (one token per
sequence). One launch performs, per (sequence, value-head):

1. 4-tap depthwise causal conv + SiLU over the packed ``q|k|v`` projections,
   with the 3-tap conv window read from / shifted into ``conv_pool`` at the
   dual page indices;
2. the low-rank decay-gate GEMV ``g = w_fb[head slice] @ f_a`` plus
   ``dt_bias``, then the safe gate
   (``lower_bound * sigmoid(exp(A_log) * g)``) or the softplus gate
   (``-exp(A_log) * softplus(g)``), exponentiated to the per-K decay;
3. QK L2 normalization (+ scale) and the delta-rule recurrence
   ``h *= exp(gk); t = h^T k; v' = (v - t) * sigmoid(beta); h += k v'^T;
   o = h^T q`` against the fp32 state page, read at ``read_indices[n]`` and
   written at ``write_indices[n]`` (negative ids: read zeros / skip stores).

Layout note (why this is not a 1:1 translation of the CUDA reference): the
reference streams a ``[HV, V, K]`` K-contiguous state, so each V-row is one
512B contiguous line and the ``<h, k>`` dot is a single warp reduction. Our
pool is the FLA-native ``[pages, HV, K, V]`` with **V contiguous** (the
transpose), so this kernel streams K-rows instead and assigns V *columns* to
lanes:

- grid ``(NV, HV, N)`` with ``NV = V / BV``; each block owns a ``BV``-column
  slice of one head's ``[K=128, V=128]`` state slab. 8 warps x 32 lanes;
  warp ``w`` streams K-rows ``[16w, 16w + 16)`` and lane ``l`` owns
  ``CPL = BV / 32`` consecutive columns, so every state access is one
  ``4 * CPL``-byte vector and each row is a coalesced ``BV * 4``-byte line
  per warp.
- ``NV`` trades launch-width against per-head redundancy: the conv/GEMV
  front is recomputed per column-split block (like the Triton program's
  ``NV`` split), so small batches want ``NV = 4`` (4x blocks in flight)
  while large batches want ``NV = 1`` (front computed once per head). The
  launcher picks ``NV`` from the batch size; every variant is numerically
  identical.
- phase 0 issues the whole state read stream as ``cp.async`` copies into
  shared memory, overlapping it with the conv/GEMV front; each thread
  re-reads exactly the slots it copied, so ``cp.async.wait_all`` alone
  orders the reads (no extra barrier).
- the front runs barrier-free: raw SiLU q/k go to smem, the L2 normalizers
  distribute over the recurrence dots (``t * k_inv``, ``(v' * k_inv) * k``,
  ``o * q_inv``), the GEMV reads ``f_a`` straight from global, and warp
  ``w``'s GEMV rows equal its recurrence rows so the decay transcendentals
  stay warp-local. One barrier publishes the front, one combines the
  per-warp ``t`` partials, one combines the per-warp ``o`` partials.
"""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_stream, make_fake_tensor

_IDX_DTYPES = {torch.int32: Int32, torch.int64: Int64}

_NUM_WARPS = 8


def _pick_nv(batch: int, num_heads: int) -> int:
    """Column-split factor: enough blocks to spread over the SMs, no more.

    Measured on B300 (148 SMs, HV = 12): NV=2 is the broad optimum (24+
    blocks at bs1 with 8-byte state vectors and 2x front redundancy); NV=1
    wins in the band where ``batch * HV`` alone already fills the device
    (front computed once per head) before per-SM queueing takes over again.
    """
    blocks_at_nv1 = batch * num_heads
    if 96 <= blocks_at_nv1 < 160:
        return 1
    return 2


@dsl_user_op
def _load_bf16x4_as_f32(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load 4 consecutive bf16 (8 bytes, 8B-aligned) as four fp32 registers.

    One ``ld.global.v2.u32`` plus bit expansion (bf16 is the top half of
    fp32), replacing four scalar 2-byte loads + converts.
    """
    address = pointer.toint(loc=loc, ip=ip)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 w0, w1, t0, t1, t2, t3;\n\t"
        "ld.global.v2.u32 {w0, w1}, [$4];\n\t"
        "shl.b32 t0, w0, 16;\n\t"
        "and.b32 t1, w0, 0xffff0000;\n\t"
        "shl.b32 t2, w1, 16;\n\t"
        "and.b32 t3, w1, 0xffff0000;\n\t"
        "mov.b32 $0, t0;\n\t"
        "mov.b32 $1, t1;\n\t"
        "mov.b32 $2, t2;\n\t"
        "mov.b32 $3, t3;\n\t"
        "}",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.f32(), loc=loc),
        [llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Float32)


@dsl_user_op
def _cp_async_shared_global(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    *,
    size: cutlass.Constexpr[int],
    loc=None,
    ip=None,
) -> None:
    """Issue one 4/8/16-byte global->shared async copy (no register staging)."""
    assert size in (4, 8, 16)
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    gmem_addr = gmem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), gmem_addr.ir_value(loc=loc, ip=ip)],
        f"cp.async.ca.shared.global [$0], [$1], {size};",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cp_async_wait_all(*, loc=None, ip=None) -> None:
    """Wait for all previously issued cp.async copies of this thread."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.wait_all;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_global_f32x2(pointer: cute.Pointer, a, b, *, loc=None, ip=None) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_global_f32x4(
    pointer: cute.Pointer, a, b, c, d, *, loc=None, ip=None
) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + cute.math.exp(-x, fastmath=True))


@cute.jit
def _silu(x: Float32) -> Float32:
    return x * _sigmoid(x)


class KdaFusedDecodeKernel:
    """One-launch KDA decode step (conv + gate GEMV + recurrence).

    Compile-time configuration (one CUBIN per combination):

    Args:
        num_heads: Per-rank value heads ``HV``.
        head_dim: ``K == V`` head dim; only 128 is supported.
        d_fa: Width of the low-rank gate input ``f_a``.
        scale: Attention scale multiplied into the normalized ``q``.
        lower_bound: Safe-gate lower bound; ``None`` selects the softplus
            gate ``-exp(A_log) * softplus(g)``.
        has_cu_seqlens: Whether token offsets come from ``cu_seqlens``
            (varlen decode; sequences with zero tokens are skipped) or each
            sequence ``n`` reads token row ``n``.
        nv: Column-split blocks per (sequence, head); 1, 2, or 4.
        enable_pdl: Launch with programmatic dependent launch and fence via
            ``griddepcontrol`` (wait on the producer GEMV at entry, release
            dependents at exit).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        d_fa: int,
        scale: float,
        lower_bound: float | None,
        has_cu_seqlens: bool,
        nv: int,
        enable_pdl: bool,
    ):
        if head_dim != 128:
            raise ValueError("KdaFusedDecodeKernel is specialized for K=V=128")
        if d_fa % 128 != 0:
            # 32 lanes x 4-element vector loads per GEMV row chunk.
            raise ValueError("f_a width must be a multiple of 128")
        if nv not in (1, 2, 4):
            raise ValueError("nv must be 1, 2, or 4")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_fa = d_fa
        self.scale = scale
        self.lower_bound = lower_bound
        self.use_lower_bound = lower_bound is not None
        self.has_cu_seqlens = has_cu_seqlens
        self.nv = nv
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        qkv_raw: cute.Tensor,  # [T, 3P] bf16
        conv_w: cute.Tensor,  # [3P, 4] bf16
        conv_pool: cute.Tensor,  # [pages, 3P, 3] bf16
        f_a: cute.Tensor,  # [T, D_FA] bf16
        w_fb: cute.Tensor,  # [P, D_FA] bf16
        beta: cute.Tensor,  # [T, HV] bf16
        A_log: cute.Tensor,  # [HV] fp32
        dt_bias: cute.Tensor,  # [P] fp32
        o: cute.Tensor,  # [T, HV, V] bf16
        h_pool: cute.Tensor,  # [pages, HV, K, V] fp32
        read_indices: cute.Tensor,  # [N] int
        write_indices: cute.Tensor,  # [N] int
        cu_seqlens: cute.Tensor,  # [N+1] int32 (placeholder when unused)
        stream: CUstream,
    ):
        n_seq = read_indices.shape[0]
        grid = (self.nv, self.num_heads, n_seq)
        block = (2 * self.head_dim, 1, 1)
        # nv > 1 launches the column-split blocks of one head as a cluster:
        # with in-place conv paging (read page == write page) the bx == 0
        # block's q/k conv-window shift must not land before a sibling block
        # has read the old window, and only a cluster barrier can order that
        # across blocks (the Triton megafusion relies on same-wave residency
        # for this, which stops holding once the grid exceeds one wave).
        cluster = (self.nv, 1, 1) if self.nv > 1 else None
        self.kernel(
            qkv_raw,
            conv_w,
            conv_pool,
            f_a,
            w_fb,
            beta,
            A_log,
            dt_bias,
            o,
            h_pool,
            read_indices,
            write_indices,
            cu_seqlens,
        ).launch(
            grid=grid,
            block=block,
            cluster=cluster,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        qkv_raw: cute.Tensor,
        conv_w: cute.Tensor,
        conv_pool: cute.Tensor,
        f_a: cute.Tensor,
        w_fb: cute.Tensor,
        beta: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h_pool: cute.Tensor,
        read_indices: cute.Tensor,
        write_indices: cute.Tensor,
        cu_seqlens: cute.Tensor,
    ):
        K = self.head_dim
        V = self.head_dim
        HV = self.num_heads
        P = HV * K
        D_FA = self.d_fa
        BV = V // self.nv  # columns per block
        CPL = BV // 32  # consecutive columns per lane (1, 2, or 4)
        ROWS = K // _NUM_WARPS  # 16 K-rows per warp

        tid, _, _ = cute.arch.thread_idx()
        i_v, i_hv, i_n = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        smem = cutlass.utils.SmemAllocator()
        s_q = smem.allocate_tensor(Float32, cute.make_layout((K,)))
        s_k = smem.allocate_tensor(Float32, cute.make_layout((K,)))
        s_dec = smem.allocate_tensor(Float32, cute.make_layout((K,)))
        s_v = smem.allocate_tensor(Float32, cute.make_layout((BV,)))
        s_norm = smem.allocate_tensor(Float32, cute.make_layout((_NUM_WARPS,)))
        # Row-major so partial writes and per-lane column reads are shared
        # memory bank conflict free.
        s_red_t = smem.allocate_tensor(
            Float32, cute.make_layout((_NUM_WARPS, BV), stride=(BV, 1))
        )
        s_red_o = smem.allocate_tensor(
            Float32, cute.make_layout((_NUM_WARPS, BV), stride=(BV, 1))
        )
        # This block's [K, BV] state slice, cp.async-prefetched at kernel
        # entry so the state read stream overlaps the whole conv/GEMV front.
        s_h = smem.allocate_tensor(
            Float32, cute.make_layout((K, BV), stride=(BV, 1)), byte_alignment=16
        )

        # PDL note: only qkv_raw / f_a / beta come from the same-stream
        # producer GEMV. Everything else (indices, cu_seqlens, both state
        # pools, conv/w_fb weights) pre-exists the producer, so under PDL the
        # kernel front-loads the whole cp.async state stream, the conv taps,
        # and the conv windows BEFORE griddepcontrol_wait — the state stream
        # then hides behind the producer's tail. (The Triton megafusion can
        # only fence the whole kernel.)
        if cutlass.const_expr(self.has_cu_seqlens):
            bos = cu_seqlens[i_n]
            t_len = cu_seqlens[i_n + 1] - bos
        else:
            bos = Int32(i_n)
            t_len = Int32(1)

        # Zero-length sequences (graph padding) do nothing, exactly like the
        # Triton kernel's early return. The branch is block-uniform, so the
        # barriers below are safe.
        if t_len > 0:
            r_page = read_indices[i_n]
            w_page = write_indices[i_n]
            read_ok = r_page >= 0
            write_ok = w_page >= 0

            col = i_v * BV + CPL * lane

            # ---- phase 0: start the state read stream (async, in smem) ----
            # Each thread copies its 16 x CPL elements as one vector per row;
            # the DMA runs behind the conv/GEMV front and each thread re-reads
            # exactly the slots it copied, so cp.async.wait_all alone orders
            # the reads.
            if read_ok:
                for j in cutlass.range_constexpr(ROWS):
                    row = warp * ROWS + j
                    _cp_async_shared_global(
                        s_h.iterator + (row * BV + CPL * lane),
                        h_pool.iterator
                        + cute.crd2idx((r_page, i_hv, row, col), h_pool.layout),
                        size=4 * CPL,
                    )

            # ---- phase 1 (single straight-line front, one barrier) ----
            # Everything before the recurrence — conv(+SiLU), the L2-norm
            # partials, the f_b gate GEMV, and the decay transcendentals —
            # runs back to back with no intervening barrier, so the conv,
            # w_fb, and f_a loads are all in flight together and overlap the
            # GEMV/SiLU arithmetic. s_q/s_k hold the *raw* SiLU outputs; the
            # L2 normalizers (and the attention scale) distribute over the
            # recurrence dots and are folded in after the barrier:
            #   t = <h, k_raw> * k_inv,  h += (v_new * k_inv) * k_raw,
            #   o = <h, q_raw> * q_inv_scaled
            # which removes the second front barrier entirely.
            #
            # Threads [0, K) own q feature `tid`; threads [K, 2K) own k
            # feature `tid - K`. The first CPL warps additionally convolve
            # this block's BV-column v-window.
            is_q = tid < K
            f_local = Int32(tid)
            if not is_q:
                f_local = Int32(tid - K)
            feat = i_hv * K + f_local
            if not is_q:
                feat = P + i_hv * K + f_local

            # Producer-independent work first: conv taps and both conv
            # windows (prior state), folded straight into partial conv
            # accumulators so only a few scalars stay live across the PDL
            # wait (full tap fragments would cost occupancy at large batch).
            w_tap = _load_bf16x4_as_f32(conv_w.iterator + feat * 4)
            st0 = Float32(0.0)
            st1 = Float32(0.0)
            st2 = Float32(0.0)
            if read_ok:
                st0 = Float32(conv_pool[r_page, feat, 0])
                st1 = Float32(conv_pool[r_page, feat, 1])
                st2 = Float32(conv_pool[r_page, feat, 2])
            acc = st0 * w_tap[0] + st1 * w_tap[1] + st2 * w_tap[2]
            w3 = w_tap[3]
            # warp % CPL keeps the address in range for the warps that do
            # not own a v-column (their loads are dead duplicates).
            vfeat = 2 * P + i_hv * V + i_v * BV + (warp % CPL) * 32 + lane
            vw_tap = _load_bf16x4_as_f32(conv_w.iterator + vfeat * 4)
            vs0 = Float32(0.0)
            vs1_keep = Float32(0.0)
            vs2_keep = Float32(0.0)
            if warp < CPL and read_ok:
                vs0 = Float32(conv_pool[r_page, vfeat, 0])
                vs1_keep = Float32(conv_pool[r_page, vfeat, 1])
                vs2_keep = Float32(conv_pool[r_page, vfeat, 2])
            vacc = vs0 * vw_tap[0] + vs1_keep * vw_tap[1] + vs2_keep * vw_tap[2]
            vw3 = vw_tap[3]

            if cutlass.const_expr(self.enable_pdl):
                # First touch of producer-written tensors happens below.
                cute.arch.griddepcontrol_wait()

            x_raw = Float32(qkv_raw[bos, feat])
            beta_raw = Float32(beta[bos, i_hv])
            acc += x_raw * w3
            y = _silu(acc)

            # L2-norm partials: warps 0..3 hold q features, 4..7 hold k.
            # Raw SiLU outputs go straight to smem; normalization is folded
            # into the recurrence dots after the barrier.
            sq_sum = cute.arch.warp_reduction_sum(y * y)
            if lane == 0:
                s_norm[warp] = sq_sum
            if is_q:
                s_q[f_local] = y
            else:
                s_k[f_local] = y

            # v conv for this block's columns (first CPL warps, one column
            # per lane). The shifted window values are kept in registers for
            # the deferred tail store.
            xv_keep = Float32(0.0)
            if warp < CPL:
                xv_keep = Float32(qkv_raw[bos, vfeat])
                vacc += xv_keep * vw3
                s_v[warp * 32 + lane] = _silu(vacc)

            # g[c] = w_fb[c, :] . f_a  for this head's K gate channels. Each
            # warp reduces 16 rows: one 8-byte vector load per lane per row
            # (coalesced 256B rows) against a register-cached f_a fragment
            # read straight from global memory (no smem staging, so the GEMV
            # needs no barrier and overlaps the conv phase above).
            E = D_FA // 32  # elements per lane per row (multiple of 4)
            fa_frag = cute.make_rmem_tensor(E, Float32)
            for e in cutlass.range_constexpr(E):
                fa_frag[e] = Float32(f_a[bos, lane * E + e])
            # Lane j keeps row j's dot so the bias + safe gate + exp run in
            # parallel on 16 lanes after the loop (warp-local rows: warp w's
            # GEMV rows equal its recurrence rows) — no barrier, and no
            # serialized per-row divergent transcendental tails.
            g_mine = Float32(0.0)
            for j in cutlass.range_constexpr(ROWS):
                row = warp * ROWS + j
                c = i_hv * K + row
                partial = Float32(0.0)
                for ch in cutlass.range_constexpr(E // 4):
                    wv = _load_bf16x4_as_f32(
                        w_fb.iterator + c * D_FA + lane * E + ch * 4
                    )
                    for e in cutlass.range_constexpr(4):
                        partial += wv[e] * fa_frag[ch * 4 + e]
                g_row = cute.arch.warp_reduction_sum(partial)
                if lane == j:
                    g_mine = g_row

            # Per-channel decay: g -> exp(gate(g)), one channel per lane.
            exp_a = cute.math.exp(Float32(A_log[i_hv]), fastmath=True)
            if lane < ROWS:
                row = warp * ROWS + lane
                g_ch = g_mine + Float32(dt_bias[i_hv * K + row])
                gk = Float32(0.0)
                if cutlass.const_expr(self.use_lower_bound):
                    gk = Float32(self.lower_bound) * _sigmoid(exp_a * g_ch)
                else:
                    sp = g_ch
                    if g_ch < 20.0:
                        sp = cute.math.log(
                            1.0 + cute.math.exp(g_ch, fastmath=True),
                            fastmath=True,
                        )
                    gk = -exp_a * sp
                s_dec[row] = cute.math.exp(gk, fastmath=True)

            cute.arch.barrier()

            # Under nv > 1, in-place conv paging means the sibling blocks
            # read the same q/k conv-window slots this cluster shifts in the
            # tail, so their stores must be held until every sibling has
            # read. The relaxed arrive is safe *here*, right after bar.sync:
            # the block barrier is a hard scheduling fence, and by this point
            # every warp's conv loads were consumed by the SiLU math (in-order
            # issue), so arrival implies the reads are complete. The matching
            # wait sits at the kernel tail before the deferred window stores,
            # hiding the barrier behind the whole recurrence. (A release
            # arrive would additionally drain the in-flight cp.async state
            # stream and cost ~0.6us; the Triton megafusion instead relies on
            # same-wave residency, which stops holding beyond one wave.)
            if cutlass.const_expr(self.nv > 1):
                cute.arch.cluster_arrive_relaxed()

            # ---- phase 2: delta-rule recurrence over the state slab ----
            # Folded L2 normalizers (see the phase-1 note).
            q_inv = (
                1.0
                / cute.math.sqrt(s_norm[0] + s_norm[1] + s_norm[2] + s_norm[3] + 1e-6)
                * self.scale
            )
            k_inv = 1.0 / cute.math.sqrt(
                s_norm[4] + s_norm[5] + s_norm[6] + s_norm[7] + 1e-6
            )

            # Warp `w` streams K-rows [16w, 16w+16); lane owns CPL columns.
            # Pass 1: decay each prefetched row into registers, accumulate
            # the per-column removal dots t = <h_decayed[:, col], k>.
            h_reg = cute.make_rmem_tensor((ROWS, CPL), Float32)
            t_acc = cute.make_rmem_tensor(CPL, Float32)
            for c in cutlass.range_constexpr(CPL):
                t_acc[c] = Float32(0.0)
            if read_ok:
                _cp_async_wait_all()
                for j in cutlass.range_constexpr(ROWS):
                    row = warp * ROWS + j
                    dec = s_dec[row]
                    kv = s_k[row]
                    for c in cutlass.range_constexpr(CPL):
                        h_val = s_h[row, CPL * lane + c] * dec
                        h_reg[j, c] = h_val
                        t_acc[c] += h_val * kv
            else:
                for j in cutlass.range_constexpr(ROWS):
                    for c in cutlass.range_constexpr(CPL):
                        h_reg[j, c] = Float32(0.0)
            for c in cutlass.range_constexpr(CPL):
                s_red_t[warp, CPL * lane + c] = t_acc[c]

            cute.arch.barrier()

            b_sig = _sigmoid(beta_raw)
            v_new_k = cute.make_rmem_tensor(CPL, Float32)
            for c in cutlass.range_constexpr(CPL):
                t_col = Float32(0.0)
                for w in cutlass.range_constexpr(_NUM_WARPS):
                    t_col += s_red_t[w, CPL * lane + c]
                v_new_k[c] = (s_v[CPL * lane + c] - t_col * k_inv) * b_sig * k_inv

            # Pass 2: rank-1 update from registers, stream rows back (one
            # vector store per row), and accumulate the per-column output
            # dots o = <h_new[:, col], q>.
            o_acc = cute.make_rmem_tensor(CPL, Float32)
            for c in cutlass.range_constexpr(CPL):
                o_acc[c] = Float32(0.0)
            h_out = cute.make_rmem_tensor(CPL, Float32)
            for j in cutlass.range_constexpr(ROWS):
                row = warp * ROWS + j
                kv = s_k[row]
                qv = s_q[row]
                for c in cutlass.range_constexpr(CPL):
                    h_val = h_reg[j, c] + v_new_k[c] * kv
                    h_out[c] = h_val
                    o_acc[c] += h_val * qv
                if write_ok:
                    dst = h_pool.iterator + cute.crd2idx(
                        (w_page, i_hv, row, col), h_pool.layout
                    )
                    if cutlass.const_expr(CPL == 1):
                        h_pool[w_page, i_hv, row, col] = h_out[0]
                    elif cutlass.const_expr(CPL == 2):
                        _store_global_f32x2(dst, h_out[0], h_out[1])
                    else:
                        _store_global_f32x4(dst, h_out[0], h_out[1], h_out[2], h_out[3])
            for c in cutlass.range_constexpr(CPL):
                s_red_o[warp, CPL * lane + c] = o_acc[c]

            cute.arch.barrier()

            # First CPL warps combine and store this block's BV outputs.
            if warp < CPL:
                oc = warp * 32 + lane
                o_col = Float32(0.0)
                for w in cutlass.range_constexpr(_NUM_WARPS):
                    o_col += s_red_o[w, oc]
                o[bos, i_hv, i_v * BV + oc] = BFloat16(o_col * q_inv)

            # ---- tail: deferred conv-window shift stores ----
            # By now every sibling block has long arrived, so the wait is
            # free; the q/k window (bx == 0 only) and this block's v window
            # are stored once per page.
            if cutlass.const_expr(self.nv > 1):
                cute.arch.cluster_wait()
            if write_ok:
                if i_v == 0:
                    conv_pool[w_page, feat, 0] = BFloat16(st1)
                    conv_pool[w_page, feat, 1] = BFloat16(st2)
                    conv_pool[w_page, feat, 2] = BFloat16(x_raw)
                if warp < CPL:
                    vfeat_w = 2 * P + i_hv * V + i_v * BV + warp * 32 + lane
                    conv_pool[w_page, vfeat_w, 0] = BFloat16(vs1_keep)
                    conv_pool[w_page, vfeat_w, 1] = BFloat16(vs2_keep)
                    conv_pool[w_page, vfeat_w, 2] = BFloat16(xv_keep)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


@cache
def _compile_kda_fused_decode(
    num_heads: int,
    head_dim: int,
    d_fa: int,
    scale: float,
    lower_bound: float | None,
    has_cu_seqlens: bool,
    idx_dtype,
    nv: int,
    enable_pdl: bool,
):
    """Compile (once per config) the fused decode kernel with symbolic shapes.

    Token count, page count, and every leading stride are symbolic, so one
    CUBIN serves all batch sizes (within one ``nv`` band) and possibly
    page-strided pool views.
    """
    p = num_heads * head_dim
    tokens = cute.sym_int()
    qkv_raw = make_fake_tensor(BFloat16, (tokens, 3 * p))
    conv_w = make_fake_tensor(BFloat16, (3 * p, 4))
    conv_pool = make_fake_tensor(BFloat16, (cute.sym_int(), 3 * p, 3))
    f_a = make_fake_tensor(BFloat16, (tokens, d_fa))
    w_fb = make_fake_tensor(BFloat16, (p, d_fa))
    beta = make_fake_tensor(BFloat16, (tokens, num_heads))
    a_log = make_fake_tensor(Float32, (num_heads,))
    dt_bias = make_fake_tensor(Float32, (p,))
    out = make_fake_tensor(BFloat16, (tokens, num_heads, head_dim))
    h_pool = make_fake_tensor(Float32, (cute.sym_int(), num_heads, head_dim, head_dim))
    n_seq = cute.sym_int()
    read_indices = make_fake_tensor(idx_dtype, (n_seq,))
    write_indices = make_fake_tensor(idx_dtype, (n_seq,))
    # When dense-indexed the launcher passes read_indices as a never-read
    # placeholder, so the fake dtype must follow the index dtype.
    cu_seqlens = make_fake_tensor(
        Int32 if has_cu_seqlens else idx_dtype, (cute.sym_int(),)
    )
    kernel = KdaFusedDecodeKernel(
        num_heads,
        head_dim,
        d_fa,
        scale,
        lower_bound,
        has_cu_seqlens,
        nv,
        enable_pdl,
    )
    return cute.compile(
        kernel,
        qkv_raw,
        conv_w,
        conv_pool,
        f_a,
        w_fb,
        beta,
        a_log,
        dt_bias,
        out,
        h_pool,
        read_indices,
        write_indices,
        cu_seqlens,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def cutedsl_fused_recurrent_kda_megafuse(
    qkv_raw: torch.Tensor,
    conv_w: torch.Tensor,
    conv_pool: torch.Tensor,
    f_a: torch.Tensor,
    w_fb: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    h_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    lower_bound: float | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Single-step KDA decode with conv1d(+SiLU) and the f_b GEMV fused in.

    CuTe DSL equivalent of
    ``tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent.
    fused_recurrent_kda_megafuse`` — same signature, same semantics, one
    kernel launch, CUDA-graph capturable.

    Args:
        qkv_raw: ``[T, 3*P]`` pre-conv packed q|k|v (token-strided slice ok),
            bf16.
        conv_w: ``[3*P, 4]`` (or any contiguous view of ``3*P*4``) fused conv
            kernel bank, bf16.
        conv_pool: ``[pages, 3*P, 3]`` bf16 conv state (updated in place,
            dual-index).
        f_a: ``[T, D]`` low-rank decay-gate input; w_fb: ``[P, D]`` up weight
            (both bf16, ``w_fb`` contiguous).
        beta: ``[T, HV]`` raw logits (sigmoid in-kernel), bf16.
        A_log: ``[HV]`` fp32 per-head log decay.
        dt_bias: fp32 gate bias with ``P`` elements (required).
        h_pool: ``[pages, HV, K, V]`` fp32 recurrent state pool (updated in
            place at ``write_indices``); the V dim must be contiguous.
        read_indices / write_indices: ``[N]`` int32/int64 page ids; negative
            ids read zeros / skip the store.
        num_heads / head_dim: per-rank head geometry (``P = num_heads *
            head_dim``; only ``head_dim == 128`` is supported).
        scale: attention scale; defaults to ``head_dim ** -0.5``.
        cu_seqlens: optional ``[N+1]`` int32 token offsets (decode: one token
            per sequence; zero-length sequences are skipped).
        lower_bound: safe-gate lower bound; ``None`` selects the softplus
            gate.
        enable_pdl: chain after the same-stream producer via programmatic
            dependent launch.

    Returns:
        o: ``[T, HV, V]`` attention output (bf16).
    """
    hv = num_heads
    k = head_dim
    p = hv * k
    t = qkv_raw.shape[0]
    d_fa = f_a.shape[-1]
    if scale is None:
        scale = k**-0.5
    if dt_bias is None:
        raise ValueError("cutedsl KDA fused decode requires dt_bias")
    assert qkv_raw.stride(-1) == 1 and qkv_raw.shape[-1] == 3 * p
    assert f_a.stride(-1) == 1 and beta.stride(-1) == 1
    assert w_fb.is_contiguous() and w_fb.shape == (p, d_fa)
    assert h_pool.stride(-1) == 1 and h_pool.shape[1:] == (hv, k, k)
    assert conv_pool.stride(-1) == 1 and conv_pool.shape[1:] == (3 * p, 3)
    conv_w = conv_w.reshape(3 * p, 4)
    dt_bias = dt_bias.reshape(p)
    A_log = A_log.reshape(hv)
    if read_indices.dtype not in _IDX_DTYPES:
        raise ValueError(f"unsupported index dtype {read_indices.dtype}")
    assert write_indices.dtype == read_indices.dtype

    n = t if cu_seqlens is None else cu_seqlens.numel() - 1
    assert read_indices.numel() == n and write_indices.numel() == n

    out = torch.empty(t, hv, k, dtype=qkv_raw.dtype, device=qkv_raw.device)
    kernel = _compile_kda_fused_decode(
        hv,
        k,
        d_fa,
        float(scale),
        None if lower_bound is None else float(lower_bound),
        cu_seqlens is not None,
        _IDX_DTYPES[read_indices.dtype],
        _pick_nv(n, hv),
        bool(enable_pdl),
    )
    kernel(
        qkv_raw,
        conv_w,
        conv_pool,
        f_a,
        w_fb,
        beta,
        A_log,
        dt_bias,
        out,
        h_pool,
        read_indices,
        write_indices,
        # Placeholder when dense-indexed: the compiled kernel never reads it.
        cu_seqlens if cu_seqlens is not None else read_indices,
    )
    return out


def is_cutedsl_kda_fused_decode_supported() -> bool:
    """Whether this platform can run the CuTe DSL fused KDA decode kernel."""
    try:
        import cutlass as _cutlass  # noqa: F401
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major == 10
