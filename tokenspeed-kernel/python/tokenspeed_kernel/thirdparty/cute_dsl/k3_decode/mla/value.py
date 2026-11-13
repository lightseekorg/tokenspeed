# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3.1 specialized decode (cutedsl/mla/value.py):
# MlaValue, without the batched-GEMM switch for many local heads. Modified:
# - Row counts are runtime values (one compile per head count and gate kind),
#   and every row is projected: no per-request lengths zero padded rows, as the
#   unfused bmm and sigmoid_mul do not.
# - The gate is optional, and its sigmoid is the Triton one (ex2.approx of
#   -g log2(e), then div.full of 1 by 1 plus it), so a gated row matches
#   sigmoid_mul on the BF16 projection; only the projection's summation order
#   differs from the unfused bmm.

"""MLA value projection (W_UV) and optional sigmoid gate of the BF16 attention latent."""

from functools import cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import architecture, dynamic_view, fake, to_cute
from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    copy_g2s,
    launch,
    load_words,
    mma_into,
    store_words,
    unpack_bf16,
    zeros,
)

LATENT, VALUE = 512, 128


@asm(Float32, pure=True)
def triton_sigmoid(value):
    """``tl.sigmoid``: 1 / (1 + exp(-x)) with ex2.approx and div.full."""
    code = "{ .reg .f32 t; mul.f32 t,$1,0fBFB8AA3B; ex2.approx.ftz.f32 t,t; "
    code += "add.f32 t,t,0f3F800000; div.full.f32 $0,0f3F800000,t; }"
    return code, "=f,f", (value,)


class MlaValue:
    """W_UV and the sigmoid gate, [B, H, 512] BF16 latent -> [B, H*128]. One
    128-thread CTA per (32 outputs, head, 16 rows); warp w owns outputs
    8w..8w+7 over the whole latent (BF16 MMA, FP32 accumulation in two chains).
    W_UV must have a contiguous latent axis (the absorbed kv_b_proj view)."""

    def __init__(self, heads: int, gated: bool):
        self.heads, self.gated = heads, gated

    @cute.jit
    def __call__(self, latent, w_uv, gate, output, rows: Int32, stream):
        grid = (4, self.heads, (rows + 15) // 16)
        launch(self.kernel, (latent, w_uv, gate, output, rows, stream), grid, 128)

    @cute.kernel
    def kernel(self, args):
        latent, w_uv, gate, output, batch = args
        rows = alloc(cutlass.utils.SmemAllocator(), BFloat16, (16, 544), 128)
        tid, _, _ = cute.arch.thread_idx()
        chunk, head, tile = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0, v = tile * 16, chunk * 32 + warp * 8 + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        for c in range(tid, 16 * 64, 128):
            r, j = c // 64, c % 64
            if t0 + r < batch:
                src = at(latent, t0 + r, head, j * 8)
                copy_g2s(rows.iterator + r * 544 + j * 8, src)
        cute.arch.cp_async_commit_group()
        col = head * VALUE + chunk * 32 + warp * 8 + 2 * quarter
        gates = []
        if cutlass.const_expr(self.gated):
            for r in cutlass.range_constexpr(2):
                word = Uint32(0)
                if t0 + group + 8 * r < batch:
                    word = load_words(
                        at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                    )[0]
                gates.append(unpack_bf16(word))
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    rows.iterator + r * 544 + 32 * c + 8 * quarter, count=4, shared=True
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            t = t0 + group + 8 * r
            if t < batch:
                lo, hi = (Float32(BFloat16(acc[2 * r + e])) for e in range(2))
                if cutlass.const_expr(self.gated):
                    lo = lo * triton_sigmoid(gates[r][0])
                    hi = hi * triton_sigmoid(gates[r][1])
                store_words(at(output, t, col), (bf16x2(lo, hi),))


@cache
def compile_value(heads: int, gated: bool, arch: str):
    """MlaValue for ``heads`` local heads over runtime row counts and strides."""

    # Declared as dynamic_fake does: dynamic_view's runtime strides only match this form.
    def rows(rank, align):
        shape = tuple(cute.sym_int() for _ in range(rank))
        return fake(
            shape, (*(cute.sym_int(64) for _ in range(rank - 1)), 1), align=align
        )

    w_uv = fake((heads, LATENT, VALUE), (LATENT * VALUE, 1, LATENT))
    return cute.compile(
        MlaValue(heads, gated),
        rows(3, 16),
        w_uv,
        rows(2, 4),
        rows(2, 4),
        Int32(0),
        make_fake_stream(),
        options=f"--gpu-arch={arch}",
    )


def mla_value(latent, w_uv, gate, output, stream):
    """``output = bf16(latent @ w_uv)`` per head, times ``sigmoid(gate)`` when given.

    ``latent`` is ``[rows, heads, 512]`` with unit inner stride and 16-byte aligned
    rows, ``w_uv`` ``[heads, 512, 128]`` with the latent axis contiguous, ``gate``
    ``[rows, heads * 128]`` with unit inner stride and 4-byte aligned rows (or
    None), and ``output`` ``[rows, heads * 128]`` with unit inner stride; all BF16.
    """
    heads = w_uv.shape[0]
    code = compile_value(heads, gate is not None, architecture(latent.device))
    # An ungated launch passes the output for the gate, which it never reads.
    code(
        dynamic_view(latent),
        to_cute(w_uv),
        dynamic_view(output if gate is None else gate, 4),
        dynamic_view(output, 4),
        Int32(latent.shape[0]),
        stream,
    )
