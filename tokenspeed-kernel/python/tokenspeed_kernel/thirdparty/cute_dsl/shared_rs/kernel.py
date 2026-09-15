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

"""Resident-grid hidden ReduceScatter; no producer or consumer fusion."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def rank_barrier(
    signals: Int64, rank: Int32, block: Int32, peer: Int32, *, loc=None, ip=None
):
    # Exactly the existing 0->1 release / 1->0 acquire handshake. Each CTA
    # owns eight slots; grids must be rank-identical and simultaneously resident.
    llvm.inline_asm(
        None,
        [x.ir_value(loc=loc, ip=ip) for x in (signals, rank, block, peer)],
        """{
        .reg .u64 remote, local, addr, offset, send, wait;
        .reg .u32 slot, old;
        .reg .pred done;
        mul.wide.u32 offset, $3, 8;
        add.u64 addr, $0, offset;
        ld.global.u64 remote, [addr];
        mul.wide.u32 offset, $1, 8;
        add.u64 addr, $0, offset;
        ld.global.u64 local, [addr];
        mad.lo.u32 slot, $2, 8, $1;
        mul.wide.u32 offset, slot, 4;
        add.u64 send, remote, offset;
        mad.lo.u32 slot, $2, 8, $3;
        mul.wide.u32 offset, slot, 4;
        add.u64 wait, local, offset;
        RS_SEND:
        atom.global.release.sys.cas.b32 old, [send], 0, 1;
        setp.eq.u32 done, old, 0;
        @!done bra RS_SEND;
        RS_WAIT:
        atom.global.acquire.sys.cas.b32 old, [wait], 1, 0;
        setp.eq.u32 done, old, 1;
        @!done bra RS_WAIT;
        }""",
        "l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def alias_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.alias;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def transfer_vectors(sources, outputs, predicates, *, loc=None, ip=None):
    # Keep all independent reduce-loads before their stores in generated PTX.
    # No numerical conversion: the four words preserve the hardware BF16 result.
    count = len(sources)
    values = []
    code = ["{"]
    for i in range(count):
        values.extend([sources[i], outputs[i], Int32(predicates[i])])
        code.extend(
            [
                f".reg .b32 v{i}_<4>;",
                f".reg .pred p{i};",
                f"setp.ne.s32 p{i}, ${3*i+2}, 0;",
            ]
        )
    for i in range(count):
        words = ", ".join(f"v{i}_{j}" for j in range(4))
        code.append(
            f"@p{i} multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
            f"{{{words}}}, [${3*i}];"
        )
    for i in range(count):
        words = ", ".join(f"v{i}_{j}" for j in range(4))
        code.append(f"@p{i} st.relaxed.sys.global.v4.b32 [${3*i+1}], {{{words}}};")
    code.append("}")
    llvm.inline_asm(
        None,
        [v.ir_value(loc=loc, ip=ip) for v in values],
        "\n".join(code),
        ",".join(["l,l,r"] * count),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class SharedRsKernel:
    def __init__(
        self, ctas: int, threads: int, vectors: int, row_aligned: bool, rank: int
    ):
        self.ctas = ctas
        self.threads = threads
        self.vectors = vectors
        self.row_aligned = row_aligned
        self.rank = rank

    @cute.jit
    def __call__(
        self,
        source: Int64,
        output: Int64,
        signals: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(source, output, signals, m).launch(
            grid=(self.ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, source: Int64, output: Int64, signals: Int64, m: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        if tid < 8:
            rank_barrier(signals, Int32(self.rank), bid, tid)
        cute.arch.sync_threads()
        alias_fence()

        if cutlass.const_expr(self.row_aligned):
            base = bid * (self.threads // 128) * self.vectors + tid // 128
            stop = m
            stride = self.ctas * (self.threads // 128) * self.vectors
        else:
            base = bid * self.threads * self.vectors + tid
            stop = m * 112
            stride = self.ctas * self.threads * self.vectors

        while base < stop:
            sources = ()
            outputs = ()
            predicates = ()
            for u in cutlass.range_constexpr(self.vectors):
                if cutlass.const_expr(self.row_aligned):
                    row = base + u * (self.threads // 128)
                    col = tid % 128
                    valid = (row < m) & (col < 112)
                else:
                    chunk = base + u * self.threads
                    row = chunk // 112
                    col = chunk % 112
                    valid = chunk < m * 112
                sources += (source + Int64(row * 896 + self.rank * 112 + col) * 16,)
                outputs += (output + Int64(row * 112 + col) * 16,)
                predicates += (Int32(valid),)
            transfer_vectors(sources, outputs, predicates)
            base += stride
        cute.arch.sync_threads()
        if tid < 8:
            rank_barrier(signals, Int32(self.rank), bid, tid)
