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

# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SM107 MLA prefill specialization of the shared FMHA execution path.

Adapted from FlashInfer PR #8716, commit
6739dd5ea4f99a06be0fa24d22db0b6745c23ce5. TokenSpeed modifications:
reuse the existing ragged interface, scheduler, empty-KV epilogue and PDL;
retain FP32 softmax and direct BF16 stores. Optional FP16 softmax, MMA row
sums, Q TMEM residency, paged GQA and TMA output are outside this port.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_mla import fmha_helpers as fmha_utils
from tokenspeed_mla.fmha import BlackwellFusedMultiHeadAttentionForward


@dsl_user_op
def _causal_mask_32_regs_setp_selp(
    regs,
    last_valid_k: Int32,
    group_start: cutlass.Constexpr,
    *,
    loc=None,
    ip=None,
):
    """Apply a causal suffix mask with scalar ``setp.ge``/``selp`` pairs.

    Each output is tied to its input register, so the inline PTX requests one
    scalar 32-bit select instead of leaving the compiler to construct
    pair-valued SSA merges for the conditional assignments in ``apply_mask``,
    which lower poorly on some architectures.
    """
    reg_count = 32
    last_valid_operand = reg_count * 2
    reg_values = [Float32(regs[i]).ir_value(loc=loc, ip=ip) for i in range(reg_count)]
    blocks = []
    for block_start in range(0, reg_count, 4):
        compares = "\n".join(
            f"    setp.ge.s32 p{i - block_start}, "
            f"${last_valid_operand}, {group_start + i};"
            for i in range(block_start, block_start + 4)
        )
        selects = "\n".join(
            f"    selp.b32 ${i}, ${reg_count + i}, 0xff800000, p{i - block_start};"
            for i in range(block_start, block_start + 4)
        )
        blocks.append(f"{compares}\n{selects}")
    asm = f"""{{
    .reg .pred p<4>;
{chr(10).join(blocks)}
    }}"""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * reg_count),
        reg_values + [Int32(last_valid_k).ir_value(loc=loc, ip=ip)],
        asm,
        ",".join(["=f"] * reg_count + [str(i) for i in range(reg_count)] + ["r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(reg_count)
    )


class RubinFusedMultiHeadAttentionForward(BlackwellFusedMultiHeadAttentionForward):
    """Native SM107 FP8 MLA prefill with the shared ragged FMHA mainloop.

    Constructor arguments have the same meaning as the Blackwell kernel.
    Only E4M3 Q/K/V, BF16 output, (192, 128) head dimensions and
    residual or bottom-right causal masking are supported.
    """

    def __init__(
        self,
        qk_acc_dtype,
        pv_acc_dtype,
        mma_tiler,
        head_dim,
        is_persistent,
        mask_type,
        enable_ex2_emulation,
        enable_skip_correction,
        compute_capability,
    ):
        if compute_capability != (10, 7):
            raise ValueError("Rubin MLA prefill requires compute_capability=(10, 7)")
        if head_dim != (192, 128) or mma_tiler != (128, 128):
            raise ValueError("SM107 MLA prefill requires D192/D128 and M128/N128")
        if enable_ex2_emulation:
            raise ValueError("SM107 MLA prefill uses native exp2")
        if mask_type not in (
            fmha_utils.MaskEnum.RESIDUAL_MASK,
            fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE,
        ):
            raise ValueError(
                "SM107 MLA prefill requires residual or bottom-right causal masking"
            )
        super().__init__(
            qk_acc_dtype,
            pv_acc_dtype,
            mma_tiler,
            head_dim,
            is_persistent,
            mask_type,
            enable_ex2_emulation,
            enable_skip_correction,
            compute_capability,
        )
        # The shared path uses 512 columns; extra SM107 columns are only
        # needed for the optional Q residency / MMA row-sum modes.
        self.num_regs_correction = 88
        self.num_regs_other = 40
        self.correction_rescale_tile_size = 64
        self.correction_epilog_unroll = 1

    def _setup_attributes(self):
        if (self.q_dtype, self.k_dtype, self.v_dtype, self.o_dtype) != (
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
        ):
            raise TypeError("SM107 MLA prefill requires E4M3 Q/K/V and BF16 output")
        super()._setup_attributes()
        self.kv_stage = 6

    def _make_qk_tiled_mma(self, cta_group):
        # K=64 changes FP32 reduction grouping from Blackwell's K=32.
        # FP8 probability and BF16 output rounding need not be bitwise identical.
        return utils.sm107.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            (*self.qk_mma_tiler[:2], 64),
        )

    def _make_pv_tiled_mma(self, cta_group, p_major_mode, p_source):
        return utils.sm107.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            (*self.pv_mma_tiler[:2], 64),
            a_source=p_source,
        )

    @cute.jit
    def _apply_mask(
        self,
        acc_qk,
        index_qk,
        seqlen_q,
        seqlen_k,
        window_size_left,
        window_size_right,
    ):
        if cutlass.const_expr(window_size_left is not None):
            raise ValueError("SM107 MLA prefill does not support a left window")
        offset = Int32(0)
        if cutlass.const_expr(
            self.mask_type is fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
        ):
            offset = seqlen_k - seqlen_q
        index_q0, index_k0 = index_qk[0]
        if cutlass.const_expr(window_size_right is not None):
            valid_k = min(
                index_q0 + offset + window_size_right + 1 - index_k0,
                seqlen_k - index_k0,
            )
        else:
            valid_k = seqlen_k - index_k0
        if index_q0 >= seqlen_q:
            valid_k = Int32(0)
        last_valid_k = max(valid_k, Int32(0)) - Int32(1)
        for group_start in cutlass.range_constexpr(0, cute.size(acc_qk), 32):
            masked = _causal_mask_32_regs_setp_selp(
                tuple(acc_qk[group_start + j] for j in range(32)),
                last_valid_k,
                group_start,
            )
            for j in cutlass.range_constexpr(32):
                acc_qk[group_start + j] = masked[j]

    @cute.jit
    def _store_o_tile_direct(
        self,
        tTMrO: cute.Tensor,
        tTMEM_LOADdO_i: cute.Tensor,
        row_idx: Int32,
        seqlen_q: Int32,
    ) -> None:
        """Alignment-annotated direct store (enables wide st.global vectors).

        Every Rubin direct-store configuration benefits; packed NVFP4 on the
        physical-D80 layout *requires* it (sub-byte elements can only be
        stored as byte-aligned groups, never element-wise).
        """
        # Preserve the destination partition layout so the direct
        # register-to-global copy can lower to vector stores.
        tDMrO = cute.make_fragment_like(tTMEM_LOADdO_i, self.o_dtype)
        tDMrO.store(tTMrO.load().to(self.o_dtype))
        if row_idx < seqlen_q:
            # TMEM partitioning loses the alignment guaranteed by the
            # output base and correction-chunk stride. Reattach that
            # fact so autovec can select a wide global store. TokenSpeed
            # guarantees 16-byte alignment at its tensor boundary.
            direct_store_alignment = 16
            tDMrO_dst = cute.make_tensor(
                cute.make_ptr(
                    self.o_dtype,
                    tTMEM_LOADdO_i.iterator.toint(),
                    cutlass.AddressSpace.gmem,
                    assumed_align=direct_store_alignment,
                ),
                tTMEM_LOADdO_i.layout,
            )
            cute.autovec_copy(tDMrO, tDMrO_dst)
