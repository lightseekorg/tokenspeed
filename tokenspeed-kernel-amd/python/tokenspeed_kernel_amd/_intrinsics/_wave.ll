; MIT License
;
; Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
;
; Permission is hereby granted, free of charge, to any person obtaining a copy
; of this software and associated documentation files (the "Software"), to deal
; in the Software without restriction, including without limitation the rights
; to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
; copies of the Software, and to permit persons to whom the Software is
; furnished to do so, subject to the following conditions:
;
; The above copyright notice and this permission notice shall be included in all
; copies or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
; OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
; SOFTWARE.

; Wave-level operations: values shared across the lanes of a wave. Every
; function is named __tokenspeed_wave_* so that Triton links this file only
; into kernels calling one (_extern.py).

; Through the intrinsic (not inline asm) the backend inserts the wait states a
; preceding VALU write of the source VGPR needs, and folds the move away when
; the value is already in an SGPR.
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)

define i32 @__tokenspeed_wave_readfirstlane_i32(i32 %x) alwaysinline {
entry:
  %r = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %x)
  ret i32 %r
}

; 1 when %a == %b (ordered, as v_cmp_eq_f32) in every active lane of the wave,
; else 0; the same in every lane, in an SGPR: one compare into a lane mask
; (llvm.amdgcn.ballot) tested against the active lanes (EXEC).
declare i64 @llvm.amdgcn.ballot.i64(i1)

define i32 @__tokenspeed_wave_all_equal_f32(float %a, float %b) alwaysinline {
entry:
  %eq = fcmp oeq float %a, %b
  %set = call i64 @llvm.amdgcn.ballot.i64(i1 %eq)
  %active = call i64 @llvm.amdgcn.ballot.i64(i1 true)
  %all = icmp eq i64 %set, %active
  %r = zext i1 %all to i32
  ret i32 %r
}

; A ballot bit is set for each active lane whose int32 predicate is nonzero.
define i64 @__tokenspeed_wave_ballot_i64(i32 %predicate) alwaysinline {
entry:
  %p = icmp ne i32 %predicate, 0
  %mask = call i64 @llvm.amdgcn.ballot.i64(i1 %p)
  ret i64 %mask
}

; Count set bits below this lane in a wave64 mask.
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)

define i32 @__tokenspeed_wave_mbcnt(i64 %mask) alwaysinline {
entry:
  %lo = trunc i64 %mask to i32
  %shifted = lshr i64 %mask, 32
  %hi = trunc i64 %shifted to i32
  %low_count = call i32 @llvm.amdgcn.mbcnt.lo(i32 %lo, i32 0)
  %count = call i32 @llvm.amdgcn.mbcnt.hi(i32 %hi, i32 %low_count)
  ret i32 %count
}

; Keep population count visible to LLVM so a uniform mask uses s_bcnt1.
declare i64 @llvm.ctpop.i64(i64)

define i32 @__tokenspeed_wave_popcount_i64(i64 %mask) alwaysinline {
entry:
  %count = call i64 @llvm.ctpop.i64(i64 %mask)
  %r = trunc i64 %count to i32
  ret i32 %r
}
