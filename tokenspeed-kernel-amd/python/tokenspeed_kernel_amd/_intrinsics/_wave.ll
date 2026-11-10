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

; gfx9 wave64 only (gfx10 and later have no DPP row broadcasts).
; Inclusive int32 sum over the 64 lanes of a wave (every lane active): DPP
; row shifts by 1, 2, 4 and 8 sum within each row of 16 lanes (lanes shifted
; in from outside the row read 0), then row broadcasts 15 and 31 carry each
; row's total into the next rows (rows the row mask leaves out add 0). The
; backend inserts the wait states each DPP read of a just-written VGPR needs.
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)

define i32 @__tokenspeed_wave_inclusive_sum_i32(i32 %x) alwaysinline {
entry:
  %d1 = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %x, i32 273, i32 15, i32 15, i1 true)
  %s1 = add i32 %x, %d1
  %d2 = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %s1, i32 274, i32 15, i32 15, i1 true)
  %s2 = add i32 %s1, %d2
  %d4 = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %s2, i32 276, i32 15, i32 15, i1 true)
  %s4 = add i32 %s2, %d4
  %d8 = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %s4, i32 280, i32 15, i32 15, i1 true)
  %s8 = add i32 %s4, %d8
  %b15 = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %s8, i32 322, i32 10, i32 15, i1 false)
  %s15 = add i32 %s8, %b15
  %b31 = call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 %s15, i32 323, i32 12, i32 15, i1 false)
  %r = add i32 %s15, %b31
  ret i32 %r
}
