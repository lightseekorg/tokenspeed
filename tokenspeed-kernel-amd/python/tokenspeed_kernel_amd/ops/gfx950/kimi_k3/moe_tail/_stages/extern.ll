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

; The MoE tail's device operations that Gluon does not expose, with no inline
; machine instructions. Every function is named __tokenspeed_moe_tail_* so
; that Triton links this file only into kernels calling one (extern.py).
declare i32 @llvm.amdgcn.readlane(i32, i32)
declare i32 @llvm.amdgcn.writelane(i32, i32, i32)
declare i64 @llvm.cttz.i64(i64, i1 immarg)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)

define i32 @__tokenspeed_moe_tail_readlane(i32 %value, i32 %lane) alwaysinline {
  %result = call i32 @llvm.amdgcn.readlane(i32 %value, i32 %lane)
  ret i32 %result
}

define i32 @__tokenspeed_moe_tail_writelane(i32 %value, i32 %lane, i32 %vector) alwaysinline {
  %result = call i32 @llvm.amdgcn.writelane(i32 %value, i32 %lane, i32 %vector)
  ret i32 %result
}

define i32 @__tokenspeed_moe_tail_cttz(i64 %value) alwaysinline {
  %wide = call i64 @llvm.cttz.i64(i64 %value, i1 false)
  %result = trunc i64 %wide to i32
  ret i32 %result
}

; A peer table entry is the same in every active lane of a wave: read it from
; the first lane, into scalar registers.
define i64 @__tokenspeed_moe_tail_peer_uniform(i64 %bits) alwaysinline {
  %lo = trunc i64 %bits to i32
  %upper = lshr i64 %bits, 32
  %hi = trunc i64 %upper to i32
  %slo = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %lo)
  %shi = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %hi)
  %lo64 = zext i32 %slo to i64
  %hi64 = zext i32 %shi to i64
  %shifted = shl i64 %hi64, 32
  %result = or i64 %lo64, %shifted
  ret i64 %result
}
