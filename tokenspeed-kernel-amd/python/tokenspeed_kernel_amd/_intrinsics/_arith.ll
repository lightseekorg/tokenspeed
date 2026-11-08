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

; Floating-point arithmetic controls. Every function is named
; __tokenspeed_arith_* so that Triton links this file only into kernels
; calling one (_extern.py).

; An f32 value LLVM must treat as new: llvm.arithmetic.fence emits no
; instruction, but no optimization looks through it, so arithmetic on the
; result is neither merged with the same arithmetic on the input (CSE) nor
; reassociated across it.
declare float @llvm.arithmetic.fence.f32(float)

define float @__tokenspeed_arith_fence_f32(float %x) alwaysinline {
entry:
  %r = call float @llvm.arithmetic.fence.f32(float %x)
  ret float %r
}
