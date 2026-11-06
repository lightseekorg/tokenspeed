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


; LLVM intrinsics of the gfx950 MoE stream kernel. Through the intrinsics (not
; inline asm) the backend sees each instruction and inserts the hazard wait
; states and counter waits around it. Triton links an external library only
; when a called symbol's name contains the library name, so every symbol
; contains "tokenspeed_stream_sync". Each function is alwaysinline and returns
; an i32 when it has no result, as Triton's extern calls require.

; The XCD running the wave: s_getreg_b32 hwreg(HW_REG_XCC_ID, 0, 4), encoded
; (size - 1) << 11 | offset << 6 | id with id 20.
declare i32 @llvm.amdgcn.s.getreg(i32 immarg)

define i32 @__tokenspeed_stream_sync_xcc_id() alwaysinline {
entry:
  %x = call i32 @llvm.amdgcn.s.getreg(i32 6164)
  ret i32 %x
}

; A queue error (s_trap 2 under the HSA ABI); never returns.
declare void @llvm.trap()

define i32 @__tokenspeed_stream_sync_trap() alwaysinline {
entry:
  call void @llvm.trap()
  ret i32 0
}

; Acquire for one wave: an IR ordering point (loads after it cannot move
; above it), with no instruction on gfx950. It ends a per-wave poll whose
; result only this wave uses, in place of a workgroup barrier.
define i32 @__tokenspeed_stream_sync_wave_acquire() alwaysinline {
entry:
  fence syncscope("wavefront") acquire
  ret i32 0
}
