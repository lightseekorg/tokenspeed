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

; Triton's HIP backend does not hash library contents. _scheduling.py supplies this
; file's SHA256 as the SCHED_LIBRARY_HASH constexpr to invalidate compiled
; kernels automatically after edits. Keep that dependency; the filename and
; symbol need no version bumps. Restart the process after editing this file,
; since its digest is cached for the process lifetime.

declare void @llvm.amdgcn.sched.barrier(i32 immarg)

define i32 @__tokenspeed_sched_barrier0() alwaysinline {
entry:
  call void @llvm.amdgcn.sched.barrier(i32 0)
  ret i32 0
}

; Through the intrinsic (not inline asm) the backend inserts the wait states a
; preceding VALU write of the source VGPR needs, and folds the move away when
; the value is already in an SGPR.
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)

define i32 @__tokenspeed_readfirstlane_i32(i32 %x) alwaysinline {
entry:
  %r = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %x)
  ret i32 %r
}

; sched_group_barrier(mask, size, 0): the next `size` instructions of class
; `mask` in the region above form one group (IGroupLP). The operands are
; immediates, so there is one function per (mask, size) pair in use.
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg)

; One MFMA.
define i32 @__tokenspeed_sgb_8_1() alwaysinline {
entry:
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  ret i32 0
}

; Three VMEM instructions.
define i32 @__tokenspeed_sgb_16_3() alwaysinline {
entry:
  call void @llvm.amdgcn.sched.group.barrier(i32 16, i32 3, i32 0)
  ret i32 0
}

; Two transcendental instructions.
define i32 @__tokenspeed_sgb_1024_2() alwaysinline {
entry:
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 2, i32 0)
  ret i32 0
}
