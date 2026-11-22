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

; Memory synchronization. Every function is named __tokenspeed_sync_* so that
; Triton links this file only into kernels calling one (_extern.py).

; Each wait is one hardware wait on this wave's vector memory counters. The
; backend sees it, so it merges the wait with the ones it inserts itself (for
; example the LDS wait before a barrier) and inserts no redundant ones after
; it. LLVM treats these intrinsics as reading and writing all memory, so
; loads and stores stay on their side of the wait.
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)
declare void @llvm.amdgcn.s.wait.storecnt(i16 immarg)
declare void @llvm.amdgcn.s.sleep(i32 immarg)
declare i64 @llvm.amdgcn.s.memrealtime()

; A deliberate memory fault: a volatile store to the null page. Unlike a
; noreturn trap intrinsic, the code after it stays reachable in the IR.
define i32 @__tokenspeed_sync_stop_on_fault() alwaysinline {
entry:
  store volatile i32 1, ptr addrspace(1) null, align 4
  ret i32 0
}

; The constant 100 MHz clock (s_memrealtime), ordered with memory accesses.
define i64 @__tokenspeed_sync_realtime_clock() alwaysinline {
entry:
  %ticks = call i64 @llvm.amdgcn.s.memrealtime()
  ret i64 %ticks
}

; s_sleep 1: pause this wave between the polls of a wait.
define i32 @__tokenspeed_sync_sleep_one() alwaysinline {
entry:
  call void @llvm.amdgcn.s.sleep(i32 1)
  ret i32 0
}

; gfx9: s_waitcnt vmcnt(0). 3952 = 0x0F70 leaves expcnt and lgkmcnt untouched.
define i32 @__tokenspeed_sync_wait_vmem() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3952)
  ret i32 0
}

; gfx9: the same wait after a workgroup-scope release fence. The intrinsic is
; declared not to access memory, so in the IR a store could still move below
; the wait; the fence keeps every earlier store above it and adds no
; instruction or cache maintenance on gfx950.
define i32 @__tokenspeed_sync_release_wait_vmem() alwaysinline {
entry:
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.waitcnt(i32 3952)
  ret i32 0
}

; gfx12 and gfx1250: s_wait_storecnt 0x0.
define i32 @__tokenspeed_sync_wait_stores() alwaysinline {
entry:
  call void @llvm.amdgcn.s.wait.storecnt(i16 0)
  ret i32 0
}
