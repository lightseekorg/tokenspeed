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

; Release drain: wait until this wave's vector memory operations completed
; (s_waitcnt vmcnt(0); 3952 = 0x0F70 leaves expcnt and lgkmcnt untouched), so
; its stores reached the L2 before a later arrival or flag store. The
; workgroup-scope release fence keeps earlier stores above the wait in the IR
; (llvm.amdgcn.s.waitcnt is declared not to touch memory, so on its own a
; store could move across it) and adds no cache maintenance on gfx950. An
; agent-scope release would also write back the whole L2 (buffer_wbl2).
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)

define i32 @__tokenspeed_sync_release_drain() alwaysinline {
entry:
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.waitcnt(i32 3952)
  ret i32 0
}
