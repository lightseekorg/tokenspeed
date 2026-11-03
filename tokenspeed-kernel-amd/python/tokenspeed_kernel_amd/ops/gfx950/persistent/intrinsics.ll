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

; LLVM intrinsics for persistent gfx950 kernels. Through the intrinsics (not
; inline asm) the backend sees each instruction: it inserts hazard wait states
; and the counter waits the results need. Every symbol contains the library
; name, "tokenspeed_persistent": Triton links an external library only when a
; called symbol's name contains it. Each function is alwaysinline and returns
; an i32 when it has no result, as Triton's extern calls require.

; The constant 100 MHz clock; the backend waits lgkmcnt only at the first use.
declare i64 @llvm.amdgcn.s.memrealtime()

define i64 @__tokenspeed_persistent_memrealtime() alwaysinline {
entry:
  %t = call i64 @llvm.amdgcn.s.memrealtime()
  ret i64 %t
}

; The XCD running the wave: s_getreg_b32 hwreg(HW_REG_XCC_ID, 0, 4), encoded
; (size - 1) << 11 | offset << 6 | id with id 20.
declare i32 @llvm.amdgcn.s.getreg(i32 immarg)

define i32 @__tokenspeed_persistent_xcc_id() alwaysinline {
entry:
  %x = call i32 @llvm.amdgcn.s.getreg(i32 6164)
  ret i32 %x
}

; A queue error (s_trap 2 under the HSA ABI); never returns.
declare void @llvm.trap()

define i32 @__tokenspeed_persistent_trap() alwaysinline {
entry:
  call void @llvm.trap()
  ret i32 0
}

; Release drain: wait until this wave's vector stores (and loads) completed,
; so they are in the L2 before a later flag store. The workgroup release fence
; keeps earlier stores above this point in the IR (llvm.amdgcn.s.waitcnt is
; declared not to touch memory, so on its own a store could move across it);
; on gfx950 the fence emits no wait. An agent-scope release fence would also
; write back the L2 (buffer_wbl2), which this must not do.
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)

define i32 @__tokenspeed_persistent_release_drain() alwaysinline {
entry:
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.waitcnt(i32 3952)
  ret i32 0
}

; Paced issue: wait until at most N of this wave's vector memory operations
; are outstanding (s_waitcnt vmcnt(N); 0x0F70 | N keeps expcnt and lgkmcnt
; untouched). The immediate is an operand of the instruction, so there is one
; function per N.
define i32 @__tokenspeed_persistent_vmcnt_1() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3953)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_2() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3954)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_3() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3955)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_4() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3956)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_5() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3957)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_6() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3958)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_7() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3959)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_8() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3960)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_9() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3961)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_10() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3962)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_11() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3963)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_12() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3964)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_13() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3965)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_14() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3966)
  ret i32 0
}

define i32 @__tokenspeed_persistent_vmcnt_15() alwaysinline {
entry:
  call void @llvm.amdgcn.s.waitcnt(i32 3967)
  ret i32 0
}

; A returning device-scope atomic add by lane 0 of the calling wave (relaxed);
; the other lanes do nothing and get undef. The address is an integer so that
; Triton's extern dispatch can key on it. The undef incoming value lets the
; result keep the atomic's own register, so the backend waits for it only
; where it is used (vmcnt(k) counting only later operations), not at the
; branch join. Callers pass a lane-indexed address (lane 0 -> the word): a
; uniform address goes through the backend's atomic optimizer, which reads
; the result back right after the atomic.
define i64 @__tokenspeed_persistent_lane0_add_i64(i64 %addr, i64 %v, i32 %lane) alwaysinline {
entry:
  %p = inttoptr i64 %addr to ptr addrspace(1)
  %is0 = icmp eq i32 %lane, 0
  br i1 %is0, label %do, label %join
do:
  %old = atomicrmw add ptr addrspace(1) %p, i64 %v syncscope("agent") monotonic, align 8
  br label %join
join:
  %r = phi i64 [ %old, %do ], [ undef, %entry ]
  ret i64 %r
}

; Acquire for one wave: an IR ordering point (loads after it cannot move
; above it), with no instruction on gfx950. It replaces the workgroup
; barrier after a per-wave poll whose result only this wave uses.
define i32 @__tokenspeed_persistent_wave_acquire() alwaysinline {
entry:
  fence syncscope("wavefront") acquire
  ret i32 0
}
