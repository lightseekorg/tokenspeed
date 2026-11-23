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

; Explicitly masked shared-memory (LDS) accesses. Every function is named
; __tokenspeed_shared_* so that Triton links this file only into kernels
; calling one (_extern.py). %address is the byte address of an allocation in
; the LDS address space and %index an element index into it; only lanes with
; a nonzero %live access memory. None of these adds a workgroup barrier: the
; caller orders them against other waves.

define i32 @__tokenspeed_shared_load_i32_masked(i32 %address, i32 %index, i32 %live) alwaysinline {
entry:
  %enabled = icmp ne i32 %live, 0
  br i1 %enabled, label %read, label %done
read:
  %base = inttoptr i32 %address to ptr addrspace(3)
  %ptr = getelementptr i32, ptr addrspace(3) %base, i32 %index
  %value = load i32, ptr addrspace(3) %ptr, align 4
  br label %done
done:
  %result = phi i32 [ %value, %read ], [ 0, %entry ]
  ret i32 %result
}

define i32 @__tokenspeed_shared_store_i32_masked(i32 %address, i32 %index, i32 %value, i32 %live) alwaysinline {
entry:
  %enabled = icmp ne i32 %live, 0
  br i1 %enabled, label %write, label %done
write:
  %base = inttoptr i32 %address to ptr addrspace(3)
  %ptr = getelementptr i32, ptr addrspace(3) %base, i32 %index
  store i32 %value, ptr addrspace(3) %ptr, align 4
  br label %done
done:
  ret i32 0
}

define i64 @__tokenspeed_shared_load_i64_masked(i32 %address, i32 %index, i32 %live) alwaysinline {
entry:
  %enabled = icmp ne i32 %live, 0
  br i1 %enabled, label %read, label %done
read:
  %base = inttoptr i32 %address to ptr addrspace(3)
  %ptr = getelementptr i64, ptr addrspace(3) %base, i32 %index
  %value = load i64, ptr addrspace(3) %ptr, align 8
  br label %done
done:
  %result = phi i64 [ %value, %read ], [ 0, %entry ]
  ret i64 %result
}

define i32 @__tokenspeed_shared_store_i64_masked(i32 %address, i32 %index, i64 %value, i32 %live) alwaysinline {
entry:
  %enabled = icmp ne i32 %live, 0
  br i1 %enabled, label %write, label %done
write:
  %base = inttoptr i32 %address to ptr addrspace(3)
  %ptr = getelementptr i64, ptr addrspace(3) %base, i32 %index
  store i64 %value, ptr addrspace(3) %ptr, align 8
  br label %done
done:
  ret i32 0
}
