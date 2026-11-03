# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from dataclasses import dataclass, replace
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims


@cute.jit
def wait(mb, phase):
    # Preserve the port's short suspend hint and parity-loop ordering.
    while not prims.mbarrier_try_wait_parity(mb, phase, time_limit=1):
        pass


@cute.jit
def arrive_expect_tx(mb, n_bytes, pred=None):
    if cutlass.const_expr(pred is None):
        prims.mbarrier_arrive_expect_tx(mb, n_bytes)
    elif pred:
        # Branch around the native op: inline-PTX predicates can lower to immediates.
        prims.mbarrier_arrive_expect_tx(mb, n_bytes)


@cute.jit
def commit_mma(mb, pred):
    # The predicate is a device value, so its branch must be inside a JIT function.
    if cutlass.const_expr(pred is None):
        prims.tcgen05_commit(mb, group=prims.CTAGroup.CTA_1)
    elif pred:
        prims.tcgen05_commit(mb, group=prims.CTAGroup.CTA_1)


class Producer(IntEnum):
    THREAD = 0
    TMA_LOAD = 1
    MMA_COMMIT = 2


@dataclass(frozen=True)
class MBarrier:
    """One single-CTA barrier ring with an explicit arrival owner and count."""

    base_ptr: object
    init_count: cutlass.Constexpr[int]
    producer: cutlass.Constexpr[int]
    stage_idx: object = 0

    def __getitem__(self, i):
        return replace(self, stage_idx=i)

    @property
    def smem_ptr(self):
        if isinstance(self.stage_idx, int) and self.stage_idx == 0:
            return self.base_ptr
        return self.base_ptr.subview(self.stage_idx)

    def init(self):
        prims.mbarrier_init(self.smem_ptr, self.init_count)

    def wait(self, phase):
        wait(self.smem_ptr, phase)

    def arrive(self, *, n_bytes=None, pred=None):
        if cutlass.const_expr(self.producer == int(Producer.THREAD)):
            if pred is not None or n_bytes is not None:
                raise TypeError("Thread arrivals accept neither pred nor n_bytes")
            prims.mbarrier_arrive(self.smem_ptr)
        elif cutlass.const_expr(self.producer == int(Producer.TMA_LOAD)):
            if n_bytes is None:
                raise TypeError("TMA arrivals require n_bytes")
            arrive_expect_tx(self.smem_ptr, n_bytes, pred=pred)
        elif cutlass.const_expr(self.producer == int(Producer.MMA_COMMIT)):
            if n_bytes is not None:
                raise TypeError("MMA commits do not accept n_bytes")
            commit_mma(self.smem_ptr, pred)
        else:
            raise ValueError(f"Unsupported barrier producer: {self.producer}")
