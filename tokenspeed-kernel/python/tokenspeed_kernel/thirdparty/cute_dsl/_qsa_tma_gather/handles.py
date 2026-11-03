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


@dataclass(frozen=True, kw_only=True)
class MmaDesc:
    M: int
    N: int
    K: int
    bpe_a: int
    bpe_b: int
    tile_k_hw: int
    btranspose: bool
    atranspose: bool
    idesc: object
    kind: object

    @property
    def num_k_steps(self):
        return self.K // self.tile_k_hw

    @staticmethod
    def _swz_from_inner(inner_bytes: int) -> int:
        if inner_bytes % 128 == 0:
            return 128
        if inner_bytes % 64 == 0:
            return 64
        return 32

    @property
    def swz_a_bytes(self):
        inner = self.M if self.atranspose else self.K
        return MmaDesc._swz_from_inner(inner * self.bpe_a)

    @property
    def swz_b_bytes(self):
        inner = self.N if self.btranspose else self.K
        return MmaDesc._swz_from_inner(inner * self.bpe_b)

    @property
    def smem_advance_A_intra(self):
        if self.atranspose:
            return self.tile_k_hw * self.swz_a_bytes
        return self.tile_k_hw * self.bpe_a

    @property
    def smem_advance_B_intra(self):
        if self.btranspose:
            return self.tile_k_hw * self.swz_b_bytes
        return self.tile_k_hw * self.bpe_b

    @property
    def smem_subtile_A(self):
        if self.atranspose:
            return self.K * self.swz_a_bytes
        return self.swz_a_bytes * self.M

    @property
    def smem_subtile_B(self):
        if self.btranspose:
            return self.K * self.swz_b_bytes
        return self.swz_b_bytes * self.N

    @property
    def sps_A(self):
        if self.atranspose:
            return self.K // self.tile_k_hw
        return (self.swz_a_bytes // self.bpe_a) // self.tile_k_hw

    @property
    def sps_B(self):
        if self.btranspose:
            return self.K // self.tile_k_hw
        return (self.swz_b_bytes // self.bpe_b) // self.tile_k_hw


@dataclass(frozen=True, kw_only=True)
class SmemTile:
    base: object
    elems_per_stage: int
    leading_byte_offset: int
    stride_byte_offset: int
    layout: int

    def __getitem__(self, stage):
        return self.shifted(stage * self.elems_per_stage)

    def shifted(self, off_elems):
        """Offset the view in backing-array elements without copying storage."""
        base = self.base.subview(off_elems)
        return replace(self, base=base)

    def desc(self):
        from cutlass.experimental import primitives as prims

        # The kernel bounds all MMA operands to the 256 KiB descriptor window.
        return prims.Tcgen05SmemDesc.build(
            self.base,
            leading_byte_offset=self.leading_byte_offset,
            stride_byte_offset=self.stride_byte_offset,
            layout=self.layout,
        )
