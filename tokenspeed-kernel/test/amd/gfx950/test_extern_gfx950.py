# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native contracts for the shared AMD extern intrinsic libraries."""

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._extern import (  # noqa: E402
    arithmetic_fence,
    extern_compile_options,
    release_wait_vmem,
    sched_group,
    wait_vmem,
    wave_all_equal,
    wave_ballot_i64,
    wave_mbcnt,
    wave_popcount_i64,
    wave_uniform_i32,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402


@gluon.jit
def _sched_group_probe(x, out):
    offset = gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))
    y = gl.exp2(gl.load(x + offset))
    sched_group(("trans",), 2)
    sched_group("mfma", 1)
    sched_group("valu", 4)
    gl.store(out + offset, y)


def test_sched_group_accepts_class_names():
    x = torch.randn(256, device="cuda")
    out = torch.empty_like(x)
    compiled = _sched_group_probe[(1,)](x, out, num_warps=4, **extern_compile_options())
    llir = compiled.asm["llir"]
    assert "@llvm.amdgcn.sched.group.barrier(i32 1024, i32 2, i32 0)" in llir
    assert "@llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)" in llir
    assert "@llvm.amdgcn.sched.group.barrier(i32 2, i32 4, i32 0)" in llir
    torch.testing.assert_close(out, torch.exp2(x), atol=0, rtol=0)


@gluon.jit
def _multi_library_probe(a, b, first_lane, all_equal, fenced):
    # Element i belongs to lane i % 64 of wave i // 64.
    offset = gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))
    gl.store(first_lane + offset, wave_uniform_i32(offset))
    x = gl.load(a + offset)
    equal = wave_all_equal(x, gl.load(b + offset))
    gl.store(all_equal + offset, equal)
    gl.store(fenced + offset, arithmetic_fence(x) * 2.0)
    wait_vmem()


def test_one_kernel_links_several_libraries():
    a = torch.randn(256, device="cuda")
    b = a.clone()
    # Waves 1 and 3 differ in one lane each; waves 0 and 2 match.
    b[64 + 17] += 1
    b[192 + 63] = float("nan")
    first_lane = torch.empty(256, dtype=torch.int32, device="cuda")
    all_equal = torch.empty_like(first_lane)
    fenced = torch.empty_like(a)
    compiled = _multi_library_probe[(1,)](
        a, b, first_lane, all_equal, fenced, num_warps=4, **extern_compile_options()
    )
    llir = compiled.asm["llir"]
    assert "@llvm.amdgcn.readfirstlane" in llir
    assert "@llvm.amdgcn.ballot.i64" in llir
    assert "@llvm.arithmetic.fence.f32" in llir
    assert "@llvm.amdgcn.s.waitcnt(i32 3952)" in llir
    wave = torch.arange(256, device="cuda") // 64
    torch.testing.assert_close(first_lane, (wave * 64).to(torch.int32))
    expected = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(all_equal, expected[wave])
    torch.testing.assert_close(fenced, a * 2.0, atol=0, rtol=0)


@gluon.jit
def _publish_probe(data, flag):
    offset = gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))
    gl.store(data + offset, offset)
    release_wait_vmem()
    gl.barrier()
    gl.store(flag + offset, offset * 0 + 1, mask=offset == 0)


def test_the_release_wait_links_a_fence_before_the_wait():
    """``release_wait_vmem`` links a workgroup-scope release fence before
    its ``vmcnt(0)`` wait, so LLVM cannot move the earlier data stores below
    the wait. The check is on the IR: a run's results cannot show whether
    a store was moved past the wait."""
    data = torch.empty(256, dtype=torch.int32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    compiled = _publish_probe[(1,)](data, flag, num_warps=4, **extern_compile_options())
    llir = compiled.asm["llir"]
    fence = llir.index('fence syncscope("workgroup") release')
    assert fence < llir.index("@llvm.amdgcn.s.waitcnt(i32 3952)", fence)


@gluon.jit
def _wave_mask_probe(flags, masks, ranks, counts):
    offset = gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))
    value = gl.load(flags + offset)
    mask = wave_ballot_i64(value)
    gl.store(masks + offset, mask)
    gl.store(ranks + offset, wave_mbcnt(mask))
    gl.store(counts + offset, wave_popcount_i64(mask))


@pytest.mark.parametrize("pattern", [0, 1, 2, 3])
def test_wave_ballot_and_rank(pattern):
    values = torch.arange(256, dtype=torch.int32)
    if pattern == 0:
        values[:] = 0
    elif pattern == 1:
        values[:] = -1
    elif pattern == 2:
        values = (values % 64 == 63).to(torch.int32)
    else:
        values = (values % 3 == 0).to(torch.int32) * -7
    expected_masks = []
    expected_ranks = []
    for wave in values.reshape(4, 64).tolist():
        mask = sum((v != 0) << lane for lane, v in enumerate(wave))
        expected_masks.extend([mask if mask < 1 << 63 else mask - (1 << 64)] * 64)
        expected_ranks.extend(
            (mask & ((1 << lane) - 1)).bit_count() for lane in range(64)
        )
    flags = values.cuda()
    masks = torch.empty(256, dtype=torch.int64, device="cuda")
    ranks = torch.empty_like(flags)
    counts = torch.empty_like(flags)
    _wave_mask_probe[(1,)](
        flags, masks, ranks, counts, num_warps=4, **extern_compile_options()
    )
    assert masks.cpu().tolist() == expected_masks
    assert ranks.cpu().tolist() == expected_ranks
    assert counts.cpu().tolist() == [
        (m & ((1 << 64) - 1)).bit_count() for m in expected_masks
    ]
