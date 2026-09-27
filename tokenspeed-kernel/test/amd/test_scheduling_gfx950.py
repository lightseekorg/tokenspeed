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

"""Native contracts for shared AMD instruction-scheduling hints."""

from pathlib import Path

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._scheduling import (  # noqa: E402
    sched_barrier,
    sched_barrier_compile_options,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402

_MMA = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=False,
        warps_per_cta=[1, 4],
    )
)


@gluon.jit
def _phase_boundary_probe(x, y, out_x, out_y, SCHED_LIBRARY_HASH: gl.constexpr):
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, _MMA))
    col = gl.arange(0, 64, layout=gl.SliceLayout(0, _MMA))
    offset = row[:, None] * 64 + col[None, :]
    c0 = gl.load(x + offset).to(gl.float32, bitcast=True)
    c1 = gl.load(y + offset).to(gl.float32, bitcast=True)
    for _ in gl.static_range(4):
        sched_barrier()
    gl.store(out_x + offset, c0.to(gl.int32, bitcast=True))
    gl.store(out_y + offset, c1.to(gl.int32, bitcast=True))


def _check_phase_boundary_bits():
    bits = torch.arange(1024, device="cuda", dtype=torch.int64)
    bits = ((bits * 0x9E3779B9 + 0x12345678) & 0xFFFFFFFF).to(torch.int32)
    boundaries = torch.tensor(
        [0, 0x80000000, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC12345, 0xFF812345],
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)
    bits[: boundaries.numel()] = boundaries
    other = bits.roll(137).clone()
    saved_bits, saved_other = bits.clone(), other.clone()
    out_x, out_y = torch.empty_like(bits), torch.empty_like(other)
    compiled = _phase_boundary_probe[(1,)](
        bits,
        other,
        out_x,
        out_y,
        num_warps=4,
        num_stages=1,
        **sched_barrier_compile_options(),
    )
    # The hint performs no arithmetic; this is not a NaN-payload math contract.
    torch.testing.assert_close(out_x, bits, atol=0, rtol=0)
    torch.testing.assert_close(out_y, other, atol=0, rtol=0)
    torch.testing.assert_close(bits, saved_bits, atol=0, rtol=0)
    torch.testing.assert_close(other, saved_other, atol=0, rtol=0)
    return compiled


def test_phase_fence_preserves_accumulator_bits():
    _check_phase_boundary_bits()


def test_scheduler_content_changes_compiled_kernel_key(monkeypatch, tmp_path):
    from tokenspeed_kernel_amd import _scheduling as _schedule

    path = tmp_path / "sched_barrier.ll"
    original = Path(_schedule._SCHED_LIBRARY_PATH).read_text()
    path.write_text(original)
    monkeypatch.setattr(_schedule, "_SCHED_LIBRARY_PATH", str(path))
    _schedule._scheduler_library_hash.cache_clear()
    try:
        first = _check_phase_boundary_bits()
        assert "@llvm.amdgcn.sched.barrier(i32 0)" in first.asm["llir"]
        # Keep the path fixed, but change the intrinsic's scheduling mask.
        path.write_text(
            original.replace("sched.barrier(i32 0)", "sched.barrier(i32 1)")
        )
        _schedule._scheduler_library_hash.cache_clear()
        second = _check_phase_boundary_bits()
        assert first.hash != second.hash
        assert "@llvm.amdgcn.sched.barrier(i32 1)" in second.asm["llir"]
        assert _check_phase_boundary_bits() is second
    finally:
        _schedule._scheduler_library_hash.cache_clear()
