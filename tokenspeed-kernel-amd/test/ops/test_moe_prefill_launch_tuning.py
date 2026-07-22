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

from __future__ import annotations

import pytest

fused_mxfp = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950",
    reason="tokenspeed-kernel-amd is required for gfx950 MoE launch tuning tests",
)


def test_prefill_launch_tuning_dispatch_m_buckets():
    xcds = fused_mxfp._CDNA4_NUM_XCDS
    assert fused_mxfp._prefill_launch_tuning(
        "dispatch", m=1024, use_slice_mn=False
    ) == (1, xcds, None, False)
    assert fused_mxfp._prefill_launch_tuning(
        "dispatch", m=2048, use_slice_mn=False
    ) == (1, 4, None, False)
    assert fused_mxfp._prefill_launch_tuning(
        "dispatch", m=4096, use_slice_mn=False
    ) == (1, xcds, True, False)
    assert fused_mxfp._prefill_launch_tuning(
        "dispatch", m=8192, use_slice_mn=False
    ) == (1, None, None, False)


def test_prefill_launch_tuning_combine_m_buckets():
    xcds = fused_mxfp._CDNA4_NUM_XCDS
    assert fused_mxfp._prefill_launch_tuning("combine", m=1024, use_slice_mn=False) == (
        1,
        xcds,
        None,
        False,
    )
    assert fused_mxfp._prefill_launch_tuning("combine", m=2048, use_slice_mn=False) == (
        1,
        4,
        None,
        False,
    )
    assert fused_mxfp._prefill_launch_tuning("combine", m=4096, use_slice_mn=False) == (
        1,
        xcds,
        True,
        False,
    )
    assert fused_mxfp._prefill_launch_tuning("combine", m=8192, use_slice_mn=False) == (
        1,
        4,
        None,
        False,
    )
    assert fused_mxfp._prefill_launch_tuning(
        "combine", m=16384, use_slice_mn=False
    ) == (1, 4, None, True)


def test_prefill_launch_tuning_slice_mn_uses_default_group_only():
    assert fused_mxfp._prefill_launch_tuning("combine", m=4096, use_slice_mn=True) == (
        1,
        None,
        None,
        False,
    )
