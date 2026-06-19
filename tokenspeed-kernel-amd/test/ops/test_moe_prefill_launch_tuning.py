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
