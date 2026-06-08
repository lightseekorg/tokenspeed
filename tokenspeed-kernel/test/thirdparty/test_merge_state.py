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

import math
from typing import Tuple

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.merge_state import (
    LSE_LN,
    LSE_LOG2,
    merge_state,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="merge_state CUDA kernel is NVIDIA-only",
)


def _reference_merge(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    lse_scale_log2: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference mirroring the kernel's log2-internal arithmetic.

    The merge math is base-agnostic — using log2 internally and rebasing input/
    output via ``lse_scale_log2`` matches the kernel exactly so tolerances stay
    tight (no exp vs exp2 lib-call drift).
    """
    s_a_log2 = s_a.float() * lse_scale_log2
    s_b_log2 = s_b.float() * lse_scale_log2
    s_max = torch.maximum(s_a_log2, s_b_log2)
    w_a = torch.exp2(s_a_log2 - s_max)
    w_b = torch.exp2(s_b_log2 - s_max)
    sum_w = w_a + w_b
    v_merged = (
        w_a.unsqueeze(-1) * v_a.float() + w_b.unsqueeze(-1) * v_b.float()
    ) / sum_w.unsqueeze(-1)
    s_merged = (torch.log2(sum_w) + s_max) * (1.0 / lse_scale_log2)
    return v_merged.to(v_a.dtype), s_merged


def _make_inputs(
    T: int, H: int, D: int, device: str, v_dtype: torch.dtype, seed: int = 0
):
    torch.manual_seed(seed)
    v_a = torch.randn(T, H, D, device=device, dtype=v_dtype)
    v_b = torch.randn(T, H, D, device=device, dtype=v_dtype)
    s_a = torch.randn(T, H, device=device, dtype=torch.float32)
    s_b = torch.randn(T, H, device=device, dtype=torch.float32)
    return v_a, s_a, v_b, s_b


def test_lse_constants() -> None:
    """LSE_LN must be log2(e), LSE_LOG2 must be 1.0 — the two named presets the
    public API documents."""
    assert LSE_LOG2 == 1.0
    assert math.isclose(LSE_LN, math.log2(math.e), rel_tol=0, abs_tol=0)


# Sweep representative DSv3/K2.5 shapes plus a non-power-of-2 H to exercise the
# NUM_HEADS_PAD masking path (TP configs typically give 8/16/32 heads, but the
# kernel must remain correct for arbitrary head counts).
SHAPES = [
    (1, 16, 128),
    (8, 16, 128),
    (256, 16, 128),
    (8192, 16, 128),
    (4096, 8, 128),
    (256, 12, 128),  # non-pow2 H — masking
    (256, 16, 64),  # different head_dim
]


@pytest.mark.parametrize("T,H,D", SHAPES)
@pytest.mark.parametrize("v_dtype", [torch.bfloat16, torch.float16])
def test_natural_log_default(
    device: str, T: int, H: int, D: int, v_dtype: torch.dtype
) -> None:
    """Default kwargs: natural-log LSE in, natural-log LSE out."""
    v_a, s_a, v_b, s_b = _make_inputs(T, H, D, device, v_dtype)

    v_out, s_out = merge_state(v_a, s_a, v_b, s_b)
    torch.cuda.synchronize()

    v_ref, s_ref = _reference_merge(v_a, s_a, v_b, s_b, LSE_LN)

    # bf16/fp16 V accumulator drift scales with H*D — ~5e-2 abs is normal.
    # LSE is fp32 throughout so a tighter bound is fine.
    assert torch.allclose(v_out.float(), v_ref.float(), atol=5e-2, rtol=1e-2)
    assert torch.allclose(s_out, s_ref, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("T,H,D", SHAPES)
def test_log2_basis(device: str, T: int, H: int, D: int) -> None:
    """Explicit log2 basis: lse_scale_log2=1.0 means "input is already log2"."""
    v_a, s_a, v_b, s_b = _make_inputs(T, H, D, device, torch.bfloat16)

    v_out, s_out = merge_state(v_a, s_a, v_b, s_b, lse_scale_log2=LSE_LOG2)
    torch.cuda.synchronize()

    v_ref, s_ref = _reference_merge(v_a, s_a, v_b, s_b, LSE_LOG2)

    assert torch.allclose(v_out.float(), v_ref.float(), atol=5e-2, rtol=1e-2)
    assert torch.allclose(s_out, s_ref, atol=1e-4, rtol=1e-5)


def test_basis_round_trip(device: str) -> None:
    """Two equivalent calls — natural-log inputs vs the same data pre-rebased
    to log2 — must produce algebraically equivalent results. The two paths
    differ only in *where* the LSE_LN multiply happens (kernel-internal vs
    PyTorch pre-multiply), so 1-ULP fp32 drift is expected; allclose with a
    tight tolerance is the right bar."""
    T, H, D = 256, 16, 128
    v_a, s_a, v_b, s_b = _make_inputs(T, H, D, device, torch.bfloat16)

    s_a_log2 = (s_a * LSE_LN).contiguous()
    s_b_log2 = (s_b * LSE_LN).contiguous()

    v_ln, s_ln = merge_state(v_a, s_a, v_b, s_b)  # default LSE_LN
    v_log2, s_log2 = merge_state(v_a, s_a_log2, v_b, s_b_log2, lse_scale_log2=LSE_LOG2)
    torch.cuda.synchronize()

    # bf16 V: 1 ULP at this scale ≈ 4e-3 abs. Tighter than the cross-reference
    # test (5e-2) because both sides ran the *same* kernel.
    assert torch.allclose(v_ln.float(), v_log2.float(), atol=5e-3, rtol=1e-3)
    # Output LSE differs by the basis multiplier, modulo fp32 round-off.
    assert torch.allclose(s_ln * LSE_LN, s_log2, atol=1e-5, rtol=1e-5)


def test_arbitrary_lse_base(device: str) -> None:
    """A non-canonical base (log10 here) must work too — the runtime knob is
    the whole point of the parameter, not just LSE_LN/LSE_LOG2."""
    T, H, D = 128, 16, 128
    v_a, s_a, v_b, s_b = _make_inputs(T, H, D, device, torch.bfloat16)

    scale = math.log2(10.0)  # caller's LSE is in log10
    v_out, s_out = merge_state(v_a, s_a, v_b, s_b, lse_scale_log2=scale)
    torch.cuda.synchronize()

    v_ref, s_ref = _reference_merge(v_a, s_a, v_b, s_b, scale)
    assert torch.allclose(v_out.float(), v_ref.float(), atol=5e-2, rtol=1e-2)
    assert torch.allclose(s_out, s_ref, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("lse_dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_rejects_non_fp32_lse(device: str, lse_dtype: torch.dtype) -> None:
    """fp32 LSE is a precondition (the kernel doesn't cast internally) — non-fp32
    must raise rather than silently mis-merge."""
    v_a = torch.randn(64, 16, 128, device=device, dtype=torch.bfloat16)
    v_b = torch.randn(64, 16, 128, device=device, dtype=torch.bfloat16)
    s_a = torch.randn(64, 16, device=device, dtype=lse_dtype)
    s_b = torch.randn(64, 16, device=device, dtype=lse_dtype)
    with pytest.raises(AssertionError, match="fp32 LSE"):
        merge_state(v_a, s_a, v_b, s_b)


def test_output_dtypes_and_shapes(device: str) -> None:
    """V output mirrors V input dtype; LSE output is always fp32."""
    T, H, D = 64, 16, 128
    for v_dtype in (torch.bfloat16, torch.float16):
        v_a, s_a, v_b, s_b = _make_inputs(T, H, D, device, v_dtype)
        v_out, s_out = merge_state(v_a, s_a, v_b, s_b)
        torch.cuda.synchronize()
        assert v_out.dtype == v_dtype
        assert v_out.shape == (T, H, D)
        assert s_out.dtype == torch.float32
        assert s_out.shape == (T, H)


def test_rejects_fp32_v(device: str) -> None:
    """The CUDA kernel only dispatches fp16/bf16 V; fp32 must raise rather than
    silently fall through."""
    T, H, D = 64, 16, 128
    v_a, s_a, v_b, s_b = _make_inputs(T, H, D, device, torch.float32)
    with pytest.raises(AssertionError, match="V must be bf16/fp16"):
        merge_state(v_a, s_a, v_b, s_b)
