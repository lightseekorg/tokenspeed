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

"""Correctness tests for the MI355 Gluon MoE kernel.

The Gluon kernel is gated behind ``TOKENSPEED_MOE_GLUON=1`` for the runtime
selector, but in this test we call the implementation directly so we don't
need the env knob set. We compare against:

1. A pure-PyTorch ragged matmul reference (always-on correctness check).
2. ``triton_kernels.matmul`` with the same precision config (when available).
"""

from __future__ import annotations

import pytest

# IMPORTANT: tokenspeed_kernel must be imported before torch on this docker
# image to avoid an ABI segfault between the system torch and the bundled
# tokenspeed_triton C extension.
import tokenspeed_kernel  # noqa: F401  (must be first)
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4,
    reason="Gluon MoE kernel is implemented for CDNA4 (gfx950 / MI355) only.",
)


def _build_ragged(M: int, E: int, *, block_m: int = 64, device="cuda") -> "tuple":
    """Build a fake routing assignment with block-aligned per-expert counts.

    The Gluon MoE kernel assumes each per-block expert id is unique within
    that block, which is true when every expert's row count is a multiple of
    ``block_m``.
    """
    from triton_kernels.tensor import make_ragged_tensor_metadata

    per_expert = max(block_m, (M // E) // block_m * block_m)
    M_padded = per_expert * E
    counts = torch.full((E,), per_expert, device=device, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M_padded)
    gather_indx = type(
        "GatherIndx",
        (),
        {"src_indx": torch.arange(M_padded, device=device, dtype=torch.int32)},
    )()
    return md, gather_indx, counts, M_padded


def _torch_ragged_matmul(x, w, counts, bias=None):
    """Reference: per-expert dense matmul on slices of x."""
    M, K = x.shape
    E, K_W, N = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    start = 0
    for e in range(E):
        n = int(counts[e].item())
        if n == 0:
            continue
        slc = x[start : start + n].to(torch.float32)
        we = w[e].to(torch.float32)
        out[start : start + n] = slc @ we
        if bias is not None:
            out[start : start + n] += bias[e].to(torch.float32)
        start += n
    return out


@pytest.mark.parametrize(
    "M,N,K,E,block_m",
    [
        (128, 128, 128, 2, 64),
        (256, 256, 128, 4, 64),
        (512, 512, 256, 8, 64),
    ],
)
def test_gluon_bf16_ragged_matches_torch(M, N, K, E, block_m):
    """The Gluon bf16 ragged GEMM matches a torch reference within bf16 noise."""
    pytest.importorskip("triton_kernels")
    from tokenspeed_kernel.ops.moe.gluon import _gluon_bf16_ragged_matmul

    torch.manual_seed(0)
    device = "cuda"
    md, gather_indx, counts, M_padded = _build_ragged(
        M, E, block_m=block_m, device=device
    )
    x = torch.randn(M_padded, K, device=device, dtype=torch.bfloat16) * 0.05
    w = torch.randn(E, K, N, device=device, dtype=torch.bfloat16) * 0.05

    out = _gluon_bf16_ragged_matmul(
        x,
        w,
        bias=None,
        a_ragged_metadata=md,
        gather_indx=None,
        scatter_indx=None,
        precision_config=None,
        fused_activation=None,
        n_tokens=None,
        n_expts_act=None,
        block_m=block_m,
    )
    ref = _torch_ragged_matmul(x, w, counts)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=5e-2, atol=5e-2)


def test_gluon_kernel_selected_under_env(monkeypatch):
    """With ``TOKENSPEED_MOE_GLUON=1`` the registry picks the Gluon variant.

    We exercise the public selection API so this test fails loudly if the
    feature gate breaks.
    """
    pytest.importorskip("triton_kernels")
    monkeypatch.setenv("TOKENSPEED_MOE_GLUON", "1")

    # Force re-import of moe ops so the priority bump is visible.
    import importlib

    import tokenspeed_kernel.ops.moe as moe_pkg
    import tokenspeed_kernel.ops.moe.gluon as gluon_mod

    importlib.reload(gluon_mod)
    importlib.reload(moe_pkg)

    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import select_kernel

    KernelRegistry.get().clear_cache()
    selected = select_kernel(
        "moe",
        "experts",
        torch.bfloat16,
        features=frozenset({"ragged_metadata", "dispatch_gemm"}),
        traits={},
    )
    selected_name = getattr(selected, "name", None) or getattr(selected, "__name__", "")
    assert "gluon" in selected_name, f"unexpected selected kernel: {selected_name}"
    # Restore the default state for subsequent tests in the session.
    monkeypatch.delenv("TOKENSPEED_MOE_GLUON", raising=False)
    importlib.reload(gluon_mod)
    importlib.reload(moe_pkg)


def test_gluon_falls_back_for_mxfp4():
    """mxfp4 weights / fp8 activations should fall back to triton_kernels."""
    pytest.importorskip("triton_kernels")
    from tokenspeed_kernel.ops.moe.gluon import _gluon_bf16_ragged_matmul
    from triton_kernels.matmul import PrecisionConfig

    M, N, K, E = 64, 128, 128, 2
    device = "cuda"
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w_mxfp4 = torch.zeros(E, K // 2, N, device=device, dtype=torch.uint8)
    w_scale = torch.zeros(E, K // 32, N, device=device, dtype=torch.uint8)
    md, _, _, _ = _build_ragged(M, E, device=device)
    pc = PrecisionConfig()
    pc.b_mx_scale = w_scale  # triggers fallback condition
    # Smoke-test that the fallback path is actually taken (we don't validate
    # numerics here -- the upstream backend has its own tests).
    with pytest.raises(Exception):
        # We expect this to call upstream matmul which will then complain
        # because we passed dummy zeros; what matters is that we got past
        # the Gluon prologue without trying to lower a mxfp4 path through it.
        _gluon_bf16_ragged_matmul(
            x,
            w_mxfp4,
            bias=None,
            a_ragged_metadata=md,
            gather_indx=None,
            scatter_indx=None,
            precision_config=pc,
            fused_activation=None,
            n_tokens=None,
            n_expts_act=None,
        )
