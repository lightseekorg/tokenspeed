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
import torch
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_renorm,
    fused_topk_topp_workspace_size,
)
from tokenspeed_kernel.ops.sampling.triton import (
    gather_and_expand_scalars,
    min_p_renorm_prob,
)
from tokenspeed_kernel.platform import current_platform

# Sentinel matching tokenspeed.runtime.sampling.sampling_params._TOP_K_DISABLED.
_TOP_K_DISABLED = 1 << 30

# The fused top-k + top-p kernel ships only as a CUDA build; on ROCm the
# Python entry point resolves to a RuntimeError stub. Gate the tests instead
# of failing loudly on AMD CI.
requires_nvidia = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="fused_topk_topp kernel is NVIDIA-only",
)


def _make_pools(pool_rows: int, device: str):
    temp = torch.linspace(0.5, 1.5, pool_rows, device=device, dtype=torch.float32)
    top_k = torch.arange(1, pool_rows + 1, device=device, dtype=torch.int32)
    top_p = torch.linspace(0.5, 1.0, pool_rows, device=device, dtype=torch.float32)
    min_p = torch.linspace(0.0, 0.2, pool_rows, device=device, dtype=torch.float32)
    seed = torch.arange(100, 100 + pool_rows, device=device, dtype=torch.int64)
    offsets = torch.arange(0, pool_rows, device=device, dtype=torch.int32) * 7
    return temp, top_k, top_p, min_p, seed, offsets


def _reference(index, pool, n: int):
    """index_select + repeat_interleave reference."""
    idx = index.long()
    return pool.index_select(0, idx).repeat_interleave(n, dim=0)


def _min_p_reference(probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    max_probs = probs.max(dim=-1, keepdim=True).values
    out = torch.where(
        probs >= min_p.to(probs.dtype).view(-1, 1) * max_probs,
        probs,
        torch.zeros_like(probs),
    )
    return out / out.sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("bs", [1, 4, 7])
@pytest.mark.parametrize("n", [1, 4, 8])
def test_gather_full(bs: int, n: int, device: str) -> None:
    pool_rows = 32
    torch.manual_seed(0)
    temp_p, top_k_p, top_p_p, min_p_p, seed_p, offsets_p = _make_pools(
        pool_rows, device
    )
    index = torch.randint(0, pool_rows, (bs,), device=device, dtype=torch.int32)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        index,
        temperature=temp_p,
        top_k=top_k_p,
        top_p=top_p_p,
        min_p=min_p_p,
        seed=seed_p,
        offsets=offsets_p,
        n=n,
    )

    torch.testing.assert_close(temps, _reference(index, temp_p, n))
    torch.testing.assert_close(top_ks, _reference(index, top_k_p, n))
    torch.testing.assert_close(top_ps, _reference(index, top_p_p, n))
    torch.testing.assert_close(min_ps, _reference(index, min_p_p, n))
    torch.testing.assert_close(seeds, _reference(index, seed_p, n))
    torch.testing.assert_close(offsets, _reference(index, offsets_p, n).to(torch.int64))


@pytest.mark.parametrize("n", [1, 5])
def test_gather_no_min_p_no_seed(n: int, device: str) -> None:
    """Verify path: drop min_p, seed, and offsets."""
    pool_rows = 16
    temp_p, top_k_p, top_p_p, _, _, _ = _make_pools(pool_rows, device)
    index = torch.arange(8, device=device, dtype=torch.int32) % pool_rows

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        index,
        temperature=temp_p,
        top_k=top_k_p,
        top_p=top_p_p,
        n=n,
    )

    assert min_ps is None
    assert seeds is None
    assert offsets is None
    torch.testing.assert_close(temps, _reference(index, temp_p, n))
    torch.testing.assert_close(top_ks, _reference(index, top_k_p, n))
    torch.testing.assert_close(top_ps, _reference(index, top_p_p, n))


def test_gather_sample_basic(device: str) -> None:
    """flashinfer.py sample(): seed + offsets, no min_p, n=1."""
    pool_rows = 16
    temp_p, top_k_p, top_p_p, _, seed_p, offsets_p = _make_pools(pool_rows, device)
    index = torch.tensor([3, 1, 0, 2], device=device, dtype=torch.int32)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        index,
        temperature=temp_p,
        top_k=top_k_p,
        top_p=top_p_p,
        seed=seed_p,
        offsets=offsets_p,
        n=1,
    )

    assert min_ps is None
    assert seeds is not None
    assert offsets is not None
    torch.testing.assert_close(temps, _reference(index, temp_p, 1))
    torch.testing.assert_close(seeds, _reference(index, seed_p, 1))
    torch.testing.assert_close(offsets, _reference(index, offsets_p, 1).to(torch.int64))
    assert offsets.dtype == torch.int64


def test_gather_min_p_only(device: str) -> None:
    """flashinfer_full.py verify(): min_p yes, seed no, offsets no."""
    pool_rows = 16
    temp_p, top_k_p, top_p_p, min_p_p, _, _ = _make_pools(pool_rows, device)
    index = torch.tensor([0, 5, 3], device=device, dtype=torch.int32)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        index,
        temperature=temp_p,
        top_k=top_k_p,
        top_p=top_p_p,
        min_p=min_p_p,
        n=4,
    )

    assert seeds is None
    assert offsets is None
    assert min_ps is not None
    torch.testing.assert_close(min_ps, _reference(index, min_p_p, 4))


def _ref_topk_topp(
    probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    """Pure-torch baseline mirroring flashinfer's ``top_k_renorm_prob`` followed
    by ``top_p_renorm_prob(is_deterministic=True)``. K >= V is treated as no
    top-k cutoff (matches both the flashinfer clamp and the K = 1<<30 sentinel).
    """
    bs, V = probs.shape
    out = probs.clone()
    for i in range(bs):
        k = min(int(top_ks[i].item()), V)
        if k < V:
            kth = torch.topk(out[i], k, sorted=False).values.min()
            out[i] = torch.where(out[i] >= kth, out[i], torch.zeros_like(out[i]))
            s = out[i].sum()
            if s > 0:
                out[i] = out[i] / s
        sorted_vals, _ = torch.sort(out[i], descending=True)
        cs = torch.cumsum(sorted_vals, 0)
        p = float(top_ps[i].item())
        # Smallest prefix with cumulative mass >= p. Clamp to V to absorb
        # fp32 rounding when p = 1.0 (cumsum's last value can fall a ulp
        # short of 1.0 and would otherwise push keep past the end).
        keep = min((cs < p).sum().item() + 1, V)
        thresh = sorted_vals[keep - 1]
        out[i] = torch.where(out[i] >= thresh, out[i], torch.zeros_like(out[i]))
        s = out[i].sum()
        if s > 0:
            out[i] = out[i] / s
    return out


@requires_nvidia
@pytest.mark.parametrize(
    "ks,ps,tag",
    [
        # Mode 3.1: top-K only (P=1.0).
        ([1, 16, 64, 128, 1, 64, 16, 128], [1.0] * 8, "topk-only"),
        # Mode 3.2: top-P only (K sentinel → radix path).
        (
            [_TOP_K_DISABLED] * 8,
            [0.5, 0.7, 0.9, 0.95, 0.99, 0.5, 0.9, 0.8],
            "topp-only",
        ),
        # Mode 3.3: top-K + top-P together.
        (
            [64, 64, 32, 128, 16, 8, 64, 128],
            [0.9, 0.5, 0.8, 0.7, 0.95, 0.6, 0.9, 0.99],
            "topk+topp",
        ),
        # Mixed batch: different rows take different paths in one launch.
        (
            [64, _TOP_K_DISABLED, 1, 128, 32, _TOP_K_DISABLED, 16, 8],
            [0.9, 0.9, 1.0, 0.7, 0.95, 0.5, 0.8, 0.99],
            "mixed",
        ),
    ],
)
def test_fused_topk_topp_matches_pipeline(
    device: str, ks: list[int], ps: list[float], tag: str
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for fused_topk_topp_renorm test")
    torch.manual_seed(0)
    bs, V = len(ks), 8192
    logits = torch.randn(bs, V, device=device, dtype=torch.float32) * 3.0
    probs = torch.softmax(logits, dim=-1)
    top_ks = torch.tensor(ks, dtype=torch.int32, device=device)
    top_ps = torch.tensor(ps, dtype=torch.float32, device=device)

    ref = _ref_topk_topp(probs, top_ks, top_ps)
    ours = fused_topk_topp_renorm(probs.clone(), top_ks, top_ps)

    # Each kept row should renormalize to 1 within fp32 ulp tolerance.
    torch.testing.assert_close(
        ours.sum(dim=-1), torch.ones(bs, device=device), atol=1e-5, rtol=1e-5
    )
    # The fused kernel must produce the same kept set on every row. Allow
    # a single position of disagreement to absorb ties at the top-p cutoff
    # (cumulative-sum rounding can include/exclude the boundary by 1 entry).
    pos_ref = ref > 0
    pos_ours = ours > 0
    pos_diff = (pos_ref != pos_ours).sum(dim=-1).max().item()
    assert pos_diff <= 1, f"[{tag}] kept-position mismatch up to {pos_diff} per row"
    # Renormalized values: sub-ulp accumulation order differs by row scale,
    # so 1e-5 is the right tolerance (matches the per-row sum bound).
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-4)


@requires_nvidia
def test_fused_topk_topp_workspace_size_grows_with_batch(device: str) -> None:
    """Workspace size must grow monotonically with batch and vocab so callers
    can pre-allocate a buffer sized for ``max_bs × vocab``."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for fused_topk_topp_workspace_size test")
    V = 8192
    small = fused_topk_topp_workspace_size(1, V)
    large = fused_topk_topp_workspace_size(64, V)
    assert small > 0
    assert large > small


@requires_nvidia
def test_fused_topk_topp_external_workspace(device: str) -> None:
    """Pre-allocated workspace path must produce the same result as the
    auto-allocated one, so the runtime can hoist the alloc out of the hot
    path."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for fused_topk_topp_external_workspace test")
    torch.manual_seed(1)
    bs, V = 4, 8192
    probs = torch.softmax(
        torch.randn(bs, V, device=device, dtype=torch.float32) * 2.5, dim=-1
    )
    top_ks = torch.tensor(
        [32, _TOP_K_DISABLED, 64, 128], dtype=torch.int32, device=device
    )
    top_ps = torch.tensor([0.9, 0.85, 0.95, 0.8], dtype=torch.float32, device=device)

    auto = fused_topk_topp_renorm(probs, top_ks, top_ps)
    ws = torch.empty(
        fused_topk_topp_workspace_size(bs, V), dtype=torch.uint8, device=device
    )
    manual = fused_topk_topp_renorm(probs, top_ks, top_ps, workspace=ws)
    torch.testing.assert_close(auto, manual, atol=0.0, rtol=0.0)


def test_gather_empty_batch(device: str) -> None:
    pool_rows = 16
    temp_p, top_k_p, top_p_p, min_p_p, seed_p, offsets_p = _make_pools(
        pool_rows, device
    )
    index = torch.empty(0, device=device, dtype=torch.int32)

    temps, top_ks, top_ps, min_ps, seeds, offsets = gather_and_expand_scalars(
        index,
        temperature=temp_p,
        top_k=top_k_p,
        top_p=top_p_p,
        min_p=min_p_p,
        seed=seed_p,
        offsets=offsets_p,
        n=5,
    )

    assert temps.numel() == 0
    assert top_ks.numel() == 0
    assert top_ps.numel() == 0
    assert min_ps.numel() == 0
    assert seeds.numel() == 0
    assert offsets.numel() == 0


@pytest.mark.parametrize("rows", [1, 3, 5])
@pytest.mark.parametrize("vocab_size", [17, 257, 1025])
def test_min_p_renorm_prob(rows: int, vocab_size: int, device: str) -> None:
    torch.manual_seed(rows * 1000 + vocab_size)
    probs = torch.rand((rows, vocab_size), device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    min_p = torch.linspace(0.0, 0.2, rows, device=device, dtype=torch.float32)

    out = min_p_renorm_prob(probs, min_p)
    ref = _min_p_reference(probs, min_p)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(rows, device=device))


def test_min_p_renorm_prob_bf16_min_p(device: str) -> None:
    torch.manual_seed(0)
    probs = torch.rand((4, 513), device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    min_p = torch.tensor([0.0, 0.01, 0.05, 0.2], device=device, dtype=torch.bfloat16)

    out = min_p_renorm_prob(probs, min_p)
    ref = _min_p_reference(probs, min_p)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_min_p_renorm_prob_empty_batch(device: str) -> None:
    probs = torch.empty((0, 32), device=device, dtype=torch.float32)
    min_p = torch.empty((0,), device=device, dtype=torch.float32)

    out = min_p_renorm_prob(probs, min_p)

    assert out.shape == probs.shape
    assert out.dtype == probs.dtype
