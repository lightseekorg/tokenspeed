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
from tokenspeed_kernel.ops.sampling.triton import (
    gather_and_expand_scalars,
    min_p_renorm_prob,
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
