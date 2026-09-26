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

"""Coverage for the warp-pipelined gfx950 MLA prefill kernel.

Every call names the kernel, so shapes below its selection threshold still run
it; the default selection is covered by test_kernel_api_selection.py.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.mla import mla_prefill
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
pytestmark = pytest.mark.skipif(not platform.is_cdna4, reason="gfx950 MLA pipeline")
_KERNEL = "gluon_mla_prefill_8wave_gfx950"
# The kernel is registered for FP8 inputs only.
_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
# FP8 rounds P to 3 (E4M3) or 2 (E5M2) mantissa bits before the PV MFMA.
_OUT_TOL = 6e-2
_LSE_TOL = 1e-3


def _randn(shape, dtype, device):
    # torch.randn has no FP8 kernels; round a BF16 sample instead.
    return torch.randn(shape, dtype=torch.bfloat16, device=device).to(dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "out_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("num_heads", [12, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_mla_prefill_gluon_8wave_strided_output(
    device, require, dtype, out_dtype, num_heads, is_causal
):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    q = _randn((257, num_heads, 192), dtype, device)
    k = _randn((385, num_heads, 192), dtype, device)
    v = _randn((385, num_heads, 128), dtype, device)
    cu_q = torch.tensor([0, 257], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, 385], dtype=torch.int32, device=device)
    storage = torch.full(
        (259, num_heads * 2, 130), float("nan"), dtype=out_dtype, device=device
    )
    destination = storage[1:-1, ::2, 1:-1]
    out, lse = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=257,
        max_seqlen_kv=385,
        softmax_scale=192**-0.5,
        is_causal=is_causal,
        return_lse=True,
        override=_KERNEL,
        out=destination,
    )
    assert out is destination
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * (192**-0.5)
    if is_causal:
        rows = torch.arange(257, device=device) + 128
        cols = torch.arange(385, device=device)
        scores.masked_fill_(cols[None, :] > rows[:, None], -float("inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    tol, lse_tol = _OUT_TOL, _LSE_TOL
    torch.testing.assert_close(out.float(), reference, rtol=tol, atol=tol)
    torch.testing.assert_close(
        lse, scores.logsumexp(-1).transpose(0, 1), rtol=lse_tol, atol=lse_tol
    )
    # The unaligned column offset and untouched rows/heads catch over-wide stores.
    assert torch.isnan(storage[0]).all()
    assert torch.isnan(storage[-1]).all()
    assert torch.isnan(storage[:, 1::2]).all()
    assert torch.isnan(storage[:, :, 0]).all()
    assert torch.isnan(storage[:, :, -1]).all()


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("q_len,kv_len", [(129, 65), (65, 129), (257, 193), (65, 0)])
def test_mla_prefill_gluon_8wave_causal_cutoff(device, require, dtype, q_len, kv_len):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    q = torch.zeros((q_len, 12, 192), dtype=dtype, device=device)
    # Poison the backing tail; masked loads must not admit keys past KV length.
    k_storage = torch.full(
        (kv_len + 64, 12, 192), float("nan"), dtype=torch.bfloat16, device=device
    )
    v_storage = torch.full(
        (kv_len + 64, 12, 128), float("nan"), dtype=torch.bfloat16, device=device
    )
    k_storage[:kv_len] = 0
    values = (torch.arange(kv_len, device=device) % 31 - 15).float() / 16
    v_storage[:kv_len] = values[:, None, None]
    k = k_storage.to(dtype)[:kv_len]
    v = v_storage.to(dtype)[:kv_len]
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    out, lse = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=q_len,
        max_seqlen_kv=kv_len,
        softmax_scale=192**-0.5,
        is_causal=True,
        return_lse=True,
        override=_KERNEL,
    )
    # Zero logits give the exact prefix mean, capped at the last real key.
    visible = (torch.arange(q_len, device=device) + max(kv_len - q_len, 0) + 1).clamp(
        max=kv_len
    )
    expected = torch.zeros_like(out, dtype=torch.float32)
    if kv_len:
        expected = v.float().cumsum(dim=0)[visible - 1] / visible[:, None, None]
    torch.testing.assert_close(out.float(), expected, rtol=4e-3, atol=4e-3)
    expected_lse = visible.float().log()[:, None].expand_as(lse)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("changed_row", [None, 13, 45, 254])
@pytest.mark.parametrize(
    "kv_len", [1, 64, 65, 128, 129, 193, 256, 257, 321, 512, 513, 576]
)
def test_mla_prefill_gluon_8wave_online_max(
    device, require, dtype, changed_row, kv_len
):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    q = torch.zeros((255, 2, 192), dtype=torch.bfloat16, device=device)
    k = torch.zeros((576, 2, 192), dtype=torch.bfloat16, device=device)
    v = torch.empty((576, 2, 128), dtype=torch.bfloat16, device=device)
    q[:, :, 0] = 1.0
    if changed_row is not None:
        # Non-leading rows in multiple waves, including the final active row.
        q[changed_row, 0, 0] = -1.0
    # Distinct values and changing maxima span several K/V-ring wraps. In
    # base-2 logits each key step of 64 moves a score by about 6.7, so 16-bit
    # inputs both keep a lagging maximum (a jump below their rescale threshold
    # of 8) and rescale (larger jumps, and only in the changed row's wave);
    # FP8 inputs rescale on every jump.
    tiles = (
        (64, 1),
        (64, 0.5),
        (-64, -1),
        (-128, -0.5),
        (128, 0.75),
        (-256, 0.25),
        (32, -0.75),
        (64, -0.25),
        (0, 0.5),
    )
    for tile, (key, value) in enumerate(tiles):
        k[tile * 64 : (tile + 1) * 64, :, 0] = key
        v[tile * 64 : (tile + 1) * 64, :, :64] = value
        v[tile * 64 : (tile + 1) * 64, :, 64:] = 0.5 * value + 0.25
    # Masked tail loads must not propagate values from outside the request.
    k[kv_len:] = float("nan")
    v[kv_len:] = float("nan")
    q, k, v = q.to(dtype), k.to(dtype)[:kv_len], v.to(dtype)[:kv_len]
    cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    out, lse = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=q.shape[0],
        max_seqlen_kv=kv_len,
        softmax_scale=192**-0.5,
        is_causal=False,
        return_lse=True,
        override=_KERNEL,
    )
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * (192**-0.5)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        lse, scores.logsumexp(-1).transpose(0, 1), rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "q_len,kv_len,num_heads", [(64, 1000, 2), (257, 257, 2), (600, 600, 1)]
)
def test_mla_prefill_gluon_8wave_repeated_launches(
    device, require, dtype, is_causal, q_len, kv_len, num_heads
):
    # Random logits with doubled keys move the running maximum in some waves
    # and not others, across the 16-bit lazy rescale threshold. A short query
    # block over a long prefix crosses both the causal diagonal and the key
    # tail. Any race on the K/V LDS rings shows up as launches that disagree
    # bitwise.
    require("attention", "mla_prefill", "gluon", dtype, "q")
    torch.manual_seed(0)
    q = _randn((q_len, num_heads, 192), dtype, device)
    # Poison the backing tail; masked loads must not admit keys past KV length.
    k_storage = torch.full(
        (kv_len + 64, num_heads, 192), float("nan"), dtype=torch.bfloat16, device=device
    )
    v_storage = torch.full(
        (kv_len + 64, num_heads, 128), float("nan"), dtype=torch.bfloat16, device=device
    )
    k_storage[:kv_len] = 2 * torch.randn_like(k_storage[:kv_len])
    v_storage[:kv_len] = torch.randn_like(v_storage[:kv_len])
    k, v = k_storage.to(dtype)[:kv_len], v_storage.to(dtype)[:kv_len]
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * (192**-0.5)
    if is_causal:
        rows = torch.arange(q_len, device=device) + max(kv_len - q_len, 0)
        cols = torch.arange(kv_len, device=device)
        scores.masked_fill_(cols[None, :] > rows[:, None], -float("inf"))
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    expected_lse = scores.logsumexp(-1).transpose(0, 1)

    first = None
    for _ in range(4):
        out, lse = mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=q_len,
            max_seqlen_kv=kv_len,
            softmax_scale=192**-0.5,
            is_causal=is_causal,
            return_lse=True,
            override=_KERNEL,
        )
        tol, lse_tol = _OUT_TOL, _LSE_TOL
        torch.testing.assert_close(out.float(), expected, rtol=tol, atol=tol)
        torch.testing.assert_close(lse, expected_lse, rtol=lse_tol, atol=lse_tol)
        if first is None:
            first = (out.clone(), lse.clone())
        else:
            torch.testing.assert_close(out, first[0], rtol=0, atol=0)
            torch.testing.assert_close(lse, first[1], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_mla_prefill_gluon_8wave_launch_runtime_bound(require, monkeypatch, dtype):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla import prefill_8wave

    launches = []

    class RecordLaunch:
        def __getitem__(self, grid):
            assert grid == (512,)

            def launch(*args, **kwargs):
                launches.append(kwargs)

            return launch

    monkeypatch.setattr(prefill_8wave, "gluon_mla_prefill_8wave_gfx950", RecordLaunch())
    for tokens in (144, 160, 256, 272, 512, 2048, 8192, 16384):
        q = torch.empty((tokens, 12, 192), dtype=dtype, device="meta")
        v = torch.empty((tokens, 12, 128), dtype=dtype, device="meta")
        cu = torch.empty((2,), dtype=torch.int32, device="meta")
        prefill_8wave.launch_gluon_mla_prefill_8wave_gfx950(
            q, q, v, cu, cu, tokens, tokens, 192**-0.5
        )
        assert launches[-1]["max_seqlen_q"] == tokens
    # Crossing query-tile and persistent-cycle boundaries changes no constexpr.
    for launch in launches:
        launch.pop("max_seqlen_q")
    assert all(launch == launches[0] for launch in launches)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_mla_prefill_gluon_8wave_matches_previous_kernel(
    device, require, dtype, is_causal
):
    # The previous kernel stays reachable by name, e.g. for A/B runs.
    require("attention", "mla_prefill", "gluon", dtype, "q")
    torch.manual_seed(0)
    q_lens, kv_lens = (300, 5, 700), (300, 900, 700)
    q = _randn((sum(q_lens), 4, 192), dtype, device)
    k = _randn((sum(kv_lens), 4, 192), dtype, device)
    v = _randn((sum(kv_lens), 4, 128), dtype, device)
    cu_q = torch.tensor([0, 300, 305, 1005], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, 300, 1200, 1900], dtype=torch.int32, device=device)
    results = [
        mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=max(q_lens),
            max_seqlen_kv=max(kv_lens),
            softmax_scale=192**-0.5,
            is_causal=is_causal,
            return_lse=True,
            override=name,
        )
        for name in (_KERNEL, "gluon_mla_prefill_gfx950")
    ]
    (out, lse), (previous_out, previous_lse) = results
    tol, lse_tol = _OUT_TOL, _LSE_TOL
    torch.testing.assert_close(out.float(), previous_out.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(lse, previous_lse, rtol=lse_tol, atol=lse_tol)
