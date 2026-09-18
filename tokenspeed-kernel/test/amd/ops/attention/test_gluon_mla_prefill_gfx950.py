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

"""FP8 pipeline, masking and output-contract coverage for gfx950 MLA prefill."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.mla import mla_prefill
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
pytestmark = pytest.mark.skipif(not platform.is_cdna4, reason="gfx950 MLA pipeline")
_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize(
    "out_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("num_heads", [12, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_mla_prefill_gluon_fp8_strided_output(
    device, require, dtype, out_dtype, num_heads, is_causal
):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    q = torch.randn((257, num_heads, 192), dtype=torch.bfloat16, device=device).to(
        dtype
    )
    k = torch.randn((385, num_heads, 192), dtype=torch.bfloat16, device=device).to(
        dtype
    )
    v = torch.randn((385, num_heads, 128), dtype=torch.bfloat16, device=device).to(
        dtype
    )
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
        solution="gluon",
        out=destination,
    )
    assert out is destination
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * (192**-0.5)
    if is_causal:
        rows = torch.arange(257, device=device) + 128
        cols = torch.arange(385, device=device)
        scores.masked_fill_(cols[None, :] > rows[:, None], -float("inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    torch.testing.assert_close(out.float(), reference, rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(
        lse, scores.logsumexp(-1).transpose(0, 1), rtol=8e-2, atol=8e-2
    )
    # The unaligned column offset and untouched rows/heads catch over-wide stores.
    assert torch.isnan(storage[0]).all()
    assert torch.isnan(storage[-1]).all()
    assert torch.isnan(storage[:, 1::2]).all()
    assert torch.isnan(storage[:, :, 0]).all()
    assert torch.isnan(storage[:, :, -1]).all()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("num_heads", [12, 128])
@pytest.mark.parametrize("q_lens", [(2048, 1), (1, 2048), (257, 257), (1,) * 40])
def test_mla_prefill_gluon_fp8_static_grid(require, dtype, num_heads, q_lens):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.prefill import get_config

    q = torch.empty((sum(q_lens), num_heads, 192), dtype=dtype, device="meta")
    config = get_config(q=q, k=q)
    assert config.grid == (512,)
    assert config.num_warps == 8


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("max_seqlen_q", [512, 65536])
def test_mla_prefill_gluon_fp8_static_grid_graph(
    device, require, dtype, is_causal, max_seqlen_q
):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.prefill import get_config

    q = torch.randn((512, 12, 192), dtype=torch.bfloat16, device=device).to(dtype)
    k = torch.randn((768, 12, 192), dtype=torch.bfloat16, device=device).to(dtype)
    v = torch.randn((768, 12, 128), dtype=torch.bfloat16, device=device).to(dtype)
    cu_q = torch.tensor([0, 64, 512], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, 256, 768], dtype=torch.int32, device=device)
    assert get_config(q=q, k=k).grid == (512,)

    def invoke():
        return mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=512,
            softmax_scale=192**-0.5,
            is_causal=is_causal,
            return_lse=True,
            solution="gluon",
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_out, actual_lse = invoke()
    # A captured launch must cover different ragged splits of the same buffers.
    for lengths in ([0, 64, 512], [0, 256, 512], [0, 1, 512], [0, 0, 512]):
        cu_q.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        expected_out, expected_lse = invoke()
        graph.replay()
        torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_mla_prefill_gluon_launch_runtime_bound(require, monkeypatch, dtype):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla import prefill

    launches = []

    class RecordLaunch:
        def __getitem__(self, grid):
            assert grid == (512,)

            def launch(*args, **kwargs):
                launches.append(kwargs)

            return launch

    monkeypatch.setattr(prefill, "_mla_prefill_kernel", RecordLaunch())
    for tokens in (144, 160, 256, 272, 512, 2048, 8192, 16384):
        q = torch.empty((tokens, 12, 192), dtype=dtype, device="meta")
        v = torch.empty((tokens, 12, 128), dtype=dtype, device="meta")
        cu = torch.empty((2,), dtype=torch.int32, device="meta")
        prefill.gluon_mla_prefill_gfx950(q, q, v, cu, cu, 65536, 65536, 192**-0.5)
        assert launches[-1]["max_seqlen_q"] == (
            tokens if dtype in _FP8_DTYPES else 65536
        )
    # Crossing query-tile and persistent-cycle boundaries changes no constexpr.
    for launch in launches:
        launch.pop("max_seqlen_q")
    assert all(launch == launches[0] for launch in launches)


@pytest.mark.parametrize("is_causal", [False, True])
def test_mla_prefill_gluon_fp8_reuses_kernel_across_query_sizes(
    device, require, monkeypatch, is_causal
):
    dtype = torch.float8_e4m3fn
    require("attention", "mla_prefill", "gluon", dtype, "q")
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla import prefill

    original = prefill._mla_prefill_kernel
    compiled = set()

    class RecordKernel:
        def __getitem__(self, grid):
            assert grid == (512,)

            def launch(*args, **kwargs):
                kernel = original[grid](*args, **kwargs)
                compiled.add(kernel.hash)
                return kernel

            return launch

    monkeypatch.setattr(prefill, "_mla_prefill_kernel", RecordKernel())
    k = torch.zeros((128, 12, 192), dtype=dtype, device=device)
    v = torch.zeros((128, 12, 128), dtype=dtype, device=device)
    cu_kv = torch.tensor([0, 128], dtype=torch.int32, device=device)
    for tokens in (144, 272, 512, 2048, 8192, 16384):
        q = torch.zeros((tokens, 12, 192), dtype=dtype, device=device)
        cu_q = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        prefill.gluon_mla_prefill_gfx950(
            q, k, v, cu_q, cu_kv, tokens, 128, 192**-0.5, is_causal=is_causal
        )
    assert len(compiled) == 1


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("num_heads", [12, 128])
@pytest.mark.parametrize(
    "q_lens", [(128,), (257,), (8193,), (11009,), (0, 1, 2048), (257,) * 40]
)
def test_mla_prefill_gluon_fp8_scheduler_coverage(
    device, require, is_causal, num_heads, q_lens
):
    dtype = torch.float8_e4m3fn
    require("attention", "mla_prefill", "gluon", dtype, "q")
    q = torch.zeros((sum(q_lens), num_heads, 192), dtype=dtype, device=device)
    kv_len = 128
    k = torch.zeros((len(q_lens) * kv_len, 1, 192), dtype=dtype, device=device)
    values = torch.arange(kv_len, device=device).float() / kv_len
    v = values.repeat(len(q_lens))[:, None, None].expand(-1, 1, 128).to(dtype)
    cu_q = torch.tensor((0, *q_lens), dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    cu_kv = torch.arange(len(q_lens) + 1, dtype=torch.int32, device=device) * kv_len
    destination = torch.full(q.shape[:2] + (128,), float("nan"), device=device)
    output, lse = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=kv_len,
        softmax_scale=192**-0.5,
        is_causal=is_causal,
        return_lse=True,
        solution="gluon",
        out=destination,
    )
    # Uniform logits give exact prefix means. Short KV spans keep this scheduler
    # test inexpensive while covering multiple query cycles and batch groups.
    visible = torch.cat(
        [
            (
                (
                    torch.arange(length, device=device) + max(kv_len - length, 0) + 1
                ).clamp(max=kv_len)
                if is_causal
                else torch.full((length,), kv_len, device=device)
            )
            for length in q_lens
        ]
    ).long()
    expected = v[:kv_len, 0, 0].float().cumsum(0)[visible - 1] / visible
    torch.testing.assert_close(
        output, expected[:, None, None].expand_as(output), rtol=6e-3, atol=6e-3
    )
    torch.testing.assert_close(
        lse, visible.float().log()[:, None].expand_as(lse), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_len,kv_len", [(129, 65), (65, 129), (257, 193), (65, 0)])
def test_mla_prefill_gluon_fp8_causal_cutoff(device, require, dtype, q_len, kv_len):
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
        solution="gluon",
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


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("changed_row", [None, 13, 45, 254])
@pytest.mark.parametrize(
    "kv_len", [1, 64, 65, 128, 129, 193, 256, 257, 321, 512, 513, 576]
)
def test_mla_prefill_gluon_fp8_online_max(device, require, dtype, changed_row, kv_len):
    require("attention", "mla_prefill", "gluon", dtype, "q")
    q = torch.zeros((255, 2, 192), dtype=torch.bfloat16, device=device)
    k = torch.zeros((576, 2, 192), dtype=torch.bfloat16, device=device)
    v = torch.empty((576, 2, 128), dtype=torch.bfloat16, device=device)
    q[:, :, 0] = 1.0
    if changed_row is not None:
        # Non-leading rows in multiple waves, including the final active row.
        q[changed_row, 0, 0] = -1.0
    # Distinct values and changing maxima span two complete V-ring wraps.
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
        solution="gluon",
    )
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * (192**-0.5)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        lse, scores.logsumexp(-1).transpose(0, 1), rtol=2e-5, atol=2e-5
    )
