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

"""Standalone kernel tests for Hy4-flash ops (no runtime imports)."""

from __future__ import annotations

import math
from test.utils import assert_no_triton_compile

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.kpool.hy4_compress import (
    _hy4_index_k_quantize_scatter_kernel,
    apply_interleaved_rope_last_dim,
    compute_hy4_mrope_cos_sin,
    hy4_pool_group_keys,
    hy4_quantize_and_store_index_k,
)
from tokenspeed_kernel.ops.attention.mla._triton.decode import mla_decode_fwd
from tokenspeed_kernel.ops.attention.sink import (
    _apply_attention_sink_kernel,
    apply_attention_sink,
)
from tokenspeed_kernel.ops.kvcache.index_k_scale import (
    decode_index_k_scale,
    encode_index_k_scale,
    index_k_scale_dtype,
    round_index_k_scale,
)
from tokenspeed_kernel.ops.residual.ihc import (
    _ihc_head_reduce_kernel,
    _ihc_post_kernel,
    _ihc_pre_reduce_kernel,
    ihc_head,
    ihc_post,
    ihc_pre,
)

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _fp8_e4m3fn_step_distance(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Return max representable E4M3FN step distance between two FP8 tensors."""

    def _rank(t: torch.Tensor) -> torch.Tensor:
        bits = t.contiguous().view(torch.uint8)
        mag = (bits & 0x7F).to(torch.int16)
        assert not bool((mag == 0x7F).any()), "unexpected FP8 NaN"
        return torch.where((bits & 0x80) != 0, 0x7E - mag, 0x7E + mag)

    return int((_rank(actual) - _rank(expected)).abs().max().item())


def _ref_attention_sink_fp64(
    output: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    scale = torch.sigmoid(lse.double() - sink.double().unsqueeze(0)).unsqueeze(-1)
    return (output.double() * scale).to(output.dtype)


def _ref_ihc_pre_fp64(
    streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_magnitude: float = 2.0,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    t, n, d = streams.shape
    flat = streams.reshape(t, n * d).double()
    inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(flat, fn.double()) * inv_rms
    pre_logits, post_logits = mixes.split(n, dim=-1)
    pre_b, post_b = base.double().split(n, dim=-1)
    pre_s, post_s = scale.double().unbind(0)
    pre = torch.sigmoid(pre_logits * pre_s + pre_b) + hc_eps
    post = hc_magnitude * torch.sigmoid(post_logits * post_s + post_b) + hc_eps
    reduced = (pre.unsqueeze(-1) * streams.double()).sum(dim=1)
    return post.float(), reduced.to(streams.dtype)


def _ref_ihc_head_fp64(
    streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    t, n, d = streams.shape
    flat = streams.reshape(t, n * d).double()
    inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(flat, fn.double()) * inv_rms
    pre = torch.sigmoid(mixes * scale.double() + base.double()) + hc_eps
    return (pre.unsqueeze(-1) * streams.double()).sum(dim=1).to(streams.dtype)


def test_hy4_mrope_and_pool_group_keys_cpu() -> None:
    torch.manual_seed(0)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0))
    pos_3d = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2], [0, 1, 0, 1]], dtype=torch.long)
    cos, sin = compute_hy4_mrope_cos_sin(pos_3d, inv_freq, (12, 10, 10))
    assert cos.shape == (4, 32)
    assert sin.shape == (4, 32)
    x = torch.randn(4, 2, 64, dtype=torch.bfloat16)
    rotated = apply_interleaved_rope_last_dim(x, cos.unsqueeze(1), sin.unsqueeze(1))
    assert rotated.shape == x.shape
    assert rotated.dtype == x.dtype
    # Rotation preserves per-vector L2 norm up to BF16 rounding.
    torch.testing.assert_close(
        rotated.float().norm(dim=-1),
        x.float().norm(dim=-1),
        rtol=2e-2,
        atol=2e-2,
    )

    grouped = torch.randn(6, 4, 128, dtype=torch.bfloat16)
    kv_w = torch.randn(128, 128, dtype=torch.float32) * 0.1
    kg_w = torch.randn(128, 128, dtype=torch.float32) * 0.1
    ape = torch.randn(4, 128, dtype=torch.float32) * 0.05
    norm_w = torch.ones(128, dtype=torch.bfloat16)
    pooled = hy4_pool_group_keys(grouped, kv_w, kg_w, ape, norm_w, eps=1e-5)
    assert pooled.shape == (6, 128)
    assert pooled.dtype == torch.bfloat16
    assert torch.isfinite(pooled).all()
    gf = grouped.float()
    vals = F.linear(gf, kv_w)
    gates = F.linear(gf, kg_w) + ape.unsqueeze(0)
    raw = (vals * F.softmax(gates, dim=1)).sum(dim=1)
    ref = norm_w.float() * (
        raw * torch.rsqrt(raw.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    )
    torch.testing.assert_close(pooled.float(), ref, rtol=2e-2, atol=2e-2)


def test_apply_attention_sink_matches_augmented_softmax_cpu() -> None:
    torch.manual_seed(1)
    q_len, kv_len, heads, dim = 6, 12, 4, 16
    scores = torch.randn(q_len, heads, kv_len)
    values = torch.randn(heads, kv_len, dim)
    sinks = torch.randn(heads)

    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    plain_out = torch.einsum("qhk,hkd->qhd", probs, values)
    rescaled = apply_attention_sink(plain_out.clone(), lse, sinks, inplace=False)

    aug_scores = torch.cat(
        [scores, sinks.view(1, heads, 1).expand(q_len, heads, 1)], dim=-1
    )
    aug_probs = torch.softmax(aug_scores, dim=-1)[..., :-1]
    ref_out = torch.einsum("qhk,hkd->qhd", aug_probs, values)
    torch.testing.assert_close(rescaled, ref_out, rtol=1e-5, atol=1e-5)


def test_hy4_quantize_and_store_index_k_strided_arena_view() -> None:
    """Verify strided paged arena views with storage_offset preserve decoded scale/value."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_pages, num_layers, rows_per_page, head_dim = 8, 3, 16, 128
    row_bytes = head_dim + 4
    arena = torch.zeros(
        (num_pages, num_layers, rows_per_page, row_bytes),
        dtype=torch.uint8,
        device=device,
    )
    buf = arena[:, 1]
    page_stride = buf.stride(0)
    values = torch.as_strided(
        buf, (num_pages, rows_per_page, head_dim), (page_stride, head_dim, 1)
    ).view(torch.float8_e4m3fn)
    scales = torch.as_strided(
        buf,
        (num_pages, rows_per_page, 4),
        (page_stride, 4, 1),
        buf.storage_offset() + rows_per_page * head_dim,
    ).view(index_k_scale_dtype())

    keys = torch.randn(24, head_dim, dtype=torch.bfloat16, device=device)
    slots = torch.tensor(
        [16 + i for i in range(12)] + [64 + i for i in range(12)],
        dtype=torch.int64,
        device=device,
    )
    hy4_quantize_and_store_index_k(keys, slots, values, scales)

    pages = slots // rows_per_page
    rows = slots % rows_per_page
    decoded_scales = decode_index_k_scale(scales[pages, rows])
    dequantized = values[pages, rows].float() * decoded_scales
    expected_scales = (
        keys.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448.0
    )
    assert torch.isfinite(decoded_scales).all()
    assert (decoded_scales >= expected_scales * 0.99).all()
    assert (decoded_scales <= expected_scales * 2.01).all()
    torch.testing.assert_close(dequantized, keys.float(), atol=0.25, rtol=0.1)


@_CUDA
def test_hy4_quantize_and_store_index_k_cuda_bits_and_no_recompile() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    scale_dtype = index_k_scale_dtype()
    rows_per_page = 16
    head_dim = 128

    def run_once(num_rows: int, num_pages: int) -> None:
        keys = (
            torch.randn(num_rows, head_dim, dtype=torch.bfloat16, device=device) * 0.6
        )
        slots = torch.arange(num_rows, dtype=torch.int32, device=device)
        if num_rows >= 3:
            slots[1] = -1
            slots[-1] = num_pages * rows_per_page + 5  # OOB must be skipped
        values = torch.zeros(
            (num_pages, rows_per_page, head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        scales = torch.zeros(
            (num_pages, rows_per_page, 1), dtype=scale_dtype, device=device
        )
        hy4_quantize_and_store_index_k(keys, slots, values, scales)

        # Verify CPU and GPU agree on raw scale encoding (exact for UE8M0 int32,
        # <= 1 ULP for FP32 SM90 division; catches any int-to-float bitcast bug).
        cpu_values = torch.zeros_like(values, device="cpu")
        cpu_scales = torch.zeros_like(scales, device="cpu")
        hy4_quantize_and_store_index_k(keys.cpu(), slots.cpu(), cpu_values, cpu_scales)
        if scale_dtype == torch.int32:
            assert torch.equal(
                scales.view(torch.int32).cpu(), cpu_scales.view(torch.int32)
            )
        else:
            torch.testing.assert_close(scales.cpu(), cpu_scales, rtol=1e-5, atol=1e-7)
            ulp_diff = (
                scales.view(torch.int32).cpu() - cpu_scales.view(torch.int32)
            ).abs()
            assert int(ulp_diff.max().item()) <= 1
        assert _fp8_e4m3fn_step_distance(values.cpu(), cpu_values) <= 1

        # Check active row 0 scale matches round_index_k_scale(absmax/448) within 1 ULP.
        expected_scale = round_index_k_scale(
            keys[0:1].float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448.0
        )
        expected_word = encode_index_k_scale(expected_scale).view(torch.int32)[0, 0]
        assert (
            abs(
                int(scales.view(torch.int32)[0, 0, 0].item())
                - int(expected_word.item())
            )
            <= 1
        )
        # Masked negative slot must remain untouched.
        if num_rows >= 3:
            assert int(scales.view(torch.int32)[0, 1, 0].item()) == 0

    run_once(7, 4)
    with assert_no_triton_compile(_hy4_index_k_quantize_scatter_kernel):
        run_once(16, 4)
        run_once(29, 8)
        run_once(64, 16)


@_CUDA
def test_hy4_quantize_and_store_index_k_int64_large_page_offset() -> None:
    """Verify page * stride_iv_page stays in int64 past 2^31 bytes."""
    device = torch.device("cuda")
    rows_per_page = 16
    head_dim = 128
    target_page = (1 << 20) + 17  # 1,048,593
    page_stride_bytes = rows_per_page * head_dim  # 2048; target_page * 2048 > 2^31

    free_bytes, _ = torch.cuda.mem_get_info(device)
    needed_bytes = (target_page + 1) * page_stride_bytes + 128 * 1024 * 1024
    if free_bytes < needed_bytes:
        pytest.skip(
            f"insufficient free VRAM for >2GiB int64 offset test ({free_bytes})"
        )

    big_vals = torch.empty(
        (target_page + 1, rows_per_page, head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    big_vals[target_page].zero_()
    big_scales = torch.empty(
        (target_page + 1, rows_per_page, 1),
        dtype=index_k_scale_dtype(),
        device=device,
    )
    big_scales[target_page].zero_()
    assert target_page * big_vals.stride(0) > (1 << 31)

    keys = torch.randn((1, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    slot = torch.tensor(
        [target_page * rows_per_page + 3], dtype=torch.int64, device=device
    )
    hy4_quantize_and_store_index_k(keys, slot, big_vals, big_scales)

    expected_scale = round_index_k_scale(
        keys.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448.0
    )
    expected_word = encode_index_k_scale(expected_scale).view(torch.int32)[0, 0]
    assert (
        abs(
            int(big_scales.view(torch.int32)[target_page, 3, 0].item())
            - int(expected_word.item())
        )
        <= 1
    )
    del big_vals, big_scales


@_CUDA
def test_apply_attention_sink_parity_and_no_recompile() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")

    def check(num_tokens: int, *, inplace: bool) -> None:
        out = torch.randn(num_tokens, 32, 128, dtype=torch.bfloat16, device=device)
        lse = torch.randn(num_tokens, 32, dtype=torch.float32, device=device) * 4.0
        lse[0, 0] = float("-inf")
        sink = torch.randn(32, dtype=torch.float32, device=device)
        expected = _ref_attention_sink_fp64(out, lse, sink)
        inp = out.clone() if inplace else out
        got = apply_attention_sink(inp, lse, sink, inplace=inplace)
        if inplace:
            assert got.data_ptr() == inp.data_ptr()
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
        assert float(got[0, 0].abs().max().item()) == 0.0

    check(5, inplace=False)
    with assert_no_triton_compile(_apply_attention_sink_kernel):
        check(16, inplace=True)
        check(33, inplace=False)
        check(128, inplace=True)


@_CUDA
def test_ihc_kernels_parity_and_no_recompile() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    hc_mult = 4
    hidden = 512

    fn_pre = (
        torch.randn(2 * hc_mult, hc_mult * hidden, dtype=torch.float32, device=device)
        * 0.02
    )
    scale_pre = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device)
    base_pre = torch.randn(2 * hc_mult, dtype=torch.float32, device=device) * 0.1

    fn_head = (
        torch.randn(hc_mult, hc_mult * hidden, dtype=torch.float32, device=device)
        * 0.02
    )
    scale_head = torch.tensor([0.15], dtype=torch.float32, device=device)
    base_head = torch.randn(hc_mult, dtype=torch.float32, device=device) * 0.1

    def check(num_tokens: int) -> None:
        streams = torch.randn(
            num_tokens, hc_mult, hidden, dtype=torch.bfloat16, device=device
        )
        branch = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)

        post_got, reduced_got = ihc_pre(streams, fn_pre, scale_pre, base_pre)
        post_ref, reduced_ref = _ref_ihc_pre_fp64(streams, fn_pre, scale_pre, base_pre)
        torch.testing.assert_close(post_got, post_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(reduced_got, reduced_ref, rtol=2e-2, atol=2e-2)

        updated_got = ihc_post(streams, branch, post_got)
        updated_ref = (
            post_ref.double().unsqueeze(-1) * branch.double().unsqueeze(1)
            + streams.double()
        ).to(streams.dtype)
        torch.testing.assert_close(updated_got, updated_ref, rtol=2e-2, atol=2e-2)

        head_got = ihc_head(updated_got, fn_head, scale_head, base_head)
        head_ref = _ref_ihc_head_fp64(updated_got, fn_head, scale_head, base_head)
        torch.testing.assert_close(head_got, head_ref, rtol=2e-2, atol=2e-2)

    check(3)
    with assert_no_triton_compile(
        _ihc_pre_reduce_kernel, _ihc_post_kernel, _ihc_head_reduce_kernel
    ):
        check(16)
        check(31)
        check(64)


@_CUDA
@pytest.mark.parametrize("q_len,window_left", [(1, -1), (6, -1), (6, 31)])
def test_triton_mla_decode_multi_q_len_and_swa_parity(
    q_len: int, window_left: int
) -> None:
    """Verify multi-token verify (q_len > 1) and SWA windowing in Triton MLA decode."""
    torch.manual_seed(19)
    device = torch.device("cuda")
    bs, num_heads, kv_lora_rank, rope_dim, page_size = 2, 8, 512, 64, 64
    num_pages = 8
    sm_scale = 1.0 / math.sqrt(128 + rope_dim)

    q = (
        torch.randn(
            bs,
            q_len,
            num_heads,
            kv_lora_rank + rope_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.2
    )
    kv_cache = (
        torch.randn(
            num_pages,
            page_size,
            1,
            kv_lora_rank + rope_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.2
    )
    page_table = torch.tensor(
        [[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.int32, device=device
    )
    cache_seqlens = torch.tensor([90, 140], dtype=torch.int32, device=device)

    out = torch.empty(
        (bs, q_len, num_heads, kv_lora_rank), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty((bs, q_len, num_heads), dtype=torch.float32, device=device)
    mla_decode_fwd(
        q,
        kv_cache,
        out,
        page_table,
        cache_seqlens,
        max_seqlen_k=140,
        qk_nope_head_dim=128,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        softmax_scale=sm_scale,
        lse=lse,
        window_left=window_left,
    )

    # Exact FP64 reference per (batch, q_idx)
    ref_out = torch.zeros_like(out, dtype=torch.float64)
    ref_lse = torch.zeros_like(lse, dtype=torch.float64)
    for b in range(bs):
        full_len = int(cache_seqlens[b].item())
        pages = page_table[b].long()
        flat_kv = (
            kv_cache[pages, :, 0, :]
            .reshape(-1, kv_lora_rank + rope_dim)[:full_len]
            .double()
        )
        for qi in range(q_len):
            q_cache_len = full_len - (q_len - 1 - qi)
            w_start = max(0, q_cache_len - 1 - window_left) if window_left >= 0 else 0
            k_slice = flat_kv[w_start:q_cache_len]  # [L, R+rope]
            v_slice = k_slice[:, :kv_lora_rank]  # [L, R]
            q_vec = q[b, qi].double()  # [H, R+rope]
            scores = (q_vec @ k_slice.T) * sm_scale  # [H, L]
            lse_row = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            ref_out[b, qi] = probs @ v_slice
            ref_lse[b, qi] = lse_row

    torch.testing.assert_close(out, ref_out.to(out.dtype), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, ref_lse.float(), rtol=1e-3, atol=1e-3)
