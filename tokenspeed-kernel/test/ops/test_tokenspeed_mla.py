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
import platform

import pytest
import torch
from tokenspeed_kernel.ops.attention import tokenspeed_mla as kernel_mla
from tokenspeed_kernel.ops.attention.tokenspeed_mla import (
    mla_kv_pack_quantize_fp8,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="tokenspeed_mla kernels are NVIDIA-only",
)

# K2.5 / DSv3 chunked-prefill shape with TP=4.
S = 256
H = 16
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128

# (is_causal, return_lse) variants exposed by tokenspeed_mla_prefill.
PREFILL_BINARY_VARIANT_FLAGS = [
    (causal, lse) for causal in (False, True) for lse in (False, True)
]

# (seq_lens_q, seq_lens_k, h_q, h_k) — varlen problem layouts.
PREFILL_BINARY_SHAPE_CASES = [
    ((64, 128, 32), (64, 128, 32), 8, 8),
    ((128, 256, 96), (128, 256, 96), 128, 128),
    ((990,), (990,), 128, 128),
    ((1024, 1139), (1024, 1139), 8, 8),
]


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _make_kv_slice_inputs(device: str, dtype: torch.dtype = torch.bfloat16):
    """Mirror the deepseek_v3.py call site: k_nope and v are slice views of
    a packed kv tensor produced by kv_b_proj."""
    torch.manual_seed(0)
    kv = torch.randn(S, H, QK_NOPE + V_HEAD, device=device, dtype=dtype)
    k_nope = kv[..., :QK_NOPE]
    v = kv[..., QK_NOPE:]
    k_pe = torch.randn(S, 1, QK_ROPE, device=device, dtype=dtype)
    return k_nope, k_pe, v


def _host_arch() -> str:
    return {
        "amd64": "x86_64",
        "arm64": "aarch64",
        "x64": "x86_64",
    }.get(platform.machine().lower(), platform.machine().lower())


def _prefill_shape_id(case) -> str:
    seq_lens_q, seq_lens_k, h_q, h_k = case
    return f"sQ{sum(seq_lens_q)}_sK{sum(seq_lens_k)}_hq{h_q}_hk{h_k}"


def _prefill_variant_id(flags: tuple[bool, bool]) -> str:
    causal, lse = flags
    return ("causal" if causal else "nocausal") + ("_lse" if lse else "")


def _require_mla_binary_prefill():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for tokenspeed-mla binary prefill")

    import tokenspeed_mla.fmha_binary as fmha_binary

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    expected_suffix = f"sm_{props.major}{props.minor}a_{_host_arch()}.so"
    so_path = fmha_binary._resolve_so_path()
    if not so_path.exists():
        pytest.skip(f"tokenspeed-mla binary prefill SO not found: {so_path}")
    assert so_path.name.endswith(expected_suffix)
    return fmha_binary, so_path


def _reference(k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype):
    """Pure-PyTorch reference: broadcast k_pe across heads, cat, scale, cast."""
    k_pe_2d = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
    k_pe_full = k_pe_2d.unsqueeze(1).expand(-1, k_nope.shape[1], -1)
    k_bf16 = torch.cat([k_nope, k_pe_full], dim=-1)
    k_fp8 = (k_bf16.float() * k_scale_inv).to(fp8_dtype)
    v_fp8 = (v.float() * v_scale_inv).to(fp8_dtype)
    return k_fp8, v_fp8


def _prefill_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens_q: tuple[int, ...],
    seq_lens_k: tuple[int, ...],
    softmax_scale: float,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    lses = []
    q_offset = 0
    k_offset = 0
    for cur_s_q, cur_s_k in zip(seq_lens_q, seq_lens_k):
        cur_q = query[q_offset : q_offset + cur_s_q]
        cur_k = key[k_offset : k_offset + cur_s_k]
        cur_v = value[k_offset : k_offset + cur_s_k]
        scores = (
            torch.einsum("qhd,khd->qkh", cur_q.float(), cur_k.float()) * softmax_scale
        )
        if is_causal:
            q_idx = torch.arange(cur_s_q, device=query.device).view(-1, 1)
            k_idx = torch.arange(cur_s_k, device=key.device).view(1, -1)
            offset = cur_s_k - cur_s_q
            mask = k_idx > q_idx + offset
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
        probs = torch.softmax(scores, dim=1)
        outputs.append(torch.einsum("qkh,khd->qhd", probs, cur_v.float()))
        lses.append(torch.logsumexp(scores, dim=1) * math.log2(math.e))
        q_offset += cur_s_q
        k_offset += cur_s_k
    return torch.cat(outputs, dim=0).to(torch.bfloat16), torch.cat(lses, dim=0)


def test_binary_prefill_so_loads() -> None:
    fmha_binary, so_path = _require_mla_binary_prefill()

    module = fmha_binary._load_module(str(so_path))

    assert getattr(module, fmha_binary._FUNC_NAMES[(False, False)], None) is not None
    assert fmha_binary.has_binary_prefill()


@pytest.mark.parametrize(
    "shape_case",
    PREFILL_BINARY_SHAPE_CASES,
    ids=[_prefill_shape_id(case) for case in PREFILL_BINARY_SHAPE_CASES],
)
@pytest.mark.parametrize(
    "variant_flags",
    PREFILL_BINARY_VARIANT_FLAGS,
    ids=[_prefill_variant_id(flags) for flags in PREFILL_BINARY_VARIANT_FLAGS],
)
def test_kernel_tokenspeed_mla_prefill_binary_e2e(
    device: str, monkeypatch, shape_case, variant_flags: tuple[bool, bool]
) -> None:
    is_causal, return_lse = variant_flags

    _require_mla_binary_prefill()

    import tokenspeed_mla.mla_prefill as mla_prefill

    monkeypatch.setattr(mla_prefill, "_PREFILL_BACKEND_ENV", "binary")
    mla_prefill._resolve_backend.cache_clear()

    seq_lens_q, seq_lens_k, h_q, h_k = shape_case
    total_q = sum(seq_lens_q)
    total_k = sum(seq_lens_k)
    cum_seq_lens_q = torch.tensor(
        [0, *torch.tensor(seq_lens_q, dtype=torch.int32).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    cum_seq_lens_k = torch.tensor(
        [0, *torch.tensor(seq_lens_k, dtype=torch.int32).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    torch.manual_seed(3)
    query = torch.randn(
        total_q,
        h_q,
        QK_NOPE + QK_ROPE,
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    key = torch.randn(
        total_k,
        h_k,
        QK_NOPE + QK_ROPE,
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    value = torch.randn(
        total_k,
        h_k,
        V_HEAD,
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    softmax_scale = 1.0 / math.sqrt(QK_NOPE + QK_ROPE)

    try:
        actual = kernel_mla.tokenspeed_mla_prefill(
            query,
            key,
            value,
            torch.tensor(seq_lens_k, device=device, dtype=torch.int32),
            cum_seq_lens_k,
            max(seq_lens_k),
            batch_size=len(seq_lens_k),
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            return_lse=return_lse,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=max(seq_lens_q),
        )
    finally:
        mla_prefill._resolve_backend.cache_clear()
    torch.cuda.synchronize()

    expected, expected_lse = _prefill_reference(
        query,
        key,
        value,
        seq_lens_q,
        seq_lens_k,
        softmax_scale,
        is_causal=is_causal,
    )
    if return_lse:
        actual, actual_lse = actual
        assert actual_lse.shape == (total_q, h_q)
        assert actual_lse.dtype == torch.float32
    tolerance = 0.25 if is_causal else 0.1
    assert actual.shape == (total_q, h_q, V_HEAD)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=tolerance, rtol=1e-5
    )
    if return_lse:
        torch.testing.assert_close(
            actual_lse.float(), expected_lse.float(), atol=tolerance, rtol=1e-5
        )


def test_pure_cast_strided_inputs(device: str) -> None:
    """k_nope/v are non-contiguous slices, scale=1.0 — the prefill call site."""
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    assert not k_nope.is_contiguous()
    assert not v.is_contiguous()

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    torch.cuda.synchronize()

    assert k_out.shape == (S, H, QK_NOPE + QK_ROPE)
    assert v_out.shape == (S, H, V_HEAD)
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


@pytest.mark.parametrize(
    "page_size,seq_len,batch_size,num_heads,return_lse,"
    "expected_split,expected_k_tiles,expected_tail_tile",
    [
        (64, 5, 1, 16, True, None, None, None),
        (64, 133, 1, 16, True, None, None, None),
        (64, 1982, 5, 12, False, 14, 16, 2),
        (64, 2246, 12, 12, False, 6, 18, 3),
        (32, 5, 1, 16, True, None, None, None),
        (32, 133, 1, 16, True, None, None, None),
    ],
)
def test_fp8_decode_ignores_dirty_tail_beyond_sequence_length(
    device: str,
    page_size: int,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    return_lse: bool,
    expected_split: int | None,
    expected_k_tiles: int | None,
    expected_tail_tile: int | None,
) -> None:
    """Reused page bytes past seq_len must not contaminate decode output."""
    import cutlass
    from tokenspeed_mla.mla_decode import _get_split_kv_and_workspace_size
    from tokenspeed_mla.mla_decode_fp8 import (
        BlackwellMultiHeadLatentAttentionForwardFP8,
    )
    from tokenspeed_mla.mla_helpers import select_mla_decode_tilers
    from tokenspeed_mla.utils import get_num_sm

    if not current_platform().is_blackwell:
        pytest.skip("tokenspeed_mla FP8 decode requires Blackwell")

    query_len = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    compute_capability = torch.cuda.get_device_capability()
    mma_qk_tiler_mn, mma_pv_tiler_mn = select_mla_decode_tilers(
        num_heads,
        query_len,
        is_fp8=True,
        compute_capability=compute_capability,
    )
    qk_tile_size = mma_qk_tiler_mn[1]
    if expected_split is not None:
        if compute_capability != (10, 3):
            pytest.skip(
                "SM103 production split geometry is not portable to SM100; "
                "the same H=12 case selects M64 there and is not verified here"
            )
        assert mma_qk_tiler_mn == (128, 128)
        assert mma_pv_tiler_mn == (128, 256)

        kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
            acc_dtype=cutlass.Float32,
            lse_dtype=cutlass.Float32,
            mma_qk_tiler_mn=mma_qk_tiler_mn,
            mma_pv_tiler_mn=mma_pv_tiler_mn,
            max_active_clusters=1,
            page_size=page_size,
            skip_correction_threshold=0.0,
            is_persistent=False,
            is_var_seq=True,
            is_var_split_kv=False,
            is_causal=True,
            num_heads=num_heads,
            seq_len_q=query_len,
        )
        assert kernel.cluster_shape_mnk == (2, 1, 1)
        assert kernel.use_2cta_instrs

        split_kv, _ = _get_split_kv_and_workspace_size(
            batch_size,
            query_len,
            num_heads,
            kv_lora_rank,
            get_num_sm(device),
            32768,
            torch.float8_e4m3fn,
            mma_qk_tiler_mn,
        )
        num_k_tiles = math.ceil(seq_len / qk_tile_size)
        k_tiles_per_split = math.ceil(num_k_tiles / split_kv)
        tail_tile_in_split = (num_k_tiles - 1) % k_tiles_per_split + 1
        assert split_kv == expected_split
        assert num_k_tiles == expected_k_tiles
        assert k_tiles_per_split == expected_tail_tile
        assert tail_tile_in_split == expected_tail_tile

        # On SM100, H=12 and q_len=1 select the M64/1CTA family. This B300
        # SM103 test intentionally does not claim runtime coverage for M64.

    num_kernel_pages = math.ceil(seq_len / qk_tile_size) * qk_tile_size // page_size

    torch.manual_seed(7)
    query = torch.randn(
        batch_size,
        query_len,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    clean_cache = torch.zeros(
        num_kernel_pages + 1,
        page_size,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    clean_cache[1:].view(-1, kv_lora_rank + qk_rope_head_dim)[:seq_len] = torch.randn(
        seq_len,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    dirty_cache = clean_cache.clone()
    dirty_cache[1:].view(torch.uint8).view(-1, kv_lora_rank + qk_rope_head_dim)[
        seq_len:, :kv_lora_rank
    ].fill_(0x7F)

    block_tables = (
        torch.arange(1, num_kernel_pages + 1, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    workspace = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.int8)
    decode_kwargs = dict(
        workspace_buffer=workspace,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=32768,
        softmax_scale=1.0 / math.sqrt(kv_lora_rank + qk_rope_head_dim),
        return_lse=return_lse,
    )

    expected_result = kernel_mla.tokenspeed_mla_decode(
        query, clean_cache, **decode_kwargs
    )
    if return_lse:
        expected, expected_lse = expected_result
    else:
        expected = expected_result
        expected_lse = None

    valid_kv = (
        clean_cache[1:].view(-1, kv_lora_rank + qk_rope_head_dim)[:seq_len].float()
    )
    reference_scores = (
        torch.einsum("hd,kd->hk", query[0, 0].float(), valid_kv)
        * decode_kwargs["softmax_scale"]
    )
    reference_probs = torch.softmax(reference_scores, dim=-1)
    reference = torch.einsum("hk,kd->hd", reference_probs, valid_kv[:, :kv_lora_rank])
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(expected[0, 0].float(), reference, atol=0.13, rtol=1e-5)
    for _ in range(16):
        actual_result = kernel_mla.tokenspeed_mla_decode(
            query, dirty_cache, **decode_kwargs
        )
        if return_lse:
            actual, actual_lse = actual_result
        else:
            actual = actual_result
            actual_lse = None
        torch.cuda.synchronize()

        if return_lse:
            assert torch.isfinite(actual_lse).all()
            torch.testing.assert_close(actual_lse, expected_lse, atol=0, rtol=0)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_scaled_independent_k_v(device: str) -> None:
    """k and v use different scales; output reflects each independently."""
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    k_scale_inv, v_scale_inv = 0.5, 1.7

    k_ref, v_ref = _reference(
        k_nope, k_pe, v, k_scale_inv, v_scale_inv, torch.float8_e4m3fn
    )
    k_out, v_out = mla_kv_pack_quantize_fp8(
        k_nope, k_pe, v, k_scale_inv=k_scale_inv, v_scale_inv=v_scale_inv
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_k_pe_2d_and_3d_equivalent(device: str) -> None:
    """k_pe is accepted as both [s, 1, rope] and [s, rope]; same output."""
    k_nope, k_pe_3d, v = _make_kv_slice_inputs(device)
    k_pe_2d = k_pe_3d.squeeze(1)

    k_3d, v_3d = mla_kv_pack_quantize_fp8(k_nope, k_pe_3d, v)
    k_2d, v_2d = mla_kv_pack_quantize_fp8(k_nope, k_pe_2d, v)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_3d, k_2d)
    assert _bitwise_equal(v_3d, v_2d)


def test_contiguous_inputs(device: str) -> None:
    """Standalone (non-slice) k_nope, v inputs also work."""
    torch.manual_seed(1)
    k_nope = torch.randn(S, H, QK_NOPE, device=device, dtype=torch.bfloat16)
    v = torch.randn(S, H, V_HEAD, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(S, 1, QK_ROPE, device=device, dtype=torch.bfloat16)
    assert k_nope.is_contiguous()
    assert v.is_contiguous()

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_fp16_input(device: str) -> None:
    torch.manual_seed(2)
    kv = torch.randn(S, H, QK_NOPE + V_HEAD, device=device, dtype=torch.float16)
    k_nope = kv[..., :QK_NOPE]
    v = kv[..., QK_NOPE:]
    k_pe = torch.randn(S, 1, QK_ROPE, device=device, dtype=torch.float16)

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_e5m2_output(device: str) -> None:
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e5m2)
    k_out, v_out = mla_kv_pack_quantize_fp8(
        k_nope, k_pe, v, fp8_dtype=torch.float8_e5m2
    )
    torch.cuda.synchronize()

    assert k_out.dtype == torch.float8_e5m2
    assert v_out.dtype == torch.float8_e5m2
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


def test_preallocated_outputs(device: str) -> None:
    k_nope, k_pe, v = _make_kv_slice_inputs(device)
    k_out = torch.empty(
        (S, H, QK_NOPE + QK_ROPE), dtype=torch.float8_e4m3fn, device=device
    )
    v_out = torch.empty((S, H, V_HEAD), dtype=torch.float8_e4m3fn, device=device)

    k_ret, v_ret = mla_kv_pack_quantize_fp8(k_nope, k_pe, v, k_out=k_out, v_out=v_out)
    torch.cuda.synchronize()

    assert k_ret.data_ptr() == k_out.data_ptr()
    assert v_ret.data_ptr() == v_out.data_ptr()

    k_ref, v_ref = _reference(k_nope, k_pe, v, 1.0, 1.0, torch.float8_e4m3fn)
    assert _bitwise_equal(k_out, k_ref)
    assert _bitwise_equal(v_out, v_ref)


@pytest.mark.parametrize("k_scale_inv,v_scale_inv", [(1.0, 1.0), (0.5, 1.7)])
def test_pdl_off_matches_pdl_on(
    device: str, k_scale_inv: float, v_scale_inv: float
) -> None:
    """PDL is a scheduling hint; output must be bitwise-identical regardless."""
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper+ (SM≥90)")
    k_nope, k_pe, v = _make_kv_slice_inputs(device)

    k_off, v_off = mla_kv_pack_quantize_fp8(
        k_nope,
        k_pe,
        v,
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        enable_pdl=False,
    )
    k_on, v_on = mla_kv_pack_quantize_fp8(
        k_nope,
        k_pe,
        v,
        k_scale_inv=k_scale_inv,
        v_scale_inv=v_scale_inv,
        enable_pdl=True,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(k_off, k_on)
    assert _bitwise_equal(v_off, v_on)
