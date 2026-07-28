"""Unit tests for the vendored MiniMax-M3 fused qk-norm + RoPE + KV/index-insert
CUDA kernel.

Validates against an independent torch golden (Gemma RMSNorm with the raw weight
-- the kernel adds ``1 + w`` internally -- then partial-NeoX RoPE) across:
  * norm+RoPE only (no cache insert),
  * K/V + index-K insert into TokenSpeed's separate flat slot-indexed buffers,
  * fp8 KV cache + fp8 index cache / index_q output.

Requires an NVIDIA GPU and the built ``minimax_m3_fused`` kernel module.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.minimax_m3_fused import (
    fused_qknorm_rope_kv_insert,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="minimax_m3_fused kernel is NVIDIA-only",
)

HD, RD, EPS = 128, 64, 1e-6
NQ, NKV, NIQ = 8, 2, 2
T, NUM_SLOTS = 37, 64
ROW = (NQ + 2 * NKV + NIQ + 1) * HD
E4M3 = torch.float8_e4m3fn
_SLICES = [("q", NQ), ("k", NKV), ("v", NKV), ("iq", NIQ), ("ik", 1)]


def _golden_norm_rope(x, w_raw, cos, sin):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + EPS) * (1.0 + w_raw.float())
    rot, pas = xn[..., :RD], xn[..., RD:]
    c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    x1, x2 = rot[..., : RD // 2], rot[..., RD // 2 :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, pas], dim=-1)


def _slice(t, name, n):
    o = 0
    for nm, nn in _SLICES:
        if nm == name:
            return t[:, o : o + nn * HD].view(T, nn, HD)
        o += nn * HD
    raise KeyError(name)


def _fixtures(dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    qkv = torch.randn(T, ROW, device=dev, dtype=torch.bfloat16, generator=g)
    weights = {
        k: (torch.randn(HD, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        for k in ("q", "k", "iq", "ik")
    }
    positions = torch.arange(T, device=dev, dtype=torch.int64)
    inv = 1.0 / (5_000_000 ** (torch.arange(0, RD, 2, device=dev).float() / RD))
    ang = torch.arange(4096, device=dev).float()[:, None] * inv[None, :]
    csc = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).float()
    cos, sin = csc[positions, : RD // 2], csc[positions, RD // 2 :]
    return qkv, weights, positions, csc, cos, sin


def test_norm_rope_only():
    dev = "cuda"
    qkv, w, positions, csc, cos, sin = _fixtures(dev, 0)
    qkv_in = qkv.clone()
    q_out = torch.empty(T, NQ * HD, device=dev, dtype=torch.bfloat16)
    iq_out = torch.empty(T, NIQ * HD, device=dev, dtype=torch.bfloat16)

    fused_qknorm_rope_kv_insert(
        qkv,
        w["q"],
        w["k"],
        csc,
        positions,
        NQ,
        NKV,
        RD,
        EPS,
        index_q_norm_weight=w["iq"],
        index_k_norm_weight=w["ik"],
        num_index_heads=NIQ,
        q_out=q_out,
        index_q_out=iq_out,
    )

    g_q = _golden_norm_rope(_slice(qkv_in, "q", NQ), w["q"], cos, sin)
    g_k = _golden_norm_rope(_slice(qkv_in, "k", NKV), w["k"], cos, sin)
    g_iq = _golden_norm_rope(_slice(qkv_in, "iq", NIQ), w["iq"], cos, sin)
    g_ik = _golden_norm_rope(_slice(qkv_in, "ik", 1), w["ik"], cos, sin)

    torch.testing.assert_close(q_out.view(T, NQ, HD).float(), g_q, atol=2e-2, rtol=0)
    torch.testing.assert_close(_slice(qkv, "k", NKV).float(), g_k, atol=2e-2, rtol=0)
    torch.testing.assert_close(iq_out.view(T, NIQ, HD).float(), g_iq, atol=2e-2, rtol=0)
    torch.testing.assert_close(_slice(qkv, "ik", 1).float(), g_ik, atol=2e-2, rtol=0)
    # V is not inserted here, so it must be untouched.
    torch.testing.assert_close(
        _slice(qkv, "v", NKV), _slice(qkv_in, "v", NKV), atol=0, rtol=0
    )


def test_kv_and_index_insert_bf16():
    dev = "cuda"
    qkv, w, positions, csc, cos, sin = _fixtures(dev, 1)
    qkv_in = qkv.clone()
    slots = torch.randperm(NUM_SLOTS, device=dev)[:T].to(torch.int64)
    k_cache = torch.zeros(NUM_SLOTS, NKV, HD, device=dev, dtype=torch.bfloat16)
    v_cache = torch.zeros(NUM_SLOTS, NKV, HD, device=dev, dtype=torch.bfloat16)
    index_cache = torch.zeros(NUM_SLOTS, HD, device=dev, dtype=torch.bfloat16)

    fused_qknorm_rope_kv_insert(
        qkv,
        w["q"],
        w["k"],
        csc,
        positions,
        NQ,
        NKV,
        RD,
        EPS,
        index_q_norm_weight=w["iq"],
        index_k_norm_weight=w["ik"],
        num_index_heads=NIQ,
        slot_mapping=slots,
        k_cache=k_cache,
        v_cache=v_cache,
        index_cache=index_cache,
        block_size=1,
    )

    gk = torch.zeros_like(k_cache)
    gv = torch.zeros_like(v_cache)
    gidx = torch.zeros_like(index_cache)
    gk[slots] = _golden_norm_rope(_slice(qkv_in, "k", NKV), w["k"], cos, sin).to(
        torch.bfloat16
    )
    gv[slots] = _slice(qkv_in, "v", NKV)  # V inserted raw (no norm/rope)
    gidx[slots] = (
        _golden_norm_rope(_slice(qkv_in, "ik", 1), w["ik"], cos, sin)
        .to(torch.bfloat16)
        .view(T, HD)
    )

    torch.testing.assert_close(k_cache, gk, atol=2e-2, rtol=0)
    torch.testing.assert_close(v_cache, gv, atol=0, rtol=0)
    torch.testing.assert_close(index_cache, gidx, atol=2e-2, rtol=0)


def test_fp8_kv_and_index():
    dev = "cuda"
    qkv, w, positions, csc, cos, sin = _fixtures(dev, 2)
    qkv_in = qkv.clone()
    slots = torch.randperm(NUM_SLOTS, device=dev)[:T].to(torch.int64)
    k_cache = torch.zeros(NUM_SLOTS, NKV, HD, device=dev, dtype=torch.uint8)
    v_cache = torch.zeros(NUM_SLOTS, NKV, HD, device=dev, dtype=torch.uint8)
    index_cache = torch.zeros(NUM_SLOTS, HD, device=dev, dtype=E4M3)
    iq_out = torch.empty(T, NIQ * HD, device=dev, dtype=E4M3)

    fused_qknorm_rope_kv_insert(
        qkv,
        w["q"],
        w["k"],
        csc,
        positions,
        NQ,
        NKV,
        RD,
        EPS,
        index_q_norm_weight=w["iq"],
        index_k_norm_weight=w["ik"],
        num_index_heads=NIQ,
        slot_mapping=slots,
        k_cache=k_cache,
        v_cache=v_cache,
        index_cache=index_cache,
        index_q_out=iq_out,
        block_size=1,
        kv_cache_dtype="fp8_e4m3",
    )

    gk = torch.zeros(NUM_SLOTS, NKV, HD, device=dev, dtype=E4M3)
    gv = torch.zeros(NUM_SLOTS, NKV, HD, device=dev, dtype=E4M3)
    gidx = torch.zeros(NUM_SLOTS, HD, device=dev, dtype=E4M3)
    gk[slots] = _golden_norm_rope(_slice(qkv_in, "k", NKV), w["k"], cos, sin).to(E4M3)
    gv[slots] = _slice(qkv_in, "v", NKV).to(E4M3)
    gidx[slots] = (
        _golden_norm_rope(_slice(qkv_in, "ik", 1), w["ik"], cos, sin)
        .to(E4M3)
        .view(T, HD)
    )
    g_iq = _golden_norm_rope(_slice(qkv_in, "iq", NIQ), w["iq"], cos, sin).to(E4M3)

    # Compare decoded fp8 (identity scale) -- expected bit-identical rounding.
    torch.testing.assert_close(k_cache.view(E4M3).float(), gk.float(), atol=0, rtol=0)
    torch.testing.assert_close(v_cache.view(E4M3).float(), gv.float(), atol=0, rtol=0)
    torch.testing.assert_close(index_cache.float(), gidx.float(), atol=0, rtol=0)
    torch.testing.assert_close(
        iq_out.view(T, NIQ, HD).float(), g_iq.float(), atol=0, rtol=0
    )
