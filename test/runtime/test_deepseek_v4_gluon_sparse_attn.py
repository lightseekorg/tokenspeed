from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.attention.deepseek_v4.gluon_sparse_attn import (
    dispatch,
)


def _inputs(tokens: int = 8192):
    q = torch.randn(tokens, 64, 512, dtype=torch.bfloat16)
    kv = torch.randn(1, tokens, 512, dtype=torch.bfloat16)
    indices = torch.zeros(tokens, 128, dtype=torch.int32)
    lens = torch.full((tokens,), 128, dtype=torch.int32)
    attn_sink = torch.zeros(64, dtype=torch.float32)
    return q, kv, indices, lens, attn_sink


def test_native_gluon_sparse_selected_attention_disabled(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_DSV4_SELECTED_ATTN_IMPL", raising=False)
    q, kv, indices, lens, attn_sink = _inputs()

    out = dispatch.native_gluon_sparse_selected_attention(
        q=q,
        kv=kv,
        indices=indices,
        lens=lens,
        attn_sink=attn_sink,
        softmax_scale=512**-0.5,
    )

    assert out is None


def test_native_gluon_sparse_selected_attention_shape_fallback(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DSV4_SELECTED_ATTN_IMPL", "gluon_sparse")
    q, kv, indices, lens, attn_sink = _inputs(tokens=1024)

    out = dispatch.native_gluon_sparse_selected_attention(
        q=q,
        kv=kv,
        indices=indices,
        lens=lens,
        attn_sink=attn_sink,
        softmax_scale=512**-0.5,
    )

    assert out is None


def test_native_gluon_sparse_selected_attention_calls_native_wrapper(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DSV4_SELECTED_ATTN_IMPL", "gluon_sparse")
    monkeypatch.setenv("TOKENSPEED_DSV4_GLUON_SPARSE_ATTN_MIN_TOKENS", "1")
    q, kv, indices, lens, attn_sink = _inputs(tokens=4)
    calls = {}

    def fake_sparse_attn(q4, kv3, sink, topk3, scale, *, topk_lens=None):
        calls["q_shape"] = tuple(q4.shape)
        calls["kv_shape"] = tuple(kv3.shape)
        calls["topk_shape"] = tuple(topk3.shape)
        calls["lens_shape"] = tuple(topk_lens.shape)
        calls["lens_dtype"] = topk_lens.dtype
        calls["scale"] = scale
        return torch.empty_like(q4)

    monkeypatch.setattr(dispatch, "load_native_gluon_sparse_attn", lambda: fake_sparse_attn)
    monkeypatch.setattr(dispatch, "_NATIVE_GLUON_SPARSE_ATTN_LOGGED", True)

    out = dispatch.native_gluon_sparse_selected_attention(
        q=q,
        kv=kv,
        indices=indices,
        lens=lens,
        attn_sink=attn_sink,
        softmax_scale=512**-0.5,
    )

    assert out is not None
    assert tuple(out.shape) == tuple(q.shape)
    assert calls == {
        "q_shape": (1, 4, 64, 512),
        "kv_shape": (1, 4, 512),
        "topk_shape": (1, 4, 128),
        "lens_shape": (1, 4),
        "lens_dtype": torch.int32,
        "scale": pytest.approx(512**-0.5),
    }


def test_native_gluon_sparse_selected_attention_rejects_unknown_impl(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DSV4_SELECTED_ATTN_IMPL", "bogus")
    q, kv, indices, lens, attn_sink = _inputs()

    with pytest.raises(RuntimeError, match="Unsupported TOKENSPEED_DSV4_SELECTED_ATTN_IMPL"):
        dispatch.native_gluon_sparse_selected_attention(
            q=q,
            kv=kv,
            indices=indices,
            lens=lens,
            attn_sink=attn_sink,
            softmax_scale=512**-0.5,
        )
