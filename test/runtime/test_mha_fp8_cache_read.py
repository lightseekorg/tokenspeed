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

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import tokenspeed.runtime.layers.attention.backends.mha as mha_backend


def _backend(is_fp8: bool, *, mixed_fp8_cache_read: bool | None = None):
    backend = mha_backend.MHAAttnBackend.__new__(mha_backend.MHAAttnBackend)
    backend.is_fp8 = is_fp8
    backend.kv_cache_dtype = torch.float8_e4m3fn if is_fp8 else torch.bfloat16
    backend.qkv_dtype = torch.bfloat16
    backend.use_mixed_fp8_cache_read = (
        is_fp8 if mixed_fp8_cache_read is None else mixed_fp8_cache_read
    )
    backend.kernel_solution = None
    backend.max_context_len = 64
    cache_dtype = backend.kv_cache_dtype
    k_cache = torch.empty((1, 64, 1, 8), dtype=cache_dtype)
    v_cache = torch.empty_like(k_cache)
    backend._get_kv_cache = lambda layer, pool: (k_cache, v_cache)
    backend._select_page_table = lambda layer, metadata: metadata.page_table
    return backend


def _layer():
    return SimpleNamespace(
        k_scale=0.25,
        v_scale=0.5,
        tp_q_head_num=2,
        v_head_dim=8,
        sliding_window_size=-1,
        logit_cap=0.0,
        non_causal=False,
    )


@pytest.mark.parametrize("is_fp8", [False, True], ids=["bf16", "fp8"])
def test_runtime_cached_extend_only_supplies_fp8_options(
    monkeypatch: pytest.MonkeyPatch, is_fp8: bool
) -> None:
    backend = _backend(is_fp8)
    captured = {}

    def extend_kernel(**kwargs):
        captured.update(kwargs)
        return torch.empty(
            kwargs["q"].shape, dtype=torch.bfloat16, device=kwargs["q"].device
        )

    monkeypatch.setattr(mha_backend, "mha_extend_with_kvcache", extend_kernel)
    monkeypatch.setattr(mha_backend, "_scrub_extend_padding", lambda *args: None)
    metadata = SimpleNamespace(
        cu_extend_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_extend_seq_len=2,
    )
    q = torch.ones((2, 2, 8), dtype=torch.bfloat16)
    k = torch.ones((2, 1, 8), dtype=torch.bfloat16)
    v = torch.ones_like(k)

    output = backend._forward_extend(
        q,
        k,
        v,
        _layer(),
        torch.tensor([0, 1], dtype=torch.int32),
        object(),
        metadata,
        save_kv_cache=False,
        sinks=None,
    )

    assert output.dtype is torch.bfloat16
    assert captured["q"].dtype is torch.bfloat16
    if is_fp8:
        assert captured["k_scale"] == 0.25
        assert captured["v_scale"] == 0.5
        assert captured["output_dtype"] is torch.bfloat16
    else:
        assert "k_scale" not in captured
        assert "v_scale" not in captured
        assert "output_dtype" not in captured


def test_runtime_mixed_fp8_extend_slices_prefill_graph_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(is_fp8=True)
    captured = {}

    def extend_kernel(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs["q"])

    def save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v):
        captured["saved_locs"] = out_cache_loc
        captured["saved_k"] = k
        captured["saved_v"] = v

    monkeypatch.setattr(mha_backend, "mha_extend_with_kvcache", extend_kernel)
    monkeypatch.setattr(mha_backend, "current_forward_ctx", lambda: object())
    monkeypatch.setattr(
        mha_backend,
        "_scrub_extend_padding",
        lambda *args: pytest.fail("mixed FA2 graph path must slice, not scrub"),
    )
    backend._save_kv_cache = save_kv_cache
    metadata = SimpleNamespace(
        cu_extend_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        cu_extend_seq_lens_cpu=[0, 2],
        cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_extend_seq_len=2,
    )
    q = torch.ones((4, 2, 8), dtype=torch.bfloat16)
    k = torch.ones((4, 1, 8), dtype=torch.bfloat16)
    v = torch.ones_like(k)
    out_cache_loc = torch.tensor([5, 6, 0, 0], dtype=torch.int32)

    output = backend._forward_extend(
        q,
        k,
        v,
        _layer(),
        out_cache_loc,
        object(),
        metadata,
        save_kv_cache=True,
        sinks=None,
    )

    assert captured["q"].shape[0] == 2
    assert captured["saved_k"].shape[0] == 2
    assert captured["saved_v"].shape[0] == 2
    assert torch.equal(captured["saved_locs"], torch.tensor([5, 6], dtype=torch.int32))
    assert output.shape == (2, 16)


def test_runtime_nonindexed_fp8_preserves_legacy_same_dtype_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(is_fp8=True, mixed_fp8_cache_read=False)
    captured = {}

    def prefill_kernel(**kwargs):
        captured["prefill"] = kwargs
        return torch.empty_like(kwargs["q"])

    def extend_kernel(**kwargs):
        captured["extend"] = kwargs
        return torch.empty_like(kwargs["q"])

    def decode_kernel(**kwargs):
        captured["decode"] = kwargs
        return torch.empty_like(kwargs["q"])

    monkeypatch.setattr(mha_backend, "mha_prefill", prefill_kernel)
    monkeypatch.setattr(mha_backend, "mha_extend_with_kvcache", extend_kernel)
    monkeypatch.setattr(mha_backend, "mha_decode_with_kvcache", decode_kernel)
    monkeypatch.setattr(mha_backend, "_scrub_extend_padding", lambda *args: None)
    extend_metadata = SimpleNamespace(
        cu_extend_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        cu_extend_seq_lens_cpu=[0, 2],
        cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_extend_seq_len=2,
    )
    decode_metadata = SimpleNamespace(
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
    )
    q = torch.ones((2, 2, 8), dtype=torch.bfloat16)
    k = torch.ones((2, 1, 8), dtype=torch.bfloat16)
    v = torch.ones_like(k)
    out_cache_loc = torch.tensor([0, 1], dtype=torch.int32)
    layer = _layer()

    backend._forward_prefill(
        q,
        k,
        v,
        layer,
        out_cache_loc,
        object(),
        extend_metadata,
        save_kv_cache=False,
        sinks=None,
    )
    backend._forward_extend(
        q,
        k,
        v,
        layer,
        out_cache_loc,
        object(),
        extend_metadata,
        save_kv_cache=False,
        sinks=None,
    )
    backend._forward_decode(
        q[:1],
        None,
        None,
        layer,
        out_cache_loc[:1],
        object(),
        decode_metadata,
        save_kv_cache=False,
        sinks=None,
    )

    for name in ("prefill", "extend", "decode"):
        assert captured[name]["q"].dtype is torch.float8_e4m3fn
    assert captured["prefill"]["k"].dtype is torch.float8_e4m3fn
    assert captured["prefill"]["v"].dtype is torch.float8_e4m3fn
    for name in ("extend", "decode"):
        assert "k_scale" not in captured[name]
        assert "v_scale" not in captured[name]
        assert "output_dtype" not in captured[name]


@pytest.mark.parametrize("is_fp8", [False, True], ids=["bf16", "fp8"])
def test_runtime_cached_decode_only_supplies_fp8_options(
    monkeypatch: pytest.MonkeyPatch, is_fp8: bool
) -> None:
    backend = _backend(is_fp8)
    captured = {}

    def decode_kernel(**kwargs):
        captured.update(kwargs)
        return torch.empty(
            kwargs["q"].shape, dtype=torch.bfloat16, device=kwargs["q"].device
        )

    monkeypatch.setattr(mha_backend, "mha_decode_with_kvcache", decode_kernel)
    metadata = SimpleNamespace(
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
    )
    q = torch.ones((1, 2, 8), dtype=torch.bfloat16)

    output = backend._forward_decode(
        q,
        None,
        None,
        _layer(),
        torch.tensor([0], dtype=torch.int32),
        object(),
        metadata,
        save_kv_cache=False,
        sinks=None,
    )

    assert output.dtype is torch.bfloat16
    assert captured["q"].dtype is torch.bfloat16
    if is_fp8:
        assert captured["k_scale"] == 0.25
        assert captured["v_scale"] == 0.5
        assert captured["output_dtype"] is torch.bfloat16
    else:
        assert "k_scale" not in captured
        assert "v_scale" not in captured
        assert "output_dtype" not in captured


@pytest.mark.parametrize(
    "activation_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_runtime_indexed_fp8_prefill_preserves_activation_dtype(
    monkeypatch: pytest.MonkeyPatch, activation_dtype: torch.dtype
) -> None:
    backend = _backend(is_fp8=True)
    captured = {}

    def prefill_kernel(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs["q"])

    def save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v):
        captured["saved_k"] = k
        captured["saved_v"] = v

    monkeypatch.setattr(mha_backend, "mha_prefill", prefill_kernel)
    monkeypatch.setattr(mha_backend, "_scrub_extend_padding", lambda *args: None)
    backend._save_kv_cache = save_kv_cache
    metadata = SimpleNamespace(
        cu_extend_seq_lens=torch.tensor([0, 2], dtype=torch.int32),
        cu_extend_seq_lens_cpu=[0, 2],
        max_extend_seq_len=2,
    )
    q = torch.empty((2, 2, 8), dtype=activation_dtype)
    k = torch.empty((2, 1, 8), dtype=activation_dtype)
    v = torch.empty_like(k)

    output = backend._forward_prefill(
        q,
        k,
        v,
        _layer(),
        torch.tensor([0, 1], dtype=torch.int32),
        object(),
        metadata,
        save_kv_cache=True,
        sinks=None,
    )

    assert captured["q"].dtype is activation_dtype
    assert captured["k"].dtype is activation_dtype
    assert captured["v"].dtype is activation_dtype
    assert captured["saved_k"].dtype is activation_dtype
    assert captured["saved_v"].dtype is activation_dtype
    assert output.dtype is activation_dtype
