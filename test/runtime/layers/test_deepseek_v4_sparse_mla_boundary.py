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

from types import SimpleNamespace

import tokenspeed_kernel
import torch

import tokenspeed.runtime.layers.attention.backends.deepseek_v4 as deepseek_v4_backend
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)


def test_deepseek_v4_decode_calls_tokenspeed_kernel_sparse_mla_boundary(
    monkeypatch,
) -> None:
    tokens, local_heads, padded_heads, head_dim = 2, 32, 64, 512
    swa_indices = torch.zeros(tokens, 128, dtype=torch.int32)
    swa_lens = torch.full((tokens,), 128, dtype=torch.int32)
    compressed_indices = torch.zeros(tokens, 1, 512, dtype=torch.int32)
    compressed_lens = torch.full((tokens,), 512, dtype=torch.int32)
    attention = SimpleNamespace(
        decode_swa_indices=swa_indices,
        decode_swa_lens=swa_lens,
        decode_swa_window_size=128,
        decode_swa_block_size=64,
    )
    metadata = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        token_to_req_indices=torch.arange(tokens, dtype=torch.int32),
        attention=attention,
    )
    backend = object.__new__(DeepseekV4AttentionBackend)
    backend.forward_metadata = metadata
    monkeypatch.setattr(backend, "_select_decode_metadata", lambda _tokens: metadata)
    monkeypatch.setattr(
        backend,
        "_decode_compressed_attention_indices_and_lens",
        lambda *_args, **_kwargs: (compressed_indices, compressed_lens),
    )

    swa_cache = torch.empty(3, 64, 1, 584, dtype=torch.uint8)
    compressed_cache = torch.empty(4, 2, 1, 584, dtype=torch.uint8)
    monkeypatch.setattr(
        backend,
        "_fp8_ds_mla_cache_view",
        lambda cache, _block_size: cache,
    )
    pool = SimpleNamespace(
        swa_block_size=64,
        get_compressed_block_size=lambda _layer_id: 2,
        get_swa_kv_buffer=lambda _layer_id: swa_cache,
        get_compressed_kv_buffer_2d=lambda _layer_id: compressed_cache,
    )
    expected = torch.randn(tokens, padded_heads, head_dim, dtype=torch.bfloat16)
    call: dict[str, object] = {}

    def fake_sparse_mla_decode(**kwargs):
        call.update(kwargs)
        return expected

    monkeypatch.setattr(
        tokenspeed_kernel,
        "dsv4_sparse_mla_decode",
        fake_sparse_mla_decode,
    )
    q = torch.randn(tokens, local_heads, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(tokens, dtype=torch.int64)
    sinks = torch.zeros(padded_heads, dtype=torch.float32)

    actual = backend.forward_deepseek_v4_decode(
        q=q,
        positions=positions,
        token_to_kv_pool=pool,
        layer_id=0,
        kind="csa",
        compress_ratio=4,
        num_local_heads=local_heads,
        padded_heads=padded_heads,
        head_dim=head_dim,
        window_size=128,
        softmax_scale=head_dim**-0.5,
        attn_sink=sinks,
        topk_indices=torch.zeros(tokens, 512, dtype=torch.int32),
    )

    torch.testing.assert_close(actual, expected[:, :local_heads])
    q_padded = call["q"]
    assert isinstance(q_padded, torch.Tensor)
    assert q_padded.shape == (tokens, padded_heads, head_dim)
    torch.testing.assert_close(q_padded[:, :local_heads], q)
    assert call == {
        "q": q_padded,
        "swa_kv_cache": swa_cache,
        "swa_indices": swa_indices,
        "swa_topk_lens": swa_lens,
        "compressed_kv_cache": compressed_cache,
        "compressed_indices": compressed_indices,
        "compressed_topk_lens": compressed_lens,
        "softmax_scale": head_dim**-0.5,
        "sinks": sinks,
    }


def test_deepseek_v4_sm120_prefill_calls_tokenspeed_kernel_sparse_mla_boundary(
    monkeypatch,
) -> None:
    tokens, local_heads, padded_heads, head_dim = 128, 32, 64, 512
    swa_indices = torch.zeros(tokens, 128, dtype=torch.int32)
    swa_lens = torch.full((tokens,), 128, dtype=torch.int32)
    compressed_indices = torch.zeros(tokens, 1, 512, dtype=torch.int32)
    compressed_lens = torch.full((tokens,), 512, dtype=torch.int32)
    backend = object.__new__(DeepseekV4AttentionBackend)
    backend.forward_metadata = SimpleNamespace()
    monkeypatch.setattr(
        deepseek_v4_backend,
        "_use_flashinfer_sm120_sparse_mla",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        backend,
        "_update_decode_swa_metadata",
        lambda *_args, **_kwargs: (swa_indices, swa_lens),
    )
    monkeypatch.setattr(
        backend,
        "_decode_compressed_attention_indices_and_lens",
        lambda *_args, **_kwargs: (compressed_indices, compressed_lens),
    )
    monkeypatch.setattr(
        backend,
        "_prefill_workspace",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("SM120 prefill must not gather for FlashMLA")
        ),
    )

    swa_cache = torch.empty(3, 64, 1, 584, dtype=torch.uint8)
    compressed_cache = torch.empty(4, 2, 1, 584, dtype=torch.uint8)
    monkeypatch.setattr(
        backend,
        "_fp8_ds_mla_cache_view",
        lambda cache, _block_size: cache,
    )
    pool = SimpleNamespace(
        swa_block_size=64,
        get_compressed_block_size=lambda _layer_id: 2,
        get_swa_kv_buffer=lambda _layer_id: swa_cache,
        get_compressed_kv_buffer_2d=lambda _layer_id: compressed_cache,
    )
    expected = torch.randn(tokens, padded_heads, head_dim, dtype=torch.bfloat16)
    call: dict[str, object] = {}

    def fake_sparse_mla(**kwargs):
        call.update(kwargs)
        return expected

    monkeypatch.setattr(tokenspeed_kernel, "dsv4_sparse_mla_decode", fake_sparse_mla)
    q = torch.randn(tokens, local_heads, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(tokens, dtype=torch.int64)
    sinks = torch.zeros(padded_heads, dtype=torch.float32)

    actual = backend._forward_deepseek_v4_prefill_chunk(
        q=q,
        positions=positions,
        token_to_kv_pool=pool,
        layer_id=0,
        kind="csa",
        compress_ratio=4,
        num_local_heads=local_heads,
        padded_heads=padded_heads,
        head_dim=head_dim,
        window_size=128,
        softmax_scale=head_dim**-0.5,
        attn_sink=sinks,
        topk_indices=torch.zeros(tokens, 512, dtype=torch.int32),
    )

    torch.testing.assert_close(actual, expected[:, :local_heads])
    q_padded = call["q"]
    assert isinstance(q_padded, torch.Tensor)
    assert q_padded.shape == (tokens, padded_heads, head_dim)
    torch.testing.assert_close(q_padded[:, :local_heads], q)
    assert call == {
        "q": q_padded,
        "swa_kv_cache": swa_cache,
        "swa_indices": swa_indices,
        "swa_topk_lens": swa_lens,
        "compressed_kv_cache": compressed_cache,
        "compressed_indices": compressed_indices,
        "compressed_topk_lens": compressed_lens,
        "softmax_scale": head_dim**-0.5,
        "sinks": sinks,
    }


def test_deepseek_v4_non_sm120_prefill_retains_flashmla_path(monkeypatch) -> None:
    tokens, heads, head_dim = 2, 32, 512
    backend = object.__new__(DeepseekV4AttentionBackend)
    backend.forward_metadata = SimpleNamespace()
    monkeypatch.setattr(
        deepseek_v4_backend,
        "_use_flashinfer_sm120_sparse_mla",
        lambda: False,
        raising=False,
    )
    kv_workspace = torch.zeros(1, 8, head_dim, dtype=torch.bfloat16)
    indices = torch.zeros(tokens, 8, dtype=torch.int32)
    lens = torch.full((tokens,), 8, dtype=torch.int32)
    monkeypatch.setattr(
        backend,
        "_prefill_workspace",
        lambda **_kwargs: (kv_workspace, indices, lens),
    )
    expected = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16)
    call: dict[str, object] = {}

    def fake_flashmla(**kwargs):
        call.update(kwargs)
        return expected, None, None

    monkeypatch.setattr(deepseek_v4_backend, "flash_mla_sparse_fwd", fake_flashmla)
    monkeypatch.setattr(
        tokenspeed_kernel,
        "dsv4_sparse_mla_decode",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-SM120 prefill must retain FlashMLA")
        ),
    )
    q = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16)
    sinks = torch.zeros(heads, dtype=torch.float32)

    actual = backend._forward_deepseek_v4_prefill_chunk(
        q=q,
        positions=torch.arange(tokens, dtype=torch.int64),
        token_to_kv_pool=SimpleNamespace(),
        layer_id=0,
        kind="csa",
        compress_ratio=4,
        num_local_heads=heads,
        padded_heads=heads,
        head_dim=head_dim,
        window_size=128,
        softmax_scale=head_dim**-0.5,
        attn_sink=sinks,
        topk_indices=torch.zeros(tokens, 8, dtype=torch.int32),
    )

    torch.testing.assert_close(actual, expected)
    assert set(call) == {
        "q",
        "kv",
        "indices",
        "sm_scale",
        "attn_sink",
        "topk_length",
    }
    torch.testing.assert_close(call["q"], q)
    torch.testing.assert_close(call["kv"], kv_workspace.view(-1, 1, head_dim))
    torch.testing.assert_close(call["indices"], indices.unsqueeze(1))
    assert call["sm_scale"] == head_dim**-0.5
    torch.testing.assert_close(call["attn_sink"], sinks)
    torch.testing.assert_close(call["topk_length"], lens)
