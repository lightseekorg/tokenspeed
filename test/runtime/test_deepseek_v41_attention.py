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

"""Attention projection fusion, shared FP8 inputs and stream ordering."""

from __future__ import annotations

import os
import sys
from test.runtime.test_deepseek_v41_model import (
    _Backend,
    _config,
    _ctx,
    _initialize,
    _mapping,
    _quant,
)
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41.triton import pack_index_inputs
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from torch import nn

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    V41DecoderView,
)
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import LinearBase
from tokenspeed.runtime.models import deepseek_v41 as v41
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41Attention,
    v41_mxfp8_config,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_blackwell,
    reason="requires a Blackwell GPU",
)


class _Projection(nn.Module):
    """Use dense test weights without distributed or quantized linear setup."""

    def __init__(self, input_size, output_size, dtype, quant_config, prefix):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size, dtype=dtype) * 0.1
        )

    def forward(self, x, block_scale, output_dtype):
        return F.linear(x, self.weight), None


def _make_indexer(monkeypatch, device, head_dim, heads, compress_ratio):
    torch.manual_seed(123)
    monkeypatch.setattr(v41, "_replicated", _Projection)
    config = SimpleNamespace(
        index_n_heads=heads,
        index_head_dim=head_dim,
        q_lora_rank=16,
        hidden_size=32,
        head_dim=512,
        rms_norm_eps=1e-6,
        qk_rope_head_dim=min(64, head_dim),
        rope_theta=10000.0,
        compress_rope_theta=1000000.0,
        max_position_embeddings=4096,
        rope_scaling={
            "original_max_position_embeddings": 128,
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )
    indexer = v41.DeepseekV41Indexer(
        config, SimpleNamespace(), True, None, "indexer"
    ).to(device)
    with torch.no_grad():
        indexer.k_norm.weight.normal_(1.0, 0.2)
    rotary = v41.DeepseekV41RotaryEmbedding(config, compress_ratio).to(device)
    return indexer, rotary


def _indexer_inputs(tokens, device):
    x = torch.randn(tokens, 32, device=device, dtype=torch.bfloat16)
    qr = torch.randn(tokens, 16, device=device, dtype=torch.bfloat16)
    latent = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int64) * 17 - 1
    return x, qr, latent, positions


def _reference_indexer(indexer, rotary, x, qr, latent, positions):
    q, _ = indexer.wq_b(qr, block_scale=None, output_dtype=None)
    q = rotary.apply_owned(
        q.unflatten(-1, (indexer.n_local_heads, indexer.head_dim)), positions, False
    )
    weights, _ = indexer.weights_proj(x, block_scale=None, output_dtype=None)
    key, _ = indexer.wk(latent, block_scale=None, output_dtype=None)
    key = rotary.apply_owned(v41._attention_norm(key, indexer.k_norm), positions, False)
    return q, weights * indexer.scale, key


def _assert_indexer_outputs(actual, expected):
    q, weights, key = actual
    assert q.dtype == weights.dtype == key.dtype == torch.bfloat16
    torch.testing.assert_close(q, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(weights, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(key, expected[2], rtol=1e-2, atol=8e-3)


def _consume_indexer_inputs(query, weights, key, positions, slots, cache):
    assert isinstance(query, dsv41.IndexQueryInput)
    assert isinstance(key, dsv41.IndexKeyInput)
    packed, scaled_weights = dsv41.index_query_rope_quant(
        query.values,
        weights,
        query.positions,
        query.cos_sin_cache,
        query.weight_scale,
        torch.bfloat16,
    )
    dsv41.index_key_norm_rope_scatter(
        key.values,
        key.norm_weight,
        positions,
        key.cos_sin_cache,
        key.norm_eps,
        cache,
        slots,
    )
    return *packed, scaled_weights, cache


def _reference_packed_indexer(indexer, rotary, x, qr, latent, positions, slots):
    q, weights, reference_key = _reference_indexer(
        indexer, rotary, x, qr, latent, positions
    )
    packed, scaled_weights = pack_index_inputs(q, weights, torch.bfloat16)
    projected_key, _ = indexer.wk(latent, block_scale=None, output_dtype=None)
    key = dsv41.index_key_norm_rope(
        projected_key,
        indexer.k_norm.weight,
        positions,
        rotary.cos_sin_cache,
        indexer.k_norm.variance_epsilon,
    )
    # RMS reductions can differ by one BF16 ULP from Torch. Compare the
    # packed cache byte-for-byte with the original kernel sequence.
    torch.testing.assert_close(key, reference_key, rtol=1e-2, atol=8e-3)
    cache = torch.full((1, 64, 68), 0xA7, device=x.device, dtype=torch.uint8)
    dsv41.cache_scatter(key, cache, slots, "index")
    return *packed, scaled_weights, cache


@requires_blackwell
@pytest.mark.parametrize("tokens", [0, 1, 17])
@pytest.mark.parametrize("compress_ratio", [0, 1])
@torch.inference_mode()
def test_indexer_fused_forward_and_key(monkeypatch, tokens, compress_ratio):
    indexer, rotary = _make_indexer(monkeypatch, "cuda", 128, 32, compress_ratio)
    x, qr, latent, positions = _indexer_inputs(tokens, "cuda")
    before = [tensor.clone() for tensor in (x, qr, latent, positions)]
    slots = torch.arange(tokens, device="cuda")
    cache = torch.full((1, 64, 68), 0xA7, device="cuda", dtype=torch.uint8)
    expected = _reference_packed_indexer(
        indexer, rotary, x, qr, latent, positions, slots
    )
    q, weights = indexer(x, qr, None, positions, rotary)
    key = indexer.key(latent, positions, rotary)
    assert isinstance(q, dsv41.IndexQueryInput)
    assert isinstance(key, dsv41.IndexKeyInput)
    assert q.values.shape == (tokens, 32, 128)
    assert weights.shape == (tokens, 32)
    assert key.values.shape == (tokens, 128)
    actual = _consume_indexer_inputs(q, weights, key, positions, slots, cache)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for original, saved in zip((x, qr, latent, positions), before, strict=True):
        torch.testing.assert_close(original, saved, rtol=0, atol=0)


@requires_blackwell
@torch.inference_mode()
def test_indexer_fused_graph_refreshes_inputs_and_positions(monkeypatch):
    indexer, rotary = _make_indexer(monkeypatch, "cuda", 128, 32, 1)
    x, qr, latent, positions = _indexer_inputs(4, "cuda")
    slots = torch.arange(4, device="cuda")
    cache = torch.full((1, 64, 68), 0xA7, device="cuda", dtype=torch.uint8)

    def run():
        q, weights = indexer(x, qr, None, positions, rotary)
        key = indexer.key(latent, positions, rotary)
        return _consume_indexer_inputs(q, weights, key, positions, slots, cache)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
    torch.cuda.current_stream().wait_stream(stream)
    addresses = [tensor.data_ptr() for tensor in actual]
    for row_positions in ([0, 31, 127, 4095], [-1, -2, -3, -4], [255, 511, 1023, 2047]):
        x.normal_()
        qr.normal_()
        latent.normal_()
        positions.copy_(torch.tensor(row_positions, device="cuda"))
        expected = _reference_packed_indexer(
            indexer, rotary, x, qr, latent, positions, slots
        )
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert [tensor.data_ptr() for tensor in actual] == addresses


@pytest.mark.parametrize(
    "device,head_dim,heads", [("cpu", 128, 32), ("cpu", 32, 4), ("cuda", 32, 4)]
)
@torch.inference_mode()
def test_indexer_cpu_and_small_geometry_keep_portable_path(
    monkeypatch, device, head_dim, heads
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    indexer, rotary = _make_indexer(monkeypatch, device, head_dim, heads, 1)
    x, qr, latent, positions = _indexer_inputs(3, device)
    expected = _reference_indexer(indexer, rotary, x, qr, latent, positions)
    q, weights = indexer(x, qr, None, positions, rotary)
    key = indexer.key(latent, positions, rotary)
    _assert_indexer_outputs((q, weights, key), expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_blackwell_plus,
    reason="requires Blackwell FP8 linears",
)
@pytest.mark.parametrize("case", ["shared", "unquantized", "narrowed"])
def test_attention_shares_q_quantization_only_with_compatible_indexer(
    monkeypatch, case
):
    torch.manual_seed(831)
    config = _config()
    config.q_lora_rank = 128
    config.index_head_dim = 64
    old_pdl = pdl_enabled()
    pdl_enabled(True)
    try:
        with torch.device("cuda"):
            attention = DeepseekV41Attention(
                config,
                _mapping(0, 1, 1),
                24,
                24 if case == "narrowed" else config.candidate_source_layer_id,
                v41_mxfp8_config(_quant()),
                "layers.24.attn",
                aux_stream=None,
            )
        _initialize(attention)
        for module in attention.modules():
            if isinstance(module, LinearBase):
                module.quant_method.process_weights_after_loading(module)
        index_linear = attention.indexer.wq_b
        if case == "unquantized":
            index_linear.weight.data = index_linear.weight.data.to(torch.bfloat16)
            index_linear.quant_method = UnquantizedLinearMethod()

        positions = torch.arange(6, device="cuda")
        requests = torch.zeros_like(positions)
        backend = _Backend(positions, requests)
        if case == "narrowed":
            keep = torch.tensor([1, 3, 5], device="cuda")
            query = SimpleNamespace(
                positions=positions[keep], request_indices=requests[keep]
            )
            backend = _Backend(
                positions, requests, V41DecoderView(query, None, (), keep, None)
            )
        x = torch.randn(6, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        ctx = _ctx(backend, 6, ForwardMode.EXTEND)
        received = []
        hook = index_linear.register_forward_pre_hook(
            lambda module, args: received.append(args[0])
        )
        method = index_linear.quant_method
        index_linear.quant_method = SimpleNamespace(apply=method.apply)
        expected = attention(positions, x, ctx, reduce_output=True)
        expected_index = backend.calls[-1][3:5]
        assert received[-1].dtype == torch.bfloat16
        index_linear.quant_method = method
        actual = attention(positions, x, ctx, reduce_output=True)
        # Narrowed rows select the shared codes after quantization.
        shared = received[-1].dtype == torch.float8_e4m3fn
        assert shared == (case in ("shared", "narrowed"))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_index, reference_index in zip(
            backend.calls[-1][3:5], expected_index
        ):
            torch.testing.assert_close(actual_index, reference_index, rtol=0, atol=0)
        hook.remove()
    finally:
        pdl_enabled(old_pdl)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_blackwell_plus,
    reason="requires Blackwell FP8 linears",
)
def test_shared_q_decode_graph_replay_with_auxiliary_indexer(monkeypatch):
    torch.manual_seed(943)
    config = _config()
    config.q_lora_rank = 128
    config.index_head_dim = 64
    with torch.device("cuda"):
        attention = DeepseekV41Attention(
            config,
            _mapping(0, 1, 1),
            24,
            config.candidate_source_layer_id,
            v41_mxfp8_config(_quant()),
            "layers.24.attn",
            aux_stream=torch.cuda.Stream(),
        )
    _initialize(attention)
    for module in attention.modules():
        if isinstance(module, LinearBase):
            module.quant_method.process_weights_after_loading(module)
    positions = torch.arange(6, device="cuda")
    backend = _Backend(positions, torch.zeros_like(positions))
    backend.sparse_topk = SimpleNamespace(
        decode=SimpleNamespace(
            logical_rows=torch.zeros((6, 1), device="cuda", dtype=torch.int32),
            lengths=torch.ones(6, device="cuda", dtype=torch.int32),
            candidates=None,
        )
    )
    selected = []

    def prepare(owner, q, weights, pos, requests, mode, group, *, projected_partition):
        assert projected_partition is None
        selected.append((q, weights))
        return q, weights

    backend.prepare_global_selection = prepare
    ctx = _ctx(backend, 6, ForwardMode.DECODE)
    x = torch.randn(6, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        attention(positions, x, ctx, reduce_output=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attention(positions, x, ctx, reduce_output=True)
    captured_index = selected[-1]
    index_linear = attention.indexer.wq_b
    index_linear.quant_method = SimpleNamespace(apply=index_linear.quant_method.apply)
    for factor in (0.0, 0.2, 2.0):
        x.normal_().mul_(factor)
        graph.replay()
        expected = attention(positions, x, ctx, reduce_output=True)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        for actual_index, expected_index in zip(captured_index, selected[-1]):
            torch.testing.assert_close(actual_index, expected_index, rtol=0, atol=0)
