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
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import _is_dflash2_mla
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.drafter.dflash2 import (
    DFlash2,
    _greedy_path_torch,
    _walk_greedy_path,
)

# Imported for its register_backend() side effects on _BACKEND_REGISTRY.
from tokenspeed.runtime.layers.attention import backends  # noqa: F401
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import _BACKEND_REGISTRY
from tokenspeed.runtime.models.dflash import DFlashAttention, DFlashDraftModel
from tokenspeed.runtime.models.dflash2 import (
    CandidateSelector,
    DFlash2DraftModel,
    DFlashGroupedConv,
    _dflash2_mla_rope,
    _dflash2_uses_mla,
    _grouped_conv,
    _score_edges,
)

_CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_dflash2_architecture_dispatches_to_its_selector_runtime() -> None:
    model = DFlash2DraftModel.__new__(DFlash2DraftModel)
    assert get_drafter_impl("DFLASH", model) is DFlash2


@pytest.mark.parametrize(
    ("architecture", "attention_mode", "expected"),
    (
        ("DFlash2DraftModel", "mla", True),
        ("DFlash2DraftModel", "gqa", False),
        ("DFlashDraftModel", "mla", False),
    ),
)
def test_dflash2_mla_attention_family_detection(
    architecture: str, attention_mode: str, expected: bool
) -> None:
    config = SimpleNamespace(
        architectures=[architecture],
        dflash_config={"attention_mode": attention_mode},
    )
    assert _is_dflash2_mla(config, config) is expected


def test_dflash2_mla_model_mode_and_yarn_config() -> None:
    config = SimpleNamespace(
        dflash_config={"attention_mode": "mla"},
        rope_theta=1_000_000.0,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 50_000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )

    assert _dflash2_uses_mla(config)
    rope_theta, scaling = _dflash2_mla_rope(config)
    assert rope_theta == 50_000.0
    assert scaling == {
        "rope_type": "deepseek_yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 32768,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }


def test_dflash_attention_uses_the_full_attention_cache_group() -> None:
    assert DFlashAttention.cache_group_id == FULL_ATTENTION


def test_candidate_logits_processor_is_created_after_target_wiring() -> None:
    drafter = DFlash2.__new__(DFlash2)
    drafter.model = SimpleNamespace(config=SimpleNamespace(vocab_size=32))
    drafter.output_multiplier = 1.25
    drafter.final_logit_softcapping = 8.0
    drafter.candidate_logits_processor = None
    drafter.lm_head = SimpleNamespace()
    drafter.logits_processor = SimpleNamespace(tp_rank=0, tp_size=1, tp_group=None)

    with mock.patch("tokenspeed.runtime.execution.drafter.dflash2.DFlash.wire_target"):
        drafter.wire_target(SimpleNamespace())

    assert drafter.candidate_logits_processor is not None
    assert drafter.candidate_logits_processor.logit_scale == 1.25
    assert drafter.candidate_logits_processor.final_logit_softcapping == 8.0


def test_candidate_unary_logits_are_promoted_to_fp32() -> None:
    drafter = DFlash2.__new__(DFlash2)
    drafter.selector_top_k = 2
    drafter.lm_head = object()
    drafter.candidate_logits_processor = mock.Mock()
    drafter.candidate_logits_processor._get_logits.return_value = torch.tensor(
        [[1.0, 4.0, 3.0]], dtype=torch.bfloat16
    )

    candidate_ids, unary_logits = drafter._compute_candidates(
        torch.zeros(1, 4, dtype=torch.bfloat16)
    )

    assert candidate_ids.dtype == torch.int64
    assert unary_logits.dtype == torch.float32
    torch.testing.assert_close(
        unary_logits.sort(dim=-1).values,
        torch.tensor([[3.0, 4.0]], dtype=torch.float32),
    )


# Backend names, spelled as --drafter-attention-backend takes them, whose
# kernel call sites forward layer.sliding_window_size.
_WINDOW_AWARE_BACKENDS = frozenset(
    ("mha", "fa3", "fa4", "triton", "flashinfer", "trtllm", "mla", "gluon")
)


def _window_validator(*windows: int, supported: bool, backend: str) -> DFlash:
    """A DFlash stub whose draft layers carry the given window_lefts."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for window in windows:
        attention = torch.nn.Module()
        attention.sliding_window_size = window
        model.layers.append(attention)
    drafter = DFlash.__new__(DFlash)
    drafter.model = model
    drafter.attn_backend = SimpleNamespace(supports_layer_sliding_window=supported)
    drafter.draft_model_runner = SimpleNamespace(
        server_args=SimpleNamespace(drafter_attention_backend=backend)
    )
    return drafter


def test_only_the_documented_backends_apply_per_layer_sliding_windows() -> None:
    claiming = {
        name
        for name, (_, backend_cls) in _BACKEND_REGISTRY.items()
        if backend_cls.supports_layer_sliding_window
    }

    assert claiming == _WINDOW_AWARE_BACKENDS & set(_BACKEND_REGISTRY)


def test_a_sliding_window_draft_rejects_a_backend_that_drops_the_window() -> None:
    drafter = _window_validator(4095, -1, supported=False, backend="tokenspeed_mla")

    with pytest.raises(ValueError, match="tokenspeed_mla.*ignores per-layer"):
        drafter._validate_draft_attention_window()


def test_draft_attention_window_validation_accepts_the_served_combinations() -> None:
    honoured = _window_validator(4095, -1, supported=True, backend="mla")
    unwindowed = _window_validator(-1, -1, supported=False, backend="mla")

    honoured._validate_draft_attention_window()
    unwindowed._validate_draft_attention_window()


@pytest.mark.parametrize("block_size", (6, 8))
@pytest.mark.parametrize(
    "device",
    ("cpu", pytest.param("cuda", marks=_CUDA_ONLY)),
)
def test_grouped_conv_matches_a_block_local_reference(
    block_size: int, device: str
) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(2 * block_size, 12, device=device)
    delta = torch.randn(2 * block_size, 3, 3, device=device)
    base = torch.randn(3, 12, device=device)
    actual = _grouped_conv(hidden, delta, base, block_size, 3, 4, 3)

    expected = torch.zeros_like(hidden)
    for row in range(2 * block_size):
        position = row % block_size
        for tap in range(min(position + 1, 3)):
            for group in range(3):
                sl = slice(group * 4, (group + 1) * 4)
                expected[row, sl] += (base[tap, sl] + delta[row, tap, group]) * hidden[
                    row - tap, sl
                ]
    torch.testing.assert_close(actual, expected)


def test_candidate_selector_edges_match_the_official_equation() -> None:
    torch.manual_seed(1)
    batch, steps, top_k, vocab, rank = 2, 3, 4, 19, 5
    predecessor = torch.randn(vocab, rank)
    successor = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessor, successor, candidate_ids, unary, hidden, anchors, top_k
    )
    expected = torch.empty_like(actual)
    for b in range(batch):
        for step in range(steps):
            for previous in range(top_k):
                predecessor_id = (
                    anchors[b] if step == 0 else candidate_ids[b, step - 1, previous]
                )
                for candidate in range(top_k):
                    successor_id = candidate_ids[b, step, candidate]
                    expected[b, step, previous, candidate] = unary[
                        b, step, candidate
                    ] + torch.dot(
                        predecessor[predecessor_id] * hidden[b, step],
                        successor[successor_id],
                    )
    torch.testing.assert_close(actual, expected)


def test_greedy_path_follows_the_selected_predecessor() -> None:
    candidate_ids = torch.tensor([[[10, 11], [20, 21], [30, 31]]])
    scores = torch.zeros(1, 3, 2, 2)
    scores[0, 0, 0, 1] = 3
    scores[0, 1, 1, 0] = 4
    scores[0, 2, 0, 1] = 5
    out = torch.empty(1, 4, dtype=torch.int32)
    _walk_greedy_path(candidate_ids, scores, torch.tensor([7]), out)
    assert out.tolist() == [[7, 11, 20, 31]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_the_path_kernel_reproduces_the_torch_walk() -> None:
    torch.manual_seed(5)
    candidate_ids = torch.rand(4, 7, 128).topk(16).indices.cuda()
    scores = torch.randn(4, 7, 16, 16, device="cuda")
    scores[:, 0] = scores[:, 0, :1].expand(-1, 16, -1)
    anchors = torch.randint(128, (4,), device="cuda")
    kernel_out = torch.empty(4, 8, dtype=torch.int32, device="cuda")
    torch_out = torch.empty(4, 8, dtype=torch.int32, device="cuda")

    _walk_greedy_path(candidate_ids, scores, anchors, kernel_out)
    _greedy_path_torch(candidate_ids, scores, anchors, torch_out)

    torch.testing.assert_close(kernel_out, torch_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_official_nodes_capture_and_replay_in_one_cuda_graph() -> None:
    device = torch.device("cuda")
    conv = DFlashGroupedConv(16, taps=2, group_size=4, block_size=8).to(device)
    selector = CandidateSelector(16, vocab_size=32, rank=4, top_k=4).to(device)
    static_hidden = torch.randn(8, 16, device=device)
    static_candidates = torch.randint(32, (1, 7, 4), device=device)
    static_unary = torch.randn(1, 7, 4, device=device)
    static_anchor = torch.tensor([3], device=device)
    static_out = torch.empty(1, 8, dtype=torch.int32, device=device)

    for _ in range(3):
        prepared, coefficients = conv.prepare(static_hidden)
        finished = conv.finish(prepared, coefficients)
        scores = selector(
            static_candidates, static_unary, finished[1:].unsqueeze(0), static_anchor
        )
        _walk_greedy_path(static_candidates, scores, static_anchor, static_out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared, coefficients = conv.prepare(static_hidden)
        finished = conv.finish(prepared, coefficients)
        scores = selector(
            static_candidates, static_unary, finished[1:].unsqueeze(0), static_anchor
        )
        _walk_greedy_path(static_candidates, scores, static_anchor, static_out)

    new_hidden = torch.randn_like(static_hidden)
    static_hidden.copy_(new_hidden)
    graph.replay()
    torch.cuda.synchronize()
    replayed = static_out.clone()

    prepared, coefficients = conv.prepare(new_hidden)
    finished = conv.finish(prepared, coefficients)
    eager_scores = selector(
        static_candidates, static_unary, finished[1:].unsqueeze(0), static_anchor
    )
    expected = torch.empty_like(static_out)
    _walk_greedy_path(static_candidates, eager_scores, static_anchor, expected)
    torch.testing.assert_close(replayed, expected)


@_CUDA_ONLY
def test_the_fused_mla_context_write_reproduces_the_per_layer_chain() -> None:
    """One GEMM + one launch must land what norm -> RoPE -> scatter lands."""
    from tokenspeed_kernel.ops.kvcache.triton import mla_latent_norm_rope_scatter

    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.layers.rotary_embedding import get_rope

    torch.manual_seed(3)
    device, dtype = torch.device("cuda"), torch.bfloat16
    layers, tokens, hidden, q_lora, kv_lora, rope_dim = 3, 16, 64, 8, 32, 16
    width = kv_lora + rope_dim
    rotary = get_rope(
        rope_dim, rotary_dim=rope_dim, max_position=128, base=10000, is_neox_style=False
    ).to(device)
    norms = [
        RMSNorm(kv_lora, eps=1e-6).to(device=device, dtype=dtype) for _ in range(layers)
    ]
    for norm in norms:
        norm.weight.data.normal_()
    weights = [
        torch.randn(q_lora + width, hidden, device=device, dtype=dtype)
        for _ in range(layers)
    ]
    hidden_states = torch.randn(tokens, hidden, device=device, dtype=dtype)
    positions = torch.randint(128, (tokens,), device=device)
    loc = torch.randperm(tokens, device=device).to(torch.int32)

    expected = torch.zeros(tokens, width, device=device, dtype=dtype)
    planes = []
    for index, weight in enumerate(weights):
        latent = torch.nn.functional.linear(hidden_states, weight[q_lora:])
        latent = torch.cat(
            (norms[index](latent[..., :kv_lora].contiguous()), latent[..., kv_lora:]),
            dim=-1,
        )
        k_pe = latent[..., kv_lora:].reshape(-1, 1, rope_dim).clone()
        _, rotated = rotary(positions, k_pe.new_empty(k_pe.shape), k_pe)
        latent[..., kv_lora:] = rotated.reshape(tokens, rope_dim)
        plane = torch.zeros_like(expected)
        plane[loc.long()] = latent
        planes.append(plane)

    stacked = torch.mm(
        hidden_states,
        torch.cat([w[q_lora:] for w in weights], dim=0).t().contiguous(),
    ).view(tokens, layers, width)
    actual = [torch.zeros_like(expected) for _ in range(layers)]
    mla_latent_norm_rope_scatter(
        stacked,
        torch.stack([norm.weight.data for norm in norms]),
        torch.full((layers,), 1e-6, device=device),
        rotary.cos_sin_cache,
        positions,
        loc,
        torch.tensor([p.data_ptr() for p in actual], dtype=torch.int64, device=device),
        actual[0].stride(0),
        dtype,
        is_neox=rotary.is_neox_style,
    )

    for index in range(layers):
        torch.testing.assert_close(
            actual[index].float(), planes[index].float(), atol=2e-2, rtol=2e-2
        )


def test_a_mixed_batch_derives_draft_lengths_without_reading_the_device() -> None:
    """Prefill+decode batches must not stop the stream once per request."""
    drafter = DFlash.__new__(DFlash)
    drafter.spec_num_tokens = 4
    drafter.device = torch.device("cpu")
    positions = torch.tensor([10, 11, 12, 13, 14, 20, 21, 22, 23, 30, 31, 32, 33])
    drafter.input_buffers = SimpleNamespace(
        input_lengths_buf=torch.tensor([5, 1, 1], dtype=torch.int32),
        req_pool_indices_buf=torch.tensor([0, 1, 2]),
        positions_buf=positions,
        out_cache_loc_buf=torch.arange(positions.numel(), dtype=torch.int32),
    )
    drafter.runtime_states = SimpleNamespace(
        valid_cache_lengths=torch.tensor([100, 200, 300], dtype=torch.int32)
    )
    drafter.draft_seq_lens_buf = torch.zeros(3, dtype=torch.int32)
    written = []
    drafter._write_native_cache = lambda *args, **kwargs: written.append(kwargs)

    ctx = SimpleNamespace(bs=3, num_extends=1, input_num_tokens=positions.numel())
    drafter._update_native_cache_from_target(
        ctx,
        SimpleNamespace(hidden_states=torch.zeros(positions.numel(), 8)),
        # Row 0 is a prefill chunk; row 1 keeps two tokens, row 2 keeps none.
        accept_lengths=torch.tensor([0, 2, 0], dtype=torch.int32),
    )

    assert drafter.draft_seq_lens_buf.tolist() == [15, 22, 300]
    assert written == [{"decode_only": False}]


def test_the_draft_residual_buffer_is_reused_and_recleared() -> None:
    """A resident residual costs one allocation, not one per draft forward."""
    model = DFlashDraftModel.__new__(DFlashDraftModel)
    model._residual_buffer = None

    first = DFlashDraftModel._zeroed_residual(model, torch.ones(4, 8))
    first.add_(3)
    second = DFlashDraftModel._zeroed_residual(model, torch.ones(2, 8))

    assert second.data_ptr() == first.data_ptr()
    assert second.shape == (2, 8)
    assert not second.any()


def test_distributed_topk_picks_what_a_whole_vocabulary_topk_would(monkeypatch) -> None:
    """Two shard-local top-16s must agree with one top-16 over the vocabulary."""
    from tokenspeed.runtime.execution.drafter import dflash2 as dflash2_runtime

    torch.manual_seed(7)
    rows, hidden, vocab, top_k, tp_size = 3, 6, 40, 4, 2
    weight = torch.randn(vocab, hidden)
    hidden_states = torch.randn(rows, hidden)
    shard = vocab // tp_size

    drafter = DFlash2.__new__(DFlash2)
    drafter.selector_top_k = top_k
    drafter.spec_num_tokens = 3
    drafter.input_buffers = SimpleNamespace(max_bs=2)
    drafter._candidate_gather_buffers = None
    drafter._distributed_topk_enabled = True
    drafter.lm_head = SimpleNamespace(
        weight=weight[:shard],
        shard_indices=SimpleNamespace(num_org_elements=shard, org_vocab_start_index=0),
    )
    drafter.candidate_logits_processor = SimpleNamespace(
        tp_size=tp_size, tp_group=None, logit_scale=None, final_logit_softcapping=None
    )

    # Stand in for rank 1: its own shard-local top-k, ids already global.
    peer = torch.matmul(hidden_states, weight[shard:].T)
    peer_values, peer_ids = torch.topk(peer, top_k, dim=-1, sorted=False)
    peer_ids = peer_ids + shard

    def fake_all_gather(out, src, group):
        out[: src.shape[0]].copy_(src)
        out[src.shape[0] :].copy_(peer_ids if src.dtype == torch.int64 else peer_values)

    monkeypatch.setattr(dflash2_runtime, "all_gather_into_tensor", fake_all_gather)
    candidate_ids, unary_logits = drafter._distributed_topk_candidates(hidden_states)

    expected = torch.topk(torch.matmul(hidden_states, weight.T), top_k, dim=-1)
    assert candidate_ids.tolist() == expected.indices.tolist()
    torch.testing.assert_close(unary_logits, expected.values.float())
