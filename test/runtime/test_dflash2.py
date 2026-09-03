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
    _walk_greedy_path,
)

# Imported for its register_backend() side effects on _BACKEND_REGISTRY.
from tokenspeed.runtime.layers.attention import backends  # noqa: F401
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import _BACKEND_REGISTRY
from tokenspeed.runtime.models.dflash import DFlashAttention
from tokenspeed.runtime.models.dflash2 import (
    CandidateSelector,
    DFlash2DraftModel,
    DFlashGroupedConv,
    _dflash2_mla_rope,
    _dflash2_uses_mla,
    _grouped_conv,
    _score_edges,
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
def test_grouped_conv_matches_a_block_local_reference(block_size: int) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(2 * block_size, 12)
    delta = torch.randn(2 * block_size, 3, 3)
    base = torch.randn(3, 12)
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
