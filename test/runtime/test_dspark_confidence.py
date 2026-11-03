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

"""Numerical and lifecycle coverage for DSpark confidence verification."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.verify_layout import VerifyQueryLayout
from tokenspeed.runtime.models.dspark import (
    AcceptRatePredictor,
    VanillaMarkov,
)
from tokenspeed.runtime.models.qwen4_exp import dspark_hc_features


def _draft_model():
    from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.dspark import DSparkDraftModel

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        block_size=2,
        target_layer_ids=[0],
        markov_rank=2,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    return DSparkDraftModel(config, Mapping(rank=0, world_size=1))


def test_confidence_checkpoint_loading_is_complete_and_numerical():
    model = _draft_model()
    weight = torch.arange(10, dtype=torch.float32).view(1, 10) / 10
    bias = torch.tensor([0.25])
    model.load_weights(
        [
            ("confidence_head.proj.weight", weight),
            ("confidence_head.proj.bias", bias),
        ]
    )
    hidden = torch.ones(1, 8)
    latent = torch.ones(1, 2)
    actual = model.confidence_head(hidden, latent)
    torch.testing.assert_close(actual, torch.tensor([4.75]).sigmoid())


@pytest.mark.parametrize(
    "weights", [[], [("confidence_head.proj.bias", torch.zeros(1))]]
)
def test_confidence_checkpoint_rejects_missing_weights(weights):
    model = _draft_model()
    with pytest.raises(ValueError, match="Missing DSpark confidence weights"):
        model.load_weights(weights)


def test_confidence_conditions_on_previous_token():
    head = AcceptRatePredictor(2, 2, with_markov=True)
    markov = VanillaMarkov(vocab_size=3, markov_rank=2)
    with torch.no_grad():
        head.proj.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        head.proj.bias.fill_(0.5)
        markov.markov_w1.weight.copy_(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
    hidden = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    rates = head(hidden, markov.get_prev_latent(torch.tensor([1, 2])))
    torch.testing.assert_close(rates, torch.tensor([4.5, 5.5]).sigmoid())
    with pytest.raises(ValueError, match="previous latent"):
        head(hidden, None)


def test_hc_capture_uses_fp32_mean_and_preserves_branch_order():
    branches = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], dtype=torch.bfloat16
    )
    actual = dspark_hc_features(branches.flatten(1), hc_count=4, hidden_size=2)
    torch.testing.assert_close(actual, torch.tensor([[4.0, 5.0]], dtype=torch.bfloat16))
    initial = branches[:, 0]
    torch.testing.assert_close(
        dspark_hc_features(initial, hc_count=4, hidden_size=2), initial
    )


def test_compact_attention_retains_window_start_and_isolates_filler():
    layout = VerifyQueryLayout(3, 6, 4, "cpu")
    rates = torch.full((3, 3), 0.5)
    ready = torch.ones(3, dtype=torch.bool)
    active = torch.tensor([True, False, True])
    seq_lens = torch.tensor([14, 100, 24], dtype=torch.int32)
    page_table = torch.tensor([[1, 2], [5, 6], [9, 10]], dtype=torch.int32)
    layout.refresh(rates, ready, active)
    pages, lengths = layout.causal_metadata(seq_lens, page_table)
    assert lengths.dtype == torch.int32
    assert pages.tolist() == [[1, 2]] * 3 + [[9, 10]] * 3
    assert lengths.tolist() == [11, 12, 13, 21, 22, 23]
    ready.fill_(False)
    layout.refresh(rates, ready, active)
    pages, lengths = layout.causal_metadata(seq_lens, page_table)
    assert pages.tolist() == [[1, 2], [9, 10]] + [[0, 0]] * 4
    assert lengths.tolist() == [11, 21, 1, 1, 1, 1]


@pytest.mark.parametrize("queries,expected", [(2, [1, 1]), (4, [3, 1]), (6, [3, 3])])
def test_budget_selects_high_confidence_prefixes(queries, expected):
    layout = VerifyQueryLayout(2, queries, 3, "cpu")
    layout.refresh(
        torch.tensor([[0.9, 0.9], [0.1, 0.1]]),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    assert layout.query_lengths.tolist() == expected
    values = torch.arange(6).reshape(6, 1)
    expanded = layout.expand(layout.pack(values, token_dim=0), token_dim=0)
    for request, length in enumerate(expected):
        torch.testing.assert_close(
            expanded[request * 3 : request * 3 + length],
            values[request * 3 : request * 3 + length],
        )


def test_fresh_and_padding_requests_never_consume_draft_scores():
    layout = VerifyQueryLayout(3, 9, 3, "cpu")
    layout.refresh(
        torch.ones(3, 2),
        torch.tensor([True, False, True]),
        torch.tensor([True, True, False]),
    )
    assert layout.query_lengths.tolist() == [3, 1, 0]
    assert layout.packed_valid.tolist() == [True] * 4 + [False] * 5
    assert layout.query_offsets.tolist() == [0, 3, 4, 4]


def test_equal_scores_choose_request_then_position():
    layout = VerifyQueryLayout(3, 6, 3, "cpu")
    layout.refresh(
        torch.ones(3, 2),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
    )
    assert layout.query_lengths.tolist() == [3, 2, 1]


def test_slot_confidence_generations_are_reset_and_restored():
    states = RuntimeStates(5, 16, 4, device="cpu")
    states.init_draft_confidence(3)
    slots = torch.tensor([1])
    first = torch.tensor([[0.9, 0.8, 0.7]])
    second = torch.tensor([[0.6, 0.5, 0.4]])
    states.publish_draft_confidence(slots, first, group=None)
    states.publish_draft_confidence(slots, second, group=None)
    torch.testing.assert_close(states.previous_draft_confidence[slots], first)
    torch.testing.assert_close(states.draft_confidence[slots], second)
    image = torch.empty(states.slot_state_bytes(), dtype=torch.uint8)
    states.export_slot_state(1, image, None, request_id="request")
    states.import_slot_state(2, image, None, request_id="request")
    torch.testing.assert_close(states.draft_confidence[2], second[0])
    torch.testing.assert_close(states.previous_draft_confidence[2], first[0])
    states.reset_states(torch.tensor([2]), torch.tensor([0], dtype=torch.int32))
    assert not states.draft_confidence[2].any()
    assert not states.previous_draft_confidence[2].any()
    torch.testing.assert_close(states.draft_confidence[1], second[0])


def test_planner_accounts_for_full_round_cost_and_prefers_smaller_ties():
    from tokenspeed_scheduler import ConfidenceBudgetPlanner

    planner = ConfidenceBudgetPlanner(3)
    planner.observe(2, 2, 1.0)
    planner.observe(2, 4, 1.5)
    planner.observe(2, 6, 3.0)
    assert planner.select(2, [[0.9, 0.9], [0.9, 0.9]]) == 4
    assert planner.select(2, [[0.0, 0.0], [0.0, 0.0]]) == 2
    with pytest.raises(ValueError):
        planner.observe(2, 4, float("nan"))
    with pytest.raises(ValueError):
        planner.select(2, [[1.0]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_layout_replay_refreshes_lengths_without_rebinding():
    device = "cuda"
    layout = VerifyQueryLayout(2, 4, 3, device)
    rates = torch.tensor([[0.9, 0.9], [0.1, 0.1]], device=device)
    ready = torch.ones(2, dtype=torch.bool, device=device)
    active = ready.clone()
    values = torch.arange(6, device=device).reshape(6, 1)
    layout.refresh(rates, ready, active)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layout.refresh(rates, ready, active)
        output = layout.expand(layout.pack(values, token_dim=0), token_dim=0)
    graph.replay()
    assert layout.query_lengths.tolist() == [3, 1]
    pointer = layout.packed_rows.data_ptr()
    rates.copy_(rates.flip(0))
    graph.replay()
    assert layout.query_lengths.tolist() == [1, 3]
    assert layout.packed_rows.data_ptr() == pointer
    torch.testing.assert_close(output[3:6], values[3:6])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_layout_packs_multimodal_positions_with_an_empty_bucket():
    layout = VerifyQueryLayout(2, 4, 3, "cuda")
    rates = torch.ones(2, 2, device="cuda")
    ready = torch.zeros(2, dtype=torch.bool, device="cuda")
    active = ready.clone()
    positions = torch.arange(18, device="cuda").view(3, 6)
    layout.refresh(rates, ready, active)
    result = layout.pack(positions, token_dim=1)
    torch.testing.assert_close(
        result, torch.zeros(3, 4, device="cuda", dtype=positions.dtype)
    )
    active.fill_(True)
    layout.refresh(rates, ready, active)
    result = layout.pack(positions, token_dim=1)
    torch.testing.assert_close(result[:, :2], positions[:, [0, 3]])
    assert result[:, 2:].count_nonzero().item() == 0
