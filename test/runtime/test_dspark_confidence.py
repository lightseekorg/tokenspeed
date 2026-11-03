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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "state_dtype,width", [(torch.bfloat16, 8), (torch.float32, 13)]
)
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_compact_gdn_prefixes_and_commits_survive_uniform_verify_tactics(
    state_dtype, width, replay, cuda_graph
):
    from test.runtime.test_gdn_state_paging import _ContractPool, _mamba_config_pair

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.state.gdn import GdnAttnBackend

    torch.manual_seed(53)
    batch, heads, dim = 3, 8, 128
    channels = 3 * heads * dim
    conv = torch.randn(8, channels, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(8, heads, dim, dim, device="cuda", dtype=state_dtype) * 0.02
    conv[0].zero_()
    state[0].zero_()
    backends, pools = [], []
    for _ in range(2):
        pool = _ContractPool({0: ("linear_attention", conv.clone(), state.clone())})
        backend = GdnAttnBackend(
            *_mamba_config_pair(
                torch,
                heads=heads,
                head_dim=dim,
                spec_tokens=width,
                max_bs=batch,
                device="cuda",
                replay_ssm=replay,
            )
        )
        backend.set_kv_pool(pool)
        backend.init_cuda_graph_state(batch)
        backend.refresh_decode_metadata(
            batch,
            2,
            torch.arange(batch, dtype=torch.int32, device="cuda"),
            torch.full((batch,), 4 + width, dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=cuda_graph,
            block_tables={
                "linear_attention": torch.tensor(
                    [[1, 5], [2, 6], [0, 0]], dtype=torch.int32, device="cuda"
                )
            },
        )
        backends.append(backend)
        pools.append(pool)
    layouts = (
        VerifyQueryLayout(batch, batch * width, width, "cuda"),
        VerifyQueryLayout(batch, width, width, "cuda"),
    )
    rates = torch.full((batch, width - 1), 0.1, device="cuda")
    rates[0].fill_(0.99)
    ready = torch.ones(batch, dtype=torch.bool, device="cuda")
    active = torch.tensor([True, True, False], device="cuda")
    for layout in layouts:
        layout.refresh(rates, ready, active)
    raw = torch.randn(batch * width, channels, device="cuda", dtype=torch.bfloat16)
    a, b = (
        torch.randn(batch * width, heads, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    weights = torch.randn(channels, 4, device="cuda", dtype=raw.dtype) * 0.1
    A_log, dt_bias = (torch.randn(heads, device="cuda") * 0.1 for _ in range(2))

    actual_bs = 2

    def forward(index):
        layout = layouts[index]
        return backends[index].forward_decode(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=pools[index],
            bs=actual_bs,
            mixed_qkv=layout.pack(raw, token_dim=0),
            conv_weights=weights,
            bias=None,
            activation="silu",
            key_dim=heads * dim,
            value_dim=heads * dim,
            attention_tp_size=1,
            head_k_dim=dim,
            head_v_dim=dim,
            a=layout.pack(a, token_dim=0),
            b=layout.pack(b, token_dim=0),
            A_log=A_log,
            dt_bias=dt_bias,
            layer_id=0,
            seq_len=layout.query_capacity,
            verify_queries=layout,
        )

    for _ in range(2):
        forward(0)
        forward(1)
    graph = None
    commit_graph = None
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward(1)
        backend = backends[1]

        def prepare_commit():
            backend.init_forward_metadata_capture_cuda_graph(
                batch,
                torch.arange(batch, dtype=torch.int32, device="cuda"),
                torch.full((batch,), 4 + width, dtype=torch.int32, device="cuda"),
                ForwardMode.DECODE,
            )
            backend.prepare_state_commit_capture()

        for _ in range(2):
            prepare_commit()
            backend.commit_verified_state(layouts[1].query_lengths, accepted_path=None)
        prepare_commit()
        commit_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(commit_graph):
            backend.commit_verified_state(layouts[1].query_lengths, accepted_path=None)
    for step in range(2):
        actual_bs = 2 - step
        active.copy_(torch.arange(batch, device="cuda") < actual_bs)
        for layout in layouts:
            layout.refresh(rates.roll(step, 0), ready, active)
        for backend in backends:
            # Fresh delivered storage on each round must not replace the
            # table addresses recorded by the commit graph.
            backend.refresh_decode_metadata(
                batch,
                actual_bs,
                torch.arange(batch, dtype=torch.int32, device="cuda"),
                torch.full((batch,), 4 + width, dtype=torch.int32, device="cuda"),
                forward_mode=ForwardMode.DECODE,
                for_graph_replay=cuda_graph,
                block_tables={
                    "linear_attention": torch.tensor(
                        [[1 + step, 5 + step], [2 - step, 6 - step]],
                        dtype=torch.int32,
                        device="cuda",
                    )
                },
            )
        expected = forward(0).reshape(-1, heads, dim)
        if graph is None:
            actual = forward(1)
        else:
            graph.replay()
        selected = layouts[1].packed_rows[layouts[1].packed_valid]
        torch.testing.assert_close(
            actual.reshape(-1, heads, dim)[: selected.numel()],
            expected.index_select(0, selected),
            atol=0,
            rtol=0,
        )
        backends[0].commit_verified_state(layouts[1].query_lengths, accepted_path=None)
        if commit_graph is None:
            backends[1].commit_verified_state(
                layouts[1].query_lengths, accepted_path=None
            )
        else:
            commit_graph.replay()
        for component in ("conv_state", "recurrent_state"):
            torch.testing.assert_close(
                pools[0].get_component(0, component),
                pools[1].get_component(0, component),
                atol=0,
                rtol=0,
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
