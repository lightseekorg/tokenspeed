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

"""Resident slots can outnumber the requests in a decode forward."""

from dataclasses import replace
from test.ci_system.ci_register import register_cuda_ci
from test.runtime.conftest import kimi_recipe

import pytest
import torch
from transformers import LlamaConfig

register_cuda_ci(est_time=15, suite="runtime-1gpu")


@pytest.mark.parametrize("dp,extra,want", [(1, 0, 8), (1, 4, 12), (2, 3, 7)])
def test_cli_extra_slots_preserve_execution_capacity(tmp_path, dp, extra, want):
    from tokenspeed.runtime.engine.rank_setup import load_model_config
    from tokenspeed.runtime.layers.attention.configs.base import model_wide_kwargs
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=256,
    ).save_pretrained(tmp_path)
    args = prepare_server_args(
        [
            "--model",
            str(tmp_path),
            "--max-num-seqs",
            "8",
            "--tensor-parallel-size",
            str(dp),
            "--data-parallel-size",
            str(dp),
            "--disaggregation-mode",
            "decode",
            "--disaggregation-decode-extra-slots",
            str(extra),
        ]
    )
    args.mapping.rank = 0
    model_config = load_model_config(args, args.model, draft_of=None)
    config = model_wide_kwargs(args, model_config, False, kv_cache_dtype=torch.bfloat16)
    assert config["max_bs"] == 8 // dp
    assert config["max_resident_requests"] == want


@pytest.mark.parametrize("role", ["null", "prefill", "encode"])
def test_positive_extra_slots_require_decode_role(role):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    with pytest.raises(ValueError, match="extra-slots"):
        ServerArgs(
            model="x", disaggregation_mode=role, disaggregation_decode_extra_slots=1
        )


@pytest.mark.parametrize("extra", [-1, True, 1.5])
def test_extra_slots_reject_invalid_capacity(extra):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    with pytest.raises(ValueError, match="extra-slots"):
        ServerArgs(
            model="x",
            max_num_seqs=8,
            disaggregation_mode="decode",
            disaggregation_decode_extra_slots=extra,
        )


def test_recipe_prices_residents_without_growing_execution_workspace():
    small = kimi_recipe(max_bs=2)
    expanded = kimi_recipe(max_bs=2)
    expanded.server_args.disaggregation_mode = "decode"
    expanded.attn_config = replace(
        expanded.attn_config,
        max_resident_requests=5,
        pd_disaggregation_enabled=True,
    )
    assert expanded.workspace_bytes() == small.workspace_bytes()
    groups = expanded.groups()
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
        capacity_model,
    )

    model = capacity_model(
        tuple(spec for spec, _ in groups),
        prefix_granularity=expanded.prefix_granularity,
        virtual_packing=expanded.packing(groups),
        limits=expanded.scheduler_limits,
    )
    pages = model.concurrent_group_pages(max_total_tokens=512, max_context_len=4096)
    assert [
        pages[index]
        for index, (group, _) in enumerate(groups)
        if group.attn_kind == "state"
    ] == [10, 10, 10]


def test_probe_keeps_fabricated_batch_small_with_extra_residents():
    def arena_bytes(residents):
        recipe = kimi_recipe(max_bs=2)
        recipe.attn_config = replace(
            recipe.attn_config, max_resident_requests=residents
        )
        recipe.probe_batch_rows = 2
        recipe.cache_budget_bytes = 0
        return recipe.setup().spec.memory_plan.arena_bytes

    assert arena_bytes(2) == arena_bytes(20)


def test_glm_tail_reserves_rows_for_resident_slots():
    from test.runtime.test_glm53_flash_cache_spec import _recipe

    recipe = _recipe(tp_size=8, mla_cache_dtype=torch.bfloat16)
    recipe.attn_config = replace(recipe.attn_config, max_bs=2, max_resident_requests=5)
    options = recipe.pool_options()
    assert options.num_request_slots == 7
    # 11 DSA layers, KPool=4, one overlap tail, 128 BF16 channels,
    # two generations, seven rows (null and padding included).
    assert options.workspace_bytes == 11 * 2 * 7 * options.tail_width * 128 * 2


@pytest.mark.parametrize("probe_rows", [None, 2])
def test_offload_prices_every_resident_partition(tmp_path, probe_rows):
    from test.runtime.test_cache_setup import _mla_config

    from tokenspeed.runtime.engine.rank_setup import load_model_config
    from tokenspeed.runtime.layers.attention.kv_cache.offload_config import (
        KVOffloadPolicy,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        OrdinaryRecipe,
    )
    from tokenspeed.runtime.utils.server_args import ServerArgs

    class OffloadRecipe(OrdinaryRecipe):
        def offload_policy(self):
            return KVOffloadPolicy(
                field_ids=("layer.0.latent_kv",),
                hot_tokens=64,
                reserved_tokens=4,
                topk=16,
                queries=1,
                host_budget_bytes=1 << 20,
                overlap=True,
                cyclic_tokens=0,
                selection_consumers=(),
            )

    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=256,
    ).save_pretrained(tmp_path)
    args = ServerArgs(
        model=str(tmp_path),
        max_num_seqs=2,
        device="cpu",
        disaggregation_mode="decode",
        disaggregation_decode_extra_slots=3,
        chunked_prefill_size=128,
    )
    recipe = OffloadRecipe(
        family="mla",
        server_args=args,
        model_config=load_model_config(args, args.model, draft_of=None),
        attn_config=replace(
            _mla_config(),
            max_resident_requests=5,
            pd_disaggregation_enabled=True,
        ),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 28,
        probe_batch_rows=probe_rows,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    storage = recipe.setup().spec.storage_plan
    assert storage.offload.request_slots == 7
    # Hot and receive/verify reserve partitions are per resident slot. Probe
    # builds retain them because the same backend persists into serving.
    assert storage.offload.hot_rows == 476
    # The MLA field has 64 rows per page; its physical workspace rounds the
    # seven 68-row request partitions up to eight complete pages.
    assert storage.offload.device_rows == 512


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_high_slot_bootstrap_and_reset_preserve_padding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates

    layout = RequestSlotLayout(5)
    states = RuntimeStates(layout.padding, 32, 32, 3, device)
    states.init_draft_trees(3)
    states.init_draft_probs(spec_num_tokens=3, reject_threshold=2.0)
    states.valid_cache_lengths.fill_(17)
    states.future_input_map.fill_(9)
    states.spec_candidates_ready.fill_(True)
    states.reset_states(
        torch.tensor([5], device=device),
        torch.tensor([100], dtype=torch.int32, device=device),
    )
    assert states.valid_cache_lengths.tolist() == [17, 17, 17, 17, 17, 100, 17]
    assert states.spec_candidates_ready.tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    assert states.future_input_map[layout.padding].tolist() == [9, 9, 9]

    # A B=2 forward may use either slot 4 or 5 when C=5. The real exporter
    # copies every persistent row through a Host image before slot reuse.
    expected = [row.clone() for row in states.slot_state_rows(5)]
    padding = [row.clone() for row in states.slot_state_rows(layout.padding)]
    image = torch.empty(
        states.slot_state_bytes(), dtype=torch.uint8, pin_memory=device == "cuda"
    )
    stream = torch.cuda.current_stream() if device == "cuda" else None
    states.export_slot_state(5, image, stream, request_id="high-slot")
    if stream is not None:
        stream.synchronize()
    for row in states.slot_state_rows(5):
        row.zero_()
    for row, saved in zip(states.slot_state_rows(4), expected, strict=True):
        row.copy_(saved.logical_not() if saved.dtype == torch.bool else saved + 1)
    states.import_slot_state(4, image, stream, request_id="high-slot")
    if stream is not None:
        stream.synchronize()
    assert all(
        torch.equal(row, saved)
        for row, saved in zip(states.slot_state_rows(4), expected, strict=True)
    )
    assert all(
        torch.equal(row, saved)
        for row, saved in zip(
            states.slot_state_rows(layout.padding), padding, strict=True
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_row_graph_replay_resets_high_resident_slots():
    from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates

    layout = RequestSlotLayout(5)
    states = RuntimeStates(layout.padding, 32, 32, 3, "cuda")
    states.init_draft_probs(spec_num_tokens=3, reject_threshold=2.0)
    indices = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    lengths = torch.tensor([11, 12], dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        states.reset_states(indices, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        states.reset_states(indices, lengths)

    # The captured forward still has B=2 rows, but their real slots change
    # from 1,2 to 4,5. Neither old capture slots nor padding 6 may be reset.
    indices.copy_(torch.tensor([4, 5], dtype=torch.int64, device="cuda"))
    lengths.copy_(torch.tensor([104, 105], dtype=torch.int32, device="cuda"))
    states.valid_cache_lengths.fill_(17)
    states.spec_candidates_ready.fill_(True)
    states.stopped.fill_(True)
    states.draft_probs[:, :-1].fill_(0.25)
    graph.replay()

    assert states.valid_cache_lengths.tolist() == [17, 17, 17, 17, 104, 105, 17]
    assert states.spec_candidates_ready.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        True,
    ]
    assert states.stopped.tolist() == [True, True, True, True, False, False, True]
    assert torch.equal(states.draft_probs[4:6, :-1].cpu(), torch.full((2, 2, 32), 3.0))
    assert torch.equal(states.draft_probs[:4, :-1].cpu(), torch.full((4, 2, 32), 0.25))
    assert torch.equal(
        states.draft_probs[layout.padding, :-1].cpu(), torch.full((2, 32), 0.25)
    )
    assert torch.equal(states.draft_probs[:, -1].cpu(), torch.zeros((7, 32)))
