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

"""History-dependent embeddings stay live outside prefill captures."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.memory_delta import NULL_MEMORY_DELTA_OBSERVER
from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext

register_cuda_ci(est_time=10, suite="runtime-1gpu")


class _TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8, device="cuda")

    def forward(self, input_ids, positions, ctx, *, input_embeds=None):
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        return input_embeds.square() + positions[:, None], None


class _HistoryTokenModel(_TokenModel):
    def compute_input_embeddings(self, input_ids, ctx, *, request_token_history):
        if request_token_history is None:
            raise ValueError("history is required when computing embeddings")
        history = request_token_history
        # A small history-dependent embedding with the runtime's append contract.
        # This intentionally uses CPU lengths: the seam must stay outside capture.
        pieces = []
        for row, slot in enumerate(history.req_pool_indices.tolist()):
            start, end = history.input_start_offsets[row : row + 2].tolist()
            committed = int(history.committed_lengths[slot])
            prior = history.history_token_ids[slot, :committed].sum() % 32
            ids = input_ids[start:end]
            pieces.append(self.embed_tokens((ids + prior) % 32))
            history.history_token_ids[slot, committed : committed + len(ids)].copy_(ids)
        return torch.cat(pieces)

    def forward(
        self,
        input_ids,
        positions,
        ctx,
        *,
        input_embeds=None,
        request_token_history=None,
    ):
        if input_embeds is None:
            input_embeds = self.compute_input_embeddings(
                input_ids, ctx, request_token_history=request_token_history
            )
        return super().forward(input_ids, positions, ctx, input_embeds=input_embeds)


class _HiddenRows(torch.nn.Module):
    def forward(self, input_ids, hidden, lm_head, metadata, aux):
        return hidden


class _TextModel(BaseCausalLM):
    def __init__(self):
        super().__init__(PretrainedConfig(), Mapping(rank=0, world_size=1))

    def resolve_model(self, config, mapping, quant_config, prefix):
        return _TokenModel()

    def resolve_lm_head(self, config, quant_config, prefix):
        return torch.nn.Identity()

    def resolve_logits_processor(self, config):
        return _HiddenRows()

    def multimodal_input_embeds(self, input_ids, ctx, multimodal_context):
        return self.model.embed_tokens(input_ids)


class _HistoryModel(_TextModel):
    def resolve_model(self, config, mapping, quant_config, prefix):
        return _HistoryTokenModel()

    def prepare_model_kwargs(self, ctx, input_ids, kwargs):
        model_kwargs = super().prepare_model_kwargs(ctx, input_ids, kwargs)
        model_kwargs["request_token_history"] = kwargs.get("request_token_history")
        return model_kwargs


class _NoAttention(AttentionBackend):
    """The test's token-only model has no attention or per-forward metadata."""

    def __init__(self):
        self._init_pool_binding()

    def init_forward_metadata(self, *args, **kwargs):
        pass


@dataclass
class _ModelConfig:
    requires_request_token_history: bool
    is_multimodal_active: bool = False


@dataclass
class _Runner:
    model: _TextModel
    model_config: _ModelConfig
    is_generation: bool = True
    is_multimodal: bool = False


def _pool():
    spec = CacheGroupSpec(
        group_id="full_attention",
        attn_kind="full_history",
        rows_per_page=16,
        entry_stride_tokens=1,
        replayable=False,
    )
    fields = tuple(
        CacheFieldSpec(f"layer.0.{p}", p, (16, 1, 8), "float32") for p in ("k", "v")
    )
    arena = CacheArena(
        pack([(spec, fields)], prefix_granularity=16).bind(4),
        "cuda",
        cache_group_specs=(spec,),
    )
    return MHATokenToKVPool(arena, torch.float32, 1, 8, 1, 0)


def _config():
    return ModelExecutorConfig(
        max_req_pool_size=4,
        output_length=1,
        enforce_eager=False,
        prefix_granularity=16,
        max_num_seqs=2,
        chunked_prefill_size=32,
        vocab_size=32,
        input_vocab_size=32,
        context_len=64,
        physical_context_len=64,
        device="cuda",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=[1, 2],
        disable_cuda_graph_padding=False,
        spec_topk=1,
        max_cudagraph_capture_size=2,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=[1],
        enable_drafter_prefill_graph=False,
        input_logprob_chunk_tokens=32,
        enable_speculative_sampling=False,
        query_shard_size=1,
        query_shard_rank=0,
        enable_memory_saver=False,
        disable_prefill_graph=False,
        prefill_graph_max_tokens=32,
        prefill_graph_capture_sizes=[16, 32],
        disable_autotune=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("history_embedding", [False, True])
@torch.inference_mode()
def test_embeddings_refresh_with_live_history_and_padding(history_embedding):
    model = _HistoryModel() if history_embedding else _TextModel()
    ib = InputBuffers(2, 32, 4)
    owner = PrefillGraph(
        _Runner(model, _ModelConfig(history_embedding)),
        _NoAttention(),
        _pool(),
        ib,
        _config(),
        num_warmup=1,
    )
    history_ids = torch.full((4, 64), 7, dtype=torch.int32, device="cuda")
    committed = torch.tensor([2, 5, 3, 0], dtype=torch.int32, device="cuda")
    before = history_ids.clone()
    try:
        owner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
        torch.testing.assert_close(history_ids, before, rtol=0, atol=0)
        for count, slots in ((15, [1]), (32, [2, 0]), (17, [0]), (16, [1, 2])):
            ctx = owner.make_dummy_batch(count, len(slots))
            ib.req_pool_indices_buf[: len(slots)].copy_(
                torch.tensor(slots, device="cuda")
            )
            ids = torch.arange(count, device="cuda") % 32
            ib.input_ids_buf[:count].copy_(ids)
            history = RequestTokenHistoryView(
                history_ids,
                committed,
                ib.req_pool_indices_buf[: len(slots)],
                ib.input_start_offsets_buf[: len(slots) + 1],
                ib.active_request_mask_buf[: len(slots)],
            )
            expected = model(
                ctx, ids, ib.positions_buf[:count], request_token_history=history
            )
            appended = history_ids.clone()
            history_ids.copy_(before)
            assert owner.can_run(ctx)
            mm_context = MultimodalForwardContext([None], [0], [count], count)
            assert owner.can_run(ctx, mm_context) is not history_embedding
            actual = owner.replay(ctx, ids, request_token_history=history)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(history_ids, appended, rtol=0, atol=0)
            history_ids.add_(1)
            before = history_ids.clone()
            committed.add_(1)
        owner.release_graphs()
        owner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
        torch.testing.assert_close(history_ids, before, rtol=0, atol=0)
        ctx = owner.make_dummy_batch(16, 1)
        ids = ib.input_ids_buf[:16]
        history = RequestTokenHistoryView(
            history_ids,
            committed,
            ib.req_pool_indices_buf[:1],
            ib.input_start_offsets_buf[:2],
            ib.active_request_mask_buf[:1],
        )
        expected = model(ctx, ids, ib.positions_buf[:16], request_token_history=history)
        appended = history_ids.clone()
        history_ids.copy_(before)
        actual = owner.replay(ctx, ids, request_token_history=history)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(history_ids, appended, rtol=0, atol=0)
    finally:
        owner.release_graphs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_history_model_without_embedding_seam_stays_eager():
    owner = PrefillGraph(
        _Runner(_TextModel(), _ModelConfig(True)),
        _NoAttention(),
        _pool(),
        InputBuffers(2, 32, 4),
        _config(),
    )
    owner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    assert not owner.can_run(owner.make_dummy_batch(16, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("active", [False, True])
def test_dp_multimodal_restriction_follows_active_model(active):
    config = _config()
    config.data_parallel_size = 2
    runner = _Runner(_HistoryModel(), _ModelConfig(True, active), is_multimodal=True)
    owner = PrefillGraph(
        runner, _NoAttention(), _pool(), InputBuffers(2, 32, 4), config
    )
    assert bool(owner.capture_plan) is not active


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_supplied_embeddings_bypass_history_in_eager_forward():
    model = _HistoryModel()
    owner = PrefillGraph(
        _Runner(model, _ModelConfig(True)),
        _NoAttention(),
        _pool(),
        InputBuffers(2, 32, 4),
        _config(),
    )
    ctx = owner.make_dummy_batch(16, 1)
    ids = torch.arange(16, device="cuda")
    positions = owner.input_buffers.positions_buf[:16]
    embeds = torch.randn(16, 8, device="cuda")
    actual = model(ctx, ids, positions, input_embeds=embeds)
    torch.testing.assert_close(actual, embeds.square() + positions[:, None])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_decode_uses_model_owned_history_embedding():
    model = _HistoryModel()
    owner = PrefillGraph(
        _Runner(model, _ModelConfig(True)),
        _NoAttention(),
        _pool(),
        InputBuffers(2, 32, 4),
        _config(),
    )
    ctx = owner.make_dummy_batch(1, 1)
    ctx.forward_mode = ForwardMode.DECODE
    ctx.num_extends = 0
    ib = owner.input_buffers
    ids = torch.ones(1, dtype=torch.int32, device="cuda")
    positions = ib.positions_buf[:1]
    history_ids = torch.full((4, 64), 7, dtype=torch.int32, device="cuda")
    committed = torch.full((4,), 2, dtype=torch.int32, device="cuda")
    history = RequestTokenHistoryView(
        history_ids,
        committed,
        ib.req_pool_indices_buf[:1],
        ib.input_start_offsets_buf[:2],
        ib.active_request_mask_buf[:1],
    )
    expected_history = history_ids.clone()
    expected_history[0, 2] = 1
    actual = model(ctx, ids, positions, request_token_history=history)
    # The two committed tokens each contribute 7 to the embedding index.
    expected = model.model.embed_tokens(ids + 14).square() + positions[:, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(history_ids, expected_history, rtol=0, atol=0)
