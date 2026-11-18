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

"""Decoder-only replay consumes live encoder rows and restores the context."""

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from test.runtime.execution.test_prefill_history_embedding import _config, _NoAttention
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.memory_delta import NULL_MEMORY_DELTA_OBSERVER
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@dataclass
class RowState:
    hidden: torch.Tensor

    @property
    def rows(self):
        return self.hidden.shape[0]

    def land_into(self, target):
        target.hidden[: self.rows].copy_(self.hidden)
        target.hidden[self.rows :].zero_()


class WindowModel(torch.nn.Module):
    max_decoder_rows_per_request = 16
    lm_head = torch.nn.Identity()

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(64, 1, device="cuda")

    def allocate_decoder_state(self, rows):
        return RowState(self.embed_tokens.weight.new_zeros((rows, 1)))

    def encoder_forward(self, input_ids, positions, ctx, *, input_embeds):
        return RowState(input_embeds + positions[:, None])

    def decoder_rows(self, ctx):
        return ctx.attn_backend.rows

    def narrowing_forward(self, state, ctx):
        return RowState(state.hidden[state.rows - self.decoder_rows(ctx) :])

    def decoder_forward(self, state, ctx):
        return state.hidden * 3, []

    def finish_forward(self, hidden, captured, ctx):
        return hidden, None

    def logits_processor(self, input_ids, hidden, head, metadata, auxiliary):
        return head(hidden), auxiliary


class WindowBackend(_NoAttention):
    def __init__(self):
        super().__init__()
        self.capacity = 0
        self.rows = 0

    def init_prefill_graph_state(self, *, max_num_tokens, max_bs):
        self.capacity = max_num_tokens

    def init_forward_metadata(self, *, extend_seq_lens_cpu, **kwargs):
        self.rows = sum(
            min(n, WindowModel.max_decoder_rows_per_request)
            for n in extend_seq_lens_cpu.tolist()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@torch.inference_mode()
def test_live_eager_encoder_feeds_captured_decoder():
    config = replace(
        _config(),
        prefill_graph_max_tokens=8,
        prefill_graph_capture_sizes=[8],
        prefill_graph_decoder_capture_token_sizes=[24, 16, 24],
    )
    model = WindowModel()
    backend = WindowBackend()
    inputs = InputBuffers(2, 32, 3)
    owner = PrefillGraph(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(
                model=model,
                lm_head=model.lm_head,
                logits_processor=model.logits_processor,
            ),
            model_config=SimpleNamespace(requires_request_token_history=False),
            is_generation=True,
        ),
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(
            arena=SimpleNamespace(cache_group_specs=()), layerwise_load_tracker=None
        ),
        input_buffers=inputs,
        config=config,
        memory_saver_adapter=TorchMemorySaverAdapter.create(enable=False),
    )
    owner.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    assert sorted(owner._encoders) == [8] and sorted(owner._decoders) == [16, 24]
    assert backend.capacity == 24

    # First fill the 16-row decoder, then reuse it with fewer live rows.
    # The last round needs the larger decoder despite the 8-row encoder limit.
    for tokens, requests, retained, capacity, precomputed in (
        (32, 1, 16, 16, True),
        (11, 1, 11, 16, False),
        (20, 2, 20, 24, False),
    ):
        ctx = owner.make_dummy_batch(tokens, requests)
        ids = torch.arange(tokens, device="cuda")
        positions = ids + tokens
        inputs.positions_buf[:tokens].copy_(positions)
        embeddings = (
            ids[:, None].float() * 2 if precomputed else model.embed_tokens(ids)
        )
        buckets = owner.select_buckets(ctx)
        assert buckets == (None, capacity)
        hidden, aux = owner.replay(
            ctx,
            ids,
            buckets=buckets,
            model_kwargs={"input_embeds": embeddings} if precomputed else None,
        )
        expected = (embeddings + positions[:, None])[-retained:] * 3
        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
        assert not owner._decoders[capacity].statics.hidden[retained:].any()
        assert aux is None and ctx.input_num_tokens == tokens and owner._ctx is None

    for ineligible in (
        replace(ctx, forward_mode=ForwardMode.DECODE),
        replace(ctx, query_shard=object()),
        replace(ctx, draft_narrowing=object()),
        replace(ctx, capture_hidden_mode=CaptureHiddenMode.FULL),
    ):
        assert owner.select_buckets(ineligible) is None
    assert owner.select_buckets(owner.make_dummy_batch(32, 2)) is None
